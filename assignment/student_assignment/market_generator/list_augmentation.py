"""Module for augmenting student preference lists to correct
strategic non-reporting bias.

Certain demographic groups systematically under-report demand
for oversubscribed schools. This module identifies those students
and augments their preference lists with oversubscribed programs
they are likely interested in but did not list.
"""

import numpy as np
import pandas as pd

# Ethnicity groups considered "targeted" when combined with CTIP
_TARGETED_ETHNICITIES = {
    "Black or African American",
    "Hispanic/Latino",
    "Filipino",
    "Pacific Islander",
    "American Indian/Alaska Native",
    "Two or More Races",
    "Decline to State",
}

_ETHNICITY_ALIASES = {
    "Hispanic/Latinx": "Hispanic/Latino",
    "Hispanic": "Hispanic/Latino",
    "Black/African American": "Black or African American",
    "Black": "Black or African American",
    "Other Pacific Islander": "Pacific Islander",
    "Native Hawaiian/Pacific Islander": "Pacific Islander",
    "Samoan": "Pacific Islander",
    "Hawaiian": "Pacific Islander",
    "Hawaiian Native": "Pacific Islander",
    "Multi-Racial": "Two or More Races",
    "Two or More": "Two or More Races",
    "Two or more races": "Two or More Races",
    "Decline To State": "Decline to State",
    "American Indian": "American Indian/Alaska Native",
    "American Indian or Alaska Native": "American Indian/Alaska Native",
    "American Indian or Alaskan Native": "American Indian/Alaska Native",
}


def _normalize_ethnicity(value: object) -> str:
    ethnicity = "" if pd.isna(value) else str(value).strip()
    return _ETHNICITY_ALIASES.get(ethnicity, ethnicity)


def identify_targeted_students(
    student_data: pd.DataFrame,
    pref_lengths: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Identify students whose lists should be augmented.

    Args:
        student_data: DataFrame indexed by studentno with
            columns ``ctip1``, ``resolved_ethnicity``.
        pref_lengths: Array of length n with each student's
            current preference list length.
        config: The ``list-augmentation`` sub-config dict.

    Returns:
        Boolean array of shape (n,) where True means the
        student is targeted for augmentation.
    """
    method = config.get("targeting-method", "ctip_x_ethnicity")

    if method == "ctip_x_ethnicity":
        ctip_vals = student_data["ctip1"].fillna(0).astype(int).to_numpy()
        ethnicity_vals = student_data["resolved_ethnicity"].map(
            _normalize_ethnicity
        )
        is_ctip = ctip_vals == 1
        is_targeted_ethn = ethnicity_vals.isin(_TARGETED_ETHNICITIES).to_numpy()
        targeted = is_ctip & is_targeted_ethn

    elif method == "short_list_threshold":
        threshold = config.get("short-list-threshold", 3)
        targeted = pref_lengths < threshold

    else:
        raise ValueError(f"Unknown targeting method: {method}")

    return targeted


def identify_oversubscribed_programs(
    prefs: np.ndarray,
    capacity: np.ndarray,
    school_to_indices: dict[int, list[int]],
    config: dict,
    *,
    pref_lengths: np.ndarray,
) -> np.ndarray:
    """Identify oversubscribed program indices.

    Args:
        prefs: Preference matrix (n_students, n_programs)
            where non-zero entries are 1-indexed program
            indices.
        capacity: Array of length n_programs with program
            capacities.
        school_to_indices: Dict mapping school_id to list
            of 1-indexed program indices.
        config: The ``list-augmentation`` sub-config dict.
        pref_lengths: Number of submitted choices per student. Entries after this
            boundary are designation options and are not applications.

    Returns:
        1-D array of 1-indexed program indices that are
        oversubscribed (sorted by oversubscription ratio,
        descending).
    """
    method = config.get("oversubscribed-method", "first_choice_per_seat")
    num_programs = len(capacity)
    threshold = config.get("oversubscribed-ratio-threshold", 1.5)

    if method == "fixed_list":
        school_ids = config.get("oversubscribed-fixed-schools", [])
        program_indices = []
        for sid in school_ids:
            program_indices.extend(school_to_indices.get(int(sid), []))
        return np.array(list(dict.fromkeys(program_indices)), dtype=int)

    pref_lengths = np.asarray(pref_lengths, dtype=int)
    if pref_lengths.shape != (prefs.shape[0],):
        raise ValueError("pref_lengths must contain one value per student")
    pref_lengths = np.clip(pref_lengths, 0, prefs.shape[1])
    submitted_mask = np.arange(prefs.shape[1]) < pref_lengths[:, np.newaxis]

    # Count applications per program (1-indexed in prefs)
    if method == "first_choice_per_seat":
        first_choices = np.where(pref_lengths > 0, prefs[:, 0], 0).astype(int)
        valid = (first_choices > 0) & (first_choices <= num_programs)
        app_counts = np.bincount(
            first_choices[valid],
            minlength=num_programs + 1,
        )[1:]

    elif method == "apps_per_seat":
        flat = prefs[submitted_mask].astype(int)
        valid = (flat > 0) & (flat <= num_programs)
        app_counts = np.bincount(flat[valid], minlength=num_programs + 1)[1:]

    else:
        raise ValueError(f"Unknown oversubscribed method: {method}")

    safe_capacity = np.maximum(capacity, 1)
    ratio = app_counts / safe_capacity

    oversub_mask = ratio > threshold
    oversub_indices = np.where(oversub_mask)[0]
    sorted_order = np.argsort(-ratio[oversub_indices])
    result = oversub_indices[sorted_order] + 1

    return result


def augment_preferences(
    prefs: np.ndarray,
    pref_lengths: np.ndarray,
    targeted_mask: np.ndarray,
    oversubscribed_programs: np.ndarray,
    student_data: pd.DataFrame,
    distance_matrix: np.ndarray,
    config: dict,
    *,
    eligibility_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Augment short preference lists of targeted students.

    For each targeted student whose list is shorter than the
    population median, inserts at the **top** of their list
    the oversubscribed program (not already listed) that is
    geographically closest to the student's home.

    Args:
        prefs: Preference matrix (n, p) with 1-indexed
            program indices.
        pref_lengths: Array (n,) of current list lengths.
        targeted_mask: Boolean array (n,) of targeted
            students.
        oversubscribed_programs: 1-indexed program indices
            to consider adding.
        student_data: Student DataFrame with ``ctip1`` and
            ``resolved_ethnicity`` columns, for subgroup
            tracking.
        distance_matrix: Array (n_students, n_programs)
            with pairwise distances in miles. Columns are
            0-indexed (program index - 1).
        config: The ``list-augmentation`` sub-config dict.
        eligibility_matrix: Boolean (n_students, n_programs) matrix combining
            program-type and zone eligibility.

    Returns:
        Tuple of (augmented_prefs, augmented_lengths,
        impact_df) where impact_df is a DataFrame with
        per-subgroup impact statistics.
    """
    max_add = max(0, int(config.get("max-augmented-programs", 1)))

    augmented_prefs = prefs.copy()
    augmented_lengths = pref_lengths.copy()
    num_programs = prefs.shape[1]
    eligibility_matrix = np.asarray(eligibility_matrix, dtype=bool)
    if eligibility_matrix.shape != prefs.shape:
        raise ValueError("eligibility_matrix must have the same shape as prefs")

    # All targeted students are eligible
    eligible = targeted_mask
    eligible_indices = np.where(eligible)[0]

    # Per-student tracking
    n_students = len(pref_lengths)
    programs_added = np.zeros(n_students, dtype=int)

    oversubscribed_programs = np.array(
        list(
            dict.fromkeys(
                int(program)
                for program in oversubscribed_programs
                if 1 <= int(program) <= num_programs
            )
        ),
        dtype=int,
    )

    if max_add and len(eligible_indices) > 0 and len(oversubscribed_programs) > 0:
        for student_idx in eligible_indices:
            current_len = int(augmented_lengths[student_idx])
            submitted = list(
                dict.fromkeys(
                    int(program)
                    for program in augmented_prefs[student_idx, :current_len]
                    if program != 0
                )
            )
            submitted_set = set(submitted)
            designation_tail = list(
                dict.fromkeys(
                    int(program)
                    for program in augmented_prefs[student_idx, current_len:]
                    if program != 0 and int(program) not in submitted_set
                )
            )

            candidates = np.array(
                [
                    program
                    for program in oversubscribed_programs
                    if program not in submitted_set
                    and eligibility_matrix[student_idx, program - 1]
                ],
                dtype=int,
            )
            if not len(candidates):
                continue

            dists = distance_matrix[student_idx, candidates - 1]
            candidates = candidates[np.argsort(dists, kind="stable")]
            tail_set = set(designation_tail)
            free_slots = num_programs - len(submitted) - len(designation_tail)
            to_add = []
            for candidate in candidates:
                if candidate not in tail_set and free_slots <= 0:
                    continue
                to_add.append(int(candidate))
                if candidate not in tail_set:
                    free_slots -= 1
                if len(to_add) == max_add:
                    break

            if not to_add:
                continue

            promoted = set(to_add)
            new_submitted = to_add + submitted
            new_tail = [
                program for program in designation_tail if program not in promoted
            ]
            combined = new_submitted + new_tail
            augmented_prefs[student_idx] = 0
            augmented_prefs[student_idx, : len(combined)] = combined
            augmented_lengths[student_idx] = len(new_submitted)
            programs_added[student_idx] = len(to_add)

    # Build per-subgroup impact DataFrame
    impact_df = _compute_subgroup_impact(
        student_data,
        targeted_mask,
        eligible,
        programs_added,
        pref_lengths,
        augmented_lengths,
    )

    return augmented_prefs, augmented_lengths, impact_df


def _compute_subgroup_impact(
    student_data: pd.DataFrame,
    targeted_mask: np.ndarray,
    eligible_mask: np.ndarray,
    programs_added: np.ndarray,
    original_lengths: np.ndarray,
    augmented_lengths: np.ndarray,
) -> pd.DataFrame:
    """Compute per-subgroup impact statistics.

    Args:
        student_data: Student DataFrame with demographics.
        targeted_mask: Boolean mask of targeted students.
        eligible_mask: Boolean mask of eligible (targeted
            + short list) students.
        programs_added: Array of how many programs were
            added per student.
        original_lengths: Original preference list lengths.
        augmented_lengths: Lengths after augmentation.

    Returns:
        DataFrame with one row per subgroup and columns
        for counts and statistics.
    """
    ethnicity = (
        student_data["resolved_ethnicity"]
        .map(_normalize_ethnicity)
        .replace("", "Unknown")
        .to_numpy()
    )
    ctip = student_data["ctip1"].fillna(0).astype(int).to_numpy()
    augmented_mask = programs_added > 0

    # Define subgroups
    rows = []

    # Overall row
    rows.append(
        _subgroup_row(
            "ALL",
            np.ones(len(ethnicity), dtype=bool),
            targeted_mask,
            eligible_mask,
            augmented_mask,
            programs_added,
            original_lengths,
            augmented_lengths,
        )
    )

    # By CTIP status
    for ctip_val, ctip_label in [
        (1, "CTIP"),
        (0, "Non-CTIP"),
    ]:
        mask = ctip == ctip_val
        if mask.sum() == 0:
            continue
        rows.append(
            _subgroup_row(
                ctip_label,
                mask,
                targeted_mask,
                eligible_mask,
                augmented_mask,
                programs_added,
                original_lengths,
                augmented_lengths,
            )
        )

    # By CTIP × ethnicity
    unique_ethnicities = sorted(set(ethnicity))
    for ctip_val, ctip_label in [
        (1, "CTIP"),
        (0, "Non-CTIP"),
    ]:
        for eth in unique_ethnicities:
            mask = (ctip == ctip_val) & (ethnicity == eth)
            if mask.sum() < 3:
                continue
            rows.append(
                _subgroup_row(
                    f"{ctip_label} × {eth}",
                    mask,
                    targeted_mask,
                    eligible_mask,
                    augmented_mask,
                    programs_added,
                    original_lengths,
                    augmented_lengths,
                )
            )

    return pd.DataFrame(rows)


def _subgroup_row(
    subgroup_name: str,
    subgroup_mask: np.ndarray,
    targeted_mask: np.ndarray,
    eligible_mask: np.ndarray,
    augmented_mask: np.ndarray,
    programs_added: np.ndarray,
    original_lengths: np.ndarray,
    augmented_lengths: np.ndarray,
) -> dict:
    """Build one row of the impact table.

    Args:
        subgroup_name: Name for this subgroup.
        subgroup_mask: Boolean mask for this subgroup.
        targeted_mask: Boolean mask of targeted students.
        eligible_mask: Boolean mask of eligible students.
        augmented_mask: Boolean mask of augmented students.
        programs_added: Per-student count of added programs.
        original_lengths: Original list lengths.
        augmented_lengths: Post-augmentation list lengths.

    Returns:
        Dict with subgroup statistics.
    """
    in_sub = subgroup_mask
    n_total = int(in_sub.sum())
    n_targeted = int((in_sub & targeted_mask).sum())
    n_eligible = int((in_sub & eligible_mask).sum())
    n_augmented = int((in_sub & augmented_mask).sum())
    added_in_sub = programs_added[in_sub]
    total_added = int(added_in_sub.sum())

    return {
        "subgroup": subgroup_name,
        "n_students": n_total,
        "n_targeted": n_targeted,
        "n_eligible": n_eligible,
        "n_augmented": n_augmented,
        "total_programs_added": total_added,
        "avg_programs_added": (
            total_added / n_augmented if n_augmented > 0 else 0.0
        ),
        "pct_augmented": (
            100.0 * n_augmented / n_total if n_total > 0 else 0.0
        ),
        "avg_list_before": (
            float(original_lengths[in_sub].mean()) if n_total > 0 else 0.0
        ),
        "avg_list_after": (
            float(augmented_lengths[in_sub].mean()) if n_total > 0 else 0.0
        ),
    }
