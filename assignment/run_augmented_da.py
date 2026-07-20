"""Run DA with augmented preference lists.

This script loads a config, initializes the market, and runs the
DA mechanism for all policy subconfigs — but intercepts
preferences after generation to augment them for targeted
demographic groups before passing to DA.

Usage:
    python run_augmented_da.py \
        --config_path configs/custom_configs/augmented_da_2324.yaml

@author: Edouard Rabasse
@date: 02-19-2026
"""

import os
import re
import sys
import warnings
from collections.abc import Generator
from itertools import product

import click
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.getcwd())

from student_assignment.configerator import (
    Configerator,  # ty:ignore[unresolved-import]
)
from student_assignment.market_generator.list_augmentation import (
    augment_preferences,
    identify_oversubscribed_programs,
    identify_targeted_students,
)
from student_assignment.market_generator.policy import (
    Policy,  # ty:ignore[unresolved-import]
)
from student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,  # ty:ignore[unresolved-import]
)


def resolve_variables(item: object, root_config: dict) -> object:
    """Recursively replace ${var} in strings.

    Args:
        item: The config item (dict, list, str, or other).
        root_config: The root config dict for variable lookup.

    Returns:
        The item with all variables resolved.
    """
    if isinstance(item, dict):
        return {k: resolve_variables(v, root_config) for k, v in item.items()}
    elif isinstance(item, list):
        return [resolve_variables(v, root_config) for v in item]
    elif isinstance(item, str):
        pattern = re.compile(r"\$\{([^\}]+)\}")

        def replace(match: re.Match) -> str:
            key = match.group(1)
            if key in root_config:
                return str(root_config[key])
            warnings.warn(
                f"Could not resolve variable ${{{key}}}", stacklevel=2
            )
            return match.group(0)

        return pattern.sub(replace, item)
    else:
        return item


class AugmentedMarketGenerator(MarketGenerator):
    """MarketGenerator that augments real preferences.

    Overrides ``_simulate_policy`` to inject list
    augmentation between preference generation and DA.
    """

    def _simulate_policy(
        self, policy: str, iteration: int
    ) -> Generator[pd.DataFrame | None, None, None]:
        """Simulate a single iteration with augmentation.

        Identical to ``MarketGenerator._simulate_policy``
        except that when using real preferences (utility
        model disabled), preferences are augmented for
        targeted demographic groups before DA.

        Args:
            policy: Policy name.
            iteration: Iteration number.

        Yields:
            Optional[pd.DataFrame]: Assignment dataframe.
        """
        self.priority_generator.generate_base_priorities(policy)

        if self.config["utility-model"]["enable"]:
            prefs = self.preference_generator.get_utility_model_preferences_after_truncation()
        else:
            prefs = self.preference_generator.initialize_real_preferences()

        # --- AUGMENTATION HOOK ---
        # Applies whether prefs came from the utility model or real submissions.
        aug_config = self.config.get("list-augmentation", {})
        if aug_config.get("enable", False):
            prefs = self._augment_real_preferences(prefs, aug_config)

        for ctip, rounds_merged, ties in product(
            self.config["ctip-options"],
            self.config["rounds-merged-options"],
            self.config["ties-options"],
        ):
            policy_data = Policy(
                name=policy,
                ctip=ctip,
                rounds_merged=rounds_merged,
                tiebreaker=ties,
            )

            priorities = self.priority_generator.set_policy_specific_priorities(
                policy_data, prefs
            )

            if self.config["guard-rails"] != -1:
                (
                    match,
                    in_zone_rank,
                    cutoffs,
                ) = self._generate_assignment_with_guardrails(prefs, priorities)
            else:
                (
                    match,
                    in_zone_rank,
                    cutoffs,
                ) = self._generate_assignment(prefs, priorities)

            yield self._save_assignment(
                prefs,
                policy_data,
                iteration,
                match,
                in_zone_rank,
                cutoffs,
            )

    def _augment_real_preferences(
        self,
        prefs: np.ndarray,
        aug_config: dict,
    ) -> np.ndarray:
        """Augment real preferences for targeted students.

        Args:
            prefs: The real preference matrix (n, p).
            aug_config: The list-augmentation config dict.

        Returns:
            Augmented preference matrix.
        """
        import pathlib

        pref_lengths = self.preference_generator.pref_length.copy()

        # Identify targeted students
        targeted = identify_targeted_students(
            self.students.student_data,
            pref_lengths,
            aug_config,
        )

        # Identify oversubscribed programs
        capacity = self.programs.capacity.to_numpy()
        oversub_programs = identify_oversubscribed_programs(
            prefs,
            capacity,
            self.programs.school_to_indices,
            aug_config,
        )

        # Build distance matrix aligned to program indices
        # distance_data: DataFrame (studentno × program_id)
        # We need a numpy array (n_students × n_programs)
        # ordered by program index (1-indexed in prefs)
        dist_df = self.students.distance_data
        # Reorder columns to match programno ordering
        program_ids_ordered = [
            self.programs.codes[i + 1]
            for i in range(self.programs.num_programs)
        ]
        dist_aligned = dist_df.reindex(columns=program_ids_ordered).fillna(9999)
        # Reindex rows to match student order
        student_order = self.students.student_data.index
        dist_aligned = dist_aligned.reindex(student_order).fillna(9999)
        distance_matrix = dist_aligned.to_numpy()

        # Augment preferences (returns impact_df)
        prefs, new_pref_lengths, impact_df = augment_preferences(
            prefs,
            pref_lengths,
            targeted,
            oversub_programs,
            self.students.student_data,
            distance_matrix,
            aug_config,
        )

        # Update pref_length on the preference generator
        self.preference_generator.pref_length = new_pref_lengths

        # Save impact stats to CSV
        if self.config.get("save-assignment", True):
            output_dir = pathlib.Path(
                self.config["paths"]["assignment-folder"]
            ).expanduser()
            subconfig_name = self.config.get("subconfig-name", "default")
            impact_dir = output_dir / subconfig_name
            impact_dir.mkdir(parents=True, exist_ok=True)
            impact_path = impact_dir / "augmentation_impact.csv"
            impact_df.to_csv(impact_path, index=False)

        return prefs


@click.command()
@click.option(
    "--config-path",
    "--config_path",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the augmented DA config file.",
)
def run_augmented_da(config_path: str) -> None:
    """Run DA with augmented preference lists.

    Args:
        config_path: Path to the YAML config file.
    """
    with open(config_path) as config_file:
        raw_config = yaml.safe_load(config_file)

    custom_config = resolve_variables(raw_config, raw_config)

    # Initialize Configerator singleton with our config
    configurator = Configerator()
    configurator._config = custom_config
    configurator._original_config = custom_config

    subconfigs_list = custom_config.get("subconfigs", [])
    configurator.subconfigs = iter(subconfigs_list)

    # Use AugmentedMarketGenerator instead of plain
    # MarketGenerator
    market = AugmentedMarketGenerator()
    market.simulate()


if __name__ == "__main__":
    run_augmented_da()
