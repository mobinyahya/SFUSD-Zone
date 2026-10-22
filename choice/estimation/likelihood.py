"""MNL with an outside option, fitted on rankings and enrolment decisions.

Inside utility is ``V_ij = school_FE[s_j] + type_FE[t_j] + X_ij . beta`` and
the outside option is ``V_i0 = Z_i . gamma``, with no constant. The constant
lives in the inside fixed effects instead: the school FEs carry the full level,
and ``type_FE`` is pinned to zero at ``reference_type``. So ``V_ij - V_i0`` is
utility relative to leaving SFUSD, and ``V_ij > V_i0`` means the program is
preferred to not enrolling.

The log-likelihood has two parts that share those utilities:

* **Rank** -- the exploded logit of exp8: each of a student's first ``top_k``
  ranked programs is chosen from the eligible programs not yet ranked. The
  outside option is not in these sets, so this part identifies differences
  between programs and says nothing about the level.
* **Enrolment** -- a student offered the programs ``O_i`` (their Main Round
  seat, plus the program they enrolled at if that is a different one) picks one
  of ``O_i`` or the outside option. Not enrolling anywhere picks the outside
  option. This is what identifies the level.

The two parts are treated as independent, although in a random-utility model
the enrolment decision reuses the Gumbel draws that produced the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from choice.estimation.exp8 import Design, inside_utilities

OUTSIDE = -1


@dataclass(frozen=True)
class ChoiceData:
    """Observed choices aligned to a :class:`Design`.

    ``rankings[i]`` is a list of design columns in rank order (empty when the
    student contributes no ranking). ``offers[i]`` lists the columns offered at
    enrolment and ``enrolled[i]`` is the chosen column, ``OUTSIDE``, or ``None``
    when the student contributes no enrolment decision.
    """

    design: Design
    rankings: list[list[int]]
    offers: list[list[int]]
    enrolled: list[int | None]
    outside_features: np.ndarray
    outside_names: tuple[str, ...]
    top_k: int = 5
    reference_type: str = "GE"


@dataclass
class _Arrays:
    ranked: np.ndarray  # (n, R) column, or -1
    available: np.ndarray  # (n, R, J) bool
    offer_cols: np.ndarray  # (m, O) column, or -1
    offer_rows: np.ndarray  # (m,)
    chosen_slot: np.ndarray  # (m,) 0 = outside, k = offer_cols[:, k - 1]
    school_index: np.ndarray
    type_index: np.ndarray
    free_types: np.ndarray = field(default_factory=lambda: np.array([], int))


@dataclass(frozen=True)
class FitResult:
    names: tuple[str, ...]
    theta: np.ndarray
    std_error: np.ndarray
    log_likelihood: float
    rank_log_likelihood: float
    enrolment_log_likelihood: float
    converged: bool
    message: str
    iterations: int
    max_abs_gradient: float


class OutsideOptionMNL:
    """Penalised maximum likelihood for :class:`ChoiceData`."""

    def __init__(self, data: ChoiceData, *, ridge: float = 1e-3):
        self.data = data
        self.ridge = ridge
        design = data.design
        n, J, K = design.features.shape
        self.schools = np.unique(design.school_ids)
        self.types = np.unique(design.program_types)
        if data.reference_type not in self.types:
            raise ValueError(f"Reference type {data.reference_type!r} is not offered.")
        free_types = np.array([t for t in self.types if t != data.reference_type])
        school_index = np.searchsorted(self.schools, design.school_ids)
        type_index = np.searchsorted(self.types, design.program_types)

        R = data.top_k
        ranked = np.full((n, R), -1, dtype=int)
        for i, ranking in enumerate(data.rankings):
            for r, j in enumerate(ranking[:R]):
                if not design.eligible[i, j]:
                    raise ValueError(
                        f"Student {design.student_ids[i]} ranks ineligible column {j}."
                    )
                ranked[i, r] = j
        available = np.repeat(design.eligible[:, None, :], R, axis=1)
        for r in range(1, R):
            available[:, r] = available[:, r - 1]
            rows = np.nonzero(ranked[:, r - 1] >= 0)[0]
            available[rows, r, ranked[rows, r - 1]] = False

        rows, cols, slots = [], [], []
        width = max([len(o) for o in data.offers] + [1])
        for i, (offer, choice) in enumerate(zip(data.offers, data.enrolled)):
            if choice is None:
                continue
            if choice != OUTSIDE and choice not in offer:
                raise ValueError(
                    f"Student {design.student_ids[i]} enrols at a program not offered."
                )
            if any(not design.eligible[i, j] for j in offer):
                raise ValueError(
                    f"Student {design.student_ids[i]} is offered an ineligible program."
                )
            rows.append(i)
            cols.append(list(offer) + [-1] * (width - len(offer)))
            slots.append(0 if choice == OUTSIDE else offer.index(choice) + 1)
        self.arrays = _Arrays(
            ranked=ranked,
            available=available,
            offer_cols=np.array(cols, dtype=int).reshape(-1, width),
            offer_rows=np.array(rows, dtype=int),
            chosen_slot=np.array(slots, dtype=int),
            school_index=school_index,
            type_index=type_index,
            free_types=np.searchsorted(self.types, free_types),
        )
        self.names = (
            *(f"school_{s}" for s in self.schools),
            *(f"program_type_{t}" for t in free_types),
            *design.feature_names,
            *data.outside_names,
        )
        self._sizes = (len(self.schools), len(free_types), K, len(data.outside_names))

    # ------------------------------------------------------------------ #
    def unpack(self, theta: np.ndarray):
        S, T, K, L = self._sizes
        school_fe = theta[:S]
        type_fe = np.zeros(len(self.types))
        type_fe[self.arrays.free_types] = theta[S : S + T]
        return school_fe, type_fe, theta[S + T : S + T + K], theta[S + T + K :]

    def utilities(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inside utilities (``-inf`` where ineligible) and outside utilities."""
        school_fe, type_fe, beta, gamma = self.unpack(theta)
        a = self.arrays
        fixed = school_fe[a.school_index] + type_fe[a.type_index]
        inside = inside_utilities(self.data.design, fixed, beta)
        return inside, self.data.outside_features @ gamma

    def _parts(self, theta: np.ndarray):
        a = self.arrays
        V, V0 = self.utilities(theta)
        n = V.shape[0]

        # Rank part.
        valid = a.ranked >= 0
        logits = np.where(a.available, V[:, None, :], -np.inf)
        lse = logsumexp(logits, axis=2)
        chosen = np.take_along_axis(V, np.maximum(a.ranked, 0), axis=1)
        rank_ll = float(np.sum(np.where(valid, chosen - lse, 0.0)))
        prob = np.exp(logits - lse[..., None]) * valid[..., None]
        dV = -prob.sum(axis=1)
        rr, kk = np.nonzero(valid)
        np.add.at(dV, (rr, a.ranked[rr, kk]), 1.0)

        # Enrolment part.
        dV0 = np.zeros(n)
        enrol_ll = 0.0
        if len(a.offer_rows):
            rows = a.offer_rows
            present = a.offer_cols >= 0
            inside = np.where(
                present, V[rows[:, None], np.maximum(a.offer_cols, 0)], -np.inf
            )
            options = np.concatenate([V0[rows, None], inside], axis=1)
            norm = logsumexp(options, axis=1)
            enrol_ll = float(
                np.sum(options[np.arange(len(rows)), a.chosen_slot] - norm)
            )
            p = np.exp(options - norm[:, None])
            g = -p
            g[np.arange(len(rows)), a.chosen_slot] += 1.0
            np.add.at(dV0, rows, g[:, 0])
            r2, k2 = np.nonzero(present)
            np.add.at(dV, (rows[r2], a.offer_cols[r2, k2]), g[r2, k2 + 1])
        return rank_ll, enrol_ll, dV, dV0

    def objective(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Penalised negative log-likelihood and its gradient."""
        rank_ll, enrol_ll, dV, dV0 = self._parts(theta)
        a = self.arrays
        S, T, K, L = self._sizes
        column_grad = dV.sum(axis=0)
        grad_school = np.bincount(a.school_index, weights=column_grad, minlength=S)
        grad_type = np.bincount(
            a.type_index, weights=column_grad, minlength=len(self.types)
        )[a.free_types]
        grad_beta = np.einsum("ij,ijk->k", dV, self.data.design.features)
        grad_gamma = self.data.outside_features.T @ dV0
        grad = np.concatenate([grad_school, grad_type, grad_beta, grad_gamma])
        value = -(rank_ll + enrol_ll) + self.ridge * float(theta @ theta)
        return value, -grad + 2 * self.ridge * theta

    def fit(
        self, *, theta0: np.ndarray | None = None, maxiter: int = 5000
    ) -> FitResult:
        theta0 = np.zeros(len(self.names)) if theta0 is None else theta0
        result = minimize(
            self.objective,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "maxfun": 4 * maxiter,
                "gtol": 1e-6,
                "ftol": 1e-14,
            },
        )
        theta = result.x
        _, grad = self.objective(theta)
        rank_ll, enrol_ll, _, _ = self._parts(theta)
        return FitResult(
            names=self.names,
            theta=theta,
            std_error=self.standard_errors(theta),
            log_likelihood=rank_ll + enrol_ll,
            rank_log_likelihood=rank_ll,
            enrolment_log_likelihood=enrol_ll,
            converged=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            max_abs_gradient=float(np.max(np.abs(grad))),
        )

    def standard_errors(self, theta: np.ndarray, step: float = 1e-5) -> np.ndarray:
        """Square roots of the inverse penalised Hessian's diagonal.

        The Hessian is a central difference of the analytic gradient.
        """
        P = len(theta)
        hessian = np.empty((P, P))
        for p in range(P):
            e = np.zeros(P)
            e[p] = step
            hessian[p] = (
                self.objective(theta + e)[1] - self.objective(theta - e)[1]
            ) / (2 * step)
        hessian = (hessian + hessian.T) / 2
        return np.sqrt(np.clip(np.diag(np.linalg.inv(hessian)), 0, None))
