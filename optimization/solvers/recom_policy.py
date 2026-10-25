"""How a ReCom walk chooses: constraint weights, pair selection, cut selection.

Every choice reads one merit, the weighted linear violation

    M(x) = sum_{z,k} w_zk max(0, g_zk(x))  [+ w_cut max(0, g_cut(x))]

where ``g_zk`` is row ``k`` of zone ``z``'s residual in percent of the zone's
students (``g <= 0`` is feasible) and ``g_cut`` the cut-edge cap's. A linear
penalty is exact and, unlike a squared one, does not reward spreading a
violation over many zones. Three independent pieces then make the choices:

``Weights`` (``recom_weights``)
    ``fixed``: ``w = 1``.  ``hedge``: multiplicative weights over the rows,
    raised by how often a row was violated during the last burst.  ``dual``:
    ``w = 1 + lambda``, projected subgradient ascent on the multipliers.
``PairSelection`` (``recom_pair_selection``)
    Which two adjacent zones merge: ``uniform``; ``proportional`` to the
    pair's pressure with an exploration floor; ``softmax`` over pairs;
    ``zone`` (a zone by pressure, then a uniform neighbour); ``exp3``, the
    proportional rule times discounted EXP3-IX weights learned per pair.
    ``recom_pair_score`` says what a pair's score is: ``pressure``, the two
    zones' own pressures added, or ``pooled``, the part of each zone's
    pressure the other's slack could absorb - an upper bound on what one
    move on the pair can repair.
``CutSelection`` (``recom_cut_selection``)
    Which tree edge to cut: ``uniform`` over the pair-feasible cuts (all cuts
    if none is), or ``softmax`` of ``-(M_ab + mu * boundary) / tau``.

A ``Temperature`` turns scores into probabilities, either at a fixed ``tau``
or, with ``recom_entropy`` set, at whatever ``tau`` gives the distribution a
normalized entropy ``h`` (``n**h`` effective choices) - the same randomness
whatever the scale of the scores.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from optimization.solvers.recom import _Cut, _ReComContext, _State

_EPS = 1e-6

WEIGHTS = ("fixed", "hedge", "dual")
PAIR_SELECTIONS = ("uniform", "proportional", "softmax", "zone", "exp3")
PAIR_SCORES = ("pressure", "pooled")
CUT_SELECTIONS = ("uniform", "softmax")
TREE_SAMPLERS = ("random_mst", "wilson_uniform")


@dataclass(frozen=True)
class ReComSettings:
    """Everything that distinguishes one ReCom method from another.

    The defaults are the recommended method; ``PRESETS`` names the others.
    ``objective_weight`` (``mu``) only applies when the objective is local
    (cut edges): it prices one boundary unit against one percentage point of
    violation in cut choice. Other objectives enter only burst selection.
    """

    burst_length: int = 25
    reject_infeasible: bool = False
    weights: str = "fixed"
    pair_selection: str = "softmax"
    pair_score: str = "pressure"
    cut_selection: str = "softmax"
    # Fresh independent U(0, 1) edge weights and their MST per proposal, or
    # Wilson's uniform spanning tree. These give different tree distributions.
    tree_sampler: str = "random_mst"
    temperature: float = 1.0
    entropy: float | None = 0.3
    exploration: float = 0.05
    learning_rate: float | None = None
    bandit_window: float = 160_000.0
    objective_weight: float = 0.0

    def __post_init__(self) -> None:
        for name, choices in (
            ("weights", WEIGHTS),
            ("pair_selection", PAIR_SELECTIONS),
            ("pair_score", PAIR_SCORES),
            ("cut_selection", CUT_SELECTIONS),
            ("tree_sampler", TREE_SAMPLERS),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"recom_{name} must be one of: {', '.join(choices)}.")
        if isinstance(self.burst_length, bool) or not isinstance(
            self.burst_length, int
        ):
            raise ValueError("recom_burst_length must be a non-negative integer.")
        if self.burst_length < 0:
            raise ValueError("recom_burst_length must be a non-negative integer.")
        if not isinstance(self.reject_infeasible, bool):
            raise ValueError("recom_reject_infeasible must be a boolean.")
        positive = {
            "temperature": self.temperature,
            "bandit_window": self.bandit_window,
        }
        if self.learning_rate is not None:
            positive["learning_rate"] = self.learning_rate
        for name, value in positive.items():
            if not _finite(value) or value <= 0:
                raise ValueError(f"recom_{name} must be a positive number.")
        if self.entropy is not None and not (
            _finite(self.entropy) and 0 < self.entropy <= 1
        ):
            raise ValueError("recom_entropy must lie in (0, 1], or be null.")
        if not (_finite(self.exploration) and 0 <= self.exploration <= 1):
            raise ValueError("recom_exploration must lie in [0, 1].")
        if not _finite(self.objective_weight) or self.objective_weight < 0:
            raise ValueError("recom_objective_weight must be a non-negative number.")

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "ReComSettings":
        return cls(
            **{
                f.name: options[f"recom_{f.name}"]
                for f in fields(cls)
                if f"recom_{f.name}" in options
            }
        )

    def as_options(self) -> dict[str, Any]:
        return {f"recom_{f.name}": getattr(self, f.name) for f in fields(self)}


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


# The studied methods, as settings. Only those that differ from the defaults
# are named.
_UNIFORM = dict(pair_selection="uniform", cut_selection="uniform", entropy=None)
PRESETS: dict[str, ReComSettings] = {
    # One chain that rejects every move leaving the feasible set.
    "chain": ReComSettings(burst_length=0, reject_infeasible=True, **_UNIFORM),
    "uniform_bursts": ReComSettings(**_UNIFORM),
    # Fixed weights; pairs and cuts at normalized entropy 0.3. The default.
    "entropy": ReComSettings(),
    "exact_penalty": ReComSettings(
        pair_selection="proportional", entropy=None, objective_weight=1.0
    ),
    "bandit": ReComSettings(pair_selection="exp3", entropy=None, objective_weight=1.0),
    "hedge": ReComSettings(
        weights="hedge",
        pair_selection="zone",
        exploration=0.0,
        temperature=1.2,
        entropy=None,
    ),
    "dual": ReComSettings(
        weights="dual",
        pair_selection="zone",
        exploration=0.5,
        entropy=None,
        objective_weight=1.0,
    ),
}


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Temperature:
    """``p_i ~ exp(s_i / tau)``, or the ``tau`` that holds entropy at ``h``."""

    tau: float = 1.0
    entropy: float | None = None

    def probabilities(self, scores: list[float]) -> list[float]:
        if self.entropy is None:
            return softmax([s / self.tau for s in scores])
        beta = entropy_beta(scores, self.entropy)
        if math.isinf(beta):  # ties alone reach the target: share them
            top = max(scores)
            return softmax([0.0 if s >= top - _EPS else -math.inf for s in scores])
        return softmax([beta * s for s in scores])


def softmax(logits: list[float]) -> list[float]:
    """Normalize log-weights; ``-inf`` entries get no mass, all ``-inf`` is uniform."""
    top = max(logits, default=-math.inf)
    if not math.isfinite(top):
        return [1.0 / len(logits)] * len(logits)
    weights = [math.exp(x - top) for x in logits]
    total = sum(weights)
    return [w / total for w in weights]


def entropy_beta(scores: list[float], target: float) -> float:
    """``beta >= 0`` with ``H(softmax(beta * scores)) / log n == target``.

    The maximum-entropy reading of the temperature: ``beta`` is the dual
    variable of an entropy constraint, so ``target`` is dimensionless.
    ``inf`` when the tied maxima alone already reach it; 0 when all tie.
    """
    n = len(scores)
    top = max(scores)
    spread = top - min(scores)
    if n <= 1 or spread <= 1e-12:
        return 0.0
    log_n = math.log(n)
    ties = sum(s >= top - 1e-12 * max(1.0, spread) for s in scores)
    if math.log(ties) / log_n >= target - 1e-9:
        return math.inf
    y = [(s - top) / spread for s in scores]
    goal = target * log_n

    def entropy(b: float) -> float:
        w = [math.exp(b * yi) for yi in y]
        z = sum(w)
        return math.log(z) - b * sum(wi * yi for wi, yi in zip(w, y)) / z

    lo, hi = 0.0, 1.0
    while entropy(hi) > goal and hi < 1e12:
        lo, hi = hi, 8.0 * hi
    while hi - lo > 1e-3 * hi:  # entropy falls monotonically in b
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if entropy(mid) > goal else (lo, mid)
    return 0.5 * (lo + hi) / spread


def _sample(rng, probabilities: list[float]) -> int:
    return rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]


# ---------------------------------------------------------------------------
# Constraint weights
# ---------------------------------------------------------------------------


class Weights:
    """Fixed weights ``w = 1``: the merit is the total (L1) violation.

    Rows are indexed ``zone * R + k``; the cut-edge cap, if any, is row
    ``Z * R``. ``zone`` and ``cut`` price clipped violations; ``pressure`` is
    what pair selection targets; ``scale`` is the unit a pair's cut scores
    are measured in. ``observe`` sees every accepted state and ``update``
    the state each burst restarts from.
    """

    def __init__(self, context: "_ReComContext", settings: ReComSettings) -> None:
        self.context = context
        self.rows = context.zone_row_count
        self.size = context.zone_count * self.rows + (context.cut_row is not None)

    def weight(self, row: int) -> float:
        return 1.0

    def zone(self, zone: int, violations: tuple[float, ...]) -> float:
        base = zone * self.rows
        return sum(self.weight(base + k) * v for k, v in enumerate(violations) if v > 0)

    def cut(self, violations: tuple[float, ...]) -> float:
        row = self.context.cut_row
        return 0.0 if row is None else self.weight(self.size - 1) * violations[row]

    def merit(self, state: "_State") -> float:
        return sum(
            self.zone(z, v) for z, v in enumerate(state.zone_violations)
        ) + self.cut(state.violations)

    def pressure(self, zone: int, violations: tuple[float, ...]) -> float:
        return self.zone(zone, violations)

    def row_pressure(self, row: int, violation: float, part: float) -> float:
        """What repairing ``part`` of one row's ``violation`` is worth in pressure."""
        return self.weight(row) * part

    def scale(self, zone_a: int, zone_b: int) -> float:
        return 1.0

    def observe(self, state: "_State") -> None:
        pass

    def update(self, state: "_State") -> None:
        pass

    def metadata(self) -> dict[str, object]:
        return {}


class HedgeWeights(Weights):
    """Fixed-share Hedge over the constraint rows (Arora-Hazan-Kale).

    A row's loss is the fraction of the burst's walk on which it was
    violated: in [0, 1], so no scale to choose, and it sees the last tiny
    residuals a magnitude loss would not. After each burst
    ``p <- p exp(eta_t l)``, ``eta_t = eta sqrt(ln K / t)`` (``eta`` is
    ``recom_learning_rate``, default 1), then fixed share ``alpha_t =
    1/(t+1)``, which keeps every row above ``alpha_t / K``. Weights are
    ``K p``, mean 1. Only their ratios matter, so a pair's cuts are scored in
    units of its mean weight, and pressure is the weight of the rows violated
    now (the mixed 0-1 loss).
    """

    def __init__(self, context, settings):
        super().__init__(context, settings)
        self.eta = 1.0 if settings.learning_rate is None else settings.learning_rate
        self.w = [1.0] * self.size
        self.hits = [0] * self.size
        self.moves = 0
        self.rounds = 0

    def weight(self, row):
        return self.w[row]

    def pressure(self, zone, violations):
        base = zone * self.rows
        return sum(self.w[base + k] for k, v in enumerate(violations) if v > _EPS)

    def row_pressure(self, row, violation, part):
        return self.w[row] * part / violation

    def scale(self, zone_a, zone_b):
        r = self.rows
        mass = sum(self.w[zone_a * r : zone_a * r + r]) + sum(
            self.w[zone_b * r : zone_b * r + r]
        )
        return mass / (2 * r)

    def observe(self, state):
        self.moves += 1
        r = self.rows
        for zone, violations in enumerate(state.zone_violations):
            for k, v in enumerate(violations):
                if v > _EPS:
                    self.hits[zone * r + k] += 1
        if (
            self.context.cut_row is not None
            and state.violations[self.context.cut_row] > _EPS
        ):
            self.hits[-1] += 1

    def update(self, state):
        self.rounds += 1
        n, k = max(1, self.moves), self.size
        eta = self.eta * math.sqrt(math.log(max(k, 2)) / self.rounds)
        w = [x * math.exp(eta * h / n) for x, h in zip(self.w, self.hits)]
        alpha = 1.0 / (self.rounds + 1)
        total = sum(w)
        self.w = [k * (1.0 - alpha) * x / total + alpha for x in w]
        self.hits = [0] * k
        self.moves = 0

    def metadata(self):
        p = [x / self.size for x in self.w]
        entropy = -sum(x * math.log(x) for x in p if x > 0)
        return {
            "hedge_rounds": self.rounds,
            "hedge_effective_rows": math.exp(entropy),
            "hedge_max_weight": max(self.w),
        }


class DualWeights(Weights):
    """``w = 1 + lambda``: dual ascent on the L1 Lagrangian.

    The 1 is the exact-penalty floor. After each burst, on the state the
    next one starts from, ``lambda <- max(0, lambda + alpha clip(g, -1, 1))``
    (``alpha`` is ``recom_learning_rate``, default 0.05): projected
    subgradient ascent on ``clip(g) <= 0``, the same feasible set, with a
    step that does not depend on whether a residual is 0.01 or 4,000
    percent. Multipliers rise while their row stays violated and fall on
    slack, so they stay bounded.
    """

    def __init__(self, context, settings):
        super().__init__(context, settings)
        self.alpha = 0.05 if settings.learning_rate is None else settings.learning_rate
        self.lam = [0.0] * self.size
        self.max_lambda = 0.0

    def weight(self, row):
        return 1.0 + self.lam[row]

    def update(self, state):
        residuals = [g for zone in state.zone_residuals for g in zone]
        if self.context.cut_row is not None:
            residuals.append(self.context.cut_residual(state.cut_edges))
        for row, g in enumerate(residuals):
            step = self.alpha * max(-1.0, min(1.0, g))
            self.lam[row] = max(0.0, self.lam[row] + step)
        self.max_lambda = max(self.max_lambda, max(self.lam))

    def metadata(self):
        return {
            "dual_max_lambda": self.max_lambda,
            "dual_positive_multipliers": sum(x > 0 for x in self.lam),
        }


_WEIGHTS = {"fixed": Weights, "hedge": HedgeWeights, "dual": DualWeights}


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------


class PairSelection:
    """Uniform over the adjacent zone pairs.

    ``mu`` is the local objective's price per boundary unit (0 when the
    objective is not local), which ``exp3`` needs to tell a move that can
    change the merit from one that cannot.
    """

    def __init__(
        self, weights: Weights, settings: ReComSettings, rng, mu: float
    ) -> None:
        self.weights = weights
        self.settings = settings
        self.rng = rng
        self.mu = mu
        self.temperature = Temperature(settings.temperature, settings.entropy)

    def choose(self, state: "_State", pairs: list[tuple[int, int]]) -> tuple[int, int]:
        return pairs[self.rng.randrange(len(pairs))]

    def feedback(self, before: float, after: float) -> None:
        """The merit change of the last chosen pair's move (``exp3`` learns it)."""

    def _pressures(self, state: "_State", pairs) -> list[float]:
        if self.settings.pair_score == "pooled":
            return [self._pooled(state, a, b) for a, b in pairs]
        zone = [
            self.weights.pressure(z, v) for z, v in enumerate(state.zone_violations)
        ]
        return [zone[a] + zone[b] for a, b in pairs]

    def _pooled(self, state: "_State", a: int, b: int) -> float:
        """The pressure one move on ``a`` and ``b`` could at most repair.

        A move only redistributes the union, and every row is linear in the
        counts, so the union's two-zone residual is ``c_a + c_b`` (counts,
        ``c = g s / 100``). Since ``max(0, c_a' + c_b') <= max(0, c_a') +
        max(0, c_b')``, no split leaves less than the pooled violation: what a
        zone can shed is at most the other's slack, ``min(c_a+, c_b-)``.
        """
        return self._transfer(state, a, b) + self._transfer(state, b, a)

    def _transfer(self, state: "_State", a: int, b: int) -> float:
        rows = self.weights.rows
        students = self.weights.context.row_students
        s_a, s_b = students(state.zone_stats[a]), students(state.zone_stats[b])
        g_b = state.zone_residuals[b]
        total = 0.0
        for k, v in enumerate(state.zone_violations[a]):
            if v <= _EPS or g_b[k] >= 0:
                continue
            slack = -g_b[k] * s_b[k] / s_a[k] if s_a[k] > 0 else math.inf
            total += self.weights.row_pressure(a * rows + k, v, min(v, slack))
        return total

    def metadata(self) -> dict[str, object]:
        return {}


class ProportionalPairs(PairSelection):
    """``(1 - gamma) P_ab / sum P + gamma / K``, uniform once nothing is violated.

    ``P_ab`` is the two zones' pressure; ``gamma`` (``recom_exploration``) is
    the exploration floor that keeps every pair reachable.
    """

    def choose(self, state, pairs):
        return pairs[_sample(self.rng, self._probabilities(state, pairs))]

    def _probabilities(self, state, pairs):
        p = self._pressures(state, pairs)
        total, k, gamma = sum(p), len(pairs), self.settings.exploration
        if total <= _EPS:
            return [1.0 / k] * k
        return [(1.0 - gamma) * x / total + gamma / k for x in p]


class SoftmaxPairs(PairSelection):
    """``p_ab ~ exp(P_ab / tau)``, or at normalized entropy ``h``."""

    def choose(self, state, pairs):
        probabilities = self.temperature.probabilities(self._pressures(state, pairs))
        return pairs[_sample(self.rng, probabilities)]


class ZonePairs(PairSelection):
    """Zone ``a`` by ``(1 - gamma) P_a / sum P + gamma / n``, then a neighbour.

    The neighbour is uniform under ``pressure``; under ``pooled`` it is drawn
    the same way by the pair's pooled score.
    """

    def choose(self, state, pairs):
        neighbours: dict[int, list[int]] = defaultdict(list)
        for a, b in pairs:
            neighbours[a].append(b)
            neighbours[b].append(a)
        zones = sorted(neighbours)
        p = [self.weights.pressure(z, state.zone_violations[z]) for z in zones]
        total, n, gamma = sum(p), len(zones), self.settings.exploration
        probabilities = (
            [(1.0 - gamma) * x / total + gamma / n for x in p]
            if total > _EPS
            else [1.0 / n] * n
        )
        zone = zones[_sample(self.rng, probabilities)]
        others = neighbours[zone]
        if self.settings.pair_score == "pooled":
            # The neighbour by what the pair can repair, with the same floor.
            q = [self._pooled(state, zone, other) for other in others]
            total, m = sum(q), len(others)
            other = others[
                _sample(self.rng, [(1.0 - gamma) * x / total + gamma / m for x in q])
                if total > _EPS
                else self.rng.randrange(m)
            ]
        else:
            other = others[self.rng.randrange(len(others))]
        return (zone, other) if zone < other else (other, zone)


class Exp3Pairs(ProportionalPairs):
    """Discounted EXP3-IX over pairs, times the proportional rule as an expert.

    ``pi_k ~ P_k exp(-eta L_k)``, ``p_k = (1 - gamma) pi_k + gamma / K``,
    ``L_k <- lambda L_k + 1{k} (1 - r) / (p_k + eta / 2)``. The reward is
    ``r = sigmoid((M_before - M_after) / tau)``: 1/2 for no change. With the
    window ``W = 1 / (1 - lambda)`` (``recom_bandit_window``) in place of the
    horizon, EXP3's bound gives ``gamma = sqrt(K ln K / ((e - 1) W))`` and
    ``eta = gamma / K`` (Auer et al. 2002). A move on two unviolated zones
    under a non-local objective repairs nothing and is not an observation.
    """

    def __init__(self, weights, settings, rng, mu):
        super().__init__(weights, settings, rng, mu)
        self.discount = 1.0 - 1.0 / settings.bandit_window
        self.scale = 1.0  # lazy discounting: stored losses are L / discount^t
        self.loss: dict[tuple[int, int], float] = defaultdict(float)
        self.pending: tuple[tuple[int, int], float, float, bool] | None = None
        self.observations = 0

    def choose(self, state, pairs):
        self.scale /= self.discount
        if self.scale > 1e200:
            for pair in self.loss:
                self.loss[pair] /= self.scale
            self.scale = 1.0
        k = len(pairs)
        gamma = min(
            1.0,
            math.sqrt(
                k * math.log(max(k, 2)) / ((math.e - 1.0) * self.settings.bandit_window)
            ),
        )
        eta = gamma / k
        p = self._pressures(state, pairs)
        top = max(p)
        prior = (
            [math.log(max(x, 1e-9 * top) / top) for x in p] if top > _EPS else [0.0] * k
        )
        learned = softmax(
            [q - eta * self.loss[pair] / self.scale for q, pair in zip(prior, pairs)]
        )
        probabilities = [(1.0 - gamma) * x + gamma / k for x in learned]
        index = _sample(self.rng, probabilities)
        null = top <= _EPS or p[index] <= _EPS
        self.pending = (pairs[index], probabilities[index], eta, null)
        return pairs[index]

    def feedback(self, before, after):
        if self.pending is None:
            return
        pair, probability, eta, null = self.pending
        self.pending = None
        if null and self.mu == 0.0:
            return
        reward = _sigmoid((before - after) / self.settings.temperature)
        self.loss[pair] += (1.0 - reward) / (probability + eta / 2.0) * self.scale
        self.observations += 1

    def metadata(self):
        return {"exp3_observations": self.observations, "exp3_arms": len(self.loss)}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


_PAIRS = {
    "uniform": PairSelection,
    "proportional": ProportionalPairs,
    "softmax": SoftmaxPairs,
    "zone": ZonePairs,
    "exp3": Exp3Pairs,
}


# ---------------------------------------------------------------------------
# The policy: weights, pairs and cuts together
# ---------------------------------------------------------------------------


class Policy:
    """The four adaptive decisions of a walk, from one merit.

    ``merit`` ranks infeasible restart candidates; ``choose_pair`` and
    ``cut_probabilities`` drive proposals; ``moved`` and ``end_burst`` feed
    the learners. ``objective_local`` says whether the objective is known per
    cut (cut edges); only then does ``objective_weight`` enter cut scores.
    """

    def __init__(
        self,
        context: "_ReComContext",
        rng,
        settings: ReComSettings,
        *,
        objective_local: bool,
    ) -> None:
        self.settings = settings
        self.weights = _WEIGHTS[settings.weights](context, settings)
        self.mu = settings.objective_weight if objective_local else 0.0
        self.pairs = _PAIRS[settings.pair_selection](
            self.weights, settings, rng, self.mu
        )
        self.temperature = Temperature(settings.temperature, settings.entropy)

    def merit(self, state: "_State") -> float:
        return self.weights.merit(state)

    def walk_merit(self, state: "_State", merit: float | None = None) -> float:
        """The merit plus the local objective's price: what a move is judged by.

        ``merit`` is ``self.merit(state)`` when the caller already has it.
        """
        merit = self.merit(state) if merit is None else merit
        return merit + self.mu * state.boundary_cost

    def choose_pair(
        self, state: "_State", pairs: list[tuple[int, int]]
    ) -> tuple[int, int]:
        return self.pairs.choose(state, pairs)

    def cut_probabilities(
        self, state: "_State", zone_a: int, zone_b: int, cuts: list["_Cut"]
    ) -> list[float]:
        if self.settings.cut_selection == "uniform":
            # Uniform over the cuts that leave both zones feasible, else all.
            feasible = [c.pair_feasible for c in cuts]
            if not any(feasible):
                feasible = [True] * len(cuts)
            return softmax([0.0 if ok else -math.inf for ok in feasible])
        weights, mu = self.weights, self.mu
        scale = weights.scale(zone_a, zone_b)
        return self.temperature.probabilities(
            [
                -(
                    weights.zone(zone_a, c.violations_a)
                    + weights.zone(zone_b, c.violations_b)
                    + weights.cut(c.violations)
                    + mu * c.boundary_cost
                )
                / scale
                for c in cuts
            ]
        )

    def moved(self, state: "_State", before: float, after: float) -> None:
        self.weights.observe(state)
        self.pairs.feedback(before, after)

    def end_burst(self, state: "_State") -> None:
        self.weights.update(state)

    def metadata(self) -> dict[str, object]:
        return {**self.weights.metadata(), **self.pairs.metadata()}
