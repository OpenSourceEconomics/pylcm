"""Outer-search strategy configurations for nested outer-choice solvers.

A nested solver (`NNBEGM`) selects a scalar continuous outer action by
comparing an exact keeper against an adjuster sweep. *How* the adjuster's
candidate actions are generated and refined is a strategy of its own, and it
is configured here, separately from the solver's economic wiring:

- `FiniteOuterGrid` — the historical behavior: candidates are exactly the
  nodes of a fixed exogenous grid, and the solve is exact relative to that
  finite candidate set. The default, and the only strategy whose result is
  grid-snapped.
- `AdaptiveOuterMesh` — the canonical continuous-outer approximation: exact
  inner solves on a shared adaptive mesh, a validated interpolant between the
  nodes, and bracket-local refinement of every node-local optimum. No global
  unimodality is assumed; golden-section refinement only ever runs inside
  brackets identified on the exact candidate mesh, and the exact keeper is
  always evaluated separately. Midpoint validation is a *mesh-relative*
  guarantee: it certifies the interpolant against exact solves at the sampled
  points and refines any interval whose exact midpoint beats the incumbent, but
  absent a smoothness bound it cannot rule out an arbitrarily narrow peak that
  falls between the samples. Set `outer_lipschitz_bound` to make refinement a
  genuine global branch-and-bound certificate under that Lipschitz constant.
- `LegacyGoldenSection` — historical-algorithm compatibility (a single
  golden-section search with source-specific endpoint and tie rules); never
  the canonical paper mode.

These are user-facing configuration leaves (re-exported through
`lcm.outer_search`); the numerical machinery that consumes them lives in
`_lcm.egm.outer_interpolation` / `_lcm.egm.outer_refinement` and in the
solver kernels.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Literal

from _lcm.grids import ContinuousGrid
from lcm.exceptions import RegimeInitializationError


class OuterSearch(ABC):  # noqa: B024
    """Configuration for a scalar outer-action search.

    Abstract marker base: a nested solver accepts any `OuterSearch` instance
    and dispatches polymorphically on the concrete strategy. It carries no
    behavior of its own — the strategies are pure configuration leaves.
    """


@dataclass(frozen=True, kw_only=True)
class FiniteOuterGrid(OuterSearch):
    """Finite-candidate outer search — exact relative to a fixed grid.

    The historical `NNBEGM` behavior: the adjuster sweep solves one exact
    inner problem per grid node and the outer axis is collapsed by a finite
    maximum. The selected outer action is always one of the grid nodes.
    """

    grid: ContinuousGrid
    """Exogenous candidate grid for the outer post-decision margin."""

    batch_size: int = 0
    """Grid nodes solved per chunk before folding into the running maximum;
    `0` solves every node at once. A memory knob only — value-invariant."""

    def __post_init__(self) -> None:
        _fail_if_batch_size_negative(self.batch_size, field="batch_size")


@dataclass(frozen=True, kw_only=True)
class AdaptiveOuterMesh(OuterSearch):
    """Canonical continuous-outer approximation.

    Exact inner solves at a shared set of outer nodes; a validated
    interpolant between them; adaptive midpoint insertion until the
    interpolant reproduces exact solves to tolerance; then bracket-local
    refinement of the per-state optimum. Inference runs must fail closed:
    unresolved intervals raise instead of degrading.

    The midpoint validation is mesh-relative: it does not, on its own, rule out
    a peak narrower than the mesh that falls between the sampled midpoints. Set
    `outer_lipschitz_bound` to upgrade refinement to a certified global
    branch-and-bound: any interval whose Lipschitz upper bound could beat the
    incumbent is refined, so the returned optimum is global to tolerance under
    that constant.
    """

    initial_grid: ContinuousGrid
    """Starting shared mesh (sorted, unique); adaptively refined."""

    # Hard static resource limits.
    max_nodes: int = 129
    """Hard cap on shared mesh nodes; exceeding it while intervals remain
    marked is a convergence failure, not a silent truncation."""

    max_refinement_rounds: int = 6
    """Hard cap on midpoint-insertion rounds."""

    batch_size: int = 0
    """Mesh nodes solved per chunk; `0` solves every node at once. A memory
    knob only — value-invariant."""

    # Actual-vs-interpolated candidate refinement.
    value_atol: float = 1e-10
    """Absolute tolerance for the exact-vs-interpolated value at proposed
    midpoints."""

    value_rtol: float = 1e-8
    """Relative tolerance for the exact-vs-interpolated value at proposed
    midpoints."""

    outer_lipschitz_bound: float | None = None
    """Optional Lipschitz constant `L` of the outer value surface in the
    outer-action metric. When set, refinement gains a certified global
    branch-and-bound guarantee: an interval `[a, b]` can hide a point above
    `max(V(a), V(b)) + L*(b-a)/2`, so any interval whose Lipschitz upper bound
    exceeds the incumbent (best node) by more than the value band is refined,
    and the returned optimum is global to tolerance under `L`. `None` keeps the
    default mesh-relative validation, which refines optima that perturb a
    sampled midpoint or the interpolant but cannot certify against a peak
    narrower than the mesh. Must be `> 0` when set; a value below the true
    Lipschitz constant forfeits the global guarantee (garbage-in), a large
    value is safe but costs nodes."""

    # Local continuous refinement.
    local_refiner: Literal["golden", "quadratic"] = "golden"
    """Bracket-local refinement method; brackets always come from the exact
    candidate mesh."""

    golden_iterations: int = 32
    """Static golden-section iteration count per retained bracket."""

    # Safeguards.
    evaluate_all_endpoints: bool = True
    """Always evaluate both domain endpoints exactly as candidates."""

    retain_second_best: bool = True
    """Track the second-best candidate for branch-margin diagnostics."""

    freeze_mesh_for_derivatives: bool = True
    """Derivative batches must re-run on one frozen union mesh rather than
    letting each perturbation adapt its own."""

    fail_closed: bool = True
    """Whether an exhausted refinement budget with marked intervals remaining
    is a hard error (`True`, the inference-grade default) or degrades to a
    best-effort mesh flagged `unresolved` with its residual validation error
    reported through the solver diagnostics (`False`, development / best-effort
    mode). Set `False` when a good-enough outer optimum suffices and the surface
    is known not to validate to tolerance (e.g. a genuinely kinked outer value),
    so a diagnostic-grade solve is reachable while a convergence fix lands. It
    does NOT loosen tolerances — the residual is measured, surfaced, and left to
    the caller to accept, not hidden."""

    def __post_init__(self) -> None:
        _fail_if_batch_size_negative(self.batch_size, field="batch_size")
        if self.max_nodes < 2:  # noqa: PLR2004
            msg = f"max_nodes must be >= 2, got {self.max_nodes}."
            raise RegimeInitializationError(msg)
        if self.max_refinement_rounds < 0:
            msg = (
                f"max_refinement_rounds must be >= 0, got {self.max_refinement_rounds}."
            )
            raise RegimeInitializationError(msg)
        if self.golden_iterations < 1:
            msg = f"golden_iterations must be >= 1, got {self.golden_iterations}."
            raise RegimeInitializationError(msg)
        for name in (
            "value_atol",
            "value_rtol",
        ):
            value = getattr(self, name)
            if not value > 0.0:
                msg = f"{name} must be > 0, got {value}."
                raise RegimeInitializationError(msg)
        if (
            self.outer_lipschitz_bound is not None
            and not self.outer_lipschitz_bound > 0.0
        ):
            msg = (
                "outer_lipschitz_bound must be > 0 when set, got "
                f"{self.outer_lipschitz_bound}."
            )
            raise RegimeInitializationError(msg)


@dataclass(frozen=True, kw_only=True)
class LegacyGoldenSection(OuterSearch):
    """Historical-algorithm compatibility; not canonical paper mode.

    A single golden-section search over the full outer domain with
    source-specific endpoint and tie rules, reproducing a historical
    implementation. It assumes unimodality the canonical mode refuses to
    assume — use only for labeled historical reproduction.
    """

    lower: float
    """Lower edge of the searched outer domain."""

    upper: float
    """Upper edge of the searched outer domain."""

    iterations: int
    """Golden-section iteration count of the historical implementation."""

    tolerance: float
    """Width/convergence tolerance of the historical implementation."""

    endpoint_rule: Literal["fortran"]
    """Which historical endpoint handling to reproduce."""

    tie_break: Literal["fortran"]
    """Which historical tie-breaking rule to reproduce."""

    def __post_init__(self) -> None:
        if not self.upper >= self.lower:
            msg = f"upper must be >= lower, got lower={self.lower}, upper={self.upper}."
            raise RegimeInitializationError(msg)
        if self.iterations < 1:
            msg = f"iterations must be >= 1, got {self.iterations}."
            raise RegimeInitializationError(msg)
        if not self.tolerance > 0.0:
            msg = f"tolerance must be > 0, got {self.tolerance}."
            raise RegimeInitializationError(msg)


def _fail_if_batch_size_negative(batch_size: int, *, field: str) -> None:
    if batch_size < 0:
        msg = f"{field} must be >= 0, got {batch_size}."
        raise RegimeInitializationError(msg)
