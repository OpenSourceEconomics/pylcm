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
  nodes, and globally safeguarded bracket-local refinement. No global
  unimodality is assumed; golden-section refinement only ever runs inside
  brackets identified on the exact candidate mesh, and the exact keeper is
  always evaluated separately.
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
    interpolant reproduces exact solves to tolerance; then globally
    safeguarded bracket-local refinement of the per-state optimum. Inference
    runs must fail closed: unresolved intervals raise instead of degrading.
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

    # Policy convergence checks.
    outer_policy_atol: float = 1e-5
    """Largest admissible move of the selected outer action under one more
    refinement round, in outer-action units."""

    inner_policy_atol: float = 1e-6
    """Largest admissible move of the interpolated inner policy under one
    more refinement round, in inner-action units."""

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
            "outer_policy_atol",
            "inner_policy_atol",
        ):
            value = getattr(self, name)
            if not value > 0.0:
                msg = f"{name} must be > 0, got {value}."
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
