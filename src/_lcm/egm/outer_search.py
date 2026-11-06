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
These are user-facing configuration leaves (re-exported through
`lcm.outer_search`); the numerical machinery that consumes them lives in
`_lcm.egm.outer_interpolation` / `_lcm.egm.outer_refinement` and in the
solver kernels.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from annotated_types import Ge, Gt
from pydantic import InstanceOf
from pydantic.dataclasses import dataclass as pydantic_dataclass

from _lcm.grids import ContinuousGrid
from _lcm.input_validation import (
    CONFIG_FLOAT,
    EXACT_INT,
    STRICT_CONFIG,
    translate_validation_errors,
)
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


# A positive tolerance or bound; positive infinity is admitted.
type _PositiveFloat = Annotated[float, CONFIG_FLOAT, Gt(0.0)]


@translate_validation_errors(RegimeInitializationError)
@pydantic_dataclass(frozen=True, kw_only=True, config=STRICT_CONFIG)
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

    initial_grid: InstanceOf[ContinuousGrid]
    """Starting shared mesh (sorted, unique); adaptively refined."""

    # Hard static resource limits.
    max_nodes: Annotated[int, EXACT_INT, Ge(2)] = 129
    """Hard cap on shared mesh nodes; exceeding it while intervals remain
    marked is a convergence failure, not a silent truncation."""

    max_refinement_rounds: Annotated[int, EXACT_INT, Ge(0)] = 6
    """Hard cap on midpoint-insertion rounds."""

    # Actual-vs-interpolated candidate refinement.
    value_atol: _PositiveFloat = 1e-10
    """Absolute tolerance for the exact-vs-interpolated value at proposed
    midpoints."""

    value_rtol: _PositiveFloat = 1e-8
    """Relative tolerance for the exact-vs-interpolated value at proposed
    midpoints."""

    outer_lipschitz_bound: _PositiveFloat | None = None
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

    # Local continuous refinement. Brackets always come from the exact
    # candidate mesh, and refinement inside one is golden section.
    golden_iterations: Annotated[int, EXACT_INT, Ge(1)] = 32
    """Static golden-section iteration count per retained bracket."""

    fail_closed: bool = True
    """Whether an exhausted refinement budget with marked intervals remaining
    is a hard error (`True`, the inference-grade default) or degrades to a
    best-effort mesh flagged `unresolved` with its residual validation error
    reported through the solver diagnostics (`False`, development / best-effort
    mode). Set `False` when a good-enough outer optimum suffices and the surface
    is known not to validate to tolerance (e.g. a genuinely kinked outer value),
    so a diagnostic-grade solve is reachable while a convergence fix lands. It
    does NOT loosen tolerances — the residual is measured, surfaced, and left to
    the caller to accept, not hidden.

    The guarantee is mesh convergence and nothing wider. These stay
    diagnostics under `True` and never raise:

    - an optimum sitting on a domain bound (`outer_at_lower_bound` /
      `outer_at_upper_bound`);
    - a close keeper-vs-adjuster branch margin
      (`keeper_adjuster_margin`);
    - a close best-vs-runner-up margin (`best_second_best_margin`);
    - a withheld simulation policy (`policy_fallback_mask`);
    - cells where every outer candidate was invalid
      (`n_outer_all_invalid_cells`).

    An inference release gate over those is the caller's to build."""
