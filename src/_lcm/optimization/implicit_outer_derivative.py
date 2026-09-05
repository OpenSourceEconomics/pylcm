"""Implicit derivative of a continuous outer optimum.

`continuous_outer_optimum` maximizes a smooth scalar objective `Q(f, theta)`
over a bracket and carries a *custom JVP*: the tangent of the winning
abscissa is the implicit-function-theorem derivative

    df*/dtheta = - Q_{f theta} / Q_{ff},        Q_f(f*, theta) = 0,

evaluated at the primal winner — never the naive derivative through the
search's comparison operations, which differentiates the bracket updates
instead of the economics. The value tangent is the envelope-theorem term
`Q_theta(f*, theta)`.

The primal is the robust two-stage search: a static exact mesh locates the
global basin (no unimodality assumed), golden section polishes inside the
winning bracket. Cells where local-normal calculus is not trustworthy are
REPORTED, not repaired, through `ImplicitOptimumDiagnostics`:

- the winner sits at a bracket bound (one-sided optimum, `Q_f != 0`);
- `|Q_ff|` is below the curvature threshold (flat top — the implicit
  tangent divides by ~0);
- the best and runner-up mesh basins are value-tied (a global argmax about
  to jump — the derivative is set-valued);
- `Q_f(f*)` is not (near) zero — the winner sits at a KINK in the value
  surface (or the primal under-polished), so the first-order condition the
  implicit-function theorem inverts does not hold. On the real Mahler-Yum
  model the consumption floor makes the value non-smooth in effort, and the
  outer optimum can pin to a floor-induced kink whose location does not move
  with the parameter; without this screen the tangent `-Q_ftheta/Q_ff`, valid
  only at a stationary point, would be reported as trustworthy.

Exact certification requires complete OWNER PROVENANCE: a formula-defining
signature that includes the exact affine segment, inner discrete choice,
floor/budget piece, active constraint side, and every other identity that
selects the local smooth formula. The signature must remain decided,
strict-primary, complete, and unchanged across action probes, parameter probes,
mixed corners, and supplied reoptimized points. A legacy `branch_id` oracle is
still a useful conservative action-direction kink screen, but cannot certify
parameter-direction stability or completeness and therefore fails closed when
certification is requested.

Without owner provenance the generic diagnostic retains a value-only HEURISTIC
— a multi-radius slope-jump contraction test. This is not a differentiability
certificate: a kink whose slope jump is below the value oracle rounding floor
at the probe radius (yet still yields an O(1) argmax-tangent error) cannot be
seen from values alone. Certified consumers must require owner provenance;
`branch_certified` is true only when the complete witness passes.

Consumers must treat `unresolved` cells as *no derivative available* and
fall back to finite differences or refuse inference there; the tangent is
still returned (guarded against division blow-up) so a vectorized caller
does not NaN-poison resolved cells.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from _lcm.optimization.golden_section import maximize_golden_section
from _lcm.utils.functools import allow_args
from lcm.typing import BoolND, FloatND

# Below this |Q_ff| the implicit tangent is numerically meaningless: the
# objective's top is flat to working precision and df*/dtheta explodes.
_CURVATURE_FLOOR = 1e-8
# Best-vs-second-best mesh-basin margin below which the global argmax is
# treated as tied (about to jump between basins under a parameter nudge).
_TIE_MARGIN = 1e-10
# A winner is treated as non-stationary when |Q_f(f*)| exceeds this many
# multiples of the residual a genuine interior optimum would still carry after
# the polish, |Q_ff| * bracket_width (since Q_f ~ Q_ff * (f - f*) and the
# polish leaves f within ~one bracket width of the true optimum). A KINK in the
# value surface — the paper-mode consumption floor makes V non-smooth in the
# outer action — pins the optimum at a point where Q_f is sign-definite and
# does NOT vanish under refinement, so it stays above this screen while a
# smooth optimum falls below it.
_STATIONARITY_RTOL = 100.0
# Absolute |Q_f| floor so a near-flat objective (tiny |Q_ff|, already flagged
# by the curvature screen) does not also trip the stationarity screen on
# rounding noise.
_STATIONARITY_ATOL = 1e-7
# Minimum radius of the symmetric slope probe that certifies stationarity. A
# single forward-mode Q_f is NOT a valid kink certificate: at an exact kink
# (a tent peak from the consumption floor) jax.jvp returns one sub-gradient
# branch and can read ~0, passing a non-smooth optimum as stationary. Probing
# |Q_f| from BOTH sides at this radius exposes the slope jump the AD value
# hides. Must exceed the final polish bracket so the probe straddles the kink.
_KINK_PROBE_ATOL = 1e-4
# Multi-radius kink screen. A single fixed-radius slope-jump
# threshold is amplitude-scaled — a kink whose jump is below rtol*|Q_ff|*delta
# slips through, and the kink amplitude can be made arbitrarily small while the
# argmax tangent error stays O(1). Instead compare the slope jump at delta and
# delta/`_KINK_CONTRACTION_RATIO`: a SMOOTH optimum's one-sided slopes both
# converge to Q_f(f*)=0, so the jump contracts ~linearly with the radius (jump ~
# |Q_ff|*delta); a KINK leaves a jump of order the kink size, INDEPENDENT of
# radius. A jump that fails to contract below `_KINK_CONTRACTION_TOL` of its
# outer value (between 1/ratio and 1) when the radius is cut is a genuine
# breakpoint, whatever its amplitude.
_KINK_CONTRACTION_RATIO = 8.0
_KINK_CONTRACTION_TOL = 0.5
# ULPs of Q's value scale below which the slope jump is rounding, not structure:
# rounding in Q propagates to a one-sided slope as ~eps*|Q|/radius, so the
# contraction ratio of two noise-level jumps is meaningless and must be gated
# out (a machine-precision-smooth optimum is not a kink).
_KINK_NOISE_ULPS = 64.0


class OwnerProvenance(NamedTuple):
    """Compact formula-defining identity for one certified derivative cell.

    ``signature`` is a fixed pytree of integer/Boolean arrays on the cell axes:
    exact affine segment, inner discrete branch, floor/budget piece,
    constraint/interval side, keeper/adjuster branch, and every other identity
    whose change selects a different smooth formula. ``decided`` is false for
    any unresolved native/status component; ``strict_primary`` is false when a
    deterministic secondary key broke an exact primary-value tie; ``complete``
    is false when any formula-defining component is unavailable.
    """

    signature: object
    decided: BoolND
    strict_primary: BoolND
    complete: BoolND


@dataclass(frozen=True)
class ImplicitOptimumDiagnostics:
    """Where the implicit derivative of the outer optimum is trustworthy."""

    at_lower_bound: BoolND
    """Winner within one polish-bracket width of the lower bound."""

    at_upper_bound: BoolND
    """Winner within one polish-bracket width of the upper bound."""

    flat_curvature: BoolND
    """`|Q_ff|` at the winner below the curvature floor."""

    basin_tie: BoolND
    """Best and runner-up mesh basins value-tied at the mesh stage."""

    nonstationary: BoolND
    """The local first-order/stationarity or formula-identity screen failed."""

    branch_certified: BoolND
    """A complete, decided, strict-primary provenance record is unchanged."""

    owner_missing: BoolND
    """No complete provenance callback was supplied for this cell."""

    owner_incomplete: BoolND
    """At least one callback result omitted a formula-defining component."""

    owner_unresolved: BoolND
    """At least one exact owner/status fact was undecided."""

    owner_primary_tie: BoolND
    """At least one point used a secondary key to break a primary-value tie."""

    owner_changed: BoolND
    """The composite formula-defining signature changed in the neighborhood."""

    unresolved: BoolND
    """No trustworthy local-normal derivative is available for this cell."""


def _mesh_and_polish(
    *,
    objective: Callable[[FloatND], FloatND],
    lower: FloatND,
    upper: FloatND,
    n_mesh: int,
    polish_iterations: int,
) -> tuple[FloatND, FloatND, FloatND]:
    """Global mesh stage plus golden-section polish; no unimodality assumed.

    Returns `(f_star, value, basin_margin)` where `basin_margin` is the
    gap between the winning mesh node and the best node OUTSIDE the
    winner's immediate neighborhood — adjacent nodes share the winner's
    basin, so excluding them makes the margin measure cross-basin
    competition (the tie diagnostic's input), not local mesh spacing.
    """
    mesh = jnp.linspace(0.0, 1.0, n_mesh)
    nodes = lower[..., None] + (upper - lower)[..., None] * mesh
    node_values = jnp.stack([objective(nodes[..., k]) for k in range(n_mesh)], axis=-1)
    node_values = jnp.where(jnp.isnan(node_values), -jnp.inf, node_values)
    best = jnp.argmax(node_values, axis=-1)
    best_value = jnp.take_along_axis(node_values, best[..., None], axis=-1)[..., 0]
    runner_values = jnp.where(
        jnp.abs(jnp.arange(n_mesh) - best[..., None]) <= 1, -jnp.inf, node_values
    )
    second_value = jnp.max(runner_values, axis=-1)
    basin_margin = best_value - second_value
    # Polish inside the bracket flanking the winning node.
    step = (upper - lower) / (n_mesh - 1)
    bracket_lower = jnp.clip(lower + (best - 1) * step, min=lower, max=upper)
    bracket_upper = jnp.clip(lower + (best + 1) * step, min=lower, max=upper)
    polished = maximize_golden_section(
        objective=objective,
        lower=bracket_lower,
        upper=bracket_upper,
        iterations=polish_iterations,
    )
    # The exact winning node is always a candidate: keep it when the polish
    # (surrogate-free but finite-precision) does not beat it.
    node_x = lower + best * step
    take_polished = polished.value >= best_value
    f_star = jnp.where(take_polished, polished.x, node_x)
    value = jnp.maximum(polished.value, best_value)
    return f_star, value, basin_margin


# keyword-only-exempt: library-callback=jax.custom_jvp
def _continuous_outer_optimum_primal(
    objective: Callable[..., FloatND],
    theta: FloatND,
    bounds: tuple[FloatND, FloatND],
    n_mesh: int = 33,
    polish_iterations: int = 32,
) -> tuple[FloatND, FloatND, FloatND]:
    """Maximize `Q(f, theta)` over `[lower, upper]` with an implicit JVP.

    Args:
        objective: Smooth scalar objective `Q(f, theta)`, vectorized over
            the cell axes of `f` (theta is shared).
        theta: Parameter (pytree-free array; the differentiable input).
        bounds: `(lower, upper)` per-cell bracket arrays.
        n_mesh: Static exact-mesh size of the global stage.
        polish_iterations: Golden-section iterations inside the winning
            bracket.

    Returns:
        `(f_star, value, basin_margin)` — the winning abscissa, the value
        at the winner, and the mesh-stage best-vs-second-best margin.
    """
    lower, upper = bounds
    return _mesh_and_polish(
        objective=_ActionSection(objective=allow_args(objective), theta=theta),
        lower=lower,
        upper=upper,
        n_mesh=n_mesh,
        polish_iterations=polish_iterations,
    )


# Explicit wrapping instead of `@jax.custom_jvp` decorator syntax: the
# beartype claw re-wraps the decorated `custom_jvp` instance into a plain
# function bound to its `__call__`, losing `defjvp` (same workaround as
# `_lcm.egm.interp`).
continuous_outer_optimum = jax.custom_jvp(
    _continuous_outer_optimum_primal, nondiff_argnums=(0, 3, 4)
)


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
@continuous_outer_optimum.defjvp
def _continuous_outer_optimum_jvp(
    objective: Callable[..., FloatND],
    n_mesh: int,
    polish_iterations: int,
    primals: tuple,
    tangents: tuple,
) -> tuple[tuple[FloatND, FloatND, FloatND], tuple[FloatND, FloatND, FloatND]]:
    theta, bounds = primals
    positional_objective = allow_args(objective)
    theta_dot, _ = tangents
    f_star, value, basin_margin = continuous_outer_optimum(
        objective, theta, bounds, n_mesh, polish_iterations
    )

    # All objective derivatives are FORWARD-mode. The objective is per-cell
    # (cell i's value depends only on f[i]), so the ones-tangent JVP is
    # exactly the elementwise derivative — and forward mode differentiates
    # through any inner control flow (while/fori loops in a nested solve)
    # that reverse-mode AD cannot.
    ones = jnp.ones_like(f_star)
    _, q_ff = jax.jvp(
        _ActionSlopeInAction(objective=positional_objective, theta=theta, ones=ones),
        (f_star,),
        (ones,),
    )
    _, q_ftheta_dot = jax.jvp(
        _ActionSlopeInParameter(objective=positional_objective, f=f_star, ones=ones),
        (theta,),
        (theta_dot,),
    )
    # At a maximum Q_ff <= 0; a flat top is guarded toward -floor so the
    # tangent stays finite (the diagnostics flag such cells as unresolved).
    guarded_curvature = jnp.where(
        jnp.abs(q_ff) < _CURVATURE_FLOOR,
        jnp.where(q_ff > 0.0, _CURVATURE_FLOOR, -_CURVATURE_FLOOR),
        q_ff,
    )
    f_dot = -q_ftheta_dot / guarded_curvature
    # Envelope theorem for the value: the argmax term vanishes at an
    # interior optimum; at a bound the reported value tangent is still the
    # partial (the bound's own movement is not differentiated here).
    _, value_dot = jax.jvp(
        _ParameterSection(objective=positional_objective, f=f_star),
        (theta,),
        (theta_dot,),
    )
    margin_dot = jnp.zeros_like(basin_margin)
    return (f_star, value, basin_margin), (f_dot, value_dot, margin_dot)


@dataclass(frozen=True, eq=False)
class _ActionSection:
    """`Q(., theta)`: the objective as a map of the action at a fixed parameter."""

    objective: Callable[..., FloatND]
    """The objective, called positionally as `objective(f, theta)`."""

    theta: FloatND
    """The parameter held fixed."""

    def __call__(self, f: FloatND) -> FloatND:
        """Evaluate `Q(f, theta)`."""
        return self.objective(f, self.theta)


@dataclass(frozen=True, eq=False)
class _ParameterSection:
    """`Q(f, .)`: the objective as a map of the parameter at a fixed action."""

    objective: Callable[..., FloatND]
    """The objective, called positionally as `objective(f, theta)`."""

    f: FloatND
    """The action held fixed."""

    def __call__(self, theta: FloatND) -> FloatND:
        """Evaluate `Q(f, theta)`."""
        return self.objective(self.f, theta)


def _action_slope(
    *, objective: Callable[..., FloatND], f: FloatND, theta: FloatND, ones: FloatND
) -> FloatND:
    """`Q_f(f, theta)` by forward mode.

    The objective is per-cell (cell i's value depends only on `f[i]`), so the
    ones-tangent JVP is exactly the elementwise derivative — and forward mode
    differentiates through any inner control flow (while/fori loops in a nested
    solve) that reverse-mode AD cannot.
    """
    return jax.jvp(_ActionSection(objective=objective, theta=theta), (f,), (ones,))[1]


@dataclass(frozen=True, eq=False)
class _ActionSlopeInAction:
    """`Q_f(., theta)`: the action slope as a map of the action."""

    objective: Callable[..., FloatND]
    """The objective, called positionally as `objective(f, theta)`."""

    theta: FloatND
    """The parameter held fixed."""

    ones: FloatND
    """The unit tangent of the cell axes, `ones_like(f)`."""

    def __call__(self, f: FloatND) -> FloatND:
        """Evaluate `Q_f(f, theta)`."""
        return _action_slope(
            objective=self.objective, f=f, theta=self.theta, ones=self.ones
        )


@dataclass(frozen=True, eq=False)
class _ActionSlopeInParameter:
    """`Q_f(f, .)`: the action slope as a map of the parameter."""

    objective: Callable[..., FloatND]
    """The objective, called positionally as `objective(f, theta)`."""

    f: FloatND
    """The action held fixed."""

    ones: FloatND
    """The unit tangent of the cell axes, `ones_like(f)`."""

    def __call__(self, theta: FloatND) -> FloatND:
        """Evaluate `Q_f(f, theta)`."""
        return _action_slope(
            objective=self.objective, f=self.f, theta=theta, ones=self.ones
        )


def _cell_bool(*, value: object, like: FloatND) -> BoolND:
    """Broadcast one provenance flag onto the diagnostic cell axes."""
    return jnp.broadcast_to(jnp.asarray(value, dtype=bool), jnp.shape(like))


def _same_owner_signature(
    *, baseline: object, candidate: object, like: FloatND
) -> tuple[BoolND, bool]:
    """Compare fixed-pytree signature fields without inventing missing fields."""
    baseline_leaves, baseline_tree = jax.tree_util.tree_flatten(baseline)
    candidate_leaves, candidate_tree = jax.tree_util.tree_flatten(candidate)
    structurally_complete = (
        baseline_tree == candidate_tree
        and bool(baseline_leaves)
        and len(baseline_leaves) == len(candidate_leaves)
    )
    if not structurally_complete:
        return jnp.zeros_like(like, dtype=bool), False
    same = jnp.ones_like(like, dtype=bool)
    for left, right in zip(baseline_leaves, candidate_leaves, strict=True):
        same &= jnp.broadcast_to(
            jnp.equal(jnp.asarray(left), jnp.asarray(right)), jnp.shape(like)
        )
    return same, True


def _owner_certificate_flags(
    *,
    owner_provenance: Callable[..., OwnerProvenance],
    f_star: FloatND,
    theta: FloatND,
    action_delta: FloatND,
    parameter_delta: FloatND,
    lower: FloatND,
    upper: FloatND,
    reoptimized_owner_points: tuple[tuple[FloatND, FloatND], ...],
) -> tuple[BoolND, BoolND, BoolND, BoolND, BoolND]:
    """Evaluate the complete action/parameter/mixed/Richardson witness."""
    f_plus = jnp.clip(f_star + action_delta, min=lower, max=upper)
    f_minus = jnp.clip(f_star - action_delta, min=lower, max=upper)
    theta_plus = theta + parameter_delta
    theta_minus = theta - parameter_delta
    points = (
        (f_star, theta),
        (f_minus, theta),
        (f_plus, theta),
        (f_star, theta_minus),
        (f_star, theta_plus),
        (f_minus, theta_minus),
        (f_minus, theta_plus),
        (f_plus, theta_minus),
        (f_plus, theta_plus),
        *reoptimized_owner_points,
    )
    records = tuple(owner_provenance(f, t) for f, t in points)
    baseline = records[0]
    owner_incomplete = ~_cell_bool(value=baseline.complete, like=f_star)
    owner_unresolved = ~_cell_bool(value=baseline.decided, like=f_star)
    owner_primary_tie = ~_cell_bool(value=baseline.strict_primary, like=f_star)
    owner_changed = jnp.zeros_like(f_star, dtype=bool)
    baseline_leaves, _ = jax.tree_util.tree_flatten(baseline.signature)
    if not baseline_leaves:
        owner_incomplete |= jnp.ones_like(f_star, dtype=bool)
    for record in records[1:]:
        owner_incomplete |= ~_cell_bool(value=record.complete, like=f_star)
        owner_unresolved |= ~_cell_bool(value=record.decided, like=f_star)
        owner_primary_tie |= ~_cell_bool(value=record.strict_primary, like=f_star)
        same, structure_complete = _same_owner_signature(
            baseline=baseline.signature, candidate=record.signature, like=f_star
        )
        owner_changed |= ~same
        if not structure_complete:
            owner_incomplete |= jnp.ones_like(f_star, dtype=bool)
    branch_certified = ~(
        owner_incomplete | owner_unresolved | owner_primary_tie | owner_changed
    )
    return (
        branch_certified,
        owner_incomplete,
        owner_unresolved,
        owner_primary_tie,
        owner_changed,
    )


def implicit_optimum_diagnostics(
    *,
    objective: Callable[..., FloatND],
    theta: FloatND,
    f_star: FloatND,
    basin_margin: FloatND,
    bounds: tuple[FloatND, FloatND],
    n_mesh: int = 33,
    polish_iterations: int = 32,
    branch_id: Callable[..., FloatND] | None = None,
    owner_provenance: Callable[..., OwnerProvenance] | None = None,
    require_owner_certificate: bool = False,
    reoptimized_owner_points: tuple[tuple[FloatND, FloatND], ...] = (),
    parameter_probe_atol: float = 1e-05,
    parameter_probe_rtol: float = 1e-06,
    curvature_floor: float = _CURVATURE_FLOOR,
    tie_margin: float = _TIE_MARGIN,
    stationarity_rtol: float = _STATIONARITY_RTOL,
    stationarity_atol: float = _STATIONARITY_ATOL,
    kink_probe_atol: float = _KINK_PROBE_ATOL,
    kink_contraction_ratio: float = _KINK_CONTRACTION_RATIO,
    kink_contraction_tol: float = _KINK_CONTRACTION_TOL,
) -> ImplicitOptimumDiagnostics:
    """Classify whether a local implicit derivative has a complete witness.

    ``owner_provenance`` is the only certification path. It is evaluated at the
    baseline optimum, both action probes, both parameter probes, all four mixed
    corners, and every supplied reoptimized Richardson point. A legacy
    ``branch_id`` remains useful as a conservative action-direction kink screen,
    but it is incomplete evidence and can never set ``branch_certified``.

    Generic callers may retain the historical value-only heuristic by leaving
    ``require_owner_certificate=False`` and supplying neither callback. Certified
    consumers must set it true. Supplying either provenance interface also
    requests certification automatically, so a legacy label fails closed rather
    than masquerading as a complete certificate.
    """
    positional_objective, positional_branch_id, positional_owner_provenance = (
        allow_args(objective),
        None if branch_id is None else allow_args(branch_id),
        None if owner_provenance is None else allow_args(owner_provenance),
    )
    lower, upper = bounds
    width = 2.0 * (upper - lower) / (n_mesh - 1) * (0.618**polish_iterations)
    ones = jnp.ones_like(f_star)
    q_f = _action_slope(
        objective=positional_objective, f=f_star, theta=theta, ones=ones
    )
    _, q_ff = jax.jvp(
        _ActionSlopeInAction(objective=positional_objective, theta=theta, ones=ones),
        (f_star,),
        (ones,),
    )
    at_lower = f_star <= lower + width
    at_upper = f_star >= upper - width
    flat = jnp.abs(q_ff) < curvature_floor
    tie = basin_margin < tie_margin
    action_delta = jnp.maximum(width, kink_probe_atol)

    slope_jump = functools.partial(
        _slope_jump,
        f_star=f_star,
        lower=lower,
        upper=upper,
        objective=positional_objective,
        theta=theta,
    )
    jump_outer = slope_jump(radius=action_delta)
    jump_inner = slope_jump(radius=action_delta / kink_contraction_ratio)
    eps = jnp.finfo(jnp.asarray(f_star).dtype).eps
    value_scale = jnp.abs(positional_objective(f_star, theta))
    jump_noise = stationarity_atol + _KINK_NOISE_ULPS * eps * value_scale * (
        kink_contraction_ratio / action_delta
    )
    kinked = (jump_outer > jump_noise) & (
        jump_inner > kink_contraction_tol * jump_outer
    )

    legacy_branch_unstable = jnp.zeros_like(kinked)
    if positional_branch_id is not None:
        f_plus = jnp.clip(f_star + action_delta, min=lower, max=upper)
        f_minus = jnp.clip(f_star - action_delta, min=lower, max=upper)
        b_star = positional_branch_id(f_star, theta)
        legacy_branch_unstable = (positional_branch_id(f_plus, theta) != b_star) | (
            positional_branch_id(f_minus, theta) != b_star
        )

    owner_missing = jnp.full_like(f_star, owner_provenance is None, dtype=bool)
    if positional_owner_provenance is None:
        branch_certified = jnp.zeros_like(f_star, dtype=bool)
        owner_incomplete = jnp.zeros_like(f_star, dtype=bool)
        owner_unresolved = jnp.zeros_like(f_star, dtype=bool)
        owner_primary_tie = jnp.zeros_like(f_star, dtype=bool)
        owner_changed = jnp.zeros_like(f_star, dtype=bool)
    else:
        theta_array = jnp.asarray(theta, dtype=jnp.asarray(f_star).dtype)
        parameter_delta = jnp.maximum(
            jnp.asarray(parameter_probe_atol, dtype=theta_array.dtype),
            jnp.abs(theta_array) * parameter_probe_rtol,
        )
        (
            branch_certified,
            owner_incomplete,
            owner_unresolved,
            owner_primary_tie,
            owner_changed,
        ) = _owner_certificate_flags(
            owner_provenance=positional_owner_provenance,
            f_star=f_star,
            theta=theta,
            action_delta=action_delta,
            parameter_delta=parameter_delta,
            lower=lower,
            upper=upper,
            reoptimized_owner_points=reoptimized_owner_points,
        )

    stationarity_threshold = (
        stationarity_rtol * jnp.abs(q_ff) * width + stationarity_atol
    )
    nonstationary = (
        (jnp.abs(q_f) > stationarity_threshold)
        | kinked
        | legacy_branch_unstable
        | owner_changed
    ) & ~(at_lower | at_upper)
    certification_requested = (
        require_owner_certificate
        or branch_id is not None
        or owner_provenance is not None
    )
    certification_failure = jnp.zeros_like(f_star, dtype=bool)
    if certification_requested:
        certification_failure = (
            owner_missing
            | owner_incomplete
            | owner_unresolved
            | owner_primary_tie
            | owner_changed
        )
    unresolved = (
        at_lower | at_upper | flat | tie | nonstationary | certification_failure
    )
    return ImplicitOptimumDiagnostics(
        at_lower_bound=at_lower,
        at_upper_bound=at_upper,
        flat_curvature=flat,
        basin_tie=tie,
        nonstationary=nonstationary,
        branch_certified=branch_certified,
        owner_missing=owner_missing,
        owner_incomplete=owner_incomplete,
        owner_unresolved=owner_unresolved,
        owner_primary_tie=owner_primary_tie,
        owner_changed=owner_changed,
        unresolved=unresolved,
    )


def _slope_jump(
    *,
    radius: FloatND,
    f_star: FloatND,
    lower: FloatND,
    upper: FloatND,
    objective: Callable[..., FloatND],
    theta: FloatND,
) -> FloatND:
    """Gap between the one-sided secant slopes at `f_star`, probed at `radius`.

    Each probe is clipped to the bracket; a side that cannot move contributes a
    zero-numerator slope over a unit step rather than a division by zero.
    """
    f_plus = jnp.clip(f_star + radius, min=lower, max=upper)
    f_minus = jnp.clip(f_star - radius, min=lower, max=upper)
    step_plus = jnp.where(f_plus > f_star, f_plus - f_star, 1.0)
    step_minus = jnp.where(f_star > f_minus, f_star - f_minus, 1.0)
    slope_plus = (objective(f_plus, theta) - objective(f_star, theta)) / step_plus
    slope_minus = (objective(f_star, theta) - objective(f_minus, theta)) / step_minus
    return jnp.abs(slope_plus - slope_minus)
