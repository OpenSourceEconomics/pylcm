"""Implicit derivative of a continuous outer optimum (plan section 19.2).

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

Consumers must treat `unresolved` cells as *no derivative available* and
fall back to finite differences or refuse inference there; the tangent is
still returned (guarded against division blow-up) so a vectorized caller
does not NaN-poison resolved cells.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from _lcm.optimization.golden_section import maximize_golden_section
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
    """`|Q_f(f*)|` too large for an interior stationary point — the winner
    sits at a KINK (or the primal under-polished), so the
    implicit-function-theorem tangent `-Q_ftheta/Q_ff`, which assumes
    `Q_f(f*)=0`, does not apply."""

    unresolved: BoolND
    """Any of the above: no trustworthy local-normal derivative here."""


def _mesh_and_polish(
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
        objective,
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


def _continuous_outer_optimum_primal(
    objective: Callable[[FloatND, FloatND], FloatND],
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
        lambda f: objective(f, theta), lower, upper, n_mesh, polish_iterations
    )


# Explicit wrapping instead of `@jax.custom_jvp` decorator syntax: the
# beartype claw re-wraps the decorated `custom_jvp` instance into a plain
# function bound to its `__call__`, losing `defjvp` (same workaround as
# `_lcm.egm.interp`).
continuous_outer_optimum = jax.custom_jvp(
    _continuous_outer_optimum_primal, nondiff_argnums=(0, 3, 4)
)


@continuous_outer_optimum.defjvp
def _continuous_outer_optimum_jvp(
    objective: Callable[[FloatND, FloatND], FloatND],
    n_mesh: int,
    polish_iterations: int,
    primals: tuple,
    tangents: tuple,
) -> tuple[tuple[FloatND, FloatND, FloatND], tuple[FloatND, FloatND, FloatND]]:
    theta, bounds = primals
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

    def q_f(f: FloatND, t: FloatND) -> FloatND:
        return jax.jvp(lambda g: objective(g, t), (f,), (ones,))[1]

    _, q_ff = jax.jvp(lambda f: q_f(f, theta), (f_star,), (ones,))
    _, q_ftheta_dot = jax.jvp(lambda t: q_f(f_star, t), (theta,), (theta_dot,))
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
    _, value_dot = jax.jvp(lambda t: objective(f_star, t), (theta,), (theta_dot,))
    margin_dot = jnp.zeros_like(basin_margin)
    return (f_star, value, basin_margin), (f_dot, value_dot, margin_dot)


def implicit_optimum_diagnostics(
    objective: Callable[[FloatND, FloatND], FloatND],
    *,
    theta: FloatND,
    f_star: FloatND,
    basin_margin: FloatND,
    bounds: tuple[FloatND, FloatND],
    n_mesh: int = 33,
    polish_iterations: int = 32,
    curvature_floor: float = _CURVATURE_FLOOR,
    tie_margin: float = _TIE_MARGIN,
    stationarity_rtol: float = _STATIONARITY_RTOL,
    stationarity_atol: float = _STATIONARITY_ATOL,
) -> ImplicitOptimumDiagnostics:
    """Classify where the implicit derivative at `f_star` is trustworthy."""
    lower, upper = bounds
    width = (upper - lower) / (n_mesh - 1) * (0.618**polish_iterations)
    ones = jnp.ones_like(f_star)
    q_f = jax.jvp(lambda f: objective(f, theta), (f_star,), (ones,))[1]
    _, q_ff = jax.jvp(
        lambda f: jax.jvp(lambda g: objective(g, theta), (f,), (ones,))[1],
        (f_star,),
        (ones,),
    )
    at_lower = f_star <= lower + width
    at_upper = f_star >= upper - width
    flat = jnp.abs(q_ff) < curvature_floor
    tie = basin_margin < tie_margin
    # A genuine interior optimum leaves |Q_f| ~ |Q_ff| * width after the polish;
    # a kink (or an under-polished primal) leaves it far larger. Suppress the
    # screen where the winner is at a bound (Q_f is one-sided by design there,
    # already reported by the bound flags).
    stationarity_threshold = (
        stationarity_rtol * jnp.abs(q_ff) * width + stationarity_atol
    )
    nonstationary = (jnp.abs(q_f) > stationarity_threshold) & ~(at_lower | at_upper)
    return ImplicitOptimumDiagnostics(
        at_lower_bound=at_lower,
        at_upper_bound=at_upper,
        flat_curvature=flat,
        basin_tie=tie,
        nonstationary=nonstationary,
        unresolved=at_lower | at_upper | flat | tie | nonstationary,
    )
