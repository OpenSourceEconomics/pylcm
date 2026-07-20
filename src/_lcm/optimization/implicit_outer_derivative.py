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
  to jump — the derivative is set-valued).

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

    def q_of_f(f: FloatND) -> FloatND:
        return objective(f, theta)

    def q_f_of_theta(t: FloatND) -> FloatND:
        grad_f = jax.grad(lambda f, tt: jnp.sum(objective(f, tt)), argnums=0)
        return grad_f(f_star, t)

    q_ff = jax.grad(lambda f: jnp.sum(jax.grad(lambda g: jnp.sum(q_of_f(g)))(f)))(
        f_star
    )
    _, q_ftheta_dot = jax.jvp(q_f_of_theta, (theta,), (theta_dot,))
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
) -> ImplicitOptimumDiagnostics:
    """Classify where the implicit derivative at `f_star` is trustworthy."""
    lower, upper = bounds
    width = (upper - lower) / (n_mesh - 1) * (0.618**polish_iterations)
    q_ff = jax.grad(
        lambda f: jnp.sum(jax.grad(lambda g: jnp.sum(objective(g, theta)))(f))
    )(f_star)
    at_lower = f_star <= lower + width
    at_upper = f_star >= upper - width
    flat = jnp.abs(q_ff) < curvature_floor
    tie = basin_margin < tie_margin
    return ImplicitOptimumDiagnostics(
        at_lower_bound=at_lower,
        at_upper_bound=at_upper,
        flat_curvature=flat,
        basin_tie=tie,
        unresolved=at_lower | at_upper | flat | tie,
    )
