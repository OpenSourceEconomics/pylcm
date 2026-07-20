"""The Bellman-objective evaluator for the two-asset G2EGM upper envelope.

The upper envelope ranks interpolated candidate policies by **recomputing** the Bellman
objective, never by transporting an interpolated value. For the two-asset model the
objective of a policy $(c, d)$ at a current state $(m, n)$ is

$$Q(m, n; c, d) = u(c) + \\beta\\, W(a, b),$$

with the post-decision balances reconstructed from the budget identities
$a = m - c - d$ and $b = n + d + \\chi\\log(1 + d)$, and $W$ read by bilinear
interpolation off the regular post-decision $(a, b)$ value grid. A candidate is
feasible only when $c > 0$, $d \\ge 0$, $a \\ge 0$, and $b \\ge 0$ — a policy
interpolated (or extrapolated) outside its segment can violate these, so the evaluator
returns the feasibility flag the envelope masks on before either maximum.
"""

from collections.abc import Callable

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from lcm.typing import BoolND, Float1D, Float2D, FloatND, ScalarFloat


def build_two_asset_objective(
    *,
    post_decision_value: Float2D,
    a_grid: Float1D,
    b_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    match_rate: ScalarFloat | float,
    post_decision_valid: BoolND | None = None,
) -> Callable[[Float1D, Float1D], tuple[FloatND, BoolND]]:
    """Build the `(state, policy) -> (value, feasible)` objective evaluator.

    Args:
        post_decision_value: Post-decision value `W(a, b)` on the regular post-decision
            grid, shape `(len(a_grid), len(b_grid))`.
        a_grid: Regular liquid post-decision grid (ascending, evenly spaced).
        b_grid: Regular pension post-decision grid (ascending, evenly spaced).
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion `rho`.
        match_rate: Pension employer-match coefficient `chi`.
        post_decision_valid: Optional in-domain mask over the post-decision grid, shape
            `(len(a_grid), len(b_grid))`. A `W(a, b)` node built by clamping an
            out-of-domain transformed state carries a fabricated continuation; where the
            mask is `False`, a candidate whose reconstructed `(a, b)` reads it (its
            bilinear stencil touches a `False` node) is infeasible. `None` treats every
            node as in-domain.

    Returns:
        A callable mapping a state `(m, n)` and policy `(c, d)` to the recomputed
        Bellman value and a feasibility flag. The value is finite even for an
        infeasible candidate (consumption is clamped before the utility), so the
        envelope can mask on the flag without NaN poisoning.

    """
    a_origin = a_grid[0]
    a_step = a_grid[1] - a_grid[0]
    b_origin = b_grid[0]
    b_step = b_grid[1] - b_grid[0]

    def objective(state: Float1D, policy: Float1D) -> tuple[FloatND, BoolND]:
        consumption, deposit = policy[0], policy[1]
        liquid_post = state[0] - consumption - deposit
        pension_post = state[1] + deposit + match_rate * jnp.log1p(deposit)
        a_index = (liquid_post - a_origin) / a_step
        b_index = (pension_post - b_origin) / b_step
        post_value = map_coordinates(
            post_decision_value,
            [jnp.atleast_1d(a_index), jnp.atleast_1d(b_index)],
            order=1,
            mode="nearest",
        )[0]
        # Clamp consumption before the utility so an infeasible candidate's value is
        # finite (it is masked out by `feasible`), never NaN.
        safe_consumption = jnp.where(consumption > 0.0, consumption, 1.0)
        value = _crra_utility(safe_consumption, crra) + discount_factor * post_value
        # The continuation reader clamps post-states to the grid boundary, so a
        # candidate whose reconstructed `(a, b)` leaves the post-decision grid gets a
        # fabricated continuation. Require the post-state inside the grid (subsuming the
        # economic `a >= 0`, `b >= 0` floors when the grid starts there), so only
        # candidates with a genuine continuation value are eligible for the envelope.
        feasible = (
            (consumption > 0.0)
            & (deposit >= 0.0)
            & (liquid_post >= a_grid[0])
            & (liquid_post <= a_grid[-1])
            & (pension_post >= b_grid[0])
            & (pension_post <= b_grid[-1])
        )
        # `W(a, b)` itself may have been built by clamping an out-of-domain
        # transformed state. Read the in-domain mask with the same bilinear stencil
        # and require it fully in-domain (`== 1.0`): a candidate whose stencil touches
        # a fabricated node is infeasible and routed to the direct-Bellman fill.
        if post_decision_valid is not None:
            valid_read = map_coordinates(
                post_decision_valid.astype(post_decision_value.dtype),
                [jnp.atleast_1d(a_index), jnp.atleast_1d(b_index)],
                order=1,
                mode="nearest",
            )[0]
            feasible = feasible & (valid_read >= 1.0 - 1e-9)
        return value, feasible

    return objective


def _crra_utility(consumption: FloatND, crra: ScalarFloat | float) -> FloatND:
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )
