"""NB-EGM targets whichever side of a jump carries the higher continuation.

Saving to land exactly on a boundary is a corner the Euler equation never produces,
so it enters the envelope as its own candidate. Which side of the boundary that
candidate should aim at depends on the direction of the jump, and the direction is a
property of the continuation, not something the solver may assume: a continuation that
jumps *up* at the limit rewards landing on the owning side of the boundary, and a
continuation that jumps *down* rewards landing just inside the other side.

Both targets are therefore offered and the envelope decides.
"""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from _lcm.egm.nbegm_step import nbegm_one_asset_step, nbegm_unified_step

ASSET_LIMIT = 1.0
DISCOUNT_FACTOR = 0.95
UPPER_SHIFT = 10.0
LOWER_SHIFT = 1.0


def _upward_jump_value(liquid: np.ndarray) -> np.ndarray:
    """Continuation that jumps up at the limit, owned above by the otherwise side."""
    return np.where(
        liquid < ASSET_LIMIT,
        np.log(liquid + LOWER_SHIFT),
        np.log(liquid + UPPER_SHIFT),
    )


def _upward_jump_marginal(liquid: np.ndarray) -> np.ndarray:
    """Derivative of `_upward_jump_value`, taken within each side of the jump."""
    return np.where(
        liquid < ASSET_LIMIT,
        1.0 / (liquid + LOWER_SHIFT),
        1.0 / (liquid + UPPER_SHIFT),
    )


def _solve_one_step() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one NB-EGM step against the upward-jumping continuation.

    Log utility, no return and no income, so next-period liquid equals savings and
    the boundary is reachable by saving exactly `ASSET_LIMIT`. The subsidy is the
    same on both sides, which leaves the boundary direction as the only thing under
    test.
    """
    liquid_grid = jnp.linspace(0.2, 4.0, 39)
    savings_grid = jnp.linspace(0.0, 3.8, 39)
    next_value = jnp.asarray(_upward_jump_value(np.asarray(liquid_grid)))
    next_marginal = jnp.asarray(_upward_jump_marginal(np.asarray(liquid_grid)))
    value, _marginal, consumption = nbegm_one_asset_step(
        next_value=next_value,
        next_marginal=next_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=DISCOUNT_FACTOR,
        crra=1.0,
        return_liquid=0.0,
        income=0.0,
        subsidy_when=0.0,
        subsidy_otherwise=0.0,
        asset_limit=ASSET_LIMIT,
        equality_owner="otherwise",
    )
    return np.asarray(liquid_grid), np.asarray(value), np.asarray(consumption)


def test_saving_onto_an_upward_jump_is_valued_at_the_higher_continuation():
    """A continuation jumping up at the limit is worth `beta * log(11)` to reach.

    With liquid `2.0` and log utility, saving exactly `1.0` lands on the owning side
    of the jump and consumes the remaining `1.0`, so the value is
    `log(1) + 0.95 * log(11)`. Staying strictly below the limit is worth far less.
    """
    liquid_grid, value, _ = _solve_one_step()
    at_two = int(np.argmin(np.abs(liquid_grid - 2.0)))
    expected = DISCOUNT_FACTOR * np.log(2.0 + UPPER_SHIFT - ASSET_LIMIT)
    assert_allclose(value[at_two], expected, rtol=1e-4)


def test_saving_onto_an_upward_jump_consumes_the_cash_above_the_limit():
    """The policy that reaches the jump consumes exactly what the limit leaves over.

    At liquid `2.0` the optimum saves `1.0` to land on the boundary, so consumption
    is `1.0` — not the `1.54` the interior Euler path below the limit would choose.
    """
    liquid_grid, _, consumption = _solve_one_step()
    at_two = int(np.argmin(np.abs(liquid_grid - 2.0)))
    assert_allclose(consumption[at_two], 1.0, rtol=1e-4)


def _solve_one_piecewise_affine_step() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one piecewise-affine NB-EGM step against the same upward jump.

    The budget is a pure jump: cash-on-hand equals liquid on both sides of the
    single breakpoint, so the boundary direction is again the only thing under
    test.
    """
    liquid_grid = jnp.linspace(0.2, 4.0, 39)
    savings_grid = jnp.linspace(0.0, 3.8, 39)
    next_value = jnp.asarray(_upward_jump_value(np.asarray(liquid_grid)))
    next_marginal = jnp.asarray(_upward_jump_marginal(np.asarray(liquid_grid)))
    value, _marginal, consumption = nbegm_unified_step(
        next_value=next_value,
        next_marginal=next_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=DISCOUNT_FACTOR,
        crra=1.0,
        gross_return=1.0,
        income=0.0,
        coh_slopes=jnp.ones(2),
        coh_intercepts=jnp.zeros(2),
        breakpoints=jnp.asarray([ASSET_LIMIT]),
        jump_mask=(True,),
        equality_owner="otherwise",
    )
    return np.asarray(liquid_grid), np.asarray(value), np.asarray(consumption)


def test_piecewise_affine_step_values_the_upward_jump_at_its_higher_side():
    """The piecewise-affine budget reaches the same `beta * log(11)` corner.

    A jump breakpoint in a piecewise-affine budget is the same corner as a
    case-boundary jump, so a node that can afford to land on the high side is
    worth the same there.
    """
    liquid_grid, value, _ = _solve_one_piecewise_affine_step()
    at_two = int(np.argmin(np.abs(liquid_grid - 2.0)))
    expected = DISCOUNT_FACTOR * np.log(2.0 + UPPER_SHIFT - ASSET_LIMIT)
    assert_allclose(value[at_two], expected, rtol=1e-4)


def test_piecewise_affine_step_consumes_the_cash_above_the_jump():
    """The piecewise-affine policy saves exactly to the breakpoint and eats the rest."""
    liquid_grid, _, consumption = _solve_one_piecewise_affine_step()
    at_two = int(np.argmin(np.abs(liquid_grid - 2.0)))
    assert_allclose(consumption[at_two], 1.0, rtol=1e-4)
