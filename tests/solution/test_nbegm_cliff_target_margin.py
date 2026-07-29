"""Save-to-cliff targets land just past the cliff, at every scale and precision.

A child value jump creates a one-sided optimum: save to just inside the cliff's
owning side. The displacement must clear the rounding of the affine savings law
that maps the target onto the child's liquid axis — and no more, or it overshoots
the next cliff.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution.nbegm import cliff_target_margin


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("jump", [1.0, 1e5])
def test_the_margin_clears_the_cliff_on_both_sides(dtype, jump: float) -> None:
    """Both nudged targets map strictly past the jump on their own side."""
    slope = jnp.asarray(1.05, dtype=dtype)
    intercept = jnp.asarray(4.0, dtype=dtype)
    jumps = jnp.asarray([jump], dtype=dtype)
    s_star = (jumps - intercept) / slope
    margin = cliff_target_margin(
        s_star=s_star, slope=slope, intercept=intercept, dtype=dtype
    )
    below = slope * (s_star - margin) + intercept
    above = slope * (s_star + margin) + intercept
    assert float(below[0]) < jump < float(above[0])


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_the_margin_never_reaches_a_neighbouring_cliff(dtype) -> None:
    """Two preimages a hair apart nudge by at most a quarter of their separation."""
    slope = jnp.asarray(1.0, dtype=dtype)
    intercept = jnp.asarray(0.0, dtype=dtype)
    s_star = jnp.asarray([1.0, 1.0 + 1e-3], dtype=dtype)
    margin = cliff_target_margin(
        s_star=s_star, slope=slope, intercept=intercept, dtype=dtype
    )
    np.testing.assert_array_less(np.asarray(margin), 0.25 * 1e-3 + 1e-12)


def _relative_margin_in_eps(dtype) -> float:
    """Displacement as a share of `s_star`, expressed in units of the dtype's eps."""
    slope = jnp.asarray(1.0, dtype=dtype)
    intercept = jnp.asarray(4.0, dtype=dtype)
    s_star = (jnp.asarray([100.0], dtype=dtype) - intercept) / slope
    margin = cliff_target_margin(
        s_star=s_star, slope=slope, intercept=intercept, dtype=dtype
    )
    return float(margin[0] / s_star[0]) / float(jnp.finfo(dtype).eps)


@pytest.mark.usefixtures("x64_enabled")
def test_the_margin_is_relatively_the_same_in_both_precisions() -> None:
    """The displacement is a fixed count of the law's roundings, not a fixed band.

    Scaling the margin by `|s_star|` and a fixed ULP count instead makes its
    relative size precision-dependent: the same nudge moves a float32 target
    orders of magnitude further, relative to the target, than a float64 one.
    """
    np.testing.assert_allclose(
        _relative_margin_in_eps(jnp.float32),
        _relative_margin_in_eps(jnp.float64),
        rtol=1e-6,
    )
