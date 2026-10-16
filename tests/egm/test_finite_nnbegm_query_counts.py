"""Count each contributing native row independently of its outer winner."""

from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution.feasibility import ContinuationReadCounts
from _lcm.typing import EconFunctionKwargs
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_finite_nnbegm_query import _bank, _FiniteResult, _query

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.mark.parametrize(
    ("passive", "value", "marginal"),
    [(0.5, 5.0, 5.0), (1.5, 7.0, 11.0)],
    ids=["interpolation", "signed-extrapolation"],
)
def test_checked_outer_winners_count_original_passive_rows(
    *, passive: float, value: float, marginal: float
) -> None:
    """Both nonzero passive weights count all original keeper and adjuster rows."""
    query = _query(actions=False)
    values = jnp.asarray(((4.0, 0.0), (0.0, 6.0)))
    grid, ordinates, marginals, data = _bank(
        values=values,
        marginals=jnp.asarray(((2.0, 13.0), (17.0, 8.0))),
        savings=jnp.full_like(values, 0.5),
    )
    result, counts = cast(
        "tuple[_FiniteResult, ContinuationReadCounts]",
        query.read_choices(
            grid=grid,
            value=ordinates,
            marginal=marginals,
            data=data,
            row_queries=jnp.full((2,), 1.0),
            row_gradients=jnp.ones((2,)),
            passive_values=(jnp.asarray(passive),),
            passive_grids=(jnp.asarray((0.0, 1.0)),),
            action_names=(),
            action_values=(),
            pool=cast(
                "EconFunctionKwargs",
                {
                    "liquid": jnp.asarray(1.0),
                    "stock": jnp.asarray(passive),
                    "floor_scale": jnp.asarray(0.0),
                    "bonus": jnp.asarray(-10.0),
                    "discount_factor": jnp.asarray(0.0),
                },
            ),
            euler_gradient=jnp.asarray(1.0),
            collect_continuation_read_counts=True,
        ),
    )
    np.testing.assert_array_equal(result.value, value)
    np.testing.assert_array_equal(result.marginal, marginal)
    assert not bool(jnp.any(result.coverage_missing))
    assert not bool(jnp.any(result.bound_status.nonfinite_floor))
    assert not bool(jnp.any(result.bound_status.invalid_shifted_grid))
    assert int(counts.eligible) == 4
    assert int(counts.below_first_live) == 0
