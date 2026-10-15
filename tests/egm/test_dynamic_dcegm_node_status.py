"""Euler candidates retain child query status through the savings map."""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.step_core import _compute_nodes_over_savings, _ComputeNode
from _lcm.solution.feasibility import BoundedResult, QueryInputStatus
from lcm.typing import Float1D, FloatND, ScalarFloat
from tests.conftest import DECIMAL_PRECISION


def _inverse(marginal: ScalarFloat) -> ScalarFloat:
    return 1.0 / marginal


def _utility(action: ScalarFloat) -> ScalarFloat:
    return jnp.log(action)


def test_savings_candidates_keep_child_status_through_a_partial_tile() -> None:
    """One read per saving carries ordered arithmetic and coverage flags beside V."""
    evaluated: list[float] = []

    def record(saving: ScalarFloat) -> None:
        evaluated.append(float(saving))

    def child(
        saving: ScalarFloat,
    ) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
        jax.debug.callback(record, saving, ordered=True)
        return BoundedResult(
            output=(jnp.log(3.0 * saving + 4.0), 3.0 / (3.0 * saving + 4.0)),
            bound_status=(),
            query_status=(
                QueryInputStatus(saving == 0.0, saving == 0.5, saving == 1.0),
            ),
        )

    compute_node = _ComputeNode(
        continuation=child,
        discount_factor=jnp.asarray(1.0),
        inverse_marginal_utility=_inverse,
        utility_of_action=_utility,
    )

    def candidates(
        savings: Float1D,
    ) -> BoundedResult[tuple[FloatND, FloatND, FloatND, FloatND]]:
        result = _compute_nodes_over_savings(
            compute_node=compute_node,
            savings_nodes=savings,
            savings_point_width=2,
            bound_summary=True,
        )
        assert isinstance(result, BoundedResult)
        return result

    result = jax.jit(candidates)(jnp.asarray([0.0, 0.5, 1.0]))
    jax.effects_barrier()
    assert isinstance(result, BoundedResult)
    action, resources, value, continuation = result.output
    tolerance = 10 ** (-DECIMAL_PRECISION)
    np.testing.assert_allclose(
        action, [4.0 / 3.0, 11.0 / 6.0, 7.0 / 3.0], rtol=tolerance
    )
    np.testing.assert_allclose(
        resources, [4.0 / 3.0, 7.0 / 3.0, 10.0 / 3.0], rtol=tolerance
    )
    np.testing.assert_allclose(
        value, np.log([16.0 / 3.0, 121.0 / 12.0, 49.0 / 3.0]), rtol=tolerance
    )
    np.testing.assert_allclose(continuation, np.log([4.0, 5.5, 7.0]), rtol=tolerance)
    assert result.bound_status == ()
    assert result.query_status is not None
    assert len(result.query_status) == 1
    np.testing.assert_array_equal(result.query_status[0], [True, True, True])
    assert evaluated.count(0.0) == 1
    assert evaluated.count(0.5) == 1
    assert evaluated.count(1.0) in (1, 2)
    assert set(evaluated) == {0.0, 0.5, 1.0}
