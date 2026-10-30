"""Original template bits through the genuine admitted preparation owner."""

import jax
import numpy as np
import pytest

from _lcm.egm.carry import build_template_egm_carry
from _lcm.egm.preparation import prepare_egm_carry, prepare_egm_query_data
from _lcm.egm.query_data import build_template_egm_query_data
from tests.egm.test_current_iid_generation_materialization import _owner

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@pytest.mark.parametrize("liquid_corner", [False, True])
def test_admitted_templates_keep_original_dtype_and_leaf_bits(
    *, liquid_corner: bool
) -> None:
    expected_query = build_template_egm_query_data(
        state_shape=(2,),
        action_shape=(3,),
        row_capacity=4,
        interval_capacity=2,
        liquid_corner=liquid_corner,
    )
    expected = build_template_egm_carry(
        n_rows=4, leading_shape=(2, 3), query_data=expected_query
    )
    owner = _owner(roots=(expected,))
    try:
        actual_query = prepare_egm_query_data(
            state_shape=(2,),
            action_shape=(3,),
            row_capacity=4,
            interval_capacity=2,
            liquid_corner=liquid_corner,
            materializer=owner,
        )
        actual = prepare_egm_carry(
            n_rows=4,
            leading_shape=(2, 3),
            query_data=actual_query,
            materializer=owner,
        )
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        for wanted, observed in zip(
            jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
        ):
            assert isinstance(wanted, jax.Array)
            assert isinstance(observed, jax.Array)
            assert observed.shape == wanted.shape
            assert observed.dtype == wanted.dtype
            assert observed.weak_type == wanted.weak_type
            assert np.asarray(observed).tobytes() == np.asarray(wanted).tobytes()
    finally:
        owner.close()
