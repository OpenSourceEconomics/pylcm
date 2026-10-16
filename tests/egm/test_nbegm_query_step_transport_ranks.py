"""Single NB rows retain rank checks while mapped records retain their axes."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.egm.nbegm_step import (
    _make_nbegm_query_data,
    _NBEGMQueryStep,
    _single_row_nbegm_query_step,
)
from _lcm.utils.dispatchers import map_over_leading_axis
from lcm.typing import ScalarFloat


def _one_row() -> _NBEGMQueryStep:
    value = jnp.asarray([3.0, 7.0])
    marginal = jnp.asarray([5.0, 11.0])
    policy = jnp.asarray([2.0, 4.0])
    data = _make_nbegm_query_data(
        value=value,
        marginal=marginal,
        policy=policy,
        savings=jnp.asarray([0.5, 1.0]),
        floor_used=jnp.asarray(0.25),
        liquid_grid=jnp.asarray([2.5, 5.0]),
        breakpoints=jnp.asarray([]),
        corner_continuation=jnp.asarray([13.0]),
    )
    return _single_row_nbegm_query_step(
        value=value,
        marginal=marginal,
        policy=policy,
        query_data=data,
    )


@pytest.mark.parametrize("channel", ["value", "marginal", "policy"])
def test_single_row_producer_rejects_rank_two_channel(
    *,
    channel: Literal["value", "marginal", "policy"],
) -> None:
    """Each ordinary economic channel must be one liquid-query row at production."""
    row = _one_row()
    value = row.value[jnp.newaxis] if channel == "value" else row.value
    marginal = row.marginal[jnp.newaxis] if channel == "marginal" else row.marginal
    policy = row.policy[jnp.newaxis] if channel == "policy" else row.policy
    with pytest.raises(BeartypeCallHintParamViolation, match=f"parameter {channel}="):
        _single_row_nbegm_query_step(
            value=value,
            marginal=marginal,
            policy=policy,
            query_data=row.query_data,
        )


@pytest.mark.parametrize("batch_size", [0, 1], ids=["vmap", "batch-one"])
def test_mapped_query_record_preserves_associated_channels_and_data(
    *,
    batch_size: int,
) -> None:
    """Mapping retains the leading branch axis and every original query-data leaf."""
    row = _one_row()

    def at_branch(offset: ScalarFloat) -> _NBEGMQueryStep:
        return _single_row_nbegm_query_step(
            value=row.value + offset,
            marginal=row.marginal,
            policy=row.policy,
            query_data=row.query_data,
        )

    mapped = map_over_leading_axis(
        func=at_branch,
        xs=jnp.asarray([0.0, 10.0]),
        batch_size=batch_size,
    )
    assert isinstance(mapped, _NBEGMQueryStep)
    assert len(mapped) == 4
    np.testing.assert_array_equal(mapped.value, [[3.0, 7.0], [13.0, 17.0]])
    np.testing.assert_array_equal(mapped.marginal, [[5.0, 11.0], [5.0, 11.0]])
    np.testing.assert_array_equal(mapped.policy, [[2.0, 4.0], [2.0, 4.0]])
    assert jax.tree.structure(mapped.query_data) == jax.tree.structure(row.query_data)
    for actual, original in zip(
        jax.tree.leaves(mapped.query_data),
        jax.tree.leaves(row.query_data),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, jnp.stack((original, original)))
