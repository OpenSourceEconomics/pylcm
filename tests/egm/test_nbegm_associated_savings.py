"""A candidate's original saving follows its value and native envelope marginal."""

import inspect
from collections.abc import Callable
from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.comparison_arithmetic import ComparisonArithmetic
from _lcm.egm.upper_envelope.query import (
    EnvelopeWinner,
    empty_envelope_winner,
    envelope_at_query,
    finish_envelope_winner,
    merge_envelope_winner,
)
from lcm.typing import Float1D, FloatND, IntND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

type _AssociatedRead = tuple[FloatND, FloatND, FloatND, FloatND, IntND]


class _Candidates(TypedDict):
    endog_grid: Float1D
    policy: Float1D
    value: Float1D
    marginal: Float1D
    segment_id: Float1D
    x_query: FloatND


def _assert_original_tuple(*, result: _AssociatedRead) -> None:
    value, policy, marginal, saving, owner = result
    for actual, expected in (
        (value, [1.5, 1.0, 1.5]),
        (policy, [10.5, 21.0, 21.5]),
        (marginal, [2.5, 10.0, 11.0]),
        (saving, [102.0, 205.0, 207.0]),
    ):
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=DECIMAL_PRECISION
        )
    np.testing.assert_array_equal(owner, [0, 2, 2])


@pytest.mark.parametrize("arithmetic", ["certified", "ordinary"])
def test_envelope_without_savings_keeps_its_state_and_published_channels(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """Absent savings retain the twelve-leaf state and three/four-channel reads."""
    query = jnp.asarray([0.5, 1.0, 1.5])
    expected = EnvelopeWinner(
        *(jnp.zeros_like(query) for _ in range(8)),
        live=jnp.zeros_like(query, dtype=bool),
        stable_index=jnp.full(query.shape, jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
        settled=jnp.ones_like(query, dtype=bool),
        unreadable=jnp.zeros_like(query, dtype=bool),
    )
    for actual in (
        empty_envelope_winner(query=query),
        jax.jit(empty_envelope_winner)(query=query),
    ):
        assert type(actual) is EnvelopeWinner
        assert len(actual) == 12
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        leaves = jax.tree.leaves(actual)
        assert len(leaves) == 12
        for actual_leaf, expected_leaf in zip(leaves, expected, strict=True):
            assert actual_leaf.shape == expected_leaf.shape
            assert actual_leaf.dtype == expected_leaf.dtype
            np.testing.assert_array_equal(actual_leaf, expected_leaf)
    candidates: _Candidates = {
        "endog_grid": jnp.asarray([0.0, 2.0]),
        "policy": jnp.asarray([10.0, 12.0]),
        "value": jnp.asarray([2.0, 0.0]),
        "marginal": jnp.asarray([2.0, 4.0]),
        "segment_id": jnp.asarray([0.0, 0.0]),
        "x_query": query,
    }
    read = jax.jit(envelope_at_query, static_argnames=("arithmetic", "return_owner"))
    for return_owner in (False, True):
        eager = envelope_at_query(
            **candidates, arithmetic=arithmetic, return_owner=return_owner
        )
        compiled = read(**candidates, arithmetic=arithmetic, return_owner=return_owner)
        assert len(eager) == len(compiled) == (4 if return_owner else 3)
        for actual, expected_channel in zip(
            eager[:3],
            ([1.5, 1.0, 0.5], [10.5, 11.0, 11.5], [2.5, 3.0, 3.5]),
            strict=True,
        ):
            np.testing.assert_array_almost_equal(
                actual, expected_channel, decimal=DECIMAL_PRECISION
            )
        for eager_channel, compiled_channel in zip(eager, compiled, strict=True):
            np.testing.assert_array_equal(eager_channel, compiled_channel)
        if return_owner:
            np.testing.assert_array_equal(eager[-1], [0, 0, 0])
    blocks = {key: value for key, value in candidates.items() if key != "x_query"}
    merge = cast("Callable[..., EnvelopeWinner]", merge_envelope_winner)
    merge_compiled = jax.jit(merge, static_argnames=("arithmetic",))
    eager_state = merge(
        **blocks,
        held=expected,
        stable_index=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        query=query,
        arithmetic=arithmetic,
    )
    compiled_state = merge_compiled(
        **blocks,
        held=expected,
        stable_index=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        query=query,
        arithmetic=arithmetic,
    )
    for state in (eager_state, compiled_state):
        assert type(state) is EnvelopeWinner
        assert len(jax.tree.leaves(state)) == 12
        assert jax.tree.structure(state) == jax.tree.structure(expected)
        finished = finish_envelope_winner(
            winner=state, query=query, arithmetic=arithmetic
        )
        assert len(finished) == 3
        for actual, expected_channel in zip(finished, eager[:3], strict=True):
            np.testing.assert_array_equal(actual, expected_channel)
    for eager_leaf, compiled_leaf in zip(eager_state, compiled_state, strict=True):
        assert eager_leaf.shape == compiled_leaf.shape
        assert eager_leaf.dtype == compiled_leaf.dtype
        np.testing.assert_array_equal(eager_leaf, compiled_leaf)


def test_associated_savings_require_matching_candidate_shape_dtype_and_state() -> None:
    """An incomplete or differently shaped saving channel fails before ownership."""
    query = jnp.asarray([0.5])
    candidates = {
        "endog_grid": jnp.asarray([0.0, 2.0]),
        "policy": jnp.asarray([10.0, 12.0]),
        "value": jnp.asarray([2.0, 0.0]),
        "marginal": jnp.asarray([2.0, 4.0]),
        "segment_id": jnp.asarray([0.0, 0.0]),
    }
    saving = jnp.asarray([101.0, 105.0])
    read = cast("Callable[..., _AssociatedRead]", envelope_at_query)
    for invalid, message in (
        (saving[:1], "shape"),
        (jnp.asarray([101.0, 105.0], dtype=jnp.float16), "dtype"),
    ):
        with pytest.raises(ValueError, match=message):
            read(**candidates, x_query=query, candidate_savings=invalid)
    standing = empty_envelope_winner(query=query, retain_savings=True)
    for held, original in (
        (empty_envelope_winner(query=query), saving),
        (standing, None),
    ):
        merge = cast("Callable[..., EnvelopeWinner]", merge_envelope_winner)
        with pytest.raises(ValueError, match="together"):
            merge(
                **candidates,
                held=held,
                candidate_savings=original,
                stable_index=jnp.asarray([0, 1, 2], dtype=jnp.int32),
                query=query,
                arithmetic="certified",
            )


@pytest.mark.parametrize("arithmetic", ["certified", "ordinary"])
def test_query_envelope_preserves_original_saving_from_its_winning_link(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """Crossing branches select original saving with the same native marginal."""
    candidates: _Candidates = {
        "endog_grid": jnp.asarray([0.0, 2.0, 0.0, 2.0]),
        "policy": jnp.asarray([10.0, 12.0, 20.0, 22.0]),
        "value": jnp.asarray([2.0, 0.0, 0.0, 2.0]),
        "marginal": jnp.asarray([2.0, 4.0, 8.0, 12.0]),
        "segment_id": jnp.asarray([0.0, 0.0, 1.0, 1.0]),
        "x_query": jnp.asarray([0.5, 1.0, 1.5]),
    }
    control = envelope_at_query(**candidates, arithmetic=arithmetic, return_owner=True)
    for actual, expected in zip(
        control[:3],
        ([1.5, 1.0, 1.5], [10.5, 21.0, 21.5], [2.5, 10.0, 11.0]),
        strict=True,
    ):
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=DECIMAL_PRECISION
        )
    np.testing.assert_array_equal(control[3], [0, 2, 2])
    assert "candidate_savings" in inspect.signature(envelope_at_query).parameters
    read = cast("Callable[..., _AssociatedRead]", envelope_at_query)
    result = jax.jit(read, static_argnames=("arithmetic", "return_owner"))(
        **candidates,
        candidate_savings=jnp.asarray([101.0, 105.0, 201.0, 209.0]),
        arithmetic=arithmetic,
        return_owner=True,
    )
    _assert_original_tuple(result=result)


@pytest.mark.parametrize("arithmetic", ["certified", "ordinary"])
def test_streamed_envelope_keeps_original_saving_through_a_partial_dead_block(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """Standing endpoint savings survive a later branch and a one-node dead block."""
    query = jnp.asarray([0.5, 1.0, 1.5])
    control = empty_envelope_winner(query=query)
    control = merge_envelope_winner(
        held=control,
        endog_grid=jnp.asarray([0.0, 2.0]),
        policy=jnp.asarray([10.0, 12.0]),
        value=jnp.asarray([2.0, 0.0]),
        marginal=jnp.asarray([2.0, 4.0]),
        segment_id=jnp.asarray([0.0, 0.0]),
        stable_index=jnp.asarray([0, 4, 5], dtype=jnp.int32),
        query=query,
        arithmetic=arithmetic,
    )
    np.testing.assert_array_almost_equal(
        finish_envelope_winner(winner=control, query=query, arithmetic=arithmetic)[0],
        [1.5, 1.0, 0.5],
        decimal=DECIMAL_PRECISION,
    )
    assert "candidate_savings" in inspect.signature(merge_envelope_winner).parameters
    assert "retain_savings" in inspect.signature(empty_envelope_winner).parameters
    initialize = cast("Callable[..., EnvelopeWinner]", empty_envelope_winner)
    merge = cast("Callable[..., EnvelopeWinner]", merge_envelope_winner)
    finish = cast("Callable[..., _AssociatedRead]", finish_envelope_winner)
    winner = initialize(query=query, retain_savings=True)
    for grid, policy, value, marginal, saving, segment, identities in (
        (
            [0.0, 2.0],
            [10.0, 12.0],
            [2.0, 0.0],
            [2.0, 4.0],
            [101.0, 105.0],
            [0.0, 0.0],
            [0, 4, 5],
        ),
        (
            [0.0, 2.0],
            [20.0, 22.0],
            [0.0, 2.0],
            [8.0, 12.0],
            [201.0, 209.0],
            [1.0, 1.0],
            [2, 6, 7],
        ),
        ([jnp.nan], [jnp.nan], [jnp.nan], [jnp.nan], [jnp.nan], [2.0], [8]),
    ):
        winner = merge(
            held=winner,
            endog_grid=jnp.asarray(grid),
            policy=jnp.asarray(policy),
            value=jnp.asarray(value),
            marginal=jnp.asarray(marginal),
            candidate_savings=jnp.asarray(saving),
            segment_id=jnp.asarray(segment),
            stable_index=jnp.asarray(identities, dtype=jnp.int32),
            query=query,
            arithmetic=arithmetic,
        )
    _assert_original_tuple(
        result=finish(winner=winner, query=query, arithmetic=arithmetic)
    )
