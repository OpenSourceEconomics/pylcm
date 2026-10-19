"""Original savings share selected link geometry and every channel's read status."""

from collections.abc import Callable
from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.comparison_arithmetic import ComparisonArithmetic
from _lcm.egm.upper_envelope.certified_sign import (
    backend_flushes_subnormals,
    is_subnormal,
)
from _lcm.egm.upper_envelope.query import (
    NO_OWNER,
    empty_envelope_winner,
    envelope_at_query,
    finish_envelope_winner,
    merge_envelope_winner,
)
from lcm.typing import Float1D, FloatND, IntND
from tests.conftest import EXACT_KERNEL_SKIP_REASON, X64_ENABLED

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

type _AssociatedRead = tuple[FloatND, FloatND, FloatND, FloatND, IntND]
type _Blocks = tuple[tuple[int, int, tuple[int, ...]], ...]


class _Candidates(TypedDict):
    endog_grid: Float1D
    policy: Float1D
    value: Float1D
    marginal: Float1D
    segment_id: Float1D
    x_query: FloatND


def _fold(
    *,
    candidates: _Candidates,
    savings: Float1D,
    blocks: _Blocks,
    arithmetic: ComparisonArithmetic,
) -> _AssociatedRead:
    winner = empty_envelope_winner(query=candidates["x_query"], retain_savings=True)
    for start, stop, identities in blocks:
        winner = merge_envelope_winner(
            held=winner,
            endog_grid=candidates["endog_grid"][start:stop],
            policy=candidates["policy"][start:stop],
            value=candidates["value"][start:stop],
            marginal=candidates["marginal"][start:stop],
            segment_id=candidates["segment_id"][start:stop],
            candidate_savings=savings[start:stop],
            stable_index=jnp.asarray(identities, dtype=jnp.int32),
            query=candidates["x_query"],
            arithmetic=arithmetic,
        )
    return finish_envelope_winner(
        winner=winner, query=candidates["x_query"], arithmetic=arithmetic
    )


def _layouts(
    *,
    candidates: _Candidates,
    savings: Float1D,
    blocks: _Blocks,
    arithmetic: ComparisonArithmetic,
) -> tuple[_AssociatedRead, ...]:
    read = cast("Callable[..., _AssociatedRead]", envelope_at_query)
    compiled_read = jax.jit(read, static_argnames=("arithmetic", "return_owner"))
    compiled_fold = jax.jit(_fold, static_argnames=("blocks", "arithmetic"))
    return (
        read(
            **candidates,
            candidate_savings=savings,
            arithmetic=arithmetic,
            return_owner=True,
        ),
        compiled_read(
            **candidates,
            candidate_savings=savings,
            arithmetic=arithmetic,
            return_owner=True,
        ),
        _fold(
            candidates=candidates, savings=savings, blocks=blocks, arithmetic=arithmetic
        ),
        compiled_fold(
            candidates=candidates,
            savings=savings,
            blocks=blocks,
            arithmetic=arithmetic,
        ),
    )


def _single_link() -> _Candidates:
    return {
        "endog_grid": jnp.asarray([0.0, 2.0]),
        "policy": jnp.asarray([10.0, 12.0]),
        "value": jnp.asarray([1.0, 3.0]),
        "marginal": jnp.asarray([2.0, 4.0]),
        "segment_id": jnp.asarray([0.0, 0.0]),
        "x_query": jnp.asarray([1.0]),
    }


def _assert_refused(*, result: _AssociatedRead) -> None:
    for channel in result[:4]:
        np.testing.assert_array_equal(jnp.isnan(channel), [True])
    np.testing.assert_array_equal(result[4], [NO_OWNER])


@pytest.mark.parametrize("arithmetic", ["certified", "ordinary"])
def test_original_savings_follow_descending_links_and_single_node_winners(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """Descending links and an isolated point retain their own original savings."""
    candidates: _Candidates = {
        "endog_grid": jnp.asarray([2.0, 0.0, 1.0, jnp.nan]),
        "policy": jnp.asarray([12.0, 10.0, 30.0, jnp.nan]),
        "value": jnp.asarray([2.0, 0.0, 3.0, jnp.nan]),
        "marginal": jnp.asarray([4.0, 2.0, 6.0, jnp.nan]),
        "segment_id": jnp.asarray([0.0, 0.0, 1.0, 2.0]),
        "x_query": jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0]),
    }
    for result in _layouts(
        candidates=candidates,
        savings=jnp.asarray([105.0, 101.0, 301.0, jnp.nan]),
        blocks=((0, 2, (0, 3, 4)), (2, 3, (5,)), (3, 4, (6,))),
        arithmetic=arithmetic,
    ):
        for channel, expected in zip(
            result,
            (
                [0.0, 0.5, 3.0, 1.5, 2.0],
                [10.0, 10.5, 30.0, 11.5, 12.0],
                [2.0, 2.5, 6.0, 3.5, 4.0],
                [101.0, 102.0, 301.0, 104.0, 105.0],
                [0, 0, 5, 0, 0],
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(channel, expected)


@pytest.mark.parametrize("arithmetic", ["certified", "ordinary"])
@pytest.mark.parametrize("invalid_saving", [np.nan, np.inf], ids=["nan", "inf"])
def test_selected_nonfinite_savings_refuse_every_published_channel(
    *, arithmetic: ComparisonArithmetic, invalid_saving: float
) -> None:
    """A selected unreadable saving withholds value, policy, marginal and owner."""
    candidates = _single_link()
    control = envelope_at_query(**candidates, arithmetic=arithmetic, return_owner=True)
    for channel, expected in zip(control, ([2.0], [11.0], [3.0], [0]), strict=True):
        np.testing.assert_array_equal(channel, expected)
    for result in _layouts(
        candidates=candidates,
        savings=jnp.asarray([101.0, invalid_saving]),
        blocks=((0, 2, (0, 1, 2)),),
        arithmetic=arithmetic,
    ):
        _assert_refused(result=result)


@pytest.mark.parametrize("arithmetic", ["certified", "ordinary"])
def test_constant_subnormal_savings_publish_their_bits_or_refuse_together(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """Native reads retain subnormal savings; flushing ordinary reads refuse."""
    candidates = _single_link()
    control = envelope_at_query(**candidates, arithmetic=arithmetic, return_owner=True)
    for channel, expected in zip(control, ([2.0], [11.0], [3.0], [0]), strict=True):
        np.testing.assert_array_equal(channel, expected)
    small = np.ldexp(1.0, -1030 if X64_ENABLED else -130)
    savings = jnp.asarray([small, small], dtype=candidates["endog_grid"].dtype)
    np.testing.assert_array_equal(is_subnormal(savings), [True, True])
    must_refuse = arithmetic == "ordinary" and backend_flushes_subnormals(
        dtype=savings.dtype
    )
    unsigned = jnp.uint64 if X64_ENABLED else jnp.uint32
    expected_bits = jax.lax.bitcast_convert_type(savings[:1], unsigned)
    for result in _layouts(
        candidates=candidates,
        savings=savings,
        blocks=((0, 2, (0, 1, 2)),),
        arithmetic=arithmetic,
    ):
        if must_refuse:
            _assert_refused(result=result)
        else:
            for channel, expected in zip(
                result[:3], ([2.0], [11.0], [3.0]), strict=True
            ):
                np.testing.assert_array_equal(channel, expected)
            np.testing.assert_array_equal(
                jax.lax.bitcast_convert_type(result[3], unsigned), expected_bits
            )
            np.testing.assert_array_equal(result[4], [0])
