"""A vmapped envelope query resolves its whole batch in one native call.

The exact winner is an opaque custom call. Batched sequentially it becomes a
loop around that call, which serialises what the caller vectorised — the EGM
step evaluates intervals in parallel within a chunk, so the loop would run once
per interval. Whichever operands carry the batch, the transformed program holds
one call for all of it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.conftest import EXACT_KERNEL_SKIP_REASON, X64_ENABLED

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

# One flat link and one rising link, both spanning `[0, 1]`. The rising link runs
# from 0 to 1 with unit marginal and policy 1; the flat link sits at `level` with
# zero marginal and policy 2, so it owns a query exactly where `level` exceeds it.
RISING_POLICY = 1.0
FLAT_POLICY = 2.0


def _jdtype():
    return jnp.float64 if X64_ENABLED else jnp.float32


def _evaluate(level, query):
    """Publish the envelope of a rising link and a flat link at height `level`."""
    dtype = _jdtype()
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    return envelope_at_query(
        endog_grid=jnp.stack([zero, one, zero, one]),
        policy=jnp.stack([one * RISING_POLICY] * 2 + [one * FLAT_POLICY] * 2),
        value=jnp.stack([zero, one, level, level]),
        marginal=jnp.stack([one, one, zero, zero]),
        segment_id=jnp.stack([zero, zero, one, one]),
        x_query=query,
    )


def _levels(*, n_rows: int):
    """One flat-link height per row, straddling both queries."""
    return jnp.asarray(np.linspace(0.1, 0.9, n_rows), dtype=_jdtype())


def _queries():
    return jnp.asarray([0.25, 0.75], dtype=_jdtype())


def _row_varying_text(*, n_rows: int) -> str:
    """Lower a batch whose candidate segments differ from row to row."""
    levels = _levels(n_rows=n_rows)
    return (
        jax.jit(jax.vmap(_evaluate, in_axes=(0, None)))
        .lower(levels, _queries())
        .as_text()
    )


def _shared_segment_text(*, n_rows: int) -> str:
    """Lower a batch whose rows share one candidate set and differ only in query."""
    queries = jnp.stack([_queries()] * n_rows)
    return (
        jax.jit(jax.vmap(_evaluate, in_axes=(None, 0)))
        .lower(jnp.asarray(0.5, dtype=_jdtype()), queries)
        .as_text()
    )


def test_row_varying_batch_lowers_without_a_sequential_loop() -> None:
    """Batching rows with their own segments emits no loop around the winner."""
    assert _row_varying_text(n_rows=4).count("stablehlo.while") == 0


def test_row_varying_batch_emits_one_batched_winner() -> None:
    """A batch of independent segment sets reaches the batch-native target once."""
    suffix = "F64" if X64_ENABLED else "F32"
    assert _row_varying_text(n_rows=4).count(f"ExactQueryWinnerBatched{suffix}") == 1


def test_row_varying_batch_leaves_no_shared_segment_call_behind() -> None:
    """Only the batch-native target is emitted; the shared-segment body is not."""
    suffix = "F64" if X64_ENABLED else "F32"
    assert _row_varying_text(n_rows=4).count(f"ExactQueryWinner{suffix}") == 0


def test_shared_segment_batch_lowers_without_a_sequential_loop() -> None:
    """Queries batched over one shared segment set emit no loop either."""
    assert _shared_segment_text(n_rows=4).count("stablehlo.while") == 0


def test_shared_segment_batch_keeps_the_shared_segment_target() -> None:
    """Shared segments fold the batch into the query axis, replicating nothing."""
    suffix = "F64" if X64_ENABLED else "F32"
    text = _shared_segment_text(n_rows=4)
    assert text.count(f"ExactQueryWinnerBatched{suffix}") == 0
    assert text.count(f"ExactQueryWinner{suffix}") == 1


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_row_varying_batch_matches_row_by_row_evaluation(channel: int) -> None:
    """The batched winner publishes what the same rows publish one at a time."""
    levels = _levels(n_rows=5)
    queries = _queries()
    batched = jax.jit(jax.vmap(_evaluate, in_axes=(0, None)))(levels, queries)
    one_at_a_time = [jax.jit(_evaluate)(level, queries) for level in levels]
    expected = np.stack([np.asarray(row[channel]) for row in one_at_a_time])
    np.testing.assert_array_equal(np.asarray(batched[channel]), expected)


def test_row_varying_batch_publishes_the_owning_links_policy() -> None:
    """The flat link owns a query exactly where its height exceeds the query."""
    levels = _levels(n_rows=5)
    queries = _queries()
    _value, policy, _marginal = jax.jit(jax.vmap(_evaluate, in_axes=(0, None)))(
        levels, queries
    )
    flat_owns = np.asarray(levels)[:, None] > np.asarray(queries)[None, :]
    expected = np.where(flat_owns, FLAT_POLICY, RISING_POLICY)
    np.testing.assert_array_equal(np.asarray(policy), expected)
