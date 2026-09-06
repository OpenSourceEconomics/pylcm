"""The production interval-block fold owns each query by global identity alone.

`merge_envelope_winner` folds one candidate block into a standing winner whose
identity is the global stored-link index of the candidate that produced it. Which
candidate owns a query is therefore a function of the candidate records and their
identities — never of the block a record is stored in, the slot it holds there, or
the dead blocks a winner is carried through. These cases put candidates within a
few units in the last place of one another, the neighbourhood a fold that decided
by storage position would resolve differently per layout, and require the same
owner from every layout. Where the documented total order decides on stored fields
alone it is also checked literally: a strictly larger stored value owns the query,
and an exact tie falls to the smallest stable identity. The published payload is the
owner's stored record, so a wrong owner is a wrong level as well.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope._exact_affine.ffi import (
    kernel_built_for_current_backend,
)
from _lcm.egm.upper_envelope.query import (
    ComparisonArithmetic,
    EnvelopeWinner,
    empty_envelope_winner,
    finish_envelope_winner,
    merge_envelope_winner,
)
from tests.conftest import X64_ENABLED

# Self-brackets are named above every consecutive link, as in the one-shot layout
# where the links precede the self-brackets in stored order.
_SELF_BASE = 10_000
_LOCAL = (2, 3, 4)
_NONLOCAL = (350, 7, 1000)
_CANONICAL = (0, 1, 2)


class _Case(NamedTuple):
    """One near-tie neighbourhood and one layout of it."""

    value_ulp: int
    """Units in the last place by which the first value exceeds the second."""
    payload_ulp: int
    """Units in the last place between the near-tied records' policy and marginal."""
    identities: tuple[int, ...]
    """Global identity of each canonical candidate."""
    order: tuple[int, ...]
    """Storage order of the canonical candidates across the layout."""
    block_size: int
    """Candidates per folded block."""
    dead_block_after: int | None
    """Index of the block after which an all-dead block is folded, if any."""


_LAYOUTS = (
    (_LOCAL, (0, 1, 2), 3, None),
    (_NONLOCAL, (2, 1, 0), 1, None),
    (_NONLOCAL, (1, 2, 0), 2, 0),
    (_LOCAL, (2, 0, 1), 1, 0),
)
_CASES = tuple(
    _Case(value_ulp, value_ulp, identities, order, block_size, dead_block_after)
    for value_ulp in (0, 1, 2, 16)
    for identities, order, block_size, dead_block_after in _LAYOUTS
)
_DECIDED = tuple(case for case in _CASES if case.value_ulp > 0)
_TIED = tuple(case for case in _CASES if case.value_ulp == 0)


def _case_id(case: _Case) -> str:
    return (
        f"v{case.value_ulp}ulp-ids{'local' if case.identities == _LOCAL else 'far'}"
        f"-order{''.join(map(str, case.order))}-block{case.block_size}"
        f"-dead{case.dead_block_after}"
    )


def _dtype() -> np.dtype:
    return np.dtype(np.float64 if X64_ENABLED else np.float32)


def _ulp_from(*, anchor: float, n_ulp: int) -> float:
    out = np.asarray(anchor, dtype=_dtype())
    step = np.asarray(np.inf if n_ulp >= 0 else -np.inf, dtype=_dtype())
    for _ in range(abs(n_ulp)):
        out = np.nextafter(out, step)
    assert np.isfinite(out)
    return float(out)


class _Records(NamedTuple):
    """Three zero-width self-brackets at the query, in canonical order."""

    value: tuple[float, ...]
    policy: tuple[float, ...]
    marginal: tuple[float, ...]


def _records(case: _Case) -> _Records:
    low = 1.0
    return _Records(
        value=(
            _ulp_from(anchor=low, n_ulp=case.value_ulp),
            low,
            _ulp_from(anchor=low, n_ulp=-32),
        ),
        policy=(_ulp_from(anchor=0.5, n_ulp=case.payload_ulp), 0.5, 0.25),
        marginal=(_ulp_from(anchor=2.0, n_ulp=case.payload_ulp), 2.0, 3.0),
    )


def _query() -> jnp.ndarray:
    return jnp.asarray([0.0], dtype=_dtype())


def _fold_block(
    *,
    held: EnvelopeWinner,
    records: _Records,
    identities: tuple[int, ...],
    block: tuple[int, ...],
    arithmetic: ComparisonArithmetic,
) -> EnvelopeWinner:
    dtype = _dtype()
    # Distinct segment labels make every consecutive link dead, so a block's live
    # candidates are its self-brackets, each carrying its own global identity.
    link_ids = [identities[position] for position in block[:-1]]
    self_ids = [_SELF_BASE + identities[position] for position in block]
    return merge_envelope_winner(
        held=held,
        endog_grid=jnp.zeros(len(block), dtype=dtype),
        policy=jnp.asarray([records.policy[p] for p in block], dtype=dtype),
        value=jnp.asarray([records.value[p] for p in block], dtype=dtype),
        marginal=jnp.asarray([records.marginal[p] for p in block], dtype=dtype),
        segment_id=jnp.asarray([float(p) for p in block], dtype=dtype),
        stable_index=jnp.asarray(link_ids + self_ids, dtype=jnp.int32),
        query=_query(),
        arithmetic=arithmetic,
    )


def _fold_dead_block(
    *, held: EnvelopeWinner, arithmetic: ComparisonArithmetic
) -> EnvelopeWinner:
    dtype = _dtype()
    dead = jnp.full(2, jnp.nan, dtype=dtype)
    return merge_envelope_winner(
        held=held,
        endog_grid=dead,
        policy=dead,
        value=dead,
        marginal=dead,
        segment_id=jnp.asarray([90.0, 91.0], dtype=dtype),
        stable_index=jnp.asarray([8000, 8001, 8002], dtype=jnp.int32),
        query=_query(),
        arithmetic=arithmetic,
    )


def _fold(
    *,
    records: _Records,
    identities: tuple[int, ...],
    order: tuple[int, ...],
    block_size: int,
    dead_block_after: int | None,
    arithmetic: ComparisonArithmetic,
) -> EnvelopeWinner:
    held = empty_envelope_winner(query=_query())
    blocks = [
        order[start : start + block_size] for start in range(0, len(order), block_size)
    ]
    for index, block in enumerate(blocks):
        held = _fold_block(
            held=held,
            records=records,
            identities=identities,
            block=block,
            arithmetic=arithmetic,
        )
        if index == dead_block_after:
            held = _fold_dead_block(held=held, arithmetic=arithmetic)
    return held


def _layout_owner(*, case: _Case, arithmetic: ComparisonArithmetic) -> int:
    winner = _fold(
        records=_records(case),
        identities=case.identities,
        order=case.order,
        block_size=case.block_size,
        dead_block_after=case.dead_block_after,
        arithmetic=arithmetic,
    )
    assert bool(winner.live[0])
    assert bool(winner.settled[0])
    return int(np.asarray(winner.stable_index)[0])


def _one_shot_owner(*, case: _Case, arithmetic: ComparisonArithmetic) -> int:
    winner = _fold(
        records=_records(case),
        identities=case.identities,
        order=_CANONICAL,
        block_size=len(_CANONICAL),
        dead_block_after=None,
        arithmetic=arithmetic,
    )
    return int(np.asarray(winner.stable_index)[0])


def _skip_without_payload(arithmetic: ComparisonArithmetic) -> None:
    if arithmetic == "certified" and not kernel_built_for_current_backend():
        pytest.skip("the certified exact-affine payload is not built for this backend")


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_every_layout_publishes_the_one_shot_owner(
    *, case: _Case, arithmetic: ComparisonArithmetic
) -> None:
    """Storage order, block width, and dead blocks never move a query's owner."""
    _skip_without_payload(arithmetic)
    assert _layout_owner(case=case, arithmetic=arithmetic) == _one_shot_owner(
        case=case, arithmetic=arithmetic
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("case", _DECIDED, ids=_case_id)
def test_a_strictly_larger_stored_value_owns_the_query(
    *, case: _Case, arithmetic: ComparisonArithmetic
) -> None:
    """One unit in the last place of stored value decides ownership in every layout."""
    _skip_without_payload(arithmetic)
    assert _layout_owner(case=case, arithmetic=arithmetic) == (
        _SELF_BASE + case.identities[0]
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("case", _TIED, ids=_case_id)
def test_an_exact_tie_falls_to_the_smallest_stable_identity(
    *, case: _Case, arithmetic: ComparisonArithmetic
) -> None:
    """Identical stored values fall to the smaller identity wherever it is stored."""
    _skip_without_payload(arithmetic)
    assert _layout_owner(case=case, arithmetic=arithmetic) == _SELF_BASE + min(
        case.identities[:2]
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_the_published_payload_is_the_owners_stored_record(
    *, case: _Case, arithmetic: ComparisonArithmetic
) -> None:
    """Value, policy and marginal are read from the owning record exactly."""
    _skip_without_payload(arithmetic)
    records = _records(case)
    winner = _fold(
        records=records,
        identities=case.identities,
        order=case.order,
        block_size=case.block_size,
        dead_block_after=case.dead_block_after,
        arithmetic=arithmetic,
    )
    owner = case.identities.index(int(np.asarray(winner.stable_index)[0]) - _SELF_BASE)
    published = finish_envelope_winner(
        winner=winner, query=_query(), arithmetic=arithmetic
    )
    np.testing.assert_array_equal(
        [float(np.asarray(channel)[0]) for channel in published],
        [records.value[owner], records.policy[owner], records.marginal[owner]],
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
def test_the_owner_follows_the_identity_not_the_slot(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """Swapping two tied records' identities in place swaps the published policy."""
    _skip_without_payload(arithmetic)
    tied = _Case(0, 16, (5, 3, 9), _CANONICAL, 3, None)
    swapped = tied._replace(identities=(3, 5, 9))
    published = [
        float(
            np.asarray(
                finish_envelope_winner(
                    winner=_fold(
                        records=_records(case),
                        identities=case.identities,
                        order=case.order,
                        block_size=case.block_size,
                        dead_block_after=None,
                        arithmetic=arithmetic,
                    ),
                    query=_query(),
                    arithmetic=arithmetic,
                )[1]
            )[0]
        )
        for case in (tied, swapped)
    ]
    assert not np.array_equal(published[0], published[1])
