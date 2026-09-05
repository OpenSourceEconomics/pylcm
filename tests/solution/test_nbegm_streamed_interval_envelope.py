"""Agreement of the streamed NB-EGM interval read-and-fold with the one-shot merge.

The algebra and standalone reduction tests prove the winner monoid independently.
These maintainer-owned tests use the project runtime and the built exact-affine
payload to check that the production step publishes the same envelope for one-shot
and streamed interval layouts at every representative partition width.

`interval_batch_size` partitions a computation it does not change: the same
candidates are compared under the same total order, and the standing winner
re-enters each block under its global stored-link index. What a partition does
change is the vmap width each block is compiled for, so the two routes can name the
same real number with adjacent bit patterns. Ownership is therefore asserted
exactly — the global stored-link identity that owns each node, and with it the
published feasible set, both of which a mis-ordered fold moves by a finite amount —
and the published levels in units of the working format's spacing. The records the
two routes fold are captured at the production seams, so the one disagreement a
width can legitimately produce — two coincident candidates at a grid node, an exact
tie, split because one route produced a record a unit in the last place apart — is
recognised from the records themselves and never by a tolerance on the decision. The
normal precision fixture runs this module at float64 and with ``--precision=32``.
"""

from collections.abc import Callable
from functools import cache
from typing import Any, NamedTuple, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import nbegm_step
from _lcm.egm.nbegm_step import nbegm_per_interval_continuation_step_savings
from _lcm.egm.preferences import Preferences
from _lcm.egm.upper_envelope._exact_affine.ffi import (
    kernel_built_for_current_backend,
)
from _lcm.egm.upper_envelope.query import (
    NO_OWNER,
    ComparisonArithmetic,
    EnvelopeWinner,
)
from lcm.typing import Float1D, FloatND, IntND, ScalarFloat
from tests.conftest import assert_agrees_to_ulp
from tests.solution._crra_preferences import crra_preferences

_N_INTERVALS = 7
_N_SAVINGS = 36
_N_LIQUID = 31

_CHANNELS = ("value", "marginal", "policy")

# The spacing budget a change of compiled vmap width may spend. It is the constant
# this repository already uses for a partition knob, and it is two orders of
# magnitude below the smallest finite gap a mis-ordered fold would open.
_PARTITION_ULP = 16


class _Geometry(TypedDict):
    """The interval budget the step solves, apart from its continuation rows."""

    liquid_grid: Float1D
    savings_grid: Float1D
    discount_factor: ScalarFloat
    preferences: Preferences
    coh_slopes: Float1D
    coh_intercepts: Float1D
    breakpoints: Float1D


def _geometry() -> _Geometry:
    return {
        "liquid_grid": jnp.linspace(0.1, 32.0, _N_LIQUID),
        "savings_grid": jnp.linspace(0.0, 29.0, _N_SAVINGS),
        "discount_factor": jnp.asarray(0.96),
        "preferences": crra_preferences(crra=2.0),
        "coh_slopes": jnp.linspace(0.95, 1.35, _N_INTERVALS),
        "coh_intercepts": jnp.linspace(0.4, 2.1, _N_INTERVALS),
        "breakpoints": jnp.asarray([1.8, 5.2, 9.1, 14.7, 21.3, 27.4]),
    }


def _continuation() -> tuple[FloatND, FloatND]:
    """The continuation rows, asserted finite in the format that will be used."""
    shift = jnp.linspace(0.0, 0.9, _N_INTERVALS)[:, None]
    cont_value = -1.0 / jnp.linspace(0.45, 5.5, _N_SAVINGS)[None, :] + shift
    cont_marginal = jnp.linspace(2.4, 0.04, _N_SAVINGS)[None, :] + 0.07 * shift
    assert bool(jnp.isfinite(cont_value).all())
    assert bool(jnp.isfinite(cont_marginal).all())
    return cont_value, cont_marginal


def _one_shot(
    *, cont_value: FloatND, cont_marginal: FloatND, arithmetic: ComparisonArithmetic
) -> tuple[FloatND, ...]:
    return nbegm_per_interval_continuation_step_savings(
        **_geometry(),
        cont_value=cont_value,
        cont_marginal=cont_marginal,
        arithmetic=arithmetic,
        interval_batch_size=0,
        return_owner=True,
    )


def _streamed(
    *,
    cont_value: FloatND,
    cont_marginal: FloatND,
    arithmetic: ComparisonArithmetic,
    interval_batch_size: int,
) -> tuple[FloatND, ...]:

    def read(interval_indices: IntND) -> tuple[FloatND, FloatND]:
        return cont_value[interval_indices], cont_marginal[interval_indices]

    return nbegm_per_interval_continuation_step_savings(
        **_geometry(),
        cont_value=None,
        cont_marginal=None,
        arithmetic=arithmetic,
        interval_block_reader=read,
        interval_batch_size=interval_batch_size,
        return_owner=True,
    )


@cache
def _published(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> tuple[np.ndarray, ...]:
    """Publish the three channels and the owner once per (arithmetic, width)."""
    cont_value, cont_marginal = _continuation()
    if interval_batch_size == 0:
        solve: Callable[..., tuple[FloatND, ...]] = jax.jit(
            lambda value, marginal: _one_shot(
                cont_value=value, cont_marginal=marginal, arithmetic=arithmetic
            )
        )
    else:
        solve = jax.jit(
            lambda value, marginal: _streamed(
                cont_value=value,
                cont_marginal=marginal,
                arithmetic=arithmetic,
                interval_batch_size=interval_batch_size,
            )
        )
    return tuple(np.asarray(channel) for channel in solve(cont_value, cont_marginal))


def _skip_without_payload(arithmetic: ComparisonArithmetic) -> None:
    if arithmetic == "certified" and not kernel_built_for_current_backend():
        pytest.skip("the certified exact-affine payload is not built for this backend")


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [1, 2, 4, 7])
def test_streamed_step_publishes_the_one_shot_feasible_set(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> None:
    """Singleton, divisor, non-divisor, and full-width partitions own the same nodes."""
    _skip_without_payload(arithmetic)
    reference = _published(arithmetic=arithmetic, interval_batch_size=0)
    candidate = _published(
        arithmetic=arithmetic, interval_batch_size=interval_batch_size
    )

    np.testing.assert_array_equal(
        [np.isfinite(channel) for channel in candidate[:3]],
        [np.isfinite(channel) for channel in reference[:3]],
    )


class _Record(NamedTuple):
    """One candidate as a layout handed it to the envelope."""

    endog_grid: float
    value: float
    bits: tuple[bytes, ...]
    """The four numeric fields as stored, for bit-level comparison across layouts."""


_RECORD_FIELDS = ("endog_grid", "value", "policy", "marginal")


def _record(*, arrays: dict[str, np.ndarray], index: int) -> _Record:
    return _Record(
        endog_grid=float(arrays["endog_grid"][index]),
        value=float(arrays["value"][index]),
        bits=tuple(np.asarray(arrays[f][index]).tobytes() for f in _RECORD_FIELDS),
    )


class _Layouts(NamedTuple):
    """Both routes' published arrays and the candidate records each one folded."""

    one_shot: tuple[np.ndarray, ...]
    streamed: tuple[np.ndarray, ...]
    one_shot_records: dict[int, _Record]
    """Position in the one-shot stack → record, every candidate."""
    streamed_records: dict[int, _Record]
    """The same positions → record as the streamed blocks produced it; live only."""
    n_candidates: int


class _Captured(NamedTuple):
    """One route's published arrays and the records it handed to the envelope."""

    published: tuple[np.ndarray, ...]
    records: dict[int, _Record]
    """Position in the one-shot stack → record: every candidate on the one-shot
    route, the live candidates on the streamed route."""
    n_candidates: int


def _capturing(
    *, attribute: str, fields: tuple[str, ...], sink: list[dict[str, np.ndarray]]
) -> Callable[..., Any]:
    """Wrap a production seam so every call records its operands through a callback."""
    production = getattr(nbegm_step, attribute)

    def store(*arrays: np.ndarray) -> None:
        sink.append(dict(zip(fields, arrays, strict=True)))

    def seam(**kwargs: Any) -> Any:
        jax.debug.callback(store, *(kwargs[f] for f in fields), ordered=True)
        return production(**kwargs)

    return seam


@cache
def _one_shot_captured(*, arithmetic: ComparisonArithmetic) -> _Captured:
    """Publish the one-shot route, recording the stack `envelope_at_query` folded."""
    stacks: list[dict[str, np.ndarray]] = []
    cont_value, cont_marginal = _continuation()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            nbegm_step,
            "envelope_at_query",
            _capturing(
                attribute="envelope_at_query", fields=_RECORD_FIELDS, sink=stacks
            ),
        )
        published = tuple(
            np.asarray(channel)
            for channel in jax.jit(
                lambda value, marginal: _one_shot(
                    cont_value=value, cont_marginal=marginal, arithmetic=arithmetic
                )
            )(cont_value, cont_marginal)
        )
    (stack,) = stacks
    n_candidates = int(stack["endog_grid"].shape[0])
    return _Captured(
        published,
        {i: _record(arrays=stack, index=i) for i in range(n_candidates)},
        n_candidates,
    )


@cache
def _streamed_captured(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> _Captured:
    """Publish the streamed route, recording every block `merge_envelope_winner` folded.

    The records are captured through `jax.debug.callback` from the production
    seam, so they are exactly the operands the envelope compared — including any
    last-place difference the route's compiled program introduces while producing
    them.
    """
    blocks: list[dict[str, np.ndarray]] = []
    cont_value, cont_marginal = _continuation()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            nbegm_step,
            "merge_envelope_winner",
            _capturing(
                attribute="merge_envelope_winner",
                fields=(*_RECORD_FIELDS, "stable_index"),
                sink=blocks,
            ),
        )
        published = tuple(
            np.asarray(channel)
            for channel in jax.jit(
                lambda value, marginal: _streamed(
                    cont_value=value,
                    cont_marginal=marginal,
                    arithmetic=arithmetic,
                    interval_batch_size=interval_batch_size,
                )
            )(cont_value, cont_marginal)
        )
    n_candidates = _one_shot_captured(arithmetic=arithmetic).n_candidates
    records: dict[int, _Record] = {}
    for block in blocks:
        n_block = int(block["endog_grid"].shape[0])
        # A block names its self-brackets after every consecutive link, at the
        # positions of the equivalent one-shot layout.
        positions = block["stable_index"][n_block - 1 :] - (n_candidates - 1)
        for local, position in enumerate(positions.tolist()):
            if np.isnan(block["endog_grid"][local]):
                continue
            records[int(position)] = _record(arrays=block, index=local)
    return _Captured(published, records, n_candidates)


def _layouts(*, arithmetic: ComparisonArithmetic, interval_batch_size: int) -> _Layouts:
    one_shot = _one_shot_captured(arithmetic=arithmetic)
    streamed = _streamed_captured(
        arithmetic=arithmetic, interval_batch_size=interval_batch_size
    )
    return _Layouts(
        one_shot.published,
        streamed.published,
        one_shot.records,
        streamed.records,
        one_shot.n_candidates,
    )


def _owner_positions(*, owner: int, n_candidates: int) -> tuple[int, ...]:
    """The candidate positions a stored-link identity is made of."""
    if owner < n_candidates - 1:
        return (owner, owner + 1)
    return (owner - (n_candidates - 1),)


def _coincident_tie(*, layouts: _Layouts, node: int, liquid: float) -> bool:
    """Whether a node's two owners are the same point, split at the last place.

    Both owners must have a candidate sitting exactly at the node, those two
    candidates' stored values must be within one spacing of each other, and at
    least one of the owners' records must differ between the two layouts at the
    last place. The last condition is what separates the boundary from a defect:
    a fold handed bit-identical records that still names a different owner has
    decided by something other than the records and identities.
    """
    owners = (int(layouts.one_shot[3][node]), int(layouts.streamed[3][node]))
    at_node: list[_Record] = []
    records_moved = False
    for owner in owners:
        positions = _owner_positions(owner=owner, n_candidates=layouts.n_candidates)
        here = [
            layouts.one_shot_records[p]
            for p in positions
            if layouts.one_shot_records[p].endog_grid == liquid
        ]
        if not here:
            return False
        at_node.append(here[0])
        records_moved |= any(
            p in layouts.streamed_records
            and layouts.streamed_records[p].bits != layouts.one_shot_records[p].bits
            for p in positions
        )
    spacing = float(
        np.spacing(np.asarray(max(abs(r.value) for r in at_node), dtype=_dtype()))
    )
    return abs(at_node[0].value - at_node[1].value) <= spacing and records_moved


def _dtype() -> np.dtype:
    return np.asarray(_geometry()["liquid_grid"]).dtype


def _assert_owners_agree(
    *, layouts: _Layouts, streamed_owner: np.ndarray | None = None
) -> None:
    """Every node has the one-shot owner or a coincident tie split at the last place."""
    reference = layouts.one_shot[3]
    candidate = layouts.streamed[3] if streamed_owner is None else streamed_owner
    liquid = np.asarray(_geometry()["liquid_grid"])
    unexplained = [
        int(node)
        for node in np.flatnonzero(candidate != reference)
        if not _coincident_tie(
            layouts=layouts, node=int(node), liquid=float(liquid[node])
        )
    ]
    assert not unexplained, (
        f"nodes {unexplained} are owned differently and are not coincident ties: "
        f"one-shot {reference[unexplained].tolist()}, streamed "
        f"{candidate[unexplained].tolist()}"
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [1, 2, 4, 7])
def test_streamed_step_publishes_the_one_shot_owner_at_every_node(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> None:
    """The same global stored-link identity owns each node at every partition.

    The one exception a partition may produce is not a partition effect: where a
    savings-node point candidate coincides with an interior candidate at a grid
    node, the two are an exact tie, and the routes' compiled programs can produce
    one of the records a unit in the last place apart, so the same exact order
    picks the other member of the tie. Such a node is accepted only when the
    captured records show exactly that; any other disagreement fails.
    """
    _skip_without_payload(arithmetic)
    _assert_owners_agree(
        layouts=_layouts(arithmetic=arithmetic, interval_batch_size=interval_batch_size)
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [1, 2, 4, 7])
def test_streamed_records_are_the_one_shot_records_at_every_live_position(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> None:
    """Every live streamed candidate sits at the one-shot position of the same point.

    The identity a block assigns is the position the same candidate holds in the
    one-shot stack, so the abscissa stored under it must be that candidate's — to
    the spacing a compiled width may spend, since the abscissa is itself a level.
    """
    _skip_without_payload(arithmetic)
    layouts = _layouts(arithmetic=arithmetic, interval_batch_size=interval_batch_size)
    positions = sorted(layouts.streamed_records)
    assert positions
    assert_agrees_to_ulp(
        got=np.asarray([layouts.streamed_records[p].endog_grid for p in positions]),
        expected=np.asarray(
            [layouts.one_shot_records[p].endog_grid for p in positions]
        ),
        n_ulp=_PARTITION_ULP,
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
def test_the_one_shot_owner_names_several_candidates(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """The owner comparison ranges over distinct identities, not a constant."""
    _skip_without_payload(arithmetic)
    owner = _published(arithmetic=arithmetic, interval_batch_size=0)[3]
    assert np.unique(owner[owner != NO_OWNER]).size > 1


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [0, 1, 2, 4, 7])
def test_an_owner_is_published_exactly_where_a_level_is(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> None:
    """A node carries an owner if and only if it carries a finite value."""
    _skip_without_payload(arithmetic)
    value, _, _, owner = _published(
        arithmetic=arithmetic, interval_batch_size=interval_batch_size
    )
    np.testing.assert_array_equal(owner != NO_OWNER, np.isfinite(value))


@pytest.mark.parametrize("channel", range(len(_CHANNELS)), ids=_CHANNELS)
@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [1, 2, 4, 7])
def test_streamed_step_agrees_with_the_one_shot_merge_to_the_format_spacing(
    *,
    arithmetic: ComparisonArithmetic,
    interval_batch_size: int,
    channel: int,
) -> None:
    """Every partition names the one-shot level in each published channel."""
    _skip_without_payload(arithmetic)
    assert_agrees_to_ulp(
        got=_published(arithmetic=arithmetic, interval_batch_size=interval_batch_size)[
            channel
        ],
        expected=_published(arithmetic=arithmetic, interval_batch_size=0)[channel],
        n_ulp=_PARTITION_ULP,
        err_msg=(
            f"channel={_CHANNELS[channel]}, arithmetic={arithmetic}, "
            f"interval_batch_size={interval_batch_size}"
        ),
    )


@cache
def _relabelled_streamed(*, interval_batch_size: int) -> tuple[np.ndarray, ...]:
    """Stream with a fold that names every candidate by its block-local slot.

    The relabelled fold still compares the same records under the same order, so
    where no two candidates tie it selects the same record and publishes the same
    levels; only the identity it reports is wrong. It is the defect class the owner
    assertion exists for, and it is invisible to the level assertions.
    """
    production = nbegm_step.merge_envelope_winner

    def relabelled(*, stable_index: IntND, **kwargs: Any) -> EnvelopeWinner:
        return production(
            stable_index=jnp.arange(stable_index.shape[0], dtype=jnp.int32), **kwargs
        )

    cont_value, cont_marginal = _continuation()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nbegm_step, "merge_envelope_winner", relabelled)
        solve = jax.jit(
            lambda value, marginal: _streamed(
                cont_value=value,
                cont_marginal=marginal,
                arithmetic="ordinary",
                interval_batch_size=interval_batch_size,
            )
        )
        return tuple(
            np.asarray(channel) for channel in solve(cont_value, cont_marginal)
        )


def test_the_owner_assertion_rejects_a_fold_that_relabels_identities() -> None:
    """The instrument fires on the defect it guards against, in this run."""
    relabelled = _relabelled_streamed(interval_batch_size=2)[3]
    layouts = _layouts(arithmetic="ordinary", interval_batch_size=2)
    with pytest.raises(AssertionError, match="not coincident ties"):
        _assert_owners_agree(layouts=layouts, streamed_owner=relabelled)


@pytest.mark.parametrize("channel", range(len(_CHANNELS)), ids=_CHANNELS)
def test_the_level_assertions_do_not_see_a_fold_that_relabels_identities(
    *, channel: int
) -> None:
    """Relabelled identities leave every level within the spacing budget."""
    assert_agrees_to_ulp(
        got=_relabelled_streamed(interval_batch_size=2)[channel],
        expected=_published(arithmetic="ordinary", interval_batch_size=0)[channel],
        n_ulp=_PARTITION_ULP,
    )


def test_the_agreement_bound_rejects_a_fold_that_publishes_another_owner() -> None:
    """The instrument fires on the defect it guards against, in this run."""
    reference = _published(arithmetic="ordinary", interval_batch_size=0)[0]
    # A fold that hands a query to the wrong candidate moves its level by a finite
    # amount. The smallest gap between two distinct published levels stands in for
    # that, and it is orders of magnitude above the spacing budget above.
    distinct = np.unique(reference[np.isfinite(reference)])
    assert distinct.size > 1
    smallest_ownership_gap = float(np.min(np.diff(distinct)))
    mis_owned = np.where(
        np.isfinite(reference), reference + smallest_ownership_gap, reference
    )

    with pytest.raises(AssertionError, match="ULP, above the"):
        assert_agrees_to_ulp(got=mis_owned, expected=reference, n_ulp=_PARTITION_ULP)


def test_streaming_reader_is_called_with_fixed_width_blocks() -> None:
    """The direct production seam asks for blocks, never the full interval matrix."""
    value, marginal = _continuation()
    seen_shapes: list[tuple[int, ...]] = []

    def read(indices: IntND) -> tuple[FloatND, FloatND]:
        seen_shapes.append(indices.shape)
        return value[indices], marginal[indices]

    with jax.disable_jit():
        nbegm_per_interval_continuation_step_savings(
            **_geometry(),
            cont_value=None,
            cont_marginal=None,
            arithmetic="ordinary",
            interval_block_reader=read,
            interval_batch_size=2,
        )
    assert seen_shapes
    assert set(seen_shapes) == {(2,)}
