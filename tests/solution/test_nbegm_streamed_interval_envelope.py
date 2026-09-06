"""Agreement of the streamed NB-EGM interval read-and-fold with the one-shot merge.

The algebra and standalone reduction tests prove the winner monoid independently.
These maintainer-owned tests use the project runtime and the built exact-affine
payload to check that the production step publishes the same envelope for one-shot
and streamed interval layouts at every representative partition width.

`interval_batch_size` partitions a computation it does not change: the same
candidates are compared under the same total order, and the standing winner
re-enters each block under its global stored-link index. What a partition does
change is the vmap width each block is compiled for, so the two routes can name the
same real number with adjacent bit patterns. Ownership is therefore certified on
each route by itself before the routes are compared. Every route runs once,
instrumented so that the candidate records, the query, and — under ordinary
arithmetic — the rank fields it compared are the very buffers its fold consumed
(an optimization barrier stops the backend from producing one copy for the
observer and another for the fold). The published owner must then be the maximum
of that admitted set under the documented total order, and the published channels
must be read from the owner's own record. Only afterwards are the routes compared:
identical admitted records must give identical owners, and where a width has
produced a record a unit in the last place apart, both owners are already
certified on the records each route folded, so no tolerance ever touches the
decision. The published levels are compared in units of the working format's
spacing. The normal precision fixture runs this module at float64 and with
``--precision=32``.
"""

from collections.abc import Callable, Iterator
from fractions import Fraction
from functools import cache
from typing import Any, NamedTuple, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import nbegm_step
from _lcm.egm.nbegm_step import nbegm_per_interval_continuation_step_savings
from _lcm.egm.preferences import Preferences
from _lcm.egm.upper_envelope import query as envelope_query
from _lcm.egm.upper_envelope._exact_affine.ffi import (
    exact_affine_read,
    kernel_built_for_current_backend,
)
from _lcm.egm.upper_envelope.query import (
    NO_OWNER,
    ComparisonArithmetic,
    EnvelopeWinner,
    empty_envelope_winner,
    finish_envelope_winner,
    merge_envelope_winner,
)
from lcm.typing import Float1D, FloatND, IntND, ScalarFloat
from tests.conftest import assert_agrees_to_ulp
from tests.solution._crra_preferences import crra_preferences

_N_INTERVALS = 7
_N_SAVINGS = 36
_N_LIQUID = 31

_CHANNELS = ("value", "marginal", "policy")
_WIDTHS = (1, 2, 4, 7)

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


def _solver(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> Callable[..., tuple[FloatND, ...]]:
    """The jitted step for one route: the one-shot at width 0, else streamed."""
    if interval_batch_size == 0:
        return jax.jit(
            lambda value, marginal: _one_shot(
                cont_value=value, cont_marginal=marginal, arithmetic=arithmetic
            )
        )
    return jax.jit(
        lambda value, marginal: _streamed(
            cont_value=value,
            cont_marginal=marginal,
            arithmetic=arithmetic,
            interval_batch_size=interval_batch_size,
        )
    )


@cache
def _published(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> tuple[np.ndarray, ...]:
    """Publish the three channels and the owner once per (arithmetic, width)."""
    cont_value, cont_marginal = _continuation()
    solve = _solver(arithmetic=arithmetic, interval_batch_size=interval_batch_size)
    return tuple(np.asarray(channel) for channel in solve(cont_value, cont_marginal))


def _skip_without_payload(arithmetic: ComparisonArithmetic) -> None:
    if arithmetic == "certified" and not kernel_built_for_current_backend():
        pytest.skip("the certified exact-affine payload is not built for this backend")


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", _WIDTHS)
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


_RECORD_FIELDS = ("endog_grid", "value", "policy", "marginal")
_STACK_FIELDS = (*_RECORD_FIELDS, "segment_id")
_HELD_FIELDS = tuple(f"held_{name}" for name in EnvelopeWinner._fields)
_PADDING_IDENTITY = int(np.iinfo(np.int32).max)


class _Trace:
    """What one instrumented execution handed its envelope, in program order."""

    def __init__(self) -> None:
        self.stacks: list[dict[str, np.ndarray]] = []
        """The one-shot candidate stack with its query: one entry."""
        self.blocks: list[dict[str, np.ndarray]] = []
        """Every streamed block: candidates, identities, query, standing winner."""
        self.link_blocks: list[dict[str, np.ndarray]] = []
        """The one-shot ordinary reduction's link partition: one entry."""
        self.block_terms: list[dict[str, np.ndarray]] = []
        """The one-shot ordinary rank fields, one entry per link block."""
        self.link_terms: list[dict[str, np.ndarray]] = []
        """The streamed ordinary rank fields, one entry per streamed block."""


def _bound(
    *, sink: list[dict[str, np.ndarray]], arrays: dict[str, Any]
) -> dict[str, Any]:
    """Materialize `arrays` once, report those buffers to `sink`, and return them.

    The optimization barrier stops the backend from producing one copy of an
    operand for the reporting callback and another for the production consumer:
    everything downstream of the seam reads the buffers the callback reported.
    """
    names = tuple(arrays)
    values = jax.lax.optimization_barrier(tuple(arrays[name] for name in names))
    bound = dict(zip(names, values, strict=True))

    def store(*reported: Any) -> None:
        sink.append(
            {name: np.asarray(r) for name, r in zip(names, reported, strict=True)}
        )

    jax.debug.callback(store, *(bound[name] for name in names), ordered=True)
    return bound


def _instrument(
    *,
    trace: _Trace,
    patch: pytest.MonkeyPatch,
    fold: Callable[..., EnvelopeWinner] | None,
) -> None:
    """Route the production seams through `trace`, folding with `fold` if given."""
    production_query = nbegm_step.envelope_at_query
    production_merge = nbegm_step.merge_envelope_winner if fold is None else fold
    production_link_blocks = envelope_query._link_blocks
    production_block_terms = envelope_query._block_query_terms
    production_link_terms = envelope_query._batched_link_terms

    def at_query(**kwargs: Any) -> Any:
        names = (*_STACK_FIELDS, "x_query")
        bound = _bound(sink=trace.stacks, arrays={n: kwargs[n] for n in names})
        return production_query(**{**kwargs, **bound})

    def merge(*, held: EnvelopeWinner, **kwargs: Any) -> EnvelopeWinner:
        names = (*_STACK_FIELDS, "stable_index", "query")
        arrays = {n: kwargs[n] for n in names}
        arrays.update(dict(zip(_HELD_FIELDS, held, strict=True)))
        bound = _bound(sink=trace.blocks, arrays=arrays)
        return production_merge(
            held=EnvelopeWinner(*(bound[n] for n in _HELD_FIELDS)),
            **{**kwargs, **{n: bound[n] for n in names}},
        )

    def link_blocks(**kwargs: Any) -> tuple[Any, ...]:
        blocks, live, stable_index = production_link_blocks(**kwargs)
        bound = _bound(
            sink=trace.link_blocks, arrays={"live": live, "stable_index": stable_index}
        )
        return blocks, bound["live"], bound["stable_index"]

    def block_terms(**kwargs: Any) -> Any:
        terms = production_block_terms(**kwargs)
        bound = _bound(
            sink=trace.block_terms, arrays=dict(zip(terms._fields, terms, strict=True))
        )
        return type(terms)(*(bound[n] for n in terms._fields))

    def link_terms(**kwargs: Any) -> tuple[Any, Any]:
        rank, brackets = production_link_terms(**kwargs)
        arrays = dict(zip(rank._fields, rank, strict=True)) | {"brackets": brackets}
        bound = _bound(sink=trace.link_terms, arrays=arrays)
        return type(rank)(*(bound[n] for n in rank._fields)), bound["brackets"]

    patch.setattr(nbegm_step, "envelope_at_query", at_query)
    patch.setattr(nbegm_step, "merge_envelope_winner", merge)
    patch.setattr(envelope_query, "_link_blocks", link_blocks)
    patch.setattr(envelope_query, "_block_query_terms", block_terms)
    patch.setattr(envelope_query, "_batched_link_terms", link_terms)


class _Candidate(NamedTuple):
    """One candidate as a route's fold consumed it."""

    endog_grid: float
    value: float
    policy: float
    marginal: float
    segment_id: float
    bits: tuple[bytes, ...]
    """The four numeric fields as stored, for bit-level comparison across layouts."""


class _Link(NamedTuple):
    """One admitted link: its stable identity and the candidate positions it joins."""

    identity: int
    left: int
    right: int
    """Equal to `left` for a self-bracket."""


class _Keys(NamedTuple):
    """The ordinary rank fields one link carried against every query, as compared."""

    brackets: np.ndarray
    value: np.ndarray
    right_available: np.ndarray
    slope_high: np.ndarray
    slope_low: np.ndarray


class _Route(NamedTuple):
    """One instrumented execution: what its envelope folded and what it published."""

    arithmetic: ComparisonArithmetic
    query: np.ndarray
    candidates: dict[int, _Candidate]
    """Position in the one-shot stack → live candidate as the fold consumed it."""
    links: dict[int, _Link]
    """Stable identity → admitted link (both endpoints live, one branch label)."""
    n_candidates: int
    published: tuple[np.ndarray, ...]
    """Value, marginal, policy, owner."""
    keys: dict[int, _Keys] | None
    """Ordinary arithmetic: identity → rank fields as compared; else `None`."""
    notes: tuple[str, ...]
    """Binding failures: one identity naming two records or two rank fields, a
    standing winner re-entering with a record that is not its own."""


def _candidate(*, arrays: dict[str, np.ndarray], index: int) -> _Candidate:
    return _Candidate(
        *(float(arrays[f][index]) for f in _STACK_FIELDS),
        bits=tuple(np.asarray(arrays[f][index]).tobytes() for f in _RECORD_FIELDS),
    )


def _is_live(*, arrays: dict[str, np.ndarray], index: int) -> bool:
    return bool(
        np.isfinite(arrays["endog_grid"][index]) and np.isfinite(arrays["value"][index])
    )


def _admit(
    *,
    arrays: dict[str, np.ndarray],
    positions: np.ndarray,
    identities: np.ndarray,
    candidates: dict[int, _Candidate],
    links: dict[int, _Link],
    notes: list[str],
) -> None:
    """Admit one stack's live candidates and the links the stack forms.

    A consecutive pair is a link where both candidates are live and share a
    branch label; every live candidate is also a zero-width self-bracket. This
    is the documented admission rule, re-derived here from the stored fields.
    """
    n_stack = int(positions.shape[0])
    live = [_is_live(arrays=arrays, index=j) for j in range(n_stack)]

    def add(link: _Link) -> None:
        prior = links.get(link.identity)
        if prior is not None and prior != link:
            notes.append(
                f"identity {link.identity} names two links: {prior} and {link}"
            )
        links.setdefault(link.identity, link)

    for j in range(n_stack):
        if not live[j]:
            continue
        candidate = _candidate(arrays=arrays, index=j)
        position = int(positions[j])
        prior = candidates.get(position)
        if prior is not None and prior.bits != candidate.bits:
            notes.append(f"position {position} was produced twice, differently")
        candidates.setdefault(position, candidate)
        add(_Link(int(identities[n_stack - 1 + j]), position, position))
    for j in range(n_stack - 1):
        if (
            live[j]
            and live[j + 1]
            and arrays["segment_id"][j] == arrays["segment_id"][j + 1]
        ):
            add(_Link(int(identities[j]), int(positions[j]), int(positions[j + 1])))


def _empty_keys(*, n_query: int, dtype: np.dtype) -> _Keys:
    return _Keys(
        np.zeros(n_query, dtype=bool),
        *(np.full(n_query, np.nan, dtype=dtype) for _ in range(4)),
    )


def _set_keys(
    *,
    keys: dict[int, _Keys],
    identity: int,
    rows: np.ndarray,
    fields: tuple[np.ndarray, ...],
    n_query: int,
    notes: list[str],
) -> None:
    """Record the rank fields `identity` carried at the bracketing `rows`."""
    held = keys.get(identity)
    if held is None:
        held = _empty_keys(n_query=n_query, dtype=fields[0].dtype)
        keys[identity] = held
    for row in np.flatnonzero(rows).tolist():
        incoming = tuple(np.asarray(f[row]).tobytes() for f in fields)
        if held.brackets[row]:
            present = tuple(np.asarray(f[row]).tobytes() for f in held[1:])
            if present != incoming:
                notes.append(f"identity {identity} carried two rank fields at {row}")
            continue
        held.brackets[row] = True
        for field, value in zip(held[1:], fields, strict=True):
            field[row] = value[row]


def _one_shot_route(
    *,
    trace: _Trace,
    published: tuple[np.ndarray, ...],
    arithmetic: ComparisonArithmetic,
) -> _Route:
    (stack,) = trace.stacks
    n_candidates = int(stack["endog_grid"].shape[0])
    query = stack["x_query"]
    n_query = int(query.shape[0])
    candidates: dict[int, _Candidate] = {}
    links: dict[int, _Link] = {}
    notes: list[str] = []
    _admit(
        arrays=stack,
        positions=np.arange(n_candidates),
        identities=np.arange(2 * n_candidates - 1),
        candidates=candidates,
        links=links,
        notes=notes,
    )
    keys: dict[int, _Keys] | None = None
    if arithmetic == "ordinary":
        (partition,) = trace.link_blocks
        keys = {}
        for block, terms in zip(
            partition["stable_index"], trace.block_terms, strict=True
        ):
            right_available = (query[:, None] < terms["upper"]).astype(
                terms["value"].dtype
            )
            for column, identity in enumerate(block.tolist()):
                if identity == _PADDING_IDENTITY:
                    continue
                _set_keys(
                    keys=keys,
                    identity=identity,
                    rows=terms["brackets"][:, column],
                    fields=(
                        terms["value"][:, column],
                        right_available[:, column],
                        np.broadcast_to(terms["slope_high"][:, column], (n_query,)),
                        np.broadcast_to(terms["slope_low"][:, column], (n_query,)),
                    ),
                    n_query=n_query,
                    notes=notes,
                )
    return _Route(
        arithmetic,
        query,
        candidates,
        links,
        n_candidates,
        published,
        keys,
        tuple(notes),
    )


def _check_re_entry(
    *,
    block: dict[str, np.ndarray],
    index: int,
    candidates: dict[int, _Candidate],
    links: dict[int, _Link],
    notes: list[str],
) -> None:
    """A standing winner re-enters with the record its identity was admitted as."""
    stored = (
        "left_grid",
        "left_value",
        "left_policy",
        "left_marginal",
        "right_grid",
        "right_value",
        "right_policy",
        "right_marginal",
    )
    for row in np.flatnonzero(block["held_live"]).tolist():
        identity = int(block["held_stable_index"][row])
        link = links.get(identity)
        if link is None:
            notes.append(f"winner {identity} re-entered block {index} unadmitted")
            continue
        expected = candidates[link.left].bits + candidates[link.right].bits
        carried = tuple(np.asarray(block[f"held_{f}"][row]).tobytes() for f in stored)
        if carried != expected:
            notes.append(
                f"winner {identity} re-entered block {index} at query {row} "
                "with a record that is not its own"
            )


def _streamed_route(
    *,
    trace: _Trace,
    published: tuple[np.ndarray, ...],
    arithmetic: ComparisonArithmetic,
    n_candidates: int,
) -> _Route:
    query = trace.blocks[0]["query"]
    n_query = int(query.shape[0])
    candidates: dict[int, _Candidate] = {}
    links: dict[int, _Link] = {}
    notes: list[str] = []
    keys: dict[int, _Keys] | None = {} if arithmetic == "ordinary" else None
    for index, block in enumerate(trace.blocks):
        if not np.array_equal(block["query"], query):
            notes.append(f"block {index} folded a different query")
        n_block = int(block["endog_grid"].shape[0])
        identities = block["stable_index"]
        # A block names its self-brackets after every consecutive link, at the
        # positions of the equivalent one-shot layout.
        positions = identities[n_block - 1 :] - (n_candidates - 1)
        _check_re_entry(
            block=block, index=index, candidates=candidates, links=links, notes=notes
        )
        _admit(
            arrays=block,
            positions=positions,
            identities=identities,
            candidates=candidates,
            links=links,
            notes=notes,
        )
        if keys is None:
            continue
        terms = trace.link_terms[index]
        # Column 0 is the standing winner, whose identity is per query; the block's
        # own links follow in storage order.
        for row in np.flatnonzero(terms["brackets"][:, 0]).tolist():
            _set_keys(
                keys=keys,
                identity=int(block["held_stable_index"][row]),
                rows=np.arange(n_query) == row,
                fields=tuple(terms[f][:, 0] for f in _Keys._fields[1:]),
                n_query=n_query,
                notes=notes,
            )
        for column, identity in enumerate(identities.tolist(), start=1):
            _set_keys(
                keys=keys,
                identity=identity,
                rows=terms["brackets"][:, column],
                fields=tuple(terms[f][:, column] for f in _Keys._fields[1:]),
                n_query=n_query,
                notes=notes,
            )
    return _Route(
        arithmetic,
        query,
        candidates,
        links,
        n_candidates,
        published,
        keys,
        tuple(notes),
    )


def _relabelled_fold(**kwargs: Any) -> EnvelopeWinner:
    """A fold that names every candidate by its block-local slot.

    It compares the same records under the same order, so away from ties it
    selects the same record and publishes the same levels; only the identity it
    reports is wrong. It is invisible to the level assertions.
    """
    kwargs["stable_index"] = jnp.arange(
        kwargs["stable_index"].shape[0], dtype=jnp.int32
    )
    return merge_envelope_winner(**kwargs)


def _inferior_fold(**kwargs: Any) -> EnvelopeWinner:
    """A fold that hands a query to the lowest-reading link that brackets it.

    The record it selects is internally consistent — its own endpoints, payloads
    and identity go through the unmodified finisher — so wherever two links
    bracket a query it publishes an inferior but plausible owner.
    """
    winner = merge_envelope_winner(**kwargs)
    links, _ = envelope_query._segment_links_from_candidates(
        **{f: kwargs[f] for f in (*_STACK_FIELDS, "stable_index")},
        feasibility_partition=None,
        feasible_interval_mask=None,
    )
    q = jnp.asarray(kwargs["query"]).reshape(-1, 1)
    lower = jnp.minimum(links.left_grid, links.right_grid)[None, :]
    upper = jnp.maximum(links.left_grid, links.right_grid)[None, :]
    brackets = links.live[None, :] & (q >= lower) & (q <= upper)
    read = envelope_query._along_link(
        left=links.left_value[None, :],
        right=links.right_value[None, :],
        query=q,
        left_grid=links.left_grid[None, :],
        right_grid=links.right_grid[None, :],
        arithmetic="ordinary",
    )
    lowest = jnp.argmin(jnp.where(brackets, read, jnp.inf), axis=1)
    bracketed = jnp.any(brackets, axis=1)
    return EnvelopeWinner(
        *(
            jnp.where(bracketed, column[lowest], held)
            for column, held in zip(links[:8], winner[:8], strict=True)
        ),
        live=winner.live | bracketed,
        stable_index=jnp.where(
            bracketed, links.stable_index[lowest], winner.stable_index
        ),
        settled=winner.settled,
        unreadable=winner.unreadable,
    )


_FOLDS: dict[str, Callable[..., EnvelopeWinner] | None] = {
    "production": None,
    "relabelled": _relabelled_fold,
    "inferior": _inferior_fold,
}


@cache
def _route(
    *,
    arithmetic: ComparisonArithmetic,
    interval_batch_size: int,
    fold: str = "production",
) -> _Route:
    """Run one route instrumented and bind what it folded to what it published."""
    trace = _Trace()
    cont_value, cont_marginal = _continuation()
    with pytest.MonkeyPatch.context() as patch:
        _instrument(trace=trace, patch=patch, fold=_FOLDS[fold])
        solve = _solver(arithmetic=arithmetic, interval_batch_size=interval_batch_size)
        published = tuple(np.asarray(c) for c in solve(cont_value, cont_marginal))
    if interval_batch_size == 0:
        return _one_shot_route(trace=trace, published=published, arithmetic=arithmetic)
    return _streamed_route(
        trace=trace,
        published=published,
        arithmetic=arithmetic,
        n_candidates=_route(arithmetic=arithmetic, interval_batch_size=0).n_candidates,
    )


def _endpoints(*, route: _Route, link: _Link) -> tuple[_Candidate, _Candidate]:
    """The link's candidates in ascending order of abscissa."""
    left, right = route.candidates[link.left], route.candidates[link.right]
    return (right, left) if left.endog_grid > right.endog_grid else (left, right)


def _bracketing(*, route: _Route, node: int) -> Iterator[tuple[_Link, Fraction]]:
    """Every admitted link that brackets the node's query, with that query."""
    q = Fraction(float(route.query[node]))
    for link in route.links.values():
        start, stop = _endpoints(route=route, link=link)
        if Fraction(start.endog_grid) <= q <= Fraction(stop.endog_grid):
            yield link, q


def _exact_owner(*, route: _Route, node: int) -> int:
    """The literal total order the certified comparison is documented to apply.

    Exact affine value at the query, then whether the link extends strictly to
    the right, then the exact slope, then the smaller stable identity; a
    zero-width link reads its stored value and has no slope.
    """
    best: tuple[tuple[Fraction, bool, Fraction, int], int] | None = None
    for link, q in _bracketing(route=route, node=node):
        start, stop = _endpoints(route=route, link=link)
        x0, x1 = Fraction(start.endog_grid), Fraction(stop.endog_grid)
        v0, v1 = Fraction(start.value), Fraction(stop.value)
        if x0 == x1:
            key = (v0, False, Fraction(0), -link.identity)
        else:
            slope = (v1 - v0) / (x1 - x0)
            key = (v0 + (q - x0) * slope, q < x1, slope, -link.identity)
        if best is None or key > best[0]:
            best = (key, link.identity)
    return NO_OWNER if best is None else best[1]


def _ordinary_owner(*, route: _Route, node: int) -> int:
    """The lexicographic maximum of the rank fields the ordinary fold compared."""
    assert route.keys is not None
    best: tuple[tuple[Any, ...], int] | None = None
    for identity, keys in route.keys.items():
        if not keys.brackets[node]:
            continue
        key = (*(field[node] for field in keys[1:]), -identity)
        if best is None or key > best[0]:
            best = (key, identity)
    return NO_OWNER if best is None else best[1]


def _expected_owners(route: _Route) -> np.ndarray:
    select = _exact_owner if route.arithmetic == "certified" else _ordinary_owner
    return np.asarray(
        [select(route=route, node=node) for node in range(route.query.shape[0])],
        dtype=np.int32,
    )


def _certify_selection(route: _Route) -> None:
    """The published owner is the maximum of the records the route itself folded."""
    assert not route.notes, route.notes
    np.testing.assert_array_equal(
        route.published[3],
        _expected_owners(route),
        err_msg=(
            f"{route.arithmetic}: the published owner is not the maximum of the "
            "admitted records under the documented order"
        ),
    )


def _certify_readout(route: _Route) -> None:
    """The published channels are read from the published owner's own record."""
    owner = route.published[3]
    nodes = np.flatnonzero(owner != NO_OWNER).tolist()
    missing = [int(owner[n]) for n in nodes if int(owner[n]) not in route.links]
    assert not missing, f"published owners {missing} name no admitted record"
    dtype = route.query.dtype
    ends = [_endpoints(route=route, link=route.links[int(owner[n])]) for n in nodes]
    x0 = np.asarray([start.endog_grid for start, _ in ends], dtype=dtype)
    x1 = np.asarray([stop.endog_grid for _, stop in ends], dtype=dtype)
    q = route.query[nodes]
    zero_width = x0 == x1
    for field in ("value", "policy", "marginal"):
        v0 = np.asarray([getattr(start, field) for start, _ in ends], dtype=dtype)
        v1 = np.asarray([getattr(stop, field) for _, stop in ends], dtype=dtype)
        v1 = np.where(zero_width, v0, v1)
        published = route.published[_CHANNELS.index(field)][nodes]
        if route.arithmetic == "certified":
            # The exact reader keeps its positive-width contract: a selected point
            # is read as a flat unit line at zero, as the production reader does.
            read, status = exact_affine_read(
                x0=jnp.where(zero_width, 0.0, x0).astype(dtype),
                x1=jnp.where(zero_width, 1.0, x1).astype(dtype),
                v0=jnp.asarray(v0),
                v1=jnp.asarray(v1),
                x_query=jnp.where(zero_width, 0.0, q).astype(dtype),
            )
            np.testing.assert_array_equal(np.asarray(status), 0)
            np.testing.assert_array_equal(
                published, np.asarray(read), err_msg=f"{field} not read from the owner"
            )
        else:
            read = envelope_query._along_link(
                left=jnp.asarray(v0),
                right=jnp.asarray(v1),
                query=jnp.asarray(q),
                left_grid=jnp.asarray(x0),
                right_grid=jnp.asarray(x1),
                arithmetic="ordinary",
            )
            assert_agrees_to_ulp(
                got=published,
                expected=np.asarray(read),
                n_ulp=_PARTITION_ULP,
                err_msg=f"{field} not read from the owner",
            )


def _certify_route(route: _Route) -> None:
    _certify_selection(route)
    _certify_readout(route)


class _Difference(NamedTuple):
    """A node whose bracketing records the two routes produced differently."""

    node: int
    identities: tuple[int, ...]
    """Bracketing links whose records or rank fields differ between the routes."""
    owners: tuple[int, int]
    """One-shot owner, streamed owner."""
    query_differs: bool
    """Whether the two routes folded different bit patterns of the query itself."""


def _differing_identities(
    *, one_shot: _Route, streamed: _Route, node: int
) -> tuple[int, ...]:
    bracketing = {link.identity for link, _ in _bracketing(route=one_shot, node=node)}
    bracketing |= {link.identity for link, _ in _bracketing(route=streamed, node=node)}
    differing = []
    for identity in sorted(bracketing):
        first = one_shot.links[identity]
        records = [
            (one_shot.candidates[p].bits, streamed.candidates[p].bits)
            for p in (first.left, first.right)
        ]
        moved = any(a != b for a, b in records)
        if one_shot.keys is not None and streamed.keys is not None:
            a, b = one_shot.keys.get(identity), streamed.keys.get(identity)
            moved |= (a is None) != (b is None) or (
                a is not None
                and b is not None
                and any(
                    np.asarray(x[node]).tobytes() != np.asarray(y[node]).tobytes()
                    for x, y in zip(a, b, strict=True)
                )
            )
        if moved:
            differing.append(identity)
    return tuple(differing)


def _compare_routes(*, one_shot: _Route, streamed: _Route) -> tuple[_Difference, ...]:
    """Identical admitted records give identical owners; different ones are reported.

    Both routes must admit the same identities joining the same candidate
    positions. At a node where every bracketing record — and, under ordinary
    arithmetic, every rank field — is bit-identical between the routes, the
    owners must be identical. Where a record differs, each route's owner is
    already certified against its own records, and the node is reported with
    the identities that moved.
    """
    assert set(streamed.links) == set(one_shot.links)
    assert all(streamed.links[i] == one_shot.links[i] for i in one_shot.links)
    differences = []
    for node in range(one_shot.query.shape[0]):
        owners = (int(one_shot.published[3][node]), int(streamed.published[3][node]))
        identities = _differing_identities(
            one_shot=one_shot, streamed=streamed, node=node
        )
        query_differs = one_shot.query[node].tobytes() != streamed.query[node].tobytes()
        if identities or query_differs:
            differences.append(_Difference(node, identities, owners, query_differs))
        else:
            assert owners[0] == owners[1], (
                f"node {node}: identical records, owners {owners}"
            )
    return tuple(differences)


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [0, *_WIDTHS])
def test_each_route_publishes_the_maximum_of_the_records_it_folded(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> None:
    """On its own records, a route's owner is the documented order's maximum.

    Under certified arithmetic the order is the literal one — exact affine value,
    right extension, exact slope, smaller stable identity — evaluated in rational
    arithmetic on the stored records. Under ordinary arithmetic the fold compares
    working-format rank fields it evaluates itself; those fields are captured as
    compared and the owner must be their lexicographic maximum, so the ordinary
    certificate covers the reduction, and the field evaluation is the documented
    ordinary contract.
    """
    _skip_without_payload(arithmetic)
    _certify_selection(
        _route(arithmetic=arithmetic, interval_batch_size=interval_batch_size)
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [0, *_WIDTHS])
def test_each_route_reads_its_channels_from_the_owner_it_publishes(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> None:
    """Value, policy and marginal at a node are read from the published owner."""
    _skip_without_payload(arithmetic)
    _certify_readout(
        _route(arithmetic=arithmetic, interval_batch_size=interval_batch_size)
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", _WIDTHS)
def test_identical_records_give_identical_owners_across_routes(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int
) -> None:
    """A partition admits the one-shot links and, on identical records, its owners."""
    _skip_without_payload(arithmetic)
    _compare_routes(
        one_shot=_route(arithmetic=arithmetic, interval_batch_size=0),
        streamed=_route(arithmetic=arithmetic, interval_batch_size=interval_batch_size),
    )


@pytest.mark.parametrize("field", _RECORD_FIELDS)
@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", _WIDTHS)
def test_the_streamed_blocks_produce_the_one_shot_records_to_the_format_spacing(
    *, arithmetic: ComparisonArithmetic, interval_batch_size: int, field: str
) -> None:
    """Every live streamed candidate is the one-shot candidate at its position.

    A record is compared at the spacing of the quantity that carries its rounding.
    An interior candidate's fields are compared at their own spacing. A point
    candidate — a savings corner or a savings-node point — consumes a grid
    point's cash-on-hand minus a savings node, so one unit in the last place of
    that cash-on-hand is many units of a small consumption and, through the
    utility, of its value and marginal; its fields are compared through that
    consumption at the cash-on-hand's spacing.
    """
    _skip_without_payload(arithmetic)
    one_shot = _route(arithmetic=arithmetic, interval_batch_size=0)
    streamed = _route(arithmetic=arithmetic, interval_batch_size=interval_batch_size)
    positions = sorted(streamed.candidates)
    assert positions == sorted(one_shot.candidates)
    got = np.asarray(
        [getattr(streamed.candidates[p], field) for p in positions], dtype=_dtype()
    )
    expected = np.asarray(
        [getattr(one_shot.candidates[p], field) for p in positions], dtype=_dtype()
    )
    budget = np.asarray(
        [
            _record_budget(candidate=one_shot.candidates[p], position=p, field=field)
            for p in positions
        ],
        dtype=_dtype(),
    )
    gap = np.abs(got - expected)
    worst = int(np.argmax(gap / budget))
    assert np.all(gap <= budget), (
        f"{field} at position {positions[worst]}: {got[worst]!r} vs "
        f"{expected[worst]!r}, budget {budget[worst]!r}"
    )


def _savings_node_of(*, position: int) -> float | None:
    """The savings node a point candidate at `position` consumes against.

    `None` for an interior candidate, whose consumption is the Euler inversion's
    and carries no cancellation.
    """
    layout = nbegm_step._streamed_interval_block_layout(
        interval_batch_size=1,
        n_intervals=_N_INTERVALS,
        interval_stride=4 * (_N_SAVINGS + _N_LIQUID),
        n_liquid=_N_LIQUID,
        n_savings=_N_SAVINGS,
    )
    savings_grid = np.asarray(_geometry()["savings_grid"])
    if position < layout.s0_offset:
        return None
    if position < layout.smax_offset:
        return float(savings_grid[0])
    if position < layout.node_offset:
        return float(savings_grid[-1])
    assert position < layout.node_offset + layout.node_family_size
    return float(savings_grid[((position - layout.node_offset) // 2) % _N_SAVINGS])


def _record_budget(*, candidate: _Candidate, position: int, field: str) -> float:
    """How far a record's field may move between widths, in the format's units.

    Sixteen units at the field's own spacing, plus — for a point candidate — the
    first-order image of sixteen units at the spacing of the cash-on-hand its
    consumption is cut from: the consumption moves by that amount, the value by
    the marginal utility times it, and the marginal by the marginal utility's
    derivative times it (scaled by the record's own cash-on-hand slope).
    """
    dtype = _dtype()
    own = _PARTITION_ULP * float(
        np.spacing(np.abs(np.asarray(getattr(candidate, field), dtype)))
    )
    savings_node = _savings_node_of(position=position)
    if savings_node is None or field in ("endog_grid", "segment_id"):
        return own
    preferences = _geometry()["preferences"]
    consumption = jnp.asarray(candidate.policy, dtype=dtype)
    cash_on_hand = np.asarray(candidate.policy + savings_node, dtype=dtype)
    moved = _PARTITION_ULP * float(np.spacing(np.abs(cash_on_hand)))
    marginal_utility = float(preferences.marginal_utility(consumption))
    if field == "policy":
        return own + moved
    if field == "value":
        return own + marginal_utility * moved
    assert field == "marginal"
    curvature = float(jax.grad(preferences.marginal_utility)(consumption))
    return own + abs(curvature) * (candidate.marginal / marginal_utility) * moved


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
def test_the_one_shot_owner_names_several_candidates(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """The owner comparison ranges over distinct identities, not a constant."""
    _skip_without_payload(arithmetic)
    owner = _published(arithmetic=arithmetic, interval_batch_size=0)[3]
    assert np.unique(owner[owner != NO_OWNER]).size > 1


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
@pytest.mark.parametrize("interval_batch_size", [0, *_WIDTHS])
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
@pytest.mark.parametrize("interval_batch_size", _WIDTHS)
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


def test_the_certificate_rejects_a_fold_that_relabels_identities() -> None:
    """A fold naming candidates by block-local slot publishes owners no record has."""
    with pytest.raises(AssertionError):
        _certify_selection(
            _route(arithmetic="ordinary", interval_batch_size=2, fold="relabelled")
        )


@pytest.mark.parametrize("channel", range(len(_CHANNELS)), ids=_CHANNELS)
def test_the_level_assertions_do_not_see_a_fold_that_relabels_identities(
    *, channel: int
) -> None:
    """Relabelled identities leave every level within the spacing budget."""
    relabelled = _route(arithmetic="ordinary", interval_batch_size=2, fold="relabelled")
    assert_agrees_to_ulp(
        got=relabelled.published[channel],
        expected=_published(arithmetic="ordinary", interval_batch_size=0)[channel],
        n_ulp=_PARTITION_ULP,
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
def test_the_certificate_rejects_a_fold_that_selects_an_inferior_bracketing_record(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """A fold publishing a plausible but inferior record, with its payloads, fails."""
    _skip_without_payload(arithmetic)
    with pytest.raises(AssertionError, match="not the maximum"):
        _certify_selection(
            _route(arithmetic=arithmetic, interval_batch_size=2, fold="inferior")
        )


def _synthetic_route(
    *,
    arithmetic: ComparisonArithmetic,
    stack: dict[str, np.ndarray],
    query: np.ndarray,
    published: tuple[np.ndarray, ...],
) -> _Route:
    """A route over a hand-built candidate stack, for the certificate's controls.

    Under ordinary arithmetic the rank fields come from the production evaluation
    of the stack's links, taken eagerly; the controls exercise the certificate's
    selection rule, not the fold that would evaluate them in a compiled program.
    """
    n_candidates = int(stack["endog_grid"].shape[0])
    identities = np.arange(2 * n_candidates - 1, dtype=np.int32)
    candidates: dict[int, _Candidate] = {}
    links: dict[int, _Link] = {}
    notes: list[str] = []
    _admit(
        arrays=stack,
        positions=np.arange(n_candidates),
        identities=identities,
        candidates=candidates,
        links=links,
        notes=notes,
    )
    keys: dict[int, _Keys] | None = None
    if arithmetic == "ordinary":
        segment_links, _ = envelope_query._segment_links_from_candidates(
            **{f: jnp.asarray(stack[f]) for f in _STACK_FIELDS},
            stable_index=jnp.asarray(identities),
            feasibility_partition=None,
            feasible_interval_mask=None,
        )
        rank, brackets = envelope_query._batched_link_terms(
            left_grid=segment_links.left_grid[None, :],
            right_grid=segment_links.right_grid[None, :],
            left_value=segment_links.left_value[None, :],
            right_value=segment_links.right_value[None, :],
            live=segment_links.live[None, :],
            stable_index=segment_links.stable_index[None, :],
            query=jnp.asarray(query),
        )
        keys = {}
        for column, identity in enumerate(identities.tolist()):
            _set_keys(
                keys=keys,
                identity=identity,
                rows=np.asarray(brackets)[:, column],
                fields=tuple(np.asarray(f)[:, column] for f in rank[:4]),
                n_query=int(query.shape[0]),
                notes=notes,
            )
    return _Route(
        arithmetic,
        query,
        candidates,
        links,
        n_candidates,
        published,
        keys,
        tuple(notes),
    )


def _dtype() -> np.dtype:
    return np.asarray(_geometry()["liquid_grid"]).dtype


def _step(*, value: float, n_ulp: int) -> np.ndarray:
    """`value` moved `n_ulp` representable steps in the working format."""
    moved = np.asarray(value, dtype=_dtype())
    towards = np.asarray(np.inf if n_ulp > 0 else -np.inf, dtype=moved.dtype)
    for _ in range(abs(n_ulp)):
        moved = np.nextafter(moved, towards)
    return moved


def _two_points(*, field: str, n_ulp: int) -> tuple[dict[str, np.ndarray], int, int]:
    """Two coincident point candidates at the origin, one a unit above the other.

    The lower point is the first candidate; its self-bracket is identity 1 and the
    higher point's is identity 2, the correct owner at the origin. The named field
    of the lower point (or, for `winner_value`, the value of the higher) is then
    moved `n_ulp` steps — a change that never makes the lower point the maximum.
    """
    dtype = _dtype()
    stack = {
        "endog_grid": np.zeros(2, dtype=dtype),
        "value": np.asarray([1.0, _step(value=1.0, n_ulp=1)], dtype=dtype),
        "policy": np.asarray([0.5, _step(value=0.5, n_ulp=1)], dtype=dtype),
        "marginal": np.asarray([2.0, 2.0], dtype=dtype),
        "segment_id": np.asarray([0.0, 1.0], dtype=dtype),
    }
    index, name = (1, "value") if field == "winner_value" else (0, field)
    stack[name][index] = _step(value=float(stack[name][index]), n_ulp=n_ulp)
    # The specimen must survive construction in the working format: the higher
    # point must still be higher, or the control tests a tie instead of a fault.
    assert stack["value"][1] > stack["value"][0]
    return stack, 1, 2


def _publish_point(*, stack: dict[str, np.ndarray], index: int, owner: int) -> tuple:
    """Publish one point's own payloads under `owner`, as a one-node route would."""
    return (
        *(np.asarray([stack[f][index]], dtype=_dtype()) for f in _CHANNELS),
        np.asarray([owner], dtype=np.int32),
    )


_MUTATIONS = [
    *(("marginal", n_ulp) for n_ulp in range(1, 17)),
    ("value", -1),
    ("winner_value", 1),
    ("policy", 1),
]


@pytest.mark.parametrize(("field", "n_ulp"), _MUTATIONS)
@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
def test_the_certificate_rejects_an_inferior_winner_however_its_record_moved(
    *, arithmetic: ComparisonArithmetic, field: str, n_ulp: int
) -> None:
    """Publishing the lower of two coincident points fails, whatever else moved.

    Moving a non-ranking field of the inferior point, lowering its value further,
    or raising the winner's value leaves the maximum unchanged; the certificate
    rejects the inferior owner and its payloads in every case, although every
    published level stays within the spacing budget of the correct one.
    """
    stack, inferior, _ = _two_points(field=field, n_ulp=n_ulp)
    route = _synthetic_route(
        arithmetic=arithmetic,
        stack=stack,
        query=np.zeros(1, dtype=_dtype()),
        published=_publish_point(stack=stack, index=0, owner=inferior),
    )
    with pytest.raises(AssertionError, match="not the maximum"):
        _certify_selection(route)


@pytest.mark.parametrize(("field", "n_ulp"), _MUTATIONS)
@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
def test_the_certificate_accepts_the_production_selection_on_the_same_records(
    *, arithmetic: ComparisonArithmetic, field: str, n_ulp: int
) -> None:
    """The production fold and finisher select and read the higher point; accepted."""
    _skip_without_payload(arithmetic)
    stack, _, correct = _two_points(field=field, n_ulp=n_ulp)
    query = jnp.zeros(1, dtype=_dtype())
    winner = merge_envelope_winner(
        held=empty_envelope_winner(query=query),
        endog_grid=jnp.asarray(stack["endog_grid"]),
        policy=jnp.asarray(stack["policy"]),
        value=jnp.asarray(stack["value"]),
        marginal=jnp.asarray(stack["marginal"]),
        segment_id=jnp.asarray(stack["segment_id"]),
        stable_index=jnp.arange(3, dtype=jnp.int32),
        query=query,
        arithmetic=arithmetic,
    )
    value, policy, marginal = finish_envelope_winner(
        winner=winner, query=query, arithmetic=arithmetic
    )
    published = tuple(
        np.asarray(c) for c in (value, marginal, policy, winner.stable_index)
    )
    assert int(published[3][0]) == correct
    _certify_route(
        _synthetic_route(
            arithmetic=arithmetic,
            stack=stack,
            query=np.asarray(query),
            published=published,
        )
    )


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
def test_the_certificate_rejects_an_injected_inferior_winner_read_by_the_finisher(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """A fold output replaced by the inferior point, then finished as usual, fails.

    The lower point's own endpoints, payloads and identity are written into the
    standing winner and the unmodified finisher reads them; every published level
    is within a unit of the correct one, and the certificate still rejects it.
    """
    _skip_without_payload(arithmetic)
    stack, inferior, _ = _two_points(field="marginal", n_ulp=1)
    query = jnp.zeros(1, dtype=_dtype())
    winner = merge_envelope_winner(
        held=empty_envelope_winner(query=query),
        endog_grid=jnp.asarray(stack["endog_grid"]),
        policy=jnp.asarray(stack["policy"]),
        value=jnp.asarray(stack["value"]),
        marginal=jnp.asarray(stack["marginal"]),
        segment_id=jnp.asarray(stack["segment_id"]),
        stable_index=jnp.arange(3, dtype=jnp.int32),
        query=query,
        arithmetic=arithmetic,
    )
    injected = winner._replace(
        **{
            f"{side}_{name}": jnp.asarray([stack[field][0]])
            for side in ("left", "right")
            for name, field in (
                ("grid", "endog_grid"),
                ("value", "value"),
                ("policy", "policy"),
                ("marginal", "marginal"),
            )
        },
        stable_index=jnp.asarray([inferior], dtype=jnp.int32),
    )
    value, policy, marginal = finish_envelope_winner(
        winner=injected, query=query, arithmetic=arithmetic
    )
    route = _synthetic_route(
        arithmetic=arithmetic,
        stack=stack,
        query=np.asarray(query),
        published=tuple(
            np.asarray(c) for c in (value, marginal, policy, injected.stable_index)
        ),
    )
    with pytest.raises(AssertionError, match="not the maximum"):
        _certify_selection(route)


@pytest.mark.parametrize("arithmetic", ["ordinary", "certified"])
def test_the_certificate_rejects_an_owner_that_names_no_admitted_record(
    *, arithmetic: ComparisonArithmetic
) -> None:
    """An identity outside the admitted set is never an owner, whatever its levels."""
    stack, _, correct = _two_points(field="marginal", n_ulp=1)
    published = _publish_point(stack=stack, index=1, owner=999_999)
    route = _synthetic_route(
        arithmetic=arithmetic,
        stack=stack,
        query=np.zeros(1, dtype=_dtype()),
        published=published,
    )
    with pytest.raises(AssertionError, match="not the maximum"):
        _certify_selection(route)
    with pytest.raises(AssertionError, match="no admitted record"):
        _certify_readout(route)
    assert _expected_owners(route).tolist() == [correct]


def test_the_agreement_bound_rejects_a_fold_that_publishes_another_owner() -> None:
    """The level instrument fires on a finite ownership gap, in this run."""
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
