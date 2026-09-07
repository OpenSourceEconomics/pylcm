"""The MSS envelope offers a certified and an ordinary arithmetic.

Both arithmetics decide the same geometry: which stored piece covers an interval,
which node owns a query, and where two branches hand over. They differ only in how
a comparison between two chords is settled — the certified one through exact
stored-operand arithmetic, the ordinary one in the working floating format, with
no native kernel required. Selecting the ordinary arithmetic buys speed and gives
up the exactness of a decision the working format cannot separate; it does not give
up the geometry.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope import mss
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import MSSEnvelope

# A knot witness: branch 1's piece over [10, 11] crosses branch 0 at 10.5, while its
# following piece over [11, 12] would cross at 10.75 if extrapolated backwards.
_KNOT_GRID = [10.0, 12.0, 10.0, 11.0, 12.0]
_KNOT_POLICY = [8.0, 8.0, 2.0, 2.0, 2.0]
_KNOT_VALUE = [11.0, 13.0, 10.0, 13.0, 18.0]
_KNOT_LABEL = [0.0, 0.0, 1.0, 1.0, 1.0]


def _refine(*, arithmetic, grid=None, policy=None, value=None, label=None):
    """Publish the refined rows for the knot witness under one arithmetic."""
    return mss.refine_envelope(
        endog_grid=jnp.asarray(_KNOT_GRID if grid is None else grid),
        policy=jnp.asarray(_KNOT_POLICY if policy is None else policy),
        value=jnp.asarray(_KNOT_VALUE if value is None else value),
        segment_id=jnp.asarray(_KNOT_LABEL if label is None else label),
        n_refined=32,
        arithmetic=arithmetic,
    )


def _live_rows(published):
    """Return the live prefix of a published row triple."""
    grid, policy, value, count = published
    n = int(count)
    return np.asarray(grid)[:n], np.asarray(policy)[:n], np.asarray(value)[:n]


def test_mss_envelope_defaults_to_the_certified_arithmetic() -> None:
    """A regime that names no arithmetic gets the certified one."""
    assert MSSEnvelope().arithmetic == "certified"


def test_mss_envelope_accepts_the_ordinary_arithmetic() -> None:
    """A regime can select the ordinary arithmetic by name."""
    assert MSSEnvelope(arithmetic="ordinary").arithmetic == "ordinary"


def test_mss_envelope_rejects_an_unknown_arithmetic() -> None:
    """An arithmetic outside the declared pair is refused when the regime is built."""
    with pytest.raises(RegimeInitializationError, match="arithmetic"):
        MSSEnvelope(arithmetic="approximate")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("arithmetic", ["certified", "ordinary"])
def test_both_arithmetics_place_the_knot_at_the_covering_pieces_root(
    arithmetic,
) -> None:
    """The event is at 10.5, the root of the pieces covering the interval.

    The incoming branch's following piece would meet the outgoing branch at 10.75,
    but it does not cover the interval below its own knot. Piece selection is
    geometry, so neither arithmetic may report 10.75.
    """
    x, policy, value = _live_rows(_refine(arithmetic=arithmetic))
    duplicated = x[:-1][x[:-1] == x[1:]]
    assert len(duplicated) == 1
    assert float(duplicated[0]) == 10.5
    ids = np.flatnonzero(x == 10.5)
    np.testing.assert_array_equal(policy[ids], [8.0, 2.0])
    np.testing.assert_array_equal(value[ids], [11.5, 11.5])


@pytest.mark.parametrize("arithmetic", ["certified", "ordinary"])
def test_both_arithmetics_read_the_incoming_policy_past_the_knot(arithmetic) -> None:
    """At 85/8 the reader returns the incoming branch's policy 2 and value 95/8."""
    grid, policy, value, _count = _refine(arithmetic=arithmetic)
    query = jnp.asarray([85 / 8])
    got_policy = float(interp_on_padded_grid(x_query=query, xp=grid, fp=policy)[0])
    got_value = float(interp_on_padded_grid(x_query=query, xp=grid, fp=value)[0])
    assert got_policy == 2.0
    assert got_value == 95 / 8


def test_the_two_arithmetics_agree_on_a_well_separated_witness() -> None:
    """Where the working format separates every comparison, both publish one answer."""
    certified = _live_rows(_refine(arithmetic="certified"))
    ordinary = _live_rows(_refine(arithmetic="ordinary"))
    for exact_channel, loose_channel in zip(certified, ordinary, strict=True):
        np.testing.assert_array_equal(exact_channel, loose_channel)


def test_the_ordinary_arithmetic_needs_no_exact_kernel() -> None:
    """The ordinary arithmetic reaches no native primitive, so it runs without one.

    Selecting it is the route for a backend whose exact-affine payload is absent;
    a call that reached the kernel anyway would fail here rather than silently
    depend on it.
    """

    def _unavailable(*_args, **_kwargs):
        msg = "the exact kernel must not be reached by the ordinary arithmetic"
        raise AssertionError(msg)

    with (
        patch.object(mss, "certified_margin_sign", _unavailable),
        patch.object(mss, "exact_affine_handover", _unavailable),
        patch.object(mss, "exact_affine_read", _unavailable),
        patch.object(mss, "exact_query_winner_batched", _unavailable),
    ):
        x, policy, value = _live_rows(_refine(arithmetic="ordinary"))
    assert np.isfinite(x).all()
    assert np.isfinite(policy).all()
    assert np.isfinite(value).all()


@pytest.mark.parametrize("arithmetic", ["certified", "ordinary"])
def test_both_arithmetics_hand_over_across_a_gap_without_poisoning(
    arithmetic,
) -> None:
    """Branches that share no support hand over, each node keeping its own value."""
    x, policy, value = _live_rows(
        _refine(
            arithmetic=arithmetic,
            grid=[0.0, 1.0, 2.0, 3.0],
            policy=[8.0, 8.0, 2.0, 2.0],
            value=[0.0, 1.0, 2.0, 3.0],
            label=[0.0, 0.0, 1.0, 1.0],
        )
    )
    np.testing.assert_array_equal(x, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(policy, [8.0, 8.0, 2.0, 2.0])
    np.testing.assert_array_equal(value, [0.0, 1.0, 2.0, 3.0])


@pytest.mark.parametrize("arithmetic", ["certified", "ordinary"])
def test_both_arithmetics_survive_jit(arithmetic) -> None:
    """The arithmetic is a static choice, so each mode compiles."""
    compiled = jax.jit(
        lambda g, p, v, s: mss.refine_envelope(
            endog_grid=g,
            policy=p,
            value=v,
            segment_id=s,
            n_refined=32,
            arithmetic=arithmetic,
        )
    )
    published = compiled(
        jnp.asarray(_KNOT_GRID),
        jnp.asarray(_KNOT_POLICY),
        jnp.asarray(_KNOT_VALUE),
        jnp.asarray(_KNOT_LABEL),
    )
    x, _policy, _value = _live_rows(published)
    assert float(x[:-1][x[:-1] == x[1:]][0]) == 10.5
