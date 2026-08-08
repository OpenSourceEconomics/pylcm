"""Ownership follows the stored values, not the level they happen to sit on.

A branch's *value* is read to a precision proportional to its magnitude, so on a
large common value level two reads whose exact difference is many orders of
magnitude above zero can still carry error bars that overlap. Deciding ownership
between those reads would hand the query to whichever branch the tie-break
happens to prefer — an outcome the level chose, not the economics.

Deciding on the difference instead removes the level from the question: it is
subtracted from every candidate exactly before anything is multiplied, so the
arithmetic that settles ownership runs at the magnitude of the gaps rather than
of the values.

Two branches here cross at an abscissa the format represents exactly, and the
query sits one representable step to its left. What the stored endpoint values
imply there is the whole specification, and an exact rational oracle over those
same stored floats says what it is. That matters at the largest level tested,
where the spacing is coarser than the offsets that define the branches: the
stored geometry is then *not* the geometry the offsets describe — the crossing
moves — and the envelope owes agreement with what is stored, not with what was
intended.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.conftest import assert_agrees_to_ulp

_X0 = 100.0
_X1 = 1100.0
_CROSSING = 434.0

# Both branches pass through `(_CROSSING, level)`. Their endpoint values are
# offsets from the common level: the flat branch rises by 2 per unit, the steep
# one by 4, and both are integers a float holds exactly at every level used here.
_FLAT_OFFSETS = (-668.0, 1332.0)
_STEEP_OFFSETS = (-1336.0, 2664.0)

_FLAT_POLICY = 0.5
_STEEP_POLICY = 0.25
_FLAT_MARGINAL = 2.0
_STEEP_MARGINAL = 4.0


def _stored_values(level: float) -> jnp.ndarray:
    """The four endpoint values as the working format actually holds them.

    On a large enough level the offsets are finer than the spacing there, so what
    is stored is not `level + offset` but its rounding. The oracle has to read the
    same numbers the envelope does, or it is describing a different geometry.
    """
    return jnp.asarray(
        [
            level + _FLAT_OFFSETS[0],
            level + _FLAT_OFFSETS[1],
            level + _STEEP_OFFSETS[0],
            level + _STEEP_OFFSETS[1],
        ]
    )


def _exact_value(*, v0: float, v1: float, query: float) -> Fraction:
    """The branch's affine value at `query`, in exact rational arithmetic."""
    x0, x1, x = Fraction(_X0), Fraction(_X1), Fraction(query)
    return (Fraction(v0) * (x1 - x) + Fraction(v1) * (x - x0)) / (x1 - x0)


def _exact_owner_policy(*, level: float, query: float) -> float:
    """The policy of the branch that is exactly highest at `query`.

    An exact tie goes to the steeper branch: that is the one that continues higher
    to the right, which is what the envelope's right-continuous convention takes.
    """
    stored = [float(entry) for entry in _stored_values(level)]
    flat = _exact_value(v0=stored[0], v1=stored[1], query=query)
    steep = _exact_value(v0=stored[2], v1=stored[3], query=query)
    return _FLAT_POLICY if flat > steep else _STEEP_POLICY


def _rows(level: float) -> dict:
    """The two branches as the envelope's own row arrays."""
    return {
        "endog_grid": jnp.asarray([_X0, _X1, _X0, _X1]),
        "policy": jnp.asarray(
            [_FLAT_POLICY, _FLAT_POLICY, _STEEP_POLICY, _STEEP_POLICY]
        ),
        "value": _stored_values(level),
        "marginal": jnp.asarray(
            [
                _FLAT_MARGINAL,
                _FLAT_MARGINAL,
                _STEEP_MARGINAL,
                _STEEP_MARGINAL,
            ]
        ),
        "segment_id": jnp.asarray([0.0, 0.0, 1.0, 1.0]),
    }


def _published(*, level: float, query: float, block_size: int) -> tuple[float, float]:
    """The policy and marginal the envelope publishes at `query`."""
    _, got_policy, got_marginal = envelope_at_query(
        **_rows(level),
        x_query=jnp.asarray([query]),
        segment_block_size=block_size,
    )
    return float(got_policy[0]), float(got_marginal[0])


def _just_left_of_crossing() -> float:
    """The largest representable abscissa strictly below the crossing."""
    crossing = np.asarray(jnp.asarray(_CROSSING))
    return float(np.nextafter(crossing, np.asarray(crossing.dtype.type(-np.inf))))


def _levels() -> tuple[float, ...]:
    """Common value levels spanning none up to far above the branches' own scale."""
    return (0.0, 2.0**10, 2.0**23, 2.0**30)


@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
@pytest.mark.parametrize("level", _levels())
def test_the_exactly_highest_branch_supplies_the_policy(level: float, block_size: int):
    """The branch the stored values put highest is the one that owns the query."""
    query = _just_left_of_crossing()

    got_policy, _ = _published(level=level, query=query, block_size=block_size)

    assert got_policy == _exact_owner_policy(level=level, query=query)


@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
@pytest.mark.parametrize("level", _levels())
def test_the_marginal_comes_from_the_same_branch_as_the_policy(
    level: float, block_size: int
):
    """Value, policy, and marginal are published from one branch."""
    query = _just_left_of_crossing()
    expected_policy = _exact_owner_policy(level=level, query=query)

    _, got_marginal = _published(level=level, query=query, block_size=block_size)

    expected = _FLAT_MARGINAL if expected_policy == _FLAT_POLICY else _STEEP_MARGINAL
    assert got_marginal == expected


@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
@pytest.mark.parametrize("level", _levels())
def test_the_steeper_branch_owns_the_crossing_itself(level: float, block_size: int):
    """At an exact tie the right-continuous rule takes the branch continuing higher."""
    got_policy, _ = _published(level=level, query=_CROSSING, block_size=block_size)

    assert got_policy == _STEEP_POLICY


@pytest.mark.parametrize("transform", ["eager", "jit", "vmap"])
@pytest.mark.parametrize("level", _levels())
def test_the_owner_is_the_same_under_every_execution_path(level: float, transform: str):
    """Tracing or batching the query changes nothing about who owns it."""
    query = _just_left_of_crossing()

    def owner(x_query: jnp.ndarray) -> jnp.ndarray:
        _, got_policy, _ = envelope_at_query(
            **_rows(level), x_query=x_query, segment_block_size=0
        )
        return got_policy

    run = {
        "eager": lambda: owner(jnp.asarray([query])),
        "jit": lambda: jax.jit(owner)(jnp.asarray([query])),
        "vmap": lambda: jax.vmap(owner)(jnp.asarray([[query], [query]])),
    }[transform]

    # The two branches' policies differ by a factor of two, so bounding the gap
    # at a few ULP still names one branch — while leaving the fused arithmetic
    # each path compiles free to land on a representable neighbour.
    assert_agrees_to_ulp(
        run().ravel()[0],
        jnp.asarray(_exact_owner_policy(level=level, query=query)),
        n_ulp=2,
    )
