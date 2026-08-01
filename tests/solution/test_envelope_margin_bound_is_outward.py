"""The bound a certified margin carries never understates its own error.

Ownership rests entirely on the claim that the exact margin between two branches
lies inside `[value - bound, value + bound]`. A bound that is outward is the whole
content of that claim: one that understates by even a fraction of an ULP turns a
certificate into an estimate, and the failure is silent — the envelope keeps
naming a winner, and the winner is right almost always, so nothing downstream
looks wrong.

An exact rational oracle over the stored floats settles it. The margin is
generated over the geometry the envelope actually meets — links of widely
different widths and value scales, queries at both endpoints, at their
representable neighbours, and across the interior, and the same geometry lifted
onto a large common value level, which is where the reads that feed the margin
are least separable.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import certified_quotient_margin
from _lcm.egm.upper_envelope.query import _value_quotient

_N_CASES = 400
_SEED = 20260731


def _exact_quotient(
    *, x0: float, x1: float, v0: float, v1: float, x: float
) -> Fraction:
    """The link's affine value at `x`, exactly, over the stored floats."""
    if x == x0:
        return Fraction(v0)
    if x == x1:
        return Fraction(v1)
    lo, hi, query = Fraction(x0), Fraction(x1), Fraction(x)
    return (Fraction(v0) * (hi - query) + Fraction(v1) * (query - lo)) / (hi - lo)


def _certified(*, left: dict, right: dict, query: float, level: float):
    """The production margin of `left` over `right` at `query`, above `level`."""
    pair = {}
    for name, link in (("left", left), ("right", right)):
        numerator, divisor = _value_quotient(
            left=jnp.asarray(link["v0"]),
            right=jnp.asarray(link["v1"]),
            query=jnp.asarray(query),
            left_grid=jnp.asarray(link["x0"]),
            right_grid=jnp.asarray(link["x1"]),
            level=jnp.asarray(level),
        )
        pair[name] = (numerator, divisor)
    return certified_quotient_margin(
        left_numerator=pair["left"][0],
        left_divisor=pair["left"][1],
        right_numerator=pair["right"][0],
        right_divisor=pair["right"][1],
    )


def _stored(value: float) -> float:
    """Round a Python float into the working format and back."""
    return float(jnp.asarray(value))


def _link(
    *, x0: float, x1: float, level: float, scale: float, rng: np.random.Generator
) -> dict:
    """One link spanning `[x0, x1]`, its values scattered around `level`."""
    return {
        "x0": x0,
        "x1": x1,
        "v0": _stored(level + float(rng.normal(0.0, scale))),
        "v1": _stored(level + float(rng.normal(0.0, scale))),
    }


def _cases() -> list[tuple[dict, dict, float, float]]:
    """Random link pairs, the query they are compared at, and a common level."""
    rng = np.random.default_rng(seed=_SEED)
    cases = []
    for index in range(_N_CASES):
        x0 = _stored(float(rng.uniform(1.0, 1e3)))
        width = _stored(float(10.0 ** rng.uniform(-1.0, 3.0)))
        x1 = _stored(x0 + width)
        if not x1 > x0:
            continue
        scale = 10.0 ** rng.uniform(-2.0, 4.0)
        level = _stored(float(0.0 if index % 2 else 10.0 ** rng.uniform(3.0, 7.0)))
        draw = {"x0": x0, "x1": x1, "level": level, "scale": scale, "rng": rng}

        share = (0.0, 1.0, float(rng.uniform(0.0, 1.0)))[index % 3]
        query = _stored(x0 + share * (x1 - x0))
        query = min(max(query, x0), x1)
        cases.append((_link(**draw), _link(**draw), query, level))
    return cases


@pytest.mark.parametrize(("left", "right", "query", "level"), _cases())
def test_the_exact_margin_lies_inside_the_certified_interval(
    left: dict, right: dict, query: float, level: float
):
    """The exact margin between two links is within the bound the margin reports."""
    margin = _certified(left=left, right=right, query=query, level=level)
    if not bool(margin.trustworthy):
        pytest.skip("the margin declares itself uncertifiable, which claims nothing")

    exact = _exact_quotient(x=query, **left) - _exact_quotient(x=query, **right)
    error = abs(Fraction(float(margin.value)) - exact)

    assert error <= Fraction(float(margin.bound))


def test_a_link_compared_with_itself_certifies_an_exact_zero():
    """Comparing a link with itself is exactly zero, with nothing left over."""
    link = {"x0": 100.0, "x1": 1100.0, "v0": -3.5, "v1": 7.25}

    margin = _certified(left=link, right=link, query=434.0, level=0.0)

    assert (float(margin.value), float(margin.bound)) == (0.0, 0.0)
