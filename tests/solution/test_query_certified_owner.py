"""The query envelope publishes the branch that is certifiably higher.

Ownership of a query is a structural decision, not a reported magnitude: one
branch is higher there, and it supplies the value, the policy and the marginal
together. A magnitude-scaled tolerance cannot make that decision, because it is
not invariant to a common additive level — put two branches on a large constant
and a strict, exactly representable difference between them falls inside any
relative band, at which point the branch that is genuinely lower can be selected
because it happens to be steeper.

`certified_margin_sign` decides the same question exactly, from the sign of the
cross-multiplied determinant in double-double arithmetic. These tests pin the
envelope to that verdict: where the certified sign is strict, the winner it
names owns every channel, on every evaluation path the kernel offers.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import certified_margin_sign
from _lcm.egm.upper_envelope.query import envelope_at_query


def _dtype() -> jnp.dtype:
    """Return the float type the suite is running at."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _straddling_pair() -> dict[str, float]:
    """Return two ascending affine branches separated by a representable margin.

    Both share endpoints and bracket the query, so both are live contenders. `A`
    is higher at the query by an exactly representable amount; `B` is steeper, so
    a selection that resolves the pair by slope rather than by height picks `B`.
    The values sit on a large constant, which is what pushes the strict margin
    inside any relative tolerance band.
    """
    if jax.config.jax_enable_x64:
        return {
            "x0": 1024.0,
            "x1": 2**53 + 1024.0,
            "query": 1024.0 + 825086954632762.6,
            "a0": -(2.0**49),
            "a1": 2.0**49,
            "b0": -(2.0**49) - 1.5,
            "b1": 2.0**49 + 14.875,
        }
    return {
        "x0": 1024.0,
        "x1": 16778240.0,
        "query": 1356758.625,
        "a0": -(2.0**20),
        "a1": 2.0**20,
        "b0": -(2.0**20) - 1.0,
        "b1": 2.0**20 + 11.375,
    }


def _exact_at_query(*, v0: float, v1: float, case: dict[str, float]) -> Fraction:
    """Return the branch's value at the query in exact rational arithmetic."""
    dtype = _dtype()

    def frac(x: float) -> Fraction:
        return Fraction.from_float(float(np.asarray(jnp.asarray(x, dtype=dtype))))

    left, right, query = frac(case["x0"]), frac(case["x1"]), frac(case["query"])
    return frac(v0) + (frac(v1) - frac(v0)) * (query - left) / (right - left)


def _evaluate(
    *, case: dict[str, float], reverse: bool, block_size: int
) -> tuple[float, float, float]:
    """Return the published (value, policy, marginal) for one evaluation path."""
    dtype = _dtype()
    grid = [case["x0"], case["x1"], case["x0"], case["x1"]]
    value = [case["a0"], case["a1"], case["b0"], case["b1"]]
    policy = [0.5, 0.5, 0.25, 0.25]
    marginal = [0.125, 0.125, 0.25, 0.25]
    segment = [0.0, 0.0, 1.0, 1.0]
    if reverse:
        order = [2, 3, 0, 1]
        grid = [grid[i] for i in order]
        value = [value[i] for i in order]
        policy = [policy[i] for i in order]
        marginal = [marginal[i] for i in order]

    published = envelope_at_query(
        endog_grid=jnp.asarray(grid, dtype=dtype),
        value=jnp.asarray(value, dtype=dtype),
        policy=jnp.asarray(policy, dtype=dtype),
        marginal=jnp.asarray(marginal, dtype=dtype),
        segment_id=jnp.asarray(segment, dtype=dtype),
        x_query=jnp.asarray(case["query"], dtype=dtype),
        segment_block_size=block_size,
    )
    return tuple(float(np.asarray(channel)) for channel in published)


def test_the_certified_sign_names_the_higher_branch() -> None:
    """The fixture is a strict certified win for `A`, not a tie.

    Guards the tests below: if this pair ever stops being strictly separated,
    they would pass for the wrong reason.
    """
    case = _straddling_pair()
    dtype = _dtype()
    margin = _exact_at_query(
        v0=case["a0"], v1=case["a1"], case=case
    ) - _exact_at_query(v0=case["b0"], v1=case["b1"], case=case)

    sign = int(
        certified_margin_sign(
            a_x0=jnp.asarray(case["x0"], dtype=dtype),
            a_x1=jnp.asarray(case["x1"], dtype=dtype),
            a_v0=jnp.asarray(case["a0"], dtype=dtype),
            a_v1=jnp.asarray(case["a1"], dtype=dtype),
            b_x0=jnp.asarray(case["x0"], dtype=dtype),
            b_x1=jnp.asarray(case["x1"], dtype=dtype),
            b_v0=jnp.asarray(case["b0"], dtype=dtype),
            b_v1=jnp.asarray(case["b1"], dtype=dtype),
            x_query=jnp.asarray(case["query"], dtype=dtype),
        )
    )

    assert margin > 0
    assert sign == 1


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
def test_the_certified_winner_supplies_every_channel(
    *, reverse: bool, block_size: int
) -> None:
    """The branch certified higher owns value, policy and marginal alike.

    The losing branch is steeper and its policy is `0.25`; publishing that is
    the failure this pins, on every row order and segment block size.
    """
    case = _straddling_pair()
    value, policy, marginal = _evaluate(
        case=case, reverse=reverse, block_size=block_size
    )
    expected_value = float(_exact_at_query(v0=case["a0"], v1=case["a1"], case=case))

    assert policy == 0.5
    assert marginal == 0.125
    assert value == pytest.approx(expected_value, rel=1e-6)


def test_the_winner_does_not_change_when_every_value_is_translated() -> None:
    """Adding a constant to both branches cannot change which one is higher.

    The difference between two branches is invariant to a common additive level,
    so the owner is too. A selection resolved by a tolerance scaled to the value
    magnitude is not, which is what makes this the discriminating case.
    """
    case = _straddling_pair()
    shift = 2.0**40 if jax.config.jax_enable_x64 else 2.0**18
    translated = dict(case)
    for key in ("a0", "a1", "b0", "b1"):
        translated[key] = case[key] + shift

    _, policy, marginal = _evaluate(case=translated, reverse=False, block_size=0)

    assert policy == 0.5
    assert marginal == 0.125


def test_the_certified_winner_survives_compilation() -> None:
    """Ownership is a property of the correspondence, not of the trace."""
    case = _straddling_pair()
    dtype = _dtype()

    compiled = jax.jit(envelope_at_query, static_argnames=("segment_block_size",))
    _, policy, marginal = compiled(
        endog_grid=jnp.asarray(
            [case["x0"], case["x1"], case["x0"], case["x1"]], dtype=dtype
        ),
        value=jnp.asarray(
            [case["a0"], case["a1"], case["b0"], case["b1"]], dtype=dtype
        ),
        policy=jnp.asarray([0.5, 0.5, 0.25, 0.25], dtype=dtype),
        marginal=jnp.asarray([0.125, 0.125, 0.25, 0.25], dtype=dtype),
        segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.asarray(case["query"], dtype=dtype),
        segment_block_size=0,
    )

    assert float(np.asarray(policy)) == 0.5
    assert float(np.asarray(marginal)) == 0.125
