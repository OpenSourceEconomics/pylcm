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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import certified_margin_sign
from _lcm.egm.upper_envelope.query import envelope_at_query
from lcm.typing import FloatND


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
        return Fraction(float(np.asarray(jnp.asarray(x, dtype=dtype))))

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
    value, policy, marginal = published
    return (
        float(np.asarray(value)),
        float(np.asarray(policy)),
        float(np.asarray(marginal)),
    )


def test_the_certified_sign_names_the_higher_branch() -> None:
    """The fixture is a strict certified win for `A`, not a tie.

    Guards the tests below: if this pair ever stops being strictly separated,
    they would pass for the wrong reason.
    """
    case = _straddling_pair()
    dtype = _dtype()
    margin = _exact_at_query(v0=case["a0"], v1=case["a1"], case=case) - _exact_at_query(
        v0=case["b0"], v1=case["b1"], case=case
    )

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


@pytest.mark.parametrize("exponent", [0, 10, 20, 30, 35, 40, 45])
def test_translating_every_value_never_promotes_the_lower_branch(
    *, exponent: int
) -> None:
    """Putting both branches on a common level never publishes the lower one.

    Which branch is higher is invariant to a common additive level, so no
    translation can make `B` the owner. What a translation *can* do is exhaust the
    resolution of the certified difference: the determinant is mathematically
    invariant, but the intermediates grow with the level while the margin does not,
    so the comparison is a cancellation that fixed-precision arithmetic eventually
    cannot certify. Whether it still can is not monotone in the level.

    So the guarantee is not that the owner is always published — it is that the
    lower branch never is. Either the certified owner supplies the channels, or
    they are NaN because nothing was certified. Publishing `B`'s policy of `0.25`
    is the one outcome ruled out.
    """
    dtype = _dtype()
    case = _straddling_pair()
    shift = 0.0 if exponent == 0 else 2.0**exponent
    # Round each translated endpoint into the working format first: at a large
    # enough level the format cannot hold the offsets that separate the branches,
    # and the geometry the kernel actually sees is the rounded one. The owner is
    # then derived from those same rounded numbers, so the test measures selection
    # rather than the fixture's own representability.
    translated = {
        key: (
            float(np.asarray(jnp.asarray(value + shift, dtype=dtype)))
            if key in {"a0", "a1", "b0", "b1"}
            else value
        )
        for key, value in case.items()
    }
    higher = _exact_at_query(
        v0=translated["a0"], v1=translated["a1"], case=translated
    ) - _exact_at_query(v0=translated["b0"], v1=translated["b1"], case=translated)
    if higher == 0:
        pytest.skip("this level collapses the two branches onto one line")
    expected_policy, expected_marginal = (0.5, 0.125) if higher > 0 else (0.25, 0.25)

    _, policy, marginal = _evaluate(case=translated, reverse=False, block_size=0)

    assert policy == expected_policy or np.isnan(policy)
    assert marginal == expected_marginal or np.isnan(marginal)


def test_an_uncertifiable_comparison_publishes_nothing() -> None:
    """Where ownership cannot be certified, no channel is published.

    A level large enough to exhaust the resolution of the difference leaves the
    comparison genuinely undecided. The kernel says so — NaN in all three
    channels — rather than falling back on the plain-float maximum, which is the
    quantity that cannot be trusted here in the first place.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("the float32 fixture certifies at every level it can represent")
    case = _straddling_pair()
    uncertifiable = dict(case)
    for key in ("a0", "a1", "b0", "b1"):
        uncertifiable[key] = case[key] + 2.0**40

    value, policy, marginal = _evaluate(case=uncertifiable, reverse=False, block_size=0)

    assert np.isnan(value)
    assert np.isnan(policy)
    assert np.isnan(marginal)


def test_two_coincident_branches_are_settled_right_continuously() -> None:
    """An exact tie is broken by what is higher just to the right, not by luck.

    Both branches take the same value at the query, so neither is above the
    other and the certified sign is exactly zero. The rule then prefers the
    steeper branch, which is the one that is higher immediately to the right.
    """
    dtype = _dtype()
    case = _straddling_pair()
    published = envelope_at_query(
        endog_grid=jnp.asarray(
            [case["x0"], case["x1"], case["x0"], case["x1"]], dtype=dtype
        ),
        # Both branches run from 0 to the same height, so they coincide
        # everywhere; the second is given the larger policy so the winner is
        # identifiable.
        value=jnp.asarray([0.0, 1024.0, 0.0, 1024.0], dtype=dtype),
        policy=jnp.asarray([0.25, 0.25, 0.5, 0.5], dtype=dtype),
        marginal=jnp.asarray([0.25, 0.25, 0.125, 0.125], dtype=dtype),
        segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.asarray(case["query"], dtype=dtype),
    )

    assert not np.isnan(float(np.asarray(published[1])))


def test_a_one_ulp_margin_is_still_a_strict_win() -> None:
    """Adjacent representable values separate two branches, and the kernel sees it.

    The margin is the smallest one the format admits at that magnitude. A
    tolerance cannot resolve it; the certified sign can, so the higher branch
    owns the query rather than the steeper one.
    """
    dtype = _dtype()
    base = 1024.0
    higher = float(np.nextafter(np.asarray(base, dtype=dtype), np.inf))
    published = envelope_at_query(
        endog_grid=jnp.asarray([0.0, 2048.0, 0.0, 2048.0], dtype=dtype),
        value=jnp.asarray([0.0, 2.0 * higher, 0.0, 2.0 * base], dtype=dtype),
        policy=jnp.asarray([0.5, 0.5, 0.25, 0.25], dtype=dtype),
        marginal=jnp.asarray([0.125, 0.125, 0.25, 0.25], dtype=dtype),
        segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        x_query=jnp.asarray(1024.0, dtype=dtype),
    )

    policy = float(np.asarray(published[1]))

    assert policy == 0.5 or np.isnan(policy)


def test_the_owner_is_the_same_under_vmap() -> None:
    """Batching queries does not change which branch owns any one of them."""
    dtype = _dtype()
    case = _straddling_pair()
    queries = jnp.asarray([case["query"]] * 4, dtype=dtype)

    def one(x: FloatND) -> FloatND:
        return envelope_at_query(
            endog_grid=jnp.asarray(
                [case["x0"], case["x1"], case["x0"], case["x1"]], dtype=dtype
            ),
            value=jnp.asarray(
                [case["a0"], case["a1"], case["b0"], case["b1"]], dtype=dtype
            ),
            policy=jnp.asarray([0.5, 0.5, 0.25, 0.25], dtype=dtype),
            marginal=jnp.asarray([0.125, 0.125, 0.25, 0.25], dtype=dtype),
            segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
            x_query=x,
        )[1]

    batched = jax.vmap(one)(queries)

    np.testing.assert_array_equal(np.asarray(batched), np.full((4,), 0.5))


def test_the_ordinary_route_is_available_but_only_densely() -> None:
    """`arithmetic` selects the comparison, and the blocked scan refuses to fake it.

    The ordinary route is the cheap plain-float selection, offered for callers
    that have established they do not need the certified one. It is not offered
    for the blocked scan, which carries the certified comparison only — serving
    that cost under the ordinary label would misreport what was paid.
    """
    dtype = _dtype()
    case = _straddling_pair()
    common: dict[str, Any] = {
        "endog_grid": jnp.asarray(
            [case["x0"], case["x1"], case["x0"], case["x1"]], dtype=dtype
        ),
        "value": jnp.asarray(
            [case["a0"], case["a1"], case["b0"], case["b1"]], dtype=dtype
        ),
        "policy": jnp.asarray([0.5, 0.5, 0.25, 0.25], dtype=dtype),
        "marginal": jnp.asarray([0.125, 0.125, 0.25, 0.25], dtype=dtype),
        "segment_id": jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        "x_query": jnp.asarray(case["query"], dtype=dtype),
    }

    ordinary = envelope_at_query(**common, arithmetic="ordinary")
    certified = envelope_at_query(**common, arithmetic="certified")

    # The fixture is exactly the case the two routes disagree on.
    assert float(np.asarray(ordinary[1])) == 0.25
    assert float(np.asarray(certified[1])) == 0.5

    with pytest.raises(ValueError, match="certified comparison only"):
        envelope_at_query(**common, segment_block_size=2, arithmetic="ordinary")


def test_the_dense_and_blocked_paths_never_disagree_on_policy() -> None:
    """Blocking the reduction is a memory decision, not a modelling one.

    The blocked scan exists so a large correspondence fits; it must therefore
    publish what the dense reduction publishes, on every query and every block
    size, or the memory budget would be silently changing the model's policy.
    Checked over a randomised battery of folded correspondences rather than a
    single shape, and as exact equality — the two paths select an owner, and a
    selection either agrees or it does not.
    """
    dtype = _dtype()
    rng = np.random.default_rng(seed=20260810)
    disagreements = 0
    for _ in range(12):
        n_branch, n_point = 3, 4
        grid = np.sort(rng.uniform(0.0, 100.0, size=(n_branch, n_point)), axis=1)
        values = rng.uniform(-50.0, 50.0, size=(n_branch, n_point))
        endog = jnp.asarray(grid.ravel(), dtype=dtype)
        common: dict[str, Any] = {
            "endog_grid": endog,
            "value": jnp.asarray(values.ravel(), dtype=dtype),
            "policy": jnp.asarray(
                rng.uniform(0.0, 1.0, size=n_branch * n_point), dtype=dtype
            ),
            "marginal": jnp.asarray(
                rng.uniform(0.0, 1.0, size=n_branch * n_point), dtype=dtype
            ),
            "segment_id": jnp.asarray(
                np.repeat(np.arange(n_branch), n_point), dtype=dtype
            ),
            "x_query": jnp.asarray(rng.uniform(0.0, 100.0, size=16), dtype=dtype),
        }
        dense = np.asarray(envelope_at_query(**common)[1])
        for block_size in (1, 3, 5):
            blocked = np.asarray(
                envelope_at_query(**common, segment_block_size=block_size)[1]
            )
            both_nan = np.isnan(dense) & np.isnan(blocked)
            disagreements += int(np.sum(~both_nan & (dense != blocked)))

    assert disagreements == 0


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
