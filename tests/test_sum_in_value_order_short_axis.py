"""A permutable axis of one or two entries needs no canonicalising sort.

`sum_in_value_order` exists because floating-point addition is not associative,
so a reduction over entries keyed by an economically inert identifier — a regime
label, a stakeholder name — would otherwise depend on that identifier. Sorting
the contributions first makes the reduction order a function of their values.

Associativity is what fails; commutativity does not. A one-entry sum has no
order to canonicalise, and a two-entry sum is a single addition, which IEEE-754
evaluates identically in either operand order. So on those axes the sort cannot
change any bit, and the ordinary singleton path — one or two reachable regimes —
should not pay for it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.zero_safe import sum_in_value_order


def _sort_primitive_count(*, n_entries: int) -> int:
    """Number of sort primitives the jaxpr carries for an axis of `n_entries`."""
    jaxpr = jax.make_jaxpr(lambda values: sum_in_value_order(values=values, axis=0))(
        jnp.zeros(n_entries)
    )
    return str(jaxpr).count("sort")


@pytest.mark.parametrize("n_entries", [1, 2])
def test_a_short_permutable_axis_carries_no_sort(n_entries: int):
    """One or two entries reduce without a sort primitive in the jaxpr."""
    assert _sort_primitive_count(n_entries=n_entries) == 0


def test_a_longer_permutable_axis_still_sorts():
    """Three entries keep the sort — the check above is not vacuous."""
    assert _sort_primitive_count(n_entries=3) > 0


@pytest.mark.parametrize(
    "pair",
    [
        (1.0, 2.0),
        (1e300, -1e300),
        (0.0, -0.0),
        (jnp.finfo(jnp.float32).tiny, 1.0),
        (float("inf"), -1.0),
    ],
)
def test_a_two_entry_sum_is_identical_in_either_order(pair: tuple[float, float]):
    """Swapping two contributions cannot move a bit, so the sort buys nothing."""
    first, second = pair
    forward = sum_in_value_order(values=jnp.asarray([first, second]), axis=0)
    reverse = sum_in_value_order(values=jnp.asarray([second, first]), axis=0)
    assert np.asarray(forward).tobytes() == np.asarray(reverse).tobytes()


def test_a_short_axis_agrees_with_the_sorted_reduction():
    """The short-circuit publishes exactly what the sorted reduction published."""
    values = jnp.asarray([[3.0, -1.0], [2.5, 7.25]])
    got = sum_in_value_order(values=values, axis=0)
    expected = jnp.sum(jnp.sort(values, axis=0), axis=0)
    assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("entries", [(jnp.nan, 1.0), (1.0, jnp.nan)])
def test_a_nan_entry_poisons_the_sum_from_either_side(entries: tuple[float, float]):
    """A NaN reaches the result whichever side of the two-entry axis it sits on."""
    assert bool(jnp.isnan(sum_in_value_order(values=jnp.asarray(entries), axis=0)))
