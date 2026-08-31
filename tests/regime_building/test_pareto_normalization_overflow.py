"""Pointwise Pareto normalization survives weights near the format's maximum.

A weight vector is admissible when every entry is finite and non-negative and
the total is strictly positive. Nothing in that contract bounds how large a
finite weight may be, so normalization has to stay finite for any admissible
declaration — including one whose raw total is not representable in the
working format.
"""

import itertools

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.regime_building.collective import build_pareto_weights
from lcm.collective import ParetoObjective
from tests.conftest import DECIMAL_PRECISION


def _working_float_max() -> float:
    """Return the largest finite value of the precision the suite runs at."""
    return float(jnp.finfo(jnp.zeros(()).dtype).max)


def _normalized(declared: dict[str, float]) -> dict[str, float]:
    """Compute the normalized share of each stakeholder in `declared`."""
    built = build_pareto_weights(
        objective=ParetoObjective(weights=dict(declared)),
        stakeholders=tuple(declared),
        state_names=frozenset(),
    )
    return {name: float(share) for name, share in built.compute().items()}


def _equal_weights_at(value: float) -> dict[str, float]:
    """Compute normalized weights for two stakeholders declared equal at `value`."""
    return _normalized({"f": value, "m": value})


def test_weights_at_the_format_maximum_normalize_to_one_half():
    """Two equal admissible weights share the household equally, however large."""
    weights = _equal_weights_at(_working_float_max())

    aaae(weights["f"], 0.5, decimal=DECIMAL_PRECISION)


def test_weights_at_the_format_maximum_select_the_intended_action():
    """The normalized weighting picks the action the intended weights pick.

    With payoffs `A = (1, 0)` and `B = (0, 2)`, weights of one half each rank
    `B` above `A`. Weights that collapse to zero tie the two and select `A`,
    which is the wrong household choice for a declaration that never said the
    stakeholders were worthless.
    """
    weights = _equal_weights_at(_working_float_max())
    payoffs = jnp.asarray([[1.0, 0.0], [0.0, 2.0]])
    stacked = jnp.asarray([weights["f"], weights["m"]])

    chosen = int(jnp.argmax(jnp.sum(payoffs * stacked, axis=-1)))

    assert chosen == 1


def test_ordinary_weights_are_unchanged_by_the_scale_safe_path():
    """Rescaling must not move a declaration the old path already handled."""
    weights = _equal_weights_at(2.0)

    aaae(weights["f"], 0.5, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(("name", "expected"), [("f", 0.75), ("m", 0.25)])
def test_unequal_weights_keep_their_declared_ratio(name, expected):
    """Normalization preserves the ratio the modeller declared."""
    weights = _normalized({"f": 3.0, "m": 1.0})

    aaae(weights[name], expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("n_stakeholders", range(2, 9))
def test_equal_weights_at_the_format_maximum_share_evenly_at_any_household_size(
    n_stakeholders,
):
    """`n` stakeholders declared equal each take `1/n`, at any household size.

    The raw total leaves the working format from two stakeholders upward, so
    every size in this range exercises the overflow, not just the pair.
    """
    names = [f"s{index}" for index in range(n_stakeholders)]
    weights = _normalized(dict.fromkeys(names, _working_float_max()))

    aaae(
        [weights[name] for name in names],
        [1.0 / n_stakeholders] * n_stakeholders,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_relabelling_the_household_leaves_every_share_untouched(order):
    """A stakeholder's share depends on her declared weight, not on her position.

    Stakeholder names are economically inert, so declaring the same three
    weights in a different order has to return the same three shares.
    """
    maximum = _working_float_max()
    declared = {"f": maximum, "m": maximum / 2, "k": maximum / 4}
    expected = _normalized(declared)

    reordered = _normalized({name: declared[name] for name in [*declared][:]})
    permuted_names = [[*declared][index] for index in order]
    permuted = _normalized({name: declared[name] for name in permuted_names})

    assert reordered == expected
    aaae(
        [permuted[name] for name in declared],
        [expected[name] for name in declared],
        decimal=DECIMAL_PRECISION,
    )


def test_weights_far_below_overflow_are_left_exactly_where_they_were():
    """A declaration whose raw total is representable keeps its plain ratio.

    Weights of `max/8` and `max/16` sum to a finite total, so rescaling must
    reproduce the ratio a plain division already gave: two thirds and one third.
    """
    maximum = _working_float_max()
    weights = _normalized({"f": maximum / 8, "m": maximum / 16})

    aaae(
        [weights["f"], weights["m"]],
        [2.0 / 3.0, 1.0 / 3.0],
        decimal=DECIMAL_PRECISION,
    )
