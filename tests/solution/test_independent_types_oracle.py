"""The fixed-type cake-eating model against its brute-force oracle.

Each preference type is an independent dynamic program. The engine's values
and simulated choices must match plain enumeration of every type on its own,
and changing one type's parameters must leave every other type's values
bitwise unchanged.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from tests.conftest import DECIMAL_PRECISION
from tests.test_models import independent_types

_N_TYPES = len(independent_types.TYPE_PARAMS["weight"])


def _solved_values() -> Mapping:
    model = independent_types.get_model()
    return model.solve(params=independent_types.get_params(), log_level="off").values


@pytest.mark.parametrize("period", range(len(independent_types.AGES.exact_values)))
def test_values_match_the_enumerated_oracle(period: int) -> None:
    """Every type's value at every wealth node equals enumeration's."""
    values = _solved_values()
    oracle, _ = independent_types.solve_by_enumeration()

    (published,) = values[period].values()
    aaae(np.asarray(published), np.asarray(oracle[period]), decimal=DECIMAL_PRECISION)


def test_simulated_consumption_follows_the_enumerated_policy() -> None:
    """Starting from every (type, wealth) pair, each period's choice is the oracle's."""
    model = independent_types.get_model()
    n_wealth = independent_types.N_WEALTH_POINTS
    types = np.repeat(np.arange(_N_TYPES), n_wealth)
    wealth = np.tile(np.arange(n_wealth, dtype=float), _N_TYPES)
    result = model.simulate(
        params=independent_types.get_params(),
        initial_conditions={
            "regime_id": jnp.full(types.size, independent_types.RegimeId.working),
            "age": jnp.zeros(types.size),
            "wealth": jnp.asarray(wealth),
            "pref_type": jnp.asarray(types, dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    )
    _, policies = independent_types.solve_by_enumeration()
    frame = result.to_dataframe()
    working = frame[frame["regime_name"] == "working"]

    expected = [
        float(policies[int(period)][int(types[subject])][int(wealth_now)])
        for period, subject, wealth_now in zip(
            working["period"].to_numpy(dtype=int),
            working["subject_id"].to_numpy(dtype=int),
            working["wealth"].to_numpy(dtype=float),
            strict=True,
        )
    ]
    assert working["consumption"].tolist() == expected


@pytest.mark.parametrize("unchanged_type", range(_N_TYPES - 1))
def test_changing_one_type_leaves_other_types_bitwise_unchanged(
    unchanged_type: int,
) -> None:
    """Scaling the last type's weights moves no other type's value by a bit."""
    model = independent_types.get_model()
    base = model.solve(params=independent_types.get_params(), log_level="off").values
    changed_params = {
        name: tuple(
            value * (1.25 if code == _N_TYPES - 1 else 1.0)
            for code, value in enumerate(values)
        )
        for name, values in independent_types.TYPE_PARAMS.items()
    }
    changed = model.solve(
        params=independent_types.get_params(type_params=changed_params),
        log_level="off",
    ).values

    assert all(
        np.array_equal(
            np.asarray(base[period][regime])[unchanged_type],
            np.asarray(changed[period][regime])[unchanged_type],
        )
        for period in base
        for regime in base[period]
    )


def test_changing_the_last_type_moves_its_own_values() -> None:
    """The control for the bitwise check: the changed type's values do move."""
    model = independent_types.get_model()
    base = model.solve(params=independent_types.get_params(), log_level="off").values
    changed_params = {
        name: (*values[:-1], values[-1] * 1.25)
        for name, values in independent_types.TYPE_PARAMS.items()
    }
    changed = model.solve(
        params=independent_types.get_params(type_params=changed_params),
        log_level="off",
    ).values

    assert not np.array_equal(
        np.asarray(base[0]["working"])[-1], np.asarray(changed[0]["working"])[-1]
    )


def test_the_oracle_comparison_rejects_a_different_discount_factor() -> None:
    """The control for the oracle check: a perturbed oracle is told apart."""
    values = _solved_values()
    perturbed, _ = independent_types.solve_by_enumeration(discount_factor=0.89)

    with pytest.raises(AssertionError):
        aaae(
            np.asarray(values[0]["working"]),
            np.asarray(perturbed[0]),
            decimal=DECIMAL_PRECISION,
        )
