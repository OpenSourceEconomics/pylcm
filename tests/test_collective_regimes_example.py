"""The collective example is small, executable, and economically checkable."""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm_examples.collective_regimes import (
    get_dissolution_model,
    get_params,
    get_shared_decision_model,
)
from tests.conftest import DECIMAL_PRECISION


def test_shared_decision_values_are_the_stakeholders_values_at_one_argmax():
    """Both stakeholder values are read at the household's shared work choice."""
    model = get_shared_decision_model()

    solution = model.solve(params=get_params(), log_level="debug")

    aaae(
        solution[0]["couple"],
        np.array([[46.0, 92.0], [78.0, 156.0]]),
        decimal=DECIMAL_PRECISION,
    )


def test_participation_constraints_dissolve_only_the_middle_wage_cell():
    """Only wage two leaves the couple without a jointly feasible action."""
    model = get_dissolution_model()

    _, dissolution_flags = model.solve(
        params=get_params(),
        log_level="debug",
        return_dissolution_flags=True,
    )

    np.testing.assert_array_equal(
        dissolution_flags[1]["married_with_participation"],
        np.array([False, True, False]),
    )


def test_dissolution_routes_the_simulated_person_to_their_single_regime():
    """The female cohort follows its own fallback route when participation fails."""
    model = get_dissolution_model()
    solution, dissolution_flags = model.solve(
        params=get_params(),
        log_level="debug",
        return_dissolution_flags=True,
    )
    initial_conditions = {
        "wage": jnp.array([1.0, 2.0, 3.0]),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(
            3,
            model.regime_names_to_ids["married"],
            dtype=jnp.int32,
        ),
        # Every subject starts in the collective regime, so each one occupies a
        # household role from the first period on, and that role decides which
        # fallback a dissolving row takes.
        "own_stakeholder": jnp.full(
            3,
            model.stakeholder_names_to_ids["f"],
            dtype=jnp.int32,
        ),
    }

    result = model.simulate(
        params=get_params(),
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        log_level="debug",
        seed=0,
    )

    np.testing.assert_array_equal(
        result.raw_results["married_with_participation"][1].in_regime,
        np.array([True, False, True]),
    )
    np.testing.assert_array_equal(
        result.raw_results["single_f"][1].in_regime,
        np.array([False, True, False]),
    )
