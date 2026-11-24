"""The collective example is small, executable, and economically checkable."""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm.exits import Unilateral
from lcm.solver_api import ResultRetention
from lcm_examples.collective_regimes import (
    get_dissolution_model,
    get_params,
    get_shared_decision_model,
)
from tests.conftest import DECIMAL_PRECISION


def test_shared_decision_values_are_the_members_values_at_one_argmax() -> None:
    """Both member values are read at the household's shared work choice."""
    model = get_shared_decision_model()

    solution = model.solve(params=get_params(), log_level="debug")

    aaae(
        solution.values[0]["couple"],
        np.array([[46.0, 92.0], [78.0, 156.0]]),
        decimal=DECIMAL_PRECISION,
    )


def test_participation_constraints_dissolve_only_the_middle_wage_cell() -> None:
    """The source prices each actual route; dissolution pays the exit's cost."""
    model = get_dissolution_model()
    solution = model.solve(params=get_params(), log_level="debug")

    # Work keeps wages one and three feasible for both partners. Wage two has
    # no participating action, so the household's own exit moves both partners
    # to their single regimes, (5.5, 1), and pays the divorce cost (0.5, 0.2):
    # (5.0, 0.8).
    aaae(
        solution.values[0]["married"],
        0.95 * np.array([[2.0, 1.0], [5.0, 0.8], [6.0, 3.0]]),
        decimal=DECIMAL_PRECISION,
    )


def test_dissolution_routes_the_simulated_person_to_their_single_regime() -> None:
    """The female cohort takes her own exit target when no household action is
    feasible."""
    model = get_dissolution_model()
    solution = model.solve(
        params=get_params(),
        log_level="debug",
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
        "own_member": jnp.full(
            3,
            model.member_names_to_ids["f"],
            dtype=jnp.int32,
        ),
    }

    result = model.simulate(
        params=get_params(),
        initial_conditions=initial_conditions,
        solution=solution,
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


def test_the_household_exit_routes_without_retained_dissolution_artifacts() -> None:
    """The household's own exit routes each original role directly."""
    model = get_dissolution_model()
    params = get_params()
    solution = model.solve(
        params=params, log_level="off", retention=ResultRetention.VALUES
    )
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wage": jnp.asarray([1.0, 2.0, 3.0]),
            "age": jnp.zeros(3),
            "regime_id": jnp.full(
                3, model.regime_names_to_ids["married"], dtype=jnp.int32
            ),
            "own_member": jnp.full(3, model.member_names_to_ids["f"], dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    ).to_dataframe()
    arrived = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(
        arrived["regime_name"],
        ["married_with_participation", "single_f", "married_with_participation"],
    )


def test_dissolution_follows_the_eckstein_keane_lifshitz_rule() -> None:
    """Either partner can leave, and a partner exactly indifferent leaves."""
    model = get_dissolution_model()
    declaration = model.user_regimes["married_with_participation"].exit
    assert getattr(declaration, "rule", None) == Unilateral(tie="leave")
