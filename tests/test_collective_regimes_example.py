"""The collective example is small, executable, and economically checkable."""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm.exits import RuleResult
from lcm.solver_api import ResultRetention
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
        solution.values[0]["couple"],
        np.array([[46.0, 92.0], [78.0, 156.0]]),
        decimal=DECIMAL_PRECISION,
    )


def test_participation_constraints_dissolve_only_the_middle_wage_cell():
    """The source prices each actual route; dissolution pays the full outside value."""
    model = get_dissolution_model()
    solution = model.solve(params=get_params(), log_level="debug")

    # Work keeps wages one and three feasible for both partners. Wage two
    # has no participating action and receives (5.5, 1), without subtracting
    # the (0.5, 0.2) participation slack from either payoff.
    aaae(
        solution.values[0]["married"],
        0.95 * np.array([[2.0, 1.0], [5.5, 1.0], [6.0, 3.0]]),
        decimal=DECIMAL_PRECISION,
    )


def test_dissolution_routes_the_simulated_person_to_their_single_regime():
    """The female cohort follows its own fallback route when participation fails."""
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
        "own_stakeholder": jnp.full(
            3,
            model.stakeholder_names_to_ids["f"],
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


def test_arrival_support_routes_without_retained_dissolution_artifacts():
    """The arrival policy's feasible set routes each original role directly."""
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
            "own_stakeholder": jnp.full(
                3, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        },
        seed=0,
        log_level="off",
    ).to_dataframe()
    arrived = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(
        arrived["regime_name"],
        ["married_with_participation", "single_f", "married_with_participation"],
    )


def test_participation_rule_preserves_a_zero_weight_members_veto():
    """An infeasible or unwilling partner cannot be ignored in a shared choice."""
    model = get_dissolution_model()
    rule = model.user_regimes["married_with_participation"].exits["dissolve"].rule
    assert callable(rule)
    cases = [
        ([2.0, 2.0], [1.0, 1.0], {"f": 0.5, "m": 0.5}, 0),
        ([0.5, 2.0], [100.0, 1.0], {"f": 0.0, "m": 1.0}, 1),
        ([-np.inf, 2.0], [-np.inf, 1.0], {"f": 0.0, "m": 1.0}, 1),
        ([0.0, 0.0], [1.0, 1.0], {"f": 0.5, "m": 0.5}, 2),
    ]
    for q_f, q_m, weights, expected in cases:
        selected = rule(
            Q_f=jnp.asarray(q_f),
            Q_m=jnp.asarray(q_m),
            exit_payoffs={"dissolve": {"f": jnp.asarray(1.5), "m": jnp.asarray(1.0)}},
            alternative_ids=jnp.arange(3, dtype=jnp.int32),
            pareto_weights={
                name: jnp.asarray(value) for name, value in weights.items()
            },
        )
        assert isinstance(selected, RuleResult)
        assert int(selected.action) == expected
