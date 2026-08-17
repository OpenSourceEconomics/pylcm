"""CPU-safe semantic checks for the Mahler-Yum retirement split."""

import jax.numpy as jnp
import numpy as np

from lcm_examples.mahler_yum_2024 import (
    RETIREMENT_REGIME,
    START_PARAMS,
    WORKING_REGIME,
    ages,
    create_inputs,
    net_income,
    retirement_is_active,
    retirement_net_income,
    retirement_period,
    retirement_to_dead_probability,
    retirement_to_retirement_probability,
    retirement_utility,
    utility,
    working_is_active,
    working_to_dead_probability,
    working_to_retirement_probability,
    working_to_working_probability,
)


def test_retirement_split_preserves_payoffs():
    """Dropping work variables leaves retirement income and utility unchanged."""
    consumption_utility = jnp.array([10.0, 12.0])
    effort_cost = jnp.array([1.0, 2.0])
    adjustment_cost = jnp.array([0.5, 0.25])

    source_utility = utility(
        adjustment_cost_penalty=adjustment_cost,
        effort_cost=effort_cost,
        work_disutility=jnp.zeros(2),
        consumption_utility=consumption_utility,
    )
    split_utility = retirement_utility(
        adjustment_cost_penalty=adjustment_cost,
        effort_cost=effort_cost,
        consumption_utility=consumption_utility,
    )
    expected_utility = np.array([8.5, 9.75])

    pension = jnp.array([2.0, 3.5])
    source_income = net_income(
        benefits=jnp.zeros(2),
        taxed_income=jnp.zeros(2),
        pension=pension,
    )
    split_income = retirement_net_income(pension)

    np.testing.assert_allclose(source_utility, expected_utility)
    np.testing.assert_allclose(split_utility, expected_utility)
    np.testing.assert_allclose(source_income, pension)
    np.testing.assert_allclose(split_income, pension)


def test_retirement_split_routes_survival_mass_at_the_boundary():
    """The split relabels survivors at age 65 and preserves death probability."""
    transition_probs = jnp.full((retirement_period + 1, 2, 2), 0.8)
    education = jnp.asarray(1, dtype=jnp.int32)
    health = jnp.asarray(0, dtype=jnp.int32)
    before_period = jnp.asarray(retirement_period - 2, dtype=jnp.int32)
    boundary_period = jnp.asarray(retirement_period - 1, dtype=jnp.int32)
    retirement_period_scalar = jnp.asarray(retirement_period, dtype=jnp.int32)

    before_retirement = np.array(
        [
            working_to_working_probability(
                period=before_period,
                education=education,
                health=health,
                transition_probs=transition_probs,
            ),
            working_to_retirement_probability(
                period=before_period,
                education=education,
                health=health,
                transition_probs=transition_probs,
            ),
            working_to_dead_probability(
                period=before_period,
                education=education,
                health=health,
                transition_probs=transition_probs,
            ),
        ]
    )
    at_retirement = np.array(
        [
            working_to_working_probability(
                period=boundary_period,
                education=education,
                health=health,
                transition_probs=transition_probs,
            ),
            working_to_retirement_probability(
                period=boundary_period,
                education=education,
                health=health,
                transition_probs=transition_probs,
            ),
            working_to_dead_probability(
                period=boundary_period,
                education=education,
                health=health,
                transition_probs=transition_probs,
            ),
        ]
    )
    after_retirement = np.array(
        [
            retirement_to_retirement_probability(
                period=retirement_period_scalar,
                education=education,
                health=health,
                transition_probs=transition_probs,
            ),
            retirement_to_dead_probability(
                period=retirement_period_scalar,
                education=education,
                health=health,
                transition_probs=transition_probs,
            ),
        ]
    )

    np.testing.assert_allclose(before_retirement, [0.8, 0.0, 0.2])
    np.testing.assert_allclose(at_retirement, [0.0, 0.8, 0.2])
    np.testing.assert_allclose(after_retirement, [0.8, 0.2])


def test_retirement_split_partitions_living_ages_and_inputs():
    """The two living regimes cover each living age exactly once."""
    final_living_age = int(ages.values[-2])
    activity = np.array(
        [
            [
                working_is_active(int(age)),
                retirement_is_active(
                    age=int(age),
                    final_age_alive=final_living_age,
                ),
            ]
            for age in ages.values[:-1]
        ]
    )
    np.testing.assert_array_equal(activity.sum(axis=1), 1)

    removed_dimensions = {"labor_supply", "productivity", "productivity_shock"}
    working_dimensions = set(WORKING_REGIME.states) | set(WORKING_REGIME.actions)
    retirement_dimensions = set(RETIREMENT_REGIME.states) | set(
        RETIREMENT_REGIME.actions
    )
    assert removed_dimensions <= working_dimensions
    assert removed_dimensions.isdisjoint(retirement_dimensions)

    model_params, initial_conditions = create_inputs(
        seed=0,
        n_simulation_subjects=4,
        params=START_PARAMS,
    )
    assert set(model_params) == {"working", "retirement"}
    assert initial_conditions["regime_name"].eq("working").all()
