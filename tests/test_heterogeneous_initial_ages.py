"""Test heterogeneous initial ages in simulation."""

import jax.numpy as jnp

from tests.test_models.deterministic.base import RegimeId, get_model, get_params


def test_simulation_with_heterogeneous_initial_ages():
    """Subjects should be able to start simulation at different ages.

    The simulation loop iterates all periods starting from 0 for every subject.
    Even though `initial_states["age"]` can specify per-subject starting ages,
    the simulation does not yet use them to offset each subject's timeline.
    """
    n_periods = 5
    model = get_model(n_periods)
    params = get_params(n_periods)

    # Subject 0 starts at age 40, subject 1 starts at age 60
    result = model.solve_and_simulate(
        params,
        initial_conditions={
            "age": jnp.array([40.0, 60.0]),
            "wealth": jnp.array([50.0, 50.0]),
            "regime_id": jnp.array([RegimeId.working_life] * 2),
        },
    )
    df = result.to_dataframe()

    # Subject 1 should not have data for ages before their starting age
    subject_1_min_age = df.loc[df["subject_id"] == 1, "age"].min()
    assert subject_1_min_age == 60.0
