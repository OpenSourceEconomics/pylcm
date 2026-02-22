"""Reproducer: heterogeneous initial ages in simulation."""

import jax.numpy as jnp
import pytest

from tests.test_models.deterministic.base import get_model, get_params


@pytest.mark.xfail(
    reason=(
        "Simulation does not support heterogeneous initial ages. All subjects "
        "are forced to start at period 0 (ages.minimum). The simulation loop "
        "does not yet use the per-subject age from initial_states."
    ),
    strict=True,
)
def test_simulation_with_heterogeneous_initial_ages():
    """Subjects should be able to start simulation at different ages.

    The simulation loop iterates all periods starting from 0 for every subject.
    Even though `initial_states["age"]` can specify per-subject starting ages,
    the simulation does not yet use them to offset each subject's timeline.
    """
    n_periods = 5
    model = get_model(n_periods)
    params = get_params(n_periods)

    # Subject 0 starts at age 0, subject 1 starts at age 2
    result = model.solve_and_simulate(
        params,
        initial_states={
            "age": jnp.array([0.0, 2.0]),
            "wealth": jnp.array([50.0, 50.0]),
        },
        initial_regimes=["working", "working"],
    )
    df = result.to_dataframe()

    # Subject 1 should not have data for ages before their starting age
    subject_1_min_age = df.loc[df["_subject_id"] == 1, "age"].min()
    assert subject_1_min_age == 2.0
