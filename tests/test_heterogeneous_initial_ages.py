"""Reproducer: heterogeneous initial ages in simulation."""

import jax.numpy as jnp
import pytest

from tests.test_models.deterministic.base import get_model, get_params


@pytest.mark.xfail(
    reason=(
        "Simulation does not support heterogeneous initial ages. All subjects "
        "are forced to start at period 0 (ages.minimum). The API has no "
        "parameter to specify per-subject starting ages/periods."
    ),
    strict=True,
)
def test_simulation_with_heterogeneous_initial_ages():
    """Subjects should be able to start simulation at different ages.

    The simulation loop iterates all periods starting from 0 for every subject.
    There is no API to specify that some subjects enter the model at later ages
    (e.g., age 2 instead of age 0). The solve_and_simulate method does not
    accept an initial_periods parameter.
    """
    n_periods = 5
    model = get_model(n_periods)
    params = get_params(n_periods)

    # Subject 0 starts at period 0 (age 0), subject 1 starts at period 2 (age 2)
    result = model.solve_and_simulate(
        params,
        initial_states={"wealth": jnp.array([50.0, 50.0])},
        initial_regimes=["working", "working"],
        initial_periods=jnp.array([0, 2]),
    )
    df = result.to_dataframe()

    # Subject 1 should not have data for ages before their starting age
    subject_1_min_age = df.loc[df["_subject_id"] == 1, "age"].min()
    assert subject_1_min_age == 2.0
