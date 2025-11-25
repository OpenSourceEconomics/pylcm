import jax.numpy as jnp
from model_spec import CONSUMPTION_SAVING_MODEL, PARAMS

n_simulation_subjects = 1_000

simulation_result = CONSUMPTION_SAVING_MODEL.solve_and_simulate(
    params=PARAMS,
    initial_regimes=["consumption_saving_regime"] * n_simulation_subjects,
    initial_states={
        "consumption_saving_regime": {
            "wealth": jnp.full(n_simulation_subjects, 1),
            "health": jnp.full(n_simulation_subjects, 1),
        }
    },
)
