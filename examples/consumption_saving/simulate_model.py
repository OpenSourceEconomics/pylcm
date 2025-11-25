import jax.numpy as jnp
from model_spec import CONSUMPTION_SAVING_MODEL, PARAMS

# number of simulated subjects
n = 1_000

simulation_result = CONSUMPTION_SAVING_MODEL.solve_and_simulate(
    params=PARAMS,
    initial_regimes=["consumption_saving_regime"] * n,
    initial_states={
        "consumption_saving_regime": {
            "wealth": jnp.full(n, 1),
            "health": jnp.full(n, 1),
        }
    },
)
