import jax.numpy as jnp
from model_spec import CONSUMPTION_SAVING_MODEL, PARAMS

# number of simulated subjects
n = 1_000

simulation_result = CONSUMPTION_SAVING_MODEL.solve_and_simulate(
    params=PARAMS,
    initial_regimes=["cons_sav_model"] * n,
    initial_states={
        "cons_sav_model": {"wealth": jnp.full(n, 1), "health": jnp.full(n, 1)}
    },
)
