import jax.numpy as jnp
from cons_sav import CONS_SAV_MODEL, PARAMS

# number of simulated subjects
n = 1000

simulation_result = CONS_SAV_MODEL.solve_and_simulate(
    params=PARAMS,
    initial_regimes=["cons_sav_model"] * n,
    initial_states={
        "cons_sav_model": {"wealth": jnp.full(n, 1), "health": jnp.full(n, 1)}
    },
)
