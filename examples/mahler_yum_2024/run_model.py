from create_model_inputs import create_inputs
from Mahler_Yum_2024 import MAHLER_YUM_MODEL, START_PARAMS

seed = 32
params, initial_states, initial_regimes = create_inputs(
    seed=seed, n=5000, **START_PARAMS
)
simulation_result = MAHLER_YUM_MODEL.solve_and_simulate(
    params={"alive": params, "dead": params},
    initial_states=initial_states,
    initial_regimes=initial_regimes,
    seed=seed,
)
