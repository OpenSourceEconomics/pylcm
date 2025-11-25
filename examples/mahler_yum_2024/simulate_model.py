from model_spec import MAHLER_YUM_MODEL, START_PARAMS, create_inputs

seed: int = 32
params, initial_states, initial_regimes = create_inputs(
    seed=seed, n=1_000, **START_PARAMS
)
simulation_result = MAHLER_YUM_MODEL.solve_and_simulate(
    params={"alive": params, "dead": params},
    initial_states=initial_states,
    initial_regimes=initial_regimes,
    seed=seed,
)
