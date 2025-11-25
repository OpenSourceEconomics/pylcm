from model_spec import MAHLER_YUM_MODEL, START_PARAMS, create_inputs

params, initial_states, initial_regimes = create_inputs(
    seed=7235, n_simulation_subjects=1_000, **START_PARAMS
)

simulation_result = MAHLER_YUM_MODEL.solve_and_simulate(
    params={"alive": params, "dead": params},
    initial_states=initial_states,
    initial_regimes=initial_regimes,
    seed=8295,
)
