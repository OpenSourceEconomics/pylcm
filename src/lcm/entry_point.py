from functools import partial

from lcm.discrete_emax import get_emax_calculator
from lcm.model_functions import get_utility_and_feasibility_function
from lcm.process_model import process_model
from lcm.solve_brute import solve
from lcm.state_space import create_state_choice_space


def get_lcm_function(model, targets="solve", interpolation_options=None):
    """Entry point for users to get high level functions generated by lcm.

    Advanced users might want to use lower level functions instead, but can read the
    source code of this function to see how the lower level components are meant to be
    used.

    Notes:
    -----
    - Further targets could be "likelihood" or "simulate"
    - We might need additional arguments such as solver_options that we want to take
      separate from a model specification.
    - create_params needs to work with a processed_model instead of a user model.
    - currently all the preparations are hardcoded to generate the arguments needed
      by solve_brute. In the long run, this needs to inspect the signature of the
      solver, or generate only what is needed using dags.
    - there is a hack to make the state_indexers empty in the last period which needs
      to be replaced by a better solution, when we want to allow for bequest motives.

    Args:
        model (dict): Model specification.
        targets (str or iterable): The requested function types. Currently only
            "solve" is supported.
        interpolation_options (dict): Dictionary of keyword arguments for interpolation
            via map_coordinates.

    Returns:
        callable: A function that takes params and returns the requested targets.
        dict: A parameter dict where all parameter values are initialized to NaN.

    """
    # ==================================================================================
    # preparations
    # ==================================================================================
    if targets != "solve":
        raise NotImplementedError

    _mod = process_model(model)
    last_period = _mod.n_periods - 1
    interpolation_options = (
        {} if interpolation_options is None else interpolation_options
    )

    # ==================================================================================
    # create list of continuous choice grids
    # ==================================================================================
    # for now they are the same in all periods but this will change.
    _subset = _mod.variable_info.query("is_continuous & is_choice").index.tolist()
    _choice_grids = {k: _mod.grids[k] for k in _subset}
    continuous_choice_grids = [_choice_grids] * _mod.n_periods

    # ==================================================================================
    # create list of emax_calculators
    # ==================================================================================
    # for now they are the same in all periods but this will change.

    _shock_type = _mod.shocks.get("additive_utility_shock", None)

    calculator = get_emax_calculator(
        shock_type=_shock_type,
        variable_info=_mod.variable_info,
    )
    emax_calculators = [calculator] * _mod.n_periods

    # ==================================================================================
    # Initialize other argument lists
    # ==================================================================================
    state_choice_spaces = []
    state_indexers = []
    utility_and_feasibility_functions = []
    choice_segments = []

    for period in range(_mod.n_periods):
        is_last_period = period == last_period
        # ==============================================================================
        # call state space creation function, append trivial items to their lists
        # ==============================================================================
        sc_space, space_info, state_indexer, segments = create_state_choice_space(
            model=_mod,
            period=period,
        )
        state_choice_spaces.append(sc_space)
        choice_segments.append(segments)

        if is_last_period:
            state_indexers.append({})
        else:
            state_indexers.append(state_indexer)

        # ==============================================================================
        # create the utility and feasibility functions and append to their list
        # ==============================================================================
        u_and_f = get_utility_and_feasibility_function(
            model=_mod,
            space_info=space_info,
            data_name="vf_arr",
            interpolation_options=interpolation_options,
            is_last_period=is_last_period,
        )
        utility_and_feasibility_functions.append(u_and_f)

    # ==================================================================================
    # select requested solver and partial arguments into it
    # ==================================================================================
    solve_model = partial(
        solve,
        state_choice_spaces=state_choice_spaces,
        state_indexers=state_indexers,
        continuous_choice_grids=continuous_choice_grids,
        utility_and_feasibility_functions=utility_and_feasibility_functions,
        emax_calculators=emax_calculators,
        choice_segments=choice_segments,
    )

    return solve_model, _mod.params
