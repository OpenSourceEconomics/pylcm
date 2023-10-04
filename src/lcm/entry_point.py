import functools
from functools import partial

import jax.numpy as jnp

from lcm.argmax import argmax
from lcm.discrete_emax import get_emax_calculator
from lcm.dispatchers import productmap
from lcm.model_functions import (
    get_utility_and_feasibility_function,
)
from lcm.process_model import process_model
from lcm.simulate import simulate
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
        model (dict): User model specification.
        targets (str or iterable): The requested function types. Currently only
            "solve", "simulate" and "solve_and_simulate" are supported.
        interpolation_options (dict): Dictionary of keyword arguments for interpolation
            via map_coordinates.

    Returns:
        callable: A function that takes params and returns the requested targets.
        dict: A parameter dict where all parameter values are initialized to NaN.

    """
    # ==================================================================================
    # preparations
    # ==================================================================================
    if targets not in {"solve", "simulate", "solve_and_simulate"}:
        raise NotImplementedError

    _mod = process_model(user_model=model)
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
    # Initialize other argument lists
    # ==================================================================================
    state_choice_spaces = []
    state_indexers = []
    space_infos = []
    compute_ccv_functions = []
    compute_ccv_policy_functions = []
    choice_segments = []
    emax_calculators = []

    # ==================================================================================
    # Create stace choice space for each period
    # ==================================================================================
    for period in range(_mod.n_periods):
        is_last_period = period == last_period

        # call state space creation function, append trivial items to their lists
        # ==============================================================================
        sc_space, space_info, state_indexer, segments = create_state_choice_space(
            model=_mod,
            period=period,
            is_last_period=is_last_period,
            jit_filter=False,
        )

        state_choice_spaces.append(sc_space)
        choice_segments.append(segments)

        if is_last_period:
            state_indexers.append({})
        else:
            state_indexers.append(state_indexer)

        space_infos.append(space_info)

    # ==================================================================================
    # Shift space info (in period t we require the space info of period t+1)
    # ==================================================================================
    space_infos = space_infos[1:] + [{}]

    # ==================================================================================
    # Create model functions
    # ==================================================================================
    for period in range(_mod.n_periods):
        is_last_period = period == last_period

        # create the compute conditional continuation value functions and append to list
        # ==============================================================================
        u_and_f = get_utility_and_feasibility_function(
            model=_mod,
            space_info=space_infos[period],
            data_name="vf_arr",
            interpolation_options=interpolation_options,
            is_last_period=is_last_period,
        )
        compute_ccv = create_compute_conditional_continuation_value(
            utility_and_feasibility=u_and_f,
            continuous_choice_variables=list(_choice_grids),
        )
        compute_ccv_functions.append(compute_ccv)

        compute_ccv_argmax = create_compute_conditional_continuation_policy(
            utility_and_feasibility=u_and_f,
            continuous_choice_variables=list(_choice_grids),
        )
        compute_ccv_policy_functions.append(compute_ccv_argmax)

        # create list of emax_calculators
        # ==============================================================================
        _shock_type = _mod.shocks.get("additive_utility_shock", None)

        calculator = get_emax_calculator(
            shock_type=_shock_type,
            variable_info=_mod.variable_info,
            is_last_period=is_last_period,
        )
        emax_calculators.append(
            partial(
                calculator,
                choice_segments=choice_segments[period],
                params=_mod.params,
            ),
        )

    # ==================================================================================
    # select requested solver and partial arguments into it
    # ==================================================================================
    solve_model = partial(
        solve,
        state_choice_spaces=state_choice_spaces,
        state_indexers=state_indexers,
        continuous_choice_grids=continuous_choice_grids,
        compute_ccv_functions=compute_ccv_functions,
        emax_calculators=emax_calculators,
    )

    simulate_model = partial(
        simulate,
        state_indexers=state_indexers,
        continuous_choice_grids=continuous_choice_grids,
        compute_ccv_policy_functions=compute_ccv_policy_functions,
        model=_mod,
    )

    if targets == "solve":
        _target = solve_model
    elif targets == "simulate":
        _target = simulate_model
    elif targets == "solve_and_simulate":
        _target = partial(simulate_model, solve_model=solve_model)

    return _target, _mod.params


def create_compute_conditional_continuation_value(
    utility_and_feasibility,
    continuous_choice_variables,
):
    """Create a function that computes the conditional continuation value.

    Note:
    -----
    This function solves the continuous choice problem conditional on a state-
    (discrete-)choice combination.

    Args:
        utility_and_feasibility (callable): A function that takes a state-choice
            combination and return the utility of that combination (float) and whether
            the state-choice combination is feasible (bool).
        continuous_choice_variables (list): List of choice variable names that are
            continuous.

    Returns:
        callable: A function that takes a state-choice combination and returns the
            conditional continuation value over the continuous choices.

    """
    u_and_f_mapped_over_cont_choices = productmap(
        func=utility_and_feasibility,
        variables=continuous_choice_variables,
    )

    @functools.wraps(u_and_f_mapped_over_cont_choices)
    def compute_ccv(*args, **kwargs):
        u, f = u_and_f_mapped_over_cont_choices(*args, **kwargs)
        return u.max(where=f, initial=-jnp.inf)

    return compute_ccv


def create_compute_conditional_continuation_policy(
    utility_and_feasibility,
    continuous_choice_variables,
):
    """Create a function that computes the conditional continuation policy.

    Note:
    -----
    This function solves the continuous choice problem conditional on a state-
    (discrete-)choice combination.

    Args:
        utility_and_feasibility (callable): A function that takes a state-choice
            combination and return the utility of that combination (float) and whether
            the state-choice combination is feasible (bool).
        continuous_choice_variables (list): List of choice variable names that are
            continuous.

    Returns:
        callable: A function that takes a state-choice combination and returns the
            conditional continuation value over the continuous choices, and the index
            that maximizes the conditional continuation value.

    """
    u_and_f_mapped_over_cont_choices = productmap(
        func=utility_and_feasibility,
        variables=continuous_choice_variables,
    )

    @functools.wraps(u_and_f_mapped_over_cont_choices)
    def compute_ccv_policy(*args, **kwargs):
        u, f = u_and_f_mapped_over_cont_choices(*args, **kwargs)
        _argmax, _max = argmax(u, where=f, initial=-jnp.inf)
        return _argmax, _max

    return compute_ccv_policy
