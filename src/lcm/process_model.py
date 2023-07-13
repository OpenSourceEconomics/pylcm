import functools
import inspect

import jax.numpy as jnp
import pandas as pd
from dags import get_ancestors
from dags.signature import with_signature

import lcm.grids as grids_module
from lcm.create_params import create_params
from lcm.interfaces import GridSpec, Model


def process_model(user_model):
    """Process the user model.

    This entails the following steps:

    - Set defaults where needed
    - Generate derived information
    - Check that the model specification is valid.

    Args:
        user_model (dict): The model as provided by the user.

    Returns:
        lcm.interfaces.Model: The processed model.

    """
    _params = create_params(user_model)
    _function_info = _get_function_info(user_model)
    _functions = _get_functions(
        user_model,
        function_info=_function_info,
        params=_params,
    )
    _variable_info = _get_variable_info(user_model, function_info=_function_info)
    _gridspecs = _get_gridspecs(user_model, variable_info=_variable_info)
    _grids = _get_grids(gridspecs=_gridspecs, variable_info=_variable_info)
    return Model(
        grids=_grids,
        gridspecs=_gridspecs,
        variable_info=_variable_info,
        functions=_functions,
        function_info=_function_info,
        params=_params,
        shocks=user_model.get("shocks", {}),
        n_periods=user_model["n_periods"],
    )


def _get_variable_info(user_model, function_info):
    """Derive information about all variables in the model.

    Args:
        user_model (dict): The model as provided by the user.
        function_info (pandas.DataFrame): A table with information about all
            functions in the model. The index contains the name of a function. The
            columns are booleans that are True if the function has the corresponding
            property. The columns are: is_filter, is_constraint, is_next.

    Returns:
        pandas.DataFrame: A table with information about all variables in the model.
            The index contains the name of a model variable. The columns are booleans
            that are True if the variable has the corresponding property. The columns
            are: is_state, is_choice, is_continuous, is_discrete, is_sparse, is_dense.

    """
    _variables = {
        **user_model["states"],
        **user_model["choices"],
    }

    info = pd.DataFrame(index=list(_variables))

    info["is_state"] = info.index.isin(user_model["states"])
    info["is_choice"] = ~info["is_state"]

    info["is_discrete"] = ["options" in spec for spec in _variables.values()]
    info["is_continuous"] = ~info["is_discrete"]

    _filtered_variables = set()
    _filter_names = function_info.query("is_filter").index.tolist()

    for name in _filter_names:
        _filtered_variables = _filtered_variables.union(
            get_ancestors(user_model["functions"], name),
        )

    info["is_sparse"] = [var in _filtered_variables for var in _variables]
    info["is_dense"] = ~info["is_sparse"]

    order = info.query("is_sparse & is_state").index.tolist()
    order += info.query("is_sparse & is_choice").index.tolist()
    order += info.query("is_dense & is_state").index.tolist()
    order += info.query("is_dense & is_choice").index.tolist()

    if set(order) != set(info.index):
        raise ValueError("Order and index do not match.")

    return info.loc[order]


def _get_gridspecs(user_model, variable_info):
    """Create a dictionary of grid specifications for each variable in the model.

    Args:
        user_model (dict): The model as provided by the user.
        variable_info (pandas.DataFrame): A table with information about all
            variables in the model. The index contains the name of a model variable.
            The columns are booleans that are True if the variable has the
            corresponding property. The columns are: is_state, is_choice, is_continuous,
            is_discrete, is_sparse, is_dense.

    Returns:
        dict: Dictionary containing all variables of the model. The keys are
            the names of the variables. The values describe which values the variable
            can take. For discrete variables these are the options. For continuous
            variables this is information about how to build the grids.

    """
    raw = {**user_model["states"], **user_model["choices"]}

    variables = {}
    for name, spec in raw.items():
        if "options" in spec:
            variables[name] = spec["options"]
        else:
            variables[name] = GridSpec(
                kind=spec["grid_type"],
                specs={k: v for k, v in spec.items() if k != "grid_type"},
            )

    order = variable_info.index.tolist()
    return {k: variables[k] for k in order}


def _get_grids(gridspecs, variable_info):
    """Create a dictionary of grids for each variable in the model.

    Args:
        gridspecs (dict): Dictionary containing all variables of the model. The keys
            are the names of the variables. The values describe which values the
            variable can take. For discrete variables these are the options. For
            continuous variables this is information about how to build the grids.
        variable_info (pandas.DataFrame): A table with information about all
            variables in the model. The index contains the name of a model variable.
            The columns are booleans that are True if the variable has the
            corresponding property. The columns are: is_state, is_choice, is_continuous,
            is_discrete, is_sparse, is_dense.

    Returns:
        dict: Dictionary containing all variables of the model. The keys are
            the names of the variables. The values are the grids.

    """
    grids = {}
    for name, grid_info in gridspecs.items():
        if variable_info.loc[name, "is_discrete"]:
            grids[name] = jnp.array(grid_info)
        else:
            func = getattr(grids_module, grid_info.kind)
            grids[name] = func(**grid_info.specs)

    order = variable_info.index.tolist()
    return {k: grids[k] for k in order}


def _get_function_info(user_model):
    """Derive information about all functions in the model.

    Args:
        user_model (dict): The model as provided by the user.

    Returns:
        pandas.DataFrame: A table with information about all functions in the model.
            The index contains the name of a model function. The columns are booleans
            that are True if the function has the corresponding property. The columns
            are: is_next, is_filter, is_constraint.

    """
    info = pd.DataFrame(index=list(user_model["functions"]))
    info["is_filter"] = info.index.str.endswith("_filter")
    info["is_constraint"] = info.index.str.endswith("_constraint")
    info["is_next"] = (
        info.index.str.startswith("next_") & ~info["is_constraint"] & ~info["is_filter"]
    )

    return info


def _get_functions(user_model, function_info, params):
    """Process the user provided model functions.

    Args:
        user_model (dict): The model as provided by the user.
        function_info (pd.DataFrame): A table with information about model functions.
        params (dict): The parameters of the model.

    Returns:
        dict: Dictionary containing all functions of the model. The keys are
            the names of the functions. The values are the processed functions.
            The main difference between processed and unprocessed functions is that
            processed functions take `params` as argument unless they are filter
            functions.

    """
    functions = {}
    for name, func in user_model["functions"].items():
        is_filter = function_info.loc[name, "is_filter"]
        if is_filter:
            if params.get(name, {}):
                raise ValueError("filters cannot depend on model parameters.")
            processed_func = func
        elif params[name]:
            processed_func = _get_extracting_function(
                func=func,
                params=params,
                name=name,
            )

        else:
            processed_func = _get_function_with_dummy_params(func=func)

        functions[name] = processed_func

    return functions


def _get_extracting_function(func, params, name):
    old_signature = list(inspect.signature(func).parameters)
    new_kwargs = [p for p in old_signature if p not in params[name]] + ["params"]

    @with_signature(kwargs=new_kwargs)
    @functools.wraps(func)
    def processed_func(**kwargs):
        _kwargs = {k: v for k, v in kwargs.items() if k in new_kwargs and k != "params"}
        return func(**_kwargs, **kwargs["params"][name])

    return processed_func


def _get_function_with_dummy_params(func):
    old_signature = list(inspect.signature(func).parameters)

    new_kwargs = [*old_signature, "params"]

    @with_signature(kwargs=new_kwargs)
    @functools.wraps(func)
    def processed_func(**kwargs):
        _kwargs = {k: v for k, v in kwargs.items() if k != "params"}
        return func(**_kwargs)

    return processed_func


def _process_stochastic_functions(functions):
    return [func for func in functions if hasattr(func, "_stochastic_info")]
