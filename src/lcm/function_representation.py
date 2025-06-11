from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax.numpy as jnp
from dags import concatenate_functions
from dags.signature import with_signature
from jax import Array

from lcm.functools import all_as_kwargs
from lcm.grids import ContinuousGrid
from lcm.interfaces import StateSpaceInfo
from lcm.ndimage import map_coordinates
from lcm.typing import Scalar


def get_value_function_representation(
    state_space_info: StateSpaceInfo,
    *,
    input_prefix: str = "next_",
    name_of_values_on_grid: str = "next_V_arr",
) -> Callable[..., Array]:
    """Create a function representation of the next value function array.

    The returned function
    ---------------------

    This function generates a function that looks up discrete values and interpolates
    values for continuous variables on the value function array. The arguments of the
    resulting function can be split in two categories:

       1. The original arguments of the function that was used to pre-calculate the
          value function on the state space grid.

       2. Auxiliary arguments, such as information about the grids, which are needed for
          example, for the interpolation.

    After partialling in all helper arguments, the resulting function behaves like an
    analytical function, i.e. it can be evaluated on points that do not lie on the grid
    points of the state variables. In particular, it can also be jitted, differentiated
    and vmapped with jax.

    How does it work?
    -----------------

    The resulting function roughly does the following steps:

    - It translates values of discrete variables into positions.
    - It translates values of continuous variables into coordinates needed for
      interpolation via jax.scipy.ndimage.map_coordinates.
    - It performs the interpolation.

    Depending on the grid, only a subset of these steps is relevant. The chosen
    implementation of each step is also adjusted to the type of grid. In particular we
    try to avoid searching for neighboring values on a grid and instead exploit
    structure in the grid to calculate where those entries are. The order in which the
    functions are called is determined by a DAG.

    Args:
        state_space_info: Class containing all information needed to interpret the
            pre-calculated values of a function.
        input_prefix: Prefix that will be added to all argument names of the resulting
            function, except for the helpers arguments. Default is "next_"; since the
            value function is typically evaluated on the next period's state space.
        name_of_values_on_grid: The name of the argument via which the pre-calculated
            values, that have been evaluated on the state-space grid, will be passed
            into the resulting function. Defaults to "next_V_arr".

    Returns:
        A callable that lets you treat the result of pre-calculating a function on the
            state space as an analytical function.

    """
    # ==================================================================================
    # check inputs
    # ==================================================================================
    _fail_if_interpolation_axes_are_not_last(state_space_info)
    _need_interpolation = bool(state_space_info.continuous_states)

    # ==================================================================================
    # create functions to look up position of discrete variables from labels
    # ==================================================================================
    funcs = {}

    for var in state_space_info.discrete_states:
        funcs[f"__{var}_pos__"] = _get_label_translator(
            in_name=input_prefix + var,
        )

    # ==================================================================================
    # create a function for the discrete lookup
    # ==================================================================================
    # lookup is positional, so the inputs of the wrapper functions need to be the
    # outcomes of tranlating labels into positions
    _internal_axes = [f"__{var}_pos__" for var in state_space_info.states_names]
    _discrete_axes = [ax for ax in _internal_axes if ax in funcs]

    _out_name = "__interpolation_data__" if _need_interpolation else "__fval__"
    funcs[_out_name] = _get_lookup_function(
        array_name=name_of_values_on_grid,
        axis_names=_discrete_axes,
    )

    if _need_interpolation:
        # ==============================================================================
        # create functions to find coordinates for the interpolation
        # ==============================================================================
        for var, grid_spec in state_space_info.continuous_states.items():
            funcs[f"__{var}_coord__"] = _get_coordinate_finder(
                in_name=input_prefix + var,
                grid=grid_spec,
            )

        # ==============================================================================
        # create interpolation function
        # ==============================================================================
        _continuous_axes = [
            f"__{var}_coord__"
            for var in state_space_info.states_names
            if var in state_space_info.continuous_states
        ]
        funcs["__fval__"] = _get_interpolator(
            name_of_values_on_grid="__interpolation_data__",
            axis_names=_continuous_axes,
        )

    return concatenate_functions(
        functions=funcs,
        targets="__fval__",
        set_annotations=True,
    )


def _get_label_translator(
    in_name: str,
) -> Callable[..., Scalar]:
    """Create a function that translates a label into a position in a list of labels.

    Currently, only labels are supported that are themselves indices. The label
    translator in this case is thus just the identity function.

    Args:
        in_name: Name of the variable that provides the label in the signature of the
            resulting function.

    Returns:
        A callable with the keyword only argument `in_name` that converts a label into a
        position in a list of labels.

    """

    @with_signature(args=dict.fromkeys([in_name], "Scalar"), return_annotation="Scalar")
    def translate_label(*args: Scalar, **kwargs: Scalar) -> Scalar:
        kwargs = all_as_kwargs(args, kwargs, arg_names=[in_name])
        return kwargs[in_name]

    return translate_label


def _get_lookup_function(
    array_name: str,
    axis_names: list[str],
) -> Callable[..., Scalar]:
    """Create a function that emulates indexing into an array via named axes.

    Args:
        array_name (str): The name of the array into which the function indexes.
        axis_names (list): List of strings with names for each axis in the array.

    Returns:
        A callable with the keyword-only arguments `[*axis_names]` that looks up values
        from an array called `array_name`.

    """
    arg_names = [*axis_names, array_name]

    @with_signature(args=dict.fromkeys(arg_names, "Scalar"), return_annotation="Scalar")
    def lookup_wrapper(*args: Scalar, **kwargs: Scalar) -> Scalar:
        kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)
        positions = tuple(kwargs[var] for var in axis_names)
        arr = cast("Array", kwargs[array_name])
        return arr[positions]

    return lookup_wrapper


def _get_coordinate_finder(
    in_name: str,
    grid: ContinuousGrid,
) -> Callable[..., Scalar]:
    """Create a function that translates a value into coordinates on a grid.

    The resulting coordinates can be used to do linear interpolation via
    jax.scipy.ndimage.map_coordinates.

    Args:
        in_name: Name via which the value to be translated into coordinates will be
            passed into the resulting function.
        grid: The continuous grid on which the value is to be translated into
            coordinates.

    Returns:
        A callable with keyword-only argument [in_name] that translates a value into
        coordinates on a grid.

    """

    @with_signature(args=dict.fromkeys([in_name], "Scalar"), return_annotation="Scalar")
    def find_coordinate(*args: Scalar, **kwargs: Scalar) -> Scalar:
        kwargs = all_as_kwargs(args, kwargs, arg_names=[in_name])
        return grid.get_coordinate(kwargs[in_name])

    return find_coordinate


def _get_interpolator(
    name_of_values_on_grid: str,
    axis_names: list[str],
) -> Callable[..., Scalar]:
    """Create a function interpolator via named axes.

    Args:
        name_of_values_on_grid: The name of the argument via which the pre-calculated
            values, that have been evaluated on a grid, will be passed into the
            resulting function.
        axis_names: Names of the axes in the data array.

    Returns:
        A callable that interpolates a function via named axes.

    """
    arg_names = [name_of_values_on_grid, *axis_names]

    @with_signature(args=dict.fromkeys(arg_names, "Scalar"), return_annotation="Scalar")
    def interpolate(*args: Scalar, **kwargs: Scalar) -> Scalar:
        kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)
        coordinates = jnp.array([kwargs[var] for var in axis_names])
        return map_coordinates(
            input=kwargs[name_of_values_on_grid],
            coordinates=coordinates,
        )

    return interpolate


def _fail_if_interpolation_axes_are_not_last(state_space_info: StateSpaceInfo) -> None:
    """Fail if the continuous variables are not the last elements in var_names.

    Args:
        state_space_info: Class containing all information needed to interpret the
            precalculated values of a function.

    Raises:
        ValueError: If the continuous variables are not the last elements in var_names.

    """
    common = set(state_space_info.continuous_states) & set(
        state_space_info.states_names
    )

    if common:
        n_common = len(common)
        if sorted(common) != sorted(state_space_info.states_names[-n_common:]):
            msg = "Continuous variables need to be the last entries in var_names."
            raise ValueError(msg)
