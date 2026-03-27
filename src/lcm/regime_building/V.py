import dataclasses
from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
from dags import concatenate_functions, with_signature
from dags.tree import qname_from_tree_path
from jax import Array

from lcm.grids import ContinuousGrid, DiscreteGrid, IrregSpacedGrid
from lcm.grids.coordinates import get_irreg_coordinate
from lcm.regime import Regime
from lcm.regime_building.ndimage import map_coordinates
from lcm.shocks import _ShockGrid
from lcm.typing import FloatND, ScalarFloat
from lcm.utils.functools import all_as_kwargs


@dataclasses.dataclass(frozen=True, kw_only=True)
class VInterpolationInfo:
    """Information to work with the output of a function evaluated on a state space.

    An example is the value function array, which is the output of the value function
    evaluated on the state space.

    """

    state_names: tuple[str, ...]
    """Tuple of state variable names."""

    discrete_states: MappingProxyType[str, DiscreteGrid | _ShockGrid]
    """Immutable mapping of discrete state names to their grids."""

    continuous_states: MappingProxyType[str, ContinuousGrid]
    """Immutable mapping of continuous state names to their grids."""


def create_v_interpolation_info(regime: Regime) -> VInterpolationInfo:
    """Create state space info for V-function interpolation.

    Args:
        regime: Regime instance.

    Returns:
        State space information for the regime.

    """
    from lcm.regime_building.variable_info import (  # noqa: PLC0415
        get_grids,
        get_variable_info,
    )

    vi = get_variable_info(regime)
    grids = get_grids(regime)

    state_names = vi.query("is_state").index.tolist()

    discrete_states = {
        name: grid_spec
        for name, grid_spec in grids.items()
        if (name in state_names and isinstance(grid_spec, DiscreteGrid))
        or isinstance(grid_spec, _ShockGrid)
    }

    continuous_states = {
        name: grid_spec
        for name, grid_spec in grids.items()
        if name in state_names
        and isinstance(grid_spec, ContinuousGrid)
        and not isinstance(grid_spec, _ShockGrid)
    }

    return VInterpolationInfo(
        state_names=tuple(state_names),
        discrete_states=MappingProxyType(discrete_states),
        continuous_states=MappingProxyType(continuous_states),
    )


def get_V_interpolator(
    *,
    v_interpolation_info: VInterpolationInfo,
    state_prefix: str,
    V_arr_name: str,
) -> Callable[..., FloatND]:
    """Create a function representation of a value function array.

    Generate a function that looks up discrete values and interpolates values for
    continuous variables on the value function array. The arguments of the resulting
    function can be split in two categories:

    1. The original arguments of the function that was used to pre-calculate the
       value function on the state space grid.
    2. Auxiliary arguments, such as information about the grids, which are needed
       for the interpolation.

    After partialling in all helper arguments, the resulting function behaves like
    an analytical function, i.e. it can be evaluated on points that do not lie on
    the grid points of the state variables. In particular, it can also be jitted,
    differentiated, and vmapped with JAX.

    Internally, the resulting function roughly does the following steps:

    - It looks up values at discrete variable positions (integer codes index directly
      into the array).
    - It translates values of continuous variables into coordinates needed for
      interpolation via jax.scipy.ndimage.map_coordinates.
    - It performs the interpolation.

    Depending on the grid, only a subset of these steps is relevant. The chosen
    implementation of each step is also adjusted to the type of grid. In particular we
    try to avoid searching for neighboring values on a grid and instead exploit
    structure in the grid to calculate where those entries are. The order in which the
    functions are called is determined by a DAG.

    Args:
        v_interpolation_info: Class containing all information needed to interpret the
            pre-calculated values of a function.
        state_prefix: Prefix that will be added to all argument names of the resulting
            function, except for the helper arguments.
        V_arr_name: The name of the argument via which the pre-calculated values, that
            have been evaluated on the state-space grid, will be passed into the
            resulting function.

    Returns:
        A callable that lets you treat the result of pre-calculating a function on the
            state space as an analytical function.

    """
    _fail_if_interpolation_axes_are_not_last(v_interpolation_info)
    _need_interpolation = bool(v_interpolation_info.continuous_states)

    funcs: dict[
        str,
        Callable[..., ScalarFloat] | Callable[..., FloatND],
    ] = {}

    _discrete_axes = [
        state_prefix + var
        for var in v_interpolation_info.state_names
        if var in v_interpolation_info.discrete_states
    ]

    _out_name = "__interpolation_data__" if _need_interpolation else "__fval__"
    funcs[_out_name] = _get_lookup_function(
        array_name=V_arr_name,
        axis_names=_discrete_axes,
    )

    if _need_interpolation:
        for var, grid_spec in v_interpolation_info.continuous_states.items():
            funcs[f"__{var}_coord__"] = _get_coordinate_finder(
                in_name=state_prefix + var,
                grid=grid_spec,
            )

        _continuous_axes = [
            f"__{var}_coord__"
            for var in v_interpolation_info.state_names
            if var in v_interpolation_info.continuous_states
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


def _get_lookup_function(
    *,
    array_name: str,
    axis_names: list[str],
) -> Callable[..., Array]:
    """Create a function that emulates indexing into an array via named axes.

    Args:
        array_name: The name of the array into which the function indexes.
        axis_names: List of strings with names for each axis in the array.

    Returns:
        A callable with the keyword-only arguments `[*axis_names]` that looks up values
        from an array called `array_name`.

    """
    arg_names = [*axis_names, array_name]

    @with_signature(args=dict.fromkeys(arg_names, "Array"), return_annotation="Array")
    def lookup_wrapper(*args: Array, **kwargs: Array) -> Array:
        kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=arg_names)
        positions = tuple(kwargs[var] for var in axis_names)
        return kwargs[array_name][positions]

    return lookup_wrapper


def _get_coordinate_finder(
    *,
    in_name: str,
    grid: ContinuousGrid,
) -> Callable[..., Array]:
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
    if isinstance(grid, IrregSpacedGrid):
        if grid.pass_points_at_runtime:
            state_name = in_name.removeprefix("next_")
            points_param = qname_from_tree_path((state_name, "points"))
            arg_names = [in_name, points_param]

            @with_signature(
                args=dict.fromkeys(arg_names, "Array"), return_annotation="Array"
            )
            def find_irreg_coordinate(*args: Array, **kwargs: Array) -> Array:
                kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=arg_names)
                return get_irreg_coordinate(
                    value=kwargs[in_name], points=kwargs[points_param]
                )

            return find_irreg_coordinate

        # Fixed points — capture in closure
        points_jax = grid.to_jax()

        @with_signature(
            args=dict.fromkeys([in_name], "Array"), return_annotation="Array"
        )
        def find_irreg_coordinate(*args: Array, **kwargs: Array) -> Array:
            kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=[in_name])
            return get_irreg_coordinate(value=kwargs[in_name], points=points_jax)

        return find_irreg_coordinate

    # All other grid types (LinSpaced, LogSpaced, Piecewise*, ShockGrid)
    @with_signature(args=dict.fromkeys([in_name], "Array"), return_annotation="Array")
    def find_coordinate(*args: Array, **kwargs: Array) -> Array:
        kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=[in_name])
        return grid.get_coordinate(kwargs[in_name])

    return find_coordinate


def _get_interpolator(
    *,
    name_of_values_on_grid: str,
    axis_names: list[str],
) -> Callable[..., Array]:
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

    @with_signature(args=dict.fromkeys(arg_names, "Array"), return_annotation="Array")
    def interpolate(*args: Array, **kwargs: Array) -> Array:
        kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=arg_names)
        coordinates = jnp.array([kwargs[var] for var in axis_names])
        return map_coordinates(
            input=kwargs[name_of_values_on_grid],
            coordinates=coordinates,
        )

    return interpolate


def _fail_if_interpolation_axes_are_not_last(
    v_interpolation_info: VInterpolationInfo,
) -> None:
    """Fail if the continuous variables are not the last elements in var_names.

    Args:
        v_interpolation_info: Class containing all information needed to interpret the
            precalculated values of a function.

    Raises:
        ValueError: If the continuous variables are not the last elements in var_names.

    """
    common = set(v_interpolation_info.continuous_states) & set(
        v_interpolation_info.state_names
    )

    if common:
        n_common = len(common)
        if sorted(common) != sorted(v_interpolation_info.state_names[-n_common:]):
            msg = "Continuous variables need to be the last entries in var_names."
            raise ValueError(msg)
