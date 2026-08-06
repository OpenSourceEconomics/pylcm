import dataclasses
from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
from dags import concatenate_functions, with_signature
from dags.tree import qname_from_tree_path

from _lcm.grids import ContinuousGrid, DiscreteGrid, IrregSpacedGrid
from _lcm.grids.coordinates import get_irreg_coordinate
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.typing import StateName
from _lcm.utils.functools import all_as_kwargs
from _lcm.variables import from_regime, get_grids
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, IntND, ScalarFloat


@dataclasses.dataclass(frozen=True, kw_only=True)
class VInterpolationInfo:
    """Information to work with the output of a function evaluated on a state space.

    An example is the value function array, which is the output of the value function
    evaluated on the state space.

    """

    state_names: tuple[StateName, ...]
    """Tuple of state variable names."""

    discrete_states: MappingProxyType[
        StateName, DiscreteGrid | _ContinuousStochasticProcess
    ]
    """Immutable mapping of discrete state names to their grids."""

    continuous_states: MappingProxyType[StateName, ContinuousGrid]
    """Immutable mapping of continuous state names to their grids."""


def create_v_interpolation_info(user_regime: UserRegime) -> VInterpolationInfo:
    """Create state space info for V-function interpolation.

    Args:
        user_regime: User-form `Regime` instance.

    Returns:
        State space information for the regime.

    """
    variables = from_regime(user_regime)
    grids = get_grids(user_regime)

    discrete_states = {name: grids[name] for name in variables.discrete_state_names}
    continuous_states = {name: grids[name] for name in variables.continuous_state_names}

    return VInterpolationInfo(
        state_names=variables.state_names,
        # `variables.{discrete,continuous}_state_names` filter on
        # topology/process; ty can't see through that to narrow grid types.
        discrete_states=MappingProxyType(discrete_states),  # ty: ignore[invalid-argument-type]
        continuous_states=MappingProxyType(continuous_states),  # ty: ignore[invalid-argument-type]
    )


def get_V_interpolator(
    *,
    v_interpolation_info: VInterpolationInfo,
    state_prefix: str,
    V_arr_name: str,
    co_map_state_names: tuple[StateName, ...] = (),
    entered_process_names: tuple[StateName, ...] = (),
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
        co_map_state_names: Tuple of discrete state names whose axes the caller has
            already sliced off `V_arr` (one device-local slice per value, via the
            backward-induction co-map). Their coordinates are dropped from the lookup
            so the interpolation reads the sliced array directly. These must be the
            leading axes of `V_arr`; only fixed (never-transitioning) states qualify.
        entered_process_names: Tuple of stochastic-process state names the caller
            enters at one declared physical value rather than drawing. Their axes
            are interpolated at that value instead of indexed, so the caller passes
            the value itself and the cost is one interpolation rather than the
            process's whole node axis.

    Returns:
        A callable that lets you treat the result of pre-calculating a function on the
            state space as an analytical function.

    """
    _fail_if_interpolation_axes_are_not_last(v_interpolation_info)
    _need_interpolation = bool(v_interpolation_info.continuous_states) or bool(
        entered_process_names
    )

    funcs: dict[
        str,
        Callable[..., ScalarFloat] | Callable[..., FloatND],
    ] = {}

    _discrete_axes = [
        state_prefix + var
        for var in v_interpolation_info.state_names
        if var in v_interpolation_info.discrete_states and var not in co_map_state_names
    ]

    _out_name = "__interpolation_data__" if _need_interpolation else "__fval__"
    funcs[_out_name] = _get_lookup_function(
        array_name=V_arr_name,
        axis_names=_discrete_axes,
        retained_axis_names=frozenset(
            state_prefix + var for var in entered_process_names
        ),
    )

    if _need_interpolation:
        for var in entered_process_names:
            funcs[f"__{var}_coord__"] = _get_entered_process_coordinate_finder(
                in_name=state_prefix + var,
                process=v_interpolation_info.discrete_states[var],
            )
        for var, grid_spec in v_interpolation_info.continuous_states.items():
            funcs[f"__{var}_coord__"] = _get_coordinate_finder(
                in_name=state_prefix + var,
                grid=grid_spec,
            )

        # An entered process keeps its axis in the array, and the lookup removes
        # every indexed axis ahead of it, so the axes left to interpolate are the
        # entered processes in state order followed by the continuous states.
        _continuous_axes = [
            f"__{var}_coord__"
            for var in v_interpolation_info.state_names
            if var in entered_process_names
            or var in v_interpolation_info.continuous_states
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
    retained_axis_names: frozenset[str] = frozenset(),
) -> Callable[..., FloatND]:
    """Create a function that emulates indexing into an array via named axes.

    Args:
        array_name: The name of the array into which the function indexes.
        axis_names: List of strings with names for each axis in the array.
        retained_axis_names: Names among `axis_names` to keep rather than index —
            an axis to be interpolated afterwards. It takes a full slice, so the
            axes left over are the retained ones in their own order followed by
            whatever trailed them.

    Returns:
        A callable with the keyword-only arguments `[*axis_names]` minus the
        retained ones, that looks up values from an array called `array_name`.

    """
    indexed_axis_names = [var for var in axis_names if var not in retained_axis_names]
    arg_names = [*indexed_axis_names, array_name]

    @with_signature(
        args=dict.fromkeys(arg_names, "FloatND | IntND"),
        return_annotation="FloatND",
    )
    def lookup_wrapper(*args: FloatND | IntND, **kwargs: FloatND | IntND) -> FloatND:
        kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=arg_names)
        positions = tuple(
            slice(None) if var in retained_axis_names else kwargs[var]
            for var in axis_names
        )
        return kwargs[array_name][positions]

    return lookup_wrapper


def _get_entered_process_coordinate_finder(
    *,
    in_name: str,
    process: DiscreteGrid | _ContinuousStochasticProcess,
) -> Callable[..., FloatND]:
    """Create a function placing a declared entry value on a process's node axis.

    A declared entry names one physical value, and the target holds its value
    function on the process's nodes. The coordinate of that value on those nodes
    is what the interpolation needs, and interpolating there is the same number
    the node basis expresses — at the cost of one interpolation rather than the
    whole node axis.

    Args:
        in_name: Name via which the declared physical value arrives.
        process: The target's stochastic process, whose nodes the value is placed
            on.

    Returns:
        A callable with the keyword-only argument `[in_name]` returning the
        coordinate, or NaN for a value the process cannot represent.

    """
    gridpoints = process.to_jax()
    lower = gridpoints[0]
    upper = gridpoints[-1]

    @with_signature(args={in_name: "FloatND"}, return_annotation="FloatND")
    def find_entered_process_coordinate(*args: FloatND, **kwargs: FloatND) -> FloatND:
        kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=[in_name])
        value = kwargs[in_name]
        coordinate = get_irreg_coordinate(value=value, points=gridpoints)
        # The interpolation extrapolates outside the node range rather than
        # refusing, and the target has no representation out there, so an entry
        # naming such a value is poisoned here and the solve-time check names the
        # regime and period.
        #
        # The test is on the physical value, not its coordinate. A coordinate only
        # stands in for the value where the map is invertible, which it is not on
        # a support of one node: there every value shares the sole index, so a
        # coordinate test would accept the whole real line.
        on_support = (value >= lower) & (value <= upper)
        return jnp.where(on_support, coordinate, jnp.nan)

    return find_entered_process_coordinate


def _get_coordinate_finder(
    *,
    in_name: str,
    grid: ContinuousGrid,
) -> Callable[..., FloatND]:
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
                args=dict.fromkeys(arg_names, "FloatND"), return_annotation="FloatND"
            )
            def find_irreg_coordinate(*args: FloatND, **kwargs: FloatND) -> FloatND:
                kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=arg_names)
                return get_irreg_coordinate(
                    value=kwargs[in_name], points=kwargs[points_param]
                )

            return find_irreg_coordinate

        # Fixed points — capture in closure
        points_jax = grid.to_jax()

        @with_signature(
            args=dict.fromkeys([in_name], "FloatND"), return_annotation="FloatND"
        )
        def find_irreg_coordinate(*args: FloatND, **kwargs: FloatND) -> FloatND:
            kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=[in_name])
            return get_irreg_coordinate(value=kwargs[in_name], points=points_jax)

        return find_irreg_coordinate

    # All other grid types (LinSpaced, LogSpaced, Piecewise*,
    # _ContinuousStochasticProcess)
    @with_signature(
        args=dict.fromkeys([in_name], "FloatND"), return_annotation="FloatND"
    )
    def find_coordinate(*args: FloatND, **kwargs: FloatND) -> FloatND:
        kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=[in_name])
        return grid.get_coordinate(kwargs[in_name])

    return find_coordinate


def _get_interpolator(
    *,
    name_of_values_on_grid: str,
    axis_names: list[str],
) -> Callable[..., FloatND]:
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

    @with_signature(
        args=dict.fromkeys(arg_names, "FloatND"), return_annotation="FloatND"
    )
    def interpolate(*args: FloatND, **kwargs: FloatND) -> FloatND:
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
