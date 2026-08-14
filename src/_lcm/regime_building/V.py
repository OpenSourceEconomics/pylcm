import dataclasses
from collections.abc import Callable
from types import MappingProxyType

import jax
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
    interpolate_process_axes: bool = False,
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
        interpolate_process_axes: When `True`, build a PROCESS-AWARE interpolator
            (`_get_V_interpolator_process_aware`) instead of the ordinary
            integer-lookup / trailing-continuous-axes interpolator. Every state
            axis (including a non-folded process axis, which is otherwise
            classified `discrete_states` for the ordinary Markov-chain solve
            path) is read through a single `map_coordinates` call, in native
            `state_names` order, so it sidesteps
            `_fail_if_interpolation_axes_are_not_last`. Use this only for a
            reader that may receive an off-grid VALUE for a process axis (a
            `SamePeriodRef` projection / gated-edge fallback); the ordinary
            continuation-value path always feeds a process axis its exact
            on-grid Markov-chain index and must keep using the fast integer
            lookup, so leave this `False` there.
        entered_process_names: Tuple of stochastic-process state names the caller
            enters at one declared physical value rather than drawing. Their axes
            are interpolated at that value instead of indexed, so the caller passes
            the value itself and the cost is one interpolation rather than the
            process's whole node axis.

    Returns:
        A callable that lets you treat the result of pre-calculating a function on the
            state space as an analytical function.

    Raises:
        ValueError: If `interpolate_process_axes` and `entered_process_names` are
            both given. The two ask for different treatment of a value outside a
            process's node range — clamping to the nearest node, and poisoning the
            read — so a reader that wants the second cannot get it from the first.

    """
    if interpolate_process_axes and entered_process_names:
        named = ", ".join(f"'{name}'" for name in entered_process_names)
        msg = (
            "get_V_interpolator takes either `interpolate_process_axes=True` or "
            f"`entered_process_names` ({named}), not both: the process-aware "
            "reader clamps an out-of-range value to the nearest node, while an "
            "entered process poisons a value the target cannot represent."
        )
        raise ValueError(msg)

    if interpolate_process_axes:
        return _get_V_interpolator_process_aware(
            v_interpolation_info=v_interpolation_info,
            state_prefix=state_prefix,
            V_arr_name=V_arr_name,
            co_map_state_names=co_map_state_names,
        )

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


def _get_V_interpolator_process_aware(
    *,
    v_interpolation_info: VInterpolationInfo,
    state_prefix: str,
    V_arr_name: str,
    co_map_state_names: tuple[StateName, ...],
) -> Callable[..., FloatND]:
    """Interpolate every axis of `V_arr` through one `map_coordinates` call.

    Companion to `get_V_interpolator` (`interpolate_process_axes=True`).
    Unlike the ordinary path — integer fancy-indexing for `discrete_states`,
    then `map_coordinates` over the trailing `continuous_states` — this
    builds ONE coordinate per axis, in `v_interpolation_info.state_names`
    (i.e. `V_arr`'s actual axis) order, and interpolates the whole array at
    once:

    - A genuine `DiscreteGrid` axis (in `discrete_states`, not a process):
      the incoming value IS the integer node index already; passed through
      unchanged as an (integer-valued) coordinate. `map_coordinates` resolves
      an integer-valued coordinate to weight 1.0 on that node (an exact
      lookup, just routed through interpolation machinery), so this is
      numerically identical to the fancy-indexing lookup for any caller that
      always feeds on-grid indices here.
    - A non-folded process axis (`discrete_states`, `_ContinuousStochasticProcess`):
      the incoming value is treated as a genuine VALUE in the process's own
      units (not a node index) and mapped to a fractional coordinate via
      `grid.get_coordinate`, clamped to the node range (see
      `_get_process_coordinate_finder`) so an off-grid projection degrades to
      clamped linear extrapolation instead of `map_coordinates`' unclamped
      linear extrapolation.
    - A `continuous_states` axis: the existing `_get_coordinate_finder`.

    Doing every axis through the same `map_coordinates` call sidesteps
    `_fail_if_interpolation_axes_are_not_last`: axis order here is simply
    `V_arr`'s own axis order, independent of which axes are "discrete" vs.
    "continuous" in the ordinary path's sense.

    Args:
        v_interpolation_info: Class containing all information needed to interpret the
            pre-calculated values of a function.
        state_prefix: Prefix added to all argument names of the resulting function.
        V_arr_name: The name of the argument via which `V_arr` is passed in.
        co_map_state_names: State names whose axes are already sliced off `V_arr`
            by the caller; dropped from the coordinate list (see
            `get_V_interpolator`).

    Returns:
        A callable that lets you treat the result of pre-calculating a function on the
            state space as an analytical function.

    """
    funcs: dict[str, Callable[..., FloatND]] = {}
    axis_coord_names: list[str] = []
    for var in v_interpolation_info.state_names:
        if var in co_map_state_names:
            continue
        coord_name = f"__{var}_coord__"
        if var in v_interpolation_info.continuous_states:
            funcs[coord_name] = _get_coordinate_finder(
                in_name=state_prefix + var,
                grid=v_interpolation_info.continuous_states[var],
            )
        else:
            grid_or_process = v_interpolation_info.discrete_states[var]
            if isinstance(grid_or_process, _ContinuousStochasticProcess):
                funcs[coord_name] = _get_process_coordinate_finder(
                    in_name=state_prefix + var,
                    grid=grid_or_process,
                )
            else:
                funcs[coord_name] = _get_identity_coordinate(in_name=state_prefix + var)
        axis_coord_names.append(coord_name)

    funcs["__fval__"] = _get_interpolator(
        name_of_values_on_grid=V_arr_name,
        axis_names=axis_coord_names,
    )

    return concatenate_functions(
        functions=funcs,
        targets="__fval__",
        set_annotations=True,
    )


def _get_process_coordinate_finder(
    *,
    in_name: str,
    grid: _ContinuousStochasticProcess,
) -> Callable[..., FloatND]:
    """Create a function that maps a process VALUE to a clamped node coordinate.

    Unlike the ordinary `_get_coordinate_finder` (whose generic branch also
    calls `grid.get_coordinate` for a `_ContinuousStochasticProcess`, but is
    only ever reached for a genuine `continuous_states` axis, never a process
    one), this clamps the result to `[0, n_points - 1]`. Both
    `get_irreg_coordinate` (inside `grid.get_coordinate`) and
    `map_coordinates` extrapolate linearly beyond the node range on their
    own; left uncomposed, a projected shock value far outside the process's
    discretized support would extrapolate TWICE, so clamping here keeps an
    out-of-range projection a graceful (bounded) read of the nearest node
    instead of a silent blow-up.

    Args:
        in_name: Name via which the value to be translated into a coordinate will be
            passed into the resulting function.
        grid: The non-folded continuous stochastic process whose node axis the
            value is translated onto.

    Returns:
        A callable with keyword-only argument [in_name] that translates a process
        value into a clamped coordinate on its node axis.

    """
    n_points = grid.n_points

    @with_signature(
        args=dict.fromkeys([in_name], "FloatND"), return_annotation="FloatND"
    )
    def find_process_coordinate(*args: FloatND, **kwargs: FloatND) -> FloatND:
        kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=[in_name])
        coordinate = grid.get_coordinate(kwargs[in_name])
        return jnp.clip(coordinate, 0.0, n_points - 1)

    return find_process_coordinate


def _get_identity_coordinate(*, in_name: str) -> Callable[..., FloatND]:
    """Create a function that places a genuine discrete axis value on its own axis.

    Used by `_get_V_interpolator_process_aware` for a genuine (non-process)
    `discrete_states` axis: the incoming value is the exact integer node
    index, so it is the `map_coordinates` coordinate directly — no
    translation needed.

    Each node of such an axis is a category, and nothing lies between two of
    them, so a coordinate strictly between two nodes names no state at all.
    `map_coordinates` would happily return the weighted blend of the two
    categories' values, which is why a non-integral coordinate is refused
    here rather than interpolated:

    - Evaluated on a concrete value (outside a JAX trace), it raises and
      names the axis.
    - Under a trace the value is not available to a Python conditional, so
      the coordinate is poisoned to NaN instead, which carries through the
      interpolation and leaves the read reportable as NaN rather than
      readable as a blend.

    Args:
        in_name: Name via which the value is passed into the resulting function.

    Returns:
        A callable with keyword-only argument [in_name] that returns that argument
        as a coordinate, refusing a value that names no category.

    """

    @with_signature(
        args=dict.fromkeys([in_name], "FloatND"), return_annotation="FloatND"
    )
    def identity_coordinate(*args: FloatND, **kwargs: FloatND) -> FloatND:
        kwargs = all_as_kwargs(args=args, kwargs=kwargs, arg_names=[in_name])
        value = kwargs[in_name]
        if jnp.issubdtype(jnp.asarray(value).dtype, jnp.integer):
            # An integer code is a category by construction, and returning it
            # untouched keeps the coordinate's dtype the caller's own.
            return value
        names_a_category = value == jnp.round(value)
        try:
            all_name_a_category = bool(jnp.all(names_a_category))
        except jax.errors.ConcretizationTypeError:
            return jnp.where(names_a_category, value, jnp.nan)
        if not all_name_a_category:
            msg = (
                f"Categorical state '{in_name}' was read at a coordinate that "
                "lies strictly between two of its categories, which names no "
                f"category: {value}. A categorical value must be one of the "
                "integer codes of its `DiscreteGrid`."
            )
            raise ValueError(msg)
        return value

    return identity_coordinate


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
