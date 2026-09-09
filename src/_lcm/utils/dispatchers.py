import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast

import jax
import jax.numpy as jnp
from jax import vmap

from _lcm.typing import ActionName, StateName
from _lcm.utils.containers import find_duplicates
from _lcm.utils.functools import allow_args, allow_only_kwargs, publish_signature
from lcm.exceptions import FunctionDispatchError
from lcm.typing import BoolND, FloatND, IntND

_MAX_FLAT_CELL_INDEX = 2**31 - 1

FunctionWithArrayReturn = TypeVar(
    "FunctionWithArrayReturn",
    bound=Callable[
        ...,
        FloatND
        | IntND
        | BoolND
        | tuple[FloatND | IntND | BoolND, FloatND | IntND | BoolND]
        | MappingProxyType[str, FloatND | IntND]
        | MappingProxyType[str, MappingProxyType[str, FloatND | IntND]],
    ],
)


def map_over_leading_axis[InputTree, OutputTree](
    *,
    func: Callable[[InputTree], OutputTree],
    xs: InputTree,
    batch_size: int,
) -> OutputTree:
    """Map one row body over a pytree leading axis, optionally in real batches.

    A positive batch size smaller than the axis is passed directly to
    ``jax.lax.map`` and therefore sets the compiled evaluation window. ``0``
    and values covering the axis use one vectorized pass. This is the common
    scheduling contract for EGM-family memory controls.
    """
    leaves = jax.tree.leaves(xs)
    if not leaves:
        raise ValueError("xs must contain at least one array")
    n_rows = int(leaves[0].shape[0])
    if any(int(leaf.shape[0]) != n_rows for leaf in leaves):
        raise ValueError("every leaf of xs must share a leading row count")
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    positional_func = allow_args(func)
    if 0 < batch_size < n_rows:
        return jax.lax.map(positional_func, xs, batch_size=batch_size)
    return jax.vmap(positional_func)(xs)


def simulation_spacemap(
    *,
    func: FunctionWithArrayReturn,
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
) -> FunctionWithArrayReturn:
    """Apply jax.lax.map so func can be evaluated on actions and simulated states.

    This function maps the function `func` over the simulation state-action-space. That
    is, it maps `func` over the Cartesian product of the action variables, and over the
    fixed simulation states. For each action variable, a leading dimension is added to
    the output object, with the length of the axis being the number of possible values
    in the grid. Importantly, it does not create a Cartesian product over the state
    variables, since these are fixed during the simulation. For the state variables,
    a single dimension is added to the output object, with the length of the axis
    being the number of simulated states.

    simulation_spacemap preserves the function signature and allows the function to be
    called with keyword arguments.

    Args:
        func: The function to be dispatched.
        action_names: Names of the action variables.
        state_names: Names of the state variables.

    Returns:
        A callable with the same arguments as func (but with an additional leading
        dimension) that returns an Array or pytree of Arrays. If `func` returns a
        scalar, the dispatched function returns an Array with k + 1 dimensions, where k
        is the length of `action_names` and the additional dimension corresponds to the
        `state_names`. The order of the dimensions is determined by the order of
        `action_names`. If the output of `func` is a jax pytree, the usual jax behavior
        applies, i.e. the leading dimensions of all arrays in the pytree are as
        described above but there might be additional dimensions.

    """
    # The model creation process ensures that in a user-created model the following
    # cannot happen. We double-check here to ensure that the post-processing does not
    # accidentally create such a situation.
    if duplicates := find_duplicates(action_names, state_names):
        msg = (
            "Same argument provided more than once in actions or states variables, "
            f"or is present in both: {duplicates}"
        )
        raise ValueError(msg)

    mappable_func = allow_args(func)

    mapped = allow_args(
        productmap(
            func=mappable_func,
            variables=action_names,
            batch_sizes=dict.fromkeys(action_names, 0),
        )
    )
    mapped = vmap_1d(func=mapped, variables=state_names, callable_with="only_args")

    publish_signature(target=mapped, signature=inspect.signature(mappable_func))

    return cast("FunctionWithArrayReturn", allow_only_kwargs(func=mapped))


def vmap_1d(
    *,
    func: FunctionWithArrayReturn,
    variables: tuple[str, ...],
    callable_with: Literal["only_args", "only_kwargs"] = "only_kwargs",
    co_mapped_in_axes: MappingProxyType[str, Any] | None = None,
) -> FunctionWithArrayReturn:
    """Apply vmap such that func is mapped over the specified variables.

    In contrast to a general vmap call, vmap_1d vectorizes along the leading axis of all
    of the requested variables simultaneously. Moreover, it preserves the function
    signature and allows the function to be called with keyword arguments.

    Args:
        func: The function to be dispatched.
        variables: Tuple with names of arguments that over which we map.
        callable_with: Whether to apply the allow_kwargs decorator to the dispatched
            function. If "only_args", the returned function can only be called with
            positional arguments. If "only_kwargs", the returned function can only be
            called with keyword arguments.
        co_mapped_in_axes: Immutable mapping of argument name to the `in_axes` spec to
            use for that argument, instead of the default `None` (unmapped). A scalar
            axis index maps that axis of the argument (broadcasting over its pytree
            leaves); use this to co-map a state's grid axis with the matching axis of a
            pytree argument so the body sees only the corresponding slice.

    Returns:
        A callable with the same arguments as func (but with an additional leading
        dimension) that returns a jax.Array or pytree of arrays. If `func`
        returns a scalar, the dispatched function returns a jax.Array with 1
        jax.Array with 1 dimension and length k, where k is the length of one of
        the mapped inputs in `variables`. The order of the dimensions is determined by
        the order of `variables` which can be different to the order of `funcs`
        arguments. If the output of `func` is a jax pytree, the usual jax behavior
        applies, i.e. the leading dimensions of all arrays in the pytree are as
        described above but there might be additional dimensions.

    """
    if duplicates := find_duplicates(variables):
        raise ValueError(
            f"Same argument provided more than once in variables: {duplicates}",
        )

    signature = inspect.signature(func)
    parameters = list(signature.parameters)

    positions = [parameters.index(var) for var in variables]

    co_mapped_in_axes = co_mapped_in_axes or MappingProxyType({})

    positional_func = allow_args(func)

    # Handle empty variables case - nothing to vmap over
    if not positions and not co_mapped_in_axes:
        vmapped = positional_func
    else:
        # Create in_axes to apply vmap over variables. This has one entry for each
        # argument of func, indicating whether the argument should be mapped over or
        # not. None means that the argument should not be mapped over, 0 means that it
        # should be mapped over the leading axis of the input. A `co_mapped_in_axes`
        # entry overrides the default for that argument — a scalar axis index there
        # maps that axis of every pytree leaf, co-mapping it with the variables.
        in_axes_for_vmap: list[Any] = cast("list[Any]", [None] * len(parameters))
        for p in positions:
            in_axes_for_vmap[p] = 0
        for name, axes in co_mapped_in_axes.items():
            in_axes_for_vmap[parameters.index(name)] = axes

        vmapped = vmap(positional_func, in_axes=in_axes_for_vmap)

    if callable_with == "only_kwargs":
        out = allow_only_kwargs(func=vmapped, enforce=False)
    else:
        out = vmapped
    publish_signature(target=out, signature=signature)

    return cast("FunctionWithArrayReturn", out)


def productmap(
    *,
    func: FunctionWithArrayReturn,
    variables: tuple[str, ...],
    batch_sizes: dict[str, int],
) -> FunctionWithArrayReturn:
    """Apply jax.lax.map so func can be evaluated on the Cartesian product of variables.

    This is achieved by an iterative application of jax.lax.map.

    In contrast to _base_productmap_batched, productmap preserves the function signature
    and allows the function to be called with keyword arguments.

    Args:
        func: The function to be dispatched.
        variables: Tuple with names of arguments that over which the Cartesian product
            should be formed.
        batch_sizes: Dict mapping each variable name to its batch size. A batch size
            of 0 means no batching.

    Returns:
        A callable with the same arguments as func (but with an additional leading
        dimension) that returns a jax.Array or pytree of arrays. If `func`
        returns a scalar, the dispatched function returns a jax.Array with k
        dimensions, where k is the length of `variables`. The order of the dimensions
        is determined by the order of `variables` which can be different to the order
        of `funcs` arguments. If the output of `func` is a jax pytree, the usual jax
        behavior applies, i.e. the leading dimensions of all arrays in the pytree are as
        described above but there might be additional dimensions.

    """
    if duplicates := find_duplicates(variables):
        raise ValueError(
            f"Same argument provided more than once in variables: {duplicates}",
        )

    func_callable_with_args = allow_args(func)

    mapped = _base_productmap_batched(
        func=func_callable_with_args,
        product_axes=variables,
        batch_sizes=batch_sizes,
    )

    # Create new signature where every parameter is kw-only as
    # batched_vmap takes only kwargs
    signature = inspect.signature(func_callable_with_args)
    new_parameters = [
        p.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        for p in signature.parameters.values()
    ]
    new_signature = signature.replace(parameters=new_parameters)
    publish_signature(target=mapped, signature=new_signature)

    return cast(
        "FunctionWithArrayReturn", allow_only_kwargs(func=mapped, enforce=False)
    )


def tiled_productmap(
    *,
    func: FunctionWithArrayReturn,
    variables: tuple[str, ...],
    width_keyword: str,
    untiled_variables: tuple[str, ...] = (),
) -> FunctionWithArrayReturn:
    """Map a bounded window of the C-order Cartesian product of named inputs.

    The static width keyword is consumed by the mapper. Each output leaf recovers
    the original product axes followed by its own trailing axes. Coordinate grids
    remain separate arrays. A window holding multiple prefix cells maps the final
    coordinate separately; other windows decode every coordinate from a flat index.
    Untiled variables use ordinary outer vmaps. Their axes are restored to their
    original positions before the result leaves this boundary.
    """
    if duplicates := find_duplicates(variables):
        msg = f"Same argument provided more than once in variables: {duplicates}"
        raise ValueError(msg)
    signature = inspect.signature(func)
    if width_keyword in signature.parameters:
        msg = f"Tile width keyword {width_keyword!r} collides with a function argument."
        raise FunctionDispatchError(msg)
    missing = set(variables).difference(signature.parameters)
    if missing:
        msg = f"Product variables are absent from the function: {sorted(missing)!r}."
        raise FunctionDispatchError(msg)
    if find_duplicates(untiled_variables) or set(untiled_variables).difference(
        variables
    ):
        msg = "Untiled variables must be a distinct subset of the product variables."
        raise FunctionDispatchError(msg)
    parameters = [
        parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        for parameter in signature.parameters.values()
    ]
    parameters.append(
        inspect.Parameter(width_keyword, inspect.Parameter.KEYWORD_ONLY, default=1)
    )
    cell_variables = tuple(name for name in variables if name not in untiled_variables)
    cell_mapper = _TiledProductMap(
        func=func, variables=cell_variables, width_keyword=width_keyword
    )
    mapped_signature = signature.replace(parameters=parameters)
    publish_signature(target=cell_mapper, signature=mapped_signature)
    mapped = cast("FunctionWithArrayReturn", cell_mapper)
    if untiled_variables:
        mapped = productmap(
            func=mapped,
            variables=untiled_variables,
            batch_sizes=dict.fromkeys(untiled_variables, 0),
        )
        mapped_order = (*untiled_variables, *cell_variables)
        if mapped_order != variables:
            restored = _RestoreProductAxisOrder(
                func=mapped,
                axes=tuple(mapped_order.index(name) for name in variables),
            )
            publish_signature(target=restored, signature=mapped_signature)
            mapped = cast("FunctionWithArrayReturn", restored)
    return mapped


@dataclass(frozen=True, kw_only=True, eq=False)
class _RestoreProductAxisOrder:
    """Restore the declared state order after mapping untiled axes outside cells."""

    func: Callable[..., Any]
    """Mapped scalar computation producing outer axes followed by cell axes."""
    axes: tuple[int, ...]
    """Permutation from mapped state axes to the original declared order."""

    def __call__(self, **kwargs: Any) -> Any:  # noqa: ANN401
        return jax.tree.map(
            partial(_transpose_product_axes, axes=self.axes), self.func(**kwargs)
        )


# keyword-only-exempt: library-callback=jax.tree.map
def _transpose_product_axes(value: jax.Array, *, axes: tuple[int, ...]) -> jax.Array:
    """Move state axes without changing a leaf's trailing roles or dtype."""
    return jnp.transpose(value, (*axes, *range(len(axes), value.ndim)))


@dataclass(frozen=True, kw_only=True, eq=False)
class _TiledProductMap:
    """Evaluate separate coordinate grids through a bounded Cartesian window."""

    func: Callable[..., Any]
    """Unchanged scalar function evaluated at each product coordinate."""
    variables: tuple[str, ...]
    """Product coordinates in canonical C order, outermost first."""
    width_keyword: str
    """Static keyword consumed by this mapping boundary."""

    def __call__(self, **kwargs: Any) -> Any:  # noqa: ANN401
        width = kwargs.pop(self.width_keyword, 1)
        if type(width) is not int or width < 1:
            msg = f"Tile width must be a positive static integer, got {width!r}."
            raise ValueError(msg)
        if not self.variables:
            return self.func(**kwargs)
        coordinates = tuple(jnp.atleast_1d(kwargs[name]) for name in self.variables)
        shape = tuple(coordinate.shape[0] for coordinate in coordinates)
        n_cells = math.prod(shape)
        if n_cells < 1 or n_cells > _MAX_FLAT_CELL_INDEX:
            msg = (
                f"State-cell product exceeds the positive int32 index range: {n_cells}."
            )
            raise ValueError(msg)
        arguments = MappingProxyType(
            {
                name: value
                for name, value in kwargs.items()
                if name not in self.variables
            }
        )
        if len(self.variables) > 1 and width // min(width, shape[-1]) > 1:
            return _map_grouped_product(
                func=self.func,
                variables=self.variables,
                coordinates=coordinates,
                shape=shape,
                arguments=arguments,
                width=width,
            )
        evaluate = _EvaluateTiledCell(
            func=self.func,
            variables=self.variables,
            coordinates=coordinates,
            strides=tuple(math.prod(shape[index + 1 :]) for index in range(len(shape))),
            arguments=arguments,
        )
        mapped = map_over_leading_axis(
            func=evaluate, xs=jnp.arange(n_cells, dtype=jnp.int32), batch_size=width
        )
        return jax.tree.map(partial(_restore_product_axes, shape=shape), mapped)


def _map_grouped_product(
    *,
    func: Callable[..., Any],
    variables: tuple[str, ...],
    coordinates: tuple[jax.Array, ...],
    shape: tuple[int, ...],
    arguments: MappingProxyType[str, Any],
    width: int,
) -> Any:  # noqa: ANN401
    """Map a flat prefix and final coordinate with at most two cell batch axes.

    The rectangle's two widths multiply to at most the requested width. Separate
    final-coordinate work can be reused across the prefix, while the decoded
    prefix keeps its scalar computations on one batch axis. The scalar cell and
    every trailing output role retain their original meanings.
    """
    inner_width = min(width, shape[-1])
    outer_width = max(1, width // inner_width)
    evaluate = _EvaluateTiledCell(
        func=_MapOverFinalCoordinate(
            func=func,
            variable=variables[-1],
            coordinate=coordinates[-1],
            width=inner_width,
        ),
        variables=variables[:-1],
        coordinates=coordinates[:-1],
        strides=tuple(
            math.prod(shape[index + 1 : -1]) for index in range(len(shape) - 1)
        ),
        arguments=arguments,
    )
    mapped = map_over_leading_axis(
        func=evaluate,
        xs=jnp.arange(math.prod(shape[:-1]), dtype=jnp.int32),
        batch_size=outer_width,
    )
    return jax.tree.map(
        partial(_restore_product_axes, shape=shape, n_flat_axes=2), mapped
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _MapOverFinalCoordinate:
    """Evaluate the final coordinate for one decoded prefix cell."""

    func: Callable[..., Any]
    """Unchanged scalar cell function."""
    variable: str
    """Name of the final Cartesian coordinate."""
    coordinate: jax.Array
    """Separate final-coordinate grid."""
    width: int
    """Active final-coordinate window within the requested cell budget."""

    def __call__(self, **kwargs: Any) -> Any:  # noqa: ANN401
        evaluate = _EvaluateTiledCell(
            func=self.func,
            variables=(self.variable,),
            coordinates=(self.coordinate,),
            strides=(1,),
            arguments=MappingProxyType(kwargs),
        )
        return map_over_leading_axis(
            func=evaluate,
            xs=jnp.arange(self.coordinate.shape[0], dtype=jnp.int32),
            batch_size=self.width,
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class _EvaluateTiledCell:
    """Decode one flat index and evaluate the unchanged scalar cell function."""

    func: Callable[..., Any]
    """Scalar cell function receiving the decoded coordinate values."""
    variables: tuple[str, ...]
    """Coordinate names in the same order as their separate arrays."""
    coordinates: tuple[jax.Array, ...]
    """Separate coordinate arrays, without a materialized product mesh."""
    strides: tuple[int, ...]
    """C-order integer strides used to decode a flat cell index."""
    arguments: MappingProxyType[str, Any]
    """Non-coordinate arguments forwarded unchanged to the cell function."""

    def __call__(self, index: jax.Array) -> Any:  # noqa: ANN401
        cell = {
            name: coordinate[(index // stride) % coordinate.shape[0]]
            for name, coordinate, stride in zip(
                self.variables, self.coordinates, self.strides, strict=True
            )
        }
        return self.func(**self.arguments, **cell)


# keyword-only-exempt: library-callback=jax.tree.map
def _restore_product_axes(
    value: jax.Array, *, shape: tuple[int, ...], n_flat_axes: int = 1
) -> jax.Array:
    """Restore the Cartesian state axes without changing a leaf's trailing axes."""
    return value.reshape((*shape, *value.shape[n_flat_axes:]))


def _base_productmap_batched(
    *,
    func: FunctionWithArrayReturn,
    product_axes: tuple[str, ...],
    batch_sizes: dict[str, int],
) -> FunctionWithArrayReturn:
    """Map func over the Cartesian product of product_axes and execute in batches.

    Like `jax.lax.map`, this function does not preserve the function signature.

    Args:
        func: The function to be dispatched. Cannot have positional-only parameters.
        product_axes: Tuple with names of arguments over which we apply
            `jax.lax.map`.
        batch_sizes: Dict with the batch sizes for each product_axis.

    Returns:
        A callable with the same arguments as func. See `productmap` for details.

    """
    parameters = inspect.signature(func).parameters
    for name, param in parameters.items():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise FunctionDispatchError(
                "Positional-only parameters are not allowed in dispatched functions. "
                f"The parameter '{name}' to the function "
                f"{getattr(func, '__name__', repr(func))} "
                "is POSITIONAL_ONLY."
            )

    return cast(
        "FunctionWithArrayReturn",
        _ProductMapBatched(
            func=func,
            product_axes=product_axes,
            batch_sizes=MappingProxyType(dict(batch_sizes)),
        ),
    )


@dataclass(frozen=True, eq=False)
class _ProductMapBatched:
    """Evaluate `func` on the Cartesian product of `product_axes`, in batches.

    Accepts whatever values the composed `func` expects (canonical JAX arrays in
    the production pipeline, but also Python scalars, non-canonical-dtype arrays,
    or `MappingProxyType` containers in callers that wrap their own pytrees) and
    returns whatever `func` returns; the wrapped `func` is responsible for its own
    contract.
    """

    func: Callable[..., Any]
    """The function evaluated at every point of the product."""
    product_axes: tuple[str, ...]
    """Names of the arguments whose values span the product, outermost first."""
    batch_sizes: MappingProxyType[str, int]
    """The `jax.lax.map` batch size per product axis, `0` for one vectorized pass."""

    def __post_init__(self) -> None:
        # Instrumentation such as `jax.named_call` reads a callable's name, so the
        # product map publishes the name of the function it evaluates, the way
        # `functools.wraps` would; the adapter wrapped around it copies it on.
        for attribute in ("__name__", "__qualname__"):
            object.__setattr__(
                self,
                attribute,
                getattr(self.func, attribute, type(self).__qualname__),
            )

    def __call__(self, **kwargs: Any) -> Any:  # noqa: ANN401
        non_array_kwargs = {
            key: val for key, val in kwargs.items() if key not in self.product_axes
        }
        loop_func = cast(
            "FunctionWithArrayReturn", partial(self.func, **non_array_kwargs)
        )
        # Map over one more product axis per step, innermost axis first, so the
        # outermost axis drives the outermost `jax.lax.map`.
        for axis in reversed(self.product_axes):
            loop_func = cast(
                "FunctionWithArrayReturn",
                _MappedOverOneMoreAxis(
                    loop_func=loop_func,
                    axis=axis,
                    axis_values=kwargs[axis],
                    batch_size=self.batch_sizes[axis],
                ),
            )
        return cast("FloatND", loop_func())


@dataclass(frozen=True, eq=False)
class _MappedOverOneMoreAxis:
    """`loop_func` mapped with `jax.lax.map` over the values of one product axis."""

    loop_func: Callable[..., Any]
    """The function evaluated once per value of `axis`."""
    axis: str
    """The argument of `loop_func` that takes one value of the axis per evaluation."""
    axis_values: Any
    """The values of the axis, mapped over their leading dimension."""
    batch_size: int
    """The `jax.lax.map` batch size, `0` for one vectorized pass."""

    def __call__(
        self,
        *already_mapped_args: Any,  # noqa: ANN401
        **already_mapped_kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        return jax.lax.map(
            partial(
                _evaluate_at_axis_value,
                loop_func=self.loop_func,
                axis=self.axis,
                mapped_args=already_mapped_args,
                mapped_kwargs=already_mapped_kwargs,
            ),
            jnp.atleast_1d(self.axis_values),
            batch_size=self.batch_size,
        )


# keyword-only-exempt: library-callback=jax.lax.map
def _evaluate_at_axis_value(
    axis_value: Any,  # noqa: ANN401
    *,
    loop_func: Callable[..., Any],
    axis: str,
    mapped_args: tuple[Any, ...],
    mapped_kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Evaluate `loop_func` at one value of `axis`, forwarding the other arguments."""
    return loop_func(*mapped_args, **{axis: axis_value}, **mapped_kwargs)
