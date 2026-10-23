"""Turn a function of scalars into a function of grids.

`productmap` maps over the Cartesian product of its named arguments,
`tiled_productmap` evaluates that product in blocks, `vmap_1d` maps over named
arguments jointly, `simulation_spacemap` maps the per-subject axes simulation
needs, and `map_over_leading_axis` maps one leading axis of a pytree.
"""

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Literal, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.core import ShapedArray

from _lcm.solution.feasibility import (
    BoundedResult,
    merge_bound_input_status,
    merge_continuation_read_counts,
    merge_query_input_status,
    reduce_bound_input_status,
    reduce_continuation_read_counts,
    reduce_query_input_status,
)
from _lcm.typing import ActionName, PytreeValue, QAndFArg, StateName
from _lcm.utils.containers import find_duplicates
from _lcm.utils.functools import allow_args, allow_only_kwargs, publish_signature
from lcm.exceptions import FunctionDispatchError
from lcm.typing import ReferenceName, RegimeName, ValueND

_MAX_FLAT_CELL_INDEX = 2**31 - 1

# A dispatched function returns any tree of arrays: the maps stack every leaf
# along the mapped axes, whatever the tree's shape.
FunctionWithArrayReturn = TypeVar(
    "FunctionWithArrayReturn", bound=Callable[..., PytreeValue]
)


# Largest share of extra rows worth adding so the batch size divides the axis.
# `jax.lax.map` traces its body a second time for a remainder batch, and when such
# maps nest the extra traces multiply, so a remainder is expensive at lowering time.
# Repeating the last row up to the next multiple of the batch size removes the
# remainder at the cost of evaluating the added rows, which is cheap only while
# they are a small share of the axis. Above this share the remainder batch keeps
# its own trace.
_MAX_PADDING_SHARE = 0.01


def map_over_leading_axis[InputTree, OutputTree](
    *,
    func: Callable[[InputTree], OutputTree],
    xs: InputTree,
    batch_size: int,
    bound_summary: bool = False,
    active: jax.Array | None = None,
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
    if bound_summary:
        return cast(
            "OutputTree",
            _map_with_bound_summary(
                func=cast(
                    "Callable[[InputTree], BoundedResult[PytreeValue]]", positional_func
                ),
                xs=xs,
                batch_size=batch_size,
                active=active,
            ),
        )
    if active is not None:
        raise TypeError("An active-lane mask requires an explicit bound summary.")
    if not 0 < batch_size < n_rows:
        return jax.vmap(positional_func)(xs)
    n_padding = -n_rows % batch_size
    if n_padding == 0 or n_padding / n_rows > _MAX_PADDING_SHARE:
        return jax.lax.map(positional_func, xs, batch_size=batch_size)
    padded = jax.tree.map(
        lambda leaf: jnp.concatenate([leaf, jnp.repeat(leaf[-1:], n_padding, axis=0)]),
        xs,
    )
    mapped = jax.lax.map(positional_func, padded, batch_size=batch_size)
    return jax.tree.map(lambda leaf: leaf[:n_rows], mapped)


def vary_carry_like_body[Carry](
    *, body: Callable[[Carry], Carry], carry: Carry
) -> Carry:
    """Give each carry leaf the manual axes its scan-body output varies over.

    An initial carry built from constants is invariant over every `shard_map`
    axis, while a body that reads sharded data returns a varying carry, and
    `lax.scan` requires both to have one type. Each leaf is cast to varying over
    the axes its abstract body output varies over and it does not, until the
    two agree. Outside `shard_map` no axis is manual and `carry` is returned
    unchanged.

    Args:
        body: The scan body with its per-step input bound, from carry to carry.
        carry: The initial carry.

    Returns:
        `carry`, varying over the same manual axes as the body's output.

    """
    while True:
        # Only leaves leave the abstract evaluation: a carry container may refuse
        # abstract leaves.
        target = jax.eval_shape(lambda state: jax.tree.leaves(body(state)), carry)
        missing = [
            tuple(sorted(_varying_axes(out) - _varying_axes(jax.typeof(leaf))))
            for leaf, out in zip(jax.tree.leaves(carry), target, strict=True)
        ]
        if not any(missing):
            return carry
        leaves = [
            jax.lax.pcast(leaf, axes, to="varying") if axes else leaf
            for leaf, axes in zip(jax.tree.leaves(carry), missing, strict=True)
        ]
        carry = jax.tree.unflatten(jax.tree.structure(carry), leaves)


def _varying_axes(
    value: jax.ShapeDtypeStruct | ShapedArray,
) -> frozenset[str]:
    """Return the manual mesh axes an abstract value varies over."""
    manual = getattr(value, "manual_axis_type", None)
    return frozenset() if manual is None else frozenset(manual.varying)


def _map_with_bound_summary[InputTree, Output: PytreeValue](
    *,
    func: Callable[[InputTree], BoundedResult[Output]],
    xs: InputTree,
    batch_size: int,
    active: jax.Array | None,
) -> BoundedResult[Output]:
    """Store ordinary rows while OR-folding active flags in a fixed scan carry.

    A single window is evaluated directly; several windows all run inside one
    scan that returns no scan history. Either way the row body is fenced off
    from the surrounding program, so each row's bits do not depend on the
    window count or on what consumes the map. Only ordinary output buffers have
    population extent; integrity lives in the current window and scalar carry.
    """
    n_rows = jax.tree.leaves(xs)[0].shape[0]
    if n_rows < 1:
        raise ValueError("A bounded summary map requires a nonempty row axis.")
    if active is not None and (active.shape != (n_rows,) or active.dtype != jnp.bool_):
        raise TypeError("Bound-summary activity must be a Boolean row mask.")
    width = n_rows if batch_size == 0 else min(batch_size, n_rows)
    n_windows = (n_rows + width - 1) // width
    offsets = jnp.arange(width, dtype=jnp.int32)

    def evaluate_window(start: jax.Array) -> tuple[BoundedResult[Output], jax.Array]:
        indices = start + offsets
        selected = jnp.minimum(indices, n_rows - 1)
        # The barriers keep XLA from fusing the row body with the ops around it:
        # a fusion can contract a multiply and an add into one rounding, so a
        # row's bits would otherwise depend on whether its window sits inside the
        # scan loop or is inlined beside the map's consumers.
        inputs = jax.lax.optimization_barrier(
            jax.tree.map(lambda leaf: leaf[selected], xs)
        )
        result = jax.vmap(func)(inputs)
        if type(result) is not BoundedResult or not (
            result.bound_status
            or result.query_status
            or result.continuation_read_counts is not None
        ):
            raise TypeError(
                "Bound-summary maps require declared BoundedResult records."
            )
        result = jax.lax.optimization_barrier(result)
        valid = indices < n_rows
        if active is not None:
            valid = valid & active[selected]
        return result, valid

    if n_windows == 1:
        # One window is evaluated directly; the row body still appears once.
        result, valid = evaluate_window(jnp.asarray(0, dtype=jnp.int32))
        return BoundedResult(
            output=result.output,
            bound_status=reduce_bound_input_status(
                statuses=result.bound_status, active=valid
            ),
            query_status=reduce_query_input_status(
                statuses=result.query_status, active=valid
            ),
            continuation_read_counts=reduce_continuation_read_counts(
                counts=result.continuation_read_counts, active=valid
            ),
        )

    # Across several windows every window, the first included, runs inside the
    # scan, so the row body is lowered once. Its abstract result sizes the
    # buffers, and folding a window with no active lane gives the all-false,
    # zero-count identity of every merge.
    # Only leaves leave the abstract evaluation, since an output container may
    # refuse abstract leaves; the zero leaves are then put back into its structure.
    structures = []

    def window_leaves(start: jax.Array) -> list[jax.Array]:
        leaves, structure = jax.tree.flatten(evaluate_window(start)[0])
        structures.append(structure)
        return leaves

    abstract = jax.eval_shape(window_leaves, jnp.asarray(0, dtype=jnp.int32))
    empty = cast(
        "BoundedResult[Output]",
        jax.tree.unflatten(
            structures[0],
            [jnp.zeros(leaf.shape, leaf.dtype) for leaf in abstract],
        ),
    )
    no_lanes = jnp.zeros((width,), dtype=jnp.bool_)
    buffers = cast(
        "Output",
        jax.tree.map(
            lambda leaf: jnp.zeros((n_windows * width, *leaf.shape[1:]), leaf.dtype),
            empty.output,
        ),
    )

    # keyword-only-exempt: library-callback=jax.lax.scan
    def one_window(
        carry: BoundedResult[Output], window: jax.Array
    ) -> tuple[BoundedResult[Output], None]:
        start = window * width
        result, valid = evaluate_window(start)
        output = cast(
            "Output",
            jax.tree.map(
                lambda buffer, leaf: jax.lax.dynamic_update_slice(
                    buffer,
                    leaf,
                    (start, *(jnp.zeros_like(start) for _ in leaf.shape[1:])),
                ),
                carry.output,
                result.output,
            ),
        )
        block = reduce_bound_input_status(statuses=result.bound_status, active=valid)
        statuses = tuple(
            merge_bound_input_status(left=left, right=right)
            for left, right in zip(carry.bound_status, block, strict=True)
        )
        return BoundedResult(
            output=output,
            bound_status=statuses,
            query_status=merge_query_input_status(
                left=carry.query_status,
                right=reduce_query_input_status(
                    statuses=result.query_status, active=valid
                ),
            ),
            continuation_read_counts=merge_continuation_read_counts(
                left=carry.continuation_read_counts,
                right=reduce_continuation_read_counts(
                    counts=result.continuation_read_counts, active=valid
                ),
            ),
        ), None

    initial = BoundedResult(
        output=buffers,
        bound_status=reduce_bound_input_status(
            statuses=empty.bound_status, active=no_lanes
        ),
        query_status=reduce_query_input_status(
            statuses=empty.query_status, active=no_lanes
        ),
        continuation_read_counts=reduce_continuation_read_counts(
            counts=empty.continuation_read_counts, active=no_lanes
        ),
    )
    final, _ = jax.lax.scan(
        one_window,
        vary_carry_like_body(
            body=lambda carry: one_window(carry, jnp.asarray(0, dtype=jnp.int32))[0],
            carry=initial,
        ),
        jnp.arange(n_windows, dtype=jnp.int32),
    )
    return BoundedResult(
        output=cast("Output", jax.tree.map(lambda leaf: leaf[:n_rows], final.output)),
        bound_status=final.bound_status,
        query_status=final.query_status,
        continuation_read_counts=final.continuation_read_counts,
    )


def simulation_spacemap(
    *,
    func: FunctionWithArrayReturn,
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    bound_summary: bool = False,
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
            bound_summary=bound_summary,
        )
    )
    mapped = vmap_1d(
        func=mapped,
        variables=state_names,
        callable_with="only_args",
        bound_summary=bound_summary,
    )

    publish_signature(target=mapped, signature=inspect.signature(mappable_func))

    return cast("FunctionWithArrayReturn", allow_only_kwargs(func=mapped))


def vmap_1d(
    *,
    func: FunctionWithArrayReturn,
    variables: tuple[str, ...],
    callable_with: Literal["only_args", "only_kwargs"] = "only_kwargs",
    co_mapped_in_axes: MappingProxyType[
        ReferenceName, int | MappingProxyType[RegimeName, int | None]
    ]
    | None = None,
    bound_summary: bool = False,
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
        in_axes_for_vmap: list[
            int | MappingProxyType[RegimeName, int | None] | None
        ] = [None] * len(parameters)
        for p in positions:
            in_axes_for_vmap[p] = 0
        for name, axes in co_mapped_in_axes.items():
            in_axes_for_vmap[parameters.index(name)] = axes

        vmapped = vmap(positional_func, in_axes=in_axes_for_vmap)
        if bound_summary:
            if not positions:
                raise TypeError(
                    "Bound-summary co-maps require a named lane coordinate."
                )
            vmapped = _BoundVmap(function=vmapped, lane_position=positions[0])
            publish_signature(target=vmapped, signature=signature)

    if callable_with == "only_kwargs":
        out = allow_only_kwargs(func=vmapped, enforce=False)
    else:
        out = vmapped
    publish_signature(target=out, signature=signature)

    return cast("FunctionWithArrayReturn", out)


@dataclass(frozen=True, kw_only=True, eq=False)
class _BoundVmap:
    """Reduce whole-coordinate temporary flags before leaving the co-map owner."""

    function: Callable[..., PytreeValue]
    lane_position: int

    def __call__(self, *args: PytreeValue) -> BoundedResult[PytreeValue]:
        result = self.function(*args)
        if type(result) is not BoundedResult or not (
            result.bound_status
            or result.query_status
            or result.continuation_read_counts is not None
        ):
            raise TypeError(
                "Bound-summary co-maps require mandatory integrity records."
            )
        extent = cast("jax.Array", args[self.lane_position]).shape[0]
        return BoundedResult(
            output=result.output,
            bound_status=reduce_bound_input_status(
                statuses=result.bound_status,
                active=jnp.ones((extent,), dtype=jnp.bool_),
            ),
            query_status=reduce_query_input_status(
                statuses=result.query_status,
                active=jnp.ones((extent,), dtype=jnp.bool_),
            ),
            continuation_read_counts=None
            if result.continuation_read_counts is None
            else reduce_continuation_read_counts(
                counts=result.continuation_read_counts,
                active=jnp.ones((extent,), dtype=jnp.bool_),
            ),
        )


def productmap(
    *,
    func: FunctionWithArrayReturn,
    variables: tuple[str, ...],
    batch_sizes: dict[str, int],
    bound_summary: bool = False,
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
        bound_summary=bound_summary,
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
    broadcast_variables: tuple[str, ...] = (),
    bound_summary: bool = False,
) -> FunctionWithArrayReturn:
    """Map a bounded window of the C-order Cartesian product of named inputs.

    The static width keyword is consumed by the mapper. Each output leaf recovers
    the original product axes followed by its own trailing axes. Coordinate grids
    remain separate arrays. A window holding multiple prefix cells maps the final
    coordinate separately; other windows decode every coordinate from a flat index.
    Two kinds of variables stay out of the flat cell and use ordinary outer vmaps:

    - untiled variables, whose extent the width does not count;
    - broadcast variables, whose extent the width does count: at a width that
      is a multiple of their product extent and narrower than the whole
      product, each window covers `width // prod(extents)` cells, so a window
      holds `width` points of the whole product. Any computation of `func` that
      does not read a broadcast variable is then evaluated once per cell and
      broadcast along it, where a flat index would batch it along that variable
      as well. At any other width the broadcast variables join the cell, so a
      window still holds exactly `width` points. A window covering the whole
      product maps every variable by its own vmap, which already leaves such a
      computation unbatched along a variable it does not read.

    Their axes are restored to their original positions before the result leaves
    this boundary.
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
    outer_variables = (*untiled_variables, *broadcast_variables)
    if find_duplicates(outer_variables) or set(outer_variables).difference(variables):
        msg = (
            "Untiled and broadcast variables must be distinct subsets of the product "
            "variables."
        )
        raise FunctionDispatchError(msg)
    parameters = [
        parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        for parameter in signature.parameters.values()
    ]
    parameters.append(
        inspect.Parameter(width_keyword, inspect.Parameter.KEYWORD_ONLY, default=1)
    )
    cell_variables = tuple(name for name in variables if name not in outer_variables)
    cell_mapper = _TiledProductMap(
        func=func,
        variables=cell_variables,
        width_keyword=width_keyword,
        bound_summary=bound_summary,
    )
    mapped_signature = signature.replace(parameters=parameters)
    publish_signature(target=cell_mapper, signature=mapped_signature)
    mapped = cast("FunctionWithArrayReturn", cell_mapper)
    if outer_variables:
        mapped = productmap(
            func=mapped,
            variables=outer_variables,
            batch_sizes=dict.fromkeys(outer_variables, 0),
            bound_summary=bound_summary,
        )
        mapped_order = (*outer_variables, *cell_variables)
        if mapped_order != variables:
            restored = _RestoreProductAxisOrder(
                func=mapped,
                axes=tuple(mapped_order.index(name) for name in variables),
                bound_summary=bound_summary,
            )
            publish_signature(target=restored, signature=mapped_signature)
            mapped = cast("FunctionWithArrayReturn", restored)
    if broadcast_variables:
        counted = _CountBroadcastExtentInWidth(
            func=mapped,
            plain=tiled_productmap(
                func=func,
                variables=variables,
                width_keyword=width_keyword,
                untiled_variables=untiled_variables,
                bound_summary=bound_summary,
            ),
            variables=broadcast_variables,
            cell_variables=tuple(
                name for name in variables if name not in untiled_variables
            ),
            width_keyword=width_keyword,
        )
        publish_signature(target=counted, signature=mapped_signature)
        mapped = cast("FunctionWithArrayReturn", counted)
    return mapped


@dataclass(frozen=True, kw_only=True, eq=False)
class _CountBroadcastExtentInWidth:
    """Hold `width` product points per window, broadcasting where the width allows.

    A width that is a multiple of the broadcast variables' product extent and
    narrower than the whole cell product runs the broadcast mapper on
    `width // extent` cells. Any other width runs the plain mapper, whose cell
    includes the broadcast variables: a window covering the whole product gains
    nothing from broadcasting, as its per-variable vmaps already batch a
    computation only along the variables it reads.
    """

    func: Callable[..., PytreeValue]
    """Mapper whose outer vmaps run over the broadcast variables."""
    plain: Callable[..., PytreeValue]
    """Mapper over the same product whose cell includes the broadcast variables."""
    variables: tuple[str, ...]
    """Broadcast variables, each mapped whole around every cell window."""
    cell_variables: tuple[str, ...]
    """Variables the plain mapper's cell spans, the broadcast ones included."""
    width_keyword: str
    """Static keyword naming the width in points of the whole product."""

    def __call__(self, **kwargs: QAndFArg) -> PytreeValue:
        width = kwargs.get(self.width_keyword, 1)
        # An invalid width passes through unchanged for the cell mapper to refuse.
        if type(width) is int and width >= 1:
            extent = math.prod(
                _axis_values(kwargs[name]).shape[0] for name in self.variables
            )
            whole = math.prod(
                _axis_values(kwargs[name]).shape[0] for name in self.cell_variables
            )
            if width % extent != 0 or width >= whole:
                return self.plain(**kwargs)
            kwargs[self.width_keyword] = width // extent
        return self.func(**kwargs)


@dataclass(frozen=True, kw_only=True, eq=False)
class _RestoreProductAxisOrder:
    """Restore the declared state order after mapping untiled axes outside cells."""

    func: Callable[..., PytreeValue]
    """Mapped scalar computation producing outer axes followed by cell axes."""
    axes: tuple[int, ...]
    """Permutation from mapped state axes to the original declared order."""

    bound_summary: bool = False
    """Whether scalar integrity is separate from ordinary product axes."""

    def __call__(self, **kwargs: QAndFArg) -> PytreeValue:
        mapped = self.func(**kwargs)
        restore = partial(_transpose_product_axes, axes=self.axes)
        if self.bound_summary:
            result = cast("BoundedResult[PytreeValue]", mapped)
            return BoundedResult(
                output=jax.tree.map(restore, result.output),
                bound_status=result.bound_status,
                query_status=result.query_status,
                continuation_read_counts=result.continuation_read_counts,
            )
        return jax.tree.map(restore, mapped)


# keyword-only-exempt: library-callback=jax.tree.map
def _transpose_product_axes(value: jax.Array, *, axes: tuple[int, ...]) -> jax.Array:
    """Move state axes without changing a leaf's trailing roles or dtype."""
    return jnp.transpose(value, (*axes, *range(len(axes), value.ndim)))


@dataclass(frozen=True, kw_only=True, eq=False)
class _TiledProductMap:
    """Evaluate separate coordinate grids through a bounded Cartesian window."""

    func: Callable[..., PytreeValue]
    """Unchanged scalar function evaluated at each product coordinate."""
    variables: tuple[str, ...]
    """Product coordinates in canonical C order, outermost first."""
    width_keyword: str
    """Static keyword consumed by this mapping boundary."""

    bound_summary: bool = False
    """Whether coordinate windows separately fold Boolean integrity."""

    def __call__(self, **kwargs: QAndFArg) -> PytreeValue:
        width = kwargs.pop(self.width_keyword, 1)
        if type(width) is not int or width < 1:
            msg = f"Tile width must be a positive static integer, got {width!r}."
            raise ValueError(msg)
        if not self.variables:
            return self.func(**kwargs)
        coordinates = tuple(_axis_values(kwargs[name]) for name in self.variables)
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
        if width >= n_cells:
            return _map_whole_product(
                func=self.func,
                variables=self.variables,
                coordinates=coordinates,
                arguments=arguments,
                bound_summary=self.bound_summary,
            )
        if len(self.variables) > 1 and width // min(width, shape[-1]) > 1:
            return _map_grouped_product(
                func=self.func,
                variables=self.variables,
                coordinates=coordinates,
                shape=shape,
                arguments=arguments,
                width=width,
                bound_summary=self.bound_summary,
            )
        evaluate = _EvaluateTiledCell(
            func=self.func,
            variables=self.variables,
            coordinates=coordinates,
            strides=tuple(math.prod(shape[index + 1 :]) for index in range(len(shape))),
            arguments=arguments,
        )
        mapped = map_over_leading_axis(
            func=evaluate,
            xs=jnp.arange(n_cells, dtype=jnp.int32),
            batch_size=width,
            bound_summary=self.bound_summary,
        )
        return _restore_mapped_product(
            mapped=mapped, shape=shape, bound_summary=self.bound_summary
        )


def _map_grouped_product(
    *,
    func: Callable[..., PytreeValue],
    variables: tuple[str, ...],
    coordinates: tuple[jax.Array, ...],
    shape: tuple[int, ...],
    arguments: MappingProxyType[ReferenceName, QAndFArg],
    width: int,
    bound_summary: bool = False,
) -> PytreeValue:
    """Map a flat prefix and final coordinate with at most two cell batch axes.

    The rectangle's two widths multiply to at most the requested width. Separate
    final-coordinate work can be reused across the prefix, while the decoded
    prefix keeps its scalar computations on one batch axis. The scalar cell and
    every trailing output role retain their original meanings.
    """
    inner_width = min(width, shape[-1])
    outer_width = max(1, width // inner_width)
    tiled_final = _MapOverFinalCoordinate(
        func=func,
        variable=variables[-1],
        coordinate=coordinates[-1],
        width=inner_width,
        bound_summary=bound_summary,
    )
    evaluate = _EvaluateTiledCell(
        func=_final_mapper(mapper=tiled_final, extent=shape[-1]),
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
        bound_summary=bound_summary,
    )
    return _restore_mapped_product(
        mapped=mapped, shape=shape, n_flat_axes=2, bound_summary=bound_summary
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _MapOverFinalCoordinate:
    """Evaluate the final coordinate for one decoded prefix cell."""

    func: Callable[..., PytreeValue]
    """Unchanged scalar cell function."""
    variable: str
    """Name of the final Cartesian coordinate."""
    coordinate: jax.Array
    """Separate final-coordinate grid."""
    width: int
    """Active final-coordinate window within the requested cell budget."""

    bound_summary: bool = False
    """Whether each final-coordinate window owns an integrity reduction."""

    def __call__(self, **kwargs: QAndFArg) -> PytreeValue:
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
            bound_summary=self.bound_summary,
        )


def _map_whole_product(
    *,
    func: Callable[..., PytreeValue],
    variables: tuple[str, ...],
    coordinates: tuple[jax.Array, ...],
    arguments: MappingProxyType[ReferenceName, QAndFArg],
    bound_summary: bool = False,
) -> PytreeValue:
    """Map coordinates whose windows cover them, without decoding a flat index.

    Enumerating a grid by index and gathering every element back out of it is an
    identity, but it replaces a compiled program's grid argument by a computed
    operand, which changes how the backend fuses the consumers that read it. A
    window covering the grid therefore maps the grid itself. The scalar cell sees
    the same values either way, so results are unchanged.
    """
    mapped = productmap(
        func=func,
        variables=variables,
        batch_sizes=dict.fromkeys(variables, 0),
        bound_summary=bound_summary,
    )
    named = dict(zip(variables, coordinates, strict=True))
    return mapped(**arguments, **named)


@dataclass(frozen=True, kw_only=True, eq=False)
class _MapWholeCoordinate:
    """Evaluate the final coordinate over its whole grid for one prefix cell."""

    func: Callable[..., PytreeValue]
    """Unchanged scalar cell function."""
    variable: str
    """Name of the final Cartesian coordinate."""
    coordinate: jax.Array
    """Separate final-coordinate grid, mapped in full."""

    bound_summary: bool = False
    """Whether ordinary rows and scalar validity have separate mapped roles."""

    def __call__(self, **kwargs: QAndFArg) -> PytreeValue:
        return _map_whole_product(
            func=self.func,
            variables=(self.variable,),
            coordinates=(self.coordinate,),
            arguments=MappingProxyType(kwargs),
            bound_summary=self.bound_summary,
        )


def _final_mapper(
    *, mapper: _MapOverFinalCoordinate, extent: int
) -> _MapOverFinalCoordinate | _MapWholeCoordinate:
    """Drop the final coordinate's flat decode once its window covers the grid."""
    if mapper.width < extent:
        return mapper
    return _MapWholeCoordinate(
        func=mapper.func,
        variable=mapper.variable,
        coordinate=mapper.coordinate,
        bound_summary=mapper.bound_summary,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _EvaluateTiledCell:
    """Decode one flat index and evaluate the unchanged scalar cell function."""

    func: Callable[..., PytreeValue]
    """Scalar cell function receiving the decoded coordinate values."""
    variables: tuple[str, ...]
    """Coordinate names in the same order as their separate arrays."""
    coordinates: tuple[jax.Array, ...]
    """Separate coordinate arrays, without a materialized product mesh."""
    strides: tuple[int, ...]
    """C-order integer strides used to decode a flat cell index."""
    arguments: MappingProxyType[ReferenceName, QAndFArg]
    """Non-coordinate arguments forwarded unchanged to the cell function."""

    def __call__(self, index: jax.Array) -> PytreeValue:
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


def _restore_mapped_product(
    *,
    mapped: PytreeValue,
    shape: tuple[int, ...],
    bound_summary: bool,
    n_flat_axes: int = 1,
) -> PytreeValue:
    """Restore ordinary Cartesian axes while leaving reduced integrity scalar."""
    restore = partial(_restore_product_axes, shape=shape, n_flat_axes=n_flat_axes)
    if bound_summary:
        result = cast("BoundedResult[PytreeValue]", mapped)
        return BoundedResult(
            output=jax.tree.map(restore, result.output),
            bound_status=result.bound_status,
            query_status=result.query_status,
            continuation_read_counts=result.continuation_read_counts,
        )
    return jax.tree.map(restore, mapped)


def _base_productmap_batched(
    *,
    func: FunctionWithArrayReturn,
    product_axes: tuple[str, ...],
    batch_sizes: dict[str, int],
    bound_summary: bool = False,
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
            bound_summary=bound_summary,
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

    func: Callable[..., PytreeValue]
    """The function evaluated at every point of the product."""
    product_axes: tuple[str, ...]
    """Names of the arguments whose values span the product, outermost first."""
    batch_sizes: MappingProxyType[str, int]
    """The `jax.lax.map` batch size per product axis, `0` for one vectorized pass."""

    bound_summary: bool = False
    """Whether each axis folds integrity before ordinary output storage."""

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

    def __call__(self, **kwargs: PytreeValue) -> PytreeValue:
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
                    axis_values=_axis_values(kwargs[axis]),
                    batch_size=self.batch_sizes[axis],
                    bound_summary=self.bound_summary,
                ),
            )
        return loop_func()


@dataclass(frozen=True, eq=False)
class _MappedOverOneMoreAxis:
    """`loop_func` mapped with `jax.lax.map` over the values of one product axis."""

    loop_func: Callable[..., PytreeValue]
    """The function evaluated once per value of `axis`."""
    axis: str
    """The argument of `loop_func` that takes one value of the axis per evaluation."""
    axis_values: ValueND
    """The values of the axis, mapped over their leading dimension."""
    batch_size: int
    """The `jax.lax.map` batch size, `0` for one vectorized pass."""

    bound_summary: bool = False
    """Whether integrity has scalar ownership rather than a product axis."""

    def __call__(
        self,
        *already_mapped_args: ValueND,
        **already_mapped_kwargs: ValueND,
    ) -> PytreeValue:
        evaluate = partial(
            _evaluate_at_axis_value,
            loop_func=self.loop_func,
            axis=self.axis,
            mapped_args=already_mapped_args,
            mapped_kwargs=already_mapped_kwargs,
        )
        if self.bound_summary:
            return map_over_leading_axis(
                func=evaluate,
                xs=jnp.atleast_1d(self.axis_values),
                batch_size=self.batch_size,
                bound_summary=True,
            )
        return jax.lax.map(
            evaluate, jnp.atleast_1d(self.axis_values), batch_size=self.batch_size
        )


def _axis_values(value: PytreeValue) -> ValueND:
    """Return the values of a product axis with at least one leading dimension."""
    if isinstance(value, jax.Array | np.ndarray | np.bool | np.number | int | float):
        return jnp.atleast_1d(value)
    msg = f"A product axis takes an array or a number, got {type(value).__name__}."
    raise FunctionDispatchError(msg)


# keyword-only-exempt: library-callback=jax.lax.map
def _evaluate_at_axis_value(
    axis_value: ValueND,
    *,
    loop_func: Callable[..., PytreeValue],
    axis: str,
    mapped_args: tuple[ValueND, ...],
    mapped_kwargs: dict[ReferenceName, ValueND],
) -> PytreeValue:
    """Evaluate `loop_func` at one value of `axis`, forwarding the other arguments."""
    return loop_func(*mapped_args, **{axis: axis_value}, **mapped_kwargs)
