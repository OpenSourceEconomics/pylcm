import inspect
from collections.abc import Callable
from functools import partial
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast

import jax
import jax.numpy as jnp
from jax import vmap

from _lcm.typing import ActionName, StateName
from _lcm.utils.containers import find_duplicates
from _lcm.utils.functools import allow_args, allow_only_kwargs
from lcm.exceptions import FunctionDispatchError
from lcm.typing import BoolND, FloatND, IntND

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


def simulation_spacemap(
    *,
    func: FunctionWithArrayReturn,
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    subjects_batch_size: int = 0,
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
        subjects_batch_size: Per-device chunk size for the per-subject vmap. `0`
            (default) keeps a single big vmap over the entire subjects axis. `>0`
            replaces the vmap with `jax.lax.map(..., batch_size=subjects_batch_size)`
            so each device's local shard is iterated in chunks of this size, with
            JAX handling any non-divisible remainder.

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
    if subjects_batch_size > 0:
        mapped = chunked_map_1d(
            func=mapped,
            variables=state_names,
            batch_size=subjects_batch_size,
            callable_with="only_args",
        )
    else:
        mapped = vmap_1d(func=mapped, variables=state_names, callable_with="only_args")

    # Callables do not necessarily have a __signature__ attribute.
    mapped.__signature__ = inspect.signature(mappable_func)  # ty: ignore[unresolved-attribute]

    return cast("FunctionWithArrayReturn", allow_only_kwargs(mapped))


def vmap_1d(
    *,
    func: FunctionWithArrayReturn,
    variables: tuple[str, ...],
    callable_with: Literal["only_args", "only_kwargs"] = "only_kwargs",
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

    # Handle empty variables case - nothing to vmap over
    if not positions:
        vmapped = func
    else:
        # Create in_axes to apply vmap over variables. This has one entry for each
        # argument of func, indicating whether the argument should be mapped over or
        # not. None means that the argument should not be mapped over, 0 means that it
        # should be mapped over the leading axis of the input.
        in_axes_for_vmap: list[int | None] = cast(
            "list[int | None]", [None] * len(parameters)
        )
        for p in positions:
            in_axes_for_vmap[p] = 0

        vmapped = vmap(func, in_axes=in_axes_for_vmap)
    vmapped.__signature__ = signature  # ty: ignore[invalid-assignment]

    if callable_with == "only_kwargs":
        out = allow_only_kwargs(vmapped, enforce=False)
    else:
        out = vmapped

    return cast("FunctionWithArrayReturn", out)


def chunked_map_1d(
    *,
    func: FunctionWithArrayReturn,
    variables: tuple[str, ...],
    batch_size: int,
    callable_with: Literal["only_args", "only_kwargs"] = "only_kwargs",
) -> FunctionWithArrayReturn:
    """Apply `jax.lax.map` so func is mapped over the leading axis of `variables` in
    chunks of size `batch_size`.

    Same calling contract as `vmap_1d` (parameter named in `variables` are mapped
    along their leading axis simultaneously; the rest are closed over), but the
    leading axis is iterated in chunks of `batch_size` via `jax.lax.map`. Within
    each chunk JAX vmaps; across chunks it scans. JAX handles a non-divisible
    leading dim by making the last chunk smaller.

    Used by `simulation_spacemap` when `subjects_batch_size > 0` so each device's
    per-subject simulate dispatch is iterated in fixed-size chunks — shrinks the
    per-iteration intermediate by `n_per_device / batch_size` and lets production
    grid sizes fit within per-device HBM budgets.

    Args:
        func: The function to be dispatched.
        variables: Tuple with names of arguments mapped along their leading axis.
        batch_size: Chunk size for `jax.lax.map`. Must be >= 1.
        callable_with: Whether the returned function accepts only positional or
            only keyword arguments.

    Returns:
        A callable with the same arguments as `func` (with an additional leading
        dimension on each mapped variable) that returns a jax.Array or pytree of
        arrays with that leading dimension preserved.

    """
    if duplicates := find_duplicates(variables):
        raise ValueError(
            f"Same argument provided more than once in variables: {duplicates}",
        )

    signature = inspect.signature(func)
    parameters = list(signature.parameters)

    if not variables:
        # Nothing to map over — match vmap_1d's empty-vars behaviour.
        mapped = func
    else:
        positions = tuple(parameters.index(var) for var in variables)

        def chunked(*args: Any) -> Any:  # noqa: ANN401
            # Split args: variables to chunk over go through jax.lax.map; the
            # rest are closed over as broadcast arguments.
            mapped_args = tuple(args[p] for p in positions)

            def one_step(chunk_slice: tuple[Any, ...]) -> Any:  # noqa: ANN401
                full = list(args)
                for i, p in enumerate(positions):
                    full[p] = chunk_slice[i]
                return func(*full)

            return jax.lax.map(one_step, mapped_args, batch_size=batch_size)

        mapped = chunked

    mapped.__signature__ = signature  # ty: ignore[invalid-assignment]

    if callable_with == "only_kwargs":
        out = allow_only_kwargs(mapped, enforce=False)
    else:
        out = mapped

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
    mapped.__signature__ = new_signature  # ty: ignore[unresolved-attribute]

    return cast("FunctionWithArrayReturn", allow_only_kwargs(mapped, enforce=False))


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

    def batched_vmap(**kwargs: Any) -> Any:  # noqa: ANN401
        # `batched_vmap` is a generic helper: it accepts whatever values the
        # composed `func` expects (canonical JAX arrays in the production
        # pipeline, but also Python scalars, non-canonical-dtype arrays, or
        # `MappingProxyType` containers in callers that wrap their own pytrees)
        # and returns whatever `func` returns. Beartype shouldn't constrain
        # the shape here — the wrapped `func` is responsible for its own
        # contract.
        non_array_kwargs = {
            key: val for key, val in kwargs.items() if key not in product_axes
        }
        func_with_partialled_args = cast(
            "FunctionWithArrayReturn", partial(func, **non_array_kwargs)
        )

        # Recursively map over one more product axis
        def map_one_more(
            loop_func: FunctionWithArrayReturn, axis: str
        ) -> FunctionWithArrayReturn:
            def func_mapped_over_one_more_axis(
                *already_mapped_args: Any,  # noqa: ANN401
                **already_mapped_kwargs: Any,  # noqa: ANN401
            ) -> Any:  # noqa: ANN401
                return jax.lax.map(
                    lambda axis_i: loop_func(
                        *already_mapped_args, **{axis: axis_i}, **already_mapped_kwargs
                    ),
                    jnp.atleast_1d(kwargs[axis]),
                    batch_size=batch_sizes[axis],
                )

            return cast("FunctionWithArrayReturn", func_mapped_over_one_more_axis)

        # Loop over all product axes
        for axis in reversed(product_axes):
            func_with_partialled_args = map_one_more(func_with_partialled_args, axis)

        return cast("FloatND", func_with_partialled_args())

    return cast("FunctionWithArrayReturn", batched_vmap)
