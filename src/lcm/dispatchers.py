from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal, TypeVar, cast

from jax import Array, vmap

from lcm.functools import allow_args, allow_only_kwargs
from lcm.utils import find_duplicates

FunctionWithArrayReturn = TypeVar(
    "FunctionWithArrayReturn",
    bound=Callable[..., Array | tuple[Array, Array] | dict[str, Array]],
)


def simulation_spacemap(
    func: FunctionWithArrayReturn,
    actions_names: tuple[str, ...],
    states_names: tuple[str, ...],
) -> FunctionWithArrayReturn:
    """Apply vmap such that func can be evaluated on actions and simulation states.

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
        actions_names: Names of the action variables.
        states_names: Names of the state variables.

    Returns:
        A callable with the same arguments as func (but with an additional leading
        dimension) that returns an Array or pytree of Arrays. If `func` returns a
        scalar, the dispatched function returns an Array with k + 1 dimensions, where k
        is the length of `actions_names` and the additional dimension corresponds to the
        `states_names`. The order of the dimensions is determined by the order of
        `actions_names`. If the output of `func` is a jax pytree, the usual jax behavior
        applies, i.e. the leading dimensions of all arrays in the pytree are as
        described above but there might be additional dimensions.

    """
    # The model creation process ensures that in a user-created model the following
    # cannot happen. We double-check here to ensure that the post-processing does not
    # accidentally create such a situation.
    if duplicates := find_duplicates(actions_names, states_names):
        msg = (
            "Same argument provided more than once in actions or states variables, "
            f"or is present in both: {duplicates}"
        )
        raise ValueError(msg)

    mappable_func = allow_args(func)

    vmapped = _base_productmap(mappable_func, actions_names)
    vmapped = vmap_1d(vmapped, variables=states_names, callable_with="only_args")

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = inspect.signature(mappable_func)  # type: ignore[attr-defined]

    return cast("FunctionWithArrayReturn", allow_only_kwargs(vmapped))


def vmap_1d(
    func: FunctionWithArrayReturn,
    variables: tuple[str, ...],
    *,
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

    # Create in_axes to apply vmap over variables. This has one entry for each argument
    # of func, indicating whether the argument should be mapped over or not. None means
    # that the argument should not be mapped over, 0 means that it should be mapped over
    # the leading axis of the input.
    in_axes_for_vmap = [None] * len(parameters)  # type: list[int | None]
    for p in positions:
        in_axes_for_vmap[p] = 0

    vmapped = vmap(func, in_axes=in_axes_for_vmap)

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = signature  # type: ignore[attr-defined]

    if callable_with == "only_kwargs":
        out = allow_only_kwargs(vmapped)
    elif callable_with == "only_args":
        out = vmapped
    else:
        raise ValueError(
            f"Invalid callable_with option: {callable_with}. Possible options are "
            "('only_args', 'only_kwargs')",
        )

    return cast("FunctionWithArrayReturn", out)


def productmap(
    func: FunctionWithArrayReturn, variables: tuple[str, ...]
) -> FunctionWithArrayReturn:
    """Apply vmap such that func is evaluated on the Cartesian product of variables.

    This is achieved by an iterative application of vmap.

    In contrast to _base_productmap, productmap preserves the function signature and
    allows the function to be called with keyword arguments.

    Args:
        func: The function to be dispatched.
        variables: Tuple with names of arguments that over which the Cartesian product
            should be formed.

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

    vmapped = _base_productmap(func_callable_with_args, variables)

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = inspect.signature(func_callable_with_args)  # type: ignore[attr-defined]

    return cast("FunctionWithArrayReturn", allow_only_kwargs(vmapped))


def _base_productmap(
    func: FunctionWithArrayReturn, product_axes: tuple[str, ...]
) -> FunctionWithArrayReturn:
    """Map func over the Cartesian product of product_axes.

    Like vmap, this function does not preserve the function signature and does not allow
    the function to be called with keyword arguments.

    Args:
        func: The function to be dispatched. Cannot have keyword-only arguments.
        product_axes: Tuple with names of arguments over which we apply vmap.

    Returns:
        A callable with the same arguments as func. See `product_map` for details.

    """
    signature = inspect.signature(func)
    parameters = list(signature.parameters)

    positions = [parameters.index(ax) for ax in product_axes if ax in parameters]

    vmap_specs = []
    # We iterate in reverse order such that the output dimensions are in the same order
    # as the input dimensions.
    for pos in reversed(positions):
        spec: list[int | None] = [None] * len(parameters)
        spec[pos] = 0
        vmap_specs.append(spec)

    vmapped = func
    for spec in vmap_specs:
        vmapped = vmap(vmapped, in_axes=spec)

    return vmapped
