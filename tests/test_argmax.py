from collections.abc import Callable, Iterator
from functools import partial
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, device_put, devices, jit, make_jaxpr
from jax.core import ShapedArray
from jax.extend.core import Jaxpr
from numpy.testing import assert_array_equal

from _lcm.execution.reductions import HARD_MAX_WITH_CARRY_REDUCTION
from _lcm.regime_building.argmax import (
    NO_ID,
    _flatten_last_n_axes,
    _float_infinity_key,
    _float_order_key,
    _move_axes_to_back,
    _reduce_float_keys,
    max_and_smallest_id,
)
from _lcm.regime_building.collective import (
    collective_argmax_and_readout,
    decider_choice,
)
from _lcm.solution.collective_action_reduction import COLLECTIVE_HARD_MAX_REDUCTION


def test_move_axes_to_back_1d() -> None:
    a = jnp.arange(4, dtype=jnp.int32)
    got = _move_axes_to_back(a=a, axes=(0,))
    assert_array_equal(got, a)


def test_move_axes_to_back_2d() -> None:
    a = jnp.arange(4, dtype=jnp.int32).reshape(2, 2)
    got = _move_axes_to_back(a=a, axes=(0,))
    assert_array_equal(got, a.transpose(1, 0))


def test_move_axes_to_back_3d() -> None:
    # 2 dimensions in back
    a = jnp.arange(8, dtype=jnp.int32).reshape(2, 2, 2)
    got = _move_axes_to_back(a=a, axes=(0, 1))
    assert_array_equal(got, a.transpose(2, 0, 1))

    # 2 dimensions in front
    a = jnp.arange(8, dtype=jnp.int32).reshape(2, 2, 2)
    got = _move_axes_to_back(a=a, axes=(1,))
    assert_array_equal(got, a.transpose(0, 2, 1))


def test_flatten_last_n_axes_1d() -> None:
    a = jnp.arange(4, dtype=jnp.int32)
    got = _flatten_last_n_axes(a=a, n=1)
    assert_array_equal(got, a)


def test_flatten_last_n_axes_2d() -> None:
    a = jnp.arange(4, dtype=jnp.int32).reshape(2, 2)

    got = _flatten_last_n_axes(a=a, n=1)
    assert_array_equal(got, a)

    got = _flatten_last_n_axes(a=a, n=2)
    assert_array_equal(got, a.reshape(4))


def test_flatten_last_n_axes_3d() -> None:
    a = jnp.arange(8, dtype=jnp.int32).reshape(2, 2, 2)

    got = _flatten_last_n_axes(a=a, n=1)
    assert_array_equal(got, a)

    got = _flatten_last_n_axes(a=a, n=2)
    assert_array_equal(got, a.reshape(2, 4))

    got = _flatten_last_n_axes(a=a, n=3)
    assert_array_equal(got, a.reshape(8))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_singleton_pair_keeps_the_identity_of_the_winning_candidate(
    *, compiled: bool
) -> None:
    """The floor has no candidate ID and loses an equal-value tie to a real ID."""
    reduce = (
        jit(partial(max_and_smallest_id, initial=-2.0))
        if compiled
        else partial(max_and_smallest_id, initial=-2.0)
    )
    maximum, index = reduce(
        values=jnp.asarray([[-8.0], [-2.0], [4.0]]),
        ids=jnp.asarray([[7], [5], [3]], dtype=jnp.int32),
    )
    assert_array_equal(maximum, [-2.0, -2.0, 4.0])
    assert_array_equal(index, [NO_ID, 5, 3])


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_singleton_floor_preserves_subnormal_bits_and_identity(
    *, compiled: bool
) -> None:
    """The zero floor admits positive subnormals and rejects negative subnormals."""
    dtype = np.dtype(jnp.asarray(0.0).dtype)
    width = dtype.itemsize * 8
    fraction_bits = 23 if width == 32 else 52
    sign = 1 << (width - 1)
    uint = np.dtype(f"uint{width}")
    magnitudes = [1, 2, (1 << fraction_bits) - 1, 1 << fraction_bits]
    encoded = np.asarray(
        magnitudes + [sign | value for value in magnitudes] + [sign, 0],
        dtype=uint,
    ).reshape(-1, 1)
    values = jnp.asarray(encoded.view(dtype))
    ids = jnp.full(values.shape, 7, dtype=jnp.int32)
    reduce = partial(max_and_smallest_id, initial=0.0)
    if compiled:
        reduce = jit(reduce)
    maximum, identity = reduce(values=values, ids=ids)
    expected_bits = np.asarray(magnitudes + [0] * (len(magnitudes) + 2), dtype=uint)
    assert_array_equal(np.asarray(maximum).view(uint), expected_bits)
    assert_array_equal(identity, [7] * 4 + [NO_ID] * 4 + [7, 7])


def test_paired_combiner_does_not_recompute_float_order_keys() -> None:
    """Bit ordering happens once, before the paired reduction, not in its combiner."""
    values = jnp.asarray([[2.0, -1.0, 3.0]])
    ids = jnp.asarray([[7, 3, 9]], dtype=jnp.int32)
    program = make_jaxpr(partial(max_and_smallest_id, initial=-jnp.inf))(
        values=values, ids=ids
    ).jaxpr
    reductions = [
        eqn
        for nested in _nested_jaxprs(program)
        for eqn in nested.eqns
        if eqn.primitive.name == "reduce"
    ]
    repeated_order_work = sum(
        eqn.primitive.name in {"optimization_barrier", "shift_right_arithmetic", "xor"}
        for reduction in reductions
        for nested in _nested_jaxprs(reduction.params["jaxpr"])
        for eqn in nested.eqns
    )
    assert (len(reductions), repeated_order_work) == (1, 0)


def _nested_jaxprs(jaxpr: Jaxpr) -> Iterator[Jaxpr]:
    yield jaxpr
    for eqn in jaxpr.eqns:
        for value in eqn.params.values():
            nested = getattr(value, "jaxpr", value)
            if isinstance(nested, Jaxpr):
                yield from _nested_jaxprs(nested)


def _key_reader_counts(jaxpr: Jaxpr) -> list[int]:
    """Per paired `(key, id)` reduction, count the reads of its key operand."""
    return [
        sum(other.invars.count(eqn.invars[0]) for other in nested.eqns)
        + nested.outvars.count(eqn.invars[0])
        for nested in _nested_jaxprs(jaxpr)
        for eqn in nested.eqns
        if eqn.primitive.name == "reduce"
    ]


def _keys_read_by_reduction_and_tie_mask(values: Array) -> tuple[Array, Array]:
    """Reduce the order keys and also compare them: a second reader of the keys."""
    keys = _float_order_key(values)
    best_key, _ = _reduce_float_keys(
        keys=keys,
        ids=jnp.zeros(values.shape, dtype=jnp.int32),
        initial_key=_float_order_key(jnp.asarray(-jnp.inf, dtype=values.dtype)),
        infinity_key=_float_infinity_key(values.dtype),
    )
    return best_key, jnp.sum(keys == best_key[..., jnp.newaxis], axis=-1)


_ACTION_IDS = jnp.arange(4, dtype=jnp.int32)

_PAIRED_REDUCTION_CALLERS = {
    "max_and_smallest_id": lambda v: max_and_smallest_id(
        values=v, ids=jnp.broadcast_to(_ACTION_IDS, v.shape), initial=0.0
    ),
    "collective_action_reduction": lambda v: COLLECTIVE_HARD_MAX_REDUCTION.add(
        accumulator=COLLECTIVE_HARD_MAX_REDUCTION.initialize(member_template=v[:, :2]),
        objectives=v,
        member_values=jnp.stack([v, -v], axis=-1),
        feasible=v > 0,
        action_ids=_ACTION_IDS,
    ),
    "reduction_fold": lambda v: HARD_MAX_WITH_CARRY_REDUCTION.finalize(
        accumulator=HARD_MAX_WITH_CARRY_REDUCTION.add(
            accumulator=HARD_MAX_WITH_CARRY_REDUCTION.initialize(
                value_template=v[:, 0]
            ),
            values=v,
            feasible=v > 0,
            action_ids=_ACTION_IDS,
        )
    ),
}


@pytest.mark.parametrize(
    ("caller", "expected"),
    [
        *(
            # The collective block is reduced, then merged into the accumulator
            # by a second paired reduction over the two partial winners.
            (name, [1, 1] if name == "collective_action_reduction" else [1])
            for name in _PAIRED_REDUCTION_CALLERS
        ),
        ("second-reader", [2]),
    ],
)
def test_float_order_keys_have_exactly_one_reader(
    *, caller: str, expected: list[int]
) -> None:
    """In the traced program, each paired reduction is the only reader of its keys.

    This is what lets a reduction over more than one element compute the keys
    inside itself without an optimization barrier. It says nothing about the
    compiled program: a reduction over one element is simplified into separate
    elementwise programs, which the lowering tests below cover. The
    `second-reader` case reads the keys in a reduction and a tie mask; it shows
    that the count sees a second reader.
    """
    function = _PAIRED_REDUCTION_CALLERS.get(
        caller, _keys_read_by_reduction_and_tie_mask
    )
    jaxpr = make_jaxpr(function)(jnp.linspace(-1.0, 1.0, 12).reshape(3, 4)).jaxpr
    assert _key_reader_counts(jaxpr) == expected


@pytest.mark.parametrize("width", [1, 2, 4])
def test_paired_reduction_holds_keys_behind_a_barrier_only_over_one_element(
    *, width: int
) -> None:
    """The lowered paired reduction has a barrier exactly when it reduces one element.

    Over several elements the compiler may compute the keys inside the reduction
    instead of writing a full-width integer array first.
    """
    values = jnp.linspace(-1.0, 1.0, 3 * width).reshape(3, width)
    text = (
        jit(partial(max_and_smallest_id, initial=-jnp.inf))
        .lower(values=values, ids=jnp.zeros(values.shape, dtype=jnp.int32))
        .as_text()
    )
    assert ("optimization_barrier" in text) == (width == 1)


@pytest.mark.parametrize("width", [1, 4])
def test_paired_reduction_evaluates_an_in_graph_producer_once(*, width: int) -> None:
    """The maximum and its identity read one evaluation of the reduced values.

    The values come from an in-graph `sin`; each `sine` in the optimized program
    is one place the values are evaluated. With a single element and a finite
    floor the compiler splits the reduction into elementwise programs for the
    maximum and for the identity, and two evaluations that rounded differently
    could publish a maximum that its identity does not attain. Fusion decisions
    are backend-specific: the count is pinned for the CPU lowering only.
    """

    def reduce(x: Array) -> tuple[Array, Array]:
        values = jnp.sin(x) * x + 0.5
        ids = jnp.broadcast_to(jnp.arange(width, dtype=jnp.int32), values.shape)
        return max_and_smallest_id(values=values, ids=ids, initial=0.0)

    x = device_put(
        jnp.linspace(-3.0, 3.0, 64 * width).reshape(64, width), devices("cpu")[0]
    )
    program = jit(reduce).lower(x).compile().as_text()
    assert program is not None
    assert program.count(" sine(") == 1


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_paired_max_rejects_unsupported_float_widths(
    *, compiled: bool, dtype: str
) -> None:
    """Reject narrower formats explicitly instead of interpreting their bit layouts."""
    values = jnp.asarray([1, 2], dtype=dtype)
    assert values.dtype.itemsize == 1
    reduce = partial(max_and_smallest_id, initial=0.0)
    if compiled:
        reduce = jit(reduce)
    with pytest.raises(ValueError, match="16-, 32- and 64-bit IEEE"):
        reduce(values=values, ids=jnp.asarray([7, 3], dtype=jnp.int32))


def _dense_household_readout(v: Array) -> tuple[Array, ...]:
    index, values, dissolution = collective_argmax_and_readout(
        member_Q={"f": v, "m": -v},
        feasibility=v > 0,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(1,),
    )
    return index, values["f"], values["m"], dissolution


def _decider_choice(v: Array) -> tuple[Array, ...]:
    return decider_choice(
        Q=jnp.stack([v, -v], axis=-1),
        feasibility=v > 0,
        weights=jnp.asarray([1.0, 0.0]),
        n_discrete_action_axes=1,
    )


def _streamed_household_block(v: Array) -> tuple[Array, ...]:
    return COLLECTIVE_HARD_MAX_REDUCTION.add(
        accumulator=COLLECTIVE_HARD_MAX_REDUCTION.initialize(member_template=v[:, :2]),
        objectives=v,
        member_values=jnp.stack([v, -v], axis=-1),
        feasible=v > 0,
        action_ids=_ACTION_IDS,
    )


_REDUCTION_PRIMITIVES = frozenset(
    {"reduce", "reduce_or", "reduce_and", "reduce_max", "reduce_min", "reduce_sum"}
)


def _action_axis_reductions(*, jaxpr: Jaxpr, n_actions: int) -> int:
    """Count the reductions over a trailing action axis of `n_actions` elements."""
    count = 0
    for nested in _nested_jaxprs(jaxpr):
        for eqn in nested.eqns:
            shape = cast("ShapedArray", eqn.invars[0].aval).shape if eqn.invars else ()
            count += (
                eqn.primitive.name in _REDUCTION_PRIMITIVES
                and shape[-1:] == (n_actions,)
                and len(shape) - 1
                in eqn.params.get("dimensions", eqn.params.get("axes", ()))
            )
    return count


@pytest.mark.parametrize(
    "reducer",
    [_dense_household_readout, _decider_choice, _streamed_household_block],
    ids=["dense-readout", "decider-choice", "streamed-block"],
)
def test_household_reduction_reads_the_action_values_once(
    *, reducer: Callable[[Array], tuple[Array, ...]]
) -> None:
    """One reduction over the action axis carries the NaN rule and feasibility.

    A second reduction over the actions, for the NaN rule or for whether any
    action is feasible, is a second reader of the action values, and the
    compiler then stores them for every cell and action.
    """
    jaxpr = make_jaxpr(reducer)(jnp.linspace(-1.0, 1.0, 12).reshape(3, 4)).jaxpr

    assert _action_axis_reductions(jaxpr=jaxpr, n_actions=4) == 1
