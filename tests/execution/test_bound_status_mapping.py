"""Owning map windows fold mandatory integrity before storing ordinary outputs."""

import inspect
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype
from beartype.roar import (
    BeartypeCallHintParamViolation,
    BeartypeCallHintReturnViolation,
)

from _lcm.execution.output_layout import StateAxesLeading
from _lcm.solution import feasibility
from _lcm.utils.dispatchers import map_over_leading_axis, productmap, tiled_productmap
from lcm.typing import BoolND, FloatND


@pytest.mark.parametrize("container", ["text", "duck-tuple"])
def test_bound_result_tree_transport_rejects_wrong_nominal_records(
    *, container: str
) -> None:
    """Typed tree transport rejects another container before accessing its fields."""
    wrong = (
        "invalid record"
        if container == "text"
        else ((jnp.asarray(1.0), jnp.asarray(a=True)), ())
    )
    invalid = cast("feasibility.BoundedResult[tuple[FloatND, BoolND]]", wrong)
    with pytest.raises(BeartypeCallHintParamViolation):
        feasibility._flatten_bounded_result(result=invalid)


def test_bound_result_return_contract_rejects_a_duck_tuple() -> None:
    """An ordinary tuple cannot replace the declared nominal result on return."""

    @beartype
    def produce() -> feasibility.BoundedResult[tuple[FloatND, BoolND]]:
        return cast(
            "feasibility.BoundedResult[tuple[FloatND, BoolND]]",
            ((jnp.asarray(1.0), jnp.asarray(a=True)), ()),
        )

    with pytest.raises(BeartypeCallHintReturnViolation):
        produce()


def test_bound_result_tree_keeps_inert_metadata_and_abstract_role_structure() -> None:
    """
    One nominal result restores concrete, abstract and role leaves with inert
    static metadata.
    """
    result = feasibility.BoundedResult(
        output=(jnp.asarray([1.0, 2.0]), jnp.asarray([True, False])),
        bound_status=(
            feasibility.BoundInputStatus(
                nonfinite_floor=jnp.asarray(a=False),
                invalid_shifted_grid=jnp.asarray(a=True),
            ),
        ),
    )
    children, aux = feasibility._flatten_bounded_result(result)
    keyed, keyed_aux = feasibility._flatten_bounded_result_with_keys(result)
    assert len(children) == len(keyed) == 2
    assert children[0] is keyed[0][1] is result.output
    assert children[1] is keyed[1][1] is result.bound_status
    assert tuple(key.name for key, _ in keyed) == ("output", "bound_status")
    assert aux is keyed_aux is None
    restored = feasibility._unflatten_bounded_result(aux, children)
    assert type(restored) is feasibility.BoundedResult
    assert restored.output is result.output
    assert restored.bound_status is result.bound_status
    leaves, definition = jax.tree.flatten(result)
    assert len(leaves) == 4
    abstract = jax.tree.unflatten(
        definition, [jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for leaf in leaves]
    )
    assert type(abstract) is feasibility.BoundedResult
    assert abstract.output[0].shape == (2,)
    assert abstract.output[1].dtype == jnp.bool_
    assert abstract.bound_status[0].nonfinite_floor.shape == ()
    abstract_children, abstract_aux = feasibility._flatten_bounded_result(abstract)
    abstract_keyed, abstract_keyed_aux = feasibility._flatten_bounded_result_with_keys(
        abstract
    )
    assert abstract_children[0] is abstract_keyed[0][1] is abstract.output
    assert abstract_children[1] is abstract_keyed[1][1] is abstract.bound_status
    assert tuple(key.name for key, _ in abstract_keyed) == ("output", "bound_status")
    assert abstract_aux is abstract_keyed_aux is None
    assert len(jax.tree.leaves(abstract)) == 4
    roles = [
        StateAxesLeading(state_names=(), dtype=leaf.dtype, shape=leaf.shape)
        for leaf in leaves
    ]
    role_tree = jax.tree.unflatten(definition, roles)
    assert type(role_tree) is feasibility.BoundedResult
    assert jax.tree.leaves(role_tree) == roles


@pytest.mark.parametrize("malformed", ["action-bank", "integer"])
def test_bound_reduction_requires_one_boolean_per_owned_lane(*, malformed: str) -> None:
    """Unreduced action banks and numeric flags cannot impersonate lane summaries."""
    flags = (
        jnp.asarray([[False, True], [False, False]])
        if malformed == "action-bank"
        else jnp.asarray([0, 1], dtype=jnp.int32)
    )
    status = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(
            feasibility.BoundInputStatus(
                nonfinite_floor=jnp.asarray(a=False),
                invalid_shifted_grid=jnp.asarray(a=False),
            )
        ),
        [flags, jnp.zeros_like(flags)],
    )
    with pytest.raises(TypeError, match=r"Boolean.*owned lane"):
        feasibility.reduce_bound_input_status(
            statuses=(status,),
            active=jnp.asarray([True, False]),
        )


@pytest.mark.parametrize("width", [0, 1, 2, 8])
@pytest.mark.parametrize("active_last", [False, True])
def test_map_windows_reduce_bound_integrity_independently_of_F(
    *, width: int, active_last: bool
) -> None:
    """Partial and whole windows return scalar validity over actual active rows."""
    record_type = getattr(feasibility, "BoundedResult", None)
    assert record_type is not None, "Mapped outputs need separate bounded summaries."
    code = inspect.unwrap(map_over_leading_axis).__code__
    assert (
        "bound_summary" in code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    ), "The owning map must reduce validity before concatenating ordinary outputs."

    def body(value: FloatND) -> feasibility.BoundedResult[tuple[FloatND, BoolND]]:
        return record_type(
            output=(value * value, value >= 0.0),
            bound_status=(
                feasibility.summarize_bound_inputs(
                    floor_values=value,
                    active=jnp.asarray(a=True),
                ),
            ),
        )

    def evaluate(
        *, values: FloatND, active: BoolND
    ) -> feasibility.BoundedResult[tuple[FloatND, BoolND]]:
        return map_over_leading_axis(
            func=body,
            xs=values,
            batch_size=width,
            bound_summary=True,
            active=active,
        )

    result = jax.jit(evaluate)(
        values=jnp.asarray([1.0, -2.0, jnp.nan]),
        active=jnp.asarray([True, True, active_last]),
    )
    np.testing.assert_array_equal(result.output[0], [1.0, 4.0, np.nan])
    np.testing.assert_array_equal(result.output[1], [True, False, False])
    record = result.bound_status[0]
    assert record.nonfinite_floor.shape == record.invalid_shifted_grid.shape == ()
    np.testing.assert_array_equal(
        jnp.stack((record.nonfinite_floor, record.invalid_shifted_grid)),
        [active_last, False],
    )


@pytest.mark.parametrize("width", [1, 4, 8])
@pytest.mark.parametrize("outer", ["none", "untiled", "broadcast"])
def test_cartesian_windows_keep_only_scalar_bound_summaries(
    *, width: int, outer: str
) -> None:
    """Cartesian and outer coordinate windows preserve ordinary axes and OR validity."""
    code = inspect.unwrap(tiled_productmap).__code__
    assert (
        "bound_summary" in code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    ), "Cartesian window owners need explicit Boolean-summary transport."
    mapped = cast(
        "Callable[..., feasibility.BoundedResult[tuple[FloatND, BoolND]]]",
        tiled_productmap(
            func=_conditional_cell,
            variables=("x", "y"),
            width_keyword="width",
            untiled_variables=("x",) if outer == "untiled" else (),
            broadcast_variables=("x",) if outer == "broadcast" else (),
            bound_summary=True,
        ),
    )
    result = jax.jit(
        lambda: mapped(
            x=jnp.asarray([0.0, 1.0]),
            y=jnp.asarray([1.0, 2.0, 3.0]),
            width=width,
        )
    )()
    _assert_conditional_cells(result=result)


@pytest.mark.parametrize("batched", [False, True])
def test_nested_product_maps_fold_validity_at_each_owning_axis(
    *, batched: bool
) -> None:
    """
    Ordinary product maps return one bounded integrity record across all
    coordinates.
    """
    code = inspect.unwrap(productmap).__code__
    assert (
        "bound_summary" in code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    ), "Ordinary coordinate maps need explicit Boolean-summary ownership."
    mapped = productmap(
        func=_conditional_cell,
        variables=("x", "y"),
        batch_sizes={"x": 1, "y": 2} if batched else {"x": 0, "y": 0},
        bound_summary=True,
    )
    result = jax.jit(mapped)(
        x=jnp.asarray([0.0, 1.0]),
        y=jnp.asarray([1.0, 2.0, 3.0]),
    )
    _assert_conditional_cells(result=result)


def _conditional_cell(
    *, x: FloatND, y: FloatND
) -> feasibility.BoundedResult[tuple[FloatND, BoolND]]:
    """Return ordinary cell values and an independent canonical floor flag."""
    return feasibility.BoundedResult(
        output=(x + 2.0 * y, x == 0.0),
        bound_status=(
            feasibility.summarize_bound_inputs(
                floor_values=jnp.where(x == 1.0, jnp.nan, 0.0),
                active=jnp.asarray(a=True),
            ),
        ),
    )


def _assert_conditional_cells(
    *, result: feasibility.BoundedResult[tuple[FloatND, BoolND]]
) -> None:
    """Check exact values, axis order and independently reduced invalid inputs."""
    np.testing.assert_array_equal(result.output[0], [[2.0, 4.0, 6.0], [3.0, 5.0, 7.0]])
    np.testing.assert_array_equal(result.output[1], [[True] * 3, [False] * 3])
    record = result.bound_status[0]
    assert record.nonfinite_floor.shape == record.invalid_shifted_grid.shape == ()
    np.testing.assert_array_equal(
        jnp.stack((record.nonfinite_floor, record.invalid_shifted_grid)),
        [True, False],
    )
