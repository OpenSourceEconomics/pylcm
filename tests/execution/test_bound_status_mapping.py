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


@pytest.mark.parametrize(("n_rows", "batch_size"), [(8, 0), (8, 1)])
def test_bounded_map_lowers_its_row_body_once(*, n_rows: int, batch_size: int) -> None:
    """One window and eight windows each place one copy of the row body."""
    values = jnp.linspace(0.5, 2.0, n_rows)
    lowered = (
        jax.jit(
            lambda xs: map_over_leading_axis(
                func=_marked_row, xs=xs, batch_size=batch_size, bound_summary=True
            )
        )
        .lower(values)
        .as_text()
    )
    assert lowered.count("stablehlo.round_nearest_even") == 1


@pytest.mark.parametrize(
    ("n_rows", "batch_size"),
    [(7, 0), (7, 7), (8, 2), (8, 4), (7, 3), (7, 1)],
    ids=["whole", "covering", "four-windows", "two-windows", "ragged", "unit"],
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_bounded_map_matches_the_plain_map_and_direct_reductions(
    *, n_rows: int, batch_size: int, compiled: bool
) -> None:
    """Windowed outputs equal the vectorized row body; summaries equal direct folds."""
    values = jnp.asarray([0.5, -1.0, jnp.nan, 2.0, -3.0, jnp.inf, 4.0, 1.5])[:n_rows]
    active = jnp.asarray([True, True, False, True, True, True, False, True])[:n_rows]

    def evaluate(
        *, xs: FloatND, mask: BoolND
    ) -> feasibility.BoundedResult[tuple[FloatND, BoolND]]:
        return map_over_leading_axis(
            func=_marked_row,
            xs=xs,
            batch_size=batch_size,
            bound_summary=True,
            active=mask,
        )

    result = (jax.jit(evaluate) if compiled else evaluate)(xs=values, mask=active)
    plain = jax.vmap(_marked_row)(values)
    assert result.query_status is not None
    assert result.continuation_read_counts is not None
    actual = [
        *result.output,
        *result.bound_status[0],
        *result.query_status[0],
        *result.continuation_read_counts,
    ]
    expected = [
        *plain.output,
        *_direct_summaries(values=np.asarray(values), active=np.asarray(active)),
    ]
    assert len(jax.tree.leaves(result)) == len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got.dtype == np.asarray(want).dtype
        np.testing.assert_array_equal(got, want)


def test_bounded_map_empty_prefix_is_the_merge_identity() -> None:
    """Reducing no active lane gives all-false and zero, which every merge keeps."""
    lanes = jax.vmap(_marked_row)(jnp.asarray([jnp.nan, -1.0, jnp.inf, 2.0]))
    nobody = jnp.zeros(4, dtype=bool)
    empty = (
        feasibility.reduce_bound_input_status(
            statuses=lanes.bound_status, active=nobody
        ),
        feasibility.reduce_query_input_status(
            statuses=lanes.query_status, active=nobody
        ),
        feasibility.reduce_continuation_read_counts(
            counts=lanes.continuation_read_counts, active=nobody
        ),
    )
    everyone = jnp.ones(4, dtype=bool)
    full = (
        feasibility.reduce_bound_input_status(
            statuses=lanes.bound_status, active=everyone
        ),
        feasibility.reduce_query_input_status(
            statuses=lanes.query_status, active=everyone
        ),
        feasibility.reduce_continuation_read_counts(
            counts=lanes.continuation_read_counts, active=everyone
        ),
    )
    merged = (
        tuple(
            feasibility.merge_bound_input_status(left=left, right=right)
            for left, right in zip(empty[0], full[0], strict=True)
        ),
        feasibility.merge_query_input_status(left=empty[1], right=full[1]),
        feasibility.merge_continuation_read_counts(left=empty[2], right=full[2]),
    )
    assert [leaf.tolist() for leaf in jax.tree.leaves(empty)] == [
        False,
        False,
        False,
        False,
        False,
        0,
        0,
    ]
    assert [leaf.tolist() for leaf in jax.tree.leaves(merged)] == [
        leaf.tolist() for leaf in jax.tree.leaves(full)
    ]


def _marked_row(
    value: FloatND,
) -> feasibility.BoundedResult[tuple[FloatND, BoolND]]:
    """Return one rounding, an exact marker op, and every bounded record for one row."""
    return feasibility.BoundedResult(
        output=(
            jax.lax.round(4.0 * value, jax.lax.RoundingMethod.TO_NEAREST_EVEN),
            value >= 0.0,
        ),
        bound_status=(
            feasibility.summarize_bound_inputs(
                floor_values=value,
                active=jnp.asarray(a=True),
                invalid_shifted_grid=value < -2.0,
            ),
        ),
        query_status=(
            feasibility.QueryInputStatus(
                nonfinite_floor=jnp.isinf(value),
                invalid_shifted_grid=value < 0.0,
                coverage_missing=value > 3.0,
            ),
        ),
        continuation_read_counts=feasibility.ContinuationReadCounts(
            below_first_live=jnp.where(value < 1.0, 2, 0).astype(jnp.int32),
            eligible=jnp.asarray(3, dtype=jnp.int32),
        ),
    )


def _direct_summaries(
    *, values: np.ndarray, active: np.ndarray
) -> list[np.ndarray | np.generic]:
    """Fold the row records of `_marked_row` over active rows with plain NumPy.

    The order is bound status, query status, then continuation read counts.
    """
    with np.errstate(invalid="ignore"):
        return [
            np.any(active & ~np.isfinite(values)),
            np.any(active & (values < -2.0)),
            np.any(active & np.isinf(values)),
            np.any(active & (values < 0.0)),
            np.any(active & (values > 3.0)),
            np.int32(np.sum(active & (values < 1.0)) * 2),
            np.int32(np.sum(active) * 3),
        ]
