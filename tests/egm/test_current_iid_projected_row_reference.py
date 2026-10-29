"""Projected child rows against an independent scalar reference, off their records.

The routing fixture's child consumes from `R = w + eps` (eps in {0, 1}, weights
1/2) with log utility and a terminal `log(s)`. On saving node `s_j` the Euler
equation gives `c_j = s_j`, so the child's resources row has nodes `R_j = 2 s_j`
with value `2 log s_j` and economic marginal `1 / s_j`. The row reads its value by
cubic Hermite in `R` (slope `1 / c_j`) and its marginal linearly. The projected
row averages that read over the two shocks; it must match it everywhere on the
domain, not only at stored records, within the projection's own budgets.
"""

import functools
from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_balanced import (
    BalancedProjectedEGMCarry,
    BalancedResourcesProgram,
)
from _lcm.egm.current_iid_case_u_balanced import (
    BalancedCaseUCarry,
    SaturatedCaseUProgram,
)
from _lcm.egm.current_iid_projection import _read_projected
from _lcm.egm.projected_child import read_projected_coordinate
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedChildAuthority,
    ProjectedEGMCarry,
)
from _lcm.solution.current_iid_case_u_native import acquire_installed_case_u_date
from lcm import Model
from lcm.solver_api import ArtifactRef
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON, X64_ENABLED
from tests.egm.test_current_iid_call_owned_generation import (
    _declarations,
    _params,
    _utility_reading_eps,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_SAVINGS = np.array(
    [
        0.25, 0.5, 0.625, 0.75, 1.0, 1.125, 1.25, 1.375, 1.5, 1.75, 1.875,
        2.0, 2.25, 2.5, 4.25, 4.75, 6.25, 6.375, 6.75, 6.875, 12.5, 12.75,
    ]
)  # fmt: skip
_QUERIES = np.linspace(1.0, 12.5, 4001)[1:-1]


def _row_read(resources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The child resources row's Hermite value and linear marginal at `resources`."""
    nodes, values, slopes = 2.0 * _SAVINGS, 2.0 * np.log(_SAVINGS), 1.0 / _SAVINGS
    j = np.clip(np.searchsorted(nodes, resources, side="right") - 1, 0, nodes.size - 2)
    h = nodes[j + 1] - nodes[j]
    t = (resources - nodes[j]) / h
    value = (
        (2 * t**3 - 3 * t**2 + 1) * values[j]
        + (t**3 - 2 * t**2 + t) * h * slopes[j]
        + (-2 * t**3 + 3 * t**2) * values[j + 1]
        + (t**3 - t**2) * h * slopes[j + 1]
    )
    return value, slopes[j] + t * (slopes[j + 1] - slopes[j])


def _solved_child_carry(*, utility_reads_eps: bool) -> ProjectedEGMCarry:
    return _solved_child(utility_reads_eps=utility_reads_eps)[1]


@functools.cache
def _solved_child(*, utility_reads_eps: bool) -> tuple[Model, ProjectedEGMCarry]:
    """Solve once per precision; the Case-U solve dominates this file's runtime."""
    declarations = _declarations()
    if utility_reads_eps:
        child = declarations.regimes["child"]
        child = replace(
            child, functions={**child.functions, "utility": _utility_reading_eps}
        )
        declarations = replace(
            declarations, regimes={**declarations.regimes, "child": child}
        )
    model = Model(
        **{
            field.name: getattr(declarations, field.name)
            for field in fields(declarations)
            if field.init
        }
    )
    result = model.solve(
        params=_params(), log_level="warning", max_compilation_workers=1
    )
    carry = result.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(carry, ProjectedEGMCarry)
    return model, carry


def _assert_value_within_budget(
    *, carry: ProjectedEGMCarry, expected_value: np.ndarray
) -> None:
    """The value budget plus the rounding of the stored precision's own reference."""
    dtype = carry.grid.dtype
    eps = float(jnp.finfo(dtype).eps)
    read = jax.vmap(lambda query: _read_projected(carry=carry, query=query))(
        jnp.asarray(_QUERIES, dtype=dtype)
    )
    assert bool(jnp.all(read.valid))
    scale = np.maximum(1.0, np.abs(expected_value))
    error = np.abs(np.asarray(read.value, dtype=np.float64) - expected_value) / scale
    assert error.max() <= max(1e-6, 64 * eps) + 4 * eps


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_r_projected_row_matches_the_row_read_off_its_records() -> None:
    """Case R's W and M equal the shock average of the child row read."""
    carry = _solved_child_carry(utility_reads_eps=False)
    low, high = _row_read(_QUERIES), _row_read(_QUERIES + 1.0)
    _assert_value_within_budget(carry=carry, expected_value=0.5 * (low[0] + high[0]))
    dtype = carry.grid.dtype
    read = jax.vmap(lambda query: _read_projected(carry=carry, query=query))(
        jnp.asarray(_QUERIES, dtype=dtype)
    )
    marginal = 0.5 * (low[1] + high[1])
    error = np.abs(np.asarray(read.economic_marginal, dtype=np.float64) - marginal)
    assert (error / np.maximum(1.0, marginal)).max() <= max(
        1e-4, 8 * np.sqrt(float(jnp.finfo(dtype).eps))
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_u_projected_row_matches_the_row_read_off_its_records() -> None:
    """Case U's W is the same average plus `0.125 * E[eps]`; its M is `u'(c)`.

    Case U reports the economic marginal at the exact interior optimum `c = R/2`,
    so `M = 1/w + 1/(w + 1)` on this domain.
    """
    carry = _solved_child_carry(utility_reads_eps=True)
    low, high = _row_read(_QUERIES), _row_read(_QUERIES + 1.0)
    _assert_value_within_budget(
        carry=carry, expected_value=0.5 * (low[0] + high[0]) + 0.0625
    )
    dtype = carry.grid.dtype
    read = jax.vmap(lambda query: _read_projected(carry=carry, query=query))(
        jnp.asarray(_QUERIES, dtype=dtype)
    )
    marginal = 1.0 / _QUERIES + 1.0 / (_QUERIES + 1.0)
    error = np.abs(np.asarray(read.economic_marginal, dtype=np.float64) - marginal)
    assert (error / np.maximum(1.0, marginal)).max() <= max(
        1e-4, 8 * np.sqrt(float(jnp.finfo(dtype).eps))
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_u_projected_row_meets_its_budget_with_few_records() -> None:
    """Case U reads W by cubic Hermite in its stored slope, so few records suffice.

    A linear read of the same smooth W needs several hundred records to meet the
    value budget; a slope-carrying read meets it with about a hundred.
    """
    carry = _solved_child_carry(utility_reads_eps=True)
    assert bool(carry.valid)
    # The value budget is 64 eps at float32 and 1e-6 at float64.
    assert int(carry.valid_length) <= (200 if X64_ENABLED else 130)


def _case_u_program(
    *, model: Model
) -> tuple[ProjectedChildAuthority, SaturatedCaseUProgram]:
    _, generation = model._prepare_call(params=_params())
    date = acquire_installed_case_u_date(
        generation=generation, regime_name="child", period=1
    ).date
    program = date.guard_program
    assert isinstance(program, SaturatedCaseUProgram)
    return date.authority, program


def _child_row_derivative(resources: np.ndarray) -> np.ndarray:
    """Derivative of `_row_read`'s Hermite value in the child's resources."""
    nodes, values, slopes = 2.0 * _SAVINGS, 2.0 * np.log(_SAVINGS), 1.0 / _SAVINGS
    j = np.clip(np.searchsorted(nodes, resources, side="right") - 1, 0, nodes.size - 2)
    h = nodes[j + 1] - nodes[j]
    t = (resources - nodes[j]) / h
    return (
        (6 * t**2 - 6 * t) * values[j] / h
        + (3 * t**2 - 4 * t + 1) * slopes[j]
        + (-6 * t**2 + 6 * t) * values[j + 1] / h
        + (3 * t**2 - 2 * t) * slopes[j + 1]
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_u_value_slope_is_the_derivative_of_the_reference_value() -> None:
    """The stored slope is dW/dw, at records (cuts included) and between them.

    W is the shock average of the child's Hermite row read, so its derivative is
    the average of that read's derivative at `w` and `w + 1`. Cut records put a
    shock's resources exactly on a child node, where the slope must still be the
    derivative of the read rather than of its bracket search. Between records it
    also agrees with a central difference of the reference value itself.
    """
    model, carry = _solved_child(utility_reads_eps=True)
    assert isinstance(carry, BalancedCaseUCarry)
    _, program = _case_u_program(model=model)
    guard = carry.guard
    conditionals = program.conditionals(guard=guard)
    slope = jax.jit(
        jax.vmap(
            lambda query: program.value_slope(
                guard=guard, conditionals=conditionals, query=query
            )
        )
    )
    records = carry.grid[: int(carry.valid_length)]
    np.testing.assert_array_equal(
        slope(records), carry.value_slope[: int(carry.valid_length)]
    )
    middles = records[:-1] + (records[1:] - records[:-1]) / 2.0
    for queries in (records, middles, jnp.asarray(_QUERIES, dtype=records.dtype)):
        at = np.asarray(queries, dtype=np.float64)
        expected = 0.5 * (_child_row_derivative(at) + _child_row_derivative(at + 1.0))
        np.testing.assert_array_almost_equal(
            slope(queries), expected, decimal=DECIMAL_PRECISION
        )
    reference = jax.jit(
        jax.vmap(lambda query: program.reference(guard=guard, query=query).value)
    )
    step = float(jnp.finfo(records.dtype).eps) ** (1.0 / 3.0)
    above, below = middles + step, middles - step
    difference = (
        np.asarray(reference(above), dtype=np.float64)
        - np.asarray(reference(below), dtype=np.float64)
    ) / (np.asarray(above, dtype=np.float64) - np.asarray(below, dtype=np.float64))
    # A central difference is accurate to the square of its step.
    error = np.abs(np.asarray(slope(middles), dtype=np.float64) - difference)
    assert (error / np.maximum(1.0, np.abs(difference))).max() <= step


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_u_refinement_reads_its_row_as_consumers_do() -> None:
    """Refinement's predicted read is bit-for-bit the consumers' read of the row."""
    model, carry = _solved_child(utility_reads_eps=True)
    assert isinstance(carry, BalancedCaseUCarry)
    authority, program = _case_u_program(model=model)
    _assert_refinement_reads_as_consumers_do(
        authority=authority, program=program, carry=carry
    )


def _assert_refinement_reads_as_consumers_do(
    *,
    authority: ProjectedChildAuthority,
    program: SaturatedCaseUProgram | BalancedResourcesProgram,
    carry: BalancedCaseUCarry | BalancedProjectedEGMCarry,
) -> None:
    unpublished = ProjectedEGMCarry(
        **{
            field.name: getattr(carry, field.name)
            for field in fields(ProjectedEGMCarry)
        }
    )
    records = carry.grid[: int(carry.valid_length)]
    smooth = carry.link_valid[: records.size - 1]
    lower, upper = records[:-1][smooth], records[1:][smooth]
    width = upper - lower
    probes = jnp.concatenate(
        (
            records,
            lower + width / 4.0,
            lower + width / 2.0,
            lower + 3.0 * width / 4.0,
            jnp.nextafter(lower, upper),
            jnp.nextafter(upper, lower),
        )
    )
    predicted = jax.jit(
        jax.vmap(
            lambda query: _read_projected(
                carry=unpublished, query=query, slopes=carry.value_slope
            )
        )
    )(probes)
    consumed = jax.jit(
        jax.vmap(
            lambda query: read_projected_coordinate(
                authority=authority,
                carry=carry,
                guard_program=program,
                query=query,
                jacobian=jnp.ones_like(query),
            )
        )
    )(probes)
    assert bool(jnp.all(predicted.valid))
    assert not bool(jnp.any(consumed.query_status[0].coverage_missing))
    np.testing.assert_array_equal(consumed.output[0], predicted.value)
    np.testing.assert_array_equal(consumed.output[1], predicted.economic_marginal)
    linear = jax.vmap(lambda query: _read_projected(carry=unpublished, query=query))(
        probes
    )
    assert not bool(jnp.all(linear.value == predicted.value))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_r_projected_row_meets_its_budget_with_few_records() -> None:
    """Case R reads W by cubic Hermite in its stored slope, so few records suffice.

    A linear read of the same smooth W needs 1393 records at float64 and 568 at
    float32 to meet the value budget; the Hermite read needs 121 at both.
    """
    carry = _solved_child_carry(utility_reads_eps=False)
    assert bool(carry.valid)
    assert int(carry.valid_length) <= 150


def _case_r_program(
    *, model: Model
) -> tuple[ProjectedChildAuthority, BalancedResourcesProgram]:
    _, generation = model._prepare_call(params=_params())
    owner = generation.solver_preparation(regime_name="child", period=1)
    date = owner.declarations["child"].date_at(period=1)
    program = date.guard_program
    assert isinstance(program, BalancedResourcesProgram)
    return date.authority, program


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_r_value_slope_is_the_derivative_of_the_reference_value() -> None:
    """The stored slope is dW/dw, at records (cuts included) and between them.

    W is the shock average of the child's Hermite row read at `w` and `w + 1`, so
    its derivative is the average of that read's derivative there. Cut records
    put a shock's resources exactly on a child node, where the slope must still
    be the derivative of the read rather than of its bracket search.
    """
    model, carry = _solved_child(utility_reads_eps=False)
    assert isinstance(carry, BalancedProjectedEGMCarry)
    _, program = _case_r_program(model=model)
    guard = carry.guard
    slope = jax.jit(
        jax.vmap(lambda query: program.value_slope(guard=guard, query=query))
    )
    records = carry.grid[: int(carry.valid_length)]
    middles = records[:-1] + (records[1:] - records[:-1]) / 2.0
    dense = jnp.asarray(_QUERIES, dtype=records.dtype)
    # Stored at the records inside the refinement loop, recomputed elsewhere.
    for queries, slopes in (
        (records, carry.value_slope[: int(carry.valid_length)]),
        (records, slope(records)),
        (middles, slope(middles)),
        (dense, slope(dense)),
    ):
        at = np.asarray(queries, dtype=np.float64)
        expected = 0.5 * (_child_row_derivative(at) + _child_row_derivative(at + 1.0))
        np.testing.assert_array_almost_equal(
            slopes, expected, decimal=DECIMAL_PRECISION
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_r_refinement_reads_its_row_as_consumers_do() -> None:
    """Refinement's predicted read is bit-for-bit the consumers' read of the row."""
    model, carry = _solved_child(utility_reads_eps=False)
    assert isinstance(carry, BalancedProjectedEGMCarry)
    authority, program = _case_r_program(model=model)
    _assert_refinement_reads_as_consumers_do(
        authority=authority, program=program, carry=carry
    )
