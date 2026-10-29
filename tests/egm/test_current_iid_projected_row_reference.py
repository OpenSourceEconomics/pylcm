"""Projected child rows against an independent scalar reference, off their records.

The routing fixture's child consumes from `R = w + eps` (eps in {0, 1}, weights
1/2) with log utility and a terminal `log(s)`. On saving node `s_j` the Euler
equation gives `c_j = s_j`, so the child's resources row has nodes `R_j = 2 s_j`
with value `2 log s_j` and economic marginal `1 / s_j`. The row reads its value by
cubic Hermite in `R` (slope `1 / c_j`) and its marginal linearly. The projected
row averages that read over the two shocks; it must match it everywhere on the
domain, not only at stored records, within the projection's own budgets.
"""

from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_projection import _read_projected
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from lcm import Model
from lcm.solver_api import ArtifactRef
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_call_owned_generation import (
    _SLOW_X64_U,
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
    return carry


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
@_SLOW_X64_U
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
