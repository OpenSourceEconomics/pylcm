"""Hoisted Case-U reads against the rebuilding reference on bounded queries.

``hoisted_reference`` reads the row's conditional bank once built; it must keep
the rebuilding reference's value, economic marginal and validity. The economic
marginal freezes the selected saving (``borrowing_limit`` at a corner, else the
bracket's saving) and differentiates ``utility(resources - saving)``; it is a
channel separate from the represented value's derivative.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.current_iid_node_adapter import NATIVE_CURRENT_IID_SWITCH
from _lcm.egm.current_iid_case_u_balanced import (
    BalancedCaseUCarry,
    SaturatedCaseUProgram,
)
from _lcm.egm.projected_continuation import ProjectedRowRead
from _lcm.solution.current_iid_case_u_native import acquire_installed_case_u_date
from lcm.typing import ScalarFloat
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_balanced import (
    _carry,
    _model,
    _solved,
)
from tests.egm.test_current_iid_projected_row_reference import (
    _case_u_program,
    _solved_child,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.fixture(scope="module", params=("routing", "genuine_u"))
def hoisted_case(
    *, request: pytest.FixtureRequest
) -> tuple[BalancedCaseUCarry, SaturatedCaseUProgram]:
    """Reuse the routing cache and solve the genuine-U fixture once per module.

    Both fixtures are built on the native current-IID route, whose rows this tests.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(NATIVE_CURRENT_IID_SWITCH, "1")
        if request.param == "routing":
            model, carry = _solved_child(utility_reads_eps=True)
            _, program = _case_u_program(model=model)
        else:
            model = _model()
            carry = _carry(solution=_solved(model=model))
            _, generation = model._prepare_call(params={"discount_factor": 1.0})
            program = acquire_installed_case_u_date(
                generation=generation, regime_name="child", period=0
            ).date.guard_program
    assert isinstance(carry, BalancedCaseUCarry)
    assert isinstance(program, SaturatedCaseUProgram)
    return carry, program


def _sample(*, values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values) - 1, maximum, dtype=int)]


def _bounded_queries(*, carry: BalancedCaseUCarry) -> np.ndarray:
    """At most 18 records, adjacent cut records and strict interior midpoints."""
    records = np.asarray(carry.grid)[: int(carry.valid_length)]
    linked = np.asarray(carry.link_valid)[: len(records) - 1]
    assert len(records) >= 2
    selected_records = _sample(values=records, maximum=6)
    # The publisher retains predecessor/successor records around pulled-back
    # cuts. Keep a few actual adjacent pairs, when the fixture contains them.
    adjacent = np.flatnonzero(np.nextafter(records[:-1], records[1:]) == records[1:])
    adjacent = _sample(values=adjacent, maximum=4)
    cuts = np.concatenate((records[adjacent], records[adjacent + 1]))
    lower, upper = records[:-1][linked], records[1:][linked]
    middle = lower + (upper - lower) / 2
    middle = middle[(middle > lower) & (middle < upper)]
    assert middle.size, "Fixture must exercise at least one linked cell interior."
    middle = _sample(values=middle, maximum=4)
    return np.unique(np.concatenate((selected_records, cuts, middle)))


def _assert_read_agreement(
    *, actual: ProjectedRowRead, expected: ProjectedRowRead, dtype: np.dtype
) -> None:
    np.testing.assert_array_equal(actual.valid, expected.valid)
    assert bool(np.all(np.asarray(expected.valid))), "Fixture queries must be valid."
    eps = float(np.finfo(dtype).eps)
    budgets = (
        ("value", max(1e-6, 64 * eps)),
        ("economic_marginal", max(1e-4, 8 * np.sqrt(eps))),
    )
    for channel, budget in budgets:
        wanted = np.asarray(getattr(expected, channel), dtype=np.float64)
        got = np.asarray(getattr(actual, channel), dtype=np.float64)
        np.testing.assert_array_equal(np.isneginf(got), np.isneginf(wanted))
        finite = np.isfinite(wanted)
        assert np.all(np.isfinite(got[finite]))
        assert np.all(np.isfinite(wanted) | np.isneginf(wanted))
        scaled = np.abs(got[finite] - wanted[finite]) / np.maximum(
            1.0, np.abs(wanted[finite])
        )
        assert np.all(scaled <= budget), (channel, scaled, budget)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_hoisted_reference_preserves_rebuilding_value_marginal_and_validity(
    *, hoisted_case: tuple[BalancedCaseUCarry, SaturatedCaseUProgram]
) -> None:
    carry, program = hoisted_case
    hoisted: Callable[..., ProjectedRowRead] | None = getattr(
        program, "hoisted_reference", None
    )
    assert callable(hoisted), "Implement SaturatedCaseUProgram.hoisted_reference first."
    conditionals = program.conditionals(guard=carry.guard)

    def old_read(query: ScalarFloat) -> ProjectedRowRead:
        return program.reference(guard=carry.guard, query=query)

    def new_read(query: ScalarFloat) -> ProjectedRowRead:
        return hoisted(guard=carry.guard, conditionals=conditionals, query=query)

    old_scalar = jax.jit(old_read)
    new_scalar = jax.jit(new_read)
    queries = _bounded_queries(carry=carry)
    for query in queries:
        point = jnp.asarray(query, dtype=carry.grid.dtype)
        _assert_read_agreement(
            actual=new_scalar(point), expected=old_scalar(point), dtype=queries.dtype
        )

    # A fixed, small batch exercises transformed execution without dense
    # repeated envelope construction. Padding is harmless repeated input.
    batch = queries[np.linspace(0, len(queries) - 1, 4, dtype=int)]
    points = jnp.asarray(batch, dtype=carry.grid.dtype)
    _assert_read_agreement(
        actual=jax.jit(jax.vmap(new_read))(points),
        expected=jax.jit(jax.vmap(old_read))(points),
        dtype=queries.dtype,
    )
