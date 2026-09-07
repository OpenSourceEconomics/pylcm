"""DC-EGM streams the child stochastic-node expectation under a planner axis.

The continuation expectation runs over the product of the child regime's
stochastic-process nodes. That product is the `stochastic_node` axis the DC-EGM
value and replay programs declare, so the width it is folded at is an execution
fact the plan owns rather than a field on the solver. Every width reads the same
nodes with the same joint weights and reorders only the floating-point adds, so
the solved value function agrees across widths to the working format's rounding.
"""

import functools
from collections.abc import Mapping

import numpy as np
import pytest

from _lcm.execution.core_program import core_program_graph
from lcm import AgeGrid, ExecutionConfig, Model
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.solvers import DCEGM, STOCHASTIC_NODE_AXIS
from lcm.typing import FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON, assert_agrees_to_ulp
from tests.solution.test_egm_process_states import (
    CONSUMPTION_GRID,
    N_INCOME_NODES,
    N_PERIODS,
    SAVINGS_GRID,
    WEALTH_GRID,
    ProcessRegimeId,
    _get_params,
    _income_process,
    dead,
    inverse_marginal_utility,
    next_regime,
    next_wealth_from_savings_iid,
    savings,
    utility_consumption_only,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@functools.cache
def _model() -> Model:
    """DC-EGM model whose child carries an IID income process."""
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = float(ages.exact_values[-1])
    working = ConsumptionSavingsRegime(
        transition=next_regime,
        active=lambda age, la=last_age: age < la,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID, "income": _income_process("iid")},
        state_transitions={"wealth": next_wealth_from_savings_iid},
        functions={
            "utility": utility_consumption_only,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(savings_grid=SAVINGS_GRID, n_constrained_points=64),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="wealth",
            post_decision_state="savings",
        ),
    )
    return Model(
        regimes={"alive": working, "dead": dead},
        ages=ages,
        regime_id_class=ProcessRegimeId,
    )


@functools.cache
def _solve(width: int | None) -> Mapping[int, Mapping[str, FloatND]]:
    """Solve with the node axis fixed at `width`, or at the plan's own choice."""
    config = (
        ExecutionConfig()
        if width is None
        else ExecutionConfig(axis_widths={STOCHASTIC_NODE_AXIS: width})
    )
    return (
        _model()
        .solve(params=_get_params("iid"), log_level="off", execution_config=config)
        .values
    )


def test_dcegm_has_no_stochastic_node_batch_size_field() -> None:
    """The stochastic-node width is an execution fact, not a solver field."""
    with pytest.raises(TypeError, match="stochastic_node_batch_size"):
        DCEGM(
            savings_grid=SAVINGS_GRID,
            stochastic_node_batch_size=4,  # ty: ignore[unknown-argument]
        )


def test_value_program_declares_the_stochastic_node_axis() -> None:
    """The DC-EGM value program declares `stochastic_node` as its reduced axis."""
    kernels = _model()._regimes["alive"].solution.period_kernels
    program = core_program_graph(kernel=next(iter(kernels.values())))["main"]

    assert program.requirements.axis_names == (STOCHASTIC_NODE_AXIS,)


def test_stochastic_node_axis_spans_the_child_process_nodes() -> None:
    """The axis runs over the child's income nodes, one coordinate per process."""
    kernels = _model()._regimes["alive"].solution.period_kernels
    program = core_program_graph(kernel=next(iter(kernels.values())))["main"]
    (axis,) = program.requirements.reduced_axes

    assert (axis.coordinate_names, axis.coordinate_extents) == (
        ("income",),
        (N_INCOME_NODES,),
    )


def test_stochastic_node_axis_folds_a_weighted_expectation() -> None:
    """The axis names the weighted-expectation contract as its reduction."""
    kernels = _model()._regimes["alive"].solution.period_kernels
    program = core_program_graph(kernel=next(iter(kernels.values())))["main"]
    (axis,) = program.requirements.reduced_axes

    assert axis.reduction.semantic_key == ("weighted-expectation", 1)


@pytest.mark.parametrize("width", [1, 2, 3, N_INCOME_NODES])
def test_value_agrees_across_stochastic_node_widths(*, width: int) -> None:
    """Folding the expectation at any width reproduces the widest fold's value.

    Includes a width (3) that does not divide the five-node income mesh, so the
    last block is short and padded with exactly-zero weights.
    """
    reference = _solve(N_INCOME_NODES)
    streamed = _solve(width)

    for period in sorted(reference):
        for regime_name in reference[period]:
            assert_agrees_to_ulp(
                got=np.asarray(streamed[period][regime_name]),
                expected=np.asarray(reference[period][regime_name]),
                n_ulp=64,
                err_msg=f"period={period}, regime={regime_name}",
            )


def test_a_width_above_the_mesh_clamps_to_the_whole_mesh() -> None:
    """A width the mesh cannot fill folds the mesh in one block instead."""
    reference = _solve(N_INCOME_NODES)
    clamped = _solve(N_INCOME_NODES + 1)

    for period in sorted(reference):
        for regime_name in reference[period]:
            np.testing.assert_array_equal(
                np.asarray(clamped[period][regime_name]),
                np.asarray(reference[period][regime_name]),
                err_msg=f"period={period}, regime={regime_name}",
            )
