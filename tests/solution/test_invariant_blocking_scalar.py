"""A blocked core with scalar local state and action products still plans reads."""

from fractions import Fraction
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import CoreExecutionDisposition, core_program_graph
from lcm import (
    AgeGrid,
    Choose,
    DiscreteGrid,
    ExecutionConfig,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import DiscreteAction, DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class _Types:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _OneAction:
    only: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    terminal: ScalarInt


def _flow(
    *, pref_type: DiscreteState, choice: DiscreteAction, weights: FloatND
) -> FloatND:
    return weights[pref_type] + choice


def _terminal(*, pref_type: DiscreteState, weights: FloatND) -> FloatND:
    return weights[pref_type]


def _to_terminal() -> ScalarInt:
    return RegimeId.terminal


def _model(
    *, blocked: bool, enable_jit: bool = True, device_memory_bytes: int | None = 2**30
) -> Model:
    grid = DiscreteGrid(category_class=_Types)
    return Model(
        regimes={
            "working": Regime(
                states={"pref_type": grid},
                state_transitions={"pref_type": fixed_transition("pref_type")},
                actions={"choice": DiscreteGrid(category_class=_OneAction)},
                functions={"utility": _flow},
                regime_transitions=Choose(func=_to_terminal, targets=("terminal",)),
            ),
            "terminal": Regime(
                states={"pref_type": grid},
                functions={"utility": _terminal},
                regime_transitions=None,
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_regimes={0: "working"},
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(
            devices=(0,),
            device_memory_bytes=device_memory_bytes,
            invariant_block_widths={"pref_type": 1} if blocked else {},
        ),
    )


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_complete_type_values_follow_the_literal_oracle_in_each_execution_mode(
    *,
    blocked: bool,
    enable_jit: bool,
    log_level: Literal["off", "warning", "progress", "debug"],
) -> None:
    """All original type codes retain their Bellman and terminal values."""
    dtype = np.dtype("float64" if jax.config.jax_enable_x64 else "float32")
    params = {
        "discount_factor": 0.5,
        "working": {"utility": {"weights": jnp.asarray((3, 1, -2), dtype=dtype)}},
        "terminal": {"utility": {"weights": jnp.asarray((8, 2, 6), dtype=dtype)}},
    }
    result = _model(
        blocked=blocked, enable_jit=enable_jit, device_memory_bytes=None
    ).solve(params=params, log_level=log_level)
    actual = tuple(
        (period, regime, value.shape, value.dtype, np.asarray(value).tobytes())
        for period, regimes in sorted(result.values.items())
        for regime, value in sorted(regimes.items())
    )
    # Code order is low/middle/high. Working values are the independently worked
    # literal 3 + 8/2, 1 + 2/2, -2 + 6/2; terminal values are 8, 2, 6.
    expected = (
        (0, "working", (3,), dtype, np.asarray((7, 2, 1), dtype=dtype).tobytes()),
        (1, "terminal", (3,), dtype, np.asarray((8, 2, 6), dtype=dtype).tobytes()),
    )
    assert actual == expected


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_eager_solve_refuses_a_compiler_memory_budget_in_every_logging_mode(
    *, blocked: bool, log_level: Literal["off", "warning", "progress", "debug"]
) -> None:
    """A budget requiring compiler workspace reports is refused without JIT."""
    dtype = np.dtype("float64" if jax.config.jax_enable_x64 else "float32")
    params = {
        "discount_factor": 0.5,
        "working": {"utility": {"weights": jnp.asarray((3, 1, -2), dtype=dtype)}},
        "terminal": {"utility": {"weights": jnp.asarray((8, 2, 6), dtype=dtype)}},
    }
    model = _model(blocked=blocked, enable_jit=False)
    with pytest.raises(
        ExecutionPlanningError,
        match=r"ExecutionConfig\.device_memory_bytes requires JIT compilation",
    ):
        model.solve(params=params, log_level=log_level)


def test_no_axis_bound_program_still_declares_planned_value_reads() -> None:
    model = _model(blocked=True)
    graph = core_program_graph(
        kernel=model._regimes["working"].solution.period_kernels[0]
    )
    assert len(graph) == 3
    for core in graph.values():
        assert core.invariant_binding is not None
        assert not core.requirements.axes
        assert core.requirements.value_reads
        assert core.disposition is CoreExecutionDisposition.PLANNED


@pytest.mark.parametrize(
    ("flow", "terminal", "expected"),
    [
        ((1, 2, 3), (4, 8, 12), (3, 6, 9)),
        ((3, 1, -2), (8, 2, 6), (7, 2, 1)),
    ],
)
def test_scalar_block_equals_literal_typewise_bellman_equation(
    *,
    flow: tuple,
    terminal: tuple,
    expected: tuple,
) -> None:
    dtype = np.dtype("float64" if jax.config.jax_enable_x64 else "float32")
    params = {
        "discount_factor": 0.5,
        "working": {"utility": {"weights": jnp.asarray(flow, dtype=dtype)}},
        "terminal": {"utility": {"weights": jnp.asarray(terminal, dtype=dtype)}},
    }
    # The reference is the literal equation u[k] + V_terminal[k] / 2,
    # not a second invocation of the blocked implementation.
    oracle = np.asarray(expected, dtype=dtype)
    for blocked in (False, True):
        result = _model(blocked=blocked).solve(params=params, log_level="off")
        got = np.asarray(result.values[0]["working"])
        assert got.shape == (3,)
        assert got.dtype == oracle.dtype
        assert got.tobytes() == oracle.tobytes()


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
@pytest.mark.parametrize("scale", [0.5, 1.0, 8.0])
def test_scalar_bound_reads_preserve_type_permutation_and_binary_scaling(
    *,
    order: tuple[int, ...],
    scale: float,
) -> None:
    dtype = np.dtype("float64" if jax.config.jax_enable_x64 else "float32")
    flows = (3, 1, -2)
    terminals = (8, 2, 6)
    exact_scale = Fraction(scale)
    expected = np.asarray(
        [
            float(exact_scale * (Fraction(flows[i]) + Fraction(terminals[i], 2)))
            for i in order
        ],
        dtype=dtype,
    )
    params = {
        "discount_factor": 0.5,
        "working": {
            "utility": {
                "weights": jnp.asarray([scale * flows[i] for i in order], dtype=dtype)
            }
        },
        "terminal": {
            "utility": {
                "weights": jnp.asarray(
                    [scale * terminals[i] for i in order], dtype=dtype
                )
            }
        },
    }
    result = _model(blocked=True).solve(params=params, log_level="off")
    got = np.asarray(result.values[0]["working"])
    assert (got.shape, got.dtype) == (expected.shape, expected.dtype)
    assert got.tobytes() == expected.tobytes()
