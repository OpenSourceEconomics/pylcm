"""A blocked core with scalar local state and action products still plans reads."""

import json
import os
import subprocess
import textwrap
from fractions import Fraction
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import CoreExecutionDisposition, core_program_graph
from lcm import (
    AgeGrid,
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


def _model(
    *, blocked: bool, enable_jit: bool = True, device_memory_bytes: int | None = 2**30
) -> Model:
    grid = DiscreteGrid(category_class=_Types)
    return Model(
        edges={"working": {"terminal": 0}},
        regimes={
            "working": Regime(
                states={"pref_type": grid},
                state_transitions={"pref_type": fixed_transition("pref_type")},
                actions={"choice": DiscreteGrid(category_class=_OneAction)},
                functions={"utility": _flow},
            ),
            "terminal": Regime(
                states={"pref_type": grid},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "working"},
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


@pytest.mark.parametrize(
    ("schedule", "partitions"),
    [("block_major", 1), ("period_major", 2), ("block_major", 2)],
)
def test_eager_schedules_and_action_partitions_preserve_all_literal_type_values(
    *, tmp_path: Path, schedule: str, partitions: int
) -> None:
    """The unique maximum at action code one retains every type's value."""
    root = Path(__file__).resolve().parents[2]
    x64 = jax.config.jax_enable_x64
    script = textwrap.dedent(
        """
        import dataclasses
        import json
        import sys

        import jax
        import jax.numpy as jnp
        import numpy as np

        jax.config.update("jax_enable_x64", bool(int(sys.argv[1])))

        from lcm import (
            DiscreteGrid, ExecutionConfig, InvariantBlockSchedule, Model, categorical,
        )
        from lcm.typing import ScalarInt
        from tests.solution.test_invariant_blocking_scalar import RegimeId, _model

        @categorical(ordered=False)
        class Actions:
            zero: ScalarInt
            one: ScalarInt

        base = _model(blocked=True, enable_jit=False, device_memory_bytes=None)
        working = dataclasses.replace(
            base.user_regimes["working"],
            actions={"choice": DiscreteGrid(category_class=Actions)}
        )
        model = Model(
            edges=base.edges,
            regimes={**base.user_regimes, "working": working},
            ages=base.ages,
            regime_id_class=RegimeId,
            initial_nodes={0: "working"},
            enable_jit=False,
            execution_config=ExecutionConfig(
                devices=(0, 1), device_memory_bytes=None,
                invariant_block_widths={"pref_type": 1},
                invariant_block_schedule=InvariantBlockSchedule(sys.argv[2]),
                action_partitions={"working": int(sys.argv[3])},
                axis_widths={"action_product": 1},
            ),
        )
        dtype = np.dtype("float64" if jax.config.jax_enable_x64 else "float32")
        params = {
            "discount_factor": 0.5,
            "working": {"utility": {"weights": jnp.asarray((3, 1, -2), dtype=dtype)}},
            "terminal": {"utility": {"weights": jnp.asarray((8, 2, 6), dtype=dtype)}},
        }
        result = model.solve(params=params, log_level="off")
        values = [
            [period, regime, list(value.shape), value.dtype.str,
             np.asarray(value).tobytes().hex()]
            for period, regimes in sorted(result.values.items())
            for regime, value in sorted(regimes.items())
        ]
        print(json.dumps({
            "backend": jax.default_backend(), "devices": jax.device_count(),
            "x64": jax.config.jax_enable_x64, "values": values,
        }))
        """
    )
    command = [
        "pixi",
        "run",
        "--frozen",
        "-e",
        "tests-cpu",
        "python",
        "-c",
        script,
        str(int(x64)),
        schedule,
        str(partitions),
    ]
    process = subprocess.run(  # noqa: S603
        command,
        cwd=root,
        env={
            **os.environ,
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    (tmp_path / "child-stdout.json").write_text(process.stdout)
    (tmp_path / "child-stderr.log").write_text(process.stderr)
    (tmp_path / "child-command.json").write_text(json.dumps(command))
    actual = json.loads(process.stdout) if process.returncode == 0 else None
    dtype = np.dtype("float64" if x64 else "float32")
    # Action one adds one to each literal Bellman value: 7, 2, 1 become 8, 3, 2.
    expected = {
        "backend": "cpu",
        "devices": 2,
        "x64": x64,
        "values": [
            [
                0,
                "working",
                [3],
                dtype.str,
                np.asarray((8, 3, 2), dtype=dtype).tobytes().hex(),
            ],
            [
                1,
                "terminal",
                [3],
                dtype.str,
                np.asarray((8, 2, 6), dtype=dtype).tobytes().hex(),
            ],
        ],
    }
    assert (process.returncode, actual) == (0, expected), process.stderr


def test_eager_state_sharded_blocks_preserve_all_literal_type_values(
    *, tmp_path: Path
) -> None:
    """A sharded discrete state retains every type and both state codes."""
    root = Path(__file__).resolve().parents[2]
    x64 = jax.config.jax_enable_x64
    script = textwrap.dedent(
        """
        import dataclasses
        import json
        import sys

        import jax
        import jax.numpy as jnp
        import numpy as np

        jax.config.update("jax_enable_x64", bool(int(sys.argv[1])))

        from lcm import (
            Model, ExecutionConfig, DiscreteGrid, categorical, fixed_transition,
        )
        from lcm.typing import DiscreteState, DiscreteAction, FloatND, ScalarInt
        from tests.solution.test_invariant_blocking_scalar import RegimeId, _model

        @categorical(ordered=False)
        class Wealth:
            zero: ScalarInt
            one: ScalarInt

        def flow(*, wealth: DiscreteState, pref_type: DiscreteState,
                 choice: DiscreteAction, weights: FloatND) -> FloatND:
            return weights[pref_type] + choice + wealth

        def terminal(*, wealth: DiscreteState, pref_type: DiscreteState,
                     weights: FloatND) -> FloatND:
            return weights[pref_type] + wealth

        base = _model(blocked=True, enable_jit=False, device_memory_bytes=None)
        model = Model(
            edges=base.edges,
            regimes={
                "working": dataclasses.replace(base.user_regimes["working"],
                                                functions={"utility": flow}),
                "terminal": dataclasses.replace(base.user_regimes["terminal"],
                                                 functions={"utility": terminal}),
            },
            states={"wealth": DiscreteGrid(category_class=Wealth)},
            state_transitions={"wealth": fixed_transition("wealth")},
            ages=base.ages, regime_id_class=RegimeId,
            initial_nodes={0: "working"}, enable_jit=False,
            execution_config=ExecutionConfig(
                devices=(0, 1), device_memory_bytes=None,
                sharded_states=("wealth",), invariant_block_widths={"pref_type": 1},
            ),
        )
        dtype = np.dtype("float64" if jax.config.jax_enable_x64 else "float32")
        params = {
            "discount_factor": 0.5,
            "working": {"utility": {"weights": jnp.asarray((3, 1, -2), dtype=dtype)}},
            "terminal": {"utility": {"weights": jnp.asarray((8, 2, 6), dtype=dtype)}},
        }
        result = model.solve(params=params, log_level="off")
        values = [
            [period, regime, list(value.shape), value.dtype.str,
             np.asarray(value).tobytes().hex()]
            for period, regimes in sorted(result.values.items())
            for regime, value in sorted(regimes.items())
        ]
        print(json.dumps({
            "backend": jax.default_backend(), "devices": jax.device_count(),
            "x64": jax.config.jax_enable_x64, "values": values,
        }))
        """
    )
    command = [
        "pixi",
        "run",
        "--frozen",
        "-e",
        "tests-cpu",
        "python",
        "-c",
        script,
        str(int(x64)),
    ]
    process = subprocess.run(  # noqa: S603
        command,
        cwd=root,
        env={
            **os.environ,
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    (tmp_path / "child-stdout.json").write_text(process.stdout)
    (tmp_path / "child-stderr.log").write_text(process.stderr)
    (tmp_path / "child-command.json").write_text(json.dumps(command))
    actual = json.loads(process.stdout) if process.returncode == 0 else None
    dtype = np.dtype("float64" if x64 else "float32")
    # Wealth codes zero/one precede the original low/middle/high type codes.
    expected = {
        "backend": "cpu",
        "devices": 2,
        "x64": x64,
        "values": [
            [
                0,
                "working",
                [2, 3],
                dtype.str,
                np.asarray(((7, 2, 1), (8.5, 3.5, 2.5)), dtype=dtype).tobytes().hex(),
            ],
            [
                1,
                "terminal",
                [2, 3],
                dtype.str,
                np.asarray(((8, 2, 6), (9, 3, 7)), dtype=dtype).tobytes().hex(),
            ],
        ],
    }
    assert (process.returncode, actual) == (0, expected), process.stderr


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
