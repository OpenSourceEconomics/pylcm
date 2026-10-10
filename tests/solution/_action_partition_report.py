"""Solve the many-actions model on eight host devices and report what it shows.

Run as `pixi run -e tests-cpu python -m tests.solution._action_partition_report`
with `--x64 <0|1> --out <json>`
in a fresh process whose `XLA_FLAGS` force eight host CPU devices. It solves the
model on the ordinary route and on the action-partitioned route in several
layouts and writes one JSON report; `test_action_partitions_devices.py` asserts
on it. Budget probes validate the charged compiler counters and retained owners.
A report that stops early has no JSON, and the test reads that as a failure.

CPU establishes semantics and placement only, never GPU performance.
"""

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypedDict, Unpack

import jax
import numpy as np

from _lcm.execution.core_program import ResolvedCoreProgram
from _lcm.execution.output_layout import ResolvedOutputLayout
from _lcm.typing import JSONValue, ReferenceName, ShapeDtypePytree
from lcm import Model
from lcm.solver_api import SolutionResult
from lcm.typing import FloatND, RegimeName, StateName


class _ModelKwargs(TypedDict, total=False):
    widths: Mapping[str, int] | None
    action_partitions: Mapping[RegimeName, int] | None
    sharded_states: tuple[StateName, ...]
    typed: bool
    invariant_block_widths: Mapping[StateName, int] | None
    devices: tuple[int, ...] | None


class _LowerCandidate(Protocol):
    def __call__(
        self,
        *,
        label: str,
        resolved: ResolvedCoreProgram,
        layout: ResolvedOutputLayout,
        donated: tuple[ReferenceName, ...],
        internal_templates: Mapping[ReferenceName, ShapeDtypePytree],
    ) -> jax.stages.Lowered: ...


_FIXED_WIDTH = 5


def main() -> None:
    """Parse the precision and the output path, then write the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--x64", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", bool(arguments.x64))
    report = _report()
    arguments.out.write_text(json.dumps(report, indent=2, sort_keys=True))


def _report() -> dict[str, JSONValue]:
    from tests.test_models import many_actions  # noqa: PLC0415

    report: dict[str, JSONValue] = {
        "device_count": jax.device_count(),
        "backend": jax.default_backend(),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "n_actions": many_actions.N_ACTIONS,
    }
    reference = _solve(widths={"action_product": _FIXED_WIDTH})
    for partitions in (2, 3, 4, 8):
        got = _solve(
            widths={"action_product": _FIXED_WIDTH},
            action_partitions={"working": partitions},
        )
        report[f"action_only_fixed_{partitions}"] = {
            "mismatches": _bitwise_mismatches(
                got=got.values, expected=reference.values
            ),
            "working_layout": _layout(got.values[0]["working"]),
            "dead_layout": _layout(got.values[3]["dead"]),
        }

    planned_reference = _solve()
    for partitions in (2, 4):
        got = _solve(action_partitions={"working": partitions})
        report[f"action_only_planned_{partitions}"] = {
            "max_ulp": _max_ulp(got=got.values, expected=planned_reference.values)
        }

    report["simulation"] = {
        name: _simulated_frames(**kwargs)
        for name, kwargs in (
            ("planned", {}),
            ("fixed", {"widths": {"action_product": _FIXED_WIDTH}}),
        )
    }

    # A partitioned regime keeps 8 // partitions wealth shards; its reference is
    # the ordinary route on that many devices, so the per-device state cells,
    # and with them the compiled cell programs, are the same on both routes.
    for partitions in (2, 4, 8):
        got = _solve(
            widths={"action_product": _FIXED_WIDTH},
            sharded_states=("wealth",),
            action_partitions={"working": partitions},
        )
        same_state_mesh = _solve(
            widths={"action_product": _FIXED_WIDTH},
            sharded_states=("wealth",),
            devices=tuple(range(8 // partitions)),
        )
        report[f"state_by_action_{partitions}"] = {
            "mismatches_to_same_state_mesh": _bitwise_mismatches(
                got=got.values, expected=same_state_mesh.values
            ),
            "max_ulp_to_one_device": _max_ulp(
                got=got.values, expected=reference.values
            ),
            "working_layout": _layout(got.values[0]["working"]),
        }

    for name, kwargs in (
        ("action_only", {}),
        ("state_by_action", {"sharded_states": ("wealth",), "devices": (0, 1)}),
    ):
        blocked: _ModelKwargs = {
            "widths": {"action_product": _FIXED_WIDTH},
            "typed": True,
            "invariant_block_widths": {"pref_type": 1},
            **kwargs,
        }
        partitioned: _ModelKwargs = {
            **blocked,
            "devices": None,
            "action_partitions": {"working": 4},
        }
        got = _solve(**partitioned)
        reference_values = _solve(**blocked).values
        report[f"invariant_blocks_{name}"] = {
            "mismatches": _bitwise_mismatches(
                got=got.values, expected=reference_values
            ),
            "max_ulp": _max_ulp(got=got.values, expected=reference_values),
            "working_layout": _layout(got.values[0]["working"]),
        }

    changed = _solve(widths={"action_product": _FIXED_WIDTH}, scale=1.7)
    model = _model(
        widths={"action_product": _FIXED_WIDTH}, action_partitions={"working": 4}
    )
    first = model.solve(params=many_actions.get_params(), log_level="off").values
    second = model.solve(params=many_actions.get_params(scale=1.7), log_level="off")
    report["changed_params"] = {
        "mismatches": _bitwise_mismatches(got=second.values, expected=changed.values),
        "differs_from_first": not np.array_equal(
            np.asarray(second.values[0]["working"]), np.asarray(first[0]["working"])
        ),
    }

    report["programs"] = _program_report()
    report["budget"] = _budget_report()
    report["block_major"] = {
        "action_only": _block_major_report(),
        "state_by_action": _block_major_report(sharded_states=("wealth",)),
    }
    return report


def _model(
    *,
    widths: Mapping[str, int] | None = None,
    action_partitions: Mapping[RegimeName, int] | None = None,
    sharded_states: tuple[StateName, ...] = (),
    typed: bool = False,
    invariant_block_widths: Mapping[StateName, int] | None = None,
    devices: tuple[int, ...] | None = None,
) -> Model:
    from lcm import ExecutionConfig  # noqa: PLC0415
    from tests.test_models import many_actions  # noqa: PLC0415

    return many_actions.get_model(
        typed=typed,
        execution_config=ExecutionConfig(
            axis_widths=dict(widths or {}),
            action_partitions=dict(action_partitions or {}),
            sharded_states=sharded_states,
            invariant_block_widths=dict(invariant_block_widths or {}),
            devices=devices,
        ),
    )


def _solve(*, scale: float = 1.0, **kwargs: Unpack[_ModelKwargs]) -> SolutionResult:
    """Solve the model `_model(**kwargs)` builds; `scale` moves its params."""
    from tests.test_models import many_actions  # noqa: PLC0415

    typed = kwargs.get("typed", False)
    return _model(**kwargs).solve(
        params=many_actions.get_params(typed=typed, scale=scale), log_level="off"
    )


def _flat(
    values: Mapping[int, Mapping[RegimeName, FloatND | np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        f"{period}/{regime}": np.asarray(value)
        for period, by_regime in values.items()
        for regime, value in by_regime.items()
    }


def _bitwise_mismatches(
    *,
    got: Mapping[int, Mapping[RegimeName, FloatND | np.ndarray]],
    expected: Mapping[int, Mapping[RegimeName, FloatND | np.ndarray]],
) -> list[str]:
    got_flat, expected_flat = _flat(got), _flat(expected)
    if sorted(got_flat) != sorted(expected_flat):
        return ["<different period-regime keys>"]
    return [
        key
        for key, value in expected_flat.items()
        if got_flat[key].dtype != value.dtype
        or got_flat[key].shape != value.shape
        or got_flat[key].tobytes() != value.tobytes()
    ]


def _max_ulp(
    *,
    got: Mapping[int, Mapping[RegimeName, FloatND | np.ndarray]],
    expected: Mapping[int, Mapping[RegimeName, FloatND | np.ndarray]],
) -> float:
    worst = 0.0
    got_flat = _flat(got)
    for key, value in _flat(expected).items():
        spacing = np.spacing(np.maximum(np.abs(value), np.abs(got_flat[key])))
        worst = max(worst, float(np.max(np.abs(got_flat[key] - value) / spacing)))
    return worst


def _layout(value: jax.Array) -> dict[str, JSONValue]:
    sharding = value.sharding
    named = isinstance(sharding, jax.NamedSharding)
    return {
        "n_devices": len(sharding.device_set),
        "mesh": dict(sharding.mesh.shape) if named else None,
        "spec": [str(axis) for axis in sharding.spec] if named else None,
    }


def _simulated_frames(
    *, widths: Mapping[str, int] | None = None
) -> dict[str, JSONValue]:
    import jax.numpy as jnp  # noqa: PLC0415

    from tests.test_models import many_actions  # noqa: PLC0415

    initial_conditions = {
        "regime_id": jnp.full(8, many_actions.RegimeId.working),
        "age": jnp.zeros(8),
        "wealth": jnp.linspace(1.0, 10.0, 8),
    }
    frames = [
        _model(widths=widths, action_partitions=partitions)
        .simulate(
            params=many_actions.get_params(),
            initial_conditions=initial_conditions,
            seed=0,
            log_level="off",
        )
        .to_dataframe()
        for partitions in ({"working": 4}, {})
    ]
    decisions = [frame.drop(columns="value") for frame in frames]
    values = [frame["value"].to_numpy() for frame in frames]
    spacing = np.spacing(np.maximum(np.abs(values[0]), np.abs(values[1])))
    return {
        "decisions_and_states_equal": bool(decisions[0].equals(decisions[1])),
        "value_max_ulp": float(np.max(np.abs(values[0] - values[1]) / spacing)),
    }


def _program_report() -> dict[str, JSONValue]:
    """Compile the working regime's period-0 program and read its exchanges."""
    from _lcm.solution import backward_induction  # noqa: PLC0415
    from tests.test_models import many_actions  # noqa: PLC0415

    report: dict[str, JSONValue] = {}
    for name, partitions in (("partitioned", {"working": 4}), ("ordinary", {})):
        lowered: dict[str, jax.stages.Lowered] = {}
        original = backward_induction._lower_resolved_candidate
        backward_induction._lower_resolved_candidate = _LoweringRecorder(  # ty: ignore[invalid-assignment]
            lower=original, lowered=lowered
        )
        try:
            _model(
                widths={"action_product": _FIXED_WIDTH, "cell": 8},
                action_partitions=partitions,
            ).solve(params=many_actions.get_params(), log_level="off")
        finally:
            backward_induction._lower_resolved_candidate = original
        label = next(
            label
            for label in lowered
            if label.startswith("working") and "age 0" in label
        )
        compiled = lowered[label].compile()
        text = compiled.as_text()
        assert text is not None
        stats = compiled.memory_analysis()
        assert stats is not None
        report[name] = {
            "all_gather_shapes": sorted(
                _all_gather_result_shapes(hlo_text=text), key=str
            ),
            "temp_bytes": int(stats.temp_size_in_bytes),
        }
    return report


class _LoweringRecorder:
    """Lower a solve candidate as the solve does, keeping each lowering by label."""

    def __init__(
        self, *, lower: _LowerCandidate, lowered: dict[str, jax.stages.Lowered]
    ) -> None:
        self._lower = lower
        self._lowered = lowered

    def __call__(
        self,
        *,
        label: str,
        resolved: ResolvedCoreProgram,
        layout: ResolvedOutputLayout,
        donated: tuple[ReferenceName, ...],
        internal_templates: Mapping[ReferenceName, ShapeDtypePytree],
    ) -> jax.stages.Lowered:
        result = self._lower(
            label=label,
            resolved=resolved,
            layout=layout,
            donated=donated,
            internal_templates=internal_templates,
        )
        self._lowered[label] = result
        return result


def _all_gather_result_shapes(*, hlo_text: str) -> list[list[int]]:
    """Return the result shape of every all-gather in optimized HLO text."""
    shapes = []
    for line in hlo_text.splitlines():
        if " all-gather(" not in line and " all-gather-start(" not in line:
            continue
        result = re.split(r" all-gather(?:-start)?\(", line.split("=", 1)[1])[0]
        shapes.extend(
            [int(extent) for extent in match.group(1).split(",") if extent]
            for match in re.finditer(r"(?:f32|f64|s32|pred)\[([0-9,]*)\]", result)
        )
    return shapes


def _block_major_report(
    *, sharded_states: tuple[StateName, ...] = ()
) -> dict[str, JSONValue]:
    """Compare complete fixed-width values, panels and retained component lifetime."""
    import jax.numpy as jnp  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    from lcm import ExecutionConfig, InvariantBlockSchedule  # noqa: PLC0415
    from lcm.solver_api import SolutionResult  # noqa: PLC0415
    from tests.simulation.test_type_grouped_simulation import (  # noqa: PLC0415
        _assert_panels_identical,
    )
    from tests.solution.test_block_major_lifetime import (  # noqa: PLC0415
        _record_component_blocks,
    )
    from tests.test_models import many_actions  # noqa: PLC0415

    def model(schedule: InvariantBlockSchedule) -> Model:
        return many_actions.get_model(
            typed=True,
            execution_config=ExecutionConfig(
                devices=tuple(range(8 if sharded_states else 4)),
                device_memory_bytes=None,
                axis_widths={"action_product": _FIXED_WIDTH, "cell": 8, "subject": 3},
                invariant_block_widths={"pref_type": 1},
                invariant_block_schedule=schedule,
                action_partitions={"working": 4},
                sharded_states=sharded_states,
            ),
        )

    reference = model(InvariantBlockSchedule.PERIOD_MAJOR)
    blocked = model(InvariantBlockSchedule.BLOCK_MAJOR)
    params = many_actions.get_params(typed=True)
    expected = reference.solve(params=params, log_level="off")
    initial = {
        "regime_id": jnp.full(7, many_actions.RegimeId.working),
        "age": jnp.zeros(7),
        "wealth": jnp.linspace(1.0, 10.0, 7),
        "pref_type": jnp.asarray([2, 0, 2, 0, 0, 2, 0], dtype=jnp.int32),
    }
    panel_mismatches: dict[str, list[str]] = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        recorded = _record_component_blocks(monkeypatch=monkeypatch)
        got = blocked.solve(params=params, log_level="off")
        value_mismatches = _bitwise_mismatches(got=got.values, expected=expected.values)
        for route in ("split", "combined"):
            panels = [
                current.simulate(
                    params=params,
                    initial_conditions=initial,
                    solution=solution if route == "split" else None,
                    seed=7,
                    log_level="off",
                )
                for current, solution in ((blocked, got), (reference, expected))
            ]
            try:
                _assert_panels_identical(got=panels[0], want=panels[1])
            except AssertionError as error:
                panel_mismatches[route] = [str(error)]
            else:
                panel_mismatches[route] = []
            if route == "combined":
                combined_solution = panels[0].solution
                if not isinstance(combined_solution, SolutionResult):
                    value_mismatches.append("<combined missing solution>")
                else:
                    value_mismatches.extend(
                        f"combined/{coordinate}"
                        for coordinate in _bitwise_mismatches(
                            got=combined_solution.values, expected=expected.values
                        )
                    )
    return {
        "value_mismatches": value_mismatches,
        "panel_mismatches": panel_mismatches,
        "release": {
            "codes": sorted({code for code, _ in recorded}),
            "live_blocks": [code for code, block in recorded if not block.is_deleted()],
        },
    }


def _budget_report() -> dict[str, JSONValue]:
    """Solve at the exact represented-memory ceiling and immediately below it."""
    import logging  # noqa: PLC0415

    import lcm  # noqa: PLC0415
    from _lcm.solution import backward_induction  # noqa: PLC0415
    from lcm import ExecutionConfig  # noqa: PLC0415
    from lcm.exceptions import ExecutionPlanningError  # noqa: PLC0415
    from tests.execution.test_core_plan_record import _PlanRecords  # noqa: PLC0415
    from tests.test_models import many_actions  # noqa: PLC0415

    def solve(budget: int) -> SolutionResult:
        return many_actions.get_model(
            execution_config=ExecutionConfig(
                devices=(0, 1, 2, 3),
                axis_widths={"action_product": _FIXED_WIDTH, "cell": 8},
                action_partitions={"working": 4},
                device_memory_bytes=budget,
            )
        ).solve(params=many_actions.get_params(), log_level="debug")

    records = _PlanRecords()
    logger = logging.getLogger("lcm")
    lowered: dict[str, jax.stages.Lowered] = {}
    original = backward_induction._lower_resolved_candidate
    backward_induction._lower_resolved_candidate = _LoweringRecorder(  # ty: ignore[invalid-assignment]
        lower=original, lowered=lowered
    )
    logger.addHandler(records)
    try:
        reference = solve(2**30)
    finally:
        logger.removeHandler(records)
        backward_induction._lower_resolved_candidate = original
    native_reservations: dict[str, int] = {}
    for label, program in lowered.items():
        stats = program.compile().memory_analysis()
        assert stats is not None
        native_reservations[label] = max(
            int(stats.peak_memory_in_bytes),
            int(stats.argument_size_in_bytes)
            + int(stats.output_size_in_bytes)
            - int(stats.alias_size_in_bytes)
            + int(stats.temp_size_in_bytes),
        )
    value_bytes = 8 if jax.config.jax_enable_x64 else 4
    totals = []
    for record in records.records:
        native = max(
            size
            for label, size in native_reservations.items()
            if label.startswith(record.regime)
        )
        # Period 0 retains two unread values; period 1 retains one. Period 2
        # charges the terminal owner plus its transfer scratch; period 3 has neither.
        resident = max(record.stored_owner_bytes.values()) + (
            {0: 2, 1: 1, 2: 2, 3: 0}[record.period] * 8 * value_bytes
        )
        assert record.compiler_reservation_bytes == native, (
            f"Charged {record.compiler_reservation_bytes} bytes, "
            f"but the native executable requires {native}."
        )
        assert record.resident_bytes == resident
        totals.append(native + resident)
    ceiling = max(totals)
    refused = False
    try:
        solve(ceiling - 1)
    except ExecutionPlanningError:
        refused = True
    admitted = solve(ceiling)
    return {
        "module_file": lcm.__file__,
        "ceiling_bytes": ceiling,
        "gathered_bytes": 4 * 8 * (value_bytes + 4 + 1),
        "refused_below_ceiling": refused,
        "mismatches_at_ceiling": _bitwise_mismatches(
            got=admitted.values, expected=reference.values
        ),
        "terminal_values": np.asarray(admitted.values[3]["dead"]).tolist(),
    }


if __name__ == "__main__":
    main()
