"""Solve the many-actions model on eight host devices and report what it shows.

Run as `python -m tests.solution._action_partition_report --x64 <0|1> --out <json>`
in a fresh process whose `XLA_FLAGS` force eight host CPU devices. It solves the
model on the ordinary route and on the action-partitioned route in several
layouts and writes one JSON report; `test_action_partitions_devices.py` asserts
on it. Nothing here asserts: a report that stops early has no JSON, and the
test reads that as a failure.

CPU establishes semantics and placement only, never GPU performance.
"""

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np

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


def _report() -> dict[str, Any]:
    from tests.test_models import many_actions  # noqa: PLC0415

    report: dict[str, Any] = {
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
        blocked = {
            "widths": {"action_product": _FIXED_WIDTH},
            "typed": True,
            "invariant_block_widths": {"pref_type": 1},
            **kwargs,
        }
        got = _solve(
            **{
                **blocked,
                "devices": None,
                "action_partitions": {"working": 4},
            }
        )
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
    return report


def _model(
    *,
    widths: Mapping[str, int] | None = None,
    action_partitions: Mapping[str, int] | None = None,
    sharded_states: tuple[str, ...] = (),
    typed: bool = False,
    invariant_block_widths: Mapping[str, int] | None = None,
    devices: tuple[int, ...] | None = None,
) -> Any:
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


def _solve(**kwargs: Any) -> Any:
    """Solve the model `_model(**kwargs)` builds; `scale` moves its params."""
    from tests.test_models import many_actions  # noqa: PLC0415

    scale = kwargs.pop("scale", 1.0)
    typed = kwargs.get("typed", False)
    return _model(**kwargs).solve(
        params=many_actions.get_params(typed=typed, scale=scale), log_level="off"
    )


def _flat(values: Mapping) -> dict[str, np.ndarray]:
    return {
        f"{period}/{regime}": np.asarray(value)
        for period, by_regime in values.items()
        for regime, value in by_regime.items()
    }


def _bitwise_mismatches(*, got: Mapping, expected: Mapping) -> list[str]:
    got_flat, expected_flat = _flat(got), _flat(expected)
    if sorted(got_flat) != sorted(expected_flat):
        return ["<different period-regime keys>"]
    return [
        key
        for key, value in expected_flat.items()
        if got_flat[key].dtype != value.dtype
        or got_flat[key].tobytes() != value.tobytes()
    ]


def _max_ulp(*, got: Mapping, expected: Mapping) -> float:
    worst = 0.0
    got_flat = _flat(got)
    for key, value in _flat(expected).items():
        spacing = np.spacing(np.maximum(np.abs(value), np.abs(got_flat[key])))
        worst = max(worst, float(np.max(np.abs(got_flat[key] - value) / spacing)))
    return worst


def _layout(value: jax.Array) -> dict[str, Any]:
    sharding = value.sharding
    named = isinstance(sharding, jax.NamedSharding)
    return {
        "n_devices": len(sharding.device_set),
        "mesh": dict(sharding.mesh.shape) if named else None,
        "spec": [str(axis) for axis in sharding.spec] if named else None,
    }


def _simulated_frames(*, widths: Mapping[str, int] | None = None) -> dict[str, Any]:
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


def _program_report() -> dict[str, Any]:
    """Compile the working regime's period-0 program and read its exchanges."""
    from _lcm.solution import backward_induction  # noqa: PLC0415
    from tests.test_models import many_actions  # noqa: PLC0415

    report: dict[str, Any] = {}
    for name, partitions in (("partitioned", {"working": 4}), ("ordinary", {})):
        lowered: dict[str, Any] = {}
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
        report[name] = {
            "all_gather_shapes": sorted(
                _all_gather_result_shapes(hlo_text=text), key=str
            ),
            "temp_bytes": int(compiled.memory_analysis().temp_size_in_bytes),
        }
    return report


class _LoweringRecorder:
    """Lower a solve candidate as the solve does, keeping each lowering by label."""

    def __init__(self, *, lower: Any, lowered: dict[str, Any]) -> None:
        self._lower = lower
        self._lowered = lowered

    def __call__(self, *, label: str, **kwargs: Any) -> Any:
        result = self._lower(label=label, **kwargs)
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


if __name__ == "__main__":
    main()
