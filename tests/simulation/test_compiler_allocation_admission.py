"""Reported temporary allocations require admission before numerical dispatch."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jaxlib
import pytest

from lcm.exceptions import ExecutionPlanningError
from tests.simulation.test_feasibility_admission import _inputs


@pytest.mark.requires(device="cpu")
def test_two_action_simulation_refuses_reported_temporary_allocation(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    """A small feasibility output cannot hide its sorting allocation."""
    captured: dict[int, dict[str, object]] = {}
    original_compile = jax.stages.Lowered.compile
    original_dispatch = jax.stages.Compiled.__call__

    def capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        executable = original_compile(self, *args, **kwargs)
        hlo = executable.as_text()
        if hlo is not None and "_batched_feasibility_check" in hlo:
            stats = executable.memory_analysis()
            assert stats is not None
            scalars = {
                name: getattr(stats, name)
                for name in dir(stats)
                if not name.startswith("_")
                and isinstance(getattr(stats, name), (int, float, bool, str))
            }
            # Serialized storage is retained separately from the scalar report.
            scalars.pop("serialized_buffer_assignment_proto", None)
            payload: dict[str, object] = {
                "jax": jax.__version__,
                "jaxlib": jaxlib.__version__,
                "precision": 64 if jax.config.x64_enabled else 32,
                "budget_bytes": 16_384,
                "n_actions": 2,
                "stats": scalars,
                "dispatch_attempted": False,
            }
            captured[id(executable)] = payload
            (tmp_path / "feasibility.hlo.txt").write_text(hlo)
            (tmp_path / "buffer_assignment.pb").write_bytes(
                stats.serialized_buffer_assignment_proto
            )
        return executable

    def guard_dispatch(self: Any, *args: Any, **kwargs: Any) -> Any:
        if id(self) in captured:
            captured[id(self)]["dispatch_attempted"] = True
            pytest.fail("Feasibility reached numerical dispatch before budget refusal")
        return original_dispatch(self, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Lowered, "compile", capture)
    monkeypatch.setattr(jax.stages.Compiled, "__call__", guard_dispatch)
    model, params, initial = _inputs(budget=16_384, n_actions=2)
    try:
        with pytest.raises(ExecutionPlanningError, match="budget"):
            model.simulate(params=params, initial_conditions=initial, log_level="debug")
    finally:
        report = tmp_path / "compiler_stats.json"
        report.write_text(json.dumps(list(captured.values()), indent=2) + "\n")
        record_property("compiler_capture", str(report))
    assert len(captured) == 1
    payload = next(iter(captured.values()))
    assert payload["dispatch_attempted"] is False
    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert stats["temp_size_in_bytes"] > 16_384
    assert stats["peak_memory_in_bytes"] < 16_384
