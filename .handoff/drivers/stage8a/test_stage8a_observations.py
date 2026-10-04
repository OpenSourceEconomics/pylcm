"""Existing compile/log helpers supply truthful per-call observations."""

import importlib
import json
import logging
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax.monitoring
import pytest
from test_stage8a_population import owner_driver as _imported_owner_driver
from test_stage8a_receipts import (
    fragment_helpers as _imported_fragment_helpers,
)

owner_driver = _imported_owner_driver
fragment_helpers = _imported_fragment_helpers


@pytest.mark.parametrize(
    "failure",
    ["success", "numeric", "numeric_cleanup", "numeric_persist", "numeric_parse"],
)
# Keep the complete concrete protocol and its negative controls adjacent.
def test_call_observations_close_files_and_preserve_actual_failure(  # noqa: C901
    *,
    owner_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    fragment_helpers: ModuleType,
) -> None:
    """Actual monitoring events and phase logs survive success or original failure.

    Only the GPU sampler/exclusivity boundary is recorded on this CPU host.
    """
    monkeypatch.syspath_prepend(
        str(Path(str(owner_driver.__file__)).resolve().parents[1] / "stage5b")
    )
    harness = importlib.import_module("stage3_arms")
    counts = importlib.import_module("benchmarks.asv._compile_counters")
    stopped = []
    write = fragment_helpers.write_json_atomically

    def persist(**kwargs: Any) -> None:
        if failure == "numeric_persist":
            raise OSError("telemetry persistence refused")
        write(**kwargs)

    monkeypatch.setattr(fragment_helpers, "write_json_atomically", persist)
    if failure == "numeric_parse":

        def parse(**_kwargs: Any) -> list[object]:
            raise OSError("telemetry parsing refused")

        monkeypatch.setattr(owner_driver, "_numerical_intervals", parse)
    readiness = []
    monkeypatch.setattr(
        jax,
        "live_arrays",
        lambda: [
            SimpleNamespace(
                block_until_ready=lambda: readiness.append("ready"),
            )
        ],
    )
    monkeypatch.setattr(
        harness,
        "_Exclusivity",
        lambda: SimpleNamespace(
            at_start={},
            record=lambda: {"exclusive": True},
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="", returncode=0),
    )

    def sampler(*_args: Any, **kwargs: Any) -> SimpleNamespace:
        kwargs["stdout"].write("2026/10/03 00:00:00.000, GPU-0, 10, 20\n")
        kwargs["stdout"].flush()

        def wait(**_kwargs: Any) -> None:
            stopped.append("wait")
            if failure == "numeric_cleanup":
                raise OSError("sampler cleanup refused")

        return SimpleNamespace(
            poll=lambda: None,
            terminate=lambda: stopped.append("terminate"),
            wait=wait,
            stderr=None,
        )

    monkeypatch.setattr(subprocess, "Popen", sampler)
    logger = logging.getLogger("lcm")
    monkeypatch.setattr(logger, "level", logging.INFO)
    handlers = list(logger.handlers)
    error = RuntimeError("actual numeric call refused")

    def call() -> int:
        readiness.append("call")
        for event, number in (
            (counts.TRACE_EVENT, 1),
            (counts.LOWERING_EVENT, 2),
            (counts.COMPILE_EVENT, 3),
        ):
            for _ in range(number):
                jax.monitoring.record_event_duration_secs(event, 0.001)
        logger.info("solve call abcdef phase public_simulate begin")
        if failure != "success":
            raise error
        logger.info(
            "solve call abcdef phase public_simulate end status=ok seconds=0.001"
        )
        return 17

    result = None
    caught = None
    try:
        result, _record = owner_driver._observe_call(
            call=call,
            out=tmp_path,
            label="worker_call",
            gpu_uuids=["GPU-0"],
        )
    except (AttributeError, RuntimeError, OSError) as current:
        caught = current
    path = tmp_path / "worker_call.observations.json"
    record = json.loads(path.read_bytes()) if path.exists() else None
    assert (
        result,
        caught is error if failure != "success" else caught,
        None if record is None else record["compile_requests"],
        None if record is None else record["status"],
        stopped,
        logger.handlers == handlers,
        None if record is None else record["debug_core_records"],
        (tmp_path / "worker_call.log").exists(),
        readiness,
    ) == (
        None if failure != "success" else 17,
        True if failure != "success" else None,
        None
        if failure == "numeric_persist"
        else {"trace_requests": 1, "lowering_requests": 2, "compile_requests": 3},
        None
        if failure == "numeric_persist"
        else "failed"
        if failure != "success"
        else "completed",
        ["terminate", "wait"],
        True,
        None,
        True,
        ["call", "ready"] if failure == "success" else ["call"],
    )
