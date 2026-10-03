"""Each production phase preserves completed and interrupted output attempts."""

import sys
from types import SimpleNamespace

import pytest
from test_stage8a_main_outputs import (
    production_runtime as _imported_production_runtime,
)
from test_stage8a_population import (
    fragment_helpers as _imported_fragment_helpers,
)
from test_stage8a_population import owner_driver as _imported_owner_driver

production_runtime = _imported_production_runtime
fragment_helpers = _imported_fragment_helpers
owner_driver = _imported_owner_driver


@pytest.mark.parametrize("phase", ["plan", "run", "reference", "collect"])
@pytest.mark.parametrize("previous", ["completed", "interrupted"])
def test_main_preserves_nonempty_previous_attempt_before_any_work(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    previous: str,
) -> None:
    """A nonempty phase output is refused before construction or artifact writes."""
    context = production_runtime
    out = context.output / "job-0000" if phase == "run" else context.output
    out.mkdir(parents=True)
    (out / "numeric.log").write_bytes(b"immutable original log\n")
    if previous == "completed":
        (out / "receipt.json").write_bytes(b'{"status":"completed"}')
    original = {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    argv = [
        "driver",
        phase,
        "--plan-directory",
        str(context.directory),
        "--out",
        str(context.output),
        "--aca-slurm-src",
        str(context.source),
    ]
    if phase in {"run", "collect"}:
        argv += [
            "--planning-receipt",
            str(out / "planning.json"),
            "--planning-receipt-sha256",
            "a" * 64,
        ]
    if phase == "run":
        monkeypatch.setenv("SLURM_PROCID", "0")
        argv += ["--job-from-slurm-procid"]
    elif phase == "collect":
        argv += [
            "--reference",
            str(out / "reference"),
            "--reference-receipt-sha256",
            "b" * 64,
            "--worker-receipts",
            str(out / "workers"),
        ]
    monkeypatch.setattr(sys, "argv", argv)
    caught = None
    try:
        context.driver.main()
    except FileExistsError as error:
        caught = (type(error).__name__, str(error))
    final = {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }

    assert (caught, context.calls, final) == (
        ("FileExistsError", "Phase output must be absent or empty"),
        [],
        original,
    )
