"""Opt-in pytest plugin: write one JSON receipt per invocation.

Enabled only when the environment variable named by `RECEIPT_ENV_VAR`
(`PYLCM_CI_RECEIPT`) is set to a writable path. Unset, this plugin does nothing
at all — no file, no extra hook work beyond one `os.environ.get` per hook. It
never changes what is collected, selected, or deselected; it only observes and
records what pytest already decided, alongside the wall-clock/monotonic timing
and the JUnit file the same invocation is already writing.

`PYLCM_CI_RECEIPT` names a directory, not a file, so `cpu.yml` can export it
ONCE per job (every CPU-suite pytest invocation already carries its own
`--junitxml`) and still get one receipt per invocation: the destination
filename is derived from that invocation's own JUnit basename. An invocation
run without `--junitxml` falls back to a start-time-based name so two such
receipts never collide inside one job's directory.

`tests/conftest.py` calls `maybe_register` from `pytest_configure`, which is a
no-op unless the env var is set, rather than declaring this as a
`pytest_plugins` entry that would import and hook it unconditionally.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

RECEIPT_ENV_VAR = "PYLCM_CI_RECEIPT"


class _ReceiptCollector:
    """Accumulates one invocation's receipt fields across pytest hooks."""

    def __init__(self, *, receipt_root: Path) -> None:
        self.receipt_root = receipt_root
        self.start_wall: str | None = None
        self.start_monotonic: float | None = None
        self.selected: list[str] = []
        self.deselected_count = 0
        self.skipped: list[str] = []

    def on_sessionstart(self) -> None:
        self.start_wall = datetime.now(UTC).isoformat()
        self.start_monotonic = time.monotonic()

    def on_collection_finish(self, items: Sequence[pytest.Item]) -> None:
        # Called after collection AND deselection have both settled, so
        # `items` is already the final selected set.
        self.selected = [item.nodeid for item in items]

    def on_deselected(self, items: Sequence[pytest.Item]) -> None:
        self.deselected_count += len(items)

    def on_logreport(self, report: pytest.TestReport) -> None:
        if report.skipped and report.nodeid not in self.skipped:
            self.skipped.append(report.nodeid)

    def on_sessionfinish(self, *, exitstatus: int, junitxml_path: str | None) -> None:
        end_wall = datetime.now(UTC).isoformat()
        elapsed = (
            None
            if self.start_monotonic is None
            else time.monotonic() - self.start_monotonic
        )
        payload: dict[str, Any] = {
            "start_wall_clock": self.start_wall,
            "end_wall_clock": end_wall,
            "elapsed_monotonic_seconds": elapsed,
            "exit_status": int(exitstatus),
            "selected_node_ids": self.selected,
            "deselected_count": self.deselected_count,
            "skipped_node_ids": self.skipped,
            "junit_path": junitxml_path,
        }
        stem = Path(junitxml_path).stem if junitxml_path else self._fallback_stem()
        destination = self.receipt_root / f"{stem}.receipt.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _fallback_stem(self) -> str:
        # No --junitxml on this invocation: fall back to a start-time-based
        # name so two receipt-less invocations in the same job/directory don't
        # collide (they would need to start in the same microsecond).
        basis = self.start_wall or datetime.now(UTC).isoformat()
        return "invocation-" + basis.replace(":", "").replace(".", "")


class _ReceiptHooks:
    """Hook implementations delegating to a `_ReceiptCollector`."""

    def __init__(self, *, collector: _ReceiptCollector) -> None:
        self._collector = collector

    def pytest_sessionstart(self) -> None:
        self._collector.on_sessionstart()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self._collector.on_collection_finish(session.items)

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self._collector.on_deselected(items)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self._collector.on_logreport(report)

    # keyword-only-exempt: library-callback=pytest
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        option = getattr(session.config, "option", None)
        junit = getattr(option, "xmlpath", None) if option is not None else None
        self._collector.on_sessionfinish(exitstatus=exitstatus, junitxml_path=junit)


def maybe_register(config: pytest.Config) -> None:
    """Register the receipt collector plugin if `PYLCM_CI_RECEIPT` is set.

    Idempotent per `config`: safe to call more than once (e.g. if a future
    conftest change calls it from two hooks) because `hasplugin` is checked
    first. Under xdist, the controller and each worker each get their own
    `Config`/`pluginmanager`, so each writes to the SAME `destination` unless
    the caller varies it per worker; `cpu.yml`'s heavy multi-device invocations
    that this plugin targets all run at `-n 0`, so no such collision occurs
    there. A future `-n>0` use should key the path on `PYTEST_XDIST_WORKER`.
    """
    raw = os.environ.get(RECEIPT_ENV_VAR)
    if not raw:
        return
    if config.pluginmanager.hasplugin("pylcm-ci-receipt"):
        return
    collector = _ReceiptCollector(receipt_root=Path(raw))
    config.pluginmanager.register(
        _ReceiptHooks(collector=collector), "pylcm-ci-receipt"
    )
