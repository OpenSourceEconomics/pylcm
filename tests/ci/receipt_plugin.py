"""Opt-in pytest plugin: publish one canonical receipt per CI invocation.

Enabled only when the environment variable named by `RECEIPT_ENV_VAR`
(`PYLCM_CI_RECEIPT`) is set to a writable directory. Unset, this plugin does
nothing at all --- no file, no extra hook work beyond one `os.environ.get` per
`maybe_register` call. It never changes what is collected, selected, or
deselected; it only observes and records what pytest already decided.

Three things this writer has to get right, because the previous one did not:

*Who publishes.* Under `xdist` the controller is the only process that sees the
whole invocation, so it alone writes the canonical record. Each worker writes a
supporting record under its own name. Nothing is ever written twice to one
path, so no participant can silently overwrite another's evidence.

*What the selection is.* `pytest_collection_finish` on an `xdist` controller
sees an empty `session.items`: the controller does not run the tests and, in
the distributed protocol, the authoritative collection arrives from the workers
through `pytest_xdist_node_collection_finished`. The canonical selection is
that collection, accepted only when every participating worker agrees on it.

*What it means.* Selected, executed, deselected and skipped are four different
populations and are recorded as four different fields. A receipt also carries
the identity of the thing that produced it --- source SHA, invocation id, OS,
precision, backend and device count, worker and distribution settings, the
active CI policy, the runtime, and the exit status --- so that two receipts can
be told apart and neither can be read as evidence about the other.

Every way the evidence can be partial --- a worker that never reported, workers
that disagree about the collection, an executed node that was never selected, a
missing identity field --- sets `inventory_complete` to `false` and names
itself in `incomplete_reasons`. An incomplete receipt must never be read as a
complete inventory, which is why the flag is written rather than inferred from
the presence of the file.

`tests/conftest.py` calls `maybe_register` from `pytest_configure`, which is a
no-op unless the env var is set, rather than declaring this as a
`pytest_plugins` entry that would import and hook it unconditionally.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import jax
import pytest

RECEIPT_ENV_VAR = "PYLCM_CI_RECEIPT"

# Bumped when the published field set changes, so a reader can refuse an old one.
SCHEMA_VERSION = 2

# Role of the single record published by the controller, or by a serial run.
CANONICAL_ROLE = "canonical"

# Role of a supporting record published by one `xdist` worker.
WORKER_ROLE = "worker"

# Where the controller keeps the id it mints, so that a second `maybe_register`
# on the same config joins the same invocation instead of starting a new one.
INVOCATION_KEY: pytest.StashKey[str] = pytest.StashKey()


def _identity_environment() -> dict[str, str | None]:
    """Return the CI identity of this process, as far as the environment knows it."""
    return {
        "source_sha": os.environ.get("GITHUB_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "workflow_job": os.environ.get("GITHUB_JOB"),
        "runner_os": os.environ.get("RUNNER_OS") or platform.system(),
    }


def _runtime_identity() -> dict[str, Any]:
    """Return the interpreter and JAX runtime this invocation actually ran on."""
    # A receipt has to stay writable in an environment where the backend
    # cannot be initialised at all, so a broken runtime is reported in the
    # record rather than raised out of a session-finish hook.
    precision: int | None = None
    backend: str | None = None
    device_count: int | None = None
    try:
        precision = 64 if jax.config.jax_enable_x64 else 32
        backend = jax.default_backend()
        device_count = len(jax.devices())
    except Exception as error:  # noqa: BLE001 - identity is reported, never raised
        backend = f"unavailable: {type(error).__name__}"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "pytest_version": pytest.__version__,
        "platform": platform.platform(),
        "precision": precision,
        "backend": backend,
        "device_count": device_count,
    }


def _policy_identity(config: pytest.Config) -> dict[str, Any]:
    """Return the CI policy and worker/distribution settings this invocation used."""
    option = getattr(config, "option", None)
    return {
        "ci_policy": getattr(option, "ci_policy", None),
        "hardware_profile": getattr(option, "hardware_profile", None),
        "policy_child": bool(getattr(option, "policy_child", False)),
        "numprocesses": getattr(option, "numprocesses", None),
        "dist": getattr(option, "dist", None),
        "markexpr": getattr(option, "markexpr", None) or None,
        "keyword": getattr(option, "keyword", None) or None,
    }


def junit_identity(*, nodeid: str) -> tuple[str, str]:
    """Return the `(classname, name)` pair JUnit writes for `nodeid`.

    pytest names a case by its dotted module (plus class) and the test's own
    name, so this is the only stable join between a recorded selection and an
    outcome in the XML.
    """
    head, _, name = nodeid.rpartition("::")
    path, _, class_path = head.partition("::")
    classname = path.removesuffix(".py").replace("/", ".").replace("\\", ".")
    if class_path:
        classname = f"{classname}.{class_path.replace('::', '.')}"
    return classname, name


def unique_junit_cases(*, path: Path) -> set[tuple[str, str]]:
    """Return each distinct test in a JUnit file, counted once.

    pytest writes one `<testcase>` per *phase* that produced a report worth
    recording, so a test that errors in setup and again in teardown appears
    twice. Counting elements would inflate the population; a test is identified
    by its `(classname, name)` pair and counted once however many phases it
    contributed.
    """
    report = ET.parse(path).getroot()  # noqa: S314 - local file written by pytest
    return {
        (case.get("classname") or "", case.get("name") or "")
        for case in report.iter("testcase")
    }


def reconcile_with_junit(
    *, receipt: Mapping[str, Any], junit_path: Path
) -> dict[str, Any]:
    """Match a receipt's recorded selection against the outcomes actually written.

    Returns both asymmetric differences and whether they are empty. Phases are
    collapsed first, so a test reported in setup, call and teardown is one test
    on both sides of the comparison rather than three.
    """
    recorded = {
        junit_identity(nodeid=nodeid) for nodeid in receipt["selected_node_ids"]
    }
    observed = unique_junit_cases(path=junit_path)
    missing = sorted(recorded - observed)
    unexpected = sorted(observed - recorded)
    return {
        "selected_count": len(recorded),
        "junit_case_count": len(observed),
        "selected_without_outcome": missing,
        "outcome_without_selection": unexpected,
        "reconciled": not missing and not unexpected,
    }


def _merge_outcome(*, previous: str | None, report: pytest.TestReport) -> str:
    """Collapse one test's setup/call/teardown reports into a single outcome.

    A failure anywhere outranks everything else; a skip outranks a pass,
    because a test skipped during setup still reports a passing teardown.
    """
    current = "failed" if report.failed else ("skipped" if report.skipped else "passed")
    order = ("passed", "skipped", "failed")
    if previous is None:
        return current
    return max((previous, current), key=order.index)


def agreed_collection(
    *, collections: Mapping[str, Sequence[str]]
) -> tuple[list[str], list[str]]:
    """Return the collection every worker reported, and why they may disagree.

    Under `xdist` each worker collects the whole suite and then executes a
    disjoint share of it, so their collections must be identical. When they are
    not, the union is published --- refusing to invent one worker's view as the
    invocation's selection --- and the disagreement is named, so the record
    cannot be read as a complete inventory.
    """
    distinct = {worker: sorted(ids) for worker, ids in collections.items()}
    if not distinct:
        return [], []
    first = distinct[min(distinct)]
    disagreeing = sorted(worker for worker, ids in distinct.items() if ids != first)
    if not disagreeing:
        return first, []
    union = sorted(set().union(*distinct.values()))
    return union, [f"workers disagree about the collection: {disagreeing}"]


def _requested_workers(*, config: pytest.Config) -> int | None:
    """Return how many `xdist` workers were asked for, when that is a plain count."""
    requested = getattr(getattr(config, "option", None), "numprocesses", None)
    return requested if isinstance(requested, int) and requested > 0 else None


def _missing_worker_reasons(
    *, config: pytest.Config, collections: Mapping[str, Sequence[str]]
) -> list[str]:
    """Name the gap when fewer workers reported than the invocation asked for."""
    requested = _requested_workers(config=config)
    if requested is None or len(collections) == requested:
        return []
    return [
        (
            f"{len(collections)} of {requested} workers reported a collection: "
            f"{sorted(collections)}"
        )
    ]


def _worker_id(*, config: pytest.Config) -> str | None:
    """Return this process's `xdist` worker id, or `None` on the controller."""
    workerinput = getattr(config, "workerinput", None)
    if workerinput is None:
        return None
    return str(workerinput.get("workerid", os.environ.get("PYTEST_XDIST_WORKER")))


def _invocation_id(*, config: pytest.Config) -> str:
    """Return the id shared by a controller and its workers.

    The controller mints one and hands it to every worker through
    `workerinput`, so an invocation's records can be joined even though their
    filenames necessarily differ.
    """
    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None and "pylcm_invocation_id" in workerinput:
        return str(workerinput["pylcm_invocation_id"])
    stash = getattr(config, "stash", None)
    if stash is None:
        return os.environ.get("PYLCM_CI_INVOCATION_ID") or uuid.uuid4().hex
    if INVOCATION_KEY not in stash:
        stash[INVOCATION_KEY] = (
            os.environ.get("PYLCM_CI_INVOCATION_ID") or uuid.uuid4().hex
        )
    return stash[INVOCATION_KEY]


def receipt_filename(
    *,
    junitxml_path: str | None,
    worker_id: str | None,
    runner_os: str | None,
    precision: int | None,
    fallback: str,
) -> str:
    """Name a receipt so that no two participants and no two lanes collide.

    The JUnit basename alone is not unique: the same invocation runs on several
    operating systems and at both precisions, and every worker of one `xdist`
    invocation shares it. The OS, the precision and the worker id are therefore
    all part of the name, and an invocation with no JUnit file falls back to
    its own invocation id.
    """
    stem = Path(junitxml_path).stem if junitxml_path else f"invocation-{fallback}"
    lane = f"{(runner_os or 'unknown-os').lower()}-fp{precision or 'unknown'}"
    role = f".worker-{worker_id}" if worker_id else ""
    return f"{stem}.{lane}{role}.receipt.json"


class _ReceiptCollector:
    """Accumulates one invocation's receipt fields across pytest hooks."""

    def __init__(self, *, receipt_root: Path, config: pytest.Config) -> None:
        self.receipt_root = receipt_root
        self.config = config
        self.worker_id: str | None = _worker_id(config=config)
        self.invocation_id = _invocation_id(config=config)
        self.start_wall: str | None = None
        self.start_monotonic: float | None = None
        self.own_collection: list[str] = []
        self.worker_collections: dict[str, list[str]] = {}
        self.deselected: list[str] = []
        self.skipped: list[str] = []
        self.outcomes: dict[str, str] = {}
        self.phase_counts: Counter[str] = Counter()

    @property
    def is_controller(self) -> bool:
        """Report whether this process publishes the canonical record."""
        return self.worker_id is None

    def on_sessionstart(self) -> None:
        """Start the clock for this invocation."""
        self.start_wall = datetime.now(UTC).isoformat()
        self.start_monotonic = time.monotonic()

    def on_collection_finish(self, *, items: Sequence[pytest.Item]) -> None:
        """Record what this process itself collected after deselection settled.

        On a serial run that is the invocation's selection. On an `xdist`
        controller it is empty, and the workers supply the real one.
        """
        self.own_collection = [item.nodeid for item in items]

    def on_worker_collection(self, *, worker_id: str, ids: Iterable[str]) -> None:
        """Record one worker's authoritative collection."""
        self.worker_collections[worker_id] = list(ids)

    def on_deselected(self, *, items: Sequence[pytest.Item]) -> None:
        """Record nodes pytest removed from the selection before running anything."""
        self.deselected.extend(item.nodeid for item in items)

    def on_logreport(self, *, report: pytest.TestReport) -> None:
        """Fold one phase report into this invocation's executed population."""
        self.phase_counts[report.nodeid] += 1
        if report.skipped and report.nodeid not in self.skipped:
            self.skipped.append(report.nodeid)
        self.outcomes[report.nodeid] = _merge_outcome(
            previous=self.outcomes.get(report.nodeid), report=report
        )

    def selection(self) -> tuple[list[str], list[str]]:
        """Return the canonical selection and every reason it may be incomplete."""
        if not self.worker_collections:
            reasons = (
                ["no worker reported its collection"]
                if _requested_workers(config=self.config)
                else []
            )
            return sorted(self.own_collection), reasons
        agreed, reasons = agreed_collection(collections=self.worker_collections)
        return agreed, [
            *reasons,
            *_missing_worker_reasons(
                config=self.config, collections=self.worker_collections
            ),
        ]

    def payload(self, *, exitstatus: int, junitxml_path: str | None) -> dict[str, Any]:
        """Assemble this process's record, complete with why it may be partial."""
        selected, reasons = self.selection()
        executed = sorted(self.outcomes)
        chosen = set(selected)
        reasons.extend(
            f"executed node was never selected: {nodeid}"
            for nodeid in executed
            if selected and nodeid not in chosen
        )
        environment = _identity_environment()
        runtime = _runtime_identity()
        reasons.extend(
            f"missing identity field: {field}"
            for field, value in (
                ("source_sha", environment["source_sha"]),
                ("precision", runtime["precision"]),
            )
            if value is None
        )
        elapsed = (
            None
            if self.start_monotonic is None
            else time.monotonic() - self.start_monotonic
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "role": CANONICAL_ROLE if self.is_controller else WORKER_ROLE,
            "worker_id": self.worker_id,
            "invocation_id": self.invocation_id,
            "start_wall_clock": self.start_wall,
            "end_wall_clock": datetime.now(UTC).isoformat(),
            "elapsed_monotonic_seconds": elapsed,
            "exit_status": int(exitstatus),
            "environment": environment,
            "runtime": runtime,
            "policy": _policy_identity(self.config),
            "selected_node_ids": selected,
            "executed_node_ids": executed,
            "deselected_node_ids": sorted(set(self.deselected)),
            "skipped_node_ids": sorted(set(self.skipped)),
            "outcomes": dict(sorted(self.outcomes.items())),
            "phase_counts": dict(sorted(self.phase_counts.items())),
            "worker_collections": {
                worker: sorted(ids)
                for worker, ids in sorted(self.worker_collections.items())
            },
            "own_collection_node_ids": sorted(self.own_collection),
            "junit_path": junitxml_path,
            "inventory_complete": not reasons,
            "incomplete_reasons": sorted(set(reasons)),
        }

    def on_sessionfinish(self, *, exitstatus: int, junitxml_path: str | None) -> None:
        """Publish this process's record under a name nothing else can claim."""
        payload = self.payload(exitstatus=exitstatus, junitxml_path=junitxml_path)
        destination = self.receipt_root / receipt_filename(
            junitxml_path=junitxml_path,
            worker_id=self.worker_id,
            runner_os=payload["environment"]["runner_os"],
            precision=payload["runtime"]["precision"],
            fallback=self.invocation_id,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


class _ReceiptHooks:
    """Hook implementations delegating to a `_ReceiptCollector`."""

    def __init__(self, *, collector: _ReceiptCollector) -> None:
        self._collector = collector

    def pytest_sessionstart(self) -> None:
        """Start this invocation's record."""
        self._collector.on_sessionstart()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Take this process's own settled collection."""
        self._collector.on_collection_finish(items=session.items)

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        """Take the deselected population, which is not the skipped one."""
        self._collector.on_deselected(items=items)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Take one phase report, controller-side even under `xdist`."""
        self._collector.on_logreport(report=report)

    def pytest_configure_node(self, node: Any) -> None:
        """Hand each worker the controller's invocation id before it starts."""
        node.workerinput["pylcm_invocation_id"] = self._collector.invocation_id

    # keyword-only-exempt: library-callback=pytest_xdist
    def pytest_xdist_node_collection_finished(
        self, node: Any, ids: Sequence[str]
    ) -> None:
        """Record one worker's authoritative collection as the controller sees it."""
        self._collector.on_worker_collection(
            worker_id=getattr(node, "gateway", node).id, ids=ids
        )

    # keyword-only-exempt: library-callback=pytest
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        """Publish the record once this process's session has settled."""
        option = getattr(session.config, "option", None)
        junit = getattr(option, "xmlpath", None) if option is not None else None
        self._collector.on_sessionfinish(exitstatus=exitstatus, junitxml_path=junit)


def pytest_configure(config: pytest.Config) -> None:
    """Register the collector when this module is loaded with `-p`.

    `tests/conftest.py` calls `maybe_register` directly for the project's own
    invocations; loading the module as a plugin is how a regression test drives
    it in a scratch directory that has no project conftest.
    """
    maybe_register(config)


def maybe_register(config: pytest.Config) -> None:
    """Register the receipt collector plugin if `PYLCM_CI_RECEIPT` is set.

    Idempotent per `config`: safe to call more than once because `hasplugin` is
    checked first. Under `xdist` the controller and each worker each get their
    own `Config`/`pluginmanager` and therefore their own collector; they no
    longer share a destination, because the worker id, the OS and the precision
    are all part of the filename and only the controller writes the canonical
    record.
    """
    raw = os.environ.get(RECEIPT_ENV_VAR)
    if not raw:
        return
    if config.pluginmanager.hasplugin("pylcm-ci-receipt"):
        return
    collector = _ReceiptCollector(receipt_root=Path(raw), config=config)
    config.pluginmanager.register(
        _ReceiptHooks(collector=collector), "pylcm-ci-receipt"
    )
