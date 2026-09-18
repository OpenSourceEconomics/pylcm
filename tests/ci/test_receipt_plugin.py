"""A receipt is a complete inventory or it says, in the file, that it is not.

The invocation-level cases run a real pytest in a scratch directory, serially
and under `xdist`, because the defect the previous writer had --- an empty
canonical selection under `-n 2` --- is invisible to anything that does not
actually distribute. The failure modes that a real run cannot be asked to
produce on demand --- a worker that never reports, two workers that disagree,
a controller that finishes before a worker's record lands --- are driven
through the collector directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from xml.etree import ElementTree as ET

import pytest

from tests.ci.receipt_plugin import (
    CANONICAL_ROLE,
    RECEIPT_ENV_VAR,
    SCHEMA_VERSION,
    WORKER_ROLE,
    _ReceiptCollector,
    agreed_collection,
    junit_identity,
    receipt_filename,
    reconcile_with_junit,
    unique_junit_cases,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SUITE = '''
"""Three passing tests, one skipped test and one parametrised family."""

import pytest


def test_first() -> None:
    """Pass."""


def test_second() -> None:
    """Pass."""


@pytest.mark.skip(reason="expected skip")
def test_third() -> None:
    """Skip, by an expectation this suite declares itself."""


@pytest.mark.parametrize("case", [1, 2])
def test_family(case: int) -> None:
    """Pass twice."""
'''

ALL_SKIPPED_SUITE = '''
"""Every test is an expected skip."""

import pytest


@pytest.mark.skip(reason="expected skip")
def test_first() -> None:
    """Skip."""


@pytest.mark.skip(reason="expected skip")
def test_second() -> None:
    """Skip."""
'''


def _run_pytest(
    *, directory: Path, suite: str, arguments: tuple[str, ...]
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run one real pytest invocation with the receipt plugin loaded.

    Returns the finished process, the receipt directory and the JUnit file, so
    a case can read back exactly what that invocation published.
    """
    (directory / "test_suite.py").write_text(suite, encoding="utf-8")
    receipts = directory / "receipts"
    junit = directory / "junit.xml"
    environment = {
        **os.environ,
        RECEIPT_ENV_VAR: str(receipts),
        "GITHUB_SHA": "0" * 40,
        "RUNNER_OS": "Linux",
        "PYTHONPATH": os.pathsep.join(
            (str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", ""))
        ),
    }
    process = subprocess.run(  # noqa: S603
        (
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.ci.receipt_plugin",
            "-p",
            "no:cacheprovider",
            str(directory / "test_suite.py"),
            f"--junitxml={junit}",
            *arguments,
        ),
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return process, receipts, junit


def _records(*, receipts: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return the one canonical record and the supporting worker records by id."""
    published = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(receipts.glob("*.receipt.json"))
    ]
    canonical = [record for record in published if record["role"] == CANONICAL_ROLE]
    assert len(canonical) == 1, published
    workers = {
        record["worker_id"]: record
        for record in published
        if record["role"] == WORKER_ROLE
    }
    assert len(workers) == len(published) - 1, published
    return canonical[0], workers


def test_a_serial_invocation_publishes_its_own_complete_selection(
    tmp_path: Path,
) -> None:
    """With no workers the process that collected is the one that publishes."""
    process, receipts, junit = _run_pytest(
        directory=tmp_path, suite=SUITE, arguments=()
    )

    assert process.returncode == 0, process.stdout
    canonical, workers = _records(receipts=receipts)
    assert workers == {}
    assert canonical["schema_version"] == SCHEMA_VERSION
    assert canonical["worker_id"] is None
    assert len(canonical["selected_node_ids"]) == 5
    assert len(canonical["executed_node_ids"]) == 5
    assert canonical["skipped_node_ids"] == ["test_suite.py::test_third"]
    assert canonical["deselected_node_ids"] == []
    assert canonical["inventory_complete"], canonical["incomplete_reasons"]
    assert reconcile_with_junit(receipt=canonical, junit_path=junit)["reconciled"]


def test_a_two_worker_invocation_publishes_the_distributed_collection(
    tmp_path: Path,
) -> None:
    """The controller's canonical selection is the workers' collection, not `[]`.

    This is the regression the previous writer failed: `session.items` on an
    `xdist` controller is empty, so 77 of 78 shipped receipts claimed an empty
    selection for an invocation that ran hundreds of tests.
    """
    process, receipts, junit = _run_pytest(
        directory=tmp_path, suite=SUITE, arguments=("-n", "2", "--dist", "loadfile")
    )

    assert process.returncode == 0, process.stdout
    canonical, workers = _records(receipts=receipts)
    assert canonical["selected_node_ids"], "the canonical selection is empty again"
    assert len(canonical["selected_node_ids"]) == 5
    assert sorted(canonical["worker_collections"]) == ["gw0", "gw1"]
    assert canonical["policy"]["numprocesses"] == 2
    assert canonical["policy"]["dist"] == "loadfile"
    assert canonical["inventory_complete"], canonical["incomplete_reasons"]
    assert set(workers) == {"gw0", "gw1"}
    assert reconcile_with_junit(receipt=canonical, junit_path=junit)["reconciled"]


def test_both_workers_collect_everything_and_execute_disjoint_shares(
    tmp_path: Path,
) -> None:
    """Identical collection, disjoint execution: two populations, not one."""
    process, receipts, _ = _run_pytest(
        directory=tmp_path, suite=SUITE, arguments=("-n", "2", "--dist", "load")
    )

    assert process.returncode == 0, process.stdout
    canonical, workers = _records(receipts=receipts)
    collections = canonical["worker_collections"]
    assert collections["gw0"] == collections["gw1"] == canonical["selected_node_ids"]
    executed = [set(record["executed_node_ids"]) for record in workers.values()]
    assert executed[0].isdisjoint(executed[1])
    assert set().union(*executed) == set(canonical["executed_node_ids"])


def test_every_participant_publishes_under_its_own_name(tmp_path: Path) -> None:
    """Three processes, three files: no participant overwrites another."""
    _, receipts, _ = _run_pytest(directory=tmp_path, suite=SUITE, arguments=("-n", "2"))

    names = sorted(path.name for path in receipts.glob("*.receipt.json"))
    assert len(names) == 3
    assert len(set(names)) == 3
    assert sum("worker-" in name for name in names) == 2


def test_the_controller_and_its_workers_share_one_invocation_id(
    tmp_path: Path,
) -> None:
    """Supporting records can be joined to the canonical one they belong to."""
    _, receipts, _ = _run_pytest(directory=tmp_path, suite=SUITE, arguments=("-n", "2"))

    canonical, workers = _records(receipts=receipts)
    assert {record["invocation_id"] for record in workers.values()} == {
        canonical["invocation_id"]
    }


def test_an_invocation_with_no_selected_tests_records_an_empty_selection(
    tmp_path: Path,
) -> None:
    """Nothing ran, nothing is claimed, and the deselected population says why."""
    process, receipts, junit = _run_pytest(
        directory=tmp_path, suite=SUITE, arguments=("-k", "nothing_matches_this")
    )

    assert process.returncode == 5, process.stdout
    canonical, _ = _records(receipts=receipts)
    assert canonical["selected_node_ids"] == []
    assert canonical["executed_node_ids"] == []
    assert len(canonical["deselected_node_ids"]) == 5
    assert canonical["exit_status"] == 5
    assert canonical["inventory_complete"], canonical["incomplete_reasons"]
    assert reconcile_with_junit(receipt=canonical, junit_path=junit)["reconciled"]


def test_a_partially_skipped_population_keeps_its_skips_out_of_the_selection(
    tmp_path: Path,
) -> None:
    """An expected skip was selected and executed; it is still recorded as skipped."""
    _, receipts, _ = _run_pytest(directory=tmp_path, suite=SUITE, arguments=())

    canonical, _ = _records(receipts=receipts)
    skipped = set(canonical["skipped_node_ids"])
    assert skipped == {"test_suite.py::test_third"}
    assert skipped <= set(canonical["selected_node_ids"])
    assert skipped <= set(canonical["executed_node_ids"])
    assert skipped.isdisjoint(canonical["deselected_node_ids"])
    assert canonical["outcomes"]["test_suite.py::test_third"] == "skipped"


def test_an_all_skipped_population_is_still_a_complete_inventory(
    tmp_path: Path,
) -> None:
    """Every test skipped is a result about the run, not a missing inventory."""
    process, receipts, junit = _run_pytest(
        directory=tmp_path, suite=ALL_SKIPPED_SUITE, arguments=()
    )

    assert process.returncode == 0, process.stdout
    canonical, _ = _records(receipts=receipts)
    assert len(canonical["selected_node_ids"]) == 2
    assert sorted(canonical["skipped_node_ids"]) == canonical["selected_node_ids"]
    assert set(canonical["outcomes"].values()) == {"skipped"}
    assert canonical["inventory_complete"], canonical["incomplete_reasons"]
    assert reconcile_with_junit(receipt=canonical, junit_path=junit)["reconciled"]


def test_a_canonical_record_carries_the_identity_of_what_produced_it(
    tmp_path: Path,
) -> None:
    """Two receipts can be told apart, and neither read as evidence about the other."""
    _, receipts, junit = _run_pytest(
        directory=tmp_path, suite=SUITE, arguments=("-n", "2")
    )

    canonical, _ = _records(receipts=receipts)
    assert canonical["environment"]["source_sha"] == "0" * 40
    assert canonical["environment"]["runner_os"] == "Linux"
    assert canonical["runtime"]["precision"] in {32, 64}
    assert canonical["runtime"]["device_count"] >= 1
    assert canonical["runtime"]["backend"]
    assert canonical["runtime"]["python_version"]
    assert canonical["policy"]["numprocesses"] == 2
    assert canonical["junit_path"] == str(junit)
    assert canonical["exit_status"] == 0
    assert canonical["start_wall_clock"] < canonical["end_wall_clock"]


def _collector(*, tmp_path: Path, **option: object) -> _ReceiptCollector:
    """Build a collector over a stand-in config, for the cases a run cannot stage."""
    settings = {
        "ci_policy": "full",
        "hardware_profile": "cpu",
        "policy_child": True,
        "numprocesses": None,
        "dist": "no",
        "markexpr": "",
        "keyword": "",
        **option,
    }
    config = cast("pytest.Config", SimpleNamespace(option=SimpleNamespace(**settings)))
    collector = _ReceiptCollector(receipt_root=tmp_path, config=config)
    collector.on_sessionstart()
    return collector


def _payload(*, collector: _ReceiptCollector) -> dict[str, Any]:
    """Assemble a record with the CI identity a real runner would supply."""
    os.environ["GITHUB_SHA"] = "0" * 40
    try:
        return collector.payload(exitstatus=0, junitxml_path="reports/junit.xml")
    finally:
        del os.environ["GITHUB_SHA"]


def test_a_worker_that_never_reported_is_not_a_complete_inventory(
    tmp_path: Path,
) -> None:
    """Two workers were asked for and one answered: the gap is named in the file."""
    collector = _collector(tmp_path=tmp_path, numprocesses=2, dist="load")
    collector.on_worker_collection(worker_id="gw0", ids=["test_suite.py::test_first"])

    payload = _payload(collector=collector)

    assert not payload["inventory_complete"]
    assert payload["incomplete_reasons"] == [
        "1 of 2 workers reported a collection: ['gw0']"
    ]


def test_no_worker_reporting_at_all_is_not_a_complete_inventory(
    tmp_path: Path,
) -> None:
    """A distributed invocation whose workers all died publishes no silent `[]`."""
    collector = _collector(tmp_path=tmp_path, numprocesses=2, dist="load")

    payload = _payload(collector=collector)

    assert payload["selected_node_ids"] == []
    assert not payload["inventory_complete"]
    assert payload["incomplete_reasons"] == ["no worker reported its collection"]


def test_workers_that_disagree_about_the_collection_are_refused(
    tmp_path: Path,
) -> None:
    """Disagreement publishes the union and says so; it never picks a winner."""
    collector = _collector(tmp_path=tmp_path, numprocesses=2, dist="load")
    collector.on_worker_collection(worker_id="gw0", ids=["a.py::one", "a.py::two"])
    collector.on_worker_collection(worker_id="gw1", ids=["a.py::one"])

    payload = _payload(collector=collector)

    assert payload["selected_node_ids"] == ["a.py::one", "a.py::two"]
    assert not payload["inventory_complete"]
    assert payload["incomplete_reasons"] == [
        "workers disagree about the collection: ['gw1']"
    ]


def test_a_late_worker_record_cannot_add_itself_to_a_published_inventory(
    tmp_path: Path,
) -> None:
    """The controller publishes what it had; a worker arriving later is still missing.

    The published record is the one the controller wrote, and it is marked
    incomplete. A worker record landing in the same directory afterwards does
    not silently repair it, because completeness is a field in the canonical
    file rather than a count of files in the directory.
    """
    collector = _collector(tmp_path=tmp_path, numprocesses=2, dist="load")
    collector.on_worker_collection(worker_id="gw0", ids=["a.py::one"])
    os.environ["GITHUB_SHA"] = "0" * 40
    try:
        collector.on_sessionfinish(exitstatus=0, junitxml_path="reports/junit.xml")
    finally:
        del os.environ["GITHUB_SHA"]
    (tmp_path / "junit.linux-fp64.worker-gw1.receipt.json").write_text(
        json.dumps({"role": WORKER_ROLE, "worker_id": "gw1"}), encoding="utf-8"
    )

    published = json.loads(
        next(tmp_path.glob("junit.*worker-gw1*")).read_text(encoding="utf-8")
    )
    canonical = json.loads(
        next(
            path
            for path in tmp_path.glob("junit.*.receipt.json")
            if "worker-" not in path.name
        ).read_text(encoding="utf-8")
    )

    assert published["worker_id"] == "gw1"
    assert not canonical["inventory_complete"]
    assert "1 of 2 workers" in canonical["incomplete_reasons"][0]


def test_an_executed_node_outside_the_selection_is_refused(tmp_path: Path) -> None:
    """An outcome for a node nobody selected means the inventory is not the run."""
    collector = _collector(tmp_path=tmp_path, numprocesses=2, dist="load")
    collector.on_worker_collection(worker_id="gw0", ids=["a.py::one"])
    collector.on_worker_collection(worker_id="gw1", ids=["a.py::one"])
    collector.on_logreport(report=_report(nodeid="a.py::two", when="call"))

    payload = _payload(collector=collector)

    assert not payload["inventory_complete"]
    assert payload["incomplete_reasons"] == [
        "executed node was never selected: a.py::two"
    ]


def _items(*, nodeid: str) -> Any:
    """Return the one-item collection the writer reads a `nodeid` off."""
    return [SimpleNamespace(nodeid=nodeid)]


def _report(*, nodeid: str, when: str, outcome: str = "passed") -> Any:
    """Return one phase report, the only part of a `TestReport` this writer reads."""
    return SimpleNamespace(
        nodeid=nodeid,
        when=when,
        outcome=outcome,
        failed=outcome == "failed",
        skipped=outcome == "skipped",
        passed=outcome == "passed",
    )


def test_three_phases_of_one_test_are_one_executed_test(tmp_path: Path) -> None:
    """Setup, call and teardown are phases of a test, not three tests."""
    collector = _collector(tmp_path=tmp_path)
    collector.on_collection_finish(items=_items(nodeid="a.py::one"))
    for when in ("setup", "call", "teardown"):
        collector.on_logreport(report=_report(nodeid="a.py::one", when=when))

    payload = _payload(collector=collector)

    assert payload["executed_node_ids"] == ["a.py::one"]
    assert payload["phase_counts"] == {"a.py::one": 3}
    assert payload["outcomes"] == {"a.py::one": "passed"}
    assert payload["inventory_complete"], payload["incomplete_reasons"]


def test_a_failure_in_any_phase_outranks_the_phases_that_passed(
    tmp_path: Path,
) -> None:
    """A test that errored in teardown did not pass because its call did."""
    collector = _collector(tmp_path=tmp_path)
    collector.on_collection_finish(items=_items(nodeid="a.py::one"))
    collector.on_logreport(report=_report(nodeid="a.py::one", when="call"))
    collector.on_logreport(
        report=_report(nodeid="a.py::one", when="teardown", outcome="failed")
    )

    payload = _payload(collector=collector)

    assert payload["outcomes"] == {"a.py::one": "failed"}


def test_a_skip_at_setup_is_not_hidden_by_its_passing_teardown(
    tmp_path: Path,
) -> None:
    """The skipped population survives the phase that reported after it."""
    collector = _collector(tmp_path=tmp_path)
    collector.on_collection_finish(items=_items(nodeid="a.py::one"))
    collector.on_logreport(
        report=_report(nodeid="a.py::one", when="setup", outcome="skipped")
    )
    collector.on_logreport(report=_report(nodeid="a.py::one", when="teardown"))

    payload = _payload(collector=collector)

    assert payload["skipped_node_ids"] == ["a.py::one"]
    assert payload["outcomes"] == {"a.py::one": "skipped"}


def test_reconciliation_counts_a_multiphase_junit_case_once(tmp_path: Path) -> None:
    """Two `<testcase>` elements for one test are one test on the JUnit side too."""
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for _ in range(2):
        ET.SubElement(suite, "testcase", classname="a", name="one")
    junit = tmp_path / "junit.xml"
    ET.ElementTree(root).write(junit)

    assert unique_junit_cases(path=junit) == {("a", "one")}
    assert reconcile_with_junit(
        receipt={"selected_node_ids": ["a.py::one"]}, junit_path=junit
    ) == {
        "selected_count": 1,
        "junit_case_count": 1,
        "selected_without_outcome": [],
        "outcome_without_selection": [],
        "reconciled": True,
    }


@pytest.mark.parametrize(
    ("nodeid", "expected"),
    [
        ("tests/ci/test_x.py::test_y", ("tests.ci.test_x", "test_y")),
        ("tests/ci/test_x.py::test_y[a-1]", ("tests.ci.test_x", "test_y[a-1]")),
        ("tests/ci/test_x.py::Klass::test_y", ("tests.ci.test_x.Klass", "test_y")),
    ],
)
def test_a_nodeid_maps_to_the_junit_name_pytest_writes_for_it(
    *, nodeid: str, expected: tuple[str, str]
) -> None:
    """The join between a recorded selection and a written outcome holds."""
    assert junit_identity(nodeid=nodeid) == expected


def test_the_same_junit_basename_on_two_lanes_publishes_two_receipts() -> None:
    """One artifact name per OS and precision is why 78 receipts were not 78 records."""
    names = {
        receipt_filename(
            junitxml_path="reports/junit-cpu-general-1.xml",
            worker_id=worker,
            runner_os=runner_os,
            precision=precision,
            fallback="unused",
        )
        for runner_os in ("Linux", "macOS", "Windows")
        for precision in (32, 64)
        for worker in (None, "gw0", "gw1")
    }

    assert len(names) == 18


def test_an_invocation_without_a_junit_file_names_itself() -> None:
    """Two receipt-less invocations in one directory still publish two records."""
    first = receipt_filename(
        junitxml_path=None,
        worker_id=None,
        runner_os="Linux",
        precision=64,
        fallback="aaaa",
    )
    second = receipt_filename(
        junitxml_path=None,
        worker_id=None,
        runner_os="Linux",
        precision=64,
        fallback="bbbb",
    )

    assert first != second
    assert first.startswith("invocation-aaaa")


def test_an_agreed_collection_is_reported_without_a_reason() -> None:
    """Identical worker collections are the invocation's selection, verbatim."""
    agreed, reasons = agreed_collection(
        collections={
            "gw0": ["a.py::two", "a.py::one"],
            "gw1": ["a.py::one", "a.py::two"],
        }
    )

    assert agreed == ["a.py::one", "a.py::two"]
    assert reasons == []
