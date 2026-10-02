"""Accept only complete, matched receipt comparisons at the command-line boundary."""

# ruff: noqa: INP001, S101

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_COMPARATOR = Path(__file__).with_name("compare_stage5b.py")
_LABELS = ("combined_cold", "combined_warm", "solve_warm", "split_simulate_warm")


def _record(arm: str) -> dict[str, Any]:
    """Build a completed receipt with the fields the arm driver publishes."""
    calls = []
    for label in _LABELS:
        call = {
            "label": label,
            "value_sha256": {"0/work": "a" * 64, "1/dead": "b" * 64},
            "wall_seconds": 1.0,
            "memory_stats_after": [{"peak_bytes_in_use": 1024}],
            "host_maxrss_kib": 2048,
        }
        if label != "solve_warm":
            call.update(panel_sha256="c" * 64, raw_sha256="d" * 64)
        calls.append(call)
    return {
        "arm": arm,
        "pylcm_sha": "1" * 40,
        "pylcm_dirty": "",
        "lcm_file": "/snapshot/src/lcm/__init__.py",
        "jax": "0.11.1",
        "x64": True,
        "devices": ["cuda:0 NVIDIA A100-SXM4-80GB"],
        "model": {
            "builder": "aca_model.benchmark.create_model",
            "grid_config": "reduced3",
            "execution_config": arm,
            "initial_conditions": "n_subjects=4096,seed=0",
            "pref_types": 3,
            "substituted_pref_type_params": {"discount_factor_by_type": [0.9] * 3},
        },
        "gpu_exclusivity": {
            "exclusive": True,
            "foreign_at_start": {},
            "foreign_seen_during_run": {},
        },
        "calls": calls,
    }


def _run_comparison(
    *, tmp_path: Path, records: list[dict[str, Any]]
) -> subprocess.CompletedProcess[str]:
    """Run the same receipt comparison command as the native job's intake."""
    directories = []
    for index, record in enumerate(records):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps(record))
        directories.append(str(directory))
    pixi_path = shutil.which("pixi")
    if pixi_path is None:
        raise RuntimeError("pixi is required for the comparator CLI check")
    return subprocess.run(  # noqa: S603
        [
            pixi_path,
            "run",
            "--as-is",
            "--manifest-path",
            os.environ["STAGE5B_TEST_MANIFEST"],
            "-e",
            "tests-cpu",
            "python",
            str(_COMPARATOR),
            *directories,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_missing_schedule_arm_is_refused(tmp_path: Path) -> None:
    """An unblocked receipt alone cannot establish schedule parity."""
    result = _run_comparison(tmp_path=tmp_path, records=[_record("unblocked")])

    assert (result.returncode, "exactly three distinct arms" in result.stderr) == (
        1,
        True,
    )


@pytest.mark.parametrize(
    "case", ["duplicate_arm", "duplicate_call", "missing_call", "unknown_call"]
)
def test_receipt_populations_are_complete_and_unique(
    *, tmp_path: Path, case: str
) -> None:
    """Every schedule and call label must occur exactly once."""
    records = [_record(arm) for arm in ("unblocked", "period_major", "block_major")]
    message = "four distinct calls"
    if case == "duplicate_arm":
        records.append(_record("block_major"))
        message = "exactly three distinct arms"
    elif case == "duplicate_call":
        records[2]["calls"].append(records[2]["calls"][0])
    elif case == "missing_call":
        records[2]["calls"].pop()
    else:
        records[2]["calls"].append({"label": "unexpected"})
    result = _run_comparison(tmp_path=tmp_path, records=records)

    assert (result.returncode, message in result.stderr) == (1, True)


def test_complete_matched_receipts_are_accepted(tmp_path: Path) -> None:
    """Equal schedule outputs pass even when the unblocked control differs."""
    records = [_record(arm) for arm in ("unblocked", "period_major", "block_major")]
    records[0]["calls"][0]["value_sha256"]["0/work"] = "f" * 64
    result = _run_comparison(tmp_path=tmp_path, records=records)

    assert (result.returncode, "block_major==period_major True" in result.stdout) == (
        0,
        True,
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing_value",
        "empty_values",
        "incomplete_values",
        "missing_panel",
        "missing_raw",
        "invalid_value_hash",
        "invalid_panel_hash",
        "invalid_raw_hash",
    ],
)
def test_required_digest_evidence_is_complete(*, tmp_path: Path, case: str) -> None:
    """Every comparison supplies valid digests for the same complete value domain."""
    records = [_record(arm) for arm in ("unblocked", "period_major", "block_major")]
    message = "digest"
    if case == "missing_value":
        del records[2]["calls"][0]["value_sha256"]
    elif case == "empty_values":
        for record in records:
            for call in record["calls"]:
                call["value_sha256"] = {}
    elif case == "incomplete_values":
        for record in records[1:]:
            del record["calls"][0]["value_sha256"]["1/dead"]
        message = "value coverage"
    elif case.startswith("missing_"):
        del records[2]["calls"][0][case.removeprefix("missing_") + "_sha256"]
    else:
        key = case.removeprefix("invalid_").removesuffix("_hash")
        for record in records[1:]:
            if key == "value":
                record["calls"][0]["value_sha256"]["0/work"] = "g" * 64
            else:
                record["calls"][0][key + "_sha256"] = "g" * 64
    result = _run_comparison(tmp_path=tmp_path, records=records)

    assert (result.returncode, message in result.stderr) == (1, True)


@pytest.mark.parametrize("key", ["value_sha256", "panel_sha256", "raw_sha256"])
def test_a_schedule_checksum_difference_is_refused(*, tmp_path: Path, key: str) -> None:
    """One differing published checksum prevents schedule parity acceptance."""
    records = [_record(arm) for arm in ("unblocked", "period_major", "block_major")]
    if key == "value_sha256":
        records[2]["calls"][0][key]["0/work"] = "e" * 64
    else:
        records[2]["calls"][0][key] = "e" * 64
    result = _run_comparison(tmp_path=tmp_path, records=records)

    assert (result.returncode, "block_major==period_major False" in result.stdout) == (
        1,
        True,
    )


@pytest.mark.parametrize(
    "field",
    [
        "pylcm_sha",
        "lcm_file",
        "jax",
        "x64",
        "devices",
        "model",
        "pylcm_dirty",
        "gpu_exclusivity",
    ],
)
def test_required_provenance_cannot_be_absent(*, tmp_path: Path, field: str) -> None:
    """Available driver provenance is mandatory for every arm."""
    records = [_record(arm) for arm in ("unblocked", "period_major", "block_major")]
    del records[2][field]
    result = _run_comparison(tmp_path=tmp_path, records=records)

    assert (result.returncode, "provenance" in result.stderr) == (1, True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pylcm_sha", "2" * 40),
        ("lcm_file", "/foreign/src/lcm/__init__.py"),
        ("jax", "0.12.0"),
        ("x64", False),
        ("devices", ["cuda:1 NVIDIA A100-SXM4-80GB"]),
        ("model", {"builder": "other"}),
    ],
)
def test_available_provenance_must_match(
    *, tmp_path: Path, field: str, value: object
) -> None:
    """Different source, environment or economic model cannot be compared."""
    records = [_record(arm) for arm in ("unblocked", "period_major", "block_major")]
    records[2][field] = value
    result = _run_comparison(tmp_path=tmp_path, records=records)

    assert (result.returncode, "provenance" in result.stderr) == (1, True)


@pytest.mark.parametrize(
    "case",
    [
        "dirty",
        "nonexclusive",
        "foreign_at_start",
        "foreign_during_run",
        "construction_error",
    ],
)
def test_unsuccessful_or_shared_arm_is_refused(*, tmp_path: Path, case: str) -> None:
    """Dirty, shared or refused arm outcomes cannot establish acceptance."""
    records = [_record(arm) for arm in ("unblocked", "period_major", "block_major")]
    if case == "dirty":
        records[2]["pylcm_dirty"] = " M src/lcm/model.py"
    elif case == "nonexclusive":
        records[2]["gpu_exclusivity"]["exclusive"] = False
    elif case == "foreign_at_start":
        records[2]["gpu_exclusivity"]["foreign_at_start"] = {"1": "other process"}
    elif case == "foreign_during_run":
        records[2]["gpu_exclusivity"]["foreign_seen_during_run"] = {
            "1": "other process"
        }
    else:
        records[2]["construction_error"] = "ExecutionPlanningError: unsupported regime"
    result = _run_comparison(tmp_path=tmp_path, records=records)

    assert (result.returncode, "provenance" in result.stderr) == (1, True)


@pytest.mark.parametrize("x64", [False, True])
def test_either_matched_precision_is_accepted(*, tmp_path: Path, x64: bool) -> None:
    """Both driver precision flags permit a complete matched comparison."""
    records = [_record(arm) for arm in ("unblocked", "period_major", "block_major")]
    for record in records:
        record["x64"] = x64
    result = _run_comparison(tmp_path=tmp_path, records=records)

    assert result.returncode == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pylcm_sha", "invalid"),
        ("lcm_file", ""),
        ("jax", ""),
        ("x64", "true"),
        ("devices", []),
        ("model", {"execution_config": "block_major"}),
    ],
)
def test_matching_empty_or_malformed_provenance_is_refused(
    *, tmp_path: Path, field: str, value: object
) -> None:
    """Identical malformed provenance is not evidence of a matched experiment."""
    records = [_record(arm) for arm in ("unblocked", "period_major", "block_major")]
    for record in records:
        record[field] = value
    result = _run_comparison(tmp_path=tmp_path, records=records)

    assert (result.returncode, "provenance" in result.stderr) == (1, True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("builder", None),
        ("grid_config", ""),
        ("initial_conditions", None),
        ("pref_types", 0),
        ("pref_types", True),
    ],
)
def test_economic_description_values_are_populated(
    *, tmp_path: Path, field: str, value: object
) -> None:
    """Matched empty descriptions cannot identify a completed economic model."""
    records = [_record(arm) for arm in ("unblocked", "period_major", "block_major")]
    for record in records:
        record["model"][field] = value
    result = _run_comparison(tmp_path=tmp_path, records=records)

    assert (result.returncode, "provenance" in result.stderr) == (1, True)
