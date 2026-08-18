"""Run the exact-kernel capability contract at one floating-point precision."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

_SMOKE_NODE = (
    "tests/test_exact_kernel_capability_contract.py::"
    "test_exact_kernel_answers_in_the_active_precision"
)
_CONTRACT_FILE = "tests/test_dcegm_validation.py"
_EXPECTED_INVENTORY = Path("tests/ci/expected-exact-kernel-skips-windows.json")


def main() -> int:
    """Run present, absent, and present-but-broken arms for one precision."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("32", "64"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"exact-kernel-matrix-fp{args.precision}"

    present_xml = args.output_dir / f"junit-{prefix}-present.xml"
    present = _run_pytest(
        name="present",
        precision=args.precision,
        pytest_args=[_CONTRACT_FILE, _SMOKE_NODE, "-m", "not slow"],
        junit_path=present_xml,
        log_path=args.output_dir / f"{prefix}-present.log",
    )
    _require(
        condition=present.returncode == 0,
        message=f"present arm failed with status {present.returncode}",
    )
    present_counts = _junit_counts(present_xml)
    _require(
        condition=present_counts["failed"] == 0,
        message="present arm reported a failure",
    )
    _require(
        condition=present_counts["skipped"] == 0,
        message="present arm skipped a test",
    )
    _require(
        condition=present_counts["tests"] >= 29,
        message="present arm did not execute the semantic-contract surface",
    )

    absent_xml = args.output_dir / f"junit-{prefix}-absent.xml"
    actual_inventory = args.output_dir / f"exact-kernel-skips-fp{args.precision}.json"
    absent = _run_pytest(
        name="absent",
        precision=args.precision,
        pytest_args=[
            _CONTRACT_FILE,
            _SMOKE_NODE,
            "-m",
            "not slow",
            "-p",
            "tests.ci.kernel_absent",
            f"--exact-kernel-skip-inventory={actual_inventory}",
            f"--expected-exact-kernel-skip-inventory={_EXPECTED_INVENTORY}",
            "--max-total-skips=1",
        ],
        junit_path=absent_xml,
        log_path=args.output_dir / f"{prefix}-absent.log",
    )
    _require(
        condition=absent.returncode == 0,
        message=f"absent arm failed with status {absent.returncode}",
    )
    absent_counts = _junit_counts(absent_xml)
    _require(
        condition=absent_counts["failed"] == 0,
        message="absent arm reported a failure",
    )
    _require(
        condition=absent_counts["skipped"] == 1,
        message="absent arm did not skip exactly once",
    )
    _require(
        condition=absent_counts["tests"] == present_counts["tests"],
        message="kernel absence changed collection rather than one declared outcome",
    )
    _require(
        condition=(
            _inventory_records(actual_inventory)
            == _inventory_records(_EXPECTED_INVENTORY)
        ),
        message="absent arm wrote an unexpected exact-kernel skip inventory",
    )

    broken_xml = args.output_dir / f"junit-{prefix}-broken.xml"
    broken = _run_pytest(
        name="broken",
        precision=args.precision,
        pytest_args=[_SMOKE_NODE, "-p", "tests.ci.kernel_broken"],
        junit_path=broken_xml,
        log_path=args.output_dir / f"{prefix}-broken.log",
    )
    _require(
        condition=broken.returncode != 0,
        message="present-but-broken arm unexpectedly passed",
    )
    broken_counts = _junit_counts(broken_xml)
    _require(
        condition=broken_counts["tests"] == 1,
        message="broken arm collected an unexpected test set",
    )
    _require(
        condition=broken_counts["skipped"] == 0,
        message="broken kernel was converted into a skip",
    )
    _require(
        condition=broken_counts["failed"] == 1,
        message="broken kernel did not fail exactly once",
    )
    _require(
        condition="ExactAffineKernelUnavailableError" in (broken.stdout or ""),
        message="broken arm failed without the exact-kernel capability error",
    )

    sys.stdout.write(
        f"exact-kernel capability matrix passed at fp{args.precision}: "
        f"present={present_counts}, absent={absent_counts}, "
        f"broken={broken_counts}\n"
    )
    return 0


def _run_pytest(
    *,
    name: str,
    precision: str,
    pytest_args: list[str],
    junit_path: Path,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Run one isolated pytest process and retain its complete transcript."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        *pytest_args,
        f"--precision={precision}",
        "-n0",
        "--tb=short",
        f"--junitxml={junit_path}",
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text("$ " + " ".join(command) + "\n\n" + completed.stdout)
    if completed.returncode != 0:
        sys.stdout.write(
            f"{name} arm returned {completed.returncode}; see {log_path}\n"
        )
    return completed


def _junit_counts(path: Path) -> dict[str, int]:
    """Return test, skip, and hard-failure counts from one JUnit report."""
    root = ElementTree.parse(path).getroot()  # noqa: S314
    cases = root.findall(".//testcase")
    skipped = sum(case.find("skipped") is not None for case in cases)
    failed = sum(
        case.find("failure") is not None or case.find("error") is not None
        for case in cases
    )
    return {"tests": len(cases), "skipped": skipped, "failed": failed}


def _inventory_records(path: Path) -> list[dict[str, str]]:
    """Read the exact node-id/reason records from one inventory."""
    payload = json.loads(path.read_text())
    records = payload.get("skipped") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        msg = f"Malformed exact-kernel inventory: {path}"
        raise RuntimeError(msg)
    return records


def _require(*, condition: bool, message: str) -> None:
    """Raise a matrix failure with a direct acceptance-condition message."""
    if not condition:
        raise RuntimeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
