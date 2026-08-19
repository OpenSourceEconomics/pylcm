"""Run the exact-kernel capability contract at one floating-point precision."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

_SMOKE_NODE = (
    "tests/test_exact_kernel_capability_contract.py::"
    "test_exact_kernel_answers_in_the_active_precision"
)
_CONTRACT_FILE = "tests/test_dcegm_validation.py"
# A file whose subject is the certified kernel itself. The contract file above
# selects portable envelopes deliberately, so on its own the absent arm proves
# only that one smoke node skips: every other collected node is indifferent to
# the kernel. This file is where absence has to show up as declared skips.
_KERNEL_SURFACE = "tests/solution/test_envelope_cell_workspace.py"
# `tests/solution/` is marked `slow` wholesale so a small runner can deselect
# it, and every exact-kernel test lives there. A bare `not slow` therefore
# deselects the entire kernel surface and the absent arm passes having observed
# nothing — so the kernel-marked nodes are added back explicitly. Only the paths
# named above are collected, so this admits no heavy solve beyond them.
_SELECTION = "not slow or requires_exact_affine_kernel"

# The collected size is pinned rather than floored: the point of the arm is to
# notice a surface that silently stopped being collected, and a floor with slack
# tolerates exactly that. Adding or removing a test in the files above changes
# this number, deliberately.
_EXPECTED_COLLECTED = 46
# Every path here is anchored to the repository, never to the caller's working
# directory: pytest runs with its own `cwd`, so a relative path would be read by
# the parent and written by the child in two different places.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_INVENTORY = _REPO_ROOT / "tests/ci/expected-exact-kernel-skips.json"


def main() -> int:
    """Run present, absent, and present-but-broken arms for one precision."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("32", "64"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    args.output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else _REPO_ROOT / args.output_dir
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"exact-kernel-matrix-fp{args.precision}"

    present_xml = args.output_dir / f"junit-{prefix}-present.xml"
    present = _run_pytest(
        name="present",
        precision=args.precision,
        pytest_args=[_CONTRACT_FILE, _KERNEL_SURFACE, _SMOKE_NODE, "-m", _SELECTION],
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
        condition=present_counts["tests"] == _EXPECTED_COLLECTED,
        message=(
            f"present arm collected {present_counts['tests']} tests, expected "
            f"{_EXPECTED_COLLECTED}"
        ),
    )

    absent_xml = args.output_dir / f"junit-{prefix}-absent.xml"
    actual_inventory = args.output_dir / f"exact-kernel-skips-fp{args.precision}.json"
    absent = _run_pytest(
        name="absent",
        precision=args.precision,
        pytest_args=[
            _CONTRACT_FILE,
            _KERNEL_SURFACE,
            _SMOKE_NODE,
            "-m",
            _SELECTION,
            "-p",
            "tests.ci.kernel_absent",
            f"--exact-kernel-skip-inventory={actual_inventory}",
            f"--expected-exact-kernel-skip-inventory={_EXPECTED_INVENTORY}",
            f"--max-total-skips={len(_inventory_records(_EXPECTED_INVENTORY))}",
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
        condition=(
            absent_counts["skipped"] == len(_inventory_records(_EXPECTED_INVENTORY))
        ),
        message=(
            f"absent arm skipped {absent_counts['skipped']} nodes, expected "
            f"{len(_inventory_records(_EXPECTED_INVENTORY))}"
        ),
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
    broken_log = args.output_dir / f"{prefix}-broken.log"
    broken = _run_pytest(
        name="broken",
        precision=args.precision,
        pytest_args=[_SMOKE_NODE, "-p", "tests.ci.kernel_broken"],
        junit_path=broken_xml,
        log_path=broken_log,
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
        condition="ExactAffineKernelUnavailableError" in broken_log.read_text(),
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
        # `-v` names each test as it starts, so a run killed before it can write
        # its JUnit report still leaves an attributable record; `--tb=short`
        # keeps the failing ones readable in that same transcript.
        "-v",
        "--tb=short",
        f"--junitxml={junit_path}",
    ]
    # The transcript is opened before the subprocess starts and handed to it
    # directly, so a run the operating system kills leaves everything it had
    # already printed. Buffering it in this process would lose exactly the
    # evidence a killed run is diagnosed from.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as transcript:
        transcript.write("$ " + " ".join(command) + "\n\n")
        transcript.flush()
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            cwd=_REPO_ROOT,
            text=True,
            stdout=transcript,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        sys.stdout.write(
            f"{name} arm returned {completed.returncode}; see {log_path}\n"
        )
    return completed


def _junit_counts(path: Path) -> dict[str, int]:
    """Return test, skip, and hard-failure counts from one JUnit report."""
    root = ET.parse(path).getroot()  # noqa: S314
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
        raise TypeError(msg)
    return records


def _require(*, condition: bool, message: str) -> None:
    """Raise a matrix failure with a direct acceptance-condition message."""
    if not condition:
        raise RuntimeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
