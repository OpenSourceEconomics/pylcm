"""Check the complete population of an isolated simulation timing report."""

import argparse
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from tests.ci.simulation_timings import UNSTABLE_HOST_MARKER

EXPECTED_CASES = Counter(
    (module, f"{function}[{witness}]")
    for module, function in (
        (
            "tests.simulation.test_compile_requests",
            "test_simulation_loop_host_time_at_progress_is_within_the_bar_of_off",
        ),
        (
            "tests.simulation.test_compile_requests",
            "test_simulate_host_time_at_progress_is_within_the_bar_of_off",
        ),
        (
            "tests.simulation.test_preflight_contract",
            "test_unstubbed_warm_full_call_progress_meets_existing_time_bar",
        ),
    )
    for witness in ("dissolution", "multi_regime")
)


def _declined_to_measure(case: ET.Element) -> bool:
    """Report whether this case skipped because the host was not steady enough.

    Any other skip is a lost row: the bar did not run and nothing said why, so
    it is refused exactly as a failure is.
    """
    return any(
        UNSTABLE_HOST_MARKER in (skip.get("message") or "")
        for skip in case.findall("skipped")
    )


def check_report(*, path: Path) -> None:
    """Require six timing cases, none failed and at least one actually measured."""
    # The workflow supplies the JUnit file written by its own pytest process.
    report = ET.parse(path).getroot()  # noqa: S314
    cases = tuple(report.iter("testcase"))
    actual = Counter((case.get("classname"), case.get("name")) for case in cases)
    declined = tuple(case for case in cases if _declined_to_measure(case))
    broken = any(
        node.tag in {"failure", "error"}
        or any(int(node.get(field, "0")) != 0 for field in ("failures", "errors"))
        for node in report.iter()
    )
    inside_cases = [skip for case in cases for skip in case.findall("skipped")]
    lost = len(inside_cases) != sum(1 for node in report.iter("skipped")) or any(
        UNSTABLE_HOST_MARKER not in (skip.get("message") or "") for skip in inside_cases
    )
    unaccounted = any(
        int(suite.get("skipped", "0")) > len(declined)
        for suite in report.iter("testsuite")
    )
    vacuous = len(declined) == len(EXPECTED_CASES)
    if actual != EXPECTED_CASES or broken or lost or unaccounted or vacuous:
        raise ValueError(
            f"Expected exactly six successful timing cases in {path}; "
            f"found {dict(actual)!r}, broken={broken}, lost={lost}, "
            f"unaccounted={unaccounted}, declined={len(declined)}."
        )


def main() -> None:
    """Validate the JUnit report named by the CPU workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    check_report(path=args.report)


if __name__ == "__main__":
    main()
