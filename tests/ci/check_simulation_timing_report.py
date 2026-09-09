"""Check the complete population of an isolated simulation timing report."""

import argparse
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


def check_report(*, path: Path) -> None:
    """Require exactly four successful simulation timing cases in JUnit."""
    # The workflow supplies the JUnit file written by its own pytest process.
    report = ET.parse(path).getroot()  # noqa: S314
    expected = Counter(
        ("tests.simulation.test_compile_requests", f"{function}[{witness}]")
        for function in (
            "test_simulation_loop_host_time_at_progress_is_within_the_bar_of_off",
            "test_simulate_host_time_at_progress_is_within_the_bar_of_off",
        )
        for witness in ("dissolution", "multi_regime")
    )
    actual = Counter(
        (case.get("classname"), case.get("name")) for case in report.iter("testcase")
    )
    unsuccessful = any(
        node.tag in {"failure", "error", "skipped"}
        or any(
            int(node.get(field, "0")) != 0
            for field in ("failures", "errors", "skipped")
        )
        for node in report.iter()
    )
    if actual != expected or unsuccessful:
        raise ValueError(
            f"Expected exactly four successful timing cases in {path}; "
            f"found {dict(actual)!r}, unsuccessful={unsuccessful}."
        )


def main() -> None:
    """Validate the JUnit report named by the CPU workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    check_report(path=args.report)


if __name__ == "__main__":
    main()
