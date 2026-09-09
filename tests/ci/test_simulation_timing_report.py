"""Timing reports require every witness and fail closed on lost execution."""

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from tests.ci.check_simulation_timing_report import check_report


@pytest.mark.parametrize(
    "defect",
    [
        "empty",
        "missing",
        "duplicate",
        "foreign",
        "skipped",
        "failure",
        "error",
        "suite_error",
        "suite_skipped",
        "suite_error_count",
    ],
)
def test_incomplete_or_unsuccessful_timing_report_is_refused(
    *, tmp_path: Path, defect: str
) -> None:
    """Refuse a missing witness, duplicate, unrelated case or unsuccessful case."""
    report = _report()
    suite = report.find("testsuite")
    assert suite is not None
    if defect == "empty":
        suite.clear()
    elif defect == "missing":
        suite.remove(suite[0])
    elif defect == "duplicate":
        suite[1].attrib.update(suite[0].attrib)
    elif defect == "foreign":
        suite[0].set("classname", "other.test_module")
    elif defect == "suite_error_count":
        suite.set("errors", "1")
    elif defect.startswith("suite_"):
        ET.SubElement(suite, defect.removeprefix("suite_"))
    else:
        ET.SubElement(suite[0], defect)
    path = tmp_path / "timing.xml"
    ET.ElementTree(report).write(path)
    with pytest.raises(ValueError, match="exactly four successful timing cases"):
        check_report(path=path)


def test_all_four_successful_timing_cases_are_accepted(tmp_path: Path) -> None:
    """Accept the two timing families with both model witnesses."""
    path = tmp_path / "timing.xml"
    ET.ElementTree(_report()).write(path)
    check_report(path=path)


def _report() -> ET.Element:
    """Build the expected JUnit case population independently of the checker."""
    report = ET.Element("testsuites")
    suite = ET.SubElement(report, "testsuite")
    for function in (
        "test_simulation_loop_host_time_at_progress_is_within_the_bar_of_off",
        "test_simulate_host_time_at_progress_is_within_the_bar_of_off",
    ):
        for witness in ("dissolution", "multi_regime"):
            ET.SubElement(
                suite,
                "testcase",
                classname="tests.simulation.test_compile_requests",
                name=f"{function}[{witness}]",
            )
    return report
