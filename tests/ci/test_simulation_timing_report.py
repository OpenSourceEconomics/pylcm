"""Timing reports require every witness and fail closed on lost execution."""

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from tests.ci.check_simulation_timing_report import check_report
from tests.ci.simulation_timings import UNSTABLE_HOST_MARKER


def _drop_every_case(suite: ET.Element) -> None:
    """Leave a report with no timing case at all."""
    suite.clear()


def _drop_one_case(suite: ET.Element) -> None:
    """Leave one of the six expected timing rows out of the report."""
    suite.remove(suite[0])


def _drop_the_preflight_family(suite: ET.Element) -> None:
    """Leave only the four cases the report carried before the preflight row."""
    for case in tuple(suite):
        if case.get("classname") == "tests.simulation.test_preflight_contract":
            suite.remove(case)


def _duplicate_a_case(suite: ET.Element) -> None:
    """Report one row twice and another not at all."""
    suite[1].attrib.update(suite[0].attrib)


def _make_a_case_foreign(suite: ET.Element) -> None:
    """Attribute one row to a module the timing lane does not run."""
    suite[0].set("classname", "other.test_module")


def _skip_a_case_without_a_reason(suite: ET.Element) -> None:
    """Skip a row with nothing recorded about why it did not measure."""
    ET.SubElement(suite[0], "skipped")


def _skip_a_case_for_an_unrecognised_reason(suite: ET.Element) -> None:
    """Skip a row for a reason that is not the steadiness precondition."""
    ET.SubElement(suite[0], "skipped", message="the runner felt slow")


def _decline_every_case(suite: ET.Element) -> None:
    """Have every row decline to measure, leaving the bar untested."""
    for case in suite:
        ET.SubElement(case, "skipped", message=f"{UNSTABLE_HOST_MARKER}: 0.4")
    suite.set("skipped", str(len(suite)))


def _fail_a_case(suite: ET.Element) -> None:
    """Fail one row."""
    ET.SubElement(suite[0], "failure")


def _error_a_case(suite: ET.Element) -> None:
    """Error one row."""
    ET.SubElement(suite[0], "error")


def _error_the_suite(suite: ET.Element) -> None:
    """Record a suite-level error outside any case."""
    ET.SubElement(suite, "error")


def _skip_the_suite(suite: ET.Element) -> None:
    """Record a suite-level skip outside any case."""
    ET.SubElement(suite, "skipped")


def _count_a_suite_error(suite: ET.Element) -> None:
    """Announce an error in the suite tally that no case element carries."""
    suite.set("errors", "1")


_DEFECTS = {
    "empty": _drop_every_case,
    "missing": _drop_one_case,
    "legacy_four_only": _drop_the_preflight_family,
    "duplicate": _duplicate_a_case,
    "foreign": _make_a_case_foreign,
    "skipped": _skip_a_case_without_a_reason,
    "unmarked_skip_message": _skip_a_case_for_an_unrecognised_reason,
    "every_case_skipped": _decline_every_case,
    "failure": _fail_a_case,
    "error": _error_a_case,
    "suite_error": _error_the_suite,
    "suite_skipped": _skip_the_suite,
    "suite_error_count": _count_a_suite_error,
}


@pytest.mark.parametrize("defect", sorted(_DEFECTS))
def test_incomplete_or_unsuccessful_timing_report_is_refused(
    *, tmp_path: Path, defect: str
) -> None:
    """Refuse a missing witness, duplicate, unrelated case or unsuccessful case."""
    report = _report()
    suite = report.find("testsuite")
    assert suite is not None
    _DEFECTS[defect](suite)
    path = tmp_path / "timing.xml"
    ET.ElementTree(report).write(path)
    with pytest.raises(ValueError, match="exactly six successful timing cases"):
        check_report(path=path)


def test_all_six_successful_timing_cases_are_accepted(tmp_path: Path) -> None:
    """Accept the three timing families with both model witnesses."""
    path = tmp_path / "timing.xml"
    ET.ElementTree(_report()).write(path)
    check_report(path=path)


def test_a_case_skipped_because_the_host_was_not_steady_is_accepted(
    tmp_path: Path,
) -> None:
    """One row that declined to measure does not condemn the run.

    The bar itself is untouched: a batch taken while the control leg was not
    delivering steady wall time is no measurement of the code, so the row says
    so in its skip reason instead of reporting a ratio as a verdict. The other
    five rows still carry the bar, which is why an all-skipped report is
    refused above.
    """
    report = _report()
    suite = report.find("testsuite")
    assert suite is not None
    ET.SubElement(
        suite[0], "skipped", message=f"{UNSTABLE_HOST_MARKER}: relative IQR 0.204"
    )
    suite.set("skipped", "1")
    path = tmp_path / "timing.xml"
    ET.ElementTree(report).write(path)

    check_report(path=path)


def test_a_suite_skip_count_above_the_marked_cases_is_refused(tmp_path: Path) -> None:
    """A skip the case elements do not account for is an unexplained lost row."""
    report = _report()
    suite = report.find("testsuite")
    assert suite is not None
    ET.SubElement(
        suite[0], "skipped", message=f"{UNSTABLE_HOST_MARKER}: relative IQR 0.204"
    )
    suite.set("skipped", "2")
    path = tmp_path / "timing.xml"
    ET.ElementTree(report).write(path)

    with pytest.raises(ValueError, match="exactly six successful timing cases"):
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
    for witness in ("dissolution", "multi_regime"):
        ET.SubElement(
            suite,
            "testcase",
            classname="tests.simulation.test_preflight_contract",
            name=(
                "test_unstubbed_warm_full_call_progress_meets_existing_time_bar"
                f"[{witness}]"
            ),
        )
    return report
