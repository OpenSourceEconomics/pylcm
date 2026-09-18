"""Every incomplete timing population is refused, not only the all-declined one.

The reviewer's 63 generated masks, adopted verbatim in substance: each mask is
one subset of the six required rows that declined to measure. Before the gate
required every row, only the single all-declined mask was refused, so 62 of
these populations were accepted with an unmeasured obligation in them.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from tests.ci.check_simulation_timing_report import check_report
from tests.ci.simulation_timings import UNSTABLE_HOST_MARKER

# The six required measurements, built here independently of the checker.
REQUIRED_ROWS = tuple(
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


@pytest.mark.parametrize("mask", range(1, 2 ** len(REQUIRED_ROWS)))
def test_no_missing_timing_obligation_is_accepted(*, tmp_path: Path, mask: int) -> None:
    """A report in which any required row declined to measure is unresolved."""
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        tests=str(len(REQUIRED_ROWS)),
        failures="0",
        errors="0",
        skipped=str(mask.bit_count()),
    )
    for index, (module, name) in enumerate(REQUIRED_ROWS):
        case = ET.SubElement(suite, "testcase", classname=module, name=name)
        if mask & (1 << index):
            ET.SubElement(
                case,
                "skipped",
                message=f"{UNSTABLE_HOST_MARKER}: synthetic control-only instability",
            )
    path = tmp_path / "timing.xml"
    ET.ElementTree(root).write(path)

    with pytest.raises(ValueError, match="exactly six successful timing cases"):
        check_report(path=path)
