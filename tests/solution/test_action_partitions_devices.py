"""The action-partitioned GridSearch route on eight host devices.

A fresh interpreter with eight forced host CPU devices solves the many-actions
model on the ordinary route and on the action-partitioned route — actions
alone, actions times a sharded continuous state, and either combined with
type-local invariant blocks — and writes a report (see
`_action_partition_report.py`). At the same action width every published value
equals the ordinary route bit for bit, for every partition count; at
independently planned widths it agrees to the last few units in the last place
and the simulated panel is unchanged; the devices exchange one accumulator per
state cell and partition, never values over actions.

CPU proves semantics and placement, not GPU performance.
"""

import json
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from tests import conftest
from tests.solution._action_partition_report import _bitwise_mismatches


class _Layout(TypedDict):
    n_devices: int
    mesh: dict[str, int] | None
    spec: list[str] | None


class _FixedPartition(TypedDict):
    mismatches: list[str]
    working_layout: _Layout
    dead_layout: _Layout


class _PlannedPartition(TypedDict):
    max_ulp: float


class _StatePartition(TypedDict):
    mismatches_to_same_state_mesh: list[str]
    max_ulp_to_one_device: float
    working_layout: _Layout


class _InvariantPartition(TypedDict):
    mismatches: list[str]
    max_ulp: float
    working_layout: _Layout


class _Simulation(TypedDict):
    decisions_and_states_equal: bool
    value_max_ulp: float


class _ChangedParams(TypedDict):
    mismatches: list[str]
    differs_from_first: bool


class _Program(TypedDict):
    all_gather_shapes: list[list[int]]
    temp_bytes: int


class _Budget(TypedDict):
    module_file: str | None
    ceiling_bytes: int
    gathered_bytes: int
    refused_below_ceiling: bool
    mismatches_at_ceiling: list[str]
    terminal_values: list[float]


class _Release(TypedDict):
    codes: list[int]
    live_blocks: list[int]


class _BlockMajor(TypedDict):
    value_mismatches: list[str]
    panel_mismatches: dict[str, list[str]]
    release: _Release


class _Report(TypedDict):
    device_count: int
    backend: str
    x64: bool
    n_actions: int
    action_only_fixed_2: _FixedPartition
    action_only_fixed_3: _FixedPartition
    action_only_fixed_4: _FixedPartition
    action_only_fixed_8: _FixedPartition
    action_only_planned_2: _PlannedPartition
    action_only_planned_4: _PlannedPartition
    state_by_action_2: _StatePartition
    state_by_action_4: _StatePartition
    state_by_action_8: _StatePartition
    invariant_blocks_action_only: _InvariantPartition
    invariant_blocks_state_by_action: _InvariantPartition
    simulation: dict[str, _Simulation]
    changed_params: _ChangedParams
    programs: dict[str, _Program]
    budget: _Budget
    block_major: dict[str, _BlockMajor]


def _action_only_fixed(*, report: _Report, partitions: int) -> _FixedPartition:
    return {
        2: report["action_only_fixed_2"],
        3: report["action_only_fixed_3"],
        4: report["action_only_fixed_4"],
        8: report["action_only_fixed_8"],
    }[partitions]


def _action_only_planned(*, report: _Report, partitions: int) -> _PlannedPartition:
    return {
        2: report["action_only_planned_2"],
        4: report["action_only_planned_4"],
    }[partitions]


def _state_by_action(*, report: _Report, partitions: int) -> _StatePartition:
    return {
        2: report["state_by_action_2"],
        4: report["state_by_action_4"],
        8: report["state_by_action_8"],
    }[partitions]


def _invariant_blocks(*, report: _Report, layout: str) -> _InvariantPartition:
    return {
        "action_only": report["invariant_blocks_action_only"],
        "state_by_action": report["invariant_blocks_state_by_action"],
    }[layout]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARTITION_AXIS = "_lcm_action_partition"

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("shape", "mismatches"), [((1, 4), []), ((2, 2), ["0/working"])]
)
def test_complete_value_comparison_distinguishes_shapes_with_the_same_bytes(
    *, shape: tuple[int, int], mismatches: list[str]
) -> None:
    """Identical data with a different shape is a different labelled value."""
    values = np.asarray([1.0, 2.0, 3.0, 4.0])

    assert (
        _bitwise_mismatches(
            got={0: {"working": values.reshape(shape).copy()}},
            expected={0: {"working": values.reshape((1, 4))}},
        )
        == mismatches
    )


@cache
def _report_for(*, x64: bool, directory: Path) -> _Report:
    out = directory / f"report_x64_{int(x64)}.json"
    env = {
        **os.environ,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": os.pathsep.join((str(_REPO_ROOT / "src"), str(_REPO_ROOT))),
    }
    pixi = shutil.which("pixi")
    assert pixi is not None
    result = subprocess.run(  # noqa: S603
        [
            pixi,
            "run",
            "-e",
            "tests-cpu",
            "python",
            "-m",
            "tests.solution._action_partition_report",
            "--x64",
            str(int(x64)),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        timeout=1800,
    )
    if result.returncode != 0 or not out.exists():
        pytest.fail(result.stderr[-6000:])
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> _Report:
    return _report_for(
        x64=conftest.X64_ENABLED,
        directory=tmp_path_factory.mktemp("action_partitions"),
    )


def test_the_report_ran_on_eight_host_devices_at_the_suite_precision(
    report: _Report,
) -> None:
    assert (report["device_count"], report["backend"], report["x64"]) == (
        8,
        "cpu",
        conftest.X64_ENABLED,
    )


@pytest.mark.parametrize("partitions", [2, 3, 4, 8])
def test_action_only_partitions_equal_the_ordinary_route_bitwise(
    *, report: _Report, partitions: int
) -> None:
    """Every period and regime, at the same fixed action width."""
    assert _action_only_fixed(report=report, partitions=partitions)["mismatches"] == []


@pytest.mark.parametrize("partitions", [2, 3, 4, 8])
def test_an_action_only_regime_is_replicated_over_its_action_group(
    *, report: _Report, partitions: int
) -> None:
    assert _action_only_fixed(report=report, partitions=partitions)[
        "working_layout"
    ] == {
        "n_devices": partitions,
        "mesh": {_PARTITION_AXIS: partitions},
        "spec": ["None"],
    }


def test_a_regime_without_a_request_stays_on_one_device(
    report: _Report,
) -> None:
    assert report["action_only_fixed_4"]["dead_layout"] == {
        "n_devices": 1,
        "mesh": None,
        "spec": None,
    }


@pytest.mark.parametrize("partitions", [2, 4])
def test_independently_planned_widths_agree_to_the_last_units_in_the_last_place(
    *, report: _Report, partitions: int
) -> None:
    """The planner narrows the action block for more devices, which changes
    the compiled vectorization of `Q`, not the reduction."""
    assert _action_only_planned(report=report, partitions=partitions)["max_ulp"] <= (
        conftest.INVARIANCE_EPS_MULTIPLE
    )


@pytest.mark.parametrize("widths", ["planned", "fixed"])
def test_simulation_from_a_partitioned_solve_takes_the_ordinary_decisions(
    *, report: _Report, widths: str
) -> None:
    """Every simulated action, state and regime equals the ordinary panel."""
    assert report["simulation"][widths]["decisions_and_states_equal"] is True


def test_simulated_values_at_the_same_width_equal_the_ordinary_panel(
    report: _Report,
) -> None:
    assert report["simulation"]["fixed"]["value_max_ulp"] == 0.0


def test_simulated_values_at_planned_widths_agree_to_the_last_units_in_the_last_place(
    report: _Report,
) -> None:
    assert (
        report["simulation"]["planned"]["value_max_ulp"]
        <= conftest.INVARIANCE_EPS_MULTIPLE
    )


@pytest.mark.parametrize("partitions", [2, 4, 8])
def test_state_by_action_partitions_equal_the_same_state_mesh_bitwise(
    *, report: _Report, partitions: int
) -> None:
    """Against the ordinary route sharding wealth over the same number of
    devices, so each device evaluates the same state cells."""
    assert (
        _state_by_action(report=report, partitions=partitions)[
            "mismatches_to_same_state_mesh"
        ]
        == []
    )


@pytest.mark.parametrize("partitions", [2, 4, 8])
def test_state_by_action_partitions_agree_with_one_device_to_the_last_units(
    *, report: _Report, partitions: int
) -> None:
    """A different count of state cells per device compiles a differently
    vectorized `Q`, as it does for the ordinary sharded route."""
    assert (
        _state_by_action(report=report, partitions=partitions)["max_ulp_to_one_device"]
        <= conftest.INVARIANCE_EPS_MULTIPLE
    )


@pytest.mark.parametrize("partitions", [2, 4, 8])
def test_a_state_by_action_regime_shards_states_and_replicates_over_actions(
    *, report: _Report, partitions: int
) -> None:
    assert _state_by_action(report=report, partitions=partitions)["working_layout"] == {
        "n_devices": 8,
        "mesh": {"wealth": 8 // partitions, _PARTITION_AXIS: partitions},
        "spec": ["wealth"],
    }


def test_partitions_compose_with_type_local_invariant_blocks_bitwise(
    report: _Report,
) -> None:
    """Against the same blocked solve on the ordinary route."""
    assert report["invariant_blocks_action_only"]["mismatches"] == []


def test_state_by_action_partitions_compose_with_invariant_blocks_to_the_last_units(
    report: _Report,
) -> None:
    """Against the same blocked solve on the ordinary route and state mesh.

    The exact merge cannot move a value; at float64 one cell has been seen one
    unit in the last place apart, the same for every partition count and moved
    to another period when the backend optimizer is off, which locates it in the
    compiled arithmetic of `Q` rather than in the reduction.
    """
    assert (
        report["invariant_blocks_state_by_action"]["max_ulp"]
        <= conftest.INVARIANCE_EPS_MULTIPLE
    )


@pytest.mark.parametrize("layout", ["action_only", "state_by_action"])
def test_partitioned_type_blocks_keep_the_block_layout(
    *, report: _Report, layout: str
) -> None:
    assert _invariant_blocks(report=report, layout=layout)["working_layout"] == (
        {"n_devices": 4, "mesh": {_PARTITION_AXIS: 4}, "spec": ["None", "None"]}
        if layout == "action_only"
        else {
            "n_devices": 8,
            "mesh": {"wealth": 2, _PARTITION_AXIS: 4},
            "spec": ["None", "wealth"],
        }
    )


def test_a_second_solve_with_changed_params_reuses_nothing_stale(
    report: _Report,
) -> None:
    assert report["changed_params"] == {"mismatches": [], "differs_from_first": True}


def test_the_partitioned_program_gathers_one_accumulator_per_cell_and_device(
    report: _Report,
) -> None:
    """Value, winning identity and feasibility of 8 wealth cells from 4 devices."""
    assert report["programs"]["partitioned"]["all_gather_shapes"] == [[8, 4]] * 3


def test_the_ordinary_program_exchanges_nothing(report: _Report) -> None:
    assert report["programs"]["ordinary"]["all_gather_shapes"] == []


def test_the_gathered_accumulators_are_reserved_as_compiled_workspace(
    report: _Report,
) -> None:
    """The exchange buffers are temporaries of the compiled program, so the
    compiler reservation admission charges covers them."""
    value_bytes = 8 if conftest.X64_ENABLED else 4
    gathered = 4 * 8 * (value_bytes + 4 + 1)

    assert report["programs"]["partitioned"]["temp_bytes"] >= gathered


def test_a_partitioned_solve_refuses_one_byte_below_its_workspace_budget(
    report: _Report,
) -> None:
    """The exchanged accumulators fit at the reported ceiling, never below it."""
    assert report["budget"]["refused_below_ceiling"] is True


def test_a_partitioned_solve_at_its_exact_workspace_budget_preserves_values(
    report: _Report,
) -> None:
    """The admitted solve publishes every ordinary fixed-width value bitwise."""
    assert report["budget"]["mismatches_at_ceiling"] == []


def test_a_partitioned_solve_at_its_exact_budget_publishes_terminal_bequests(
    report: _Report,
) -> None:
    """Terminal wealth nodes receive their square-root bequests."""
    np.testing.assert_allclose(
        report["budget"]["terminal_values"],
        np.sqrt(np.linspace(1.0, 10.0, 8)),
        rtol=10**-conftest.DECIMAL_PRECISION,
        atol=0.0,
    )


def test_block_major_action_partitions_publish_complete_period_major_values(
    report: _Report,
) -> None:
    """Every labelled value keeps its dtype, shape and bytes at fixed widths."""
    assert {
        layout: result["value_mismatches"]
        for layout, result in report["block_major"].items()
    } == {"action_only": [], "state_by_action": []}


@pytest.mark.parametrize("route", ["split", "combined"])
def test_block_major_action_partitions_preserve_the_complete_simulated_panel(
    *, report: _Report, route: str
) -> None:
    """Same-subject panels keep their metadata and raw bytes, including signed zero."""
    assert {
        layout: result["panel_mismatches"][route]
        for layout, result in report["block_major"].items()
    } == {"action_only": [], "state_by_action": []}


def test_block_major_action_partitions_release_every_retained_component(
    report: _Report,
) -> None:
    """All three preference codes release their device values after host retention."""
    assert {
        layout: result["release"] for layout, result in report["block_major"].items()
    } == {
        layout: {"codes": [0, 1, 2], "live_blocks": []}
        for layout in ("action_only", "state_by_action")
    }
