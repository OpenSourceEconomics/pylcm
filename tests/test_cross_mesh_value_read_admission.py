"""Reading a value stored on another regime's device mesh, and paying for it.

Two regimes that retain different sharded states hold device axes with
different names, different extents and — where the device count allows — no
device in common.  Solving the earlier regime reads the later one's stored
value across exactly that boundary.

The required layout of such a read belongs to the value being read and to the
core reading it, never to the core that produced the value: the producing
core's partition spec names its own axes and its own rank, neither of which the
value shares once the two regimes retain different states.  A value that is not
resident on the reading core's mesh therefore arrives replicated on that mesh.
The replica is what the reading core can address, and it is also the largest
thing a read can cost, so planning sizes it before it is issued: the full value
on every device of the reading mesh, with the stored value still alive on its
own devices for the duration of the copy.

The witnesses here pin that pair of claims together:

- the layout — one placement matrix over retained subsets, declared axis order,
  which regime owns which device block, and extents that do not divide the
  device count, each against the single-device solution and simulation of the
  same model;
- the price — the destination bytes, the operator's own scratch and the devices
  the operator touches, read off the planned transfer, and the budget that
  admits or refuses a core at that exact quantity.

The placement witnesses run in child processes, because the placements under
test only exist where several devices do, and a topology is pinned before JAX
initializes a backend.
"""

import json
import re
import subprocess
import sys
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import tests.conftest
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.execution import ExecutionConfig
from lcm.typing import FloatND, ScalarInt

_REPO_ROOT = Path(__file__).parent.parent

#: Devices the two-axis witness runs on: one `a` mesh and one `b` mesh, disjoint.
_N_DEVICES = 6

#: Extent of `a`, and so the mesh a regime retaining only `a` takes.
_A_EXTENT = 2

#: Extent of `b`, and so the mesh a regime retaining only `b` takes.
_B_EXTENT = 3

#: Points of the continuous state both regimes retain.
_N_WEALTH = 6

#: A budget no placement of this model fits in, so admission refuses by name.
_IMPOSSIBLE_BUDGET_BYTES = 64

#: A budget every placement of this model fits in comfortably.
_AMPLE_BUDGET_BYTES = 2**28

_PARAMS = {"discount_factor": 0.95}

_INITIAL_CONDITIONS = {
    "age": jnp.zeros(4),
    "wealth": jnp.asarray([20.0, 40.0, 60.0, 80.0]),
    "a": jnp.asarray([0, 1, 1, 0], dtype=jnp.int32),
    "b": jnp.asarray([0, 1, 2, 1], dtype=jnp.int32),
    "regime_id": jnp.zeros(4, dtype=jnp.int32),
}


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt


@categorical(ordered=True)
class _A:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=True)
class _B:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


_WEALTH = LinSpacedGrid(start=1.0, stop=100.0, n_points=_N_WEALTH)
_CONSUMPTION = LinSpacedGrid(start=1.0, stop=10.0, n_points=4)


def _next_wealth(*, wealth: FloatND, consumption: FloatND) -> FloatND:
    return wealth - consumption


def _utility(*, consumption: FloatND) -> FloatND:
    return jnp.log(consumption)


def _utility_of_a(*, consumption: FloatND, a: ScalarInt) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.1 * a)


def _utility_of_b(*, consumption: FloatND, b: ScalarInt) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.2 * b)


def _utility_of_a_and_b(*, consumption: FloatND, a: ScalarInt, b: ScalarInt) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.1 * a + 0.2 * b)


def _bequest(*, wealth: FloatND) -> FloatND:
    return jnp.log(wealth)


def _bequest_of_a(*, wealth: FloatND, a: ScalarInt) -> FloatND:
    return jnp.log(wealth) * (1.0 + 0.1 * a)


def _bequest_of_b(*, wealth: FloatND, b: ScalarInt) -> FloatND:
    return jnp.log(wealth) * (1.0 + 0.2 * b)


def _bequest_of_a_and_b(*, wealth: FloatND, a: ScalarInt, b: ScalarInt) -> FloatND:
    return jnp.log(wealth) * (1.0 + 0.1 * a + 0.2 * b)


_WORKING_UTILITY = {
    ("a",): _utility_of_a,
    ("b",): _utility_of_b,
    ("a", "b"): _utility_of_a_and_b,
    (): _utility,
}

_RETIRED_UTILITY = {
    ("a",): _bequest_of_a,
    ("b",): _bequest_of_b,
    ("a", "b"): _bequest_of_a_and_b,
    (): _bequest,
}


def _retire() -> FloatND:
    return jnp.asarray(1.0)


def _enter_b(*, wealth: FloatND) -> FloatND:
    """Probabilities over `_B`, richer entrants arriving in higher categories."""
    high = jnp.clip(wealth / 100.0, 0.0, 1.0)
    return jnp.stack([1.0 - high, 0.5 * high, 0.5 * high], axis=-1)


def build_model(
    *,
    devices: tuple[int, ...],
    sharded: tuple[str, ...],
    working_reads: tuple[str, ...] = ("a",),
    retired_reads: tuple[str, ...] = ("b",),
    states_lead_with_b: bool = False,
    retired_declared_first: bool = False,
    **config: Any,
) -> Model:
    """Build the working/retired pair whose solve reads across regime meshes.

    `working` is active in the first period and leaves into the terminal
    `retired` regime, so solving `working` reads `retired`'s stored value.
    Which discrete states each regime's utility reads decides which states it
    retains, and therefore which device axis it holds.

    Args:
        devices: Device ids the model is planned for.
        sharded: Names of the model-level discrete states given a device axis.
        working_reads: Discrete states the working regime's utility reads.
        retired_reads: Discrete states the bequest function reads.
        states_lead_with_b: Whether `b` is declared before `a`.
        retired_declared_first: Whether `retired` is declared before `working`.
        **config: Further `ExecutionConfig` fields.

    Returns:
        The model.

    """
    working = Regime(
        active=lambda age: age < 1,
        transition={"retired": MarkovTransition(_retire)},
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        functions={"utility": _WORKING_UTILITY[tuple(working_reads)]},
        state_transitions={"wealth": _next_wealth},
    )
    retired = Regime(
        active=lambda age: age >= 1,
        transition=None,
        states={"wealth": _WEALTH},
        functions={"utility": _RETIRED_UTILITY[tuple(retired_reads)]},
    )
    regimes = (
        {"retired": retired, "working": working}
        if retired_declared_first
        else {"working": working, "retired": retired}
    )
    grids = {
        "a": DiscreteGrid(category_class=_A),
        "b": DiscreteGrid(category_class=_B),
    }
    states = (
        {"b": grids["b"], "a": grids["a"]}
        if states_lead_with_b
        else {"a": grids["a"], "b": grids["b"]}
    )
    return Model(
        regimes=regimes,
        states=states,
        state_transitions={
            "a": {"retired": fixed_transition("a")},
            "b": {"retired": MarkovTransition(_enter_b)},
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(
            devices=devices, sharded_states=sharded, **config
        ),
    )


def _solve_capturing_transfers(*, model: Model) -> tuple[Any, list[Any]]:
    """Solve one model and return its solution with every planned core."""
    from _lcm.solution import backward_induction  # noqa: PLC0415

    captured: list[Any] = []
    original = backward_induction._attach_resolved_output_layout

    def capture(**kwargs: Any) -> Any:
        core = original(**kwargs)
        captured.append(core)
        return core

    with pytest.MonkeyPatch.context() as probe:
        probe.setattr(
            backward_induction, "_attach_resolved_output_layout", capture, raising=True
        )
        solution = model.solve(params=_PARAMS, log_level="off")
    return solution, captured


def _placement_report(
    *, model: Model, reference: Model, decimal: int
) -> dict[str, Any]:
    """Report placement, shapes, transfer kinds and agreement with the reference."""
    solution, captured = _solve_capturing_transfers(model=model)
    reference_solution = reference.solve(params=_PARAMS, log_level="off")

    devices: dict[str, list[int]] = {}
    value_shapes: dict[str, list[int]] = {}
    mismatches: list[str] = []
    for period, values in solution.values.items():
        for regime_name, value in values.items():
            devices[regime_name] = sorted(
                device.id for device in value.sharding.device_set
            )
            value_shapes[regime_name] = list(value.shape)
            expected = np.asarray(reference_solution.values[period][regime_name])
            try:
                np.testing.assert_array_almost_equal(
                    np.asarray(value), expected, decimal=decimal
                )
            except AssertionError as mismatch:
                mismatches.append(f"period {period}, {regime_name}: {mismatch}")

    transfers = [
        {
            "target_regime": transfer.target.regime,
            "source_regime": transfer.source.source_regime,
            "kind": str(transfer.kind),
            "per_device_bytes": transfer.cost.per_device_bytes,
            "logical_bytes": transfer.cost.logical_bytes,
            "temporary_bytes": transfer.cost.temporary_bytes,
            "devices": list(transfer.cost.devices),
            "stored_devices": sorted(
                device.id for device in transfer.stored_sharding.device_set
            ),
            "required_devices": sorted(
                device.id for device in transfer.source_sharding.device_set
            ),
            "required_spec": str(transfer.source_sharding.spec),
            "stored_spec": str(transfer.stored_sharding.spec),
            "expected_shape": list(transfer.expected_shape),
        }
        for core in captured
        for transfer in core.input_transfer_plan
    ]
    return {
        "pruned_variables": {
            name: sorted(pruned) for name, pruned in model.pruned_variables.items()
        },
        "devices": devices,
        "value_shapes": value_shapes,
        "solution_matches_reference": not mismatches,
        "solution_mismatches": mismatches,
        "transfer_kinds": sorted({item["kind"] for item in transfers}),
        "transfers": transfers,
        "_solution": solution,
        "_reference_solution": reference_solution,
    }


def _simulation_mismatch(
    *,
    model: Model,
    solution: Any,
    reference: Model,
    reference_solution: Any,
    decimal: int,
) -> str:
    """Return the frame mismatch between a model and its reference, empty if none."""
    result = model.simulate(
        params=_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        solution=solution,
        log_level="off",
        seed=42,
    )
    expected = reference.simulate(
        params=_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        solution=reference_solution,
        log_level="off",
        seed=42,
    )
    tolerance = 10.0**-decimal
    try:
        pd.testing.assert_frame_equal(
            result.to_dataframe(use_labels=False),
            expected.to_dataframe(use_labels=False),
            rtol=tolerance,
            atol=tolerance,
        )
    except AssertionError as mismatch:
        return str(mismatch)
    return ""


def _reference_model(**variant: Any) -> Model:
    """Return the single-device, unsharded model the placements are checked against."""
    return build_model(devices=(0,), sharded=(), **variant)


def report_two_axis(*, decimal: int) -> dict[str, Any]:
    """Report the two-axis placement: `working` holds `a`, `retired` holds `b`."""
    model = build_model(devices=tuple(range(_N_DEVICES)), sharded=("a", "b"))
    reference = _reference_model()
    report = _placement_report(model=model, reference=reference, decimal=decimal)
    solution = report.pop("_solution")
    reference_solution = report.pop("_reference_solution")
    report["simulation_mismatch"] = _simulation_mismatch(
        model=model,
        solution=solution,
        reference=reference,
        reference_solution=reference_solution,
        decimal=decimal,
    )
    report["simulation_matches_reference"] = not report["simulation_mismatch"]
    return report


def report_matrix(*, decimal: int) -> dict[str, Any]:
    """Report every placement variant of the same topology against its reference."""
    variants: dict[str, dict[str, Any]] = {
        "source_a_target_b": {},
        "both_read_b": {
            "working_reads": ("b",),
            "retired_reads": ("b",),
            "sharded": ("b",),
        },
        "source_reads_both": {"working_reads": ("a", "b"), "retired_reads": ("b",)},
        "target_reads_both": {"working_reads": ("a",), "retired_reads": ("a", "b")},
        "b_declared_first": {"states_lead_with_b": True},
        "retired_declared_first": {"retired_declared_first": True},
    }
    reports: dict[str, Any] = {}
    for name, variant in variants.items():
        variant: dict[str, Any] = {"sharded": ("a", "b"), **variant}  # noqa: PLW2901
        reference_variant = {
            key: value for key, value in variant.items() if key != "sharded"
        }
        model = build_model(devices=tuple(range(_N_DEVICES)), **variant)
        reference = _reference_model(**reference_variant)
        report = _placement_report(model=model, reference=reference, decimal=decimal)
        solution = report.pop("_solution")
        reference_solution = report.pop("_reference_solution")
        report["simulation_mismatch"] = _simulation_mismatch(
            model=model,
            solution=solution,
            reference=reference,
            reference_solution=reference_solution,
            decimal=decimal,
        )
        report["simulation_matches_reference"] = not report["simulation_mismatch"]
        reports[name] = report
    return reports


def report_indivisible_extents() -> dict[str, Any]:
    """Report how placement answers extents the visible device count cannot host."""
    from lcm.exceptions import ExecutionPlanningError  # noqa: PLC0415

    report: dict[str, Any] = {}
    try:
        model = build_model(
            devices=(0, 1, 2, 3),
            sharded=("a", "b"),
            working_reads=("a", "b"),
            retired_reads=("b",),
        )
        model.solve(params=_PARAMS, log_level="off")
    except ExecutionPlanningError as refusal:
        report["product_refused"] = True
        report["product_message"] = str(refusal)
    else:
        report["product_refused"] = False
        report["product_message"] = ""

    smaller = build_model(devices=(0, 1), sharded=("a", "b"))
    solution, _ = _solve_capturing_transfers(model=smaller)
    report["smaller_mesh_devices"] = {
        regime_name: sorted(device.id for device in value.sharding.device_set)
        for values in solution.values.values()
        for regime_name, value in values.items()
    }
    return report


def report_subject_sharded_simulation(*, decimal: int) -> dict[str, Any]:
    """Report the two-axis model simulated with subjects spread over the devices."""
    model = build_model(
        devices=tuple(range(_N_DEVICES)),
        sharded=("a", "b"),
        simulation_sharding="subjects",
        axis_widths={"subject": 4},
    )
    reference = _reference_model()
    mismatch = _simulation_mismatch(
        model=model,
        solution=model.solve(params=_PARAMS, log_level="off"),
        reference=reference,
        reference_solution=reference.solve(params=_PARAMS, log_level="off"),
        decimal=decimal,
    )
    return {
        "simulation_matches_reference": not mismatch,
        "simulation_mismatch": mismatch,
    }


def report_solution_round_trip(*, decimal: int) -> dict[str, Any]:
    """Report simulating the two-axis model from its persisted, reloaded solution."""
    import tempfile  # noqa: PLC0415

    from lcm.persistence import load_solution  # noqa: PLC0415

    model = build_model(devices=tuple(range(_N_DEVICES)), sharded=("a", "b"))
    reference = _reference_model()
    solution = model.solve(params=_PARAMS, log_level="off")
    reference_solution = reference.solve(params=_PARAMS, log_level="off")
    with tempfile.TemporaryDirectory() as directory:
        # The reloaded entries read their leaves lazily, so the archive has to
        # outlive the simulation that consumes them.
        reloaded = load_solution(path=solution.save(path=Path(directory) / "values.h5"))
        mismatch = _simulation_mismatch(
            model=model,
            solution=reloaded,
            reference=reference,
            reference_solution=reference_solution,
            decimal=decimal,
        )
    return {
        "simulation_matches_reference": not mismatch,
        "simulation_mismatch": mismatch,
    }


def report_admission() -> dict[str, Any]:
    """Report what the cross-mesh read reserves and the budget that refuses it."""
    from _lcm.execution.footprint import (  # noqa: PLC0415
        ArtifactFootprint,
        ResidentInventory,
    )
    from _lcm.solution.backward_induction import _triples_within_budget  # noqa: PLC0415
    from lcm.exceptions import ExecutionPlanningError  # noqa: PLC0415

    model = build_model(devices=tuple(range(_N_DEVICES)), sharded=("a", "b"))
    _, captured = _solve_capturing_transfers(model=model)
    crossings = [
        transfer
        for core in captured
        for transfer in core.input_transfer_plan
        if str(transfer.kind) == "cross_mesh_copy"
    ]
    transfer = crossings[0]
    cost = transfer.cost
    destination_devices = tuple(
        sorted(device.id for device in transfer.source_sharding.device_set)
    )

    # One shared destination is reserved for the whole period it is read in,
    # at the bytes the required layout holds on each device it lands on.
    reservation: dict[Hashable, ArtifactFootprint] = {
        (transfer.target, transfer.source_sharding): ArtifactFootprint(
            bytes_per_device=cost.per_device_bytes, device_ids=destination_devices
        )
    }
    inventory = ResidentInventory(
        device_ids=destination_devices,
        live={},
        peer_bytes=dict.fromkeys(destination_devices, 0),
        declared_inputs=(),
        shared_copies=reservation,
    )
    boundary = inventory.resident_bytes(consumes=(), consumed_copies=frozenset())
    triple = (transfer.source.source_regime, transfer.target.period, "core")
    admitted = {
        offset: bool(
            _triples_within_budget(
                candidates_by_triple={triple: ()},
                resident_bytes_by_triple={triple: boundary},
                budget_bytes=boundary + offset,
            )
        )
        for offset in (-1, 0, 1)
    }

    try:
        build_model(
            devices=tuple(range(_N_DEVICES)),
            sharded=("a", "b"),
            device_memory_bytes=_IMPOSSIBLE_BUDGET_BYTES,
        ).solve(params=_PARAMS, log_level="off")
    except ExecutionPlanningError as refusal:
        budgeted_refusal = str(refusal)
    else:
        budgeted_refusal = ""
    resident = re.search(r"keeps (\d+) bytes resident", budgeted_refusal)

    budgeted = build_model(
        devices=tuple(range(_N_DEVICES)),
        sharded=("a", "b"),
        device_memory_bytes=_AMPLE_BUDGET_BYTES,
    )
    reference = _reference_model()
    budgeted_solution = budgeted.solve(params=_PARAMS, log_level="off")
    reference_solution = reference.solve(params=_PARAMS, log_level="off")
    ample_mismatches = [
        f"period {period}, {regime_name}"
        for period, values in budgeted_solution.values.items()
        for regime_name, value in values.items()
        if not np.allclose(
            np.asarray(value),
            np.asarray(reference_solution.values[period][regime_name]),
            rtol=1e-4,
            atol=1e-5,
        )
    ]

    return {
        "kind": str(transfer.kind),
        "expected_shape": list(transfer.expected_shape),
        "logical_bytes": cost.logical_bytes,
        "per_device_bytes": cost.per_device_bytes,
        "temporary_bytes": cost.temporary_bytes,
        "operation_class": str(cost.operation_class),
        "operator_devices": list(cost.devices),
        "stored_devices": sorted(
            device.id for device in transfer.stored_sharding.device_set
        ),
        "destination_devices": list(destination_devices),
        "required_spec": str(transfer.source_sharding.spec),
        "boundary_bytes": boundary,
        "admitted_below": admitted[-1],
        "admitted_at": admitted[0],
        "admitted_above": admitted[1],
        "budgeted_refusal": budgeted_refusal,
        "budgeted_resident_bytes": int(resident.group(1)) if resident else 0,
        "ample_budget_matches_reference": not ample_mismatches,
        "ample_budget_mismatches": ample_mismatches,
    }


def _run_in_child_process(
    *, entry_point: str, n_devices: int, arguments: str = ""
) -> dict[str, Any]:
    """Run one module-level report function on `n_devices` and return its report.

    The child carries this run's float policy — `jax_enable_x64`, the matmul
    precision and the matching `DECIMAL_PRECISION` — because a topology pin
    needs a fresh process and a fresh process reads none of pytest's options.

    Args:
        entry_point: Name of a function in this module returning a
            JSON-serializable mapping.
        n_devices: Number of CPU devices to pin the child to.
        arguments: Call arguments for the entry point, as source text.

    Returns:
        Dictionary of the report the child process produced.

    """
    code = (
        "import json, sys; import jax; "
        f"jax.config.update('jax_num_cpu_devices', {n_devices}); "
        "jax.config.update('jax_platform_name', 'cpu'); "
        f"jax.config.update('jax_enable_x64', {tests.conftest.X64_ENABLED!r}); "
        "jax.config.update('jax_default_matmul_precision', 'highest'); "
        "from tests.test_cross_mesh_value_read_admission import "
        f"{entry_point} as entry; "
        f"sys.stdout.write('@@' + json.dumps(entry({arguments})) + '@@')"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    _, _, rest = completed.stdout.partition("@@")
    payload, _, _ = rest.partition("@@")
    assert payload, completed.stdout
    return json.loads(payload)


def _tolerance_argument() -> str:
    """Return the call argument handing a child process this run's tolerance."""
    return f"decimal={tests.conftest.DECIMAL_PRECISION!r}"


@pytest.fixture(scope="module")
def two_axis() -> dict[str, Any]:
    """Return the two-axis report from one six-device child process."""
    return _run_in_child_process(
        entry_point="report_two_axis",
        n_devices=_N_DEVICES,
        arguments=_tolerance_argument(),
    )


@pytest.fixture(scope="module")
def matrix() -> dict[str, Any]:
    """Return the placement-variant reports from one six-device child process."""
    return _run_in_child_process(
        entry_point="report_matrix",
        n_devices=_N_DEVICES,
        arguments=_tolerance_argument(),
    )


@pytest.fixture(scope="module")
def indivisible() -> dict[str, Any]:
    """Return the indivisible-extent report from one four-device child process."""
    return _run_in_child_process(entry_point="report_indivisible_extents", n_devices=4)


@pytest.fixture(scope="module")
def subject_sharded() -> dict[str, Any]:
    """Return the subject-sharded simulation report from six devices."""
    return _run_in_child_process(
        entry_point="report_subject_sharded_simulation",
        n_devices=_N_DEVICES,
        arguments=_tolerance_argument(),
    )


@pytest.fixture(scope="module")
def round_trip() -> dict[str, Any]:
    """Return the persisted-solution simulation report from six devices."""
    return _run_in_child_process(
        entry_point="report_solution_round_trip",
        n_devices=_N_DEVICES,
        arguments=_tolerance_argument(),
    )


@pytest.fixture(scope="module")
def admission() -> dict[str, Any]:
    """Return the admission report from one six-device child process."""
    return _run_in_child_process(entry_point="report_admission", n_devices=_N_DEVICES)


def test_each_regime_retains_only_the_sharded_state_its_dag_reads(
    two_axis: dict[str, Any],
) -> None:
    """The reading regime keeps `a` and drops `b`; the stored regime the reverse."""
    assert (
        two_axis["pruned_variables"]["working"],
        two_axis["pruned_variables"]["retired"],
    ) == (["b"], ["a"])


def test_the_two_regimes_take_meshes_of_their_own_extents(
    two_axis: dict[str, Any],
) -> None:
    """Each regime spans one device per category of the state it retains."""
    assert (two_axis["devices"]["working"], two_axis["devices"]["retired"]) == (
        [0, 1],
        [2, 3, 4],
    )


def test_each_regimes_value_leads_with_its_own_category_axis(
    two_axis: dict[str, Any],
) -> None:
    """A regime's value carries the axis of the state it retains, and no other."""
    assert (
        two_axis["value_shapes"]["working"],
        two_axis["value_shapes"]["retired"],
    ) == ([_A_EXTENT, _N_WEALTH], [_B_EXTENT, _N_WEALTH])


def test_a_value_stored_on_another_mesh_is_read_as_a_cross_mesh_copy(
    two_axis: dict[str, Any],
) -> None:
    """The read across the two device axes is served by one mesh copy."""
    assert two_axis["transfer_kinds"] == ["cross_mesh_copy"]


def test_a_cross_mesh_read_is_delivered_replicated_on_the_reading_mesh(
    two_axis: dict[str, Any],
) -> None:
    """The reading core addresses none of the stored value's axes, so it gets all."""
    assert [transfer["required_spec"] for transfer in two_axis["transfers"]] == ["P()"]


def test_the_two_axis_solution_equals_the_single_device_solution(
    two_axis: dict[str, Any],
) -> None:
    """Placement partitions the solve; it never changes the values published."""
    assert two_axis["solution_matches_reference"] is True


def test_the_two_axis_simulation_equals_the_single_device_simulation(
    two_axis: dict[str, Any],
) -> None:
    """The simulated frames agree with the unsharded reference model's."""
    assert two_axis["simulation_matches_reference"] is True


@pytest.mark.parametrize(
    "variant",
    [
        "source_a_target_b",
        "both_read_b",
        "source_reads_both",
        "target_reads_both",
        "b_declared_first",
        "retired_declared_first",
    ],
)
def test_every_placement_variant_publishes_the_reference_values(
    *, matrix: dict[str, Any], variant: str
) -> None:
    """Retained subset, declared order and block ownership never move a value."""
    assert matrix[variant]["solution_matches_reference"] is True


@pytest.mark.parametrize(
    "variant",
    [
        "source_a_target_b",
        "both_read_b",
        "source_reads_both",
        "target_reads_both",
        "b_declared_first",
        "retired_declared_first",
    ],
)
def test_every_placement_variant_simulates_the_reference_frames(
    *, matrix: dict[str, Any], variant: str
) -> None:
    """Placement never moves a simulated path either."""
    assert matrix[variant]["simulation_matches_reference"] is True


def test_declaring_the_regimes_in_the_other_order_swaps_the_device_blocks(
    matrix: dict[str, Any],
) -> None:
    """Sharded regimes take consecutive blocks in the order they are declared."""
    assert (
        matrix["retired_declared_first"]["devices"]["retired"],
        matrix["retired_declared_first"]["devices"]["working"],
    ) == ([0, 1, 2], [3, 4])


def test_two_regimes_retaining_one_state_still_hold_separate_device_blocks(
    matrix: dict[str, Any],
) -> None:
    """A shared state name does not make one mesh: each regime gets its own block."""
    assert (
        matrix["both_read_b"]["devices"]["working"],
        matrix["both_read_b"]["devices"]["retired"],
    ) == ([0, 1, 2], [3, 4, 5])


def test_a_read_between_separate_blocks_of_one_state_is_a_cross_mesh_copy(
    matrix: dict[str, Any],
) -> None:
    """Sharing an axis name buys nothing while the two meshes share no device."""
    assert matrix["both_read_b"]["transfer_kinds"] == ["cross_mesh_copy"]


def test_a_regime_retaining_both_states_spans_the_product_of_their_extents(
    matrix: dict[str, Any],
) -> None:
    """Several retained sharded states scatter one grid point per device."""
    assert matrix["source_reads_both"]["devices"]["working"] == list(range(_N_DEVICES))


def test_extents_whose_product_exceeds_the_devices_are_refused_while_planning(
    indivisible: dict[str, Any],
) -> None:
    """A regime needing more devices than exist has no mesh, so planning stops."""
    assert indivisible["product_refused"] is True


def test_the_product_refusal_names_the_points_and_the_devices(
    indivisible: dict[str, Any],
) -> None:
    """The refusal says what was asked for and what was available."""
    message = indivisible["product_message"]
    assert "Gridpoints product: 6" in message
    assert "Available devices: 4" in message


def test_an_extent_the_devices_do_not_divide_takes_the_largest_divisor_mesh(
    indivisible: dict[str, Any],
) -> None:
    """Three categories over two devices divide only one way: a single device."""
    assert indivisible["smaller_mesh_devices"]["retired"] == [0]


def test_the_two_axis_model_simulates_the_reference_over_sharded_subjects(
    subject_sharded: dict[str, Any],
) -> None:
    """Spreading subjects over the devices leaves the simulated frames unchanged."""
    assert subject_sharded["simulation_matches_reference"] is True


def test_a_persisted_cross_mesh_solution_simulates_the_reference_frames(
    round_trip: dict[str, Any],
) -> None:
    """Saving and reloading values across the two layouts reproduces the frames."""
    assert round_trip["simulation_matches_reference"] is True


def test_the_cross_mesh_read_costs_the_whole_value_on_every_reading_device(
    admission: dict[str, Any],
) -> None:
    """A replica holds no shard, so its per-device claim is the whole value."""
    assert admission["per_device_bytes"] == admission["logical_bytes"]


def test_the_cross_mesh_read_reserves_its_own_scratch_at_the_replica_size(
    admission: dict[str, Any],
) -> None:
    """The operator holds a second copy of what it delivers while it runs."""
    assert admission["temporary_bytes"] == admission["per_device_bytes"]


def test_the_planned_operator_names_both_meshes_devices(
    admission: dict[str, Any],
) -> None:
    """Nothing the copy touches is left out of the devices it is charged to."""
    assert admission["operator_devices"] == sorted(
        admission["stored_devices"] + admission["destination_devices"]
    )


def test_the_stored_value_and_its_replica_occupy_different_devices(
    admission: dict[str, Any],
) -> None:
    """The stored value stays where it is, so both copies are live at once."""
    assert not set(admission["stored_devices"]) & set(admission["destination_devices"])


def test_the_reserved_destination_bytes_are_the_replicas_own_size(
    admission: dict[str, Any],
) -> None:
    """The whole-period reservation charges each reading device the full value."""
    assert admission["boundary_bytes"] == admission["per_device_bytes"]


def test_a_budget_below_the_reserved_replica_admits_no_core(
    admission: dict[str, Any],
) -> None:
    """A device with less room than the replica needs hosts no workspace."""
    assert admission["admitted_below"] is False


def test_a_budget_equal_to_the_reserved_replica_admits_no_core(
    admission: dict[str, Any],
) -> None:
    """A budget the reservation exhausts leaves nothing for a workspace."""
    assert admission["admitted_at"] is False


def test_a_budget_above_the_reserved_replica_admits_the_core(
    admission: dict[str, Any],
) -> None:
    """One byte of room beyond the reservation is what admission asks for."""
    assert admission["admitted_above"] is True


def test_a_budget_too_small_for_the_cross_mesh_solve_is_refused_by_name(
    admission: dict[str, Any],
) -> None:
    """A solve that cannot fit its reads names the regime that cannot host them."""
    assert "'working'" in admission["budgeted_refusal"]


def test_a_budgeted_cross_mesh_solve_publishes_the_reference_values(
    admission: dict[str, Any],
) -> None:
    """Admitting the read changes what is reserved, never what is computed."""
    assert admission["ample_budget_matches_reference"] is True


def test_the_budgeted_refusal_counts_the_replica_it_would_have_to_hold(
    admission: dict[str, Any],
) -> None:
    """The bytes a refused cell reports include the replica the read delivers."""
    assert admission["budgeted_resident_bytes"] >= admission["per_device_bytes"]
