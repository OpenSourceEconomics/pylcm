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

import dataclasses
import json
import re
import subprocess
import sys
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


def _capture_reading_inventory(*, model: Model) -> tuple[Any, Any, Any]:
    """Solve one model and return the admission inventory of its cross-mesh read.

    The residency builder is observed, never replaced: the solve that produces
    the inventory is the budgeted one the model would run anyway, so the
    quantities reported are the ones admission actually compared.

    Args:
        model: The budgeted model to solve.

    Returns:
        Tuple of the reading core's triple, its planned transfer and the
        inventory admission consulted at that core's scheduled position.

    """
    from _lcm.solution import backward_induction  # noqa: PLC0415

    captured: list[tuple[Any, Any]] = []
    original = backward_induction._resident_inventory_by_triple

    def observe(**kwargs: Any) -> Any:
        inventories = original(**kwargs)
        captured.append((kwargs["program_metadata"], inventories))
        return inventories

    with pytest.MonkeyPatch.context() as probe:
        probe.setattr(
            backward_induction, "_resident_inventory_by_triple", observe, raising=True
        )
        model.solve(params=_PARAMS, log_level="off")

    for metadata, inventories in captured:
        for triple, program in metadata.items():
            for transfer in program.input_transfer_plan:
                if str(transfer.kind) == "cross_mesh_copy":
                    return triple, transfer, inventories[triple]
    raise AssertionError("No planned core reads across the two regime meshes.")


def _endpoint_bytes(*, inventory: Any, device: int, scratch_bytes: int) -> int:
    """Charge one endpoint device, with the transfer scratch set to a given size.

    Nothing is excluded: no compiler pruning, no consumed destination copy. The
    number is the declared envelope that device carries while the copy is in
    flight, so passing zero yields the same envelope with the operator's own
    storage left out.
    """
    restricted = dataclasses.replace(
        inventory,
        device_ids=(device,) if device in inventory.device_ids else (),
        transfer_scratch_bytes={device: scratch_bytes},
    )
    return restricted.resident_bytes(consumes=(), consumed_copies=frozenset())


def _refusal_at(*, budget_bytes: int) -> str:
    """Return the two-axis model's refusal at one budget, empty if it solves."""
    from lcm.exceptions import ExecutionPlanningError  # noqa: PLC0415

    try:
        build_model(
            devices=tuple(range(_N_DEVICES)),
            sharded=("a", "b"),
            device_memory_bytes=budget_bytes,
        ).solve(params=_PARAMS, log_level="off")
    except ExecutionPlanningError as refusal:
        return str(refusal)
    return ""


def _refusal_without_transfer_scratch(*, budget_bytes: int) -> str:
    """Return the same refusal with the operator's own storage left uncharged.

    The control an accounting claim needs: an inventory that omits the declared
    temporary bytes has to reach a different verdict at the same budget, or the
    budget never tested them.
    """
    from _lcm.solution import backward_induction  # noqa: PLC0415

    original = backward_induction._resident_inventory_by_triple

    def without_scratch(**kwargs: Any) -> Any:
        return {
            triple: dataclasses.replace(inventory, transfer_scratch_bytes={})
            for triple, inventory in original(**kwargs).items()
        }

    with pytest.MonkeyPatch.context() as probe:
        probe.setattr(
            backward_induction,
            "_resident_inventory_by_triple",
            without_scratch,
            raising=True,
        )
        return _refusal_at(budget_bytes=budget_bytes)


def _refused_before_compiling(*, message: str) -> bool:
    """Whether a refusal came from the position gate rather than from a width.

    A core whose scheduled position already fills the budget is left out of the
    compilation waves and refused by name; one that passes that gate is refused
    only after every candidate width has been compiled and priced.
    """
    return "resident at the node's position" in message


def report_admission() -> dict[str, Any]:
    """Report what the cross-mesh read reserves and the budget that refuses it."""
    model = build_model(
        devices=tuple(range(_N_DEVICES)),
        sharded=("a", "b"),
        device_memory_bytes=_AMPLE_BUDGET_BYTES,
    )
    triple, transfer, inventory = _capture_reading_inventory(model=model)
    cost = transfer.cost
    destination_devices = tuple(
        sorted(device.id for device in transfer.source_sharding.device_set)
    )
    endpoints = inventory.admission_device_ids
    scratch = inventory.transfer_scratch_bytes

    # The position gate compares this exact number against the budget, so the
    # budget one byte above it is the first at which the core is compiled at all.
    boundary = inventory.resident_bytes()
    refusals = {
        offset: _refusal_at(budget_bytes=boundary + offset) for offset in (-1, 0, 1)
    }
    mutant_refusal = _refusal_without_transfer_scratch(budget_bytes=boundary)

    budgeted_refusal = _refusal_at(budget_bytes=_IMPOSSIBLE_BUDGET_BYTES)
    resident = re.search(r"keeps (\d+) bytes resident", budgeted_refusal)
    mutant_resident = re.search(r"keeps (\d+) bytes resident", mutant_refusal)

    reference = _reference_model()
    budgeted_solution = model.solve(params=_PARAMS, log_level="off")
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
        "reading_regime": triple[0],
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
        "workspace_devices": list(inventory.device_ids),
        "admission_devices": list(endpoints),
        "transfer_scratch_bytes": {
            str(device): reserved for device, reserved in scratch.items()
        },
        "endpoint_bytes": {
            str(device): _endpoint_bytes(
                inventory=inventory, device=device, scratch_bytes=scratch[device]
            )
            for device in endpoints
        },
        "endpoint_bytes_without_scratch": {
            str(device): _endpoint_bytes(
                inventory=inventory, device=device, scratch_bytes=0
            )
            for device in endpoints
        },
        "boundary_bytes": boundary,
        "admitted_below": not _refused_before_compiling(message=refusals[-1]),
        "admitted_at": not _refused_before_compiling(message=refusals[0]),
        "admitted_above": not _refused_before_compiling(message=refusals[1]),
        "refusal_at_boundary": refusals[0],
        "mutant_admitted_at_boundary": not _refused_before_compiling(
            message=mutant_refusal
        ),
        "mutant_resident_bytes": (
            int(mutant_resident.group(1)) if mutant_resident else 0
        ),
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


def test_admission_covers_every_device_the_copy_touches(
    admission: dict[str, Any],
) -> None:
    """A device that only sources the copy is compared against the budget too."""
    assert admission["admission_devices"] == admission["operator_devices"]


def test_a_device_that_only_sources_the_copy_is_no_workspace_device(
    admission: dict[str, Any],
) -> None:
    """The reading core runs on its own mesh; the stored value sits on another."""
    assert not set(admission["stored_devices"]) & set(admission["workspace_devices"])


@pytest.mark.parametrize("endpoint", ["source", "destination"])
def test_every_endpoint_device_is_charged_the_operators_own_storage(
    *, admission: dict[str, Any], endpoint: str
) -> None:
    """Each device the copy touches carries a second whole value while it runs."""
    key = "stored_devices" if endpoint == "source" else "destination_devices"
    charged = [
        admission["endpoint_bytes"][str(device)]
        - admission["endpoint_bytes_without_scratch"][str(device)]
        for device in admission[key]
    ]
    assert charged == [admission["temporary_bytes"]] * len(admission[key])


def test_a_source_device_is_charged_its_shard_and_the_operators_storage(
    admission: dict[str, Any],
) -> None:
    """Sourcing a copy costs the stored shard plus the whole value in flight."""
    shard = admission["logical_bytes"] // len(admission["stored_devices"])
    source = admission["stored_devices"][0]
    assert admission["endpoint_bytes"][str(source)] >= (
        shard + admission["temporary_bytes"]
    )


def test_a_budget_below_the_planned_position_admits_no_core(
    admission: dict[str, Any],
) -> None:
    """A device with less room than the plan already keeps hosts no workspace."""
    assert admission["admitted_below"] is False


def test_a_budget_equal_to_the_planned_position_admits_no_core(
    admission: dict[str, Any],
) -> None:
    """A budget the position exhausts leaves nothing for a workspace."""
    assert admission["admitted_at"] is False


def test_a_budget_above_the_planned_position_admits_the_core(
    admission: dict[str, Any],
) -> None:
    """One byte of room beyond the position is what admission asks for."""
    assert admission["admitted_above"] is True


def test_the_admitting_boundary_counts_the_operators_own_storage(
    admission: dict[str, Any],
) -> None:
    """Leaving the declared temporary bytes out moves the boundary by their size."""
    assert (
        admission["boundary_bytes"] - admission["mutant_resident_bytes"]
        == (admission["temporary_bytes"])
    )


def test_an_inventory_without_the_operators_storage_admits_the_refused_core(
    admission: dict[str, Any],
) -> None:
    """The budget that refuses the complete footprint passes the incomplete one."""
    assert admission["mutant_admitted_at_boundary"] is True


def test_the_refusal_names_the_transfer_charge_it_counted(
    admission: dict[str, Any],
) -> None:
    """A cell refused over a copy says how much of its budget the copy took."""
    assert (
        f"transfer operators reserve up to {admission['temporary_bytes']} bytes"
        in admission["refusal_at_boundary"]
    )


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
