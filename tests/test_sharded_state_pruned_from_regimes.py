"""A discrete sharded state may be dropped from regimes whose DAG never reads it.

Sharding spreads a discrete state's grid axis over devices, and only the regimes
that carry the state need that axis. A regime that prunes the state runs on a
single device; the regimes that retain it keep their sharded submesh, and values
crossing between the two placements move through the ordinary transfer
catalogue. Whether a regime prunes the state is a property of that regime's own
DAG, so the state may be dropped from a non-terminal regime just as from a
terminal one.

Both directions are covered: a state read in the early regimes and dropped by
the later ones, and the mirror where only a later regime reads it while an
earlier one enters it through a target-keyed law.

The placement witnesses run in a four-CPU-device child process, because the
placement under test only exists where several devices do, and a topology is
pinned before JAX initializes a backend. The refusals need no particular
topology and run in the test process itself.
"""

import json
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
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.typing import FloatND, ScalarInt

_REPO_ROOT = Path(__file__).parent.parent

#: Extent of the sharded `kind` grid, and so the mesh a retaining regime takes.
_KIND_EXTENT = 3

#: Devices the child process runs on: one full `kind` mesh plus one spare.
_N_DEVICES = 4


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _Kind:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


_WEALTH = LinSpacedGrid(start=1.0, stop=100.0, n_points=6)
_CONSUMPTION = LinSpacedGrid(start=1.0, stop=10.0, n_points=4)
_PARAMS = {"discount_factor": 0.95}
_INITIAL_CONDITIONS = {
    "age": jnp.zeros(4),
    "wealth": jnp.asarray([20.0, 40.0, 60.0, 80.0]),
    "kind": jnp.asarray([0, 1, 2, 1], dtype=jnp.int32),
    "regime_id": jnp.full(4, _RegimeId.working),
}


def _next_wealth(*, wealth: FloatND, consumption: FloatND) -> FloatND:
    return wealth - consumption


def _utility_of_consumption(consumption: FloatND) -> FloatND:
    return jnp.log(consumption)


def _utility_by_kind(*, consumption: FloatND, kind: ScalarInt) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.1 * kind)


def _bequest_utility(wealth: FloatND) -> FloatND:
    return jnp.log(wealth)


def _retire_at_one(age: FloatND) -> FloatND:
    return jnp.where(age < 1, _RegimeId.working, _RegimeId.retired)


def _die_at_three(age: FloatND) -> FloatND:
    return jnp.where(age >= 2, _RegimeId.dead, _RegimeId.retired)


def _entry_kind(wealth: FloatND) -> FloatND:
    """Probabilities over `_Kind`, richer entrants arriving in higher categories."""
    high = jnp.clip(wealth / 100.0, 0.0, 1.0)
    return jnp.stack([1.0 - high, 0.5 * high, 0.5 * high], axis=-1)


def _working(**overrides: Any) -> Regime:
    spec: dict[str, Any] = {
        "active": lambda age: age < 2,
        "transition": _retire_at_one,
        "states": {"wealth": _WEALTH},
        "actions": {"consumption": _CONSUMPTION},
        "functions": {"utility": _utility_of_consumption},
        "state_transitions": {"wealth": _next_wealth},
    }
    spec.update(overrides)
    return Regime(**spec)


def _retired(**overrides: Any) -> Regime:
    spec: dict[str, Any] = {
        "active": lambda age: (age >= 2) & (age < 3),
        "transition": _die_at_three,
        "states": {"wealth": _WEALTH},
        "actions": {"consumption": _CONSUMPTION},
        "functions": {"utility": _utility_of_consumption},
        "state_transitions": {"wealth": _next_wealth},
    }
    spec.update(overrides)
    return Regime(**spec)


def _dead() -> Regime:
    return Regime(
        active=lambda age: age >= 3,
        transition=None,
        states={"wealth": _WEALTH},
        functions={"utility": _bequest_utility},
    )


def _build(
    *, regimes: dict[str, Regime], sharded: tuple[str, ...], **config: Any
) -> Model:
    """Build the three-regime model on a fixed device set and grid vocabulary."""
    return Model(
        regimes=regimes,
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(sharded_states=sharded, **config),
        states={"kind": DiscreteGrid(category_class=_Kind)},
    )


def forward_model(*, devices: tuple[int, ...], sharded: tuple[str, ...]) -> Model:
    """`kind` read by the working regime and dropped by every later regime."""
    return _build(
        regimes={
            "working": _working(
                functions={"utility": _utility_by_kind},
                state_transitions={
                    "wealth": _next_wealth,
                    "kind": {"working": fixed_transition("kind")},
                },
            ),
            "retired": _retired(),
            "dead": _dead(),
        },
        sharded=sharded,
        devices=devices,
    )


def mirror_model(*, devices: tuple[int, ...], sharded: tuple[str, ...]) -> Model:
    """`kind` read only by the retirement regime, which the working regime enters."""
    return Model(
        regimes={
            "working": _working(),
            "retired": _retired(functions={"utility": _utility_by_kind}),
            "dead": _dead(),
        },
        states={"kind": DiscreteGrid(category_class=_Kind)},
        state_transitions={"kind": {"retired": MarkovTransition(_entry_kind)}},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(devices=devices, sharded_states=sharded),
    )


def build_unread_sharded_state() -> Model:
    """Build a model naming a sharded state no regime's DAG reads."""
    return _build(
        regimes={"working": _working(), "retired": _retired(), "dead": _dead()},
        sharded=("kind",),
        devices=(0,),
    )


def build_pruned_continuous_sharded_state() -> Model:
    """Build a model sharding a continuous state one regime prunes."""
    assets = LinSpacedGrid(start=1.0, stop=50.0, n_points=4)
    return Model(
        regimes={
            "working": _working(
                functions={
                    "utility": lambda consumption, assets: (
                        jnp.log(consumption) + 0.01 * assets
                    )
                },
                state_transitions={
                    "wealth": _next_wealth,
                    "assets": {"working": fixed_transition("assets")},
                },
            ),
            "retired": _retired(),
            "dead": _dead(),
        },
        states={"assets": assets},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(devices=(0,), sharded_states=("assets",)),
    )


def _solve_and_compare(
    *, model: Model, reference: Model, decimal: int
) -> dict[str, Any]:
    """Solve both models and report placement, shapes and agreement."""
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
    reference_solution = reference.solve(params=_PARAMS, log_level="off")

    devices: dict[str, list[int]] = {}
    value_shapes: dict[str, list[int]] = {}
    reference_shapes: dict[str, list[int]] = {}
    mismatches: list[str] = []
    for period, values in solution.values.items():
        for regime_name, value in values.items():
            devices[regime_name] = sorted(
                device.id for device in value.sharding.device_set
            )
            value_shapes[regime_name] = list(value.shape)
            expected = np.asarray(reference_solution.values[period][regime_name])
            reference_shapes[regime_name] = list(expected.shape)
            try:
                np.testing.assert_array_almost_equal(
                    np.asarray(value), expected, decimal=decimal
                )
            except AssertionError as mismatch:
                mismatches.append(f"period {period}, {regime_name}: {mismatch}")
    transfers = {
        transfer.target.regime: str(transfer.kind)
        for core in captured
        for transfer in core.input_transfer_plan
    }
    return {
        "pruned_variables": {
            name: sorted(pruned) for name, pruned in model.pruned_variables.items()
        },
        "devices": devices,
        "value_shapes": value_shapes,
        "reference_shapes": reference_shapes,
        "solution_matches_reference": not mismatches,
        "solution_mismatches": mismatches,
        "transfers": transfers,
        "_solution": solution,
        "_reference_solution": reference_solution,
    }


def _simulations_agree(
    *,
    model: Model,
    solution: Any,
    reference: Model,
    reference_solution: Any,
    decimal: int,
) -> tuple[str, Any]:
    """Simulate both models from the same conditions and compare their frames.

    Returns:
        Tuple of the mismatch report — empty when the frames agree — and the
        simulation result of the model under test.

    """
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
        return str(mismatch), result
    return "", result


def report_forward(*, decimal: int) -> dict[str, Any]:
    """Report the forward model's placement, values, transfers and persistence."""
    import tempfile  # noqa: PLC0415

    from lcm.result import SimulationResult  # noqa: PLC0415

    model = forward_model(devices=tuple(range(_N_DEVICES)), sharded=("kind",))
    reference = forward_model(devices=(0,), sharded=())
    report = _solve_and_compare(model=model, reference=reference, decimal=decimal)
    mismatch, result = _simulations_agree(
        model=model,
        solution=report.pop("_solution"),
        reference=reference,
        reference_solution=report.pop("_reference_solution"),
        decimal=decimal,
    )
    report["simulation_matches_reference"] = not mismatch
    report["simulation_mismatch"] = mismatch
    report["transfer_from_pruning_regime"] = report["transfers"]["retired"]
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "result"
        result.save(directory=target)
        reloaded = SimulationResult.load(directory=target)
        report["save_load_round_trips"] = reloaded.to_dataframe(
            use_labels=False
        ).equals(result.to_dataframe(use_labels=False))
    return report


def report_mirror(*, decimal: int) -> dict[str, Any]:
    """Report the mirror model's placement, values and transfers."""
    model = mirror_model(devices=tuple(range(_N_DEVICES)), sharded=("kind",))
    reference = mirror_model(devices=(0,), sharded=())
    report = _solve_and_compare(model=model, reference=reference, decimal=decimal)
    mismatch, _ = _simulations_agree(
        model=model,
        solution=report.pop("_solution"),
        reference=reference,
        reference_solution=report.pop("_reference_solution"),
        decimal=decimal,
    )
    report["simulation_matches_reference"] = not mismatch
    report["simulation_mismatch"] = mismatch
    report["transfer_from_retaining_regime"] = report["transfers"]["retired"]
    return report


def report_subject_sharded(*, decimal: int) -> dict[str, Any]:
    """Report the forward model simulated with subjects spread over every device."""
    model = _build(
        regimes={
            "working": _working(
                functions={"utility": _utility_by_kind},
                state_transitions={
                    "wealth": _next_wealth,
                    "kind": {"working": fixed_transition("kind")},
                },
            ),
            "retired": _retired(),
            "dead": _dead(),
        },
        sharded=("kind",),
        devices=tuple(range(_N_DEVICES)),
        simulation_sharding="subjects",
        axis_widths={"subject": _N_DEVICES},
    )
    reference = forward_model(devices=(0,), sharded=())
    mismatch, _ = _simulations_agree(
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


def _run_in_four_device_process(*, entry_point: str) -> dict[str, Any]:
    """Run one module-level report function on four CPU devices and return it.

    The child carries this run's float policy — `jax_enable_x64`, the matmul
    precision and the matching `DECIMAL_PRECISION` — because a topology pin
    needs a fresh process and a fresh process reads none of pytest's options.

    Args:
        entry_point: Name of a function in this module taking the tolerance in
            decimals and returning a JSON-serializable mapping.

    Returns:
        Dictionary of the report the child process produced.

    """
    code = (
        "import json, sys; import jax; "
        f"jax.config.update('jax_num_cpu_devices', {_N_DEVICES}); "
        "jax.config.update('jax_platform_name', 'cpu'); "
        f"jax.config.update('jax_enable_x64', {tests.conftest.X64_ENABLED!r}); "
        "jax.config.update('jax_default_matmul_precision', 'highest'); "
        "from tests.test_sharded_state_pruned_from_regimes import "
        f"{entry_point} as entry; "
        "sys.stdout.write('@@' + json.dumps("
        f"entry(decimal={tests.conftest.DECIMAL_PRECISION!r})) + '@@')"
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


@pytest.fixture(scope="module")
def forward() -> dict[str, Any]:
    """Return the forward model's report from one four-device child process."""
    return _run_in_four_device_process(entry_point="report_forward")


@pytest.fixture(scope="module")
def mirror() -> dict[str, Any]:
    """Return the mirror model's report from one four-device child process."""
    return _run_in_four_device_process(entry_point="report_mirror")


@pytest.fixture(scope="module")
def subject_sharded() -> dict[str, Any]:
    """Return the subject-sharded report from one four-device child process."""
    return _run_in_four_device_process(entry_point="report_subject_sharded")


def test_a_non_terminal_regime_may_prune_a_sharded_state(
    forward: dict[str, Any],
) -> None:
    """The retirement regime drops the sharded state its DAG never reads."""
    assert forward["pruned_variables"]["retired"] == ["kind"]


def test_a_reading_regime_keeps_the_sharded_state(forward: dict[str, Any]) -> None:
    """The working regime, which reads the state, retains it."""
    assert forward["pruned_variables"]["working"] == []


def test_a_retaining_regime_takes_one_device_per_category(
    forward: dict[str, Any],
) -> None:
    """The regime carrying the state runs on one device per grid point."""
    assert forward["devices"]["working"] == list(range(_KIND_EXTENT))


def test_a_pruning_non_terminal_regime_runs_on_one_device(
    forward: dict[str, Any],
) -> None:
    """The regime without the axis is placed as a single-device regime."""
    assert len(forward["devices"]["retired"]) == 1


def test_a_pruning_regimes_value_has_the_unsharded_shape(
    forward: dict[str, Any],
) -> None:
    """Dropping the state drops its axis from the value the regime publishes."""
    assert forward["value_shapes"]["retired"] == forward["reference_shapes"]["retired"]


def test_a_retaining_regimes_value_leads_with_the_category_axis(
    forward: dict[str, Any],
) -> None:
    """The sharded axis is the leading axis of the retaining regime's value."""
    assert forward["value_shapes"]["working"][0] == _KIND_EXTENT


def test_the_sharded_solution_equals_the_single_device_solution(
    forward: dict[str, Any],
) -> None:
    """Placement partitions the solve; it never changes the values published."""
    assert forward["solution_matches_reference"] is True


def test_the_sharded_simulation_equals_the_single_device_simulation(
    forward: dict[str, Any],
) -> None:
    """The simulated frames agree with the unsharded reference model's."""
    assert forward["simulation_matches_reference"] is True


def test_reading_a_single_device_value_onto_a_mesh_is_a_copy_to_source_layout(
    forward: dict[str, Any],
) -> None:
    """The pruning regime's value reaches the sharded reader as a device copy."""
    assert forward["transfer_from_pruning_regime"] == "copy_to_source_layout"


def test_a_saved_sharded_result_reloads_unchanged(forward: dict[str, Any]) -> None:
    """Persisting and reloading the result reproduces the same frame."""
    assert forward["save_load_round_trips"] is True


def test_the_mirror_model_prunes_the_state_from_the_entering_regime(
    mirror: dict[str, Any],
) -> None:
    """A state only a later regime reads is dropped from the earlier one."""
    assert mirror["pruned_variables"]["working"] == ["kind"]


def test_the_mirror_models_reading_regime_takes_the_sharded_block(
    mirror: dict[str, Any],
) -> None:
    """The later regime that reads the state carries the device axis."""
    assert mirror["devices"]["retired"] == list(range(_KIND_EXTENT))


def test_the_mirror_models_entering_regime_runs_on_one_device(
    mirror: dict[str, Any],
) -> None:
    """The regime that only enters the state needs no axis of its own."""
    assert len(mirror["devices"]["working"]) == 1


def test_reading_a_sharded_value_onto_one_device_is_a_copy_to_source_layout(
    mirror: dict[str, Any],
) -> None:
    """An unsharded regime reads a sharded target's full value by device copy."""
    assert mirror["transfer_from_retaining_regime"] == "copy_to_source_layout"


def test_the_mirror_solution_equals_the_single_device_solution(
    mirror: dict[str, Any],
) -> None:
    """The mirror model's values agree with its unsharded reference."""
    assert mirror["solution_matches_reference"] is True


def test_the_mirror_simulation_equals_the_single_device_simulation(
    mirror: dict[str, Any],
) -> None:
    """The mirror model's simulated frames agree with its unsharded reference."""
    assert mirror["simulation_matches_reference"] is True


def test_subject_sharded_simulation_equals_the_single_device_simulation(
    subject_sharded: dict[str, Any],
) -> None:
    """Sharding forward subjects over every device leaves the frames unchanged."""
    assert subject_sharded["simulation_matches_reference"] is True


def test_a_sharded_state_every_regime_prunes_is_refused() -> None:
    """A state no regime reads defines no device axis anywhere, so it is refused."""
    with pytest.raises(ExecutionPlanningError, match="no regime retains"):
        build_unread_sharded_state()


def test_a_pruned_continuous_sharded_state_is_refused() -> None:
    """Continuous sharding still requires the state in every regime."""
    with pytest.raises(ExecutionPlanningError, match="retained in every regime"):
        build_pruned_continuous_sharded_state()
