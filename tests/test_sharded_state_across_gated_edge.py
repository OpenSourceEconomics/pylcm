"""A gated edge whose projection reads a state the source shards.

Sharding a discrete state gives it a device axis, and a state whose law of motion
is the identity everywhere is co-mapped: its axis is sliced off every value array
before the continuation is read, so the interpolation places no coordinate on it.

A gated edge's projected references are not read off that array. A gate reference
or a leg fallback names ANOTHER regime's value at coordinates a projection
produces, so it is evaluated at the point the source lands on — and a projection
may map the co-mapped state onto the referenced regime's grid. It consumes the
state as a value rather than as an index, so the landing coordinate is supplied
even though the axis it would have indexed is gone.

Three placements of one topology are covered, each against the single-device
solution and simulation of the same model:

- `y` sharded on one device, which is the declaration alone with no device
  spread, so the co-map is what the witness isolates;
- `y` sharded over four devices, one per category, which is the same solve
  partitioned;
- the same edge with a projection reading only the target's wealth, the control
  that separates the projection's read of the sharded state from the sharding.

A fourth control drops the gate, so the sharded self-loop is an ordinary edge.
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
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.execution import ExecutionConfig
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)

_REPO_ROOT = Path(__file__).parent.parent

#: Extent of the sharded `level` grid, and so the mesh a retaining regime takes.
_LEVEL_EXTENT = 4

#: Devices the spread witness runs on: one full `level` mesh.
_N_DEVICES = _LEVEL_EXTENT

_BETA = 0.9

#: Wealth above which the gate opens and the projection lands in `high`.
_GATE_WEALTH_THRESHOLD = 2.0


@categorical(ordered=False)
class _RegimeId:
    solo: ScalarInt
    pair: ScalarInt
    mate: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _Category:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _Level:
    first: ScalarInt
    second: ScalarInt
    third: ScalarInt
    fourth: ScalarInt


_WEALTH = LinSpacedGrid(start=1.0, stop=4.0, n_points=3)
_CONSUMPTION = LinSpacedGrid(start=0.5, stop=2.0, n_points=3)

_PARAMS = {
    "solo": {"koopmans_aggregator": {"discount_factor": _BETA}},
    "pair": {"koopmans_aggregator": {"discount_factor": _BETA}},
    "mate": {"koopmans_aggregator": {"discount_factor": _BETA}},
    "dead": {},
}

_INITIAL_CONDITIONS = {
    "age": jnp.zeros(4),
    "wealth": jnp.asarray([1.0, 2.5, 4.0, 2.5]),
    "x": jnp.asarray([0, 1, 1, 0], dtype=jnp.int32),
    "level": jnp.asarray([0, 3, 1, 2], dtype=jnp.int32),
    "regime_id": jnp.full(4, _RegimeId.pair),
}


def _utility_of_x(*, consumption: ContinuousAction, x: DiscreteState) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.1 * x)


def _utility_of_level(
    *, consumption: ContinuousAction, level: DiscreteState
) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.2 * level)


def _bequest_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def _all_before_age_two(age: FloatND) -> FloatND:
    return jnp.where(age < 2.0, 1.0, 0.0)


def _none_before_age_two(age: FloatND) -> FloatND:
    return jnp.where(age < 2.0, 0.0, 1.0)


def _half_before_age_two(age: FloatND) -> FloatND:
    return jnp.where(age < 2.0, 0.5, 0.0)


def _gate_open_above_the_middle(wealth: ContinuousState) -> BoolND:
    """Open where the landing regime's own wealth exceeds the grid's midpoint."""
    return wealth > _GATE_WEALTH_THRESHOLD


def _projected_wealth(wealth: ContinuousState) -> FloatND:
    return wealth


def _projected_x_from_level(level: DiscreteState) -> DiscreteState:
    """Land the closed branch in the high category at the upper levels."""
    return jnp.where(level > 1, 1, 0).astype(jnp.int32)


def _projected_x_from_wealth(wealth: ContinuousState) -> DiscreteState:
    """Land the closed branch in the high category above the wealth threshold."""
    return jnp.where(wealth > _GATE_WEALTH_THRESHOLD, 1, 0).astype(jnp.int32)


def build_model(
    *,
    devices: tuple[int, ...],
    sharded: tuple[str, ...],
    projection_reads: str = "level",
    gated: bool = True,
) -> Model:
    """Build the solo/pair/mate/dead topology whose gated edge falls back into `solo`.

    `pair` leaves into `mate` through a gated edge whose one leg falls back into
    `solo` at a projection, so solving `pair` reads `solo`'s stored value at a
    point that projection names. Both `pair` and `mate` retain `level`, and
    `level` never transitions, so declaring it sharded co-maps it.

    Args:
        devices: Device ids the model is planned for.
        sharded: Names of the model-level discrete states given a device axis.
        projection_reads: What the fallback's `x` projection reads — `"level"`
            gives the projection the co-mapped state, `"wealth"` the target's
            own continuous state.
        gated: Whether `pair` leaves into `mate` through a gated edge or an
            ordinary Markov one.

    Returns:
        The model.

    """
    projected_x = (
        _projected_x_from_level
        if projection_reads == "level"
        else _projected_x_from_wealth
    )
    leaving = (
        ValueDependentTransition(
            probability=MarkovTransition(_half_before_age_two),
            gate=_gate_open_above_the_middle,
            routes={
                "only": StakeholderRoute(
                    fallback=ProjectedRegimeValue(
                        regime="solo",
                        projection={"wealth": _projected_wealth, "x": projected_x},
                    )
                )
            },
        )
        if gated
        else MarkovTransition(_half_before_age_two)
    )
    solo = Regime(
        transition={
            "solo": MarkovTransition(_all_before_age_two),
            "dead": MarkovTransition(_none_before_age_two),
        },
        active=lambda age: age < 3,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        functions={"utility": _utility_of_x},
        state_transitions={
            "wealth": _next_wealth,
            "x": {"solo": fixed_transition("x")},
        },
    )
    pair = Regime(
        transition={
            "pair": MarkovTransition(_half_before_age_two),
            "mate": leaving,
            "dead": MarkovTransition(_none_before_age_two),
        },
        active=lambda age: age < 2,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        functions={"utility": _utility_of_level},
        state_transitions={
            "wealth": _next_wealth,
            "level": {
                "pair": fixed_transition("level"),
                "mate": fixed_transition("level"),
            },
        },
    )
    mate = Regime(
        transition={
            "mate": MarkovTransition(_all_before_age_two),
            "dead": MarkovTransition(_none_before_age_two),
        },
        active=lambda age: age < 3,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        functions={"utility": _utility_of_level},
        state_transitions={
            "wealth": _next_wealth,
            "level": {"mate": fixed_transition("level")},
        },
    )
    dead = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": _WEALTH},
        functions={"utility": _bequest_utility},
    )
    return Model(
        regimes={"solo": solo, "pair": pair, "mate": mate, "dead": dead},
        states={
            "x": DiscreteGrid(category_class=_Category),
            "level": DiscreteGrid(category_class=_Level),
        },
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(devices=devices, sharded_states=sharded),
    )


def _report(
    *, decimal: int, devices: tuple[int, ...], **variant: Any
) -> dict[str, Any]:
    """Solve and simulate one placement against its single-device reference."""
    model = build_model(devices=devices, **variant)
    reference = build_model(
        devices=(0,),
        sharded=(),
        projection_reads=variant.get("projection_reads", "level"),
        gated=variant.get("gated", True),
    )

    solution = model.solve(params=_PARAMS, log_level="off")
    reference_solution = reference.solve(params=_PARAMS, log_level="off")

    devices_by_regime: dict[str, list[int]] = {}
    mismatches: list[str] = []
    for period, values in solution.values.items():
        for regime_name, value in values.items():
            devices_by_regime[regime_name] = sorted(
                device.id for device in value.sharding.device_set
            )
            expected = np.asarray(reference_solution.values[period][regime_name])
            try:
                np.testing.assert_array_almost_equal(
                    np.asarray(value), expected, decimal=decimal
                )
            except AssertionError as mismatch:
                mismatches.append(f"period {period}, {regime_name}: {mismatch}")

    simulated = model.simulate(
        params=_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        solution=solution,
        log_level="off",
        seed=42,
    )
    expected_frame = reference.simulate(
        params=_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        solution=reference_solution,
        log_level="off",
        seed=42,
    )
    tolerance = 10.0**-decimal
    try:
        pd.testing.assert_frame_equal(
            simulated.to_dataframe(use_labels=False),
            expected_frame.to_dataframe(use_labels=False),
            rtol=tolerance,
            atol=tolerance,
        )
        simulation_mismatch = ""
    except AssertionError as mismatch:
        simulation_mismatch = str(mismatch)

    return {
        "pruned_variables": {
            name: sorted(pruned) for name, pruned in model.pruned_variables.items()
        },
        "devices": devices_by_regime,
        "solution_matches_reference": not mismatches,
        "solution_mismatches": mismatches,
        "simulation_matches_reference": not simulation_mismatch,
        "simulation_mismatch": simulation_mismatch,
    }


def report_one_device(*, decimal: int) -> dict[str, Any]:
    """Report the declaration alone: `level` sharded with nothing to spread over."""
    return _report(decimal=decimal, devices=(0,), sharded=("level",))


def report_spread(*, decimal: int) -> dict[str, Any]:
    """Report the same solve partitioned one device per `level` category."""
    return _report(
        decimal=decimal, devices=tuple(range(_N_DEVICES)), sharded=("level",)
    )


def report_projection_reads_wealth(*, decimal: int) -> dict[str, Any]:
    """Report the control whose projection never reads the co-mapped state."""
    return _report(
        decimal=decimal,
        devices=tuple(range(_N_DEVICES)),
        sharded=("level",),
        projection_reads="wealth",
    )


def report_ungated(*, decimal: int) -> dict[str, Any]:
    """Report the control whose sharded self-loop carries no gate at all."""
    return _report(
        decimal=decimal,
        devices=tuple(range(_N_DEVICES)),
        sharded=("level",),
        gated=False,
    )


def _run_in_child_process(*, entry_point: str, n_devices: int) -> dict[str, Any]:
    """Run one module-level report function on `n_devices` and return its report.

    The child carries this run's float policy — `jax_enable_x64`, the matmul
    precision and the matching `DECIMAL_PRECISION` — because a topology pin
    needs a fresh process and a fresh process reads none of pytest's options.

    Args:
        entry_point: Name of a function in this module taking the tolerance in
            decimals and returning a JSON-serializable mapping.
        n_devices: Number of CPU devices to pin the child to.

    Returns:
        Dictionary of the report the child process produced.

    """
    code = (
        "import json, sys; import jax; "
        f"jax.config.update('jax_num_cpu_devices', {n_devices}); "
        "jax.config.update('jax_platform_name', 'cpu'); "
        f"jax.config.update('jax_enable_x64', {tests.conftest.X64_ENABLED!r}); "
        "jax.config.update('jax_default_matmul_precision', 'highest'); "
        "from tests.test_sharded_state_across_gated_edge import "
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
def one_device() -> dict[str, Any]:
    """Return the single-device report from one one-device child process."""
    return _run_in_child_process(entry_point="report_one_device", n_devices=1)


@pytest.fixture(scope="module")
def spread() -> dict[str, Any]:
    """Return the spread report from one four-device child process."""
    return _run_in_child_process(entry_point="report_spread", n_devices=_N_DEVICES)


@pytest.fixture(scope="module")
def projection_reads_wealth() -> dict[str, Any]:
    """Return the wealth-projection control from one four-device child process."""
    return _run_in_child_process(
        entry_point="report_projection_reads_wealth", n_devices=_N_DEVICES
    )


@pytest.fixture(scope="module")
def ungated() -> dict[str, Any]:
    """Return the ungated control from one four-device child process."""
    return _run_in_child_process(entry_point="report_ungated", n_devices=_N_DEVICES)


def test_the_regimes_reading_the_sharded_state_retain_it(
    spread: dict[str, Any],
) -> None:
    """Only the regimes whose DAG reads `level` keep it, so `solo` drops it."""
    assert spread["pruned_variables"]["solo"] == ["level"]


def test_the_retaining_regimes_span_one_device_per_category(
    spread: dict[str, Any],
) -> None:
    """Each regime retaining `level` takes a mesh one device wide per category."""
    assert (spread["devices"]["pair"], spread["devices"]["mate"]) == (
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    )


def test_the_single_device_solution_equals_the_unsharded_solution(
    one_device: dict[str, Any],
) -> None:
    """Declaring a state sharded never changes the values published."""
    assert one_device["solution_matches_reference"] is True


def test_the_single_device_simulation_equals_the_unsharded_simulation(
    one_device: dict[str, Any],
) -> None:
    """Declaring a state sharded never changes the simulated frame."""
    assert one_device["simulation_matches_reference"] is True


def test_the_spread_solution_equals_the_unsharded_solution(
    spread: dict[str, Any],
) -> None:
    """Placement partitions the solve; it never changes the values published."""
    assert spread["solution_matches_reference"] is True


def test_the_spread_simulation_equals_the_unsharded_simulation(
    spread: dict[str, Any],
) -> None:
    """The simulated frames agree with the unsharded reference model's."""
    assert spread["simulation_matches_reference"] is True


def test_a_projection_reading_only_wealth_equals_the_unsharded_solution(
    projection_reads_wealth: dict[str, Any],
) -> None:
    """A gated edge whose projection skips the sharded state publishes the same."""
    assert projection_reads_wealth["solution_matches_reference"] is True


def test_a_projection_reading_only_wealth_equals_the_unsharded_simulation(
    projection_reads_wealth: dict[str, Any],
) -> None:
    """A gated edge whose projection skips the sharded state simulates the same."""
    assert projection_reads_wealth["simulation_matches_reference"] is True


def test_an_ungated_sharded_loop_equals_the_unsharded_solution(
    ungated: dict[str, Any],
) -> None:
    """An ordinary edge out of a sharded regime publishes the reference's values."""
    assert ungated["solution_matches_reference"] is True


def test_an_ungated_sharded_loop_equals_the_unsharded_simulation(
    ungated: dict[str, Any],
) -> None:
    """An ordinary edge out of a sharded regime simulates the reference's frames."""
    assert ungated["simulation_matches_reference"] is True
