"""A gated edge's fallback read crossing regimes sharded on different states.

Sharding names each regime's device axis after the discrete state that defines
it, so two regimes that retain different states run on meshes with different
axis names and, when the device count allows, on different devices.  A gated
edge whose leg fallback is a `ProjectedRegimeValue` makes the source regime read
the fallback regime's stored value across exactly that boundary.

The required layout of such a read belongs to the value being read, not to the
core reading it: the consuming core's partition spec names its own axes and its
own rank, neither of which the other regime's value shares.  A value that is not
resident on the consuming core's mesh therefore arrives replicated on that mesh,
which the transfer catalogue classifies from the two concrete layouts.

Three placements of one topology are covered, each against the single-device
solution and simulation of the same model:

- `x` and `y` both sharded, so the two regimes hold different mesh axes on
  disjoint device blocks and the fallback read is a `cross_mesh_copy`;
- only `x` sharded, so the reading regime runs on one device and the read is a
  `copy_to_source_layout`;
- both regimes reading the same sharded state, so they share one mesh and the
  read stays `aligned_local`.

A fourth witness pins the refusal: two meshes that overlap without either
containing the other are served by no single operator, and planning says which
regime reads which and over what axis, rather than failing on a shape.

The witnesses run in child processes, because the placements under test only
exist where several devices do, and a topology is pinned before JAX initializes
a backend.
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

#: Devices the child process runs on: one full mesh per sharded regime.
_N_DEVICES = 4

#: Extent of every sharded grid, and so the mesh a retaining regime takes.
_EXTENT = 2

#: Devices the refusal witness runs on: two four-device meshes sharing two of them.
_OVERLAP_DEVICES = 6

_BETA = 0.9


@categorical(ordered=False)
class _RegimeId:
    solo: ScalarInt
    pair: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _Category:
    low: ScalarInt
    high: ScalarInt


_WEALTH = LinSpacedGrid(start=1.0, stop=4.0, n_points=3)
_CONSUMPTION = LinSpacedGrid(start=0.5, stop=2.0, n_points=3)
_HEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)

#: Wealth above which the gate opens and the closed branch lands in `high`.
_GATE_WEALTH_THRESHOLD = 2.0

_PARAMS = {
    "solo": {"koopmans_aggregator": {"discount_factor": _BETA}},
    "pair": {"koopmans_aggregator": {"discount_factor": _BETA}},
    "dead": {},
}

_INITIAL_CONDITIONS = {
    "age": jnp.zeros(4),
    "wealth": jnp.asarray([1.0, 2.5, 4.0, 2.5]),
    "health": jnp.asarray([0.0, 1.0, 0.0, 1.0]),
    "x": jnp.asarray([0, 1, 1, 0], dtype=jnp.int32),
    "y": jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
    "regime_id": jnp.full(4, _RegimeId.pair),
}


def _utility_of_x(*, consumption: ContinuousAction, x: DiscreteState) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.1 * x)


def _utility_of_y(
    *, consumption: ContinuousAction, y: DiscreteState, health: ContinuousState
) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.2 * y) + 0.05 * health


def _utility_of_x_and_health(
    *, consumption: ContinuousAction, x: DiscreteState, health: ContinuousState
) -> FloatND:
    return jnp.log(consumption) * (1.0 + 0.2 * x) + 0.05 * health


def _bequest_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def _solo_stays_before_age_two(age: FloatND) -> FloatND:
    return jnp.where(age < 2.0, 1.0, 0.0)


def _solo_leaves_from_age_two(age: FloatND) -> FloatND:
    return jnp.where(age < 2.0, 0.0, 1.0)


def _stay_before_age_one(age: FloatND) -> FloatND:
    return jnp.where(age < 1.0, 0.5, 0.0)


def _leave_from_age_one(age: FloatND) -> FloatND:
    return jnp.where(age < 1.0, 0.5, 1.0)


def _gate_open_above_the_middle(wealth: ContinuousState) -> BoolND:
    """Open where the landing regime's own wealth exceeds the grid's midpoint."""
    return wealth > _GATE_WEALTH_THRESHOLD


def _projected_wealth(wealth: ContinuousState) -> FloatND:
    return wealth


def _projected_x(wealth: ContinuousState) -> DiscreteState:
    """Land the closed branch in the high category above the wealth threshold."""
    return jnp.where(wealth > _GATE_WEALTH_THRESHOLD, 1, 0).astype(jnp.int32)


def build_model(
    *,
    devices: tuple[int, ...],
    sharded: tuple[str, ...],
    pair_reads: str = "y",
) -> Model:
    """Build the solo/pair/dead topology whose gated edge falls back into `solo`.

    `pair` leaves through a gated edge into `dead` whose one leg falls back into
    `solo` at a projection, so solving `pair` reads `solo`'s stored value. The
    gate opens on the landing wealth alone, so both branches are priced.

    Args:
        devices: Device ids the model is planned for.
        sharded: Names of the model-level discrete states given a device axis.
        pair_reads: Discrete state `pair`'s utility reads — `"y"` gives the two
            regimes different sharded states, `"x"` the same one.

    Returns:
        The model.

    """
    pair_utility = _utility_of_y if pair_reads == "y" else _utility_of_x_and_health
    solo = Regime(
        transition={
            "solo": MarkovTransition(_solo_stays_before_age_two),
            "dead": MarkovTransition(_solo_leaves_from_age_two),
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
            "pair": MarkovTransition(_stay_before_age_one),
            "dead": ValueDependentTransition(
                probability=MarkovTransition(_leave_from_age_one),
                gate=_gate_open_above_the_middle,
                routes={
                    "only": StakeholderRoute(
                        fallback=ProjectedRegimeValue(
                            regime="solo",
                            projection={
                                "wealth": _projected_wealth,
                                "x": _projected_x,
                            },
                        )
                    )
                },
            ),
        },
        active=lambda age: age < 2,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        functions={"utility": pair_utility},
        state_transitions={
            "wealth": _next_wealth,
            pair_reads: {"pair": fixed_transition(pair_reads)},
            "health": {"pair": fixed_transition("health")},
        },
    )
    dead = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": _WEALTH},
        functions={"utility": _bequest_utility},
    )
    return Model(
        regimes={"solo": solo, "pair": pair, "dead": dead},
        states={
            "x": DiscreteGrid(category_class=_Category),
            "y": DiscreteGrid(category_class=_Category),
            "health": _HEALTH,
        },
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(devices=devices, sharded_states=sharded),
    )


def _report(
    *, decimal: int, devices: tuple[int, ...], **variant: Any
) -> dict[str, Any]:
    """Solve and simulate one placement against its single-device reference."""
    from _lcm.solution import backward_induction  # noqa: PLC0415

    model = build_model(devices=devices, **variant)
    reference = build_model(devices=(0,), sharded=(), pair_reads=variant["pair_reads"])

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

    devices_by_regime: dict[str, list[int]] = {}
    value_shapes: dict[str, list[int]] = {}
    mismatches: list[str] = []
    for period, values in solution.values.items():
        for regime_name, value in values.items():
            devices_by_regime[regime_name] = sorted(
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

    reads = [
        {
            "source_regime": transfer.source.source_regime,
            "target_regime": transfer.target.regime,
            "channel": str(transfer.source.channel),
            "kind": str(transfer.kind),
        }
        for core in captured
        for transfer in core.input_transfer_plan
    ]
    fallback_reads = [
        read
        for read in reads
        if read["source_regime"] == "pair" and read["target_regime"] == "solo"
    ]
    fallback_kinds = sorted({read["kind"] for read in fallback_reads})
    fallback_channels = sorted({read["channel"] for read in fallback_reads})

    simulated = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions_read_by(model),
        solution=solution,
        log_level="off",
        seed=42,
    )
    expected_frame = reference.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions_read_by(reference),
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
        "value_shapes": value_shapes,
        "solution_matches_reference": not mismatches,
        "solution_mismatches": mismatches,
        "reads": reads,
        "fallback_kinds": fallback_kinds,
        "fallback_channels": fallback_channels,
        "simulation_matches_reference": not simulation_mismatch,
        "simulation_mismatch": simulation_mismatch,
    }


def report_cross_axis(*, decimal: int) -> dict[str, Any]:
    """Report the placement where the two regimes hold different mesh axes."""
    return _report(
        decimal=decimal,
        devices=tuple(range(_N_DEVICES)),
        sharded=("x", "y"),
        pair_reads="y",
    )


def report_only_source_sharded(*, decimal: int) -> dict[str, Any]:
    """Report the placement where only the fallback regime carries a mesh axis."""
    return _report(
        decimal=decimal,
        devices=tuple(range(_N_DEVICES)),
        sharded=("x",),
        pair_reads="y",
    )


def report_same_axis(*, decimal: int) -> dict[str, Any]:
    """Report the placement where both regimes are sharded on the same state."""
    return _report(
        decimal=decimal,
        devices=(0, 1),
        sharded=("x",),
        pair_reads="x",
    )


def _tolerance_argument() -> str:
    """Return the call argument handing a child process this run's tolerance."""
    return f"decimal={tests.conftest.DECIMAL_PRECISION!r}"


def report_overlapping_meshes() -> dict[str, Any]:
    """Report how a read between partially overlapping device meshes is refused."""
    import jax  # noqa: PLC0415

    from _lcm.solution import backward_induction  # noqa: PLC0415
    from lcm.exceptions import ExecutionPlanningError  # noqa: PLC0415

    devices = jax.devices()
    stored = jax.NamedSharding(
        mesh=jax.make_mesh((4,), ("x",), devices=tuple(devices[:4])),
        spec=jax.P("x"),
    )
    reading = jax.NamedSharding(
        mesh=jax.make_mesh((4,), ("y",), devices=tuple(devices[2:6])),
        spec=jax.P("y"),
    )
    try:
        backward_induction._resolve_value_transfer_layout(
            stored_sharding=stored,
            source_execution_sharding=reading,
            target_regime="solo",
            source_regime="pair",
        )
    except ExecutionPlanningError as refusal:
        return {"refused": True, "message": str(refusal)}
    return {"refused": False, "message": ""}


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
        "from tests.test_sharded_fallback_across_named_axes import "
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


@pytest.fixture(scope="module")
def cross_axis() -> dict[str, Any]:
    """Return the different-mesh-axis report from one four-device child process."""
    return _run_in_child_process(
        entry_point="report_cross_axis",
        n_devices=_N_DEVICES,
        arguments=_tolerance_argument(),
    )


@pytest.fixture(scope="module")
def only_source_sharded() -> dict[str, Any]:
    """Return the single-mesh report from one four-device child process."""
    return _run_in_child_process(
        entry_point="report_only_source_sharded",
        n_devices=_N_DEVICES,
        arguments=_tolerance_argument(),
    )


@pytest.fixture(scope="module")
def same_axis() -> dict[str, Any]:
    """Return the shared-mesh report from one two-device child process."""
    return _run_in_child_process(
        entry_point="report_same_axis",
        n_devices=_EXTENT,
        arguments=_tolerance_argument(),
    )


@pytest.fixture(scope="module")
def overlapping_meshes() -> dict[str, Any]:
    """Return the refusal report from one six-device child process."""
    return _run_in_child_process(
        entry_point="report_overlapping_meshes", n_devices=_OVERLAP_DEVICES
    )


def test_each_regime_retains_only_the_state_its_dag_reads(
    cross_axis: dict[str, Any],
) -> None:
    """The reading regime keeps `y` and drops `x`; the fallback regime the reverse."""
    assert cross_axis["pruned_variables"]["pair"] == ["x"]


def test_the_fallback_regime_drops_the_readers_sharded_state(
    cross_axis: dict[str, Any],
) -> None:
    """The fallback regime never reads `y`, so it carries no `y` axis."""
    assert cross_axis["pruned_variables"]["solo"] == ["health", "y"]


def test_the_two_sharded_regimes_take_disjoint_device_blocks(
    cross_axis: dict[str, Any],
) -> None:
    """Each regime's own mesh spans one device per category of its own state."""
    assert (cross_axis["devices"]["solo"], cross_axis["devices"]["pair"]) == (
        [0, 1],
        [2, 3],
    )


def test_the_read_under_test_is_the_gated_edges_reference_channel(
    cross_axis: dict[str, Any],
) -> None:
    """`pair` reads `solo` only through the gated edge's reference channel."""
    assert cross_axis["fallback_channels"] == ["edge_reference_regime_to_V_arr"]


def test_a_fallback_read_across_named_axes_is_a_cross_mesh_copy(
    cross_axis: dict[str, Any],
) -> None:
    """A value stored on another regime's mesh reaches the reader as a mesh copy."""
    assert cross_axis["fallback_kinds"] == ["cross_mesh_copy"]


def test_the_cross_axis_solution_equals_the_single_device_solution(
    cross_axis: dict[str, Any],
) -> None:
    """Placement partitions the solve; it never changes the values published."""
    assert cross_axis["solution_matches_reference"] is True


def test_the_cross_axis_simulation_equals_the_single_device_simulation(
    cross_axis: dict[str, Any],
) -> None:
    """The simulated frames agree with the unsharded reference model's."""
    assert cross_axis["simulation_matches_reference"] is True


def test_a_fallback_read_onto_one_device_is_a_copy_to_source_layout(
    only_source_sharded: dict[str, Any],
) -> None:
    """An unsharded reader collects the sharded fallback value by device copy."""
    assert only_source_sharded["fallback_kinds"] == ["copy_to_source_layout"]


def test_the_single_mesh_solution_equals_the_single_device_solution(
    only_source_sharded: dict[str, Any],
) -> None:
    """Sharding only the fallback regime leaves the published values unchanged."""
    assert only_source_sharded["solution_matches_reference"] is True


def test_the_single_mesh_simulation_equals_the_single_device_simulation(
    only_source_sharded: dict[str, Any],
) -> None:
    """Sharding only the fallback regime leaves the simulated frames unchanged."""
    assert only_source_sharded["simulation_matches_reference"] is True


def test_a_fallback_read_within_one_mesh_stays_aligned(
    same_axis: dict[str, Any],
) -> None:
    """Two regimes sharded on the same state share a mesh, so no transfer is made."""
    assert same_axis["fallback_kinds"] == ["aligned_local"]


def test_the_shared_mesh_solution_equals_the_single_device_solution(
    same_axis: dict[str, Any],
) -> None:
    """The same-state control publishes the unsharded reference's values."""
    assert same_axis["solution_matches_reference"] is True


def test_the_shared_mesh_simulation_equals_the_single_device_simulation(
    same_axis: dict[str, Any],
) -> None:
    """The same-state control simulates the unsharded reference's frames."""
    assert same_axis["simulation_matches_reference"] is True


def test_an_unservable_route_is_refused_while_planning(
    overlapping_meshes: dict[str, Any],
) -> None:
    """Meshes that overlap without containment admit no operator, so planning stops."""
    assert overlapping_meshes["refused"] is True


def test_a_refused_route_names_both_regimes(
    overlapping_meshes: dict[str, Any],
) -> None:
    """The refusal says which regime reads which, not only which shapes disagree."""
    message = overlapping_meshes["message"]
    assert "'pair'" in message
    assert "'solo'" in message


def test_a_refused_route_names_both_device_axes(
    overlapping_meshes: dict[str, Any],
) -> None:
    """The refusal names each side's mesh axis, which is what has to be changed."""
    message = overlapping_meshes["message"]
    assert "x=4" in message
    assert "y=4" in message


def _initial_conditions_read_by(model: Model) -> dict[str, Any]:
    """Return the initial conditions restricted to the states `model` simulates."""
    read = {"age", "regime_id"}.union(
        *(regime.simulation.state_names for regime in model._regimes.values())
    )
    return {name: value for name, value in _INITIAL_CONDITIONS.items() if name in read}
