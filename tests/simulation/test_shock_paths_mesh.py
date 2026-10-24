"""Given paths reach every subject on an eight-device subject mesh.

The cases skip unless the process has eight CPU host devices before JAX starts;
CI runs this file in its own fresh eight-device process.
"""

import gc
import weakref
from types import MappingProxyType
from typing import Unpack

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from _lcm.simulation import entry_allocations as entry_allocations_module
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.entry_inputs import SimulationEntryInputs
from _lcm.simulation.operand_placement import _OperandValue
from _lcm.simulation.residency import measure_buffer_footprint, resident_bytes_by_device
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.typing import PytreeValue
from lcm import (
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Periods,
    Regime,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.shocks import (
    BaseInnovations,
    GaussHermite,
    Normal,
    PhysicalValues,
    Product,
    ShockGroup,
)
from lcm.typing import FloatND, ReferenceName, ScalarInt
from tests.simulation._callback_types import OperandPlacement, RuntimeDispatch
from tests.simulation.test_aot_collective_and_gated import _capture_compiled_dispatches
from tests.simulation.test_current_exits import (
    _adjustment,
    _Binary,
    _DatedOverrideRegimes,
    _location_mean,
    _mapped_location,
    _shared_source_flow,
    _zero,
)
from tests.simulation.test_current_iid_overrides import _model as current_iid_model
from tests.simulation.test_current_iid_overrides import _Regimes as CurrentIIDRegimes
from tests.simulation.test_shock_paths import RegimeId, _model

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@pytest.mark.parametrize("budget", [None, 100_000_000])
def test_supplied_paths_on_subject_mesh(*, budget: int | None) -> None:
    devices = tuple(range(8))
    if len(jax.devices()) < len(devices):
        pytest.skip("Select this control in a fresh eight-device CPU process.")
    paths = np.tile(np.array([1.0, -1.0, 2.0]), (9, 1))
    paths[1::2] *= -1
    initial = {
        "age": jnp.array([20, 21, 20, 21, 20, 21, 20, 21, 20]),
        "shock": jnp.zeros(9),
        "regime_id": jnp.full(9, RegimeId.alive),
    }
    model = _model(
        execution=ExecutionConfig(
            devices=devices,
            simulation_sharding="subjects",
            axis_widths={"subject": 3},
            device_memory_bytes=budget,
        )
    )
    result = model.simulate(
        params={"discount_factor": 0.9},
        initial_conditions=initial,
        shock_paths={"shock": paths},
        seed=0,
        log_level="off",
    )
    summary = result.plan_summary
    assert summary is not None
    assert (summary.route, summary.subject_device_ids, summary.subject_backend) == (
        "subjects",
        devices,
        "cpu",
    )
    rows = (
        result.to_dataframe(use_categorical_codes=True)
        .reset_index()
        .query("regime_name == 'alive'")
    )
    expected = paths[rows.subject_id.to_numpy(), rows.period.to_numpy()]
    np.testing.assert_array_equal(rows.shock, expected)
    np.testing.assert_array_equal(rows.choice, expected)


@pytest.mark.parametrize("physical", [False, True])
def test_tagged_current_paths_keep_original_rows_on_subject_mesh(
    *, physical: bool
) -> None:
    """A budgeted subject mesh preserves tagged paths through late entry and padding."""
    devices = tuple(range(8))
    if jax.default_backend() != "cpu" or len(jax.devices()) < len(devices):
        pytest.skip("Select this control in a fresh eight-device CPU process.")
    starts = np.asarray([2, 0, 1, 0, 2, 1, 0, 1, 0], dtype=np.int32)
    paths = np.arange(starts.size * 3, dtype=jnp.asarray(0.0).dtype).reshape(-1, 3) + 2
    paths[1::2] *= -1
    assert starts.size % len(devices) != 0
    assert np.unique(paths).size == paths.size
    initial = {"period": starts, "regime_id": starts}
    declaration = current_iid_model()
    frames = []
    for sharded in (False, True):
        model = Model(
            regimes=declaration.user_regimes,
            shocks=declaration.shocks,
            n_periods=3,
            initial_nodes=declaration.initial_nodes,
            edges=declaration.edges,
            regime_id_class=CurrentIIDRegimes,
            execution_config=ExecutionConfig(
                devices=devices if sharded else (0,),
                simulation_sharding="subjects" if sharded else "legacy",
                axis_widths={"subject": 3},
                device_memory_bytes=100_000_000,
            ),
        )
        result = model.simulate(
            params={"discount_factor": 0.5, "risk": {"eps__sigma": 2.0}},
            initial_conditions=initial,
            shock_paths={
                "eps": PhysicalValues(values=paths)
                if physical
                else BaseInnovations(values=paths)
            },
            seed=91,
            log_level="off",
            runtime_checks=False,
        )
        if sharded:
            summary = result.plan_summary
            assert summary is not None
            assert (
                summary.route,
                summary.subject_device_ids,
                summary.subject_backend,
            ) == (
                "subjects",
                devices,
                "cpu",
            )
            assert summary.budget_mode == "budgeted"
            assert summary.effective_device_memory_bytes is not None
            assert 0 < summary.effective_device_memory_bytes <= 100_000_000
            assert sum(summary.admitted_chunk_widths) > starts.size
            assert all(
                width % len(devices) == 0 for width in summary.admitted_chunk_widths
            )
        frame = (
            result.to_dataframe(use_categorical_codes=True)
            .sort_values(["subject_id", "period"])
            .reset_index(drop=True)
        )
        pairs = np.asarray(
            [
                (row, period)
                for row, start in enumerate(starts)
                for period in range(start, 3)
            ]
        )
        np.testing.assert_array_equal(frame[["subject_id", "period"]], pairs)
        expected = paths if physical else 1 + 2 * paths
        values = expected[frame["subject_id"], frame["period"]]
        np.testing.assert_array_equal(frame["eps"], values)
        np.testing.assert_array_equal(frame["action"], values > 0)
        frames.append(frame)
    assert_frame_equal(frames[1], frames[0], check_exact=True)


def _exit_mesh_target_flow(*, shared_signal: FloatND, choice: ScalarInt) -> FloatND:
    return 20.0 + (2 * choice - 1) * shared_signal


def _exit_mesh_source_next(*, location: FloatND) -> FloatND:
    return location + 100


def _exit_mesh_target_next(*, location: FloatND, choice: ScalarInt) -> FloatND:
    return location + 1 + choice


def _exit_mesh_model(*, execution: ExecutionConfig) -> Model:
    """Reuse dated entry and mapped shared laws, with observable final transitions."""
    source_grid = LinSpacedGrid(start=0, stop=1, n_points=2)
    return Model(
        regimes={
            "start": Regime(
                functions={"utility": _zero},
                states={"location": source_grid},
                state_transitions={"location": _location_mean},
            ),
            "source": Regime(
                functions={"utility": _shared_source_flow},
                states={"location": source_grid},
                state_transitions={"location": _exit_mesh_source_next},
                shocks=("shared",),
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target",
                                projection={"location": _mapped_location},
                                utility_adjustment=_adjustment,
                            )
                        },
                        rule="unilateral",
                    )
                },
            ),
            "target": Regime(
                functions={"utility": _exit_mesh_target_flow},
                states={"location": LinSpacedGrid(start=10, stop=11, n_points=2)},
                actions={"choice": DiscreteGrid(category_class=_Binary)},
                state_transitions={"location": _exit_mesh_target_next},
                shocks=("shared",),
            ),
            "source_done": Regime(
                functions={"utility": _location_mean},
                states={"location": LinSpacedGrid(start=100, stop=101, n_points=2)},
            ),
            "target_done": Regime(
                functions={"utility": _location_mean},
                states={"location": LinSpacedGrid(start=11, stop=13, n_points=3)},
            ),
        },
        shocks={
            "shared": ShockGroup(
                shocks={"shared_signal": Normal(mu=_location_mean, sigma=1)},
                nodes=Product(shared_signal=GaussHermite(3)),
            )
        },
        n_periods=3,
        regime_id_class=_DatedOverrideRegimes,
        initial_nodes=InitialNodes(by_period={0: "start", 1: "source"}),
        edges={
            "start": {"source": Periods(values=(0,))},
            "source": {"source_done": Periods(values=(1,))},
            "target": {"target_done": Periods(values=(1,))},
        },
        execution_config=execution,
    )


def _capture_exit_mesh_placement(
    *, monkeypatch: pytest.MonkeyPatch, devices: tuple[int, ...]
) -> set[str]:
    seen: set[str] = set()
    native_dispatch = SimulationRuntime.dispatch

    def observe_native(
        self: SimulationRuntime, **call: Unpack[RuntimeDispatch]
    ) -> PytreeValue:
        output = native_dispatch(self, **call)
        family = call["program"].name
        if self.execution.simulation_sharding == "subjects" and family in {
            "simulate_exit_full",
            "simulate_exit_participating",
        }:
            leaves = jax.tree.leaves(output)
            assert leaves
            for leaf in leaves:
                assert isinstance(leaf, jax.Array)
                assert leaf.ndim > 0
                assert leaf.sharding.spec == jax.P("X")
                assert {device.id for device in leaf.sharding.device_set} == set(
                    devices
                )
            seen.add(family)
        return output

    monkeypatch.setattr(SimulationRuntime, "dispatch", observe_native)
    return seen


@pytest.mark.parametrize("physical", [False, True])
def test_tagged_exit_paths_keep_producing_law_on_subject_mesh(
    *, physical: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit movers keep their original innovation, final policy and sole transition."""
    devices = tuple(range(8))
    if jax.default_backend() != "cpu" or len(jax.devices()) < len(devices):
        pytest.skip("Select this control in a fresh eight-device CPU process.")
    starts = np.asarray([1, 0, 1, 0, 1, 0, 0, 1, 0], dtype=np.int32)
    location = np.arange(starts.size, dtype=jnp.asarray(0.0).dtype) % 2
    bases = np.arange(starts.size * 2, dtype=location.dtype).reshape(-1, 2) - 20.25
    paths = bases + location[:, None] if physical else bases
    expected_signal = location + 10 + bases[:, 1]
    expected_choice = expected_signal > 0
    assert np.any(expected_choice)
    assert np.any(~expected_choice)
    assert starts.size % len(devices) != 0
    compiled = _capture_compiled_dispatches(monkeypatch=monkeypatch)
    seen = _capture_exit_mesh_placement(monkeypatch=monkeypatch, devices=devices)
    frames = []
    for sharded in (False, True):
        model = _exit_mesh_model(
            execution=ExecutionConfig(
                devices=devices if sharded else (0,),
                simulation_sharding="subjects" if sharded else "legacy",
                axis_widths={"subject": 3},
                device_memory_bytes=100_000_000,
            )
        )
        result = model.simulate(
            params={"discount_factor": 0.0},
            initial_conditions={
                "period": starts,
                "regime_id": starts,
                "location": location,
            },
            shock_paths={
                "shared_signal": PhysicalValues(values=paths)
                if physical
                else BaseInnovations(values=paths)
            },
            seed=91,
            log_level="off",
            runtime_checks=False,
        )
        if sharded:
            summary = result.plan_summary
            assert summary is not None
            assert (
                summary.route,
                summary.subject_device_ids,
                summary.subject_backend,
            ) == ("subjects", devices, "cpu")
            assert summary.budget_mode == "budgeted"
            assert summary.effective_device_memory_bytes is not None
            assert 0 < summary.effective_device_memory_bytes <= 100_000_000
            assert sum(summary.admitted_chunk_widths) > starts.size
            assert all(
                width % len(devices) == 0 for width in summary.admitted_chunk_widths
            )
        frame = (
            result.to_dataframe(use_categorical_codes=True)
            .sort_values(["subject_id", "period"])
            .reset_index(drop=True)
        )
        pairs = np.asarray(
            [
                (row, period)
                for row, start in enumerate(starts)
                for period in range(start, 3)
            ]
        )
        np.testing.assert_array_equal(frame[["subject_id", "period"]], pairs)
        acting = frame.loc[frame["period"] == 1]
        final = frame.loc[frame["period"] == 2]
        np.testing.assert_array_equal(acting["regime_name"], "target")
        np.testing.assert_array_equal(acting["location"], location + 10)
        np.testing.assert_array_equal(acting["shared_signal"], expected_signal)
        np.testing.assert_array_equal(acting["choice"], expected_choice)
        np.testing.assert_array_equal(acting["value"], 19.75 + abs(expected_signal))
        expected_final = location + 11 + expected_choice
        np.testing.assert_array_equal(final["regime_name"], "target_done")
        np.testing.assert_array_equal(final["location"], expected_final)
        np.testing.assert_array_equal(final["value"], expected_final)
        frames.append(frame)
    assert compiled
    assert seen == {"simulate_exit_full", "simulate_exit_participating"}
    assert_frame_equal(frames[1], frames[0], check_exact=True)


@pytest.mark.parametrize("refuse_later", [False, True])
def test_query_parameter_copies_keep_sources_and_completed_views(
    *, refuse_later: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each admitted query copy stays charged through later placement and release."""
    if jax.default_backend() != "cpu" or len(jax.devices()) < 8:
        pytest.skip("Select this control in a fresh eight-device CPU process.")
    devices = tuple(jax.devices()[:8])
    model = _exit_mesh_model(
        execution=ExecutionConfig(
            devices=tuple(range(8)), device_memory_bytes=100_000_000
        )
    )
    source_device = model._regimes["source"].solution.placed_devices()[0]
    target_device = model._regimes["target"].solution.placed_devices()[0]
    assert source_device != target_device
    source = jax.device_put(jnp.asarray(0.5), source_device)
    target = jax.device_put(jnp.asarray(0.25), target_device)
    original = (source, target)
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=original),
        solution=None,
        model_roots=(),
        devices=devices,
        budget_bytes=100_000_000,
    )
    name = "koopmans_aggregator__discount_factor"
    flat = MappingProxyType(
        {
            "target": MappingProxyType({name: target}),
            "source": MappingProxyType({name: source}),
        }
    )
    copies = []
    native_place = entry_allocations_module.place_simulation_arguments

    def observe[Operand: _OperandValue](
        **kwargs: Unpack[OperandPlacement[Operand]],
    ) -> MappingProxyType[ReferenceName, Operand]:
        live = kwargs["live_footprint"]
        assert live is not None
        missing = resident_bytes_by_device(
            live=measure_buffer_footprint(tree=original),
            arguments=live,
            devices=devices,
        )
        assert all(value == 0 for value in missing.values())
        if copies:
            assert copies[0]() is not None
            missing = resident_bytes_by_device(
                live=measure_buffer_footprint(tree=copies[0]()),
                arguments=live,
                devices=devices,
            )
            assert all(value == 0 for value in missing.values())
        output = native_place(**kwargs)
        for leaf in jax.tree.leaves(output):
            if isinstance(leaf, jax.Array) and all(
                leaf is not item for item in original
            ):
                copies.append(weakref.ref(leaf))
                if refuse_later:
                    # The next native admission must refuse while this completed
                    # copy and both original source owners remain charged.
                    owner.budget_bytes = 1
        return output

    monkeypatch.setattr(entry_allocations_module, "place_simulation_arguments", observe)
    if refuse_later:
        with pytest.raises(ExecutionPlanningError, match="budget"):
            owner._place_query_solve_parameters(
                flat_params=flat, regimes=model._regimes
            )
    else:
        owner._place_query_solve_parameters(flat_params=flat, regimes=model._regimes)
    copied = owner.solve_parameter_views["source"]["target"][name]
    assert isinstance(copied, jax.Array)
    assert copied is not target
    assert copied.devices() == {source_device}
    np.testing.assert_array_equal(copied, 0.25)
    np.testing.assert_array_equal(target, 0.25)
    assert target.devices() == {target_device}
    assert owner.solve_parameter_views["source"]["source"][name] is source
    assert copies
    retained = weakref.ref(copied)
    owner.close()
    del copied
    gc.collect()
    assert retained() is None
