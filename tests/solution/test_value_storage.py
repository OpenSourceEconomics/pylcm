"""`ExecutionConfig.value_storage="host"` keeps solved values in host memory.

The values, the reads and the simulated panel are those of the default device
storage, bit for bit. What changes is where a solved period's values live once no
period still to be solved reads them: in pinned host memory instead of on the
device. During the solve the device holds the resident window only, the values the
running period reads, while under device storage every solved period stays.

The models are the grid-search models of `test_period_target_value_reads`, whose
value arrays have two axes and so cannot be mistaken for a one-dimensional grid.
CPU devices have a pinned host memory kind too, so the copy path runs here as on
an accelerator.
"""

import dataclasses
import filecmp
import gc
import importlib
import json
import subprocess
import weakref
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Unpack, cast

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import lcm.model as model_module
from _lcm.engine import Regime
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueArtifactKind
from _lcm.execution.workspace_planning import WorkspaceBudgetError
from _lcm.persistence.solution import (
    BlockedSolutionArchive,
    DiskPolicyEntry,
    DiskValueEntry,
    StreamedSolutionArchive,
)
from _lcm.simulation import chunk_admission
from _lcm.simulation import simulate as simulate_module
from _lcm.simulation.memory import SimulationMemory
from _lcm.solution import backward_induction, block_major, host_values
from _lcm.solution.host_values import HostValueEntry, copy_values_to_host
from _lcm.solution.result_snapshot import own_value_store
from _lcm.typing import (
    ArtifactPayload,
    PeriodToRegimeToSimulationPolicy,
    SimulationPolicy,
)
from _lcm.utils.logging import get_logger
from lcm import ExecutionConfig, InvariantBlockSchedule, Model
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidValueFunctionError,
    ValueStorageError,
)
from lcm.persistence import SolveSnapshot, load_snapshot, load_solution
from lcm.solver_api import (
    SIMULATION_POLICY,
    ArtifactRef,
    SolutionResult,
    ValueStore,
)
from lcm.typing import Int1D, RegimeName
from tests import test_nan_diagnostics as nan_diagnostics
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.regime_building.test_collective_regime_simulate import (
    _DISSOLUTION_PARAMS,
    _make_dissolution_model,
)
from tests.simulation import test_type_grouped_simulation as life_cycle
from tests.solution import test_block_major_lifetime as lifetime
from tests.solution._callback_types import (
    ResidentInventoryKwargs,
    ResidentInventoryResult,
    RunPeriodKernelKwargs,
    RunPeriodKernelResult,
)
from tests.solution.test_invariant_blocking import _independent_types_model
from tests.solution.test_period_target_value_reads import (
    _gated_model,
    _GatedRegimeId,
    _RegimeId,
    _three_regime_model,
)
from tests.test_models import independent_types, n_nbegm_toy
from tests.test_models import stochastic as stochastic_model

_PARAMS = {"discount_factor": 0.95}

_MODELS = pytest.mark.parametrize(
    "build_model",
    [_three_regime_model, _gated_model],
    ids=["three-regime", "gated-edge"],
)


# What a read-through view reads and hands out, for wrappers of its reads.
type _Stored = jax.Array | SimulationPolicy | DiskValueEntry | DiskPolicyEntry


def _host() -> ExecutionConfig:
    return ExecutionConfig(value_storage="host")


def test_value_storage_defaults_to_the_device() -> None:
    """Without a choice, solved values stay on the device."""
    assert ExecutionConfig().value_storage == "device"


def test_value_storage_refuses_an_unknown_target() -> None:
    """Only the device and the host are storage targets."""
    with pytest.raises(BeartypeCallHintParamViolation, match="value_storage"):
        ExecutionConfig(value_storage="disk")  # ty: ignore[invalid-argument-type]


def test_value_storage_takes_an_existing_directory(tmp_path: Path) -> None:
    """A directory keeps each solve's values on disk."""
    config = ExecutionConfig(value_storage=tmp_path)

    assert config.value_storage == tmp_path
    assert config.value_storage_kind == "disk"


def test_value_storage_kind_names_device_and_host_storage() -> None:
    assert ExecutionConfig().value_storage_kind == "device"
    assert _host().value_storage_kind == "host"


def test_value_storage_refuses_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"value_storage.*directory"):
        ExecutionConfig(value_storage=tmp_path / "missing")


def test_value_storage_refuses_a_file(tmp_path: Path) -> None:
    file = tmp_path / "values.h5"
    file.touch()
    with pytest.raises(ValueError, match=r"value_storage.*directory"):
        ExecutionConfig(value_storage=file)


@_MODELS
def test_host_storage_publishes_the_device_storage_values(
    *, build_model: Callable[[ExecutionConfig], Model]
) -> None:
    """Every (period, regime) value is bitwise the one device storage publishes."""
    device = build_model(ExecutionConfig()).solve(params=_PARAMS, log_level="off")
    host = build_model(_host()).solve(params=_PARAMS, log_level="off")

    assert {p: set(v) for p, v in host.values.items()} == {
        p: set(v) for p, v in device.values.items()
    }
    for period, regime_to_value in device.values.items():
        for regime_name, value in regime_to_value.items():
            np.testing.assert_array_equal(
                np.asarray(host.value(period=period, regime=regime_name)),
                np.asarray(value),
                strict=True,
            )


_STOCHASTIC_INITIAL_CONDITIONS = MappingProxyType(
    {
        "health": jnp.array([1, 1, 0, 0], dtype=jnp.int32),
        "partner": jnp.array([0, 0, 1, 0], dtype=jnp.int32),
        "wealth": jnp.array([10.0, 50.0, 30.0, 80.0]),
        "age": jnp.array([40.0, 40.0, 40.0, 40.0]),
        "regime_id": jnp.array([stochastic_model.RegimeId.working_life] * 4),
    }
)


@pytest.mark.parametrize(
    "widths",
    [MappingProxyType({}), MappingProxyType({"subject": 1})],
    ids=["one-chunk", "one-subject-per-chunk"],
)
def test_every_storage_publishes_and_simulates_the_stochastic_model_alike(
    *, tmp_path: Path, widths: Mapping[str, int]
) -> None:
    """Issue test 1 on four periods, a stochastic state and regimes that end early.

    Every value and the panel are bitwise those of device storage, with host
    storage and with a directory.
    """
    params = stochastic_model.get_params(n_periods=4)
    values = []
    panels = []
    for storage in ("device", "host", tmp_path):
        model = stochastic_model.build_model(
            n_periods=4,
            execution_config=ExecutionConfig(
                value_storage=storage, axis_widths=widths, device_memory_bytes=None
            ),
        )
        solution = model.solve(params=params, log_level="off")
        values.append(
            {
                (period, regime): np.asarray(
                    solution.value(period=period, regime=regime)
                )
                for period, regime_to_value in solution.values.items()
                for regime in regime_to_value
            }
        )
        panels.append(
            model.simulate(
                params=params,
                initial_conditions=dict(_STOCHASTIC_INITIAL_CONDITIONS),
                solution=solution,
                seed=7,
                log_level="off",
            ).to_dataframe()
        )

    assert {regime for _, regime in values[0]} == {
        "working_life",
        "retirement",
        "dead",
    }
    for stored, panel in zip(values[1:], panels[1:], strict=True):
        assert stored.keys() == values[0].keys()
        for coordinate, value in values[0].items():
            np.testing.assert_array_equal(stored[coordinate], value, strict=True)
        assert panel.equals(panels[0])


@pytest.mark.parametrize("storage", ["host", "disk"])
def test_lookup_policy_reads_only_the_periods_it_needs(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: str
) -> None:
    """A lookup off the device answers as on it, copying only what it reads.

    Under both storages values are read through the read-through view, so the
    copies it makes are the values of the looked-up period and the next.
    """
    params = stochastic_model.get_params(n_periods=4)
    states = {
        "health": jnp.array([0, 1], dtype=jnp.int32),
        "partner": jnp.array([1, 0], dtype=jnp.int32),
        "wealth": jnp.array([20.0, 60.0]),
    }
    lookups = []
    copied: list[tuple[int, RegimeName]] = []
    read = host_values._RecentDeviceCopies.read

    # keyword-only-exempt: primary-argument=self
    def _recording_read(
        self: host_values._RecentDeviceCopies,
        *,
        period: int,
        regime: RegimeName,
        stored: _Stored,
    ) -> jax.Array | SimulationPolicy:
        if regime not in self._copies.get(period, {}):
            copied.append((period, regime))
        return read(self, period=period, regime=regime, stored=stored)

    for config in (
        ExecutionConfig(),
        ExecutionConfig(value_storage="host" if storage == "host" else tmp_path),
    ):
        model = stochastic_model.build_model(n_periods=4, execution_config=config)
        solution = model.solve(params=params, log_level="off")
        copied.clear()
        with monkeypatch.context() as patch:
            patch.setattr(host_values._RecentDeviceCopies, "read", _recording_read)
            lookups.append(
                model.lookup_policy(
                    params=params,
                    solution=solution,
                    period=1,
                    regime_name="working_life",
                    states=states,
                )
            )

    device, stored = lookups
    np.testing.assert_array_equal(
        np.asarray(stored.value), np.asarray(device.value), strict=True
    )
    assert stored.actions is not None
    assert device.actions is not None
    for name, action in device.actions.items():
        np.testing.assert_array_equal(
            np.asarray(stored.actions[name]), np.asarray(action), strict=True
        )
    assert copied
    assert {period for period, _ in copied} <= {1, 2}


@pytest.mark.parametrize("storage", ["host", "disk"])
def test_a_debug_solve_snapshot_holds_the_device_storage_values(
    *, tmp_path: Path, storage: str
) -> None:
    """The snapshot writer reads stored values back for its file."""
    params = stochastic_model.get_params(n_periods=4)
    snapshots = tmp_path / "snapshots"
    storage_directory = tmp_path / "storage"
    storage_directory.mkdir()
    device = stochastic_model.build_model(n_periods=4).solve(
        params=params, log_level="off"
    )
    stochastic_model.build_model(
        n_periods=4,
        execution_config=ExecutionConfig(
            value_storage="host" if storage == "host" else storage_directory
        ),
    ).solve(params=params, log_level="debug", log_path=snapshots)

    (directory,) = snapshots.glob("solve_snapshot_*/")
    snapshot = load_snapshot(directory)
    assert isinstance(snapshot, SolveSnapshot)
    assert snapshot.period_to_regime_to_V_arr is not None
    for period, regime_to_value in device.values.items():
        for regime, value in regime_to_value.items():
            np.testing.assert_array_equal(
                np.asarray(snapshot.period_to_regime_to_V_arr[period][regime]),
                np.asarray(value),
                strict=True,
            )


@_MODELS
def test_host_storage_reads_a_value_on_the_device(
    *, build_model: Callable[[ExecutionConfig], Model]
) -> None:
    """A read hands out a device array, as under device storage."""
    solution = build_model(_host()).solve(params=_PARAMS, log_level="off")
    for period, regime_to_value in solution.values.items():
        for regime_name in regime_to_value:
            value = solution.value(period=period, regime=regime_name)
            assert value.sharding.memory_kind == "device"


def _three_regime_initial_conditions() -> dict[str, jax.Array]:
    return {
        "regime_id": jnp.array([_RegimeId.a, _RegimeId.a, _RegimeId.b]),
        "age": jnp.array([0.0, 0.0, 1.0]),
        "wealth": jnp.array([1.0, 3.5, 2.0]),
        "health": jnp.array([0, 1, 1]),
    }


def _gated_initial_conditions() -> dict[str, jax.Array]:
    return {
        "regime_id": jnp.array([_GatedRegimeId.src, _GatedRegimeId.src]),
        "age": jnp.array([0.0, 0.0]),
        "wealth": jnp.array([1.0, 4.5]),
        "health": jnp.array([1, 0]),
    }


@pytest.mark.parametrize(
    ("build_model", "initial_conditions"),
    [
        (_three_regime_model, _three_regime_initial_conditions),
        (_gated_model, _gated_initial_conditions),
    ],
    ids=["three-regime", "gated-edge"],
)
def test_host_storage_simulates_the_device_storage_panel(
    *,
    build_model: Callable[[ExecutionConfig], Model],
    initial_conditions: Callable[[], dict[str, jax.Array]],
) -> None:
    """A panel simulated from host-stored values is the device-storage panel."""
    panels = [
        build_model(config)
        .simulate(
            params=_PARAMS,
            initial_conditions=initial_conditions(),
            seed=3,
            log_level="off",
        )
        .to_dataframe()
        for config in (ExecutionConfig(), _host())
    ]

    assert panels[0].equals(panels[1])


def _value_periods_on_the_device_at_each_kernel(
    *, monkeypatch: pytest.MonkeyPatch, model: Model
) -> list[tuple[int, frozenset[int]]]:
    """Return, per kernel start, its period and the periods of live device values.

    Each value is followed by a weak reference, not its buffer address: an
    accelerator's allocator hands a freed address to the next buffer.
    """
    published: list[tuple[int, weakref.ref[jax.Array]]] = []
    samples: list[tuple[int, frozenset[int]]] = []
    run_period_kernel = backward_induction._run_period_kernel

    def _run(**kwargs: Unpack[RunPeriodKernelKwargs]) -> RunPeriodKernelResult:
        gc.collect()
        samples.append(
            (
                kwargs["period"],
                frozenset(
                    period
                    for period, reference in published
                    if (value := reference()) is not None and not value.is_deleted()
                ),
            )
        )
        output = run_period_kernel(**kwargs)
        assert isinstance(output.value, jax.Array)
        published.append((kwargs["period"], weakref.ref(output.value)))
        return output

    with monkeypatch.context() as patch:
        patch.setattr(backward_induction, "_run_period_kernel", _run)
        model.solve(params=_PARAMS, log_level="off")
    return samples


def test_host_storage_keeps_only_the_resident_window_on_the_device(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kernel finds on the device only values of its period and the two after.

    Its own period holds the values its peers already solved, the next period the
    values it reads, and the one after may still be moving to the host. Under
    device storage the check fires: every solved period is resident.
    """
    host = _value_periods_on_the_device_at_each_kernel(
        monkeypatch=monkeypatch, model=_three_regime_model(_host())
    )
    device = _value_periods_on_the_device_at_each_kernel(
        monkeypatch=monkeypatch, model=_three_regime_model(ExecutionConfig())
    )

    def _within_window(*, resident: frozenset[int], period: int) -> bool:
        return resident <= {period, period + 1, period + 2}

    assert all(
        _within_window(resident=resident, period=period) for period, resident in host
    ), host
    assert not all(
        _within_window(resident=resident, period=period) for period, resident in device
    ), device


def _nbegm_model(execution_config: ExecutionConfig) -> Model:
    """The nested NB-EGM toy, whose simulation reads a published policy.

    It runs unbudgeted, as on a CPU by default: its budgeted simulation is refused
    on an accelerator whatever the storage.
    """
    return n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=3,
        execution_config=dataclasses.replace(
            execution_config, device_memory_bytes=None
        ),
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_host_storage_simulates_the_device_storage_policy_panel() -> None:
    """A panel read off host-stored policies is the device-storage panel."""
    initial_conditions = {
        "regime_id": jnp.array([n_nbegm_toy.RegimeId.alive] * 2),
        "age": jnp.array([20.0, 20.0]),
        "wealth": jnp.array([1.0, 7.5]),
        "illiquid": jnp.array([0.0, 4.0]),
    }
    panels = [
        _nbegm_model(config)
        .simulate(
            params=_PARAMS,
            initial_conditions=initial_conditions,
            seed=5,
            log_level="off",
        )
        .to_dataframe()
        for config in (ExecutionConfig(), _host())
    ]

    assert panels[0].equals(panels[1])


def _published_policies(solution: SolutionResult) -> dict[ArtifactRef, list[jax.Array]]:
    """Every simulation policy the result's public store hands out, by address."""
    return {
        ref: jax.tree.leaves(solution.replay_artifacts[ref])
        for ref in solution.replay_artifacts
        if ref.key == SIMULATION_POLICY
    }


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_host_storage_keeps_simulation_policies_in_pinned_host_memory() -> None:
    """Policies move like values; the public store hands out device copies.

    Each published policy reads back bitwise equal to the device-storage one.
    """
    device = _nbegm_model(ExecutionConfig()).solve(params=_PARAMS, log_level="off")
    host = _nbegm_model(_host()).solve(params=_PARAMS, log_level="off")
    engine_view = host._engine_view
    assert engine_view is not None
    stored = [
        leaf
        for regime_to_policy in host_values.stored_form(
            engine_view.simulation_policies
        ).values()
        for policy in regime_to_policy.values()
        for leaf in jax.tree.leaves(policy)
    ]

    assert stored
    assert all(host_values.is_host_value(leaf) for leaf in stored)
    expected = _published_policies(device)
    published = _published_policies(host)
    assert expected
    assert published.keys() == expected.keys()
    for ref, leaves in published.items():
        assert all(leaf.sharding.memory_kind == "device" for leaf in leaves)
        for leaf, reference in zip(leaves, expected[ref], strict=True):
            np.testing.assert_array_equal(
                np.asarray(leaf), np.asarray(reference), strict=True
            )


_NBEGM_INITIAL_CONDITIONS = MappingProxyType(
    {
        "regime_id": jnp.array([n_nbegm_toy.RegimeId.alive] * 2),
        "age": jnp.array([20.0, 20.0]),
        "wealth": jnp.array([1.0, 7.5]),
        "illiquid": jnp.array([0.0, 4.0]),
    }
)


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_disk_storage_writes_simulation_policies_to_the_archive(tmp_path: Path) -> None:
    """Policies are written with the values and read back from the file.

    Each published policy reads back on the device, bitwise equal to the
    device-storage one, and the saved archive simulates the device panel.
    """
    device_model = _nbegm_model(ExecutionConfig())
    device = device_model.solve(params=_PARAMS, log_level="off")
    disk = _nbegm_model(_disk(tmp_path)).solve(params=_PARAMS, log_level="off")
    stored = [
        entry
        for ref, entry in disk.replay_artifacts._entries.items()
        if ref.key == SIMULATION_POLICY
    ]
    saved = disk.save(path=tmp_path / "kept.h5")

    assert stored
    assert all(isinstance(entry, DiskPolicyEntry) for entry in stored)
    expected = _published_policies(device)
    published = _published_policies(disk)
    assert published.keys() == expected.keys()
    for ref, leaves in published.items():
        assert all(leaf.sharding.memory_kind == "device" for leaf in leaves)
        for leaf, reference in zip(leaves, expected[ref], strict=True):
            np.testing.assert_array_equal(
                np.asarray(leaf), np.asarray(reference), strict=True
            )
    panels = [
        device_model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_NBEGM_INITIAL_CONDITIONS),
            solution=solution,
            seed=5,
            log_level="off",
        ).to_dataframe()
        for solution in (device, load_solution(path=saved))
    ]
    assert panels[0].equals(panels[1])


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_disk_storage_simulates_the_device_storage_policy_panel(
    tmp_path: Path,
) -> None:
    initial_conditions = {
        "regime_id": jnp.array([n_nbegm_toy.RegimeId.alive] * 2),
        "age": jnp.array([20.0, 20.0]),
        "wealth": jnp.array([1.0, 7.5]),
        "illiquid": jnp.array([0.0, 4.0]),
    }
    panels = [
        _nbegm_model(config)
        .simulate(
            params=_PARAMS,
            initial_conditions=initial_conditions,
            seed=5,
            log_level="off",
        )
        .to_dataframe()
        for config in (ExecutionConfig(), _disk(tmp_path))
    ]

    assert panels[0].equals(panels[1])


def _stored_reads_while_simulating(
    *, monkeypatch: pytest.MonkeyPatch, model: Model
) -> int:
    """Count the copies from pinned memory and the policy reads from disk.

    Only those made while the engine simulates count.
    """
    reads = [0]
    simulating = [False]
    device_put = jax.device_put
    materialize = DiskPolicyEntry.materialize

    # keyword-only-exempt: library-callback=jax.device_put
    def _counting_device_put(
        x: ArtifactPayload,
        device: jax.sharding.Sharding | jax.Device | None = None,
        **kwargs: bool | jax.sharding.Sharding | None,
    ) -> ArtifactPayload:
        if simulating[0] and any(
            host_values.is_host_value(leaf)
            for leaf in jax.tree.leaves(x)
            if isinstance(leaf, jax.Array)
        ):
            reads[0] += 1
        return device_put(x, device, **kwargs)

    # keyword-only-exempt: primary-argument=self
    def _counting_materialize(
        self: DiskPolicyEntry, *, template: ArtifactPayload | None = None
    ) -> ArtifactPayload:
        if simulating[0]:
            reads[0] += 1
        return materialize(self, template=template)

    solution = model.solve(params=_PARAMS, log_level="off")
    with monkeypatch.context() as patch:
        patch.setattr(jax, "device_put", _counting_device_put)
        patch.setattr(DiskPolicyEntry, "materialize", _counting_materialize)
        patch.setattr(
            model_module,
            "simulate",
            _while_flagged(function=model_module.simulate, flag=simulating),
        )
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_NBEGM_INITIAL_CONDITIONS),
            solution=solution,
            seed=5,
            log_level="off",
        )
    return reads[0]


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize("storage", ["host", "disk"])
def test_several_chunks_read_the_stored_policies_a_single_chunk_reads(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: str
) -> None:
    """The chunks of a period share one device copy of each stored policy."""
    reads = [
        _stored_reads_while_simulating(
            monkeypatch=monkeypatch,
            model=_nbegm_model(
                ExecutionConfig(
                    value_storage="host" if storage == "host" else tmp_path,
                    axis_widths=widths,
                )
            ),
        )
        for widths in (MappingProxyType({}), _ONE_SUBJECT_PER_CHUNK)
    ]

    assert reads[0]
    assert reads[1] == reads[0]


def test_host_storage_keeps_dissolution_flags_in_host_memory() -> None:
    """A collective regime's dissolution flags are stored like its values."""
    model = _make_dissolution_model(_host())
    assert model.ages is not None
    result = backward_induction.solve(
        program_fingerprint="test_value_storage",
        flat_params=model._process_params(_DISSOLUTION_PARAMS),
        ages=model.ages,
        regimes=model._regimes,
        logger=get_logger(log_level="off"),
        enable_jit=model.enable_jit,
        execution=model._execution,
        retain_dissolution_flags=True,
    )
    flags = [
        flag
        for regime_to_flag in result.dissolution_flags.values()
        for flag in regime_to_flag.values()
    ]

    assert flags
    assert {flag.sharding.memory_kind for flag in flags} == {"pinned_host"}


def test_host_storage_routes_dissolutions_as_device_storage_does() -> None:
    """Flags read back from the host route the panel exactly as on the device."""
    panels = []
    for config in (ExecutionConfig(), _host()):
        model = _make_dissolution_model(config)
        initial_conditions = {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [model.regime_names_to_ids["married"]] * 3, dtype=jnp.int32
            ),
            "own_stakeholder": jnp.full(
                3, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        }
        solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
        panels.append(
            model.simulate(
                params=_DISSOLUTION_PARAMS,
                initial_conditions=initial_conditions,
                solution=solution,
                seed=0,
                log_level="off",
            ).to_dataframe()
        )

    assert panels[0].equals(panels[1])


# A budget no model here comes near, so the planner walks every cell's residency
# without refusing one.
_AMPLE_BUDGET = 2**32


def _value_periods_charged_at_each_cell(
    *, monkeypatch: pytest.MonkeyPatch, model: Model
) -> list[tuple[int, frozenset[int]]]:
    """Return, per planned cell, its period and the periods of the values it charges."""
    inventories: list[ResidentInventoryResult] = []
    resident_inventory_by_triple = backward_induction._resident_inventory_by_triple

    def _capture(
        **kwargs: Unpack[ResidentInventoryKwargs],
    ) -> ResidentInventoryResult:
        inventory = resident_inventory_by_triple(**kwargs)
        inventories.append(inventory)
        return inventory

    with monkeypatch.context() as patch:
        patch.setattr(backward_induction, "_resident_inventory_by_triple", _capture)
        model.solve(params=_PARAMS, log_level="off")
    return [
        (
            period,
            frozenset(
                artifact.period
                for group in position.live.values()
                for artifact in group
                if isinstance(artifact, ValueArtifactAddress)
                and artifact.kind is ValueArtifactKind.REGIME_VALUE
            ),
        )
        for inventory in inventories
        for (_regime, period, _core), position in inventory.items()
    ]


def test_host_storage_charges_the_planner_only_the_values_on_the_device(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission charges a cell its own period's values and the period it reads.

    Under device storage every solved period's values stay resident, and the
    planner charges them all.
    """
    host = _value_periods_charged_at_each_cell(
        monkeypatch=monkeypatch,
        model=_three_regime_model(
            ExecutionConfig(value_storage="host", device_memory_bytes=_AMPLE_BUDGET)
        ),
    )
    device = _value_periods_charged_at_each_cell(
        monkeypatch=monkeypatch,
        model=_three_regime_model(ExecutionConfig(device_memory_bytes=_AMPLE_BUDGET)),
    )

    assert host
    assert all(charged <= {period, period + 1} for period, charged in host), host
    assert not all(charged <= {period, period + 1} for period, charged in device), (
        device
    )


def _cell_refusal(*, value_storage: Literal["device", "host"]) -> str:
    """Return the message a cell's workspace refusal carries under one storage."""
    error = backward_induction._width_selection_failure(
        triple=("a", 0, "core"),
        resident_bytes=200,
        transfer_scratch_bytes={},
        budget_bytes=100,
        execution=ResolvedExecution(
            device_ids=(0,),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({}),
            device_memory_bytes=100,
            value_storage=value_storage,
        ),
        error=WorkspaceBudgetError("No width fits."),
    )
    return str(error)


def test_a_residency_refusal_under_device_storage_names_host_storage() -> None:
    """A cell refused over what the plan keeps resident points to host storage."""
    assert "value_storage='host'" in _cell_refusal(value_storage="device")


def test_a_residency_refusal_under_host_storage_does_not_name_it_again() -> None:
    """Host storage already keeps solved values off the device."""
    assert "value_storage" not in _cell_refusal(value_storage="host")


_STORED_VALUE = jnp.ones((4, 2))


def _host_store(*, budget_bytes: int | None) -> ValueStore:
    """A store of three periods' host-stored values under one budget."""
    return own_value_store(
        {
            period: copy_values_to_host(period=period, values={"a": _STORED_VALUE})
            for period in range(3)
        },
        host_budget_bytes=budget_bytes,
    )


def test_materializing_host_values_above_the_budget_is_refused() -> None:
    """Placing every stored value on the device at once must fit the budget."""
    with pytest.raises(ExecutionPlanningError, match="value_storage"):
        _host_store(budget_bytes=2 * _STORED_VALUE.nbytes).materialize()


def test_materializing_host_values_within_the_budget_reads_them_on_the_device() -> None:
    """A set that fits is placed on the device, one array per stored value."""
    snapshot = _host_store(budget_bytes=3 * _STORED_VALUE.nbytes).materialize()

    assert {
        value.sharding.memory_kind
        for regime_to_value in snapshot.values()
        for value in regime_to_value.values()
    } == {"device"}


def test_an_unbudgeted_host_store_materializes_without_admission() -> None:
    """Without a budget nothing is refused."""
    assert len(_host_store(budget_bytes=None).materialize()) == 3


def test_a_host_storage_solve_hands_its_budget_to_the_value_store() -> None:
    """The store admits a full materialization against the solve's budget."""
    model = _three_regime_model(
        ExecutionConfig(value_storage="host", device_memory_bytes=_AMPLE_BUDGET)
    )
    solution = model.solve(params=_PARAMS, log_level="off")
    entries = cast("ValueStore", solution.values)._entries.values()

    assert entries
    assert {
        entry.budget_bytes for entry in entries if isinstance(entry, HostValueEntry)
    } == {model._execution.device_memory_bytes}
    assert all(isinstance(entry, HostValueEntry) for entry in entries)


def test_a_period_copy_is_dispatched_before_the_next_period_computes(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy of period p's values goes out between p's kernels and p-1's.

    A copy dispatched before a kernel runs beside it; one dispatched after waits
    for it. The order of dispatch is therefore what decides whether the copy is
    hidden, and it is what this checks, not wall time.
    """
    events: list[tuple[str, int]] = []
    run_period_kernel = backward_induction._run_period_kernel
    copy_values_to_host = backward_induction.copy_values_to_host

    def _run(**kwargs: Unpack[RunPeriodKernelKwargs]) -> RunPeriodKernelResult:
        events.append(("kernel", kwargs["period"]))
        return run_period_kernel(**kwargs)

    def _copy(
        *, period: int, values: Mapping[RegimeName, jax.Array]
    ) -> MappingProxyType[RegimeName, jax.Array]:
        events.append(("copy", len(events)))
        return copy_values_to_host(period=period, values=values)

    with monkeypatch.context() as patch:
        patch.setattr(backward_induction, "_run_period_kernel", _run)
        patch.setattr(backward_induction, "copy_values_to_host", _copy)
        _three_regime_model(_host()).solve(params=_PARAMS, log_level="off")

    # Label each copy with the period of the kernels dispatched just before it.
    order: list[tuple[str, int]] = []
    for kind, payload in events:
        if kind == "kernel":
            order.append(("kernel", payload))
        else:
            order.append(("copy", next(p for k, p in reversed(order) if k == "kernel")))
    copies = [index for index, (kind, _) in enumerate(order) if kind == "copy"]
    assert copies
    for index in copies:
        period = order[index][1]
        before = [p for kind, p in order[:index] if kind == "kernel"]
        after = [p for kind, p in order[index + 1 :] if kind == "kernel"]
        assert period in before
        assert all(p < period for p in after), order
        assert not after or after[0] == period - 1, order


def test_the_transfer_of_a_period_starts_before_the_period_before_it_ends(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue test 3: a transfer recording its start and end, against compute ends.

    A transfer starts when its copy is dispatched and ends when the solve's wait
    for it returns; a period's computation ends when its kernels' results are
    ready. No wall time is asserted, only the order of these events.
    """
    events: list[tuple[str, int]] = []
    run_period_kernel = backward_induction._run_period_kernel
    copy_values_to_host = backward_induction.copy_values_to_host
    wait_for_host_copies = backward_induction.wait_for_host_copies

    def _run(**kwargs: Unpack[RunPeriodKernelKwargs]) -> RunPeriodKernelResult:
        result = jax.block_until_ready(run_period_kernel(**kwargs))
        events.append(("compute end", kwargs["period"]))
        return result

    def _copy(
        *, period: int, values: Mapping[RegimeName, jax.Array]
    ) -> MappingProxyType[RegimeName, jax.Array]:
        events.append(("transfer start", period))
        return copy_values_to_host(period=period, values=values)

    def _wait(*, copies: tuple[tuple[int, RegimeName, jax.Array], ...]) -> None:
        wait_for_host_copies(copies=copies)
        events.extend(("transfer end", period) for period in {c[0] for c in copies})

    with monkeypatch.context() as patch:
        patch.setattr(backward_induction, "_run_period_kernel", _run)
        patch.setattr(backward_induction, "copy_values_to_host", _copy)
        patch.setattr(backward_induction, "wait_for_host_copies", _wait)
        _three_regime_model(_host()).solve(params=_PARAMS, log_level="off")

    def first(*, kind: str, period: int) -> int:
        return events.index((kind, period))

    def last(*, kind: str, period: int) -> int:
        return len(events) - 1 - events[::-1].index((kind, period))

    transferred = sorted(
        {period for kind, period in events if kind == "transfer start"}
    )
    computed = {period for kind, period in events if kind == "compute end"}
    overlapping = [p for p in transferred if p - 1 in computed]
    assert overlapping
    for period in overlapping:
        start = first(kind="transfer start", period=period)
        assert start < last(kind="compute end", period=period - 1), events
        assert start < first(kind="transfer end", period=period)


def _blocked(*, value_storage: Literal["device", "host"]) -> ExecutionConfig:
    """Blocked by preference type, unbudgeted: a budgeted blocked simulation is
    refused on an accelerator whatever the storage."""
    return ExecutionConfig(
        invariant_block_widths={"pref_type": 1},
        value_storage=value_storage,
        device_memory_bytes=None,
    )


def test_a_blocked_host_storage_solve_publishes_the_blocked_device_values() -> None:
    """Assembled block values are stored and read back bit for bit."""
    solutions = [
        _independent_types_model(
            execution_config=_blocked(value_storage=storage)
        ).solve(params=independent_types.get_params(), log_level="off")
        for storage in ("device", "host")
    ]

    for period, regime_to_value in solutions[0].values.items():
        for regime_name, value in regime_to_value.items():
            np.testing.assert_array_equal(
                np.asarray(solutions[1].value(period=period, regime=regime_name)),
                np.asarray(value),
                strict=True,
            )


def test_a_blocked_host_storage_solve_simulates_the_blocked_device_panel() -> None:
    """Simulation from host-stored block values is the device-storage panel."""
    n_wealth = independent_types.N_WEALTH_POINTS
    n_types = len(independent_types.TYPE_PARAMS["weight"])
    types = np.repeat(np.arange(n_types), n_wealth)
    initial_conditions = {
        "regime_id": jnp.full(types.size, independent_types.RegimeId.working),
        "age": jnp.zeros(types.size),
        "wealth": jnp.asarray(np.tile(np.arange(n_wealth, dtype=float), n_types)),
        "pref_type": jnp.asarray(types, dtype=jnp.int32),
    }
    panels = [
        _independent_types_model(execution_config=_blocked(value_storage=storage))
        .simulate(
            params=independent_types.get_params(),
            initial_conditions=initial_conditions,
            seed=0,
            log_level="off",
        )
        .to_dataframe()
        for storage in ("device", "host")
    ]

    assert panels[0].equals(panels[1])


# keyword-only-exempt: primary-argument=directory
def _disk(
    directory: Path, *, device_memory_bytes: int | None = None
) -> ExecutionConfig:
    return ExecutionConfig(
        value_storage=directory,
        device_memory_bytes="device"
        if device_memory_bytes is None
        else device_memory_bytes,
    )


def _archives(directory: Path) -> list[str]:
    return sorted(child.name for child in directory.iterdir())


@_MODELS
def test_disk_storage_publishes_the_device_storage_values(
    *, tmp_path: Path, build_model: Callable[[ExecutionConfig], Model]
) -> None:
    """Every value read back from disk is bitwise the device-storage value."""
    device = build_model(ExecutionConfig()).solve(params=_PARAMS, log_level="off")
    disk = build_model(_disk(tmp_path)).solve(params=_PARAMS, log_level="off")

    assert {p: set(v) for p, v in disk.values.items()} == {
        p: set(v) for p, v in device.values.items()
    }
    for period, regime_to_value in device.values.items():
        for regime_name, value in regime_to_value.items():
            read = disk.value(period=period, regime=regime_name)
            assert read.sharding.memory_kind == "device"
            np.testing.assert_array_equal(
                np.asarray(read), np.asarray(value), strict=True
            )


def test_disk_storage_keeps_each_solve_in_a_file_of_its_own(tmp_path: Path) -> None:
    """Two results of one model read from two complete archives."""
    model = _three_regime_model(_disk(tmp_path))
    first = model.solve(params=_PARAMS, log_level="off")
    second = model.solve(params=_PARAMS, log_level="off")
    names = _archives(tmp_path)

    assert len(names) == 2
    assert all(n.startswith("solution-") and n.endswith(".h5") for n in names)
    assert {
        type(entry) for entry in cast("ValueStore", first.values)._entries.values()
    } == {DiskValueEntry}
    np.testing.assert_array_equal(
        np.asarray(first.value(period=0, regime="a")),
        np.asarray(second.value(period=0, regime="a")),
        strict=True,
    )


def test_a_disk_archive_is_deleted_with_the_last_reader_of_its_values(
    tmp_path: Path,
) -> None:
    solution = _three_regime_model(_disk(tmp_path)).solve(
        params=_PARAMS, log_level="off"
    )
    assert len(_archives(tmp_path)) == 1

    del solution
    gc.collect()

    assert _archives(tmp_path) == []


def test_saving_a_disk_result_copies_its_archive(tmp_path: Path) -> None:
    """The saved file is the archive the solve wrote, and it outlives the result."""
    storage = tmp_path / "storage"
    storage.mkdir()
    solution = _three_regime_model(_disk(storage)).solve(
        params=_PARAMS, log_level="off"
    )
    (archive,) = storage.iterdir()

    saved = solution.save(path=tmp_path / "kept.h5")
    value = np.asarray(solution.value(period=4, regime="c"))

    assert filecmp.cmp(saved, archive, shallow=False)
    del solution
    gc.collect()
    assert list(storage.iterdir()) == []
    np.testing.assert_array_equal(
        np.asarray(load_solution(path=saved).value(period=4, regime="c")),
        value,
        strict=True,
    )


@pytest.mark.parametrize(
    ("build_model", "initial_conditions"),
    [
        (_three_regime_model, _three_regime_initial_conditions),
        (_gated_model, _gated_initial_conditions),
    ],
    ids=["three-regime", "gated-edge"],
)
def test_disk_storage_simulates_the_device_storage_panel(
    *,
    tmp_path: Path,
    build_model: Callable[[ExecutionConfig], Model],
    initial_conditions: Callable[[], dict[str, jax.Array]],
) -> None:
    panels = [
        build_model(config)
        .simulate(
            params=_PARAMS,
            initial_conditions=initial_conditions(),
            seed=3,
            log_level="off",
        )
        .to_dataframe()
        for config in (ExecutionConfig(), _disk(tmp_path))
    ]

    assert panels[0].equals(panels[1])


def _host_value_periods_at_each_kernel(
    *, monkeypatch: pytest.MonkeyPatch, model: Model
) -> list[tuple[int, frozenset[int]]]:
    """Return, per kernel start, its period and the periods of live host copies."""
    period_of: dict[int, int] = {}
    samples: list[tuple[int, frozenset[int]]] = []
    run_period_kernel = backward_induction._run_period_kernel
    copy_values_to_host = backward_induction.copy_values_to_host

    def _copy(
        *, period: int, values: Mapping[RegimeName, jax.Array]
    ) -> MappingProxyType[RegimeName, jax.Array]:
        copies = copy_values_to_host(period=period, values=values)
        if values and all(value.ndim >= 2 for value in values.values()):
            period = current[-1]
            for copy in copies.values():
                period_of[copy.unsafe_buffer_pointer()] = period
        return copies

    current: list[int] = []

    def _run(**kwargs: Unpack[RunPeriodKernelKwargs]) -> RunPeriodKernelResult:
        current.append(kwargs["period"])
        samples.append(
            (
                kwargs["period"],
                frozenset(
                    period_of[pointer]
                    for array in jax.live_arrays()
                    if array.sharding.memory_kind == "pinned_host"
                    and (pointer := array.unsafe_buffer_pointer()) in period_of
                ),
            )
        )
        return run_period_kernel(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(backward_induction, "_run_period_kernel", _run)
        patch.setattr(backward_induction, "copy_values_to_host", _copy)
        model.solve(params=_PARAMS, log_level="off")
    return samples


def test_disk_storage_keeps_at_most_two_unwritten_periods_in_host_memory(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kernel finds host copies of the two periods after its own at most.

    Under host storage the check fires: every solved period stays in host memory.
    """
    disk = _host_value_periods_at_each_kernel(
        monkeypatch=monkeypatch, model=_three_regime_model(_disk(tmp_path))
    )
    host = _host_value_periods_at_each_kernel(
        monkeypatch=monkeypatch, model=_three_regime_model(_host())
    )

    def _bounded(*, resident: frozenset[int], period: int) -> bool:
        return resident <= {period + 1, period + 2}

    assert all(_bounded(resident=r, period=p) for p, r in disk), disk
    assert not all(_bounded(resident=r, period=p) for p, r in host), host


def _checkout_commit() -> str:
    """The commit of this checkout, marked when tracked files have changed."""

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no") != ""
    return git("rev-parse", "HEAD") + ("-dirty" if dirty else "")


def test_a_failed_disk_solve_leaves_an_incomplete_record(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record names how far the solve got; nothing else is left behind."""
    run_period_kernel = backward_induction._run_period_kernel

    def _fail_at_period_1(
        **kwargs: Unpack[RunPeriodKernelKwargs],
    ) -> RunPeriodKernelResult:
        if kwargs["period"] == 1:
            msg = "injected failure"
            raise RuntimeError(msg)
        return run_period_kernel(**kwargs)

    monkeypatch.setattr(backward_induction, "_run_period_kernel", _fail_at_period_1)
    with pytest.raises(RuntimeError, match="injected failure"):
        _three_regime_model(_disk(tmp_path)).solve(params=_PARAMS, log_level="off")

    (name,) = _archives(tmp_path)
    assert name.startswith("solution-")
    assert name.endswith(".incomplete")
    with h5py.File(tmp_path / name, "r") as file:
        record = json.loads(bytes(file["incomplete"][()]).decode())
    assert record["last_completed_period"] == 2
    assert record["dtype"] == (
        "float64" if jax.config.read("jax_enable_x64") else "float32"
    )
    assert set(record) == {
        "last_completed_period",
        "dtype",
        "model_fingerprint",
        "params_fingerprint",
        "pylcm_version",
        "pylcm_commit",
    }
    assert record["pylcm_commit"] == _checkout_commit()


@pytest.mark.parametrize("storage", ["host", "disk"])
def test_a_nan_is_diagnosed_without_reading_stored_values_back(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: str
) -> None:
    """The NaN check runs while the period's values are still on the device.

    The error and its breakdown are those of device storage, and no stored value
    is copied back onto the device to produce them.
    """
    with pytest.raises(InvalidValueFunctionError) as device:
        nan_diagnostics._build_nan_model()[0].solve(
            params=nan_diagnostics._build_nan_model()[1], log_level="off"
        )
    read_back: list[jax.Array | str] = []
    to_device = host_values.to_device
    materialize = DiskValueEntry.materialize

    def _counting_to_device(value: jax.Array) -> jax.Array:
        read_back.append(value)
        return to_device(value)

    # keyword-only-exempt: primary-argument=self
    def _counting_materialize(
        self: DiskValueEntry, *, template: ArtifactPayload | None = None
    ) -> jax.Array:
        read_back.append(self.dataset)
        return materialize(self, template=template)

    monkeypatch.setattr(host_values, "to_device", _counting_to_device)
    monkeypatch.setattr(DiskValueEntry, "materialize", _counting_materialize)
    model, params = nan_diagnostics._build_nan_model(
        _host() if storage == "host" else _disk(tmp_path)
    )
    with pytest.raises(InvalidValueFunctionError) as stored:
        model.solve(params=params, log_level="off")

    assert str(stored.value) == str(device.value)
    assert stored.value.diagnostics == device.value.diagnostics
    assert read_back == []


def test_a_failed_host_copy_names_its_period_regime_and_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first copy is of the last period; its failure says where it went."""

    def _fail(value: jax.Array) -> jax.Array:
        del value
        msg = "injected copy failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(host_values, "_to_pinned_host", _fail)
    with pytest.raises(
        ValueStorageError,
        match=r"period 4, regime '\w+' to pinned host memory",
    ) as raised:
        _three_regime_model(_host()).solve(params=_PARAMS, log_level="off")
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_a_failed_wait_for_a_host_copy_names_its_period_and_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(value: jax.Array) -> jax.Array:
        del value
        msg = "injected copy failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(host_values.jax, "block_until_ready", _fail)
    with pytest.raises(
        ValueStorageError, match=r"period 3, regime 'b' to pinned host memory"
    ):
        host_values.wait_for_host_copies(copies=((3, "b", _STORED_VALUE),))


def test_a_failed_disk_write_names_its_period_regime_and_archive(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error names the file, and the archive records the periods written."""
    write_value = StreamedSolutionArchive.write_value

    # keyword-only-exempt: primary-argument=self
    def _fail_at_period_1(
        self: StreamedSolutionArchive,
        *,
        period: int,
        regime: RegimeName,
        value: np.ndarray,
        sharding: jax.sharding.Sharding,
    ) -> DiskValueEntry:
        if period == 1:
            msg = "injected write failure"
            raise OSError(msg)
        return write_value(
            self, period=period, regime=regime, value=value, sharding=sharding
        )

    monkeypatch.setattr(StreamedSolutionArchive, "write_value", _fail_at_period_1)
    with pytest.raises(
        ValueStorageError,
        match=rf"period 1, regime '\w+' to {tmp_path}/solution-\S+\.h5\.partial",
    ) as raised:
        _three_regime_model(_disk(tmp_path)).solve(params=_PARAMS, log_level="off")
    assert isinstance(raised.value.__cause__, OSError)

    (name,) = _archives(tmp_path)
    assert name.endswith(".incomplete")
    with h5py.File(tmp_path / name, "r") as file:
        record = json.loads(bytes(file["incomplete"][()]).decode())
    assert record["last_completed_period"] == 2


def test_a_disk_value_entry_carries_the_solve_budget(tmp_path: Path) -> None:
    model = _three_regime_model(_disk(tmp_path, device_memory_bytes=_AMPLE_BUDGET))
    solution = model.solve(params=_PARAMS, log_level="off")
    entries = cast("ValueStore", solution.values)._entries.values()

    assert {
        entry.budget_bytes for entry in entries if isinstance(entry, DiskValueEntry)
    } == {model._execution.device_memory_bytes}


# One subject per chunk, so every population here runs in several chunks.
_ONE_SUBJECT_PER_CHUNK = MappingProxyType({"subject": 1})


def _while_flagged[**P, R](
    *, function: Callable[P, R], flag: list[bool]
) -> Callable[P, R]:
    """Wrap `function` so that `flag[0]` is true exactly while it runs."""

    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        flag[0] = True
        try:
            return function(*args, **kwargs)
        finally:
            flag[0] = False

    return _wrapped


def _value_copies_while_simulating(
    *,
    monkeypatch: pytest.MonkeyPatch,
    model: Model,
    initial_conditions: dict[str, jax.Array],
    outside_the_loop: bool = False,
) -> tuple[list[tuple[int, RegimeName]], pd.DataFrame]:
    """Return each stored value the simulation loop copies onto the device.

    Only copies made while the engine simulates count, or with `outside_the_loop`
    only those made before and after it, while the call prepares and plans.
    """
    copied: list[tuple[int, RegimeName]] = []
    simulating: list[bool] = [False]
    read = host_values._RecentDeviceCopies.read

    # keyword-only-exempt: primary-argument=self
    def _counting_read(
        self: host_values._RecentDeviceCopies,
        *,
        period: int,
        regime: RegimeName,
        stored: _Stored,
    ) -> jax.Array | SimulationPolicy:
        if simulating[0] != outside_the_loop and regime not in self._copies.get(
            period, {}
        ):
            copied.append((period, regime))
        return read(self, period=period, regime=regime, stored=stored)

    solution = model.solve(params=_PARAMS, log_level="off")
    with monkeypatch.context() as patch:
        patch.setattr(host_values._RecentDeviceCopies, "read", _counting_read)
        patch.setattr(
            model_module,
            "simulate",
            _while_flagged(function=model_module.simulate, flag=simulating),
        )
        panel = model.simulate(
            params=_PARAMS,
            initial_conditions=initial_conditions,
            solution=solution,
            seed=3,
            log_level="off",
        ).to_dataframe()
    return copied, panel


@pytest.mark.parametrize(
    ("build_model", "initial_conditions"),
    [
        (_three_regime_model, _three_regime_initial_conditions),
        (_gated_model, _gated_initial_conditions),
    ],
    ids=["three-regime", "gated-edge"],
)
def test_several_chunks_find_each_later_period_loaded_before_it_starts(
    *,
    monkeypatch: pytest.MonkeyPatch,
    build_model: Callable[[ExecutionConfig], Model],
    initial_conditions: Callable[[], dict[str, jax.Array]],
) -> None:
    """A period's reads are copied while the period before it runs.

    After the first period no chunk step copies a stored value itself: what it
    reads was copied before the period's first chunk started.
    """
    step_period: list[int | None] = [None]
    copied_in_a_step: list[tuple[int, int, RegimeName]] = []
    read = host_values._RecentDeviceCopies.read
    activate = simulate_module.activate_subject_membership

    # keyword-only-exempt: primary-argument=self
    def _recording_read(
        self: host_values._RecentDeviceCopies,
        *,
        period: int,
        regime: RegimeName,
        stored: _Stored,
    ) -> jax.Array | SimulationPolicy:
        current = step_period[0]
        if current is not None and regime not in self._copies.get(period, {}):
            copied_in_a_step.append((current, period, regime))
        return read(self, period=period, regime=regime, stored=stored)

    def _marking_activate(
        *,
        period: int,
        starting_periods: Int1D,
        initial_regime_ids: Int1D,
        initial_own_stakeholder: Int1D,
        regime_ids: Int1D,
        own_stakeholder: Int1D,
        memory: SimulationMemory | None = None,
    ) -> tuple[Int1D, Int1D]:
        step_period[0] = period
        return activate(
            period=period,
            starting_periods=starting_periods,
            initial_regime_ids=initial_regime_ids,
            initial_own_stakeholder=initial_own_stakeholder,
            regime_ids=regime_ids,
            own_stakeholder=own_stakeholder,
            memory=memory,
        )

    model = build_model(
        ExecutionConfig(
            value_storage="host",
            axis_widths=_ONE_SUBJECT_PER_CHUNK,
            device_memory_bytes=None,
        )
    )
    solution = model.solve(params=_PARAMS, log_level="off")
    monkeypatch.setattr(host_values._RecentDeviceCopies, "read", _recording_read)
    monkeypatch.setattr(
        simulate_module, "activate_subject_membership", _marking_activate
    )
    monkeypatch.setattr(
        simulate_module,
        "_prefetch_period_reads",
        _while_cleared(
            function=simulate_module._prefetch_period_reads, marker=step_period
        ),
        raising=False,
    )
    model.simulate(
        params=_PARAMS,
        initial_conditions=initial_conditions(),
        solution=solution,
        seed=3,
        log_level="off",
    )

    assert [copy for copy in copied_in_a_step if copy[0] > 0] == []


def _while_cleared[**P, R](
    *, function: Callable[P, R], marker: list[int | None]
) -> Callable[P, R]:
    """Wrap `function` so that `marker[0]` is `None` exactly while it runs."""

    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        saved, marker[0] = marker[0], None
        try:
            return function(*args, **kwargs)
        finally:
            marker[0] = saved

    return _wrapped


@pytest.mark.parametrize(
    ("build_model", "initial_conditions"),
    [
        (_three_regime_model, _three_regime_initial_conditions),
        (_gated_model, _gated_initial_conditions),
    ],
    ids=["three-regime", "gated-edge"],
)
def test_several_chunks_copy_the_stored_values_a_single_chunk_copies(
    *,
    monkeypatch: pytest.MonkeyPatch,
    build_model: Callable[[ExecutionConfig], Model],
    initial_conditions: Callable[[], dict[str, jax.Array]],
) -> None:
    """Chunks run each period in turn, so copies do not grow with the chunks."""
    copies = [
        sorted(
            _value_copies_while_simulating(
                monkeypatch=monkeypatch,
                model=build_model(
                    ExecutionConfig(
                        value_storage="host",
                        axis_widths=widths,
                        device_memory_bytes=None,
                    )
                ),
                initial_conditions=initial_conditions(),
            )[0]
        )
        for widths in (MappingProxyType({}), _ONE_SUBJECT_PER_CHUNK)
    ]

    assert copies[0]
    assert copies[1] == copies[0]


@pytest.mark.parametrize(
    ("build_model", "initial_conditions"),
    [
        (_three_regime_model, _three_regime_initial_conditions),
        (_gated_model, _gated_initial_conditions),
    ],
    ids=["three-regime", "gated-edge"],
)
def test_several_chunks_under_host_storage_simulate_the_device_panel(
    *,
    monkeypatch: pytest.MonkeyPatch,
    build_model: Callable[[ExecutionConfig], Model],
    initial_conditions: Callable[[], dict[str, jax.Array]],
) -> None:
    """Every subject keeps its keys and row: the panel is the chunked device one."""
    device = (
        build_model(
            ExecutionConfig(
                axis_widths=_ONE_SUBJECT_PER_CHUNK, device_memory_bytes=None
            )
        )
        .simulate(
            params=_PARAMS,
            initial_conditions=initial_conditions(),
            seed=3,
            log_level="off",
        )
        .to_dataframe()
    )
    _, host = _value_copies_while_simulating(
        monkeypatch=monkeypatch,
        model=build_model(
            ExecutionConfig(
                value_storage="host",
                axis_widths=_ONE_SUBJECT_PER_CHUNK,
                device_memory_bytes=None,
            )
        ),
        initial_conditions=initial_conditions(),
    )

    assert host.equals(device)


def test_several_chunks_under_disk_storage_simulate_the_device_panel(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = (
        _three_regime_model(
            ExecutionConfig(
                axis_widths=_ONE_SUBJECT_PER_CHUNK, device_memory_bytes=None
            )
        )
        .simulate(
            params=_PARAMS,
            initial_conditions=_three_regime_initial_conditions(),
            seed=3,
            log_level="off",
        )
        .to_dataframe()
    )
    copied, disk = _value_copies_while_simulating(
        monkeypatch=monkeypatch,
        model=_three_regime_model(
            ExecutionConfig(
                value_storage=tmp_path,
                axis_widths=_ONE_SUBJECT_PER_CHUNK,
                device_memory_bytes=None,
            )
        ),
        initial_conditions=_three_regime_initial_conditions(),
    )

    assert disk.equals(device)
    assert copied


def test_several_chunks_report_dissolutions_in_the_device_order() -> None:
    """Rerouted households are reported in the order device storage reports them."""
    results = []
    for storage in ("device", "host"):
        model = _make_dissolution_model(
            ExecutionConfig(
                value_storage=storage,
                axis_widths=_ONE_SUBJECT_PER_CHUNK,
                device_memory_bytes=None,
            )
        )
        initial_conditions = {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [model.regime_names_to_ids["married"]] * 3, dtype=jnp.int32
            ),
            "own_stakeholder": jnp.full(
                3, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        }
        results.append(
            model.simulate(
                params=_DISSOLUTION_PARAMS,
                initial_conditions=initial_conditions,
                seed=0,
                log_level="off",
            )
        )

    assert results[1].to_dataframe().equals(results[0].to_dataframe())
    assert results[1]._dissolution_fallbacks == results[0]._dissolution_fallbacks


def _budgeted(*, value_storage: Literal["device", "host"]) -> ExecutionConfig:
    return ExecutionConfig(
        value_storage=value_storage, device_memory_bytes=_AMPLE_BUDGET
    )


def _one_subject_per_budgeted_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the budgeted planner consider one subject per chunk only.

    A budget picks the widest admitted chunk, which for these small models is
    the whole population; only the candidate list forces several chunks.
    """
    monkeypatch.setattr(
        chunk_admission, "_independent_outer_candidates", lambda **_: (1,)
    )


def test_budgeted_chunks_copy_the_stored_values_a_single_chunk_copies(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a budget too, several chunks run period by period."""
    single, _ = _value_copies_while_simulating(
        monkeypatch=monkeypatch,
        model=_three_regime_model(_budgeted(value_storage="host")),
        initial_conditions=_three_regime_initial_conditions(),
    )
    _one_subject_per_budgeted_chunk(monkeypatch)
    several, _ = _value_copies_while_simulating(
        monkeypatch=monkeypatch,
        model=_three_regime_model(_budgeted(value_storage="host")),
        initial_conditions=_three_regime_initial_conditions(),
    )

    assert single
    assert sorted(several) == sorted(single)


def test_budgeted_chunks_under_host_storage_simulate_the_device_panel(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    _one_subject_per_budgeted_chunk(monkeypatch)
    device = (
        _three_regime_model(_budgeted(value_storage="device"))
        .simulate(
            params=_PARAMS,
            initial_conditions=_three_regime_initial_conditions(),
            seed=3,
            log_level="off",
        )
        .to_dataframe()
    )
    _, host = _value_copies_while_simulating(
        monkeypatch=monkeypatch,
        model=_three_regime_model(_budgeted(value_storage="host")),
        initial_conditions=_three_regime_initial_conditions(),
    )

    assert host.equals(device)


def test_the_planner_reserves_every_subjects_state_for_several_chunks() -> None:
    """The reservation covers each initial-condition column of every subject."""
    initial_conditions = _three_regime_initial_conditions()
    devices = (jax.devices()[0],)

    reserved = chunk_admission._population_carry_bytes(
        initial_conditions=initial_conditions, subject_devices=devices
    )

    columns = sum(int(column.nbytes) for column in initial_conditions.values())
    assert reserved[devices[0]] >= columns


def test_the_planner_reserves_what_one_period_copies_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reservation covers every read set the early copies bring in."""
    copied_early: list[int] = []
    prefetch = simulate_module._prefetch_period_reads

    def _measuring_prefetch(
        *,
        period: int,
        active_by_period: tuple[MappingProxyType[RegimeName, Regime], ...],
        values: Mapping[int, Mapping[RegimeName, jax.Array]],
        flags: Mapping[int, Mapping[RegimeName, jax.Array]],
        policies: PeriodToRegimeToSimulationPolicy,
    ) -> None:
        before = host_values.device_copies_of_view(view_id=id(values))
        prefetch(
            period=period,
            active_by_period=active_by_period,
            values=values,
            flags=flags,
            policies=policies,
        )
        after = host_values.device_copies_of_view(view_id=id(values))
        assert before is not None
        assert after is not None
        known = {id(copy) for copy in before}
        copied_early.append(
            sum(int(copy.nbytes) for copy in after if id(copy) not in known)
        )

    monkeypatch.setattr(simulate_module, "_prefetch_period_reads", _measuring_prefetch)
    model = _three_regime_model(
        ExecutionConfig(
            value_storage="host",
            axis_widths=_ONE_SUBJECT_PER_CHUNK,
            device_memory_bytes=None,
        )
    )
    solution = model.solve(params=_PARAMS, log_level="off")
    engine_view = solution._engine_view
    assert engine_view is not None
    model.simulate(
        params=_PARAMS,
        initial_conditions=_three_regime_initial_conditions(),
        solution=solution,
        seed=3,
        log_level="off",
    )
    reserved = chunk_admission._prefetch_bytes(
        regimes=model._regimes,
        values=host_values.planning_stand_in(engine_view.values),
        flags=host_values.planning_stand_in(engine_view.dissolution_flags),
        policies=None,
        n_periods=model.n_periods,
        devices=tuple(jax.devices()),
    )

    assert max(copied_early) > 0
    assert sum(reserved.values()) >= max(copied_early)


def test_a_budget_without_room_for_the_early_copies_is_refused(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    _one_subject_per_budgeted_chunk(monkeypatch)
    monkeypatch.setattr(
        chunk_admission,
        "_prefetch_bytes",
        lambda **_: dict.fromkeys(jax.devices(), _AMPLE_BUDGET),
    )
    model = _three_regime_model(_budgeted(value_storage="host"))
    solution = model.solve(params=_PARAMS, log_level="off")

    with pytest.raises(ExecutionPlanningError, match="copied early"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_three_regime_initial_conditions(),
            solution=solution,
            seed=3,
            log_level="off",
        )


def test_a_budget_without_room_for_every_subjects_state_is_refused(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal names the remedies that shrink the population's state."""
    _one_subject_per_budgeted_chunk(monkeypatch)
    monkeypatch.setattr(
        chunk_admission,
        "_population_carry_bytes",
        lambda **_: dict.fromkeys(jax.devices(), _AMPLE_BUDGET),
    )
    model = _three_regime_model(_budgeted(value_storage="host"))
    solution = model.solve(params=_PARAMS, log_level="off")

    with pytest.raises(ExecutionPlanningError, match="fewer subjects"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_three_regime_initial_conditions(),
            solution=solution,
            seed=3,
            log_level="off",
        )


def test_planning_a_budgeted_call_copies_no_more_values_than_an_unbudgeted_one(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measuring and profiling read the values' metadata, not copies of them."""
    _one_subject_per_budgeted_chunk(monkeypatch)
    copies = [
        sorted(
            _value_copies_while_simulating(
                monkeypatch=monkeypatch,
                model=_three_regime_model(config),
                initial_conditions=_three_regime_initial_conditions(),
                outside_the_loop=True,
            )[0]
        )
        for config in (
            ExecutionConfig(value_storage="host", device_memory_bytes=None),
            _budgeted(value_storage="host"),
        )
    ]

    assert copies[1] == copies[0]


def _life_cycle(
    *,
    schedule: InvariantBlockSchedule | None,
    value_storage: Literal["device", "host"] | Path,
) -> Model:
    """The typed life cycle, blocked by preference type when `schedule` is given."""
    model = life_cycle._model(typed_dead=True)
    config = dataclasses.replace(
        lifetime._config(schedule=schedule, subject_width=3),
        value_storage=value_storage,
    )
    return Model(
        edges=model.edges,
        regimes=model.user_regimes,
        ages=model.ages,
        regime_id_class=life_cycle._RegimeId,
        initial_nodes={0: "work"},
        execution_config=config,
    )


_LIFE_CYCLE_PARAMS = life_cycle._params(typed_dead=True)


def _life_cycle_values(solution: SolutionResult) -> dict[tuple[int, str], np.ndarray]:
    return {
        (period, regime): np.asarray(solution.value(period=period, regime=regime))
        for period, regime_to_value in solution.values.items()
        for regime in regime_to_value
    }


def test_block_major_host_storage_publishes_the_block_major_values() -> None:
    """Each code's values move to pinned host memory and assemble bit for bit.

    The reference is block-major on the device: block-major and period-major
    already differ in the last bits of some values.
    """
    device, host = (
        _life_cycle_values(
            _life_cycle(
                schedule=InvariantBlockSchedule.BLOCK_MAJOR, value_storage=storage
            ).solve(params=_LIFE_CYCLE_PARAMS, log_level="off")
        )
        for storage in ("device", "host")
    )

    assert host.keys() == device.keys()
    for coordinate, value in device.items():
        np.testing.assert_array_equal(host[coordinate], value, strict=True)


def test_block_major_host_storage_retains_pinned_blocks() -> None:
    """The retention keeps the blocks where the solve stored them."""
    solution = _life_cycle(
        schedule=InvariantBlockSchedule.BLOCK_MAJOR, value_storage="host"
    ).solve(params=_LIFE_CYCLE_PARAMS, log_level="off")
    engine_view = solution._engine_view
    assert engine_view is not None
    retained = engine_view.component_values
    assert retained is not None

    blocks = [
        block
        for code in retained.coverage.codes
        for block in retained.host_blocks(code=code).values()
    ]
    assert blocks
    assert {
        block.sharding.memory_kind for block in blocks if isinstance(block, jax.Array)
    } == {"pinned_host"}
    assert all(isinstance(block, jax.Array) for block in blocks)


@pytest.mark.parametrize("combined", [False, True], ids=["from-solution", "combined"])
def test_block_major_host_storage_simulates_the_block_major_panel(
    *, combined: bool
) -> None:
    """Reading each code back from pinned blocks gives the device-storage panel."""
    panels = []
    for value_storage in ("device", "host"):
        model = _life_cycle(
            schedule=InvariantBlockSchedule.BLOCK_MAJOR, value_storage=value_storage
        )
        solution = (
            None
            if combined
            else model.solve(params=_LIFE_CYCLE_PARAMS, log_level="off")
        )
        panels.append(
            model.simulate(
                params=_LIFE_CYCLE_PARAMS,
                initial_conditions=life_cycle._initial(),
                solution=solution,
                seed=0,
                log_level="off",
            ).to_dataframe()
        )

    assert panels[1].equals(panels[0])


# Blocking already moves some values from the unblocked solve, and block-major
# some from period-major, by at most this many units in the last place (measured
# at fp64 and fp32 on these models); storage adds nothing on top.
_BLOCKING_ULPS = 2


def _assert_within_blocking_ulps(
    *,
    actual: Mapping[tuple[int, str], np.ndarray],
    expected: Mapping[tuple[int, str], np.ndarray],
) -> None:
    """Every value is the expected one but for blocking's rounding."""
    assert actual.keys() == expected.keys()
    for coordinate, value in expected.items():
        finite = np.isfinite(value)
        np.testing.assert_array_equal(np.isfinite(actual[coordinate]), finite)
        np.testing.assert_array_equal(actual[coordinate][~finite], value[~finite])
        np.testing.assert_array_max_ulp(
            actual[coordinate][finite], value[finite], maxulp=_BLOCKING_ULPS
        )


def test_a_blocked_host_storage_solve_matches_the_unblocked_solve() -> None:
    """Issue test 9: blocked host storage against the unblocked solve."""
    params = independent_types.get_params()
    unblocked, blocked = (
        _life_cycle_values(
            _independent_types_model(execution_config=config).solve(
                params=params, log_level="off"
            )
        )
        for config in (ExecutionConfig(), _blocked(value_storage="host"))
    )

    _assert_within_blocking_ulps(actual=blocked, expected=unblocked)


@pytest.mark.parametrize("storage", ["device", "host", "disk"])
def test_block_major_in_every_storage_matches_period_major(
    *, tmp_path: Path, storage: str
) -> None:
    """Issue test 9: block-major with each storage against period-major."""
    period_major = _life_cycle_values(
        _life_cycle(schedule=None, value_storage="device").solve(
            params=_LIFE_CYCLE_PARAMS, log_level="off"
        )
    )
    block_major = _life_cycle_values(
        _life_cycle(
            schedule=InvariantBlockSchedule.BLOCK_MAJOR,
            value_storage=tmp_path
            if storage == "disk"
            else cast("Literal['device', 'host']", storage),
        ).solve(params=_LIFE_CYCLE_PARAMS, log_level="off")
    )

    _assert_within_blocking_ulps(actual=block_major, expected=period_major)


def test_component_jobs_refuse_value_storage_off_the_device(tmp_path: Path) -> None:
    """Jobs keep their own block files; a storage mode would be ignored."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    model = _life_cycle(
        schedule=InvariantBlockSchedule.BLOCK_MAJOR, value_storage="host"
    )

    with pytest.raises(ExecutionPlanningError, match="value_storage"):
        component_jobs.plan_component_jobs(
            model=model,
            params=_LIFE_CYCLE_PARAMS,
            directory=tmp_path / "jobs",
            n_jobs=2,
        )


def _block_major_disk(directory: Path) -> Model:
    return _life_cycle(
        schedule=InvariantBlockSchedule.BLOCK_MAJOR, value_storage=directory
    )


def test_block_major_disk_storage_publishes_the_block_major_values(
    tmp_path: Path,
) -> None:
    """Every code's blocks go to one archive and assemble bit for bit."""
    device = _life_cycle_values(
        _life_cycle(
            schedule=InvariantBlockSchedule.BLOCK_MAJOR, value_storage="device"
        ).solve(params=_LIFE_CYCLE_PARAMS, log_level="off")
    )
    solution = _block_major_disk(tmp_path).solve(
        params=_LIFE_CYCLE_PARAMS, log_level="off"
    )
    disk = _life_cycle_values(solution)
    (name,) = _archives(tmp_path)

    assert name.startswith("solution-")
    assert name.endswith(".h5")
    assert disk.keys() == device.keys()
    for coordinate, value in device.items():
        np.testing.assert_array_equal(disk[coordinate], value, strict=True)


def test_block_major_disk_storage_keeps_no_block_in_memory(tmp_path: Path) -> None:
    solution = _block_major_disk(tmp_path).solve(
        params=_LIFE_CYCLE_PARAMS, log_level="off"
    )
    engine_view = solution._engine_view
    assert engine_view is not None
    retained = engine_view.component_values
    assert retained is not None

    assert {
        type(block).__name__
        for blocks in retained._blocks.values()
        for block in blocks.values()
    } == {"_ArchivedBlock"}


def test_a_block_major_disk_archive_loads_saves_and_goes_with_its_result(
    tmp_path: Path,
) -> None:
    """The archive is an ordinary saved solution, copied by save and then deleted."""
    storage = tmp_path / "storage"
    storage.mkdir()
    solution = _block_major_disk(storage).solve(
        params=_LIFE_CYCLE_PARAMS, log_level="off"
    )
    (archive,) = storage.iterdir()
    values = _life_cycle_values(solution)

    saved = solution.save(path=tmp_path / "kept.h5")

    assert filecmp.cmp(saved, archive, shallow=False)
    del solution
    gc.collect()
    assert list(storage.iterdir()) == []
    reloaded = _life_cycle_values(load_solution(path=saved))
    for coordinate, value in values.items():
        np.testing.assert_array_equal(reloaded[coordinate], value, strict=True)


@pytest.mark.parametrize("combined", [False, True], ids=["from-solution", "combined"])
def test_block_major_disk_storage_simulates_the_block_major_panel(
    *, tmp_path: Path, combined: bool
) -> None:
    panels = []
    for value_storage in ("device", tmp_path):
        model = _life_cycle(
            schedule=InvariantBlockSchedule.BLOCK_MAJOR, value_storage=value_storage
        )
        solution = (
            None
            if combined
            else model.solve(params=_LIFE_CYCLE_PARAMS, log_level="off")
        )
        panels.append(
            model.simulate(
                params=_LIFE_CYCLE_PARAMS,
                initial_conditions=life_cycle._initial(),
                solution=solution,
                seed=0,
                log_level="off",
            ).to_dataframe()
        )

    assert panels[1].equals(panels[0])


def test_a_failed_block_major_disk_write_names_its_code_period_regime_and_archive(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # keyword-only-exempt: primary-argument=self
    def _fail(
        self: BlockedSolutionArchive,
        *,
        code: int,
        coordinate: tuple[int, RegimeName],
        block: np.ndarray,
        axis: int,
        full_shape: tuple[int, ...],
    ) -> None:
        del self, code, coordinate, block, axis, full_shape
        msg = "injected write failure"
        raise OSError(msg)

    monkeypatch.setattr(BlockedSolutionArchive, "write_block", _fail)
    with pytest.raises(
        ValueStorageError,
        match=(
            rf"code \d+ of period \d+, regime '\w+' to "
            rf"{tmp_path}/solution-\S+\.h5\.partial"
        ),
    ) as raised:
        _block_major_disk(tmp_path).solve(params=_LIFE_CYCLE_PARAMS, log_level="off")
    assert isinstance(raised.value.__cause__, OSError)


def test_a_failed_block_major_disk_solve_records_its_completed_codes(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solve_component = block_major.ComponentSchedule.solve_component
    solved: list[int] = []

    # keyword-only-exempt: primary-argument=self
    def _fail_on_the_second_code(
        self: block_major.ComponentSchedule, *, code: int
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, jax.Array]]:
        if solved:
            msg = "injected failure"
            raise RuntimeError(msg)
        solved.append(code)
        return solve_component(self, code=code)

    monkeypatch.setattr(
        block_major.ComponentSchedule, "solve_component", _fail_on_the_second_code
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        _block_major_disk(tmp_path).solve(params=_LIFE_CYCLE_PARAMS, log_level="off")

    (name,) = _archives(tmp_path)
    assert name.endswith(".incomplete")
    with h5py.File(tmp_path / name, "r") as file:
        record = json.loads(bytes(file["incomplete"][()]).decode())
    assert record["completed_codes"] == solved
