import json
import logging
from pathlib import Path
from types import MappingProxyType, ModuleType
from unittest.mock import patch

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbax.checkpoint._src.metadata import sharding as orbax_sharding

import lcm
from _lcm import variables as _variables
from _lcm.engine import PeriodRegimeSimulationData
from _lcm.persistence import snapshots as _snapshot_module
from _lcm.persistence.io import _get_platform
from lcm import (
    AgeGrid,
    DeterministicTransition,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    SimulateSnapshot,
    SolveSnapshot,
    Transition,
    categorical,
    load_snapshot,
)
from lcm.exceptions import SolutionIntegrityError
from lcm.persistence import load_legacy_solution, load_solution, save_solution
from lcm.regime import Regime as UserRegime
from lcm.result import SimulationResult as _PublicSimulationResult
from lcm.solver_api import SolutionResult, ValueStore
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt


def test_forward_refs_bound_after_import() -> None:
    """`Model` and `SimulationResult` are bound in `_lcm.persistence.snapshots`.

    The package claw rewrites their string annotations on `_save_simulate_snapshot`
    into runtime forward references resolved against that module's globals at
    call time. Missing the binding leaves those calls failing with
    `BeartypeCallHintForwardRefException`.
    """
    assert _snapshot_module.Model is lcm.Model
    assert _snapshot_module.SimulationResult is _PublicSimulationResult
    assert _variables.UserRegime is lcm.Regime


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt


def _retired_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _build_tiny_model(*, enable_jit: bool):
    def utility(*, consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
        return jnp.log(consumption + wealth)

    def next_wealth(
        *, wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption

    def next_regime(period: int) -> ScalarInt:
        return jnp.where(period >= 1, 1, 0)

    working = UserRegime(
        states={"wealth": LinSpacedGrid(start=1, stop=5, n_points=3)},
        state_transitions={"wealth": next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1, n_points=3)},
        functions={"utility": utility},
    )
    retired = UserRegime(
        states={"wealth": LinSpacedGrid(start=1, stop=5, n_points=3)},
        functions={"utility": _retired_utility},
    )
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    model = Model(
        regimes={"working": working, "retired": retired},
        ages=ages,
        edges={
            "working": Transition(
                targets={"working": 0, "retired": (0, 1)},
                law=DeterministicTransition(func=next_regime),
            )
        },
        regime_id_class=_RegimeId,
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        # The simulation starts at age zero; an empty later period is intentional.
        initial_nodes={0: "working"},
    )
    params = {"discount_factor": 0.95}
    return model, params


@pytest.mark.parametrize("enable_jit", [False, True])
def test_persistence_fixture_does_not_fill_an_unrequired_final_period(
    *, enable_jit: bool
) -> None:
    """Archive round trips use the declared entry domain, including its empty tail."""
    model, _params = _build_tiny_model(enable_jit=enable_jit)
    expected = frozenset(
        {(0, "working"), (1, "working"), (1, "retired"), (2, "retired")}
    )
    assert model.graph.initial_nodes == frozenset({(0, "working")})
    assert model.reachability.visited_nodes == expected
    assert model.reachability.nodes == expected


@pytest.mark.parametrize("enable_jit", [False, True])
def test_persistence_roundtrip_preserves_the_exact_sparse_domain(
    *, tmp_path: Path, enable_jit: bool
) -> None:
    """A saved and reloaded solution holds exactly the demanded pairs, same values."""
    model, params = _build_tiny_model(enable_jit=enable_jit)
    expected = {(0, "working"), (1, "working"), (1, "retired"), (2, "retired")}
    solution = model.solve(params=params, log_level="debug", log_path=tmp_path)
    path = tmp_path / "sparse-solution.lcm"
    save_solution(solution=solution, path=path)

    loaded = load_solution(path=path)

    for store in (solution.values, loaded.values):
        assert {
            (period, name) for period in store for name in store[period]
        } == expected
    for period, name in expected:
        np.testing.assert_array_equal(
            np.asarray(loaded.value(period=period, regime=name)),
            np.asarray(solution.value(period=period, regime=name)),
        )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_solve_publishes_only_periods_that_hold_a_demanded_node(
    *, enable_jit: bool
) -> None:
    """The unrequired final period is absent from the published value periods."""
    model, params = _build_tiny_model(enable_jit=enable_jit)

    solution = model.solve(params=params, log_level="off")

    assert set(solution.values) == {0, 1, 2}


def _initial_conditions():
    return {
        "wealth": jnp.array([2.0, 3.0]),
        "age": jnp.array([0.0, 0.0]),
        "regime_id": jnp.array([_RegimeId.working] * 2),
    }


def _save_cpu_checkpoint(
    *, tmp_path: Path, named: bool = False, subject_rows: np.ndarray | None = None
) -> tuple[Path, jax.Array]:
    """Save literal values through the public simulation checkpoint interface."""
    cpu = jax.local_devices(backend="cpu")[0]
    placement = cpu
    if named:
        placement = jax.sharding.NamedSharding(
            jax.sharding.Mesh(
                np.asarray(jax.local_devices(backend="cpu")), ("subject",)
            ),
            jax.sharding.PartitionSpec("subject"),
        )
    values = jax.device_put(np.array([-0.0, 3.125], dtype=np.float32), placement)
    model, _params = _build_tiny_model(enable_jit=False)
    raw = PeriodRegimeSimulationData(
        V_arr=values,
        actions=MappingProxyType({"consumption": jax.device_put(jnp.ones(2), cpu)}),
        states=MappingProxyType({"wealth": jax.device_put(jnp.array([2.0, 3.0]), cpu)}),
        in_regime=jax.device_put(jnp.array([True, True]), cpu),
        own_stakeholder=jax.device_put(jnp.array([-1, -1], dtype=jnp.int32), cpu),
        nested_policy_fallback=jax.device_put(jnp.array([False, False]), cpu),
    )
    result = _PublicSimulationResult(
        raw_results=MappingProxyType(
            {"working": MappingProxyType({0: raw}), "retired": MappingProxyType({})}
        ),
        regimes=model._regimes,
        flat_params=MappingProxyType({"working": MappingProxyType({})}),
        period_to_regime_to_V_arr=MappingProxyType(
            {0: MappingProxyType({"working": values})}
        ),
        ages=model.ages,
        simulation_output_dtypes={},
    )
    result._subject_rows = subject_rows
    return result.save(directory=tmp_path / "result"), values


@pytest.mark.parametrize(
    "subject_rows", [None, np.array([4, 7], dtype=np.int64)], ids=["all", "selected"]
)
def test_load_restores_the_subject_rows_of_the_saved_result(
    *, tmp_path: Path, subject_rows: np.ndarray | None
) -> None:
    """A loaded result names the same simulated rows as the result that was saved."""
    directory, _values = _save_cpu_checkpoint(
        tmp_path=tmp_path, subject_rows=subject_rows
    )

    loaded = _PublicSimulationResult.load(directory=directory)

    assert repr(loaded._subject_rows) == repr(subject_rows)


@pytest.mark.parametrize("solution_only", [False, True])
@pytest.mark.parametrize("named", [False, True])
def test_cpu_checkpoint_preserves_its_device_when_default_enumeration_excludes_cpu(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, solution_only: bool, named: bool
) -> None:
    """Restore CPU leaf bytes and placement independently of the default backend."""
    directory, values = _save_cpu_checkpoint(tmp_path=tmp_path, named=named)

    # Only Orbax's external default enumeration differs; actual CPU devices exist.
    orbax_jax = ModuleType("jax")
    orbax_jax.__dict__.update(vars(jax))
    monkeypatch.setattr(orbax_jax, "local_devices", list)
    monkeypatch.setattr(orbax_sharding, "jax", orbax_jax)
    if solution_only:
        loaded_values = _PublicSimulationResult.load_solution(directory=directory)
        restored = loaded_values[0]["working"]
    else:
        loaded = _PublicSimulationResult.load(directory=directory)
        restored = loaded.raw_results["working"][0].V_arr

    assert (
        restored.shape,
        restored.dtype,
        np.asarray(restored).tobytes(),
        restored.sharding,
    ) == (
        (2,),
        np.dtype("float32"),
        np.array([-0.0, 3.125], dtype=np.float32).tobytes(),
        values.sharding,
    )


@pytest.mark.parametrize("solution_only", [False, True])
@pytest.mark.parametrize("device_name", ["cpu:999", "cuda:999"])
def test_a_checkpoint_with_an_unavailable_saved_device_is_refused(
    *, tmp_path: Path, solution_only: bool, device_name: str
) -> None:
    """Refuse unavailable saved devices without gathering onto another device."""
    directory, _values = _save_cpu_checkpoint(tmp_path=tmp_path)
    path = directory / ("V_arr" if solution_only else "arrays") / "_sharding"
    shardings = json.loads(path.read_text())
    path.write_text(
        json.dumps(
            {
                key: json.dumps({**json.loads(value), "device_str": device_name})
                for key, value in shardings.items()
            }
        )
    )

    loader = (
        _PublicSimulationResult.load_solution
        if solution_only
        else _PublicSimulationResult.load
    )
    with pytest.raises(ValueError, match=r"unavailable|was not found"):
        loader(directory=directory)


@pytest.fixture
def model_and_params():
    return _build_tiny_model(enable_jit=False)


@pytest.fixture
def solved(model_and_params):
    model, params = model_and_params
    return model.solve(params=params, log_level="debug")


# -- save_solution / load_solution ---------------------------------------------------


def test_save_and_load_solution_roundtrip(*, tmp_path, solved):
    path = tmp_path / "solution.lcm"
    save_solution(solution=solved, path=path)

    loaded = load_solution(path=path)

    assert isinstance(loaded, SolutionResult)
    assert isinstance(loaded.values, ValueStore)
    assert set(loaded.values) == set(solved.values)
    for period in solved.values:
        assert set(loaded.values[period]) == set(solved.values[period])
        for regime_name in solved.values[period]:
            assert jnp.allclose(
                loaded.value(period=period, regime=regime_name),
                solved.value(period=period, regime=regime_name),
            )


def test_save_solution_missing_parent_dir(*, tmp_path, solved):
    path = tmp_path / "nonexistent" / "solution.lcm"
    with pytest.raises(FileNotFoundError):
        save_solution(solution=solved, path=path)


def test_legacy_value_only_archive_requires_the_explicit_migration_reader(
    *, tmp_path: Path, solved: SolutionResult
) -> None:
    """Keep value-only HDF5 readable only through its migration-specific API."""
    path = tmp_path / "legacy-solution.h5"
    with h5py.File(path, "w") as archive:
        for period in solved.values:
            for regime_name in solved.values[period]:
                archive.create_dataset(
                    f"{period}/{regime_name}/V_arr",
                    data=solved.value(period=period, regime=regime_name),
                )

    with pytest.raises(SolutionIntegrityError, match="manifest"):
        load_solution(path=path)

    migrated_values = load_legacy_solution(path=path)
    assert not isinstance(migrated_values, SolutionResult)
    assert set(migrated_values) == set(solved.values)
    for period in solved.values:
        assert set(migrated_values[period]) == set(solved.values[period])
        for regime_name in solved.values[period]:
            assert jnp.allclose(
                migrated_values[period][regime_name],
                solved.value(period=period, regime=regime_name),
            )


# -- debug snapshots ------------------------------------------------------------------


def test_solve_debug_persists_snapshot(*, tmp_path, model_and_params):
    model, params = model_and_params
    period_to_regime_to_V_arr = model.solve(
        params=params, log_level="debug", log_path=tmp_path
    )

    dirs = sorted(tmp_path.glob("solve_snapshot_*/"))
    assert len(dirs) == 1

    snapshot = load_snapshot(dirs[0])
    assert isinstance(snapshot, SolveSnapshot)
    assert snapshot.period_to_regime_to_V_arr is not None
    for period in period_to_regime_to_V_arr.values:
        for regime_name in period_to_regime_to_V_arr.values[period]:
            assert jnp.allclose(
                snapshot.period_to_regime_to_V_arr[period][regime_name],
                period_to_regime_to_V_arr.values[period][regime_name],
            )


def test_simulate_debug_persists_snapshot(*, tmp_path, model_and_params):
    model, params = model_and_params
    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    model.simulate(
        params=params,
        initial_conditions=_initial_conditions(),
        solution=period_to_regime_to_V_arr,
        log_level="debug",
        log_path=tmp_path,
    )

    dirs = sorted(tmp_path.glob("simulate_snapshot_*/"))
    assert len(dirs) == 1

    snapshot = load_snapshot(path=dirs[0])
    assert isinstance(snapshot, SimulateSnapshot)
    assert snapshot.result is not None


def test_simulate_debug_persists_snapshot_from_a_loaded_solution(
    *, tmp_path, model_and_params, solved
):
    """A simulation consuming a solution read back from disk still writes its snapshot.

    A loaded solution's value store reads lazily from the archive and holds handles
    the snapshot pickle cannot serialize; the snapshot keeps the value arrays in its
    own HDF5 file, so the consumed solution is dropped from the pickled result.
    """
    model, params = model_and_params
    archive = tmp_path / "solution.lcm"
    save_solution(solution=solved, path=archive)
    loaded = load_solution(path=archive)

    model.simulate(
        params=params,
        initial_conditions=_initial_conditions(),
        solution=loaded,
        log_level="debug",
        log_path=tmp_path / "snapshots",
    )

    dirs = sorted((tmp_path / "snapshots").glob("simulate_snapshot_*/"))
    assert len(dirs) == 1
    snapshot = load_snapshot(path=dirs[0])
    assert isinstance(snapshot, SimulateSnapshot)
    assert snapshot.result is not None
    assert snapshot.result.solution is None


def test_simulate_with_solve_debug_persists_snapshot(*, tmp_path, model_and_params):
    model, params = model_and_params
    model.simulate(
        params=params,
        initial_conditions=_initial_conditions(),
        log_level="debug",
        log_path=tmp_path,
    )

    dirs = sorted(tmp_path.glob("simulate_snapshot_*/"))
    assert len(dirs) == 1

    snapshot = load_snapshot(path=dirs[0])
    assert isinstance(snapshot, SimulateSnapshot)
    assert snapshot.period_to_regime_to_V_arr is not None
    assert snapshot.result is not None


def test_simulate_debug_persists_snapshot_with_compiled_runtime(tmp_path):
    """Debug snapshots serialize canonical regimes after compiled runtime dispatch."""
    model, params = _build_tiny_model(enable_jit=True)
    model.simulate(
        params=params,
        initial_conditions=_initial_conditions(),
        log_level="debug",
        log_path=tmp_path,
    )

    dirs = sorted(tmp_path.glob("simulate_snapshot_*/"))
    assert len(dirs) == 1
    snapshot = load_snapshot(path=dirs[0])
    assert isinstance(snapshot, SimulateSnapshot)
    assert snapshot.result is not None


@pytest.mark.parametrize("log_level", ["off", "warning", "progress"])
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_solve_no_persistence_when_not_debug(
    *, tmp_path, model_and_params, log_level, runtime_checks
):
    model, params = model_and_params
    model.solve(
        params=params,
        log_level=log_level,
        log_path=tmp_path,
        runtime_checks=runtime_checks,
    )

    assert len(list(tmp_path.iterdir())) == 0


def test_simulate_no_persistence_when_not_debug(*, tmp_path, model_and_params):
    model, params = model_and_params
    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    model.simulate(
        params=params,
        initial_conditions=_initial_conditions(),
        solution=period_to_regime_to_V_arr,
        log_level="warning",
        log_path=tmp_path,
    )

    assert len(list(tmp_path.iterdir())) == 0


def test_debug_without_log_path_solves(model_and_params):
    """`log_level="debug"` runs without `log_path` — it just writes no snapshot."""
    model, params = model_and_params
    model.solve(params=params, log_level="debug")


def test_log_keep_n_latest_deletes_old_snapshots(*, tmp_path, model_and_params):
    model, params = model_and_params

    for _ in range(5):
        model.solve(
            params=params, log_level="debug", log_path=tmp_path, log_keep_n_latest=3
        )

    dirs = sorted(tmp_path.glob("solve_snapshot_*/"))
    assert len(dirs) == 3
    assert dirs[0].name == "solve_snapshot_003"
    assert dirs[1].name == "solve_snapshot_004"
    assert dirs[2].name == "solve_snapshot_005"


def test_snapshot_contains_environment_files(*, tmp_path, model_and_params):
    model, params = model_and_params
    model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = min(tmp_path.glob("solve_snapshot_*/"))

    assert (snap_dir / "metadata.json").exists()
    assert (snap_dir / "REPRODUCE.md").exists()

    with (snap_dir / "metadata.json").open(encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["snapshot_type"] == "solve"
    assert "platform" in metadata
    assert "fields" in metadata

    reproduce = (snap_dir / "REPRODUCE.md").read_text(encoding="utf-8")
    assert _get_platform() in reproduce
    assert "pixi install --frozen" in reproduce


def test_snapshot_contains_pixi_lock_and_pyproject(*, tmp_path, model_and_params):
    model, params = model_and_params
    model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = min(tmp_path.glob("solve_snapshot_*/"))

    assert (snap_dir / "pyproject.toml").exists()
    assert (snap_dir / "pixi.lock").exists()


def test_snapshot_contains_h5_arrays(*, tmp_path, model_and_params):
    model, params = model_and_params
    model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = min(tmp_path.glob("solve_snapshot_*/"))
    assert (snap_dir / "arrays.h5").exists()
    assert (snap_dir / "model.pkl").exists()
    assert (snap_dir / "params.pkl").exists()


def test_load_snapshot_warns_on_platform_mismatch(
    *, tmp_path, model_and_params, caplog
):
    model, params = model_and_params
    model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = min(tmp_path.glob("solve_snapshot_*/"))

    with (
        patch("lcm.persistence._get_platform", return_value="fake_arch-FakeOS"),
        caplog.at_level(logging.WARNING, logger="lcm.persistence"),
    ):
        load_snapshot(path=snap_dir)
    assert "environment may not match" in caplog.text


def test_load_snapshot_with_exclude(*, tmp_path, model_and_params):
    model, params = model_and_params
    model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = min(tmp_path.glob("solve_snapshot_*/"))

    snapshot = load_snapshot(path=snap_dir, exclude=["period_to_regime_to_V_arr"])
    assert isinstance(snapshot, SolveSnapshot)
    assert snapshot.period_to_regime_to_V_arr is None
    assert snapshot.model is not None
    assert snapshot.params is not None


def test_solve_snapshot_round_trip(*, tmp_path, model_and_params):
    model, params = model_and_params
    period_to_regime_to_V_arr = model.solve(
        params=params, log_level="debug", log_path=tmp_path
    )

    snap_dir = min(tmp_path.glob("solve_snapshot_*/"))
    snapshot = load_snapshot(path=snap_dir)

    # Verify the loaded model can re-solve
    assert isinstance(snapshot.model, Model)
    assert snapshot.params is not None
    period_to_regime_to_V_arr_2 = snapshot.model.solve(
        params=snapshot.params, log_level="debug"
    ).values
    for period in period_to_regime_to_V_arr.values:
        for regime_name in period_to_regime_to_V_arr.values[period]:
            assert jnp.allclose(
                period_to_regime_to_V_arr_2[period][regime_name],
                period_to_regime_to_V_arr.values[period][regime_name],
            )
