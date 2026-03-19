import json
import logging
from unittest.mock import patch

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    Regime,
    SimulateSnapshot,
    SolveSnapshot,
    categorical,
    load_snapshot,
)
from lcm.persistence import _get_platform
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    working: int
    retired: int


def _retired_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _build_tiny_model():
    def utility(consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
        return jnp.log(consumption + wealth)

    def next_wealth(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption

    def next_regime(period: int) -> ScalarInt:
        return jnp.where(period >= 1, 1, 0)

    working = Regime(
        transition=next_regime,
        states={"wealth": LinSpacedGrid(start=1, stop=5, n_points=3)},
        state_transitions={"wealth": next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1, n_points=3)},
        functions={"utility": utility},
        active=lambda age: age < 2,
    )
    retired = Regime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=1, stop=5, n_points=3)},
        functions={"utility": _retired_utility},
        active=lambda age: age >= 2,
    )
    ages = AgeGrid(start=0, stop=3, step="Y")
    model = Model(
        regimes={"working": working, "retired": retired},
        ages=ages,
        regime_id_class=_RegimeId,
        enable_jit=False,
    )
    params = {"discount_factor": 0.95}
    return model, params


@pytest.fixture
def model_and_params():
    return _build_tiny_model()


def _initial_conditions():
    return {
        "wealth": jnp.array([2.0, 3.0]),
        "age": jnp.array([0.0, 0.0]),
        "regime": jnp.array([_RegimeId.working] * 2),
    }


def test_solve_debug_persists_snapshot(tmp_path, model_and_params):
    model, params = model_and_params
    V_arr_dict = model.solve(params=params, log_level="debug", log_path=tmp_path)

    dirs = sorted(tmp_path.glob("solve_snapshot_*/"))
    assert len(dirs) == 1

    snapshot = load_snapshot(dirs[0])
    assert isinstance(snapshot, SolveSnapshot)
    assert snapshot.V_arr_dict is not None
    for period in V_arr_dict:
        for regime_name in V_arr_dict[period]:
            assert jnp.allclose(
                snapshot.V_arr_dict[period][regime_name],
                V_arr_dict[period][regime_name],
            )


def test_simulate_debug_persists_snapshot(tmp_path, model_and_params):
    model, params = model_and_params
    V_arr_dict = model.solve(params=params, log_level="off")

    model.simulate(
        params=params,
        initial_conditions=_initial_conditions(),
        V_arr_dict=V_arr_dict,
        log_level="debug",
        log_path=tmp_path,
    )

    dirs = sorted(tmp_path.glob("simulate_snapshot_*/"))
    assert len(dirs) == 1

    snapshot = load_snapshot(dirs[0])
    assert isinstance(snapshot, SimulateSnapshot)
    assert snapshot.result is not None


def test_simulate_with_solve_debug_persists_snapshot(tmp_path, model_and_params):
    model, params = model_and_params
    model.simulate(
        params=params,
        initial_conditions=_initial_conditions(),
        V_arr_dict=None,
        log_level="debug",
        log_path=tmp_path,
    )

    dirs = sorted(tmp_path.glob("simulate_snapshot_*/"))
    assert len(dirs) == 1

    snapshot = load_snapshot(dirs[0])
    assert isinstance(snapshot, SimulateSnapshot)
    assert snapshot.V_arr_dict is not None
    assert snapshot.result is not None


def test_solve_no_persistence_when_not_debug(tmp_path, model_and_params):
    model, params = model_and_params
    model.solve(params=params, log_level="progress", log_path=tmp_path)

    assert len(list(tmp_path.iterdir())) == 0


def test_simulate_no_persistence_when_not_debug(tmp_path, model_and_params):
    model, params = model_and_params
    V_arr_dict = model.solve(params=params, log_level="off")

    model.simulate(
        params=params,
        initial_conditions=_initial_conditions(),
        V_arr_dict=V_arr_dict,
        log_level="warning",
        log_path=tmp_path,
    )

    assert len(list(tmp_path.iterdir())) == 0


def test_debug_without_log_path_raises(model_and_params):
    model, params = model_and_params
    with pytest.raises(ValueError, match="log_path is required"):
        model.solve(params=params, log_level="debug")


def test_log_keep_n_latest_deletes_old_snapshots(tmp_path, model_and_params):
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


def test_snapshot_contains_environment_files(tmp_path, model_and_params):
    model, params = model_and_params
    model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = sorted(tmp_path.glob("solve_snapshot_*/"))[0]

    assert (snap_dir / "metadata.json").exists()
    assert (snap_dir / "REPRODUCE.md").exists()

    with (snap_dir / "metadata.json").open() as fh:
        metadata = json.load(fh)
    assert metadata["snapshot_type"] == "solve"
    assert "platform" in metadata
    assert "fields" in metadata

    reproduce = (snap_dir / "REPRODUCE.md").read_text()
    assert _get_platform() in reproduce
    assert "pixi install --frozen" in reproduce


def test_snapshot_contains_pixi_lock_and_pyproject(tmp_path, model_and_params):
    model, params = model_and_params
    model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = sorted(tmp_path.glob("solve_snapshot_*/"))[0]

    # These files should exist if the project root was found
    assert (snap_dir / "pyproject.toml").exists()
    assert (snap_dir / "pixi.lock").exists()


def test_snapshot_contains_h5_arrays(tmp_path, model_and_params):
    model, params = model_and_params
    model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = sorted(tmp_path.glob("solve_snapshot_*/"))[0]
    assert (snap_dir / "arrays.h5").exists()
    assert (snap_dir / "model.pkl").exists()
    assert (snap_dir / "params.pkl").exists()


def test_load_snapshot_warns_on_platform_mismatch(tmp_path, model_and_params, caplog):
    model, params = model_and_params
    model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = sorted(tmp_path.glob("solve_snapshot_*/"))[0]

    with (
        patch("lcm.persistence._get_platform", return_value="fake_arch-FakeOS"),
        caplog.at_level(logging.WARNING, logger="lcm.persistence"),
    ):
        load_snapshot(snap_dir)
    assert "environment may not match" in caplog.text


def test_load_snapshot_with_exclude(tmp_path, model_and_params):
    model, params = model_and_params
    model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = sorted(tmp_path.glob("solve_snapshot_*/"))[0]

    snapshot = load_snapshot(snap_dir, exclude=["V_arr_dict"])
    assert isinstance(snapshot, SolveSnapshot)
    assert snapshot.V_arr_dict is None
    assert snapshot.model is not None
    assert snapshot.params is not None


def test_solve_snapshot_round_trip(tmp_path, model_and_params):
    model, params = model_and_params
    V_arr_dict = model.solve(params=params, log_level="debug", log_path=tmp_path)

    snap_dir = sorted(tmp_path.glob("solve_snapshot_*/"))[0]
    snapshot = load_snapshot(snap_dir)

    # Verify the loaded model can re-solve
    assert isinstance(snapshot.model, Model)
    assert snapshot.params is not None
    V_arr_dict_2 = snapshot.model.solve(params=snapshot.params, log_level="off")
    for period in V_arr_dict:
        for regime_name in V_arr_dict[period]:
            assert jnp.allclose(
                V_arr_dict_2[period][regime_name],
                V_arr_dict[period][regime_name],
            )
