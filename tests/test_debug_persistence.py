import jax.numpy as jnp
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, SimulationResult, categorical
from lcm.persistence import load_solution
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical
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
    initial_states = {
        "wealth": jnp.array([2.0, 3.0]),
        "age": jnp.array([0.0, 0.0]),
    }
    initial_regimes = ["working", "working"]
    return initial_states, initial_regimes


def test_solve_debug_persists_solution(tmp_path, model_and_params):
    model, params = model_and_params
    V_arr_dict = model.solve(params, debug=True, debug_path=tmp_path)

    files = list(tmp_path.glob("solution_*.pkl"))
    assert len(files) == 1

    loaded = load_solution(path=files[0])
    for period in V_arr_dict:
        for regime_name in V_arr_dict[period]:
            assert jnp.allclose(
                loaded[period][regime_name], V_arr_dict[period][regime_name]
            )


def test_simulate_debug_persists_result(tmp_path, model_and_params):
    model, params = model_and_params
    V_arr_dict = model.solve(params, debug=False)

    initial_states, initial_regimes = _initial_conditions()
    model.simulate(
        params,
        initial_states=initial_states,
        initial_regimes=initial_regimes,
        V_arr_dict=V_arr_dict,
        debug=True,
        debug_path=tmp_path,
    )

    files = list(tmp_path.glob("simulation_result_*.pkl"))
    assert len(files) == 1

    loaded = SimulationResult.from_pickle(files[0])
    assert loaded.n_subjects == 2


def test_solve_and_simulate_debug_persists_both(tmp_path, model_and_params):
    model, params = model_and_params
    initial_states, initial_regimes = _initial_conditions()

    model.solve_and_simulate(
        params,
        initial_states=initial_states,
        initial_regimes=initial_regimes,
        debug=True,
        debug_path=tmp_path,
    )

    assert len(list(tmp_path.glob("solution_*.pkl"))) == 1
    assert len(list(tmp_path.glob("simulation_result_*.pkl"))) == 1


def test_solve_debug_false_no_persistence(tmp_path, model_and_params):
    model, params = model_and_params
    model.solve(params, debug=False, debug_path=tmp_path)

    assert len(list(tmp_path.glob("*.pkl"))) == 0


def test_simulate_debug_false_no_persistence(tmp_path, model_and_params):
    model, params = model_and_params
    V_arr_dict = model.solve(params, debug=False)

    initial_states, initial_regimes = _initial_conditions()
    model.simulate(
        params,
        initial_states=initial_states,
        initial_regimes=initial_regimes,
        V_arr_dict=V_arr_dict,
        debug=False,
        debug_path=tmp_path,
    )

    assert len(list(tmp_path.glob("*.pkl"))) == 0


def test_keep_n_latest_deletes_old_files(tmp_path, model_and_params):
    model, params = model_and_params

    for _ in range(5):
        model.solve(params, debug=True, debug_path=tmp_path, keep_n_latest=3)

    files = sorted(tmp_path.glob("solution_*.pkl"))
    assert len(files) == 3
    # The remaining files should be the 3 most recent (003, 004, 005)
    assert files[0].name == "solution_003.pkl"
    assert files[1].name == "solution_004.pkl"
    assert files[2].name == "solution_005.pkl"
