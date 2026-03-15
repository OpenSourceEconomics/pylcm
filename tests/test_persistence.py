import cloudpickle
import jax.numpy as jnp
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.persistence import load_solution, save_solution
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
def solved():
    model, params = _build_tiny_model()
    return model.solve(params, log_level="off")


def test_save_and_load_solution_roundtrip(tmp_path, solved):
    path = tmp_path / "solution.pkl"
    save_solution(V_arr_dict=solved, path=path)

    loaded = load_solution(path=path)

    assert set(loaded.keys()) == set(solved.keys())
    for period in solved:
        assert set(loaded[period].keys()) == set(solved[period].keys())
        for regime_name in solved[period]:
            assert jnp.allclose(
                loaded[period][regime_name], solved[period][regime_name]
            )


def test_save_solution_missing_parent_dir(tmp_path, solved):
    path = tmp_path / "nonexistent" / "solution.pkl"
    with pytest.raises(FileNotFoundError):
        save_solution(V_arr_dict=solved, path=path)


def test_load_solution_type_check(tmp_path):
    path = tmp_path / "wrong_type.pkl"
    with path.open("wb") as f:
        cloudpickle.dump({"not": "a MappingProxyType"}, f)

    with pytest.raises(TypeError, match="MappingProxyType"):
        load_solution(path=path)
