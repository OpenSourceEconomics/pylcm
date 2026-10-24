"""Rank actual finite replay tuples against their current callable floor."""

import functools
import importlib
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.params.edges import regime_kernel_params
from _lcm.simulation import policy_programs
from _lcm.solution.feasibility import BoundedResult
from _lcm.typing import PytreeValue
from lcm import Model, NestedConsumptionSavingsRegime, post_decision_lower_bound
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import SIMULATION_POLICY, SolutionResult
from lcm.typing import (
    ActionName,
    BoolND,
    ContinuousState,
    Float1D,
    FloatND,
    IntND,
    RegimeName,
    StateName,
    UserFunction,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution import test_nested_dynamic_borrowing as original

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_replay = importlib.import_module("_lcm.simulation.simulate")
_PARAMS = {"discount_factor": 0.0}


def _service(*, new_illiquid: ContinuousState) -> FloatND:
    return new_illiquid * jnp.log(1.5)


def _interior_floor(*, illiquid: ContinuousState) -> FloatND:
    return 0.5 * illiquid * (1.0 - illiquid)


def _higher_floor(*, illiquid: ContinuousState) -> FloatND:
    return illiquid * (1.0 - illiquid)


def _model(*, lower: UserFunction) -> Model:
    """Keep the finite toy's grids and budget beside an interior current floor."""
    seed = original._model(family="nnbegm", lower=0.0)
    child = seed.user_regimes["alive"]
    assert isinstance(child, NestedConsumptionSavingsRegime)
    alive = child.replace(
        functions={**child.functions, "service": _service},
        constraints={
            "borrowing": post_decision_lower_bound(margin=child.liquid, lower=lower)
        },
    )
    return Model(
        regimes={**seed.user_regimes, "alive": alive},
        ages=seed.ages,
        edges=seed.edges,
        initial_nodes={10: "alive"},
        regime_id_class=original.RegimeId,
    )


def test_public_finite_replay_masks_the_higher_value_illegal_keeper() -> None:
    """Choose the legal adjustment tuple over a keeper below the current floor."""
    model = _model(lower=_interior_floor)
    solution = model.solve(params=_PARAMS, log_level="off")
    policy = solution.replay_artifacts.project(SIMULATION_POLICY)[0]["alive"]
    assert isinstance(policy, NNBEGMSimPolicy)
    assert policy.state_names == ("liquid", "illiquid")
    assert policy.n_keeper_candidates == 1
    states = {"liquid": jnp.asarray([0.0]), "illiquid": jnp.asarray([0.5])}
    regime = model._regimes["alive"]
    flat = regime_kernel_params(
        model._process_params(_PARAMS, runtime_checks=True), regime_name="alive"
    )
    assert model.ages is not None
    age = model.ages.values[0]
    inner, outer, live, represented = _replay._prepare_nnbegm_candidate_bank(
        regime=regime,
        sim_policy=policy,
        states=states,
        flat_params=flat,
        period=0,
        age=age,
    )
    np.testing.assert_array_equal(live[:, 0], [True, True, True, False])
    np.testing.assert_array_equal(represented[:, 0], [True, True, True, False])
    np.testing.assert_array_almost_equal(
        inner[:3, 0], [1.25, 1.0, 1.0], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        outer[:3, 0], [0.0, -0.5, 0.5], decimal=DECIMAL_PRECISION
    )
    scored = _replay._score_nnbegm_candidate_bank(
        optimal_actions=MappingProxyType({}),
        regime=regime,
        sim_policy=policy,
        candidate_inner=inner,
        candidate_outer=outer,
        flat_params=flat,
        period=0,
        age=age,
        canonical_states=states,
        action_names=regime.simulation.action_names,
        next_regime_to_V_arr=MappingProxyType(dict(solution.values[1])),
    )
    assert type(scored) is tuple
    values, feasible = cast("tuple[FloatND, BoolND]", scored)
    np.testing.assert_array_almost_equal(
        values[:3, 0],
        [np.log(1.25) + np.log(1.5) / 2.0, 0.0, np.log(1.5)],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(feasible[:3, 0], [False, True, True])
    assert float(values[0, 0]) > float(values[2, 0])
    result = model.simulate(
        params=_PARAMS,
        solution=solution,
        initial_conditions={
            **states,
            "age": jnp.asarray([10]),
            "regime_id": jnp.asarray([original.RegimeId.alive]),
        },
        log_level="off",
        seed=528,
    )
    rows = result.to_dataframe().query("regime_name == 'alive' and period == 0")
    assert len(rows) == 1
    row = rows.iloc[0]
    np.testing.assert_array_almost_equal(
        [row["consumption"], row["investment"], row["value"]],
        [1.0, 0.5, np.log(1.5)],
        decimal=DECIMAL_PRECISION,
    )
    resource = 1.25 - float(row["investment"]) ** 2 / 2.0
    np.testing.assert_array_almost_equal(
        [resource - float(row["consumption"]), 0.5 + float(row["investment"])],
        [0.125, 1.0],
        decimal=DECIMAL_PRECISION,
    )


def test_finite_replay_does_not_replace_an_all_illegal_published_bank() -> None:
    """Return the declared missing tuple when every represented candidate fails."""
    model = _model(lower=_higher_floor)
    solution = model.solve(params=_PARAMS, log_level="off")
    policy = solution.replay_artifacts.project(SIMULATION_POLICY)[0]["alive"]
    assert isinstance(policy, NNBEGMSimPolicy)
    states = {"liquid": jnp.asarray([0.0]), "illiquid": jnp.asarray([0.5])}
    regime = model._regimes["alive"]
    flat = regime_kernel_params(
        model._process_params(_PARAMS, runtime_checks=True), regime_name="alive"
    )
    assert model.ages is not None
    age = model.ages.values[0]
    inner, outer, live, represented = _replay._prepare_nnbegm_candidate_bank(
        regime=regime,
        sim_policy=policy,
        states=states,
        flat_params=flat,
        period=0,
        age=age,
    )
    np.testing.assert_array_equal(live[:, 0], [True, True, True, False])
    np.testing.assert_array_equal(represented[:, 0], [True, True, True, False])
    np.testing.assert_array_almost_equal(
        inner[:3, 0], [1.25, 1.0, 1.0], decimal=DECIMAL_PRECISION
    )
    ranked = _replay._rank_nnbegm_candidate_bank(
        candidate_inner=inner,
        candidate_outer=outer,
        represented=represented,
        optimal_actions=MappingProxyType({}),
        regime=regime,
        sim_policy=policy,
        flat_params=flat,
        period=0,
        age=age,
        canonical_states=states,
        action_names=regime.simulation.action_names,
        next_regime_to_V_arr=MappingProxyType(dict(solution.values[1])),
        referenced_value_kwargs=MappingProxyType({}),
    )
    assert type(ranked) is tuple
    actions, value = cast(
        "tuple[MappingProxyType[ActionName, FloatND | IntND], FloatND]", ranked
    )
    np.testing.assert_array_equal(actions["consumption"], [np.nan])
    np.testing.assert_array_equal(actions["investment"], [np.nan])
    np.testing.assert_array_equal(value, [-np.inf])


@pytest.mark.parametrize(
    "runtime_checks", [False, True], ids=["checks-off", "checks-on"]
)
def test_public_finite_replay_rejects_nonfinite_floor_parameter(
    *, runtime_checks: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nonfinite actual floors fail despite finite floors on the solve grid."""
    params = {**_PARAMS, "floor_offset": float("nan")}
    model, solution, states, _bank = _solve_healthy_status_fixture(
        lower=_parameter_floor, params=params
    )
    _require_transformed_bound_refusal(
        model=model,
        solution=solution,
        states=states,
        params=params,
        runtime_checks=runtime_checks,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize(
    "runtime_checks", [False, True], ids=["checks-off", "checks-on"]
)
def test_public_finite_replay_retains_invalid_flags_for_an_excluded_bank(
    *, runtime_checks: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An actual infeasible bank cannot erase its nonfinite current floor."""
    model, solution, states, bank = _solve_healthy_status_fixture(
        lower=_off_grid_nonfinite_floor, params=_PARAMS
    )
    regime = model._regimes["alive"]
    flat = regime_kernel_params(
        model._process_params(_PARAMS, runtime_checks=False), regime_name="alive"
    )
    assert model.ages is not None
    actual = regime.simulation.Q_and_F[0](
        **states,
        consumption=bank[0][0],
        investment=bank[1][0],
        next_regime_to_V_arr=MappingProxyType(dict(solution.values[1])),
        **flat,
        period=jnp.int32(0),
        age=model.ages.values[0],
    )
    assert type(actual) is BoundedResult
    bounded = cast("BoundedResult[tuple[FloatND, BoolND]]", actual)
    assert len(bounded.bound_status) == 1
    np.testing.assert_array_equal(jnp.stack(bounded.bound_status[0]), [True, False])
    np.testing.assert_array_equal(bounded.output[1], [False])
    _require_transformed_bound_refusal(
        model=model,
        solution=solution,
        states=states,
        params=_PARAMS,
        runtime_checks=runtime_checks,
        monkeypatch=monkeypatch,
    )


def _parameter_floor(*, illiquid: ContinuousState, floor_offset: float) -> FloatND:
    """Keep the current floor DAG independent of every action."""
    return _interior_floor(illiquid=illiquid) + jnp.where(
        illiquid == 0.5, floor_offset, 0.0
    )


def _off_grid_nonfinite_floor(*, illiquid: ContinuousState) -> FloatND:
    """Keep solve-grid floors finite and expose one invalid actual passive state."""
    return jnp.where(
        jnp.isclose(illiquid, 0.5, rtol=0.0, atol=0.0),
        jnp.nan,
        _interior_floor(illiquid=illiquid),
    )


def _solve_healthy_status_fixture(
    *, lower: UserFunction, params: UserParams
) -> tuple[
    Model,
    SolutionResult,
    dict[StateName, Float1D],
    tuple[FloatND, FloatND, BoolND, BoolND],
]:
    """Qualify the genuine solved bank before an off-grid replay."""
    model = _model(lower=lower)
    solution = model.solve(params=params, log_level="off")
    policy = solution.replay_artifacts.project(SIMULATION_POLICY)[0]["alive"]
    assert isinstance(policy, NNBEGMSimPolicy)
    assert policy.state_names == ("liquid", "illiquid")
    assert policy.n_keeper_candidates == 1
    np.testing.assert_array_equal(original._STOCK.to_jax(), [0.0, 1.0, 2.0])
    states = {"liquid": jnp.asarray([0.0]), "illiquid": jnp.asarray([0.5])}
    regime = model._regimes["alive"]
    flat = regime_kernel_params(
        model._process_params(params, runtime_checks=False), regime_name="alive"
    )
    assert model.ages is not None
    assert regime.simulation.replay_route.consumer_route == "nnbegm_finite"
    bank = _replay._prepare_nnbegm_candidate_bank(
        regime=regime,
        sim_policy=policy,
        states=states,
        flat_params=flat,
        period=0,
        age=model.ages.values[0],
    )
    np.testing.assert_array_equal(bank[2][:, 0], [True, True, True, False])
    np.testing.assert_array_equal(bank[3][:, 0], [True, True, True, False])
    np.testing.assert_array_almost_equal(
        bank[0][:3, 0], [1.25, 1.0, 1.0], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        bank[1][:3, 0], [0.0, -0.5, 0.5], decimal=DECIMAL_PRECISION
    )
    return model, solution, states, bank


def _require_transformed_bound_refusal(
    *,
    model: Model,
    solution: SolutionResult,
    states: dict[StateName, Float1D],
    params: UserParams,
    runtime_checks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require rank without diagnostics and attributed refusal in both modes."""
    rank = policy_programs._Rank.__call__
    reached: list[tuple[RegimeName, int]] = []

    @functools.wraps(rank)
    def observe_rank(
        self: policy_programs._Rank, **arguments: PytreeValue
    ) -> PytreeValue:
        reached.append((self.regime.name, self.period))
        return rank(self, **arguments)

    with monkeypatch.context() as hooks:
        hooks.setattr(policy_programs._Rank, "__call__", observe_rank)
        with pytest.raises(
            InvalidValueFunctionError,
            match=r"borrowing.*nonfinite.*alive.*period 0",
        ):
            model.simulate(
                params=params,
                solution=solution,
                initial_conditions={
                    **states,
                    "age": jnp.asarray([10]),
                    "regime_id": jnp.asarray([original.RegimeId.alive]),
                },
                log_level="off",
                seed=528,
                runtime_checks=runtime_checks,
            )
    if not runtime_checks:
        assert reached, "The public simulation must reach its real finite rank program"
    assert all(identity == ("alive", 0) for identity in reached)
