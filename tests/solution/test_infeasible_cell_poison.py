"""Infeasible utility values cannot change public values or decisions.

Continuous poison is confined to nonpositive consumption, outside the physical
consumption domain. Grid search enumerates those forbidden actions directly;
endogenous-grid families generate their own consumption candidates. The discrete
routes use either an explicit feasibility constraint or an unaffordable branch.
NEGM does not admit a discrete-action row axis.
"""

from collections.abc import Callable
from functools import wraps
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import step as egm_step
from lcm import (
    AdditiveShockTransition,
    AgeGrid,
    ByAge,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Regime,
    Transition,
    categorical,
)
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    outer_unchanged,
)
from lcm.exceptions import InvalidValueFunctionError
from lcm.solvers import DCEGM, EGM, NBEGM, NEGM, NNBEGM, FiniteOuterGrid, GridSearch
from lcm.transition import StochasticTransition
from lcm.typing import (
    ActionName,
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    FunctionName,
    ScalarInt,
    StateName,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

type SolverKind = Literal["grid", "egm", "dcegm", "nbegm", "negm", "nnbegm"]
type PoisonDomain = Literal["discrete", "continuous", "feasible", "raw", "simulate_nan"]
_ALL_SOLVERS: tuple[SolverKind, ...] = (
    "grid",
    "egm",
    "dcegm",
    "nbegm",
    "negm",
    "nnbegm",
)
_DISCRETE_SOLVERS: tuple[SolverKind, ...] = ("grid", "dcegm", "nbegm", "nnbegm")
_POISONS = (
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-inf"),
    pytest.param(-float("inf"), id="negative-inf"),
    pytest.param(1e6, id="finite-rival"),
)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _Work:
    allowed: ScalarInt
    forbidden: ScalarInt


@pytest.mark.parametrize("solver_kind", _DISCRETE_SOLVERS)
@pytest.mark.parametrize("poison", _POISONS)
def test_infeasible_discrete_utility_preserves_values_policies_and_paths(
    *, solver_kind: SolverKind, poison: float
) -> None:
    """A forbidden discrete branch cannot contaminate an earlier decision."""
    _assert_same_public_result(
        solver_kind=solver_kind, domain="discrete", poison=poison
    )


@pytest.mark.parametrize("solver_kind", _ALL_SOLVERS)
@pytest.mark.parametrize("poison", _POISONS)
def test_infeasible_continuous_utility_preserves_values_policies_and_paths(
    *, solver_kind: SolverKind, poison: float
) -> None:
    """Nonpositive consumption cannot alter the feasible decision."""
    _assert_same_public_result(
        solver_kind=solver_kind, domain="continuous", poison=poison
    )


@pytest.mark.parametrize("solver_kind", _ALL_SOLVERS)
def test_a_feasible_nan_is_detected_by_checked_solve(
    *, solver_kind: SolverKind
) -> None:
    """Numerical checking still rejects poison in an admissible utility cell."""
    model = _build_model(solver_kind=solver_kind, domain="feasible")
    with pytest.raises(InvalidValueFunctionError, match="alive"):
        model.solve(params={"discount_factor": 0.9, "poison": 1.0}, log_level="off")


@pytest.mark.parametrize("poison", _POISONS)
def test_asset_row_infeasible_discrete_utility_preserves_native_paths(
    *, poison: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wealth-dependent mortality exercises the real asset-row producer."""
    calls: list[None] = []
    original = egm_step._get_solve_one_combo_asset_rows
    monkeypatch.setattr(
        egm_step,
        "_get_solve_one_combo_asset_rows",
        _observe_calls(function=original, calls=calls),
    )
    _assert_same_public_result(
        solver_kind="dcegm", domain="discrete", poison=poison, asset_row=True
    )
    assert calls, "Native solves must actually select the asset-row producer."


def test_asset_row_a_feasible_nan_is_detected_by_checked_solve() -> None:
    model = _build_model(solver_kind="dcegm", domain="feasible", asset_row=True)
    with pytest.raises(InvalidValueFunctionError, match="alive"):
        model.solve(params={"discount_factor": 0.9, "poison": 1.0}, log_level="off")


def _observe_calls[**P, R](
    *, function: Callable[P, R], calls: list[None]
) -> Callable[P, R]:
    """Record producer selection while returning its actual native computation."""

    @wraps(function)
    def observed(*args: P.args, **kwargs: P.kwargs) -> R:
        calls.append(None)
        return function(*args, **kwargs)

    return observed


def _survival(wealth: ContinuousState) -> FloatND:
    return 0.9 + 0.005 * wealth


def _mortality(wealth: ContinuousState) -> FloatND:
    return 1.0 - _survival(wealth)


def _nan_simulation_utility(
    *, consumption: ContinuousAction, wealth: ContinuousState, poison: float
) -> FloatND:
    # Solve-grid wealth is integral; valid off-grid native simulation states are not.
    return -1.0 / consumption + jnp.where(wealth == jnp.floor(wealth), poison, jnp.nan)


def test_grid_search_feasible_nan_is_detected_in_cached_solution_simulation() -> None:
    """A healthy cached solution cannot hide a feasible off-grid utility NaN."""
    model = _build_model(solver_kind="grid", domain="simulate_nan")
    params = {"discount_factor": 0.9, "poison": 0.0}
    solution = model.solve(params=params, log_level="off")
    initial = {
        "wealth": jnp.asarray([4.5, 7.5, 10.5]),
        "age": jnp.zeros(3, dtype=jnp.int32),
        "regime_id": jnp.full(3, _RegimeId.alive),
    }
    finite = _build_model(solver_kind="grid", domain="continuous")
    finite_solution = finite.solve(params=params, log_level="off")
    for period, regimes in solution.values.items():
        for regime, value in regimes.items():
            np.testing.assert_array_equal(value, finite_solution.values[period][regime])
    healthy = finite.simulate(
        params=params,
        solution=finite_solution,
        initial_conditions=initial,
        seed=19,
        log_level="off",
        runtime_checks=True,
    )
    assert np.isfinite(healthy.to_dataframe(use_labels=False)["value"].to_numpy()).all()
    with pytest.raises(InvalidValueFunctionError, match="alive"):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            seed=19,
            log_level="off",
            runtime_checks=True,
        )


def _assert_same_public_result(
    *,
    solver_kind: SolverKind,
    domain: PoisonDomain,
    poison: float,
    asset_row: bool = False,
) -> None:
    model = _build_model(solver_kind=solver_kind, domain=domain, asset_row=asset_row)
    reference_params = {"discount_factor": 0.9, "poison": 0.0}
    poisoned_params = {"discount_factor": 0.9, "poison": _poison_selector(poison)}
    reference = model.solve(params=reference_params, log_level="off")
    poisoned = model.solve(params=poisoned_params, log_level="off")
    assert set(reference.values) == {0, 1, 2}
    for period, regimes in reference.values.items():
        for regime_name, expected in regimes.items():
            assert np.isfinite(np.asarray(expected)).all()
            np.testing.assert_array_almost_equal(
                np.asarray(poisoned.values[period][regime_name]),
                np.asarray(expected),
                decimal=DECIMAL_PRECISION,
            )
    state_name = "liquid" if solver_kind == "nbegm" else "wealth"
    states = {state_name: jnp.asarray([4.0, 7.0, 10.0])}
    if solver_kind in ("negm", "nnbegm"):
        states["illiquid"] = jnp.asarray([0.0, 1.0, 2.0])
    # NNBEGM exposes its published policy through native replay, not lookup_policy.
    for period in () if solver_kind == "nnbegm" else (0, 1):
        expected_policy = model.lookup_policy(
            params=reference_params,
            solution=reference,
            period=period,
            regime_name="alive",
            states=states,
        )
        actual_policy = model.lookup_policy(
            params=poisoned_params,
            solution=poisoned,
            period=period,
            regime_name="alive",
            states=states,
        )
        for action_name, expected in expected_policy.actions.items():
            np.testing.assert_array_almost_equal(
                np.asarray(actual_policy.actions[action_name]),
                np.asarray(expected),
                decimal=DECIMAL_PRECISION,
            )
        np.testing.assert_array_almost_equal(
            np.asarray(actual_policy.value),
            np.asarray(expected_policy.value),
            decimal=DECIMAL_PRECISION,
        )
        if domain == "discrete":
            np.testing.assert_array_equal(
                np.asarray(actual_policy.actions["work"]), np.zeros(3, dtype=int)
            )
    initial = {**states, "age": jnp.zeros(3), "regime_id": jnp.full(3, _RegimeId.alive)}
    expected_path = model.simulate(
        params=reference_params,
        solution=reference,
        initial_conditions=initial,
        seed=19,
        log_level="off",
    ).to_dataframe(use_labels=False)
    actual_path = model.simulate(
        params=poisoned_params,
        solution=poisoned,
        initial_conditions=initial,
        seed=19,
        log_level="off",
    ).to_dataframe(use_labels=False)
    assert tuple(actual_path.columns) == tuple(expected_path.columns)
    assert "value" in actual_path.columns
    assert set(actual_path.loc[actual_path["regime_name"] == "alive", "period"]) == {
        0,
        1,
    }
    if domain == "discrete":
        np.testing.assert_array_equal(
            actual_path.loc[actual_path["regime_name"] == "alive", "work"].to_numpy(),
            0,
        )
    for column in expected_path.columns:
        if expected_path[column].to_numpy().dtype.kind in "biufc":
            np.testing.assert_array_almost_equal(
                actual_path[column].to_numpy(),
                expected_path[column].to_numpy(),
                decimal=DECIMAL_PRECISION,
            )
        else:
            np.testing.assert_array_equal(
                actual_path[column].to_numpy(), expected_path[column].to_numpy()
            )


def _poison_selector(poison: float) -> float:
    """Finite parameters preserve the independent float32 admission contract."""
    if np.isnan(poison):
        return 1.0
    if np.isposinf(poison):
        return 2.0
    if np.isneginf(poison):
        return 3.0
    return poison


def _poison_payload(selector: float) -> FloatND:
    return jnp.where(
        selector == 1.0,
        jnp.nan,
        jnp.where(
            selector == 2.0, jnp.inf, jnp.where(selector == 3.0, -jnp.inf, selector)
        ),
    )


@pytest.mark.parametrize("poison", _POISONS)
def test_finite_selector_produces_exact_forbidden_utility_payload(
    poison: float,
) -> None:
    selector = _poison_selector(poison)
    assert np.isfinite(selector)
    discrete = _discrete_utility(
        consumption=jnp.asarray(2.0),
        work=jnp.asarray(_Work.forbidden),
        poison=selector,
    )
    continuous = _continuous_utility(consumption=jnp.asarray(-1.0), poison=selector)
    for actual in (discrete, continuous):
        if np.isnan(poison):
            assert bool(jnp.isnan(actual))
        else:
            np.testing.assert_array_equal(actual, poison)
    allowed = _discrete_utility(
        consumption=jnp.asarray(2.0),
        work=jnp.asarray(_Work.allowed),
        poison=selector,
    )
    np.testing.assert_array_equal(allowed, -0.5)
    healthy = _discrete_utility(
        consumption=jnp.asarray(2.0),
        work=jnp.asarray(_Work.forbidden),
        poison=0.0,
    )
    np.testing.assert_array_equal(healthy, -0.5)


def _plain_utility(consumption: ContinuousAction) -> FloatND:
    return -1.0 / consumption


def _discrete_utility(
    *, consumption: ContinuousAction, work: DiscreteAction, poison: float
) -> FloatND:
    return jnp.where(
        (work == _Work.forbidden) & (poison != 0.0),
        _poison_payload(poison),
        -1.0 / consumption,
    )


def _continuous_utility(*, consumption: ContinuousAction, poison: float) -> FloatND:
    return jnp.where(consumption <= 0.0, _poison_payload(poison), -1.0 / consumption)


def _feasible_utility(*, consumption: ContinuousAction, poison: float) -> FloatND:
    return -1.0 / consumption + _poison_payload(poison)


def _resources(wealth: ContinuousState) -> FloatND:
    return wealth + 3.0


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _next_wealth(savings: FloatND) -> ContinuousState:
    return savings + 3.0


def _inverse_marginal(marginal_continuation: FloatND) -> FloatND:
    return marginal_continuation**-0.5


def _budget(*, consumption: ContinuousAction, resources: FloatND) -> BoolND:
    return consumption <= resources


def _allowed_work(work: DiscreteAction) -> BoolND:
    return work == _Work.allowed


def _new_illiquid(
    *, illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    return illiquid + illiquid_investment


def _credited(*, illiquid: ContinuousState, new_illiquid: ContinuousState) -> FloatND:
    return new_illiquid - illiquid


def _outer_resources(*, wealth: ContinuousState, credited: FloatND) -> FloatND:
    return wealth + 3.0 - credited


def _next_illiquid(new_illiquid: ContinuousState) -> ContinuousState:
    return new_illiquid


def _terminal_utility(wealth: ContinuousState) -> FloatND:
    return -1.0 / (wealth + 1.0)


def _nested_terminal_utility(
    *, wealth: ContinuousState, illiquid: ContinuousState
) -> FloatND:
    return -1.0 / (wealth + 1.0) - 1.0 / (illiquid + 1.0)


def _build_model(
    *, solver_kind: SolverKind, domain: PoisonDomain, asset_row: bool = False
) -> Model:
    wealth_grid = LinSpacedGrid(start=3.0, stop=15.0, n_points=13)
    # GridSearch enumerates forbidden nonpositive actions. Endogenous-grid
    # solvers declare a positive action domain and generate their own candidates.
    consumption_grid = (
        LinSpacedGrid(start=-1.0, stop=20.0, n_points=43)
        if solver_kind == "grid" and domain in ("continuous", "raw", "simulate_nan")
        else LinSpacedGrid(start=0.25, stop=20.0, n_points=40)
    )
    savings_grid = LinSpacedGrid(start=0.0, stop=15.0, n_points=40)
    liquid = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    utility: UserFunction = (
        _nan_simulation_utility
        if domain == "simulate_nan"
        else _raw_sqrt_utility
        if domain == "raw"
        else _feasible_utility
        if domain == "feasible"
        else _discrete_utility
        if domain == "discrete"
        else _continuous_utility
    )
    functions: dict[FunctionName, UserFunction] = {
        "utility": utility,
        "resources": _resources,
        "savings": _savings,
    }
    constraints: dict[FunctionName, UserFunction] = (
        {"allowed_work": _allowed_work} if domain == "discrete" else {}
    )
    actions: dict[ActionName, LinSpacedGrid | DiscreteGrid] = {
        "consumption": consumption_grid
    }
    if domain == "discrete":
        actions["work"] = DiscreteGrid(_Work)
    if solver_kind == "grid":
        alive = Regime(
            states={"wealth": wealth_grid},
            actions=actions,
            state_transitions={"wealth": _next_wealth},
            functions=functions,
            constraints={
                **constraints,
                "budget": _budget,
                "positive_consumption": _positive_consumption,
            },
            solver=GridSearch(),
        )
    elif solver_kind in ("egm", "dcegm", "nbegm"):
        state_name: StateName = "liquid" if solver_kind == "nbegm" else "wealth"
        if solver_kind == "egm":
            functions["resources"] = _wealth_resources
        elif solver_kind == "nbegm":
            functions["resources"] = (
                _liquid_discrete_resources
                if domain == "discrete"
                else _liquid_resources
            )
            constraints = {}
            liquid = LiquidMargin(
                state="liquid",
                action="consumption",
                resources="resources",
                post_decision_state="savings",
            )
        solver = (
            EGM(savings_grid=savings_grid)
            if solver_kind == "egm"
            else DCEGM(savings_grid=savings_grid)
            if solver_kind == "dcegm"
            else NBEGM(savings_grid=savings_grid)
        )
        alive = ConsumptionSavingsRegime(
            states={state_name: wealth_grid},
            actions=actions,
            state_transitions={state_name: _next_wealth},
            functions=functions,
            constraints=constraints,
            liquid=liquid,
            solver=solver,
        )
    else:
        illiquid_grid = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)
        actions["illiquid_investment"] = LinSpacedGrid(start=-2.0, stop=2.0, n_points=5)
        functions.update({"new_illiquid": _new_illiquid, "credited": _credited})
        if solver_kind == "negm":
            del functions["resources"]
            functions["resources_before_cost"] = _resources
            functions["inverse_marginal_utility"] = (
                _inverse_sqrt_marginal if domain == "raw" else _inverse_marginal
            )
            nested_liquid = LiquidMargin(
                state="wealth",
                action="consumption",
                resources=NetOfAdjustmentCost(
                    output="resources",
                    before_cost="resources_before_cost",
                    cost="credited",
                ),
                post_decision_state="savings",
            )
            nested_solver = NEGM(
                inner=DCEGM(savings_grid=savings_grid), outer_grid=illiquid_grid
            )
        else:
            functions["resources"] = (
                _outer_discrete_resources if domain == "discrete" else _outer_resources
            )
            constraints = {}
            nested_liquid = liquid
            nested_solver = NNBEGM(
                inner=NBEGM(savings_grid=savings_grid),
                outer_search=FiniteOuterGrid(grid=illiquid_grid),
            )
        alive = NestedConsumptionSavingsRegime(
            states={"wealth": wealth_grid, "illiquid": illiquid_grid},
            actions=actions,
            state_transitions={"wealth": _next_wealth, "illiquid": _next_illiquid},
            functions=functions,
            constraints=constraints,
            liquid=nested_liquid,
            outer_continuous=OuterContinuousMargin(
                state="illiquid",
                action="illiquid_investment",
                post_decision_state="new_illiquid",
                no_adjustment=outer_unchanged,
            ),
            solver=nested_solver,
        )
    dead = (
        Regime(
            states={"liquid": wealth_grid},
            functions={"utility": _terminal_liquid_utility},
        )
        if solver_kind == "nbegm"
        else Regime(
            states={"wealth": wealth_grid}, functions={"utility": _terminal_utility}
        )
        if solver_kind not in ("negm", "nnbegm")
        else Regime(
            states={"wealth": wealth_grid, "illiquid": illiquid_grid},
            functions={"utility": _nested_terminal_utility},
        )
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        edges={
            "alive": Transition(
                targets={"alive": (0,), "dead": (0, 1)}
                if asset_row
                else {"alive": (0,), "dead": (1,)},
                law=ByAge.until(
                    stop_age_exclusive=2,
                    law={
                        "alive": StochasticTransition(func=_survival),
                        "dead": StochasticTransition(func=_mortality),
                    },
                    then="dead",
                )
                if asset_row
                else ByAge.until(stop_age_exclusive=2, law="alive", then="dead"),
            )
        },
        initial_nodes={0: "alive"},
    )


def _positive_consumption(consumption: ContinuousAction) -> BoolND:
    return consumption > 0.0


def _wealth_resources(wealth: ContinuousState) -> FloatND:
    return wealth


def _liquid_resources(liquid: ContinuousState) -> FloatND:
    return liquid + 3.0


def _liquid_discrete_resources(
    *, liquid: ContinuousState, work: DiscreteAction
) -> FloatND:
    return liquid + 3.0 - 100.0 * work


def _outer_discrete_resources(
    *, wealth: ContinuousState, credited: FloatND, work: DiscreteAction
) -> FloatND:
    return wealth + 3.0 - credited - 100.0 * work


def _terminal_liquid_utility(liquid: ContinuousState) -> FloatND:
    return -1.0 / (liquid + 1.0)


@categorical(ordered=False)
class _Health:
    low_bill: ScalarInt
    high_bill: ScalarInt
    unreachable: ScalarInt


def _fold_cost(*, next_xi: ContinuousState, next_health: DiscreteState) -> FloatND:
    return jnp.where(
        next_health == _Health.unreachable,
        100.0,
        0.5 * (1.0 + next_health) * jnp.exp(next_xi),
    )


def _negative_cost(cost: FloatND) -> FloatND:
    return -cost


def _before_bill(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption + 3.0


def _health_probs() -> FloatND:
    return jnp.asarray([0.7, 0.3, 0.0])


def _raw_domain_utility(
    *, consumption: ContinuousAction, health: DiscreteState, unsafe: bool
) -> FloatND:
    return (
        jnp.sqrt(
            jnp.where(
                unsafe, consumption, jnp.where(consumption > 0.0, consumption, 1.0)
            )
        )
        + 0.1 * health
    )


def _fold_bequest(*, wealth: ContinuousState, health: DiscreteState) -> FloatND:
    return jnp.sqrt(wealth + 4.0) + 0.1 * health


def _fold_model() -> Model:
    wealth_grid = LinSpacedGrid(start=3.0, stop=15.0, n_points=13)
    health_grid = DiscreteGrid(_Health)
    alive = Regime(
        states={
            "wealth": wealth_grid,
            "health": health_grid,
            "xi": NormalIIDProcess(
                n_points=3, mu=0.0, sigma=0.25, gauss_hermite=True, fold=True
            ),
        },
        actions={"consumption": IrregSpacedGrid(points=(-1.0, 0.0, 1.0, 2.0, 3.0))},
        state_transitions={
            "wealth": AdditiveShockTransition(
                base="before_bill", shock="negative_cost"
            ),
            "health": StochasticTransition(func=_health_probs),
        },
        functions={
            "utility": _raw_domain_utility,
            "resources": _resources,
            "cost": _fold_cost,
            "negative_cost": _negative_cost,
            "before_bill": _before_bill,
        },
        constraints={"budget": _budget, "positive_consumption": _positive_consumption},
        solver=GridSearch(),
    )
    dead = Regime(
        states={"wealth": wealth_grid, "health": health_grid},
        functions={"utility": _fold_bequest},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        edges={
            "alive": Transition(
                targets={"alive": (0,), "dead": (1,)},
                law=ByAge.until(stop_age_exclusive=2, law="alive", then="dead"),
            )
        },
    )


def _linear_read(*, at: float, nodes: np.ndarray, values: np.ndarray) -> float:
    index = int(
        np.clip(np.searchsorted(nodes, at, side="right") - 1, 0, len(nodes) - 2)
    )
    fraction = (at - nodes[index]) / (nodes[index + 1] - nodes[index])
    return float((1.0 - fraction) * values[index] + fraction * values[index + 1])


def _fold_enumerated_oracle() -> dict[int, np.ndarray]:
    """Literal feasible-action and live-node enumeration with linear reads.

    The unreachable health node is omitted before interpolation. The live shock
    readings are averaged after interpolation, never by replacing the bill with
    its mean. The utility expression here is the analytical square root over
    strictly positive actions and does not call the model's utility function.
    """
    nodes = np.linspace(3.0, 15.0, 13)
    process = NormalIIDProcess(
        n_points=3, mu=0.0, sigma=0.25, gauss_hermite=True, fold=True
    )
    shocks = np.asarray(process.to_jax())
    weights = np.asarray(process.get_transition_probs())[0]
    continuation = np.sqrt(nodes + 4.0)[:, None] + 0.1 * np.arange(3)[None, :]
    result: dict[int, np.ndarray] = {}
    for period in (1, 0):
        values = np.empty((len(nodes), 3))
        for i, wealth in enumerate(nodes):
            candidates = []
            for consumption in (1.0, 2.0, 3.0):
                expectation = 0.0
                for health, probability in ((0, 0.7), (1, 0.3)):
                    for shock, weight in zip(shocks, weights, strict=True):
                        landing = (
                            wealth
                            - consumption
                            + 3.0
                            - 0.5 * (1.0 + health) * np.exp(shock)
                        )
                        expectation += (
                            probability
                            * weight
                            * _linear_read(
                                at=float(landing),
                                nodes=nodes,
                                values=continuation[:, health],
                            )
                        )
                candidates.append(np.sqrt(consumption) + 0.9 * expectation)
            values[i, :] = max(candidates) + 0.1 * np.arange(3)
        result[period] = values
        continuation = values
    return result


def test_raw_domain_invalid_utility_matches_native_fold_and_literal_oracle() -> None:
    """Invalid primal and marginal cells are harmless outside feasibility."""
    assert np.isnan(
        float(
            _raw_domain_utility(
                consumption=jnp.asarray(-1.0), health=jnp.asarray(0), unsafe=True
            )
        )
    )
    assert np.isnan(
        float(
            jax.grad(
                lambda c: _raw_domain_utility(
                    consumption=c, health=jnp.asarray(0), unsafe=True
                )
            )(jnp.asarray(-1.0))
        )
    )
    model = _fold_model()
    reference_params = {"discount_factor": 0.9, "unsafe": False}
    poisoned_params = {"discount_factor": 0.9, "unsafe": True}
    reference = model.solve(params=reference_params, log_level="off")
    poisoned = model.solve(params=poisoned_params, log_level="off")
    order = model.state_names(regime_name="alive")
    for period, expected in _fold_enumerated_oracle().items():
        actual = np.transpose(
            np.asarray(poisoned.values[period]["alive"]),
            [order.index("wealth"), order.index("health")],
        )
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=DECIMAL_PRECISION
        )
        np.testing.assert_array_almost_equal(
            np.asarray(poisoned.values[period]["alive"]),
            np.asarray(reference.values[period]["alive"]),
            decimal=DECIMAL_PRECISION,
        )
    states = {"wealth": jnp.asarray([4.0, 7.0, 10.0]), "health": jnp.asarray([0, 1, 0])}
    for period in (0, 1):
        expected_policy = model.lookup_policy(
            params=reference_params,
            solution=reference,
            states=states,
            period=period,
            regime_name="alive",
        )
        actual_policy = model.lookup_policy(
            params=poisoned_params,
            solution=poisoned,
            states=states,
            period=period,
            regime_name="alive",
        )
        np.testing.assert_array_equal(
            np.asarray(actual_policy.actions["consumption"]),
            np.asarray(expected_policy.actions["consumption"]),
        )
        np.testing.assert_array_almost_equal(
            np.asarray(actual_policy.value),
            np.asarray(expected_policy.value),
            decimal=DECIMAL_PRECISION,
        )
    initial = {**states, "age": jnp.zeros(3), "regime_id": jnp.full(3, _RegimeId.alive)}
    expected_path = model.simulate(
        params=reference_params,
        solution=reference,
        initial_conditions=initial,
        seed=19,
        log_level="off",
    ).to_dataframe(use_labels=False)
    actual_path = model.simulate(
        params=poisoned_params,
        solution=poisoned,
        initial_conditions=initial,
        seed=19,
        log_level="off",
    ).to_dataframe(use_labels=False)
    assert tuple(actual_path.columns) == tuple(expected_path.columns)
    for column in expected_path.columns:
        if expected_path[column].to_numpy().dtype.kind in "biufc":
            np.testing.assert_array_almost_equal(
                actual_path[column].to_numpy(),
                expected_path[column].to_numpy(),
                decimal=DECIMAL_PRECISION,
            )
        else:
            np.testing.assert_array_equal(
                actual_path[column].to_numpy(), expected_path[column].to_numpy()
            )


def _raw_sqrt_utility(*, consumption: ContinuousAction, poison: float) -> FloatND:
    return jnp.sqrt(
        jnp.where(
            poison != 0.0, consumption, jnp.where(consumption > 0.0, consumption, 1.0)
        )
    )


def _inverse_sqrt_marginal(marginal_continuation: FloatND) -> FloatND:
    return 0.25 / marginal_continuation**2


@pytest.mark.parametrize("solver_kind", _ALL_SOLVERS)
def test_domain_invalid_utility_and_marginal_preserve_supported_native_paths(
    *, solver_kind: SolverKind
) -> None:
    """Square-root utility is unchanged on every positive consumption candidate.

    GridSearch evaluates forbidden negative action cells under an explicit
    positivity constraint. Endogenous-grid solvers declare positive action
    grids; their native generation/replay is checked without claiming negative
    candidate execution. NNBEGM's policy is observed through native simulation.
    """
    assert np.isnan(float(_raw_sqrt_utility(consumption=jnp.asarray(-1.0), poison=1.0)))
    assert np.isnan(
        float(
            jax.grad(lambda c: _raw_sqrt_utility(consumption=c, poison=1.0))(
                jnp.asarray(-1.0)
            )
        )
    )
    np.testing.assert_allclose(
        float(
            jax.grad(lambda c: _raw_sqrt_utility(consumption=c, poison=1.0))(
                jnp.asarray(4.0)
            )
        ),
        0.25,
        rtol=0.0,
        atol=0.0,
    )
    _assert_same_public_result(solver_kind=solver_kind, domain="raw", poison=1.0)


@categorical(ordered=False)
class _Status:
    ok: ScalarInt
    broke: ScalarInt


def _next_status(*, status: DiscreteState, work: DiscreteAction) -> DiscreteState:
    return jnp.where(work == _Work.forbidden, _Status.broke, status)


def _status_ok(status: DiscreteState) -> BoolND:
    return status == _Status.ok


def _no_work(work: DiscreteAction) -> BoolND:
    return work == _Work.allowed


def _status_terminal_utility(
    *, wealth: ContinuousState, status: DiscreteState
) -> FloatND:
    return -1.0 / (wealth + 1.0) + 0.0 * status


def _dead_branch_model(*, forbid_work: bool) -> Model:
    """DC-EGM model whose forbidden-work branch leads to an all-infeasible state.

    No action is allowed when `status` is broke, so in every alive period
    before the last the `(ok, forbidden)` combo has a `-inf` continuation at
    every savings node.
    """
    wealth_grid = LinSpacedGrid(start=3.0, stop=15.0, n_points=13)
    status_grid = DiscreteGrid(_Status)
    constraints: dict[FunctionName, UserFunction] = {"status_ok": _status_ok}
    if forbid_work:
        constraints["no_work"] = _no_work
    alive = ConsumptionSavingsRegime(
        states={"wealth": wealth_grid, "status": status_grid},
        actions={
            "consumption": LinSpacedGrid(start=0.25, stop=20.0, n_points=40),
            "work": DiscreteGrid(_Work),
        },
        state_transitions={"wealth": _next_wealth, "status": _next_status},
        functions={
            "utility": _plain_utility,
            "resources": _resources,
            "savings": _savings,
        },
        constraints=constraints,
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        solver=DCEGM(savings_grid=LinSpacedGrid(start=0.0, stop=15.0, n_points=40)),
    )
    dead = Regime(
        states={"wealth": wealth_grid, "status": status_grid},
        functions={"utility": _status_terminal_utility},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=_RegimeId,
        edges={
            "alive": Transition(
                targets={"alive": (0, 1), "dead": (2,)},
                law=ByAge.until(stop_age_exclusive=3, law="alive", then="dead"),
            )
        },
        initial_nodes={0: "alive"},
    )


@pytest.mark.parametrize("runtime_checks", [True, False])
def test_dcegm_combo_without_live_candidate_matches_forbidden_combo(
    *, runtime_checks: bool
) -> None:
    """A branch whose continuation is `-inf` everywhere acts as an infeasible one.

    Allowing forbidden work, which leads only into a state without any feasible
    action, yields the same values bitwise as excluding it by a constraint.
    """
    params = {"discount_factor": 0.9}
    expected = _dead_branch_model(forbid_work=True).solve(
        params=params, log_level="off"
    )
    actual = _dead_branch_model(forbid_work=False).solve(
        params=params, log_level="off", runtime_checks=runtime_checks
    )
    assert set(actual.values) == {0, 1, 2, 3}
    for period, regimes in expected.values.items():
        for regime_name, value in regimes.items():
            np.testing.assert_array_equal(
                np.asarray(actual.values[period][regime_name]), np.asarray(value)
            )
