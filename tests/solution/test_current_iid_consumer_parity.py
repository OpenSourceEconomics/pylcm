"""Declared current-IID shocks reproduce the node-state IID models of actual consumers.

An IID process declared as a state is a value axis: the value array holds one slice
per quadrature node. The same law declared as a `ShockGroup` is integrated within the
period, so its value array equals the node-weighted average of those slices, and a
subject whose draw is supplied at a node takes the node-state model's policy.
"""

import functools
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from lcm import (
    ConsumptionSavingsRegime,
    DiscreteGrid,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    NormalIIDProcess,
    Periods,
    Regime,
    UniformIIDProcess,
    categorical,
    fixed_transition,
    post_decision_lower_bound,
)
from lcm._solver_api.result import SolutionResult
from lcm.exceptions import (
    InvalidSimulationInputError,
    OffGridStateError,
    ShockDeclarationError,
)
from lcm.persistence import load_solution
from lcm.result import SimulationResult
from lcm.shocks import (
    BaseInnovations,
    Discrete,
    EqualSpacedUniform,
    GaussHermite,
    MonteCarlo,
    Normal,
    PhysicalValues,
    Product,
    ShockGroup,
    Smolyak,
    Uniform,
    UserNodes,
)
from lcm.solvers import DCEGM, NBEGM, NEGM, NNBEGM, FiniteOuterGrid
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    RegimeName,
    ScalarInt,
    StateName,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm._nnbegm_toy import _declarations, _oracle, _params

_N_HCC_NODES = 3


@categorical(ordered=False)
class _CostRegimeId:
    alive: ScalarInt
    done: ScalarInt


def _total_costs(*, hcc_transitory: FloatND, cost_scale: float) -> FloatND:
    return cost_scale * hcc_transitory


def _resources(*, wealth: ContinuousState, total_costs: FloatND) -> FloatND:
    return wealth + 2.0 - total_costs


def _log_consumption(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _inverse_marginal_utility(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _next_wealth_from_savings(*, savings: FloatND) -> ContinuousState:
    return savings


def _next_wealth_from_resources(
    *, resources: FloatND, consumption: ContinuousAction
) -> ContinuousState:
    return resources - consumption


def _affordable(*, consumption: ContinuousAction, resources: FloatND) -> BoolND:
    return consumption <= resources


def _log_wealth(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _cost_shifted_floor(*, hcc_transitory: FloatND, floor_scale: float) -> FloatND:
    """A floor on savings that moves with this period's health-cost draw."""
    return 2.0 - floor_scale * hcc_transitory


def _literal_floor(*, solver: str) -> float:
    """A literal floor: binding under GridSearch, the savings-grid origin under EGM.

    An endogenous-grid solver accepts only its savings grid's origin as a literal
    floor, as the ACA model declares it.
    """
    return 1.5 if solver == "grid_search" else _SAVINGS_FLOOR


_MARGIN = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)


# Steps of 1/2 hold every optimal saving R/2 for the integer resources R that the
# unit-scaled Gauss-Hermite costs produce, so no solver interpolates off a node.
_SAVINGS_FLOOR = 0.5
_SAVINGS_GRID = IrregSpacedGrid(points=np.linspace(_SAVINGS_FLOOR, 6.0, 12))


def _health_cost_model(
    *, solver: str, declared: bool, bound: str | None = None
) -> Model:
    """Two-period savings model whose transitory health cost is a current IID draw.

    The cost function reads the shock value, as `total_costs` reads `hcc_transitory`
    in the ACA model, under that model's three solvers. `bound` adds a declared
    post-decision floor: `"literal"` a number, `"shock"` one reading the draw.
    """
    states: dict[StateName, LinSpacedGrid | NormalIIDProcess] = {
        "wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=5)
    }
    if not declared:
        states["hcc_transitory"] = NormalIIDProcess(
            n_points=_N_HCC_NODES, gauss_hermite=True, mu=0.0, sigma=1.0
        )
    functions = {
        "utility": _log_consumption,
        "total_costs": _total_costs,
        "resources": _resources,
    }
    shocks = ("hcc",) if declared else ()
    constraints = {}
    if bound is not None:
        functions["savings"] = _savings
        constraints["borrowing_limit"] = post_decision_lower_bound(
            margin=_MARGIN,
            lower=_literal_floor(solver=solver)
            if bound == "literal"
            else _cost_shifted_floor,
        )
    # Node-state EGM replays its policy on the declared action grid; this one holds
    # every optimum R/2.
    consumption = LinSpacedGrid(start=0.5, stop=6.0, n_points=12)
    if solver == "grid_search":
        alive = Regime(
            states=states,
            actions={"consumption": consumption},
            functions=functions,
            constraints={"affordable": _affordable, **constraints},
            state_transitions={"wealth": {"done": _next_wealth_from_resources}},
            shocks=shocks,
        )
    else:
        alive = ConsumptionSavingsRegime(
            states=states,
            actions={"consumption": consumption},
            functions={
                **functions,
                "savings": _savings,
                "inverse_marginal_utility": _inverse_marginal_utility,
            },
            state_transitions={"wealth": {"done": _next_wealth_from_savings}},
            constraints=constraints,
            liquid=_MARGIN,
            solver=(DCEGM if solver == "dcegm" else NBEGM)(savings_grid=_SAVINGS_GRID),
            shocks=shocks,
        )
    return Model(
        regimes={
            "alive": alive,
            "done": Regime(
                states={"wealth": _SAVINGS_GRID}, functions={"utility": _log_wealth}
            ),
        },
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"done": Periods(values=(0,))}},
        regime_id_class=_CostRegimeId,
        fixed_params={"discount_factor": 1.0},
        shocks=(
            {
                "hcc": ShockGroup(
                    shocks={"hcc_transitory": Normal(mu=0.0, sigma=1.0)},
                    nodes=Product(hcc_transitory=GaussHermite(n=_N_HCC_NODES)),
                )
            }
            if declared
            else {}
        ),
    )


def _integrate_node_axis(
    *,
    model: Model,
    values: FloatND,
    regime_name: RegimeName,
    shock: StateName,
    weights: FloatND,
) -> FloatND:
    axis = model.state_names(regime_name=regime_name).index(shock)
    return jnp.tensordot(values, weights, axes=([axis], [0]))


@pytest.mark.parametrize(
    "solver",
    [
        "grid_search",
        pytest.param(
            "dcegm",
            marks=pytest.mark.requires_exact_affine_kernel(
                reason=EXACT_KERNEL_SKIP_REASON
            ),
        ),
        "nbegm",
    ],
)
def test_declared_health_cost_shock_reproduces_node_state_model(solver: str) -> None:
    """Values are the node-weighted node-state values; node draws keep its policy."""
    _assert_reproduces_node_state_model(solver=solver, bound=None)


@pytest.mark.parametrize(
    ("solver", "bound"),
    [
        ("grid_search", "literal"),
        ("grid_search", "shock"),
        pytest.param(
            "dcegm",
            "literal",
            marks=pytest.mark.requires_exact_affine_kernel(
                reason=EXACT_KERNEL_SKIP_REASON
            ),
        ),
        ("nbegm", "literal"),
    ],
)
def test_declared_health_cost_shock_with_savings_floor_reproduces_node_state_model(
    *, solver: str, bound: str
) -> None:
    """A declared savings floor binds at the realized draw, as on the node state.

    The floor is either a number or reads the draw itself; both bind for some
    wealth levels and nodes, so dropping the floor or reading it at another node
    changes values and policies.
    """
    _assert_reproduces_node_state_model(solver=solver, bound=bound)


def _assert_reproduces_node_state_model(*, solver: str, bound: str | None) -> None:
    # Unit costs at the Gauss-Hermite nodes -sqrt(3), 0, sqrt(3).
    params = {
        "alive": {
            "total_costs": {"cost_scale": 1 / np.sqrt(3.0)},
            "borrowing_limit_lower": {"floor_scale": 1 / np.sqrt(3.0)},
        }
        if bound == "shock"
        else {"total_costs": {"cost_scale": 1 / np.sqrt(3.0)}}
    }
    legacy = _health_cost_model(solver=solver, declared=False, bound=bound)
    declared = _health_cost_model(solver=solver, declared=True, bound=bound)
    legacy_solution = legacy.solve(params=params, log_level="off")
    declared_solution = declared.solve(params=params, log_level="off")

    # The cost table inputs (node values and weights) are the same law's nodes.
    process = NormalIIDProcess(
        n_points=_N_HCC_NODES, gauss_hermite=True, mu=0.0, sigma=1.0
    )
    group = declared.shocks["hcc"]
    nodes = group.resolve_nodes(group_name="hcc", age=0, coordinate_kind="period")
    realisation = group.transform(base_nodes=nodes.base_nodes)
    assert_array_almost_equal(
        realisation.values["hcc_transitory"],
        legacy.state_grid(
            params=params, regime_name="alive", state_name="hcc_transitory"
        ),
        decimal=DECIMAL_PRECISION,
    )
    weights = process.get_transition_probs()[0]
    assert_array_almost_equal(nodes.weights, weights, decimal=DECIMAL_PRECISION)

    assert_array_almost_equal(
        declared_solution.values[0]["alive"],
        _integrate_node_axis(
            model=legacy,
            values=legacy_solution.values[0]["alive"],
            regime_name="alive",
            shock="hcc_transitory",
            weights=weights,
        ),
        decimal=DECIMAL_PRECISION,
    )

    wealth = jnp.repeat(jnp.arange(1.0, 6.0), _N_HCC_NODES)
    draws = jnp.tile(jnp.asarray(realisation.values["hcc_transitory"]), 5)
    common = {
        "wealth": wealth,
        "period": jnp.zeros(wealth.shape, dtype=jnp.int32),
        "regime_id": jnp.zeros(wealth.shape, dtype=jnp.int32),
    }
    legacy_panel = legacy.simulate(
        params=params,
        solution=legacy_solution,
        initial_conditions={**common, "hcc_transitory": draws},
        seed=3,
        log_level="off",
    ).to_dataframe()
    declared_panel = declared.simulate(
        params=params,
        solution=declared_solution,
        initial_conditions={**common, "hcc_transitory": PhysicalValues(values=draws)},
        seed=3,
        log_level="off",
    ).to_dataframe()
    for panel in (legacy_panel, declared_panel):
        panel.sort_values(["subject_id", "period"], inplace=True)
    assert_array_almost_equal(
        declared_panel["consumption"].to_numpy(dtype=float),
        legacy_panel["consumption"].to_numpy(dtype=float),
        decimal=DECIMAL_PRECISION,
    )
    assert_array_almost_equal(
        declared_panel["wealth"].to_numpy(dtype=float),
        legacy_panel["wealth"].to_numpy(dtype=float),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize(
    "cost_scale",
    [
        1 / np.sqrt(3.0),
        pytest.param(
            0.5,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "the NBEGM ride-along solve snaps savings to the grid with a "
                    "process co-state; lands with wip/607-nbsnap"
                ),
            ),
        ),
    ],
)
def test_nbegm_health_cost_simulates_fresh_period_zero_draws(cost_scale: float) -> None:
    """Omitted draws are realised fresh, including beyond the outermost nodes.

    Resources `wealth + 2 - cost` then leave the range the quadrature nodes span on
    the wealth grid; the policy is still the interior Euler solution `resources / 2`.
    The node-state axis replays its policy on the declared action grid, so this
    continuous-policy check runs on the current-IID route.
    """
    params = {"alive": {"total_costs": {"cost_scale": cost_scale}}}
    model = _health_cost_model(solver="nbegm", declared=True)
    solution = model.solve(params=params, log_level="off")
    wealth = jnp.tile(jnp.arange(1.0, 6.0), 20)
    panel = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": wealth,
            "period": jnp.zeros(wealth.shape, dtype=jnp.int32),
            "regime_id": jnp.zeros(wealth.shape, dtype=jnp.int32),
        },
        seed=3,
        log_level="off",
    ).to_dataframe()
    panel.sort_values(["subject_id", "period"], inplace=True)
    first = panel[panel["period"] == 0]
    last = panel[panel["period"] == 1]
    draws = first["hcc_transitory"].to_numpy(dtype=float)
    resources = first["wealth"].to_numpy(dtype=float) + 2.0 - cost_scale * draws
    node_reach = cost_scale * np.sqrt(3.0)
    assert np.any((resources < 3.0 - node_reach) | (resources > 7.0 + node_reach))
    consumption = first["consumption"].to_numpy(dtype=float)
    assert_array_almost_equal(consumption, resources / 2, decimal=DECIMAL_PRECISION)
    assert_array_almost_equal(
        last["wealth"].to_numpy(dtype=float),
        resources - consumption,
        decimal=DECIMAL_PRECISION,
    )
    assert np.all(np.isfinite(panel["value"].to_numpy(dtype=float)))


@categorical(ordered=True)
class _Effort:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _CareerRegimeId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _consumption(
    *, wealth: ContinuousState, saving: ContinuousAction, income: FloatND
) -> FloatND:
    return wealth + income - saving


def _wage_income(*, effort: DiscreteAction) -> FloatND:
    return 1.0 + 0.5 * effort


def _pension_income() -> FloatND:
    return jnp.asarray(0.8)


def _adjustment_cost_penalty(
    *,
    adjustment_cost: ContinuousState,
    effort: DiscreteAction,
    lagged_effort: DiscreteState,
    adjustment_cost_scale: float,
) -> FloatND:
    return jnp.where(
        effort == lagged_effort, 0.0, adjustment_cost * adjustment_cost_scale
    )


def _effort_utility(
    *, consumption: FloatND, effort: DiscreteAction, adjustment_cost_penalty: FloatND
) -> FloatND:
    return jnp.log(consumption) - 0.1 * effort - adjustment_cost_penalty


def _positive_consumption(*, consumption: FloatND) -> BoolND:
    return consumption > 0


def _next_wealth_from_saving(*, saving: ContinuousAction) -> ContinuousState:
    return saving


def _next_lagged_effort(*, effort: DiscreteAction) -> DiscreteState:
    return effort


def _bequest(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(1.0 + wealth)


_WEALTH = LinSpacedGrid(start=0.0, stop=4.0, n_points=6)


def _effort_regime(*, income: Callable[..., FloatND], declared: bool) -> Regime:
    states: dict[StateName, LinSpacedGrid | DiscreteGrid | UniformIIDProcess] = {
        "wealth": _WEALTH,
        "lagged_effort": DiscreteGrid(category_class=_Effort),
    }
    if not declared:
        states["adjustment_cost"] = UniformIIDProcess(n_points=5, start=0, stop=1)
    transitions = {
        "wealth": _next_wealth_from_saving,
        "lagged_effort": _next_lagged_effort,
    }
    return Regime(
        states=states,
        actions={
            "effort": DiscreteGrid(category_class=_Effort),
            "saving": LinSpacedGrid(start=0.0, stop=4.0, n_points=9),
        },
        functions={
            "utility": _effort_utility,
            "consumption": _consumption,
            "income": income,
            "adjustment_cost_penalty": _adjustment_cost_penalty,
        },
        constraints={"positive_consumption": _positive_consumption},
        state_transitions=transitions,
        shocks=("adjustment",) if declared else (),
    )


def _effort_adjustment_model(*, declared: bool) -> Model:
    """Mahler-Yum's effort adjustment cost: a uniform IID draw in two regimes."""
    return Model(
        regimes={
            "working": _effort_regime(income=_wage_income, declared=declared),
            "retired": _effort_regime(income=_pension_income, declared=declared),
            "dead": Regime(states={"wealth": _WEALTH}, functions={"utility": _bequest}),
        },
        n_periods=3,
        initial_nodes=InitialNodes(by_period={0: "working"}),
        edges={
            "working": {"retired": Periods(values=(0,))},
            "retired": {"dead": Periods(values=(1,))},
        },
        regime_id_class=_CareerRegimeId,
        fixed_params={"discount_factor": 0.95, "adjustment_cost_scale": 0.6},
        shocks=(
            {
                "adjustment": ShockGroup(
                    shocks={"adjustment_cost": Uniform(start=0, stop=1)},
                    nodes=Product(adjustment_cost=EqualSpacedUniform(n=5)),
                )
            }
            if declared
            else {}
        ),
    )


def test_declared_uniform_adjustment_cost_reproduces_node_state_model() -> None:
    """Both regimes' values integrate the node slices; node paths keep the policy."""
    legacy = _effort_adjustment_model(declared=False)
    declared = _effort_adjustment_model(declared=True)
    legacy_solution = legacy.solve(params={}, log_level="off")
    declared_solution = declared.solve(params={}, log_level="off")
    weights = jnp.full(5, 0.2)
    for period, regime_name in ((0, "working"), (1, "retired")):
        assert_array_almost_equal(
            declared_solution.values[period][regime_name],
            _integrate_node_axis(
                model=legacy,
                values=legacy_solution.values[period][regime_name],
                regime_name=regime_name,
                shock="adjustment_cost",
                weights=weights,
            ),
            decimal=DECIMAL_PRECISION,
        )

    nodes = jnp.linspace(0.0, 1.0, 5)
    n_subjects = 2 * 6 * 5
    wealth = jnp.tile(jnp.repeat(jnp.linspace(0.0, 4.0, 6), 5), 2)
    lagged_effort = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 30)
    paths = jnp.stack(
        [jnp.tile(nodes, 12), jnp.tile(jnp.roll(nodes, 2), 12), jnp.tile(nodes, 12)],
        axis=1,
    )
    common = {
        "wealth": wealth,
        "lagged_effort": lagged_effort,
        "period": jnp.zeros(n_subjects, dtype=jnp.int32),
        "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
    }
    legacy_panel = legacy.simulate(
        params={},
        solution=legacy_solution,
        initial_conditions={**common, "adjustment_cost": paths[:, 0]},
        shock_paths={"adjustment_cost": paths},
        seed=5,
        log_level="off",
        use_categorical_codes=True,
    ).to_dataframe(use_categorical_codes=True)
    declared_panel = declared.simulate(
        params={},
        solution=declared_solution,
        initial_conditions=common,
        shock_paths={"adjustment_cost": PhysicalValues(values=paths)},
        seed=5,
        log_level="off",
        use_categorical_codes=True,
    ).to_dataframe(use_categorical_codes=True)
    for panel in (legacy_panel, declared_panel):
        panel.sort_values(["subject_id", "period"], inplace=True)
    for column in ("effort", "saving", "wealth", "adjustment_cost"):
        assert_array_almost_equal(
            declared_panel[column].to_numpy(dtype=float),
            legacy_panel[column].to_numpy(dtype=float),
            decimal=DECIMAL_PRECISION,
            err_msg=column,
        )


@categorical(ordered=True)
class _Health:
    bad: ScalarInt
    good: ScalarInt


def _shocked_log_consumption(
    *, consumption: ContinuousAction, taste: FloatND
) -> FloatND:
    return (1.0 + 0.2 * taste) * jnp.log(consumption)


def _shocked_inverse_marginal_utility(
    *, marginal_continuation: FloatND, taste: FloatND
) -> FloatND:
    return (1.0 + 0.2 * taste) / marginal_continuation


def _health_resources(
    *, wealth: ContinuousState, good_health: DiscreteState
) -> FloatND:
    return wealth + 1.5 + 0.5 * good_health


def _health_bequest(*, wealth: ContinuousState, good_health: DiscreteState) -> FloatND:
    return jnp.log(wealth) + 0.1 * good_health


def _carried_health_model(*, declared: bool) -> Model:
    """NBEGM source whose passive discrete state is carried into its terminal.

    As aca's `pref_type`, the terminal regime keeps the discrete state, so the
    child carry has a leading discrete axis that the current-IID source must index.
    """
    states: dict[StateName, LinSpacedGrid | DiscreteGrid | NormalIIDProcess] = {
        "wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=5),
        "good_health": DiscreteGrid(category_class=_Health),
    }
    if not declared:
        states["taste"] = NormalIIDProcess(
            n_points=_N_HCC_NODES, gauss_hermite=True, mu=0.0, sigma=1.0
        )
    alive = ConsumptionSavingsRegime(
        states=states,
        actions={"consumption": LinSpacedGrid(start=0.5, stop=6.0, n_points=12)},
        functions={
            "utility": _shocked_log_consumption,
            "resources": _health_resources,
            "savings": _savings,
            "inverse_marginal_utility": _shocked_inverse_marginal_utility,
        },
        state_transitions={
            "wealth": {"done": _next_wealth_from_savings},
            "good_health": {"done": fixed_transition("good_health")},
        },
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        solver=NBEGM(savings_grid=_SAVINGS_GRID),
        shocks=("taste",) if declared else (),
    )
    done = Regime(
        states={
            "wealth": _SAVINGS_GRID,
            "good_health": DiscreteGrid(category_class=_Health),
        },
        functions={"utility": _health_bequest},
    )
    return Model(
        regimes={"alive": alive, "done": done},
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"done": Periods(values=(0,))}},
        regime_id_class=_CostRegimeId,
        fixed_params={"discount_factor": 1.0},
        shocks=(
            {
                "taste": ShockGroup(
                    shocks={"taste": Normal(mu=0.0, sigma=1.0)},
                    nodes=Product(taste=GaussHermite(n=_N_HCC_NODES)),
                )
            }
            if declared
            else {}
        ),
    )


def test_nbegm_carries_passive_discrete_state_into_terminal() -> None:
    """A terminal carrying a discrete state reproduces the node-state NBEGM values."""
    legacy = _carried_health_model(declared=False)
    declared = _carried_health_model(declared=True)
    legacy_solution = legacy.solve(params={}, log_level="off")
    declared_solution = declared.solve(params={}, log_level="off")
    weights = NormalIIDProcess(
        n_points=_N_HCC_NODES, gauss_hermite=True, mu=0.0, sigma=1.0
    ).get_transition_probs()[0]
    assert_array_almost_equal(
        declared_solution.values[0]["alive"],
        _integrate_node_axis(
            model=legacy,
            values=legacy_solution.values[0]["alive"],
            regime_name="alive",
            shock="taste",
            weights=weights,
        ),
        decimal=DECIMAL_PRECISION,
    )

    good_health = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 5)
    wealth = jnp.tile(jnp.arange(1.0, 6.0), 2)
    panel = declared.simulate(
        params={},
        solution=declared_solution,
        initial_conditions={
            "wealth": wealth,
            "good_health": good_health,
            "period": jnp.zeros(wealth.shape, dtype=jnp.int32),
            "regime_id": jnp.zeros(wealth.shape, dtype=jnp.int32),
        },
        seed=7,
        log_level="off",
        use_categorical_codes=True,
    ).to_dataframe(use_categorical_codes=True)
    panel.sort_values(["subject_id", "period"], inplace=True)
    first = panel[panel["period"] == 0]
    last = panel[panel["period"] == 1]
    np.testing.assert_array_equal(last["good_health"].to_numpy(), good_health)
    consumption = first["consumption"].to_numpy(dtype=float)
    resources = np.asarray(wealth) + 1.5 + 0.5 * np.asarray(good_health)
    assert np.all(consumption > 0)
    assert_array_almost_equal(
        last["wealth"].to_numpy(dtype=float),
        resources - consumption,
        decimal=DECIMAL_PRECISION,
    )


@categorical(ordered=False)
class _ChainRegimeId:
    a: ScalarInt
    b: ScalarInt
    done: ScalarInt


@categorical(ordered=True)
class _Work:
    no: ScalarInt
    yes: ScalarInt


def _log_normal_costs(*, hcc_transitory: FloatND) -> FloatND:
    return 0.5 * jnp.exp(0.3 * hcc_transitory)


def _scaled_log_normal_costs(*, hcc_transitory: FloatND, hcc_scale: FloatND) -> FloatND:
    return hcc_scale * 0.5 * jnp.exp(0.3 * hcc_transitory)


def _health_costs(*, hcc_transitory: FloatND, good_health: DiscreteState) -> FloatND:
    return (0.7 - 0.3 * good_health) * jnp.exp(0.3 * hcc_transitory)


def _working_resources(
    *, wealth: ContinuousState, total_costs: FloatND, work: DiscreteAction
) -> FloatND:
    return wealth + 1.0 + work - total_costs


def _work_utility(*, consumption: ContinuousAction, work: DiscreteAction) -> FloatND:
    return jnp.log(consumption) - 0.2 * work


def _hcc_shocked_utility(
    *, consumption: ContinuousAction, hcc_transitory: FloatND
) -> FloatND:
    return (1.0 + 0.2 * hcc_transitory) * jnp.log(consumption)


def _hcc_shocked_inverse_marginal_utility(
    *, marginal_continuation: FloatND, hcc_transitory: FloatND
) -> FloatND:
    return (1.0 + 0.2 * hcc_transitory) / marginal_continuation


_HCC = ShockGroup(
    shocks={"hcc_transitory": Normal(mu=0.0, sigma=1.0)},
    nodes=Product(hcc_transitory=GaussHermite(n=_N_HCC_NODES)),
)
_SCALE_NODES = 3
_EGM_SHAPES = (
    "multi_date",
    "chain",
    "passive_state",
    "discrete_action",
    "utility_shock",
    "aca",
)
_SOLVERS = (
    pytest.param(
        "dcegm",
        marks=pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON),
    ),
    "nbegm",
)


def _aca_shaped_model(
    *,
    solver: str,
    declared: bool,
    shape: str,
    group: ShockGroup = _HCC,
    scaled: bool = False,
) -> Model:
    """Three-period savings model whose EGM sources read the health cost at each date.

    `shape` selects one feature of the ACA model's EGM regimes:

    - `multi_date`: one source at dates 0 and 1;
    - `chain`: two sources, `a -> b -> done`;
    - `passive_state`: a deterministic health state carried into the terminal;
    - `discrete_action`: a work choice;
    - `utility_shock`: the draw enters utility, beside the carried health state;
    - `aca`: the carried health state, the work choice and a literal savings floor
      at the savings grid's origin together.

    `scaled` multiplies the cost by a second, uniform shock of the same group.
    """
    carried = shape in {"passive_state", "utility_shock", "aca"}
    states: dict[
        StateName,
        LinSpacedGrid | DiscreteGrid | NormalIIDProcess | UniformIIDProcess,
    ] = {"wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=5)}
    if carried:
        states["good_health"] = DiscreteGrid(category_class=_Health)
    if not declared:
        states["hcc_transitory"] = NormalIIDProcess(
            n_points=_N_HCC_NODES, gauss_hermite=True, mu=0.0, sigma=1.0
        )
        if scaled:
            states["hcc_scale"] = UniformIIDProcess(
                n_points=_SCALE_NODES, start=0.8, stop=1.2
            )
    functions = {
        "utility": _log_consumption,
        "total_costs": _scaled_log_normal_costs if scaled else _log_normal_costs,
        "resources": _resources,
        "savings": _savings,
        "inverse_marginal_utility": _inverse_marginal_utility,
    }
    actions: dict[str, LinSpacedGrid | DiscreteGrid] = {
        "consumption": LinSpacedGrid(start=0.5, stop=6.0, n_points=12)
    }
    if shape in {"passive_state", "aca"}:
        functions["total_costs"] = _health_costs
    if shape in {"discrete_action", "aca"}:
        actions["work"] = DiscreteGrid(category_class=_Work)
        functions["utility"] = _work_utility
        functions["resources"] = _working_resources
    if shape == "utility_shock":
        del functions["total_costs"]
        functions["utility"] = _hcc_shocked_utility
        functions["inverse_marginal_utility"] = _hcc_shocked_inverse_marginal_utility
        functions["resources"] = _health_resources
    sources, targets = (
        ({"a": ("b",), "b": ("done",)}, {"a": (0,), "b": (1,)})
        if shape == "chain"
        else ({"alive": ("alive", "done")}, {"alive": (0, 1)})
    )

    def source(name: RegimeName) -> Regime:
        transitions = {
            "wealth": dict.fromkeys(sources[name], _next_wealth_from_savings)
        }
        if carried:
            transitions["good_health"] = dict.fromkeys(
                sources[name], fixed_transition("good_health")
            )
        return ConsumptionSavingsRegime(
            states=states,
            actions=actions,
            functions=functions,
            state_transitions=transitions,
            constraints={
                "borrowing_limit": post_decision_lower_bound(
                    margin=_MARGIN, lower=_SAVINGS_FLOOR
                )
            }
            if shape == "aca"
            else {},
            liquid=_MARGIN,
            solver=(DCEGM if solver == "dcegm" else NBEGM)(savings_grid=_SAVINGS_GRID),
            shocks=("hcc",) if declared else (),
        )

    done = Regime(
        states={"wealth": _SAVINGS_GRID}
        | ({"good_health": DiscreteGrid(category_class=_Health)} if carried else {}),
        functions={"utility": _health_bequest if carried else _log_wealth},
    )
    edges = (
        {"a": {"b": Periods(values=(0,))}, "b": {"done": Periods(values=(1,))}}
        if shape == "chain"
        else {"alive": {"alive": Periods(values=(0,)), "done": Periods(values=(1,))}}
    )
    first = "a" if shape == "chain" else "alive"
    return Model(
        regimes={**{name: source(name) for name in targets}, "done": done},
        n_periods=3,
        initial_nodes=InitialNodes(by_period={0: first}),
        edges=edges,
        regime_id_class=_ChainRegimeId if shape == "chain" else _CostRegimeId,
        fixed_params={"discount_factor": 0.95},
        shocks={"hcc": group} if declared else {},
    )


def _assert_parity(*, actual: FloatND, expected: FloatND, label: str) -> None:
    """Equal at float64 bit for bit, and to the precision's decimals at float32."""
    if jnp.asarray(actual).dtype == jnp.float64:
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected), err_msg=label
        )
    else:
        assert_array_almost_equal(
            actual, expected, decimal=DECIMAL_PRECISION, err_msg=label
        )


def _assert_values_integrate_node_state_values(
    *, legacy: Model, declared: Model, nodes: dict[StateName, FloatND]
) -> tuple[SolutionResult, SolutionResult]:
    legacy_solution = legacy.solve(params={}, log_level="off")
    declared_solution = declared.solve(params={}, log_level="off")
    for period, regimes in legacy_solution.values.items():
        for regime_name, value in regimes.items():
            expected = value
            for shock in reversed(legacy.state_names(regime_name=regime_name)):
                if shock in nodes:
                    expected = _integrate_node_axis(
                        model=legacy,
                        values=expected,
                        regime_name=regime_name,
                        shock=shock,
                        weights=nodes[shock],
                    )
            _assert_parity(
                actual=declared_solution.values[period][regime_name],
                expected=expected,
                label=f"V period={period} regime={regime_name}",
            )
            assert not set(nodes) & set(
                declared_solution.value_array(period=period, regime=regime_name).dims
            )
    return legacy_solution, declared_solution


def _node_paths(
    *, model: Model, n_subjects: int, shock: StateName = "hcc_transitory"
) -> FloatND:
    """Node values cycling over subjects and periods, as a (subjects, periods) path."""
    grid = jnp.asarray(
        NormalIIDProcess(
            n_points=_N_HCC_NODES, gauss_hermite=True, mu=0.0, sigma=1.0
        ).get_gridpoints()
        if shock == "hcc_transitory"
        else UniformIIDProcess(
            n_points=_SCALE_NODES, start=0.8, stop=1.2
        ).get_gridpoints()
    )
    codes = (jnp.arange(n_subjects)[:, None] + jnp.arange(model.n_periods)) % len(grid)
    return grid[codes]


def _start_frame(*, shape: str, n_subjects: int) -> dict[StateName, FloatND]:
    wealth = jnp.tile(jnp.arange(1.0, 6.0), n_subjects // 5)
    common = {
        "wealth": wealth,
        "period": jnp.zeros(wealth.shape, dtype=jnp.int32),
        "regime_id": jnp.zeros(wealth.shape, dtype=jnp.int32),
    }
    if shape in {"passive_state", "utility_shock", "aca"}:
        common["good_health"] = jnp.arange(n_subjects, dtype=jnp.int32) % 2
    return common


@pytest.mark.parametrize("shape", _EGM_SHAPES)
@pytest.mark.parametrize("solver", _SOLVERS)
def test_declared_shock_reproduces_node_state_model_for_aca_egm_shapes(
    *, solver: str, shape: str
) -> None:
    """Each ACA-shaped EGM source solves and simulates like its node-state model.

    Every published value is the node-weighted node-state value, without a node
    axis, and subjects given node draws at every date take the node-state policy.
    """
    legacy = _aca_shaped_model(solver=solver, declared=False, shape=shape)
    declared = _aca_shaped_model(solver=solver, declared=True, shape=shape)
    weights = NormalIIDProcess(
        n_points=_N_HCC_NODES, gauss_hermite=True, mu=0.0, sigma=1.0
    ).get_transition_probs()[0]
    legacy_solution, declared_solution = _assert_values_integrate_node_state_values(
        legacy=legacy, declared=declared, nodes={"hcc_transitory": weights}
    )

    n_subjects = 30
    paths = _node_paths(model=legacy, n_subjects=n_subjects)
    common = _start_frame(shape=shape, n_subjects=n_subjects)
    legacy_panel = legacy.simulate(
        params={},
        solution=legacy_solution,
        initial_conditions={**common, "hcc_transitory": paths[:, 0]},
        shock_paths={"hcc_transitory": paths},
        seed=3,
        log_level="off",
        use_categorical_codes=True,
    ).to_dataframe(use_categorical_codes=True)
    declared_panel = declared.simulate(
        params={},
        solution=declared_solution,
        initial_conditions=common,
        shock_paths={"hcc_transitory": PhysicalValues(values=paths)},
        seed=3,
        log_level="off",
        use_categorical_codes=True,
    ).to_dataframe(use_categorical_codes=True)
    for panel in (legacy_panel, declared_panel):
        panel.sort_values(["subject_id", "period"], inplace=True)
    assert len(declared_panel) == len(legacy_panel) == 3 * n_subjects
    for column in legacy_panel.columns:
        if column in {"regime", "regime_name"}:
            np.testing.assert_array_equal(
                declared_panel[column].to_numpy(), legacy_panel[column].to_numpy()
            )
            continue
        _assert_parity(
            actual=declared_panel[column].to_numpy(dtype=float),
            expected=legacy_panel[column].to_numpy(dtype=float),
            label=column,
        )


@pytest.mark.parametrize("rule", ["user_nodes", "smolyak", "product"])
def test_declared_node_rule_reproduces_node_state_model(rule: str) -> None:
    """Supplied nodes, a one-dimensional sparse rule and a product of two laws.

    Supplied Gauss-Hermite nodes and weights reproduce the Gauss-Hermite node state
    exactly. The level-one sparse normal rule is the three-point Gauss-Hermite rule.
    A product rule over a normal and a uniform shock is the two node states'
    tensor product.
    """
    process = NormalIIDProcess(
        n_points=_N_HCC_NODES, gauss_hermite=True, mu=0.0, sigma=1.0
    )
    nodes = {"hcc_transitory": process.get_transition_probs()[0]}
    group = {
        "user_nodes": ShockGroup(
            shocks={"hcc_transitory": Normal(mu=0.0, sigma=1.0)},
            nodes=UserNodes(
                nodes=np.asarray(process.get_gridpoints())[:, None],
                weights=np.asarray(nodes["hcc_transitory"]),
            ),
        ),
        "smolyak": ShockGroup(
            shocks={"hcc_transitory": Normal(mu=0.0, sigma=1.0)},
            nodes=Smolyak(level=1),
        ),
        "product": ShockGroup(
            shocks={
                "hcc_transitory": Normal(mu=0.0, sigma=1.0),
                "hcc_scale": Uniform(start=0.8, stop=1.2),
            },
            nodes=Product(
                hcc_transitory=GaussHermite(n=_N_HCC_NODES),
                hcc_scale=EqualSpacedUniform(n=_SCALE_NODES),
            ),
        ),
    }[rule]
    scaled = rule == "product"
    if scaled:
        nodes["hcc_scale"] = UniformIIDProcess(
            n_points=_SCALE_NODES, start=0.8, stop=1.2
        ).get_transition_probs()[0]
    legacy = _aca_shaped_model(
        solver="nbegm", declared=False, shape="multi_date", scaled=scaled
    )
    declared = _aca_shaped_model(
        solver="nbegm", declared=True, shape="multi_date", group=group, scaled=scaled
    )
    if rule == "smolyak":
        legacy_solution = legacy.solve(params={}, log_level="off")
        declared_solution = declared.solve(params={}, log_level="off")
        assert_array_almost_equal(
            declared_solution.values[0]["alive"],
            _integrate_node_axis(
                model=legacy,
                values=legacy_solution.values[0]["alive"],
                regime_name="alive",
                shock="hcc_transitory",
                weights=nodes["hcc_transitory"],
            ),
            decimal=DECIMAL_PRECISION,
        )
        return
    _assert_values_integrate_node_state_values(
        legacy=legacy, declared=declared, nodes=nodes
    )


@pytest.mark.parametrize(
    ("group", "feature"),
    [
        (
            ShockGroup(
                shocks={"hcc_transitory": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[-1.0], [0.0], [1.0]]),
                    weights=np.asarray([-0.1, 1.2, -0.1]),
                    signed=True,
                ),
            ),
            "signed node weights",
        ),
        (
            ShockGroup(
                shocks={
                    "hcc_transitory": Normal(mu=0.0, sigma=1.0),
                    "hcc_other": Normal(mu=0.0, sigma=1.0),
                },
                nodes=Product(
                    hcc_transitory=GaussHermite(n=3), hcc_other=GaussHermite(n=3)
                ),
                correlation=((1.0, 0.5), (0.5, 1.0)),
            ),
            "correlated shocks",
        ),
        (
            ShockGroup(
                shocks={"hcc_transitory": Normal(mu=0.0, sigma=1.0)},
                nodes=MonteCarlo(n=4, seed=1),
            ),
            "Monte Carlo rules",
        ),
        (
            ShockGroup(
                shocks={
                    "hcc_transitory": Normal(mu=0.0, sigma=1.0),
                    "hcc_other": Normal(mu=0.0, sigma=1.0),
                },
                nodes=UserNodes(
                    nodes=np.asarray([[-1.0, 0.0], [1.0, 0.0]]),
                    weights=np.asarray([0.5, 0.5]),
                ),
            ),
            "joint node rule",
        ),
        (
            ShockGroup(
                shocks={
                    "hcc_transitory": Normal(mu=0.0, sigma=1.0),
                    "hcc_state": Discrete(categories=_Health, probs=(0.5, 0.5)),
                },
                nodes=Product(hcc_transitory=GaussHermite(n=3)),
            ),
            "discrete laws",
        ),
        (
            ShockGroup(
                shocks={"hcc_transitory": Normal(mu=None, sigma=1.0)},
                nodes=Product(hcc_transitory=GaussHermite(n=3)),
            ),
            "runtime slots",
        ),
    ],
)
def test_node_state_axis_refuses_unrepresented_declarations(
    *, group: ShockGroup, feature: str
) -> None:
    """A group the node-state axis cannot represent is refused at model build."""
    with pytest.raises(ShockDeclarationError, match=feature):
        _aca_shaped_model(
            solver="nbegm", declared=True, shape="multi_date", group=group
        )


def test_saved_declared_solution_publishes_and_replays_integrated_values(
    tmp_path: Path,
) -> None:
    """An archive holds the published values and simulates like the solve it saved."""
    model = _aca_shaped_model(solver="nbegm", declared=True, shape="passive_state")
    solution = model.solve(params={}, log_level="off")
    loaded = load_solution(path=solution.save(path=tmp_path / "solution.lcm"))
    for period, regimes in solution.values.items():
        for regime_name, value in regimes.items():
            np.testing.assert_array_equal(
                np.asarray(loaded.values[period][regime_name]), np.asarray(value)
            )
            assert (
                "hcc_transitory"
                not in loaded.metadata.value_schemas[period, regime_name].axis_names
            )
    common = _start_frame(shape="passive_state", n_subjects=20)
    panels = [
        model.simulate(
            params={},
            solution=result,
            initial_conditions=common,
            seed=11,
            log_level="off",
            use_categorical_codes=True,
        ).to_dataframe(use_categorical_codes=True)
        for result in (solution, loaded)
    ]
    for panel in panels:
        panel.sort_values(["subject_id", "period"], inplace=True)
    for column in ("consumption", "wealth", "hcc_transitory", "value"):
        assert_array_almost_equal(
            panels[1][column].to_numpy(dtype=float),
            panels[0][column].to_numpy(dtype=float),
            decimal=DECIMAL_PRECISION,
            err_msg=column,
        )


@functools.cache
def _declared_shock_surfaces(
    solver: str,
) -> Mapping[str, tuple[str | tuple[str, ...], ...]]:
    """Collect what a user reads of the regimes, initial states and panel."""
    model = _health_cost_model(solver=solver, declared=True)
    wealth = jnp.linspace(1.0, 5.0, 4)
    result = model.simulate(
        params={"alive": {"total_costs": {"cost_scale": 1 / np.sqrt(3.0)}}},
        initial_conditions={
            "wealth": wealth,
            "period": jnp.zeros(wealth.shape, dtype=jnp.int32),
            "regime_id": jnp.zeros(wealth.shape, dtype=jnp.int32),
        },
        seed=3,
        log_level="off",
    )
    names = tuple(model.regime_names_to_ids)
    return {
        "state_names": tuple(model.state_names(regime_name=name) for name in names),
        "initial_state_names": tuple(
            model.initial_state_names(regime_name=name) for name in names
        ),
        "initial_state_grids": tuple(
            tuple(model.initial_state_grids(regime_name=name)) for name in names
        ),
        "regime_states": tuple(
            tuple(model.user_regimes[name].states) for name in names
        ),
        "regime_shocks": tuple(model.user_regimes[name].shocks for name in names),
        "result_state_names": result.state_names,
        "panel_columns": tuple(result.to_dataframe().columns),
    }


@pytest.mark.parametrize(
    "surface",
    [
        "state_names",
        "initial_state_names",
        "initial_state_grids",
        "regime_states",
        "regime_shocks",
        "result_state_names",
        "panel_columns",
    ],
)
def test_declared_shock_read_by_nbegm_shows_as_on_the_grid_search_route(
    surface: str,
) -> None:
    """The shock stays a declared shock: no state of its own, a column by the actions.

    The node-state axis that solves it is internal, so the regimes, the state
    listings, the initial-condition names and the simulated panel read exactly as
    they do when a GridSearch regime reads the same group.
    """
    assert (
        _declared_shock_surfaces("nbegm")[surface]
        == _declared_shock_surfaces("grid_search")[surface]
    )


def test_declared_fresh_draws_follow_the_current_iid_stream() -> None:
    """A seed realises the same draws on the node-state axis as on the current-IID path.

    A grid-search consumer keeps the current-IID route, so its realised draws are the
    stream the node-state axis of an EGM consumer must reproduce.
    """
    params = {"alive": {"total_costs": {"cost_scale": 0.5}}}
    initial = {
        "wealth": jnp.tile(jnp.arange(1.0, 6.0), 8),
        "period": jnp.zeros(40, dtype=jnp.int32),
        "regime_id": jnp.zeros(40, dtype=jnp.int32),
    }
    current_iid = _health_cost_model(solver="grid_search", declared=True)
    adapted = _health_cost_model(solver="nbegm", declared=True)
    draws = [
        model.simulate(
            params=params, initial_conditions=initial, seed=17, log_level="off"
        )
        .to_dataframe()
        .sort_values(["subject_id", "period"])
        .query("period == 0")["hcc_transitory"]
        .to_numpy(dtype=float)
        for model in (current_iid, adapted)
    ]
    assert np.unique(draws[0]).size == 40
    assert_array_almost_equal(draws[1], draws[0], decimal=DECIMAL_PRECISION)


def test_declared_shock_inputs_keep_their_tags() -> None:
    """A base entry maps through the law at entry; a physical path wins everywhere."""
    group = ShockGroup(
        shocks={"hcc_transitory": Normal(mu=0.2, sigma=0.5)},
        nodes=Product(hcc_transitory=GaussHermite(n=_N_HCC_NODES)),
    )
    model = _aca_shaped_model(
        solver="nbegm", declared=True, shape="multi_date", group=group
    )
    solution = model.solve(params={}, log_level="off")
    common = _start_frame(shape="multi_date", n_subjects=10)
    base = jnp.linspace(-1.5, 1.5, 10)
    path = jnp.stack([jnp.linspace(-0.4, 0.6, 10), jnp.linspace(0.3, -0.9, 10)], 1)

    entry = {"hcc_transitory": BaseInnovations(values=base)}

    def shocks(*, path_record: PhysicalValues | None) -> np.ndarray:
        panel = model.simulate(
            params={},
            solution=solution,
            initial_conditions={**common, **entry},
            shock_paths=None
            if path_record is None
            else {"hcc_transitory": path_record},
            seed=5,
            log_level="off",
        ).to_dataframe()
        panel = panel.sort_values(["subject_id", "period"])
        return np.stack(
            [
                panel.query(f"period == {period}")["hcc_transitory"].to_numpy(float)
                for period in (0, 1)
            ],
            axis=1,
        )

    entered = shocks(path_record=None)
    assert_array_almost_equal(
        entered[:, 0], 0.2 + 0.5 * base, decimal=DECIMAL_PRECISION
    )
    assert np.all(entered[:, 1] != 0.2 + 0.5 * base)
    supplied = shocks(path_record=PhysicalValues(values=path))
    assert_array_almost_equal(supplied, path, decimal=DECIMAL_PRECISION)


def test_declared_physical_value_outside_its_support_is_refused() -> None:
    """A supplied uniform draw outside the law's interval is not simulated."""
    group = ShockGroup(
        shocks={"hcc_transitory": Uniform(start=0.0, stop=1.0)},
        nodes=Product(hcc_transitory=EqualSpacedUniform(n=3)),
    )
    model = _aca_shaped_model(
        solver="nbegm", declared=True, shape="multi_date", group=group
    )
    common = _start_frame(shape="multi_date", n_subjects=5)
    with pytest.raises(InvalidSimulationInputError, match="outside the law's support"):
        model.simulate(
            params={},
            initial_conditions={
                **common,
                "hcc_transitory": PhysicalValues(values=jnp.full(5, 2.0)),
            },
            seed=5,
            log_level="off",
        )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize("solver", ["negm", "nnbegm"])
def test_declared_shock_solves_nested_egm_on_the_node_axis(*, solver: str) -> None:
    """Keeper and adjuster are chosen per shock node before the published average."""
    declarations = _declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, NestedConsumptionSavingsRegime)
    assert isinstance(child.solver, NEGM)
    if solver == "nnbegm":
        child = replace(
            child,
            solver=NNBEGM(
                inner=NBEGM(
                    savings_grid=child.solver.inner.savings_grid, jump_read="bridged"
                ),
                outer_search=FiniteOuterGrid(grid=child.solver.outer_grid),
            ),
        )
    model = Model(
        **{
            field.name: getattr(declarations, field.name)
            for field in fields(declarations)
            if field.init and field.name != "regimes"
        },
        regimes={**declarations.regimes, "child": child},
    )
    solution = model.solve(params=_params(), log_level="off")
    assert _oracle(wealth=2.0, housing=1.0, shock=0.0)[6] == 0
    assert _oracle(wealth=2.0, housing=1.0, shock=3.0)[6] == 1
    expected = np.mean(
        [_oracle(wealth=2.0, housing=1.0, shock=z)[0] for z in (0.0, 3.0)]
    )
    assert np.shape(solution.values[1]["child"]) == (3, 3)
    np.testing.assert_almost_equal(
        solution.values[1]["child"][1, 1], expected, decimal=DECIMAL_PRECISION
    )
    physical = jnp.asarray([0.0, 2.0, 3.0])
    result = model.simulate(
        params=_params(),
        solution=solution,
        log_level="off",
        seed=31,
        initial_conditions={
            "wealth": jnp.full((3,), 2.0),
            "housing": jnp.ones((3,)),
            "age": jnp.ones((3,)),
            "regime_id": jnp.full((3,), model.regime_names_to_ids["child"], jnp.int32),
            "eps": PhysicalValues(values=physical),
        },
    )
    panel = result.to_dataframe().query("period == 1")
    assert_array_almost_equal(panel["eps"], physical, decimal=DECIMAL_PRECISION)
    assert np.all(np.isfinite(panel["value"]))


def test_finite_nnbegm_extrapolates_a_draw_beyond_its_outer_nodes() -> None:
    """A draw outside the node hull reads the policy along the shock's edge segment.

    The rule's nodes sit inside the law's support, so a draw can lie beyond the
    last node. The finite NNBEGM replay reads it as it reads a draw between nodes,
    linearly in the node coordinate, here extrapolating the last segment: the
    consumption of a subject at the same wealth and housing continues the line
    through the consumptions at the two nodes.
    """
    declarations = _declarations()
    model = Model(
        **{
            field.name: getattr(declarations, field.name)
            for field in fields(declarations)
            if field.init and field.name not in ("regimes", "shocks")
        },
        regimes={**declarations.regimes, "child": _finite_nnbegm_child()},
        shocks={
            "income": ShockGroup(
                shocks={"eps": Uniform(start=0.0, stop=3.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[0.25], [0.75]]), weights=np.asarray([0.5, 0.5])
                ),
            )
        },
    )
    # The base nodes 0.25 and 0.75 of Uniform(0, 3) are the physical nodes 0.75
    # and 2.25; 2.7 lies beyond the last one.
    physical = jnp.asarray([0.75, 2.25, 2.7])
    panel = (
        model.simulate(
            params=_params(),
            log_level="off",
            seed=31,
            initial_conditions={
                "wealth": jnp.full((3,), 2.0),
                "housing": jnp.zeros((3,)),
                "age": jnp.ones((3,)),
                "regime_id": jnp.full(
                    (3,), model.regime_names_to_ids["child"], jnp.int32
                ),
                "eps": PhysicalValues(values=physical),
            },
        )
        .to_dataframe()
        .query("period == 1")
        .sort_values("subject_id")
    )
    at_nodes = panel["consumption"].to_numpy(dtype=float)
    np.testing.assert_almost_equal(
        at_nodes[2],
        at_nodes[1] + (2.7 - 2.25) / (2.25 - 0.75) * (at_nodes[1] - at_nodes[0]),
        decimal=DECIMAL_PRECISION,
    )


def _finite_nnbegm_child() -> NestedConsumptionSavingsRegime:
    """The toy's housing regime solved by NNBEGM over its finite outer grid."""
    child = _declarations().regimes["child"]
    assert isinstance(child, NestedConsumptionSavingsRegime)
    assert isinstance(child.solver, NEGM)
    return replace(
        child,
        solver=NNBEGM(
            inner=NBEGM(
                savings_grid=child.solver.inner.savings_grid, jump_read="bridged"
            ),
            outer_search=FiniteOuterGrid(grid=child.solver.outer_grid),
        ),
    )


@categorical(ordered=False)
class _ParentChildRegimeId:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


def _parent_resources(*, wealth: ContinuousState) -> FloatND:
    return wealth + 5.0


def _landing_housing(*, landing_q: FloatND) -> ContinuousState:
    return landing_q


def _finite_nnbegm_model(*, with_parent: bool) -> Model:
    """The finite NNBEGM housing regime at age 1, optionally after a GridSearch one.

    The GridSearch parent saves at most `wealth + 5 - 0.1`, far above the child's
    wealth grid, which ends at 3.
    """
    declarations = _declarations()
    regimes = {**declarations.regimes, "child": _finite_nnbegm_child()}
    edges: dict[RegimeName, dict[RegimeName, tuple[int, ...]]] = {
        "child": {"done": (1,)}
    }
    starts = {1: "child"}
    if with_parent:
        regimes["parent"] = Regime(
            states={"wealth": IrregSpacedGrid(points=np.asarray([0.5, 1.75]))},
            actions={"consumption": IrregSpacedGrid(points=np.asarray([0.1, 5.0]))},
            functions={
                "utility": _log_consumption,
                "resources": _parent_resources,
                "savings": _savings,
            },
            constraints={"affordable": _affordable},
            state_transitions={
                "wealth": {"child": _next_wealth_from_savings},
                "housing": {"child": _landing_housing},
            },
        )
        edges["parent"] = {"child": (0,)}
        starts[0] = "parent"
    return Model(
        **{
            field.name: getattr(declarations, field.name)
            for field in fields(declarations)
            if field.init
            and field.name
            not in ("regimes", "edges", "initial_nodes", "regime_id_class")
        },
        regimes=regimes,
        edges=edges,
        initial_nodes=InitialNodes(by_age=starts),
        regime_id_class=_ParentChildRegimeId if with_parent else _ChildRegimeId,
    )


@categorical(ordered=False)
class _ChildRegimeId:
    child: ScalarInt
    done: ScalarInt


def _simulate_finite_nnbegm(
    *, model: Model, regime: RegimeName, wealth: FloatND, eps: FloatND | None
) -> SimulationResult:
    """Simulate from `regime`; `eps` is the entry draw of a start in the child."""
    n_subjects = wealth.shape[0]
    return model.simulate(
        params={**_params(), "parent": {"next_housing": {"landing_q": 0.0}}}
        if "parent" in model.regime_names_to_ids
        else _params(),
        log_level="off",
        seed=3,
        initial_conditions={
            "wealth": wealth,
            "housing": jnp.zeros((n_subjects,)),
            "age": jnp.full((n_subjects,), 0.0 if regime == "parent" else 1.0),
            "regime_id": jnp.full(
                (n_subjects,), model.regime_names_to_ids[regime], jnp.int32
            ),
            **({} if eps is None else {"eps": PhysicalValues(values=eps)}),
        },
    )


def test_finite_nnbegm_replays_a_start_on_its_wealth_grid_exactly() -> None:
    """A subject inside the solved wealth grid takes the closed-form consumption."""
    panel = (
        _simulate_finite_nnbegm(
            model=_finite_nnbegm_model(with_parent=False),
            regime="child",
            wealth=jnp.full((2,), 2.5),
            eps=jnp.asarray([0.0, 3.0]),
        )
        .to_dataframe()
        .query("period == 1")
    )
    assert_array_almost_equal(
        panel.sort_values("subject_id")["consumption"].to_numpy(dtype=float),
        [_oracle(wealth=2.5, housing=0.0, shock=z)[2] for z in (0.0, 3.0)],
        decimal=DECIMAL_PRECISION,
    )


def test_finite_nnbegm_refuses_a_start_off_its_wealth_grid() -> None:
    """A start above the solved wealth grid is refused, naming the grid to widen."""
    with pytest.raises(
        OffGridStateError,
        match=(
            r"(?s)Regime 'child', period 1:.*'wealth'.*\[5\.9, 5\.9\].*"
            r"grid \[1\.5, 3\].*Widen the grid of 'wealth' in regime 'child'"
        ),
    ):
        _simulate_finite_nnbegm(
            model=_finite_nnbegm_model(with_parent=False),
            regime="child",
            wealth=jnp.asarray([2.5, 5.9]),
            eps=jnp.asarray([0.0, 3.0]),
        )


def test_finite_nnbegm_refuses_a_grid_search_parents_landing_off_its_grid() -> None:
    """A GridSearch parent saving beyond the child's grid is refused at the child.

    The message points at the predecessor's landings as the cause.
    """
    with pytest.raises(
        OffGridStateError,
        match=(
            r"(?s)Regime 'child', period 1:.*'wealth'.*grid \[1\.5, 3\].*"
            r"predecessor regime.*GridSearch or NBEGM"
        ),
    ):
        _simulate_finite_nnbegm(
            model=_finite_nnbegm_model(with_parent=True),
            regime="parent",
            wealth=jnp.ones((3,)),
            eps=None,
        )
