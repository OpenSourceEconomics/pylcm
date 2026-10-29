"""Declared current-IID shocks reproduce the node-state IID models of actual consumers.

An IID process declared as a state is a value axis: the value array holds one slice
per quadrature node. The same law declared as a `ShockGroup` is integrated within the
period, so its value array equals the node-weighted average of those slices, and a
subject whose draw is supplied at a node takes the node-state model's policy.
"""

from collections.abc import Callable

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
    NormalIIDProcess,
    Periods,
    Regime,
    UniformIIDProcess,
    categorical,
    post_decision_lower_bound,
)
from lcm.shocks import (
    EqualSpacedUniform,
    GaussHermite,
    Normal,
    PhysicalValues,
    Product,
    ShockGroup,
    Uniform,
)
from lcm.solvers import DCEGM, NBEGM
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
