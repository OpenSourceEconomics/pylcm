"""State-dependent borrowing limits compared with a dense GridSearch solve.

Plain EGM varies its floor with age and parameters. DC-EGM carries Markov
income; the taxed NB-EGM fixture also declares a kinked liquid-wealth tax.
NB-EGM uses its required `liquid` state name; EGM and DC-EGM use `wealth`.

The GridSearch comparison writes the floor as an independent predicate and
uses a different optimization algorithm. Model processing, transitions,
continuation interpolation and simulation scoring remain shared, so this is a
refinement check rather than independent coverage of those components.
"""

import functools
from collections.abc import Callable, Mapping

import jax.numpy as jnp
import numpy as np
import pytest
from dags import rename_arguments

import lcm
from lcm import (
    AgeGrid,
    AgeRange,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    StochasticTransition,
    categorical,
)
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    post_decision_lower_bound,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.solvers import DCEGM, EGM, NBEGM, GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserFunction,
    UserParams,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON

_TAXED_MARGIN = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)

# Share of current income the household may borrow against.
_LIMIT_SHARE = 0.5
# The constant limit of the float-versus-function comparison.
_CONSTANT_LIMIT = -0.5
# Income by Markov node (low, high) and by age for the age-only EGM variant.
_INCOME_LOW = 1.0
_INCOME_HIGH = 2.0

# Every reachable next wealth is `savings + income >= (1 - share) * income`, so
# 0.5 is the lowest reachable node in every variant: no continuation read leaves
# the state grid on the left.
_WEALTH_GRID = LinSpacedGrid(start=0.5, stop=20.0, n_points=120)
# The same dense consumption grid on both sides: DC-EGM simulation may
# re-optimise on it, and the oracle's resolution error is bounded by its step.
_CONSUMPTION_GRID = LinSpacedGrid(start=0.01, stop=25.0, n_points=2500)
_CONSUMPTION_STEP = (25.0 - 0.01) / 2499


def _savings_points(start: float) -> tuple[float, ...]:
    """Savings nodes clustered cubically toward the limit, where V curves most."""
    return tuple(start + 22.0 * (i / 149) ** 3 for i in range(150))


_SOLVERS: Mapping[str, Callable[[IrregSpacedGrid], EGM | DCEGM | NBEGM]] = {
    "egm": lambda grid: EGM(savings_grid=grid),
    "dcegm": lambda grid: DCEGM(savings_grid=grid),
    "nbegm": lambda grid: NBEGM(savings_grid=grid),
}

# What `lower=` receives: today's float, or a function of fixed states and params.
type _Lower = float | UserFunction
type _Values = Mapping[int, Mapping[str, FloatND]]

_EXACT = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
_SOLVER_PARAMS = [
    pytest.param("egm", id="egm"),
    pytest.param("dcegm", id="dcegm", marks=_EXACT),
    pytest.param("nbegm", id="nbegm", marks=_EXACT),
]


@categorical(ordered=False)
class RegimeId:
    saving: ScalarInt
    done: ScalarInt


@categorical(ordered=True)
class Income:
    low: ScalarInt
    high: ScalarInt


@lcm.piecewise_affine(
    output="tax",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint(threshold="tax_exemption", kind="continuous_kink"),
    ),
)
def tax(*, liquid: ContinuousState, tax_rate: float, tax_exemption: float) -> FloatND:
    """Tax only wealth above the exemption."""
    return tax_rate * jnp.maximum(liquid - tax_exemption, 0.0)


def taxed_resources(*, liquid: ContinuousState, tax: FloatND) -> FloatND:
    return liquid - tax


def taxed_savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def terminal_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def savings(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def income_of_node(income: DiscreteState) -> FloatND:
    return jnp.where(income == Income.high, _INCOME_HIGH, _INCOME_LOW)


def income_of_age(age: int) -> FloatND:
    return jnp.where(age == 0, _INCOME_LOW, _INCOME_HIGH)


def next_wealth_constant(savings: FloatND) -> ContinuousState:
    return savings + _INCOME_LOW


def next_wealth_age(*, savings: FloatND, age: int) -> ContinuousState:
    return savings + income_of_age(age)


def next_wealth_income(*, savings: FloatND, income: DiscreteState) -> ContinuousState:
    return savings + income_of_node(income)


def income_transition(income: DiscreteState) -> FloatND:
    """Persistent two-node income: stay with probability 0.8."""
    p_high = jnp.where(income == Income.high, 0.8, 0.2)
    return jnp.stack([1.0 - p_high, p_high])


def constant_lower(limit_level: float) -> FloatND:
    return jnp.asarray(limit_level)


def age_lower(*, age: int, limit_share: float) -> FloatND:
    return -limit_share * income_of_age(age)


def income_lower(*, income: DiscreteState, limit_share: float) -> FloatND:
    return -limit_share * income_of_node(income)


def liquid_lower(*, wealth: ContinuousState, limit_share: float) -> FloatND:
    return -limit_share * wealth


def oracle_constant_limit(savings: FloatND) -> BoolND:
    return savings >= _CONSTANT_LIMIT


def oracle_age_limit(*, savings: FloatND, age: int) -> BoolND:
    return savings >= -_LIMIT_SHARE * income_of_age(age)


def oracle_income_limit(*, savings: FloatND, income: DiscreteState) -> BoolND:
    return savings >= -_LIMIT_SHARE * income_of_node(income)


def _model(*, solver: str, limit: str, lower: _Lower, taxed: bool) -> Model:
    """A three-period consumption-saving model; periods 0 and 1 choose.

    Period 0 reads the period-1 solution of the same regime, so the continuation
    read crosses cells whose savings grids differ when the limit varies by cell.

    Args:
        solver: `"egm"`, `"dcegm"`, `"nbegm"` or `"grid"` (the oracle).
        limit: `"constant"`, `"age"` or `"income"`.
        lower: The `lower=` passed to `post_decision_lower_bound`; ignored by the
            oracle, which writes the limit as its own predicate.
        taxed: Whether resources are wealth net of the kinked tax.

    """
    next_wealth = {
        "constant": next_wealth_constant,
        "age": next_wealth_age,
        "income": next_wealth_income,
    }[limit]
    state_name = "liquid" if solver == "nbegm" or taxed else "wealth"
    states: dict[str, LinSpacedGrid | DiscreteGrid] = {state_name: _WEALTH_GRID}
    transitions: dict[str, dict[str, UserFunction | StochasticTransition]] = {
        state_name: {"saving": next_wealth, "done": next_wealth}
    }
    if limit == "income":
        states["income"] = DiscreteGrid(category_class=Income)
        transitions["income"] = {"saving": StochasticTransition(func=income_transition)}
    functions: dict[str, UserFunction] = {"utility": utility}
    if taxed:
        functions.update(
            {"tax": tax, "resources": taxed_resources, "savings": taxed_savings}
        )
    else:
        functions["savings"] = (
            rename_arguments(savings, mapper={"wealth": "liquid"})
            if state_name == "liquid"
            else savings
        )
    margin = (
        _TAXED_MARGIN
        if taxed
        else LiquidMargin(
            state=state_name,
            action="consumption",
            resources=state_name,
            post_decision_state="savings",
        )
    )
    if solver == "grid":
        oracle = {
            "constant": oracle_constant_limit,
            "age": oracle_age_limit,
            "income": oracle_income_limit,
        }[limit]
        saving = Regime(
            actions={"consumption": _CONSUMPTION_GRID},
            states=states,
            state_transitions=transitions,
            constraints={"borrowing_limit": oracle},
            functions=functions,
            solver=GridSearch(),
        )
    else:
        grid_start = lower if isinstance(lower, float) else 0.0
        if solver != "egm":
            functions["inverse_marginal_utility"] = inverse_marginal_utility
        saving = ConsumptionSavingsRegime(
            actions={"consumption": _CONSUMPTION_GRID},
            states=states,
            state_transitions=transitions,
            constraints={
                "borrowing_limit": post_decision_lower_bound(
                    margin=margin,
                    lower=lower,
                )
            },
            functions=functions,
            liquid=margin,
            solver=_SOLVERS[solver](
                IrregSpacedGrid(points=_savings_points(grid_start))
            ),
        )
    done = Regime(
        states={state_name: _WEALTH_GRID},
        functions={
            "utility": (
                rename_arguments(terminal_utility, mapper={"wealth": "liquid"})
                if state_name == "liquid"
                else terminal_utility
            )
        },
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving, "done": done},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={"saving": {"saving": AgeRange(exclusive_stop=1), "done": 1}},
        regime_id_class=RegimeId,
        initial_nodes={0: "saving"},
    )


def _params(*, limit_param: str | None, taxed: bool) -> UserParams:
    params: dict[str, float] = {"discount_factor": 0.95}
    if taxed:
        params.update({"tax_rate": 0.2, "tax_exemption": 10.0})
    if limit_param == "limit_level":
        params["limit_level"] = _CONSTANT_LIMIT
    elif limit_param == "limit_share":
        params["limit_share"] = _LIMIT_SHARE
    return params


@functools.cache
def _values(*, solver: str, limit: str, lower_kind: str, taxed: bool) -> _Values:
    lower, limit_param = _lower_and_param(limit=limit, lower_kind=lower_kind)
    model = _model(solver=solver, limit=limit, lower=lower, taxed=taxed)
    return model.solve(
        params=_params(
            limit_param=None if solver == "grid" else limit_param, taxed=taxed
        ),
        log_level="off",
    ).values


def _lower_and_param(*, limit: str, lower_kind: str) -> tuple[_Lower, str | None]:
    if lower_kind == "float":
        return _CONSTANT_LIMIT, None
    if limit == "constant":
        return constant_lower, "limit_level"
    return (age_lower if limit == "age" else income_lower), "limit_share"


def _period0_consumption(
    *, solver: str, limit: str, lower_kind: str, taxed: bool
) -> np.ndarray:
    """Simulated period-0 consumption on a fixed panel of wealth and income."""
    lower, limit_param = _lower_and_param(limit=limit, lower_kind=lower_kind)
    model = _model(solver=solver, limit=limit, lower=lower, taxed=taxed)
    params = _params(limit_param=None if solver == "grid" else limit_param, taxed=taxed)
    wealth = np.linspace(0.5, 15.0, 30)
    initial: dict[str, jnp.ndarray] = {
        "age": jnp.zeros(60),
        "regime_id": jnp.zeros(60, dtype=int),
    }
    initial["liquid" if solver == "nbegm" or taxed else "wealth"] = jnp.asarray(
        np.concatenate([wealth, wealth])
    )
    if limit == "income":
        initial["income"] = jnp.asarray([Income.low] * 30 + [Income.high] * 30)
    simulated = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=model.solve(params=params, log_level="off"),
        seed=528,
        log_level="off",
    ).to_dataframe(use_labels=False)
    period0 = simulated.query("period == 0").sort_values("subject_id")
    return period0["consumption"].to_numpy()


def _solved_periods(values: _Values) -> list[int]:
    return [period for period in sorted(values) if "saving" in values[period]]


@pytest.mark.parametrize("solver", _SOLVER_PARAMS)
def test_constant_lower_function_reproduces_the_float_declaration(solver: str) -> None:
    """A `lower` function returning a constant solves the float declaration's model.

    The shifted grid `lower + (grid - grid[0])` is the float model's grid up to
    the rounding of one subtraction and one addition per node, so values agree to
    a small multiple of the working epsilon.
    """
    taxed = solver == "nbegm"
    want = _values(solver=solver, limit="constant", lower_kind="float", taxed=taxed)
    got = _values(solver=solver, limit="constant", lower_kind="function", taxed=taxed)
    for period in _solved_periods(want):
        reference = np.asarray(want[period]["saving"])
        tolerance = 1e3 * float(np.finfo(reference.dtype).eps)
        np.testing.assert_allclose(
            np.asarray(got[period]["saving"]),
            reference,
            rtol=tolerance,
            atol=tolerance,
            err_msg=f"period={period}",
        )


_AGREEMENT_CASES = [
    pytest.param("egm", "age", id="egm-age"),
    pytest.param("dcegm", "income", id="dcegm-income", marks=_EXACT),
    pytest.param("nbegm", "income", id="nbegm-income", marks=_EXACT),
]


@pytest.mark.parametrize(("solver", "limit"), _AGREEMENT_CASES)
def test_state_dependent_limit_value_agrees_with_grid_search(
    *, solver: str, limit: str
) -> None:
    """The value under a state-dependent limit matches the dense brute-force solve.

    All wealth nodes are compared, including the ones where the limit binds:
    that region is the feature. The oracle interpolates next-period V linearly
    in wealth and picks consumption on a 0.01-step grid; the tolerances are the
    ones the float-limit control in this module meets on the same grids.
    """
    taxed = solver == "nbegm"
    got = _values(solver=solver, limit=limit, lower_kind="function", taxed=taxed)
    want = _values(solver="grid", limit=limit, lower_kind="function", taxed=taxed)
    for period in _solved_periods(want):
        np.testing.assert_allclose(
            np.asarray(got[period]["saving"]),
            np.asarray(want[period]["saving"]),
            rtol=1e-3,
            atol=5e-3,
            err_msg=f"period={period}",
        )


_POLICY_AGREEMENT_CASES = [
    pytest.param("egm", "age", id="egm-age"),
    pytest.param("dcegm", "income", id="dcegm-income", marks=_EXACT),
]


@pytest.mark.parametrize(("solver", "limit"), _POLICY_AGREEMENT_CASES)
def test_state_dependent_limit_policy_agrees_with_grid_search(
    *, solver: str, limit: str
) -> None:
    """Simulated period-0 consumption matches brute force to a few grid steps.

    The panel spans wealth from the lowest node, where the limit binds and
    consumption is `wealth - lower(cell)`, to well inside the interior.
    """
    taxed = solver == "nbegm"
    got = _period0_consumption(
        solver=solver, limit=limit, lower_kind="function", taxed=taxed
    )
    want = _period0_consumption(
        solver="grid", limit=limit, lower_kind="function", taxed=taxed
    )
    np.testing.assert_allclose(got, want, rtol=0.0, atol=3 * _CONSUMPTION_STEP)


@pytest.mark.parametrize("solver", _SOLVER_PARAMS)
def test_float_limit_control_agrees_with_grid_search(solver: str) -> None:
    """A literal floor meets the dense GridSearch value agreement tolerance."""
    taxed = solver == "nbegm"
    got = _values(solver=solver, limit="constant", lower_kind="float", taxed=taxed)
    want = _values(solver="grid", limit="constant", lower_kind="float", taxed=taxed)
    for period in _solved_periods(want):
        np.testing.assert_allclose(
            np.asarray(got[period]["saving"]),
            np.asarray(want[period]["saving"]),
            rtol=1e-3,
            atol=5e-3,
            err_msg=f"period={period}",
        )


@pytest.mark.parametrize("solver", _SOLVER_PARAMS)
def test_lower_reading_the_liquid_state_is_refused_at_model_build(
    solver: str,
) -> None:
    """A `lower` reading the liquid state is not fixed during the decision.

    The shifted grid needs one limit per cell before the Euler inversion, so a
    limit that moves with the liquid state cannot be represented; the model is
    refused, naming the offending argument.
    """
    state_name = "liquid" if solver == "nbegm" else "wealth"
    lower = (
        rename_arguments(liquid_lower, mapper={"wealth": "liquid"})
        if solver == "nbegm"
        else liquid_lower
    )
    with pytest.raises(
        ModelInitializationError, match=rf"(?s)borrowing_limit.*{state_name}"
    ):
        _model(solver=solver, limit="constant", lower=lower, taxed=False)
