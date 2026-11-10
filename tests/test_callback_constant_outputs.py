"""User functions may return Python scalars where the engine expects arrays.

A felicity (under grid search and every endogenous-grid family), a constraint,
a next state, a regime-transition probability, a stochastic weight, an exit
utility adjustment, exit availability, a gate or participation predicate, a
fallback projection, a joint-transition output or probability, a current-IID
law parameter and a borrowing floor may each return `0`, `0.0`, `True` or an
array. Each solves and simulates to the same values as its array-returning
twin, at the active precision.
"""

from collections.abc import Callable, Mapping
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    DiscreteGrid,
    Exit,
    ExitTarget,
    Gate,
    InitialNodes,
    IrregSpacedGrid,
    JointTransition,
    LinSpacedGrid,
    Model,
    ParticipationCheck,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
    post_decision_lower_bound,
    ref,
)
from lcm.collective import CollectiveUtility, ValueDependentConstraint
from lcm.exceptions import InvalidValueFunctionError
from lcm.regime import StateTransitionEntry
from lcm.shocks import Discrete, GaussHermite, Normal, Product, ShockGroup
from lcm.transition import TransitionLaw
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserFunction,
    UserFunctionResult,
)
from tests.collective_fixtures import (
    AGES,
    STATELESS_TARGET_V_PERIOD_0,
    STATELESS_TARGET_V_TERMINAL,
    CoupleRegimeId,
    make_stateless_collective_target_model,
)
from tests.constraints.test_dynamic_lower_bound import LIQUID
from tests.constraints.test_dynamic_lower_bound import _model as _floor_model
from tests.solution import test_egm_model_validation as egm_fixture
from tests.test_models import dcegm_paper_twin as dcegm_twin
from tests.test_models import nbegm_tax_toy as tax_toy
from tests.test_models.nbegm_common import make_alive_dead_model, resolve_solver


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    final: ScalarInt


@categorical(ordered=True)
class Health:
    bad: ScalarInt
    good: ScalarInt


_GRID = IrregSpacedGrid(points=jnp.asarray([1.0, 2.0]))
_FORMS = ("python_int", "python_float", "python_bool", "int_array", "float_array")


def _constant(*, form: str, value: int) -> UserFunctionResult:
    """Spell `value` in one of the return forms a user function may use."""
    if form == "python_int":
        return value
    if form == "python_float":
        return float(value)
    if form == "python_bool":
        return bool(value)
    if form == "int_array":
        return jnp.asarray(value, dtype=jnp.int32)
    return jnp.asarray(float(value))


def _identity_wealth(*, wealth: ContinuousState) -> ContinuousState:
    return wealth


def _wealth_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _zero_array_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth * 0.0


def _two_period_model(
    *,
    alive_utility: UserFunction = _zero_array_utility,
    final_utility: UserFunction = _zero_array_utility,
    next_wealth: UserFunction = _identity_wealth,
    constraints: dict[str, UserFunction] | None = None,
    enable_jit: bool = True,
) -> Model:
    alive = Regime(
        states={"wealth": _GRID},
        state_transitions={"wealth": next_wealth},
        functions={"utility": alive_utility},
        constraints={} if constraints is None else constraints,
    )
    final = Regime(
        states={"wealth": _GRID},
        functions={"utility": final_utility},
        constraints={} if constraints is None else constraints,
    )
    return Model(
        regimes={"alive": alive, "final": final},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "alive"},
        edges={"alive": {"final": 0}},
        enable_jit=enable_jit,
    )


_PARAMS = {"alive": {"koopmans_aggregator": {"discount_factor": 1.0}}, "final": {}}
_INITIAL = {
    "age": jnp.array([0.0, 0.0]),
    "wealth": jnp.array([1.0, 2.0]),
    "regime_id": jnp.array([RegimeId.alive, RegimeId.alive]),
}


def _assert_values(*, model: Model, alive: np.ndarray, final: np.ndarray) -> None:
    values = model.solve(params=_PARAMS, log_level="off").values
    for got, expected in ((values[0]["alive"], alive), (values[1]["final"], final)):
        assert jnp.issubdtype(got.dtype, jnp.floating)
        np.testing.assert_array_equal(np.asarray(got), expected)


def _simulated(model: Model) -> pd.DataFrame:
    return (
        model.simulate(params=_PARAMS, initial_conditions=_INITIAL, log_level="off")
        .to_dataframe()
        .sort_values(["subject_id", "period"])
        .reset_index(drop=True)
    )


@pytest.mark.parametrize("value", [-1, 0, 1])
@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("enable_jit", [False, True])
def test_constant_felicity_solves_to_its_literal_sum(
    *, value: int, form: str, enable_jit: bool
) -> None:
    """A constant felicity `c` gives terminal value `c` and earlier value `2c`."""
    if form == "python_bool" and value == -1:
        pytest.skip("A Boolean spells only 0 and 1.")
    raw = _constant(form=form, value=value)

    def utility(*, wealth: ContinuousState) -> FloatND:  # noqa: ARG001
        return raw  # ty: ignore[invalid-return-type]

    model = _two_period_model(
        alive_utility=utility, final_utility=utility, enable_jit=enable_jit
    )
    _assert_values(model=model, alive=np.full(2, 2.0 * value), final=np.full(2, value))


@pytest.mark.parametrize("form", _FORMS)
def test_constant_felicity_simulates_like_its_array_twin(*, form: str) -> None:
    """Simulated values of a constant felicity equal those of its array twin."""
    raw = _constant(form=form, value=1)

    def utility(*, wealth: ContinuousState) -> FloatND:  # noqa: ARG001
        return raw  # ty: ignore[invalid-return-type]

    def twin(*, wealth: ContinuousState) -> FloatND:
        return wealth * 0.0 + 1.0

    got = _simulated(_two_period_model(alive_utility=utility, final_utility=utility))
    expected = _simulated(_two_period_model(alive_utility=twin, final_utility=twin))
    np.testing.assert_array_equal(got["value"], expected["value"])
    np.testing.assert_array_equal(got["value"], [2.0, 1.0, 2.0, 1.0])


def _integer_utility_f(work: DiscreteAction) -> FloatND:
    return 10 * work


def _integer_utility_m(work: DiscreteAction) -> FloatND:
    return 5 * (1 - work)


def test_integer_collective_felicities_match_their_float_twins() -> None:
    """Integer-valued stakeholder felicities solve like their float spellings."""
    couple, couple_terminal = (
        make_stateless_collective_target_model()[0].user_regimes[name]
        for name in ("couple", "couple_terminal")
    )
    model = Model(
        regimes={
            "couple": couple,
            "couple_terminal": couple_terminal.replace(
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _integer_utility_f, "m": _integer_utility_m}
                    )
                }
            ),
        },
        ages=AGES,
        regime_id_class=CoupleRegimeId,
        initial_nodes={0: "couple"},
        edges={"couple": {"couple_terminal": 0}},
    )
    _, params = make_stateless_collective_target_model()
    values = model.solve(params=params, log_level="off").values
    np.testing.assert_array_equal(
        values[1]["couple_terminal"], STATELESS_TARGET_V_TERMINAL
    )
    np.testing.assert_allclose(
        values[0]["couple"], STATELESS_TARGET_V_PERIOD_0, rtol=1e-6
    )


def test_nan_diagnostics_read_a_constant_felicity() -> None:
    """The NaN diagnostics evaluate a regime whose felicity returns `0`."""

    def utility(*, wealth: ContinuousState) -> FloatND:  # noqa: ARG001
        return 0  # ty: ignore[invalid-return-type]

    model = _two_period_model(alive_utility=utility)
    params = {
        "alive": {"koopmans_aggregator": {"discount_factor": float("nan")}},
        "final": {},
    }
    with pytest.raises(InvalidValueFunctionError) as exc_info:
        model.solve(params=params, log_level="off")
    diagnostics = exc_info.value.diagnostics
    assert diagnostics is not None
    fractions = cast("Mapping[str, Mapping[str, float]]", diagnostics)
    assert fractions["U_nan_fraction"]["overall"] == 0.0
    assert fractions["Q_nan_fraction"]["overall"] == 1.0


@pytest.mark.parametrize("form", ["python_bool", "python_int", "bool_array"])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_constant_constraint_admits_every_action(
    *, form: str, enable_jit: bool
) -> None:
    """A constraint returning `True` leaves the unconstrained values unchanged."""
    raw = jnp.asarray(a=True) if form == "bool_array" else _constant(form=form, value=1)

    def allowed() -> BoolND:
        return raw  # ty: ignore[invalid-return-type]

    model = _two_period_model(
        final_utility=_wealth_utility,
        constraints={"allowed": allowed},
        enable_jit=enable_jit,
    )
    _assert_values(model=model, alive=np.array([1.0, 2.0]), final=np.array([1.0, 2.0]))
    df = _simulated(model)
    np.testing.assert_array_equal(df["value"], [1.0, 1.0, 2.0, 2.0])


@pytest.mark.parametrize("form", _FORMS)
def test_constant_continuous_next_state_lands_on_its_float_node(*, form: str) -> None:
    """`next_wealth = 1` moves every subject to the float node `1.0`."""
    raw = _constant(form=form, value=1)

    def next_wealth(*, wealth: ContinuousState) -> ContinuousState:  # noqa: ARG001
        return raw  # ty: ignore[invalid-return-type]

    model = _two_period_model(final_utility=_wealth_utility, next_wealth=next_wealth)
    _assert_values(model=model, alive=np.array([1.0, 1.0]), final=np.array([1.0, 2.0]))
    df = _simulated(model)
    final_rows = df[df["period"] == 1]
    assert final_rows["wealth"].dtype.kind == "f"
    np.testing.assert_array_equal(final_rows["wealth"], [1.0, 1.0])


def _health_utility(*, health: DiscreteState) -> FloatND:
    return jnp.where(health == Health.good, 1.0, 0.0)


def _no_health_utility(*, health: DiscreteState) -> FloatND:
    return jnp.where(health == Health.good, 0.0, 0.0)


def _health_model(*, health_law: StateTransitionEntry) -> Model:
    alive = Regime(
        states={"health": DiscreteGrid(category_class=Health)},
        state_transitions={"health": health_law},
        functions={"utility": _no_health_utility},
    )
    final = Regime(
        states={"health": DiscreteGrid(category_class=Health)},
        functions={"utility": _health_utility},
    )
    return Model(
        regimes={"alive": alive, "final": final},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "alive"},
        edges={"alive": {"final": 0}},
    )


@pytest.mark.parametrize("form", ["python_int", "int_array"])
def test_constant_discrete_next_state_names_a_code(*, form: str) -> None:
    """`next_health = Health.good` (an int code) moves everyone to good health."""
    raw = _constant(form=form, value=int(Health.good))

    def next_health() -> DiscreteState:
        return raw  # ty: ignore[invalid-return-type]

    values = _health_model(health_law=next_health).solve(
        params=_PARAMS, log_level="off"
    )
    np.testing.assert_array_equal(values.values[0]["alive"], [1.0, 1.0])


@pytest.mark.parametrize("form", ["int_array", "float_array"])
def test_integer_stochastic_weights_match_their_float_twin(*, form: str) -> None:
    """Integer probabilities `[0, 1]` over health codes act as `[0.0, 1.0]`."""
    raw = (
        jnp.asarray([0, 1], dtype=jnp.int32)
        if form == "int_array"
        else jnp.asarray([0.0, 1.0])
    )

    def next_health() -> FloatND:
        return raw

    model = _health_model(health_law=StochasticTransition(func=next_health))
    values = model.solve(params=_PARAMS, log_level="off").values
    assert jnp.issubdtype(values[0]["alive"].dtype, jnp.floating)
    np.testing.assert_array_equal(values[0]["alive"], [1.0, 1.0])


@categorical(ordered=False)
class BranchRegimeId:
    alive: ScalarInt
    final: ScalarInt
    other: ScalarInt


def _regime_model(*, law: TransitionLaw) -> Model:
    alive = Regime(
        states={"wealth": _GRID},
        state_transitions={"wealth": _identity_wealth},
        functions={"utility": _zero_array_utility},
    )
    final = Regime(states={"wealth": _GRID}, functions={"utility": _wealth_utility})
    other = Regime(states={"wealth": _GRID}, functions={"utility": _zero_array_utility})
    return Model(
        regimes={"alive": alive, "final": final, "other": other},
        regime_id_class=BranchRegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "alive"},
        edges={"alive": Transition(targets={"final": 0, "other": 0}, law=law)},
    )


_BRANCH_PARAMS = {
    "alive": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "final": {},
    "other": {},
}


@pytest.mark.parametrize("form", _FORMS)
def test_constant_per_target_regime_probability(*, form: str) -> None:
    """Per-target probabilities `1` and `0` put all mass on the final regime."""
    one = _constant(form=form, value=1)
    zero = _constant(form=form, value=0)

    def to_final() -> FloatND:
        return one  # ty: ignore[invalid-return-type]

    def to_other() -> FloatND:
        return zero  # ty: ignore[invalid-return-type]

    model = _regime_model(
        law={
            "final": StochasticTransition(func=to_final),
            "other": StochasticTransition(func=to_other),
        }
    )
    values = model.solve(params=_BRANCH_PARAMS, log_level="off").values
    assert jnp.issubdtype(values[0]["alive"].dtype, jnp.floating)
    np.testing.assert_array_equal(values[0]["alive"], [1.0, 2.0])
    df = (
        model.simulate(
            params=_BRANCH_PARAMS,
            initial_conditions={
                **_INITIAL,
                "regime_id": jnp.array([BranchRegimeId.alive] * 2),
            },
            log_level="off",
        )
        .to_dataframe()
        .sort_values(["subject_id", "period"])
    )
    np.testing.assert_array_equal(df["regime_name"], ["alive", "final"] * 2)


@pytest.mark.parametrize("form", ["int_array", "float_array"])
def test_integer_regime_probability_vector(*, form: str) -> None:
    """A coarse probability vector `[0, 1, 0]` puts all mass on the final regime."""
    raw = (
        jnp.asarray([0, 1, 0], dtype=jnp.int32)
        if form == "int_array"
        else jnp.asarray([0.0, 1.0, 0.0])
    )

    def next_regime() -> FloatND:
        return raw

    model = _regime_model(law=StochasticTransition(func=next_regime))
    values = model.solve(params=_BRANCH_PARAMS, log_level="off").values
    np.testing.assert_array_equal(values[0]["alive"], [1.0, 2.0])


@pytest.mark.parametrize("form", ["python_int", "int_array"])
def test_constant_deterministic_regime_code(*, form: str) -> None:
    """A deterministic regime selector may return a constant regime code."""
    raw = _constant(form=form, value=int(BranchRegimeId.final))

    def next_regime() -> ScalarInt:
        return raw  # ty: ignore[invalid-return-type]

    values = _regime_model(law=next_regime).solve(
        params=_BRANCH_PARAMS, log_level="off"
    )
    np.testing.assert_array_equal(values.values[0]["alive"], [1.0, 2.0])


@categorical(ordered=False)
class ExitRegimeId:
    source: ScalarInt
    target: ScalarInt


def _source_utility() -> FloatND:
    return jnp.asarray(1.0)


def _stay(*, wealth: FloatND) -> FloatND:
    return wealth


def _exit_model(
    *,
    adjustment: UserFunction,
    projection: UserFunction = _stay,
    available: UserFunction | None = None,
) -> Model:
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": _source_utility},
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target",
                                projection={"wealth": projection},
                                utility_adjustment=adjustment,
                            )
                        },
                        rule="unilateral",
                        available=available,
                    )
                },
            ),
            "target": Regime(functions={"utility": _stay}),
        },
        states={"wealth": LinSpacedGrid(start=0.0, stop=2.0, n_points=3)},
        n_periods=1,
        regime_id_class=ExitRegimeId,
        initial_nodes=InitialNodes(by_period={0: ("source", "target")}),
        edges={},
    )


@pytest.mark.parametrize("form", _FORMS)
def test_constant_exit_utility_adjustment(*, form: str) -> None:
    """A zero adjustment leaves the outside option at the target's value."""
    raw = _constant(form=form, value=0)

    def adjustment() -> FloatND:
        return raw  # ty: ignore[invalid-return-type]

    model = _exit_model(adjustment=adjustment)
    values = model.solve(params={}, log_level="off")
    np.testing.assert_array_equal(values.values[0]["source"], [1.0, 1.0, 2.0])
    df = model.simulate(
        params={},
        initial_conditions={
            "period": jnp.array([0, 0]),
            "wealth": jnp.array([0.0, 2.0]),
            "regime_id": jnp.array([ExitRegimeId.source] * 2),
        },
        log_level="off",
    ).to_dataframe()
    np.testing.assert_array_equal(df.sort_values("subject_id")["value"], [1.0, 2.0])


@pytest.mark.parametrize("form", ["python_int", "python_float", "int_array"])
@pytest.mark.parametrize("handwritten", [False, True])
def test_constant_borrowing_floor(*, form: str, handwritten: bool) -> None:
    """A zero floor written as `0` binds like the floating floor `0.0`."""
    raw = _constant(form=form, value=0)

    def zero_floor() -> FloatND:
        return raw  # ty: ignore[invalid-return-type]

    def float_floor() -> FloatND:
        return jnp.asarray(0.0)

    def solved(lower: UserFunction) -> FloatND:
        model = _floor_model(
            declaration=ref("savings") >= ref("floor")
            if handwritten
            else post_decision_lower_bound(margin=LIQUID, lower=lower),
            helper=None,
            functions={"floor": lower} if handwritten else None,
        )
        return model.solve(params={}, log_level="off").values[0]["done"]

    np.testing.assert_array_equal(solved(zero_floor), solved(float_floor))


@pytest.mark.parametrize(
    "form", ["python_int", "python_float", "int_array", "float_array"]
)
def test_constant_exit_projection_lands_on_its_float_node(*, form: str) -> None:
    """Projecting every leaver to wealth `2` values the exit at the target's `2.0`."""
    raw = _constant(form=form, value=2)

    def projection() -> FloatND:
        return raw  # ty: ignore[invalid-return-type]

    def no_adjustment() -> FloatND:
        return jnp.asarray(0.0)

    model = _exit_model(adjustment=no_adjustment, projection=projection)
    values = model.solve(params={}, log_level="off").values
    np.testing.assert_array_equal(values[0]["source"], [2.0, 2.0, 2.0])


def _egm_solved(*, raw: UserFunctionResult, monkeypatch: pytest.MonkeyPatch) -> FloatND:
    def terminal(*, wealth: ContinuousState) -> FloatND:  # noqa: ARG001
        return raw  # ty: ignore[invalid-return-type]

    monkeypatch.setattr(egm_fixture, "terminal_utility", terminal)
    model = egm_fixture._model()
    values = model.solve(params={"discount_factor": 0.95}, log_level="off").values
    return values[0]["saving"]


def _nbegm_solved(
    *,
    raw: UserFunctionResult,
    monkeypatch: pytest.MonkeyPatch,  # noqa: ARG001
) -> FloatND:
    def terminal(*, liquid: ContinuousState) -> FloatND:  # noqa: ARG001
        return raw  # ty: ignore[invalid-return-type]

    params = tax_toy.build_params()
    params.pop("dead")
    model = _nbegm_model(terminal=terminal)
    return model.solve(params=params, log_level="off").values[0]["alive"]


def _dcegm_solved(
    *, raw: UserFunctionResult, monkeypatch: pytest.MonkeyPatch
) -> FloatND:
    def terminal(*, wealth: ContinuousState) -> FloatND:  # noqa: ARG001
        return raw  # ty: ignore[invalid-return-type]

    done = Regime(
        states={"wealth": dcegm_twin.WEALTH_GRID}, functions={"utility": terminal}
    )
    monkeypatch.setattr(dcegm_twin, "done_retired", done)
    model = dcegm_twin.build_dcegm_model()
    values = model.solve(params=dcegm_twin.get_params(), log_level="off").values
    return values[0]["working_life"]


@pytest.mark.parametrize("form", ["python_int", "python_float", "int_array"])
@pytest.mark.parametrize(
    "solved", [_egm_solved, _nbegm_solved, _dcegm_solved], ids=["egm", "nbegm", "dcegm"]
)
def test_constant_terminal_utility_under_endogenous_grids(
    *, form: str, solved, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal felicity `0` solves like the floating `0.0` in every EGM family."""
    got = solved(raw=_constant(form=form, value=0), monkeypatch=monkeypatch)
    expected = solved(
        raw=_constant(form="float_array", value=0), monkeypatch=monkeypatch
    )
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def _no_adjustment() -> FloatND:
    return jnp.asarray(0.0)


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize(
    "form", ["python_int", "python_bool", "int_array", "bool_array"]
)
@pytest.mark.parametrize("reads_state", [False, True])
def test_constant_exit_availability(
    *, value: int, form: str, reads_state: bool
) -> None:
    """Availability `1` or `0` opens or closes the exit like `True` or `False`."""
    raw = (
        jnp.asarray(bool(value))
        if form == "bool_array"
        else _constant(form=form, value=value)
    )

    def constant() -> BoolND:
        return raw  # ty: ignore[invalid-return-type]

    def by_state(*, wealth: FloatND) -> BoolND:  # noqa: ARG001
        return raw  # ty: ignore[invalid-return-type]

    model = _exit_model(
        adjustment=_no_adjustment, available=by_state if reads_state else constant
    )
    values = model.solve(params={}, log_level="off").values[0]["source"]
    np.testing.assert_array_equal(values, [1.0, 1.0, 2.0] if value else [1.0] * 3)


def test_floating_exit_availability_is_refused() -> None:
    """A floating availability is not a 0/1 spelling and stays refused."""

    def availability(*, wealth: FloatND) -> FloatND:
        return wealth - 0.5

    model = _exit_model(adjustment=_no_adjustment, available=availability)
    with pytest.raises(ValueError, match="scalar Boolean or integer"):
        model.solve(params={}, log_level="off")


@categorical(ordered=False)
class GateRegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


_GATE_WEALTH = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)


def _certain(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _ten_plus_wealth(*, wealth: ContinuousState) -> FloatND:
    return wealth + 10.0


def _gate_model(
    *, predicate: UserFunction, fallback: UserFunction = _stay, gated: bool = False
) -> Model:
    route = StakeholderRoute(
        fallback=ProjectedRegimeValue(
            regime="fallback", projection={"wealth": fallback}
        )
    )
    gates = {"target": Gate(predicate=predicate, routes={"self": route})}
    checks = {"target": ParticipationCheck(predicate=predicate, routes={"self": route})}
    return Model(
        regimes={
            "source": Regime(
                states={"wealth": _GATE_WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _zero_array_utility},
            ),
            "target": Regime(
                states={"wealth": _GATE_WEALTH}, functions={"utility": _ten_plus_wealth}
            ),
            "fallback": Regime(
                states={"wealth": _GATE_WEALTH}, functions={"utility": _stay}
            ),
        },
        ages=AgeGrid(start=40, inclusive_stop=45, step="5Y"),
        regime_id_class=GateRegimeId,
        initial_nodes={40: "source"},
        edges={
            "source": Transition(
                targets={"target": 40, "fallback": 40},
                law=ByAge(cases={40: {"target": StochasticTransition(func=_certain)}}),
                gates=gates if gated else {},
                participation={} if gated else checks,
            )
        },
    )


def _gated_values(model: Model) -> np.ndarray:
    values = model.solve(params={"discount_factor": 1.0}, log_level="off").values
    return np.asarray(values[0]["source"])


def _gated_landings(model: Model) -> pd.DataFrame:
    """Each subject's regime and wealth in the period after the gated edge."""
    frame = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "age": jnp.array([40.0, 40.0, 40.0]),
            "wealth": jnp.array([0.0, 1.0, 2.0]),
            "regime_id": jnp.array([GateRegimeId.source] * 3),
        },
        log_level="off",
    ).to_dataframe()
    landed = frame[frame["period"] == 1].sort_values("subject_id")
    return landed[["regime_name", "wealth"]].reset_index(drop=True)


_GATE_KINDS = pytest.mark.parametrize("gated", [False, True], ids=["check", "gate"])


@_GATE_KINDS
@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("form", ["python_int", "python_bool", "int_array"])
def test_constant_landing_predicate(*, gated: bool, value: int, form: str) -> None:
    """A gate or participation predicate `1` or `0` admits or diverts like a Boolean."""
    raw = _constant(form=form, value=value)

    def predicate() -> BoolND:
        return raw  # ty: ignore[invalid-return-type]

    model = _gate_model(predicate=predicate, gated=gated)
    np.testing.assert_array_equal(
        _gated_values(model), [10.0, 11.0, 12.0] if value else [0.0, 1.0, 2.0]
    )
    landed = _gated_landings(model)
    assert list(landed["regime_name"]) == ["target" if value else "fallback"] * 3
    np.testing.assert_array_equal(landed["wealth"], [0.0, 1.0, 2.0])


@_GATE_KINDS
@pytest.mark.parametrize(
    "form", ["python_int", "python_float", "int_array", "float_array"]
)
def test_constant_gated_fallback_projection(*, gated: bool, form: str) -> None:
    """A fallback projecting every subject to wealth `2` lands there at 2.0."""
    raw = _constant(form=form, value=2)

    def projection() -> FloatND:
        return raw  # ty: ignore[invalid-return-type]

    def never() -> BoolND:
        return jnp.asarray(a=False)

    model = _gate_model(predicate=never, fallback=projection, gated=gated)
    np.testing.assert_array_equal(_gated_values(model), [2.0] * 3)
    landed = _gated_landings(model)
    assert list(landed["regime_name"]) == ["fallback"] * 3
    assert landed["wealth"].dtype.kind == "f"
    np.testing.assert_array_equal(landed["wealth"], [2.0] * 3)


def _nbegm_model(
    *,
    terminal: UserFunction | None = None,
    constraints: Mapping[str, UserFunction] | None = None,
) -> Model:
    return make_alive_dead_model(
        n_periods=2,
        n_liquid=8,
        liquid_max=30.0,
        n_consumption=8,
        alive_functions={
            "utility": tax_toy.utility,
            "tax": tax_toy.tax,
            "resources": tax_toy.resources,
            "savings": tax_toy.savings,
        },
        liquid_law=tax_toy.next_liquid_from_savings,
        alive_solver=resolve_solver(
            variant="nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=28.0, n_points=8),
        ),
        constraints={} if constraints is None else constraints,
        dead_functions=None if terminal is None else {"utility": terminal},
    )


@pytest.mark.parametrize("form", ["python_bool", "python_int", "int_array"])
def test_constant_constraint_under_endogenous_grids(*, form: str) -> None:
    """An NBEGM predicate returning `1` admits every action like `True`."""
    raw = _constant(form=form, value=1)

    def allowed(*, liquid: ContinuousState) -> BoolND:  # noqa: ARG001
        return raw  # ty: ignore[invalid-return-type]

    def admitted(*, liquid: ContinuousState) -> BoolND:  # noqa: ARG001
        return jnp.asarray(a=True)

    def solved(constraint: UserFunction) -> np.ndarray:
        model = _nbegm_model(constraints={"allowed": constraint})
        values = model.solve(params=tax_toy.build_params(), log_level="off").values
        return np.asarray(values[0]["alive"])

    np.testing.assert_array_equal(solved(allowed), solved(admitted))


@categorical(ordered=False)
class JointRegimeId:
    source: ScalarInt
    target: ScalarInt


def _matched_income(match: Mapping[str, FloatND]) -> FloatND:
    return match["income"]


def _joint_model(
    *, wealth: UserFunction, probabilities: Callable[..., FloatND]
) -> Model:
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": _source_utility},
                joint_transitions={
                    "target": {
                        "match": JointTransition(
                            support_size=2,
                            support={"income": jnp.asarray([0.0, 1.0])},
                            probabilities=probabilities,
                            outputs={"wealth": wealth, "income": _matched_income},
                        )
                    }
                },
            ),
            "target": Regime(
                states={"wealth": grid, "income": grid},
                functions={"utility": _joint_utility},
            ),
        },
        ages=AgeGrid(start=20, inclusive_stop=21, step="Y"),
        regime_id_class=JointRegimeId,
        initial_nodes={20: "source"},
        edges={"source": {"target": 20}},
    )


def _joint_utility(*, wealth: ContinuousState, income: ContinuousState) -> FloatND:
    return wealth + 2 * income


_JOINT_PARAMS = {
    "source": {
        "target": {
            "match": {"support": {}, "probabilities": {}},
            "next_wealth": {},
            "next_income": {},
        },
        "koopmans_aggregator": {"discount_factor": 1.0},
    },
    "target": {"utility": {}},
}


def _uniform_match() -> FloatND:
    return jnp.asarray([0.5, 0.5])


def _joint_landings(model: Model) -> pd.DataFrame:
    frame = model.simulate(
        params=_JOINT_PARAMS,
        initial_conditions={
            "age": jnp.full(4, 20.0),
            "regime_id": jnp.full(4, JointRegimeId.source),
        },
        log_level="off",
    ).to_dataframe(terminal_rows="all")
    return frame[frame["regime_name"] == "target"]


@pytest.mark.parametrize("form", _FORMS)
def test_constant_joint_transition_output(*, form: str) -> None:
    """A joint output `1` lands every subject on wealth 1.0."""
    raw = _constant(form=form, value=1)

    def wealth() -> FloatND:
        return raw  # ty: ignore[invalid-return-type]

    model = _joint_model(wealth=wealth, probabilities=_uniform_match)
    values = model.solve(params=_JOINT_PARAMS, log_level="off").values
    np.testing.assert_array_equal(np.asarray(values[0]["source"]), 3.0)
    landed = _joint_landings(model)
    assert landed["wealth"].dtype.kind == "f"
    np.testing.assert_array_equal(landed["wealth"], [1.0] * 4)


def _stay_wealth() -> FloatND:
    return jnp.asarray(0.0)


def test_integer_joint_transition_probabilities() -> None:
    """An integer one-hot probability vector selects its node like a float one."""

    def probabilities() -> FloatND:
        return jnp.asarray([0, 1], dtype=jnp.int32)

    model = _joint_model(wealth=_stay_wealth, probabilities=probabilities)
    values = model.solve(params=_JOINT_PARAMS, log_level="off").values
    np.testing.assert_array_equal(np.asarray(values[0]["source"]), 3.0)
    np.testing.assert_array_equal(_joint_landings(model)["income"], [1.0] * 4)


@categorical(ordered=False)
class ShockRegimeId:
    alive: ScalarInt


@categorical(ordered=True)
class Outcome:
    low: ScalarInt
    high: ScalarInt


def _shock_value(*, eps: FloatND) -> FloatND:
    return eps


def _shock_model(*, law: Normal | Discrete) -> Model:
    return Model(
        regimes={
            "alive": Regime(functions={"utility": _shock_value}, shocks=("innovation",))
        },
        n_periods=1,
        regime_id_class=ShockRegimeId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": law},
                nodes=Product(eps=GaussHermite(n=1))
                if isinstance(law, Normal)
                else Product(),
            )
        },
    )


@pytest.mark.parametrize(
    "form", ["python_int", "python_float", "int_array", "float_array"]
)
def test_constant_current_iid_law_parameters(*, form: str) -> None:
    """A degenerate law centred by `2` and a one-hot integer law both read 2.0."""
    raw = _constant(form=form, value=2)

    def location() -> FloatND:
        return raw  # ty: ignore[invalid-return-type]

    def one_hot() -> FloatND:
        if form == "float_array":
            return jnp.asarray([0.0, 1.0])
        return jnp.asarray([0, 1], dtype=jnp.int32)

    normal = _shock_model(law=Normal(mu=location, sigma=0))
    values = normal.solve(params={}, log_level="off").values[0]["alive"]
    np.testing.assert_array_equal(np.asarray(values), 2.0)
    discrete = _shock_model(law=Discrete(categories=Outcome, probs=one_hot))
    values = discrete.solve(params={}, log_level="off").values[0]["alive"]
    np.testing.assert_array_equal(np.asarray(values), 1.0)


@pytest.mark.parametrize("form", ["python_bool", "python_int", "int_array"])
def test_constant_value_dependent_constraint(*, form: str) -> None:
    """A value-dependent predicate `1` admits every cell like `True`."""
    raw = _constant(form=form, value=1)

    def participation(*, Q_f: FloatND) -> BoolND:  # noqa: ARG001
        return raw  # ty: ignore[invalid-return-type]

    base, params = make_stateless_collective_target_model()
    couple = base.user_regimes["couple"]
    model = Model(
        regimes={
            "couple": couple.replace(
                constraints={
                    **dict(couple.constraints),
                    "participation": ValueDependentConstraint(predicate=participation),
                }
            ),
            "couple_terminal": base.user_regimes["couple_terminal"],
        },
        ages=AGES,
        regime_id_class=CoupleRegimeId,
        initial_nodes={0: "couple"},
        edges={"couple": {"couple_terminal": 0}},
    )
    values = model.solve(params=params, log_level="off").values
    np.testing.assert_allclose(
        values[0]["couple"], STATELESS_TARGET_V_PERIOD_0, rtol=1e-6
    )
