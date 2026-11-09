"""User functions may return Python scalars where the engine expects arrays.

A felicity, a constraint, a next state, a regime-transition probability, a
stochastic weight, an exit utility adjustment and a borrowing floor may each
return `0`, `0.0`, `True` or an array. Each solves and simulates to the same
values as its array-returning twin, at the active precision.
"""

from collections.abc import Mapping
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    Exit,
    ExitTarget,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    post_decision_lower_bound,
)
from lcm.collective import CollectiveUtility
from lcm.exceptions import InvalidValueFunctionError
from lcm.regime import StateTransitionEntry
from lcm.transition import TransitionLaw
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
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
        durable_identity=False,
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
        durable_identity=False,
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
        durable_identity=False,
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


def _exit_model(*, adjustment: UserFunction) -> Model:
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": _source_utility},
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target",
                                projection={"wealth": _stay},
                                utility_adjustment=adjustment,
                            )
                        },
                        rule="unilateral",
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

    values = _exit_model(adjustment=adjustment).solve(params={}, log_level="off")
    np.testing.assert_array_equal(values.values[0]["source"], [1.0, 1.0, 2.0])


@pytest.mark.parametrize("form", ["python_int", "python_float", "int_array"])
def test_constant_borrowing_floor(*, form: str) -> None:
    """A zero floor written as `0` binds like the floating floor `0.0`."""
    raw = _constant(form=form, value=0)

    def zero_floor() -> FloatND:
        return raw  # ty: ignore[invalid-return-type]

    def float_floor() -> FloatND:
        return jnp.asarray(0.0)

    def solved(lower: UserFunction) -> IntND:
        model = _floor_model(
            declaration=post_decision_lower_bound(margin=LIQUID, lower=lower),
            helper=None,
        )
        return model.solve(params={}, log_level="off").values[0]["done"]

    np.testing.assert_array_equal(solved(zero_floor), solved(float_floor))
