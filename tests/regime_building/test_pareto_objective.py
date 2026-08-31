"""A household's Pareto scalarization is a declaration, not an ordinary function.

`ParetoObjective(weights=...)` names one weight per stakeholder and the engine
owns what a Pareto weight means: one per stakeholder, finite and non-negative,
with a strictly positive total, and multiplied into each stakeholder's action
value zero-safely so an excluded stakeholder's admissible `-inf` cannot poison
the household's choice.

The models below are terminal-continuation-free and use one binary action, so
every household decision is an exact small-integer comparison and the assertions
are on the decision itself rather than on a tolerance around it.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    Model,
    ParetoObjective,
    Phased,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    ModelInitializationError,
    PyLCMError,
    RegimeInitializationError,
)
from lcm.typing import DiscreteAction, DiscreteState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_AGES = AgeGrid(start=0, stop=2, step="Y")


@categorical(ordered=False)
class Choice:
    a: ScalarInt
    b: ScalarInt


@categorical(ordered=False)
class Power:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class RegimeId:
    couple: ScalarInt
    couple_terminal: ScalarInt


def _next_regime() -> ScalarInt:
    return RegimeId.couple_terminal


def _utility_f(choice: DiscreteAction) -> FloatND:
    """Prefers `a`, worth 3 against 0."""
    return jnp.where(choice == Choice.a, 3.0, 0.0)


def _utility_m(choice: DiscreteAction) -> FloatND:
    """Prefers `b`, worth 4 against 0."""
    return jnp.where(choice == Choice.a, 0.0, 4.0)


def _terminal_zero() -> FloatND:
    return jnp.asarray(0.0)


def _weight_f(pareto_weight_f: float) -> FloatND:
    return jnp.asarray(pareto_weight_f)


def _weight_m(pareto_weight_f: float) -> FloatND:
    return 1.0 - jnp.asarray(pareto_weight_f)


def _weight_f_by_power(power: DiscreteState) -> FloatND:
    """Her weight is 0.8 where she holds the bargaining power, 0.2 elsewhere."""
    return jnp.where(power == Power.high, 0.8, 0.2)


def _weight_m_by_power(power: DiscreteState) -> FloatND:
    return jnp.where(power == Power.high, 0.2, 0.8)


def _weight_by_choice(choice: DiscreteAction) -> FloatND:
    """The defect: a weight that depends on the choice it is used to make."""
    return jnp.where(choice == Choice.a, 0.9, 0.1)


def _build_model(
    *,
    objective: ParetoObjective | None,
    states: dict | None = None,
    state_transitions: dict | None = None,
    utility_f=_utility_f,
    utility_m=_utility_m,
) -> Model:
    couple = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states=states or {},
        state_transitions=state_transitions or {},
        actions={"choice": DiscreteGrid(Choice)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": utility_f, "m": utility_m}, objective=objective
            )
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _terminal_zero, "m": _terminal_zero}
            )
        },
    )
    return Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=_AGES,
        regime_id_class=RegimeId,
    )


def _params(**pareto: float) -> dict:
    return {
        "couple": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            **({"pareto_objective": pareto} if pareto else {}),
        },
        "couple_terminal": {},
    }


def test_equal_constant_weights_pick_the_larger_total_payoff() -> None:
    """Equal weights choose `b`, worth `(0, 4)` against `a`'s `(3, 0)`.

    The household compares `0.5 * 3 = 1.5` against `0.5 * 4 = 2`. This is the
    control: it is the answer an undeclared objective already gives, so the
    skewed case below is a change of weights and not of anything else.
    """
    model = _build_model(
        objective=ParetoObjective(weights={"f": 0.5, "m": 0.5}),
    )

    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["couple"]),
        np.array([0.0, 4.0]),
        decimal=DECIMAL_PRECISION,
    )


def test_a_skewed_constant_weight_moves_the_household_choice() -> None:
    """Weighting her 0.8 picks `a`, worth `(3, 0)`.

    `0.8 * 3 = 2.4` beats `0.2 * 4 = 0.8`, so the pair the household reads off
    is hers rather than his. The two candidates differ by a whole unit of
    payoff, so this is the discrete decision and not a rounding difference.
    """
    model = _build_model(
        objective=ParetoObjective(weights={"f": 0.8, "m": 0.2}),
    )

    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["couple"]),
        np.array([3.0, 0.0]),
        decimal=DECIMAL_PRECISION,
    )


def test_a_weight_parameter_is_estimable_without_rebuilding_the_model() -> None:
    """One model answers at `0.8` and at `0.5`, set per call.

    A weight declared as an ordinary function of a free parameter reaches
    `get_params_template()` under the `pareto_objective` key, so a estimation
    loop varies it the way it varies any other parameter.
    """
    model = _build_model(
        objective=ParetoObjective(weights={"f": _weight_f, "m": _weight_m}),
    )

    assert set(model.get_params_template()["couple"]["pareto_objective"]) == {
        "pareto_weight_f"
    }

    hers = model.solve(params=_params(pareto_weight_f=0.8), log_level="debug")
    his = model.solve(params=_params(pareto_weight_f=0.5), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(hers[0]["couple"]), np.array([3.0, 0.0]), decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        np.asarray(his[0]["couple"]), np.array([0.0, 4.0]), decimal=DECIMAL_PRECISION
    )


def test_a_state_dependent_weight_decides_cell_by_cell() -> None:
    """Her cell picks `a`; his picks `b`, in one solve.

    The weights read the `power` state, so the household's choice differs
    across the state grid: `(3, 0)` where she holds the power and `(0, 4)`
    where he does. The `Power` codes are `low = 0`, `high = 1`, so the value
    array's first row is his cell.
    """
    model = _build_model(
        objective=ParetoObjective(
            weights={"f": _weight_f_by_power, "m": _weight_m_by_power}
        ),
        states={"power": DiscreteGrid(Power)},
        state_transitions={"power": fixed_transition("power")},
    )

    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["couple"]),
        np.array([[0.0, 4.0], [3.0, 0.0]]),
        decimal=DECIMAL_PRECISION,
    )


def _utility_f_minus_inf(choice: DiscreteAction) -> FloatND:
    """Her value of `a` is an admissible `-inf`, not an infeasibility."""
    return jnp.where(choice == Choice.a, -jnp.inf, 0.0)


def _utility_m_prefers_a(choice: DiscreteAction) -> FloatND:
    return jnp.where(choice == Choice.a, 5.0, 1.0)


def test_a_zero_weight_annihilates_an_admissible_minus_infinity() -> None:
    """With her weight at zero the household picks `a` on his payoff alone.

    Her `-inf` at `a` is a value, not an infeasibility, and she carries no
    weight in the decision. Multiplying naively makes the objective `0 * -inf`,
    i.e. NaN, which loses to `1 * 1` and hands the household `b` — the wrong
    partner's preference deciding. The published pair keeps her own `-inf`.
    """
    model = _build_model(
        objective=ParetoObjective(weights={"f": 0.0, "m": 1.0}),
        utility_f=_utility_f_minus_inf,
        utility_m=_utility_m_prefers_a,
    )

    solution = model.solve(params=_params(), log_level="debug")

    assert np.asarray(solution[0]["couple"]).tolist() == [-np.inf, 5.0]


def test_a_missing_stakeholder_weight_is_rejected() -> None:
    """The error names the stakeholder the objective does not weight."""
    with pytest.raises(RegimeInitializationError) as excinfo:
        _build_model(objective=ParetoObjective(weights={"f": 1.0}))

    assert "'m'" in str(excinfo.value)


def test_a_weight_for_an_unknown_stakeholder_is_rejected() -> None:
    """The error names the stakeholder the regime does not have."""
    with pytest.raises(RegimeInitializationError) as excinfo:
        _build_model(
            objective=ParetoObjective(weights={"f": 0.5, "m": 0.5, "child": 0.1})
        )

    assert "'child'" in str(excinfo.value)


def test_a_negative_constant_weight_is_rejected() -> None:
    """The error names the stakeholder and the value."""
    with pytest.raises(RegimeInitializationError) as excinfo:
        _build_model(objective=ParetoObjective(weights={"f": -0.5, "m": 1.5}))

    message = str(excinfo.value)
    assert "'f'" in message
    assert "-0.5" in message


def test_an_all_zero_weight_vector_is_rejected() -> None:
    """A household that weights nobody has no objective to maximize."""
    with pytest.raises(RegimeInitializationError) as excinfo:
        _build_model(objective=ParetoObjective(weights={"f": 0.0, "m": 0.0}))

    assert "total" in str(excinfo.value).lower()


def test_an_action_dependent_weight_is_rejected() -> None:
    """A weight may read the state, never the choice it is used to make.

    A weight that varies with the action turns the scalarization into a
    different objective per candidate, and the maximizer of that is not a
    Pareto optimum of any fixed weighting.
    """
    with pytest.raises(ModelInitializationError) as excinfo:
        _build_model(
            objective=ParetoObjective(
                weights={"f": _weight_by_choice, "m": _weight_by_choice}
            )
        )

    message = str(excinfo.value)
    assert "'choice'" in message
    assert "'f'" in message


def test_a_negative_weight_parameter_is_rejected_when_the_model_is_solved() -> None:
    """A weight that only becomes negative at a parameter value stops the solve.

    The declaration is admissible — the defect is the value supplied for it —
    so it is caught where the value arrives, and the message names the
    parameter's own path rather than the stakeholder alone.
    """
    model = _build_model(
        objective=ParetoObjective(weights={"f": _weight_f, "m": _weight_m}),
    )

    with pytest.raises(PyLCMError) as excinfo:
        model.solve(params=_params(pareto_weight_f=1.5), log_level="debug")

    message = str(excinfo.value)
    assert "pareto_objective" in message
    assert "'m'" in message


# The scalarization's own closure cases, taken at the kernel it is evaluated in
# rather than through a whole model: the household objective is a weighted sum,
# and what has to hold of it is exact.
_EXTREME_WEIGHTS = [
    ({"f": 0.0, "m": 1.0}, [-np.inf, 5.0]),
    ({"f": 1.0, "m": 0.0}, [0.0, 1.0]),
]


@pytest.mark.parametrize(("weights", "expected"), _EXTREME_WEIGHTS)
def test_the_endpoints_of_the_weight_interval_read_one_partner(
    weights: dict[str, float], expected: list[float]
) -> None:
    """At `w = 0` and `w = 1` the household follows exactly one stakeholder.

    Her value of `a` is an admissible `-inf`. Weighting only him picks `a` for
    his 5, and her `-inf` is published as her own value there. Weighting only
    her picks `b`, where she is at 0 and he is at 1 — so each endpoint reads
    off the pair at the excluded partner's preferred action as readily as at
    the deciding one.
    """
    model = _build_model(
        objective=ParetoObjective(weights=weights),
        utility_f=_utility_f_minus_inf,
        utility_m=_utility_m_prefers_a,
    )

    solution = model.solve(params=_params(), log_level="debug")

    assert np.asarray(solution[0]["couple"]).tolist() == expected


_NEAR_ZERO = [1e-300, 5e-324, np.finfo(np.float64).tiny]


@pytest.mark.parametrize("weight", _NEAR_ZERO)
def test_a_weight_just_above_zero_still_weighs_its_stakeholder(
    weight: float,
) -> None:
    """A subnormal weight is a weight, not an exclusion.

    Her weight is far below his but not zero, so the household still compares
    `w * 3` against `(1 - w) * 4` and picks his `b`. The point is that the
    subnormal is not flushed to an exact zero on the way in, which would make
    the comparison one-sided and, for a stakeholder holding `-inf`, silently
    drop her from the objective.
    """
    model = _build_model(objective=ParetoObjective(weights={"f": weight, "m": 1.0}))

    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["couple"]),
        np.array([0.0, 4.0]),
        decimal=DECIMAL_PRECISION,
    )


@categorical(ordered=False)
class ThreeRegimeId:
    household: ScalarInt
    household_terminal: ScalarInt


def _next_three_regime() -> ScalarInt:
    return ThreeRegimeId.household_terminal


def _utility_child(choice: DiscreteAction) -> FloatND:
    """Prefers `a`, worth 1 against 0."""
    return jnp.where(choice == Choice.a, 1.0, 0.0)


def _three_params() -> dict:
    return {
        "household": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "household_terminal": {},
    }


def _three_stakeholder_solution(order: tuple[str, ...]) -> np.ndarray:
    """Solve the same household with its stakeholders declared in `order`."""
    utilities = {
        "f": _utility_f,
        "m": _utility_m,
        "child": _utility_child,
    }
    weights = {"f": 0.5, "m": 0.25, "child": 0.25}
    household = Regime(
        transition=_next_three_regime,
        active=lambda age: age < 1,
        actions={"choice": DiscreteGrid(Choice)},
        functions={
            "utility": CollectiveUtility(
                utilities={name: utilities[name] for name in order},
                objective=ParetoObjective(
                    weights={name: weights[name] for name in order}
                ),
            )
        },
    )
    household_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={
            "utility": CollectiveUtility(utilities=dict.fromkeys(order, _terminal_zero))
        },
    )
    model = Model(
        regimes={
            "household": household,
            "household_terminal": household_terminal,
        },
        ages=_AGES,
        regime_id_class=ThreeRegimeId,
    )
    solution = model.solve(params=_three_params(), log_level="debug")
    by_name = dict(zip(order, np.asarray(solution[0]["household"]), strict=True))
    return np.array([by_name["f"], by_name["m"], by_name["child"]])


_ORDERS = [
    ("f", "m", "child"),
    ("child", "f", "m"),
    ("m", "child", "f"),
]


@pytest.mark.parametrize("order", _ORDERS)
def test_declaration_order_does_not_decide_a_three_party_household(
    order: tuple[str, ...],
) -> None:
    """Relabelling three stakeholders leaves the household's choice alone.

    `0.5 * 3 = 1.5` for `a` against `0.25 * 4 = 1` for `b`, plus the child's
    `0.25 * 1 = 0.25` for `a`, so `a` wins under every declaration order and
    the published triple is `(3, 0, 1)` read back by name. From three terms on
    the sum has a reduction tree to choose, and choosing it by declaration
    order would let an economically inert relabelling move a close decision.
    """
    np.testing.assert_array_almost_equal(
        _three_stakeholder_solution(order),
        np.array([3.0, 0.0, 1.0]),
        decimal=DECIMAL_PRECISION,
    )


def _impute_power() -> ScalarInt:
    """Backward induction assumes she holds the bargaining power."""
    return Power.high


def _power_signal() -> ScalarInt:
    """An ordinary solve-function ancestor of the carried-state imputation."""
    return Power.high


def _impute_power_from_signal(power_signal: ScalarInt) -> ScalarInt:
    """Resolve the carried state through an ordinary regime-function output."""
    return power_signal


def _carry_power(power: DiscreteState) -> ScalarInt:
    """The seeded bargaining power persists through the panel."""
    return power


def _build_carried_power_model() -> Model:
    """A household whose Pareto weights read a carried bargaining-power state."""
    couple = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={
            "power": Phased(solve=_impute_power, simulate=DiscreteGrid(Power)),
        },
        state_transitions={"power": _carry_power},
        actions={"choice": DiscreteGrid(Choice)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _utility_f, "m": _utility_m},
                objective=ParetoObjective(
                    weights={"f": _weight_f_by_power, "m": _weight_m_by_power}
                ),
            )
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _terminal_zero, "m": _terminal_zero}
            )
        },
    )
    return Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=_AGES,
        regime_id_class=RegimeId,
    )


def test_a_weight_reading_a_carried_state_solves_on_its_imputation() -> None:
    """A weight may read a carried state; the solve uses that state's imputation.

    The imputation puts the bargaining power with her, so her weight is `0.8`
    and the household takes `a`, worth `3` to her and `0` to him. A carried
    state contributes no solve axis, so the value array carries the stakeholder
    axis alone.
    """
    model = _build_carried_power_model()

    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["couple"]),
        np.array([3.0, 0.0]),
        decimal=DECIMAL_PRECISION,
    )


def test_a_carried_weight_preserves_ordinary_imputation_dependencies() -> None:
    """The carried imputation reaches an ordinary helper as a DAG dependency.

    `weight_f(power)` reads a carried state, whose solve imputation is
    `impute_power(power_signal)`. The ordinary `power_signal` function produces
    `high`; it is not a parameter named `power__power_signal`. The resulting
    weights select her preferred action and publish stakeholder values `(3, 0)`.
    """
    couple = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={
            "power": Phased(
                solve=_impute_power_from_signal,
                simulate=DiscreteGrid(Power),
            ),
        },
        state_transitions={"power": _carry_power},
        actions={"choice": DiscreteGrid(Choice)},
        functions={
            "power_signal": _power_signal,
            "utility": CollectiveUtility(
                utilities={"f": _utility_f, "m": _utility_m},
                objective=ParetoObjective(
                    weights={"f": _weight_f_by_power, "m": _weight_m_by_power}
                ),
            ),
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _terminal_zero, "m": _terminal_zero}
            )
        },
    )
    model = Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=_AGES,
        regime_id_class=RegimeId,
    )

    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["couple"]),
        np.array([3.0, 0.0]),
        decimal=DECIMAL_PRECISION,
    )


def test_a_carried_weight_decides_on_the_imputation_not_the_seeded_value() -> None:
    """The household follows the imputed power even when the seed disagrees.

    Both subjects are seeded with `power = low`, under which his weight would
    be `0.8` and the household would take `b`. The decision is made on the
    solve-phase imputation, which puts the power with her, so both rows choose
    `a`.
    """
    model = _build_carried_power_model()

    result = model.simulate(
        params=_params(),
        initial_conditions={
            "age": jnp.full(2, model.ages.values[0]),
            "power": jnp.array([Power.low, Power.low]),
            "regime_id": jnp.array([RegimeId.couple, RegimeId.couple], dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )

    chosen = result.to_dataframe().query("regime_name == 'couple'")["choice"]
    assert list(chosen) == ["a", "a"]


def _zero_argument_weight_negative() -> FloatND:
    return jnp.asarray(-1.0)


def _zero_argument_weight_two() -> FloatND:
    return jnp.asarray(2.0)


def _zero_argument_weight_zero() -> FloatND:
    return jnp.asarray(0.0)


def test_a_negative_zero_argument_callable_weight_is_rejected() -> None:
    """A weight declared as a constant-valued callable is judged like a constant.

    `-1` is refused whether it is written as a float or returned by a callable
    that reads nothing.
    """
    model = _build_model(
        objective=ParetoObjective(
            weights={
                "f": _zero_argument_weight_negative,
                "m": _zero_argument_weight_two,
            }
        )
    )

    with pytest.raises(PyLCMError, match="negative"):
        model.solve(params=_params(), log_level="debug")


def test_all_zero_zero_argument_callable_weights_are_rejected() -> None:
    """Weights that are identically zero leave the household argmax undefined.

    Without this the objective is zero everywhere and the household silently
    takes whichever action the tie-break happens to reach first.
    """
    model = _build_model(
        objective=ParetoObjective(
            weights={
                "f": _zero_argument_weight_zero,
                "m": _zero_argument_weight_zero,
            }
        )
    )

    with pytest.raises(PyLCMError, match="positive total"):
        model.solve(params=_params(), log_level="debug")
