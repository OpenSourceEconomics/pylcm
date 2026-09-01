"""Backward induction keeps a collective regime's flag `D` only where it is read.

A collective regime's kernel publishes a boolean dissolution flag `D` on its
state axes alongside `V`, one array per period. Two things can consume the
accumulated per-period mapping: a gated edge whose gate declares the `D_target`
operand (forward simulation recomputes such a gate from the flag), and a caller
that asks for the flags with `Model.solve(return_dissolution_flags=True)`.

When neither holds, the arrays are retained for the whole backward induction and
nobody reads them. Whether a gate declares `D_target` is decided by the gate's
own signature, so backward induction settles it before the first kernel runs and
drops the accumulation for the models where it is dead. The flags a reader does
ask for, and every value function, are unaffected.

The two models here differ in exactly one respect — whether the source's gate
declares `D_target` — so a difference between them is attributable to that.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.utils.logging import get_logger
from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentConstraint,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)

_BETA = 0.95

# Three wage nodes, so the individual-rationality mask can empty at the middle
# one and give the dissolution model a `True` cell to route on.
_WAGE = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class ConsentRegimeId:
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt
    married_terminal: ScalarInt


@categorical(ordered=False)
class DissolutionRegimeId:
    married: ScalarInt
    married_ir: ScalarInt
    married_terminal: ScalarInt
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m: ScalarInt
    single_m_terminal: ScalarInt


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _u_zero(wage: ContinuousState) -> FloatND:
    return 0.0 * wage


def _u_zero_collective(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.0 * wage * work


def _u_single_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage * work


def _u_single_f_terminal(wage: ContinuousState) -> FloatND:
    return 1.5 * wage


def _u_single_m_terminal(wage: ContinuousState) -> FloatND:
    return jnp.where(wage < 1.5, 0.5, 3.0)


def _u_married_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 2.0 * wage + 0.0 * work


def _u_married_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage + 0.0 * work


def _u_married_ir_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 3.0 * (1.0 - work) + 2.0 * wage * work


def _u_married_ir_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.5 * (1.0 - work) + wage * work


def _u_single_f_ir(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    target = jnp.where((wage > 1.5) & (wage < 2.5), 5.5, 1.5)
    return target * work


def _u_single_m_ir(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 1.0 * work + 0.0 * wage


def _ir_f(*, Q_f: FloatND, V_single_f_ref: FloatND, delta_f: FloatND) -> BoolND:
    return Q_f >= V_single_f_ref - delta_f


def _ir_m(*, Q_m: FloatND, V_single_m_ref: FloatND, delta_m: FloatND) -> BoolND:
    return Q_m >= V_single_m_ref - delta_m


def _consent_gate(
    *,
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
) -> BoolND:
    """Mutual consent: reads only value operands, never the target's flag."""
    return (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)


def _no_dissolution_gate(D_target: BoolND) -> BoolND:
    """Stay married exactly where the target household has not dissolved."""
    return ~D_target


def _make_consent_model() -> tuple[Model, dict]:
    """A collective TARGET reached by a gate that reads only value operands.

    `single_f` consents into the collective `married_terminal`, whose kernel
    publishes `D`. No gate in the model declares `D_target`.
    """
    single_f = Regime(
        transition={
            "married_terminal": ValueDependentTransition(
                probability=MarkovTransition(_prob_one),
                gate=_consent_gate,
                routes={
                    "f": StakeholderRoute(
                        target_stakeholder="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f_terminal",
                            projection={"wage": _identity_wage},
                        ),
                    )
                },
                gate_references={
                    "V_single_f_ref": ProjectedRegimeValue(
                        regime="single_f_terminal",
                        projection={"wage": _identity_wage},
                    ),
                    "V_single_m_ref": ProjectedRegimeValue(
                        regime="single_m_terminal",
                        projection={"wage": _identity_wage},
                    ),
                },
            )
        },
        active=lambda age: age < 1,
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _u_single_f_terminal},
    )
    single_m_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _u_single_m_terminal},
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_married_f, "m": _u_married_m}
            )
        },
    )
    model = Model(
        regimes={
            "single_f": single_f,
            "single_f_terminal": single_f_terminal,
            "single_m_terminal": single_m_terminal,
            "married_terminal": married_terminal,
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=ConsentRegimeId,
    )
    return model, {"discount_factor": _BETA}


def _make_dissolution_model() -> tuple[Model, dict]:
    """A collective SOURCE whose gate reads the target's dissolution flag."""
    married = Regime(
        transition={
            "married_ir": ValueDependentTransition(
                probability=MarkovTransition(_prob_one),
                gate=_no_dissolution_gate,
                routes={
                    "f": StakeholderRoute(
                        target_stakeholder="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f", projection={"wage": _identity_wage}
                        ),
                    ),
                    "m": StakeholderRoute(
                        target_stakeholder="m",
                        fallback=ProjectedRegimeValue(
                            regime="single_m", projection={"wage": _identity_wage}
                        ),
                    ),
                },
            )
        },
        active=lambda age: age < 1,
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    married_ir = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_married_ir_f, "m": _u_married_ir_m}
            )
        },
        constraints={
            "ir_f": ValueDependentConstraint(
                predicate=_ir_f,
                references={
                    "V_single_f_ref": ProjectedRegimeValue(
                        regime="single_f", projection={"wage": _identity_wage}
                    )
                },
            ),
            "ir_m": ValueDependentConstraint(
                predicate=_ir_m,
                references={
                    "V_single_m_ref": ProjectedRegimeValue(
                        regime="single_m", projection={"wage": _identity_wage}
                    )
                },
            ),
        },
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wage": _WAGE},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    single_f = Regime(
        transition={"single_f_terminal": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f_ir},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wage": _WAGE},
        functions={"utility": _u_zero},
    )
    single_m = single_f.replace(
        transition={"single_m_terminal": MarkovTransition(_prob_one)},
        functions={"utility": _u_single_m_ir},
    )
    model = Model(
        regimes={
            "married": married,
            "married_ir": married_ir,
            "married_terminal": married_terminal,
            "single_f": single_f,
            "single_f_terminal": single_f_terminal,
            "single_m": single_m,
            "single_m_terminal": single_f_terminal.replace(),
        },
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=DissolutionRegimeId,
    )
    params = {"discount_factor": _BETA, "delta_f": 0.5, "delta_m": 0.2}
    return model, params


def _solve_internal(
    *,
    model: Model,
    params: dict,
    collect_simulation_policies: bool = False,
    retain_dissolution_flags: bool = False,
):
    """Run the engine solve behind `Model.solve` and return its full result.

    Policy collection is off by default: these tests ask which dissolution flags
    survive backward induction, and `collect_simulation_policies` retains arrays
    for an unrelated reason, so leaving it on would blur what is being measured.
    """
    return model._solve_compiled(
        flat_params=model._process_params(params),
        params=params,
        log=get_logger(log_level="off"),
        log_path=None,
        log_keep_n_latest=3,
        max_compilation_workers=None,
        collect_simulation_policies=collect_simulation_policies,
        retain_dissolution_flags=retain_dissolution_flags,
    )


def _retained_flag_arrays(dissolution_flags) -> list:
    """Every flag array the backward induction kept, across all periods."""
    return [
        array
        for regime_to_flag in dissolution_flags.values()
        for array in regime_to_flag.values()
    ]


def test_solve_retains_no_flags_when_no_gate_reads_the_dissolution_flag():
    """A collective target whose gate reads only values leaves no flag retained.

    `married_terminal` is collective, so its kernel publishes `D` every period;
    the only gate in the model is the value-operand consent gate, so nothing
    can read those arrays and backward induction keeps none of them.
    """
    model, params = _make_consent_model()

    result = _solve_internal(model=model, params=params)

    assert _retained_flag_arrays(result.dissolution_flags) == []


def test_solve_retains_flags_when_a_gate_reads_the_dissolution_flag():
    """A gate declaring `D_target` keeps the flags without anyone asking.

    Forward simulation recomputes such a gate from the flag, so the arrays must
    survive the backward induction whether or not the caller requested them.
    """
    model, params = _make_dissolution_model()

    result = _solve_internal(model=model, params=params)

    assert _retained_flag_arrays(result.dissolution_flags) != []


def test_opting_in_retains_flags_even_without_a_flag_reading_gate():
    """`return_dissolution_flags=True` surfaces `D` for any collective model.

    The request stands on its own: a caller inspecting a collective regime's
    empty-mask cells gets them even though no gate in the model reads them.
    """
    model, params = _make_consent_model()

    _value_functions, dissolution_flags = model.solve(
        params=params, log_level="off", return_dissolution_flags=True
    )

    assert any(len(regime_map) > 0 for regime_map in dissolution_flags.values())


@pytest.mark.parametrize("make_model", [_make_consent_model, _make_dissolution_model])
def test_value_functions_do_not_depend_on_whether_flags_are_retained(make_model):
    """Retaining `D` changes what is kept, never what is computed.

    Every value function is bit-identical between a solve that keeps the flags
    and one that drops them, for a flag-reading gate and a value-operand one
    alike.
    """
    model, params = make_model()

    dropped = _solve_internal(model=model, params=params).value_functions
    retained = _solve_internal(
        model=model, params=params, retain_dissolution_flags=True
    ).value_functions

    assert set(dropped) == set(retained)
    for period, regime_to_V in retained.items():
        assert set(dropped[period]) == set(regime_to_V)
        for regime_name, V_arr in regime_to_V.items():
            np.testing.assert_array_equal(
                np.asarray(dropped[period][regime_name]),
                np.asarray(V_arr),
                err_msg=f"period {period}, regime {regime_name}",
            )


def test_simulate_is_unchanged_when_the_solve_drops_unread_flags():
    """A gated edge whose gate reads no flag simulates the same without them.

    The consent gate's operands are value components and gate refs, so the
    simulated path is identical whether the collective target's flags are
    threaded into `simulate` or absent entirely.
    """
    model, params = _make_consent_model()
    initial_conditions = {
        "wage": jnp.array([1.0, 2.0, 3.0]),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, ConsentRegimeId.single_f, dtype=jnp.int32),
    }
    value_functions, flags = model.solve(
        params=params, log_level="off", return_dissolution_flags=True
    )

    with_flags = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=value_functions,
        period_to_regime_to_dissolution_flags=flags,
        log_level="off",
        seed=0,
    ).to_dataframe()
    without_flags = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=value_functions,
        period_to_regime_to_dissolution_flags=MappingProxyType({}),
        log_level="off",
        seed=0,
    ).to_dataframe()

    assert with_flags.equals(without_flags)
