"""Integration tests for E3': gated edge objects (slice 4).

A source regime declares `gated_edges` keyed by TARGET regime name. At the end of
each period the engine folds, per declared edge and source stakeholder ``s``, a
gated continuation object on the target regime's grid (design doc
`pylcm-extension-collective-regimes.md` §2 E3'; EKL 2019 eqs. 9/12/27)::

    Wbar^s(x) = jnp.where(gate(x), V_target^{leg_s}(x), V_fallback^s(pi_s(x)))

The source's continuation reads ``Wbar`` in place of the raw target V. Two edges
matter for EKL: the singles->married CONSENT edge (a singleton source reaches a
collective target only by mutual consent, eq. 27) and the married self DIVORCE
edge (a collective source routes per-stakeholder to the single regimes when the
IR mask empties, ``D=True``). The mixture is the strict ``jnp.where`` — never a
linear ``kappa*V + (1-kappa)*V`` — so a ``-inf`` target cell never leaks NaN.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, LinSpacedGrid, Model, categorical, fixed_transition
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    work: ScalarInt


_WAGE = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)  # {1.0, 2.0}
_BETA = 0.95


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


# --------------------------------------------------------------------------------------
# Consent edge: single_f (singleton) -> married_terminal (collective)
# --------------------------------------------------------------------------------------


def _u_single_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage * work


def _u_single_f_terminal(wage: ContinuousState) -> FloatND:
    return 1.5 * wage  # {1.5, 3.0}


def _u_single_m_terminal(wage: ContinuousState) -> FloatND:
    return jnp.where(wage < 1.5, 0.5, 3.0)  # {0.5, 3.0}


def _u_married_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 2.0 * wage + 0.0 * work  # {2, 4}


def _u_married_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage + 0.0 * work  # {1, 2}


def _consent_gate(
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
) -> BoolND:
    # EKL eq. 27: strict, unanimous mutual consent.
    return (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)


def _make_consent_regimes() -> dict[str, Regime]:
    single_f = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_single_f},
        gated_edges={
            "married_terminal": GatedEdge(
                gate=_consent_gate,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="single_f_terminal",
                            projection={"wage": _identity_wage},
                        ),
                    )
                },
                gate_refs={
                    "V_single_f_ref": SamePeriodRef(
                        regime="single_f_terminal",
                        projection={"wage": _identity_wage},
                    ),
                    "V_single_m_ref": SamePeriodRef(
                        regime="single_m_terminal",
                        projection={"wage": _identity_wage},
                    ),
                },
            )
        },
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
        stakeholders=("f", "m"),
        states={"wage": _WAGE},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_married_f, "utility_m": _u_married_m},
    )
    return {
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m_terminal": single_m_terminal,
        "married_terminal": married_terminal,
    }


def _solve_consent(*, enable_jit: bool = False):
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=_make_consent_regimes(), derived_categoricals={}
        ),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {
                "single_f": jnp.int32(0),
                "single_f_terminal": jnp.int32(1),
                "single_m_terminal": jnp.int32(2),
                "married_terminal": jnp.int32(3),
            }
        ),
        enable_jit=enable_jit,
    )
    flat_params = MappingProxyType(
        {
            "single_f": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "single_f_terminal": MappingProxyType({}),
            "single_m_terminal": MappingProxyType({}),
            "married_terminal": MappingProxyType({}),
        }
    )
    return solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=enable_jit,
    )


def test_consent_gate_routes_marriage_value_and_fallback():
    """UNANIMITY: gate open -> married value flows; one partner short -> fallback.

    wage=1: (V_married_f=2 > V_single_f_ref=1.5) & (V_married_m=1 > V_single_m_ref
        =0.5) -> OPEN. The wife's continuation reads V_married_f = 2.
    wage=2: (4 > 3) & (2 > 3) -> husband's outside option beats marriage -> CLOSED
        (unanimity). The wife's continuation reads her single fallback = 3.

    Wbar = [2, 3]; V_single_f(period 0) = wage + beta*Wbar(wage):
        wage=1: 1 + 0.95*2 = 2.9;  wage=2: 2 + 0.95*3 = 4.85.
    """
    solution, _sim, _divorce = _solve_consent()
    V_single_f = np.asarray(solution[0]["single_f"])
    np.testing.assert_allclose(V_single_f, np.array([2.9, 4.85]), rtol=1e-6)


def test_consent_gate_matches_under_jit():
    """The public (jitted) path folds the same edge."""
    solution, _sim, _divorce = _solve_consent(enable_jit=True)
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"]), np.array([2.9, 4.85]), rtol=1e-6
    )


# --------------------------------------------------------------------------------------
# Divorce edge: married (collective) -> married_ir (collective), ~D gate, per-
# stakeholder fallback to single_f / single_m (reusing the slice-3 IR miniature)
# --------------------------------------------------------------------------------------

_WAGE3 = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)  # {1, 2, 3}


def _u_married_ir_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 3.0 * (1.0 - work) + 2.0 * wage * work


def _u_married_ir_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.5 * (1.0 - work) + wage * work


def _u_single_f_ir(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    target = jnp.where((wage > 1.5) & (wage < 2.5), 5.5, 1.5)
    return target * work


def _u_single_m_ir(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 1.0 * work + 0.0 * wage


def _u_zero(wage: ContinuousState) -> FloatND:
    return 0.0 * wage


def _u_zero_collective(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.0 * wage * work


def _ir_f(Q_f: FloatND, V_single_f_ref: FloatND, delta_f: FloatND) -> BoolND:
    return Q_f >= V_single_f_ref - delta_f


def _ir_m(Q_m: FloatND, V_single_m_ref: FloatND, delta_m: FloatND) -> BoolND:
    return Q_m >= V_single_m_ref - delta_m


def _no_divorce_gate(D_target: BoolND) -> BoolND:
    return ~D_target


def _make_divorce_regimes() -> dict[str, Regime]:
    married = Regime(
        transition={"married_ir": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _u_zero_collective,
            "utility_m": _u_zero_collective,
        },
        gated_edges={
            "married_ir": GatedEdge(
                gate=_no_divorce_gate,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="single_f", projection={"wage": _identity_wage}
                        ),
                    ),
                    "m": EdgeLeg(
                        target_stakeholder="m",
                        fallback=SamePeriodRef(
                            regime="single_m", projection={"wage": _identity_wage}
                        ),
                    ),
                },
            )
        },
    )
    married_ir = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        stakeholders=("f", "m"),
        states={"wage": _WAGE3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_married_ir_f, "utility_m": _u_married_ir_m},
        value_constraints={"ir_f": _ir_f, "ir_m": _ir_m},
        same_period_refs={
            "V_single_f_ref": SamePeriodRef(
                regime="single_f", projection={"wage": _identity_wage}
            ),
            "V_single_m_ref": SamePeriodRef(
                regime="single_m", projection={"wage": _identity_wage}
            ),
        },
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        stakeholders=("f", "m"),
        states={"wage": _WAGE3},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_zero_collective, "utility_m": _u_zero_collective},
    )
    single_f = Regime(
        transition={"single_f_terminal": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        states={"wage": _WAGE3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_single_f_ir},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wage": _WAGE3},
        functions={"utility": _u_zero},
    )
    single_m = single_f.replace(
        transition={"single_m_terminal": MarkovTransition(_prob_one)},
        functions={"utility": _u_single_m_ir},
    )
    single_m_terminal = single_f_terminal.replace()
    return {
        "married": married,
        "married_ir": married_ir,
        "married_terminal": married_terminal,
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m": single_m,
        "single_m_terminal": single_m_terminal,
    }


def _solve_divorce(*, enable_jit: bool = False):
    ages = AgeGrid(start=0, stop=3, step="Y")
    names = list(_make_divorce_regimes())
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=_make_divorce_regimes(), derived_categoricals={}
        ),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {name: jnp.int32(i) for i, name in enumerate(names)}
        ),
        enable_jit=enable_jit,
    )
    flat_params = MappingProxyType(
        {
            "married": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "married_ir": MappingProxyType(
                {
                    "H__discount_factor": jnp.asarray(_BETA),
                    "ir_f__delta_f": jnp.asarray(0.5),
                    "ir_m__delta_m": jnp.asarray(0.2),
                }
            ),
            "married_terminal": MappingProxyType({}),
            "single_f": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "single_f_terminal": MappingProxyType({}),
            "single_m": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "single_m_terminal": MappingProxyType({}),
        }
    )
    return solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=enable_jit,
    )


# married_ir period-1 solution (slice-3 IR miniature): D at wage=2, -inf there.
# Wbar = where(~D, V_married_ir, own single fallback):
#   wage=1 (D=F): (2, 1);  wage=2 (D=T): (V_single_f=5.5, V_single_m=1.0);
#   wage=3 (D=F): (6, 3).
# V_married(period 0) = beta * Wbar (utilities 0, wage fixed).
_EXPECTED_WBAR = np.array([[2.0, 1.0], [5.5, 1.0], [6.0, 3.0]])
_EXPECTED_V_MARRIED_0 = _BETA * _EXPECTED_WBAR


def test_divorce_edge_routes_each_stakeholder_to_own_fallback_no_nan():
    """D=True cell: each stakeholder's continuation reads its OWN single fallback.

    The two components differ at the divorce cell (wife 5.225 vs husband 0.9025),
    and NOTHING is NaN despite the target being -inf there (the -inf/where guard).
    """
    solution, _sim, divorce_flags = _solve_divorce()
    V_married_0 = np.asarray(solution[0]["married"])
    assert V_married_0.shape == (3, 2)
    assert not np.any(np.isnan(V_married_0))
    np.testing.assert_allclose(V_married_0, _EXPECTED_V_MARRIED_0, rtol=1e-6)
    # The divorce cell's two stakeholder continuations differ (own fallbacks).
    assert not np.isclose(V_married_0[1, 0], V_married_0[1, 1])
    # The source couple itself has no value constraints -> D all False.
    np.testing.assert_array_equal(
        np.asarray(divorce_flags[0]["married"]), np.array([False, False, False])
    )


def test_divorce_edge_matches_under_jit():
    solution, _sim, _divorce = _solve_divorce(enable_jit=True)
    np.testing.assert_allclose(
        np.asarray(solution[0]["married"]), _EXPECTED_V_MARRIED_0, rtol=1e-6
    )


# --------------------------------------------------------------------------------------
# Scope fences and build-time validation (pins)
# --------------------------------------------------------------------------------------


def test_raw_ungated_mixed_transition_still_rejected():
    """A singleton regime reaching a collective target WITHOUT an edge is rejected."""
    regimes = _make_consent_regimes()
    # Drop the gated edge but keep the singleton -> collective transition.
    regimes["single_f"] = regimes["single_f"].replace(gated_edges={})
    ages = AgeGrid(start=0, stop=2, step="Y")
    with pytest.raises(NotImplementedError, match="stakeholders"):
        process_regimes(
            user_regimes=finalize_regimes(
                user_regimes=regimes, derived_categoricals={}
            ),
            ages=ages,
            regime_names_to_ids=MappingProxyType(
                {n: jnp.int32(i) for i, n in enumerate(regimes)}
            ),
            enable_jit=False,
        )


def test_probabilistic_gate_is_rejected():
    """A stochastic (MarkovTransition) gate is out of scope — boolean only."""
    with pytest.raises(RegimeInitializationError, match="boolean"):
        Regime(
            transition={"married_terminal": MarkovTransition(_prob_one)},
            active=lambda age: age < 1,
            states={"wage": _WAGE},
            state_transitions={"wage": fixed_transition("wage")},
            actions={"work": DiscreteGrid(Work)},
            functions={"utility": _u_single_f},
            gated_edges={
                "married_terminal": GatedEdge(
                    gate=MarkovTransition(_prob_one),
                    legs={
                        "f": EdgeLeg(
                            target_stakeholder="f",
                            fallback=SamePeriodRef(
                                regime="single_f_terminal",
                                projection={"wage": _identity_wage},
                            ),
                        )
                    },
                )
            },
        )


def test_edge_fallback_to_unknown_regime_is_rejected():
    """A gated edge whose fallback names a missing regime is rejected at build."""
    regimes = _make_consent_regimes()
    bad_edge = GatedEdge(
        gate=_consent_gate,
        legs={
            "f": EdgeLeg(
                target_stakeholder="f",
                fallback=SamePeriodRef(
                    regime="no_such_regime", projection={"wage": _identity_wage}
                ),
            )
        },
        gate_refs={
            "V_single_f_ref": SamePeriodRef(
                regime="single_f_terminal", projection={"wage": _identity_wage}
            ),
            "V_single_m_ref": SamePeriodRef(
                regime="single_m_terminal", projection={"wage": _identity_wage}
            ),
        },
    )
    regimes["single_f"] = regimes["single_f"].replace(
        gated_edges={"married_terminal": bad_edge}
    )
    ages = AgeGrid(start=0, stop=2, step="Y")
    with pytest.raises(ModelInitializationError, match="no_such_regime"):
        process_regimes(
            user_regimes=finalize_regimes(
                user_regimes=regimes, derived_categoricals={}
            ),
            ages=ages,
            regime_names_to_ids=MappingProxyType(
                {n: jnp.int32(i) for i, n in enumerate(regimes)}
            ),
            enable_jit=False,
        )


def test_edge_leg_naming_a_missing_target_stakeholder_is_rejected():
    """An edge leg naming a target stakeholder the target lacks is rejected."""
    regimes = _make_consent_regimes()
    bad_edge = GatedEdge(
        gate=_consent_gate,
        legs={
            "f": EdgeLeg(
                target_stakeholder="not_a_stakeholder",
                fallback=SamePeriodRef(
                    regime="single_f_terminal", projection={"wage": _identity_wage}
                ),
            )
        },
        gate_refs={
            "V_single_f_ref": SamePeriodRef(
                regime="single_f_terminal", projection={"wage": _identity_wage}
            ),
            "V_single_m_ref": SamePeriodRef(
                regime="single_m_terminal", projection={"wage": _identity_wage}
            ),
        },
    )
    regimes["single_f"] = regimes["single_f"].replace(
        gated_edges={"married_terminal": bad_edge}
    )
    ages = AgeGrid(start=0, stop=2, step="Y")
    with pytest.raises(ModelInitializationError, match="not_a_stakeholder"):
        process_regimes(
            user_regimes=finalize_regimes(
                user_regimes=regimes, derived_categoricals={}
            ),
            ages=ages,
            regime_names_to_ids=MappingProxyType(
                {n: jnp.int32(i) for i, n in enumerate(regimes)}
            ),
            enable_jit=False,
        )


# --------------------------------------------------------------------------------------
# Full mini-EKL topology in ONE model via the public API
# --------------------------------------------------------------------------------------


@categorical(ordered=False)
class EKLRegimeId:
    single_f: ScalarInt
    single_m: ScalarInt
    single_f_p1: ScalarInt
    single_m_p1: ScalarInt
    married: ScalarInt
    married_terminal: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt


def _make_full_topology_regimes() -> dict[str, Regime]:
    def _consent_leg(fallback_regime: str, stakeholder: str) -> dict[str, EdgeLeg]:
        return {
            stakeholder: EdgeLeg(
                target_stakeholder=stakeholder,
                fallback=SamePeriodRef(
                    regime=fallback_regime, projection={"wage": _identity_wage}
                ),
            )
        }

    _consent_gate_refs = {
        "V_single_f_ref": SamePeriodRef(
            regime="single_f_p1", projection={"wage": _identity_wage}
        ),
        "V_single_m_ref": SamePeriodRef(
            regime="single_m_p1", projection={"wage": _identity_wage}
        ),
    }
    single_f = Regime(
        transition={"married": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wage": _WAGE3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_single_f},
        gated_edges={
            "married": GatedEdge(
                gate=_consent_gate,
                legs=_consent_leg("single_f_p1", "f"),
                gate_refs=_consent_gate_refs,
            )
        },
    )
    single_m = single_f.replace(
        gated_edges={
            "married": GatedEdge(
                gate=_consent_gate,
                legs=_consent_leg("single_m_p1", "m"),
                gate_refs=_consent_gate_refs,
            )
        },
    )
    single_f_p1 = Regime(
        transition={"single_f_terminal": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        states={"wage": _WAGE3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_single_f_ir},
    )
    single_m_p1 = single_f_p1.replace(
        transition={"single_m_terminal": MarkovTransition(_prob_one)},
        functions={"utility": _u_single_m_ir},
    )
    married = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        stakeholders=("f", "m"),
        states={"wage": _WAGE3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_married_ir_f, "utility_m": _u_married_ir_m},
        value_constraints={"ir_f": _ir_f, "ir_m": _ir_m},
        same_period_refs={
            "V_single_f_ref": SamePeriodRef(
                regime="single_f_p1", projection={"wage": _identity_wage}
            ),
            "V_single_m_ref": SamePeriodRef(
                regime="single_m_p1", projection={"wage": _identity_wage}
            ),
        },
        gated_edges={
            "married_terminal": GatedEdge(
                gate=_no_divorce_gate,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="single_f_terminal",
                            projection={"wage": _identity_wage},
                        ),
                    ),
                    "m": EdgeLeg(
                        target_stakeholder="m",
                        fallback=SamePeriodRef(
                            regime="single_m_terminal",
                            projection={"wage": _identity_wage},
                        ),
                    ),
                },
            )
        },
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        stakeholders=("f", "m"),
        states={"wage": _WAGE3},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_zero_collective, "utility_m": _u_zero_collective},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wage": _WAGE3},
        functions={"utility": _u_zero},
    )
    single_m_terminal = single_f_terminal.replace()
    return {
        "single_f": single_f,
        "single_m": single_m,
        "single_f_p1": single_f_p1,
        "single_m_p1": single_m_p1,
        "married": married,
        "married_terminal": married_terminal,
        "single_f_terminal": single_f_terminal,
        "single_m_terminal": single_m_terminal,
    }


def test_full_ekl_topology_via_public_model_api():
    """single_f/single_m + married with consent edge + divorce edge + IR, one model.

    married (period 1) is the slice-3 IR miniature (divorce cell at wage=2). Its
    divorce edge routes into the (zero-valued) terminal couple — gate ~D is open
    there, so its continuation is zero and its V equals the felicity readout.
    single_f (period 0) reaches married through the CONSENT edge:

    gate(wage) = (V_married_f > V_single_f) & (V_married_m > V_single_m):
      w=1: (2>1.5)&(1>1)  -> closed (husband indifferent, strict).
      w=2: (-inf>5.5)&...  -> closed.
      w=3: (6>1.5)&(3>1)  -> OPEN.
    Wbar_f = [1.5, 5.5, 6]; V_single_f(0) = wage + 0.95*Wbar_f = [2.425, 7.225, 8.7].
    """
    ages = AgeGrid(start=0, stop=3, step="Y")
    model = Model(
        regimes=_make_full_topology_regimes(),
        ages=ages,
        regime_id_class=EKLRegimeId,
    )
    solution = model.solve(
        params={"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2},
        log_level="off",
    )
    # married period-1: the slice-3 IR miniature (continuation zero via divorce edge).
    np.testing.assert_allclose(
        np.asarray(solution[1]["married"]),
        np.array([[2.0, 1.0], [-np.inf, -np.inf], [6.0, 3.0]]),
        rtol=1e-6,
    )
    # single_f period-0: mutual-consent continuation.
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"]),
        np.array([2.425, 7.225, 8.7]),
        rtol=1e-6,
    )


def test_singleton_source_with_two_legs_is_rejected():
    """A singleton source must declare exactly one edge leg."""
    with pytest.raises(RegimeInitializationError, match="exactly one leg"):
        Regime(
            transition={"married_terminal": MarkovTransition(_prob_one)},
            active=lambda age: age < 1,
            states={"wage": _WAGE},
            state_transitions={"wage": fixed_transition("wage")},
            actions={"work": DiscreteGrid(Work)},
            functions={"utility": _u_single_f},
            gated_edges={
                "married_terminal": GatedEdge(
                    gate=_consent_gate,
                    legs={
                        "f": EdgeLeg(
                            target_stakeholder="f",
                            fallback=SamePeriodRef(
                                regime="single_f_terminal",
                                projection={"wage": _identity_wage},
                            ),
                        ),
                        "extra": EdgeLeg(
                            target_stakeholder="m",
                            fallback=SamePeriodRef(
                                regime="single_m_terminal",
                                projection={"wage": _identity_wage},
                            ),
                        ),
                    },
                )
            },
        )
