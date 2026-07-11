"""Integration tests for E4: forward simulation of collective regimes (slice 6).

pylcm simulate recomputes the argmax per period against the STORED next-period
V rather than storing policies (design doc `pylcm-extension-collective-regimes.md`
§2 E4). For a collective regime this means, per simulated household:

1. **Household argmax at the realized state** — the joint action `a*` both
   stakeholders follow, recomputed the same way `get_max_Q_over_a`'s collective
   branch does at solve (`collective_argmax_and_readout`, `Q_arr` carrying a
   trailing stakeholder axis); each stakeholder's own value is read off at
   that shared argmax and both continue to be tracked (a trailing stakeholder
   axis on the recorded `V_arr`, mirroring the solve-side V topology).
2. **The value router (gated edges)** — a regime declaring `gated_edges`
   substitutes the target's gated continuation `Wbar` for the raw target V
   before choosing this period's action (exactly like the solve-side kernel's
   `_with_edge_substitution`, but computed from the already-solved solution),
   then interpolates the SAME boolean `gate` at the realized candidate
   target-state draw to decide ACTUAL regime routing: the target when open, a
   leg's fallback regime (with its own projected states) when closed.
3. **The stochastic offer draw** — a target-only `MarkovTransition` state
   (e.g. a drawn spouse type) is realized by the ordinary (pre-existing,
   collective-agnostic) `calculate_next_states` machinery; nothing
   gated-edge-specific is needed for the draw itself.

Divorce (a COLLECTIVE source's multi-leg gated edge) is a documented SCOPE
FENCE: forward simulation is a fixed-size population pass, so one row cannot
literally become two independently-tracked future rows. What this slice
delivers — and what is tested here — is that EVERY leg's own fallback
(regime, state) is computed correctly and written into that fallback
regime's per-subject state slot, and that the row's own continuing
membership follows the FIRST declared leg (deterministic, documented
convention); see `_lcm.simulation.gated_routing`'s module docstring.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.simulation.simulate import simulate
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, LinSpacedGrid, categorical, fixed_transition
from lcm.ages import AgeGrid
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
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


_BETA = 0.95


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _solve_and_process(
    *, regimes_dict: dict[str, Regime], ages: AgeGrid, regime_names: list[str]
):
    """Shared build+solve harness (kernel-level, mirrors the slice-4/5 tests)."""
    regime_names_to_ids = MappingProxyType(
        {name: jnp.int32(i) for i, name in enumerate(regime_names)}
    )
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=regimes_dict, derived_categoricals={}
        ),
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=False,
    )
    return regimes, regime_names_to_ids


# ----------------------------------------------------------------------------------
# Test 1: a solved (non-gated) collective regime simulates the recomputed joint
# argmax over two periods, both stakeholders tracked (reuses the
# test_nonterminal_collective_solve.py couple model).
# ----------------------------------------------------------------------------------


@categorical(ordered=False)
class CoupleRegimeId:
    couple: ScalarInt
    couple_terminal: ScalarInt


def _utility_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work)


def _utility_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    consumption = wage * work
    return 2.0 * consumption


def _next_wage(work: DiscreteAction) -> ContinuousState:
    return 40.0 * work + 8.0 * (1.0 - work)


def _next_couple_regime() -> ScalarInt:
    return CoupleRegimeId.couple_terminal


_WAGE_GRID_2 = LinSpacedGrid(start=8.0, stop=40.0, n_points=2)


def _make_couple_regimes() -> dict[str, Regime]:
    couple = Regime(
        transition=_next_couple_regime,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID_2},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID_2},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
    )
    return {"couple": couple, "couple_terminal": couple_terminal}


def test_couple_simulates_recomputed_joint_argmax_two_periods():
    """A married couple's simulated actions/values match the hand-computed path.

    Hand computation (see `test_nonterminal_collective_solve.py`): at BOTH
    wage=8 and wage=40, the period-0 household argmax is `work` (the
    continuation through tomorrow's high wage dominates); V_0 = (46, 92) at
    wage=8 and (78, 156) at wage=40. Both households transition to wage=40
    and, at the terminal period, again choose `work`; V_1 = (40, 80) for
    both.
    """
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes_dict = _make_couple_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType(
        {
            "couple": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "couple_terminal": MappingProxyType({}),
        }
    )
    solution, _sim_policies, divorce_flags = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )

    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([8.0, 40.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_divorce_flags=divorce_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )

    period_0 = result.raw_results["couple"][0]
    np.testing.assert_array_equal(np.asarray(period_0.actions["work"]), [1, 1])
    np.testing.assert_allclose(
        np.asarray(period_0.V_arr), [[46.0, 92.0], [78.0, 156.0]], rtol=1e-6
    )

    period_1 = result.raw_results["couple_terminal"][1]
    np.testing.assert_array_equal(np.asarray(period_1.in_regime), [True, True])
    np.testing.assert_array_equal(np.asarray(period_1.states["wage"]), [40.0, 40.0])
    np.testing.assert_array_equal(np.asarray(period_1.actions["work"]), [1, 1])
    np.testing.assert_allclose(
        np.asarray(period_1.V_arr), [[40.0, 80.0], [40.0, 80.0]], rtol=1e-6
    )


def test_couple_simulate_with_runtime_validation_enabled():
    """The trailing stakeholder axis on `V_arr` broadcasts against `in_regime`.

    `log_level="off"` (used by every other test in this module) skips the
    `validate_V` / NaN-diagnostic code paths entirely; those paths compare
    `V_arr` against `in_regime`/`subject_ids_in_regime`, which for a
    collective regime's `(n_subjects, n_stakeholders)` V need an explicit
    broadcast (a singleton regime's `V_arr` is already `(n_subjects,)`).
    Regression pin for that reshape, at `log_level="debug"` (the strictest
    level — raises on any validation failure).
    """
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes_dict = _make_couple_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType(
        {
            "couple": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "couple_terminal": MappingProxyType({}),
        }
    )
    solution, _sim_policies, divorce_flags = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([8.0, 40.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="debug"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_divorce_flags=divorce_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )
    np.testing.assert_allclose(
        np.asarray(result.raw_results["couple"][0].V_arr),
        [[46.0, 92.0], [78.0, 156.0]],
        rtol=1e-6,
    )


# ----------------------------------------------------------------------------------
# Test 2: consent routing — a singleton source (single_f) with a gated marriage
# edge; households whose realized wage clears mutual consent marry, others stay
# single. Exact per-seed routing (deterministic wage draws, no stochastic offer).
# ----------------------------------------------------------------------------------

_WAGE_2 = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)  # {1.0, 2.0}


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
    return (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)


def _make_consent_regimes() -> dict[str, Regime]:
    single_f = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wage": _WAGE_2},
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
        states={"wage": _WAGE_2},
        functions={"utility": _u_single_f_terminal},
    )
    single_m_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_2},
        functions={"utility": _u_single_m_terminal},
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_2},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_married_f, "utility_m": _u_married_m},
    )
    return {
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m_terminal": single_m_terminal,
        "married_terminal": married_terminal,
    }


def _solve_consent():
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes_dict = _make_consent_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType(
        {
            "single_f": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "single_f_terminal": MappingProxyType({}),
            "single_m_terminal": MappingProxyType({}),
            "married_terminal": MappingProxyType({}),
        }
    )
    solution, _sim_policies, divorce_flags = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return ages, regimes, regime_names_to_ids, flat_params, solution, divorce_flags


def test_consent_routing_simulate_matches_gate_exactly():
    """wage=1 clears mutual consent (marries); wage=2 does not (stays single).

    Gate at wage=1: (V_married_f=2 > V_single_f_ref=1.5) & (V_married_m=1 >
    V_single_m_ref=0.5) -> OPEN. Gate at wage=2: (4 > 3) & (2 > 3) -> the
    husband's outside option beats marriage -> CLOSED (unanimity).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, divorce_flags = (
        _solve_consent()
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array(
                [regime_names_to_ids["single_f"]] * 2, dtype=jnp.int32
            ),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_divorce_flags=divorce_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )

    # Period-0 own value matches the hand-computed / solve-verified path
    # (test_gated_edges_collective_solve.py::test_consent_gate_routes_...).
    period_0 = result.raw_results["single_f"][0]
    np.testing.assert_allclose(np.asarray(period_0.V_arr), [2.9, 4.85], rtol=1e-6)

    # wage=1 -> married_terminal (gate open); wage=2 -> single_f_terminal
    # (gate closed, the source's own — sole — leg fallback).
    married = result.raw_results["married_terminal"][1]
    single_f_term = result.raw_results["single_f_terminal"][1]
    np.testing.assert_array_equal(np.asarray(married.in_regime), [True, False])
    np.testing.assert_array_equal(np.asarray(single_f_term.in_regime), [False, True])

    # The routed households' values match the OPEN / CLOSED branch exactly.
    np.testing.assert_allclose(
        np.asarray(married.V_arr)[0], [2.0, 1.0], rtol=1e-6
    )  # u_married_f(wage=1)=2, u_married_m(wage=1)=1
    np.testing.assert_allclose(
        np.asarray(single_f_term.V_arr)[1], 3.0, rtol=1e-6
    )  # u_single_f_terminal(wage=2) = 1.5*2


def test_consent_routing_never_populates_the_non_routed_target():
    """single_m_terminal has no source edge in this topology, so it stays empty."""
    ages, regimes, regime_names_to_ids, flat_params, solution, divorce_flags = (
        _solve_consent()
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array(
                [regime_names_to_ids["single_f"]] * 2, dtype=jnp.int32
            ),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_divorce_flags=divorce_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )
    single_m_term = result.raw_results["single_m_terminal"][1]
    np.testing.assert_array_equal(np.asarray(single_m_term.in_regime), [False, False])


# ----------------------------------------------------------------------------------
# Test 3: divorce routing — a married cohort where slice-3 IR empties the mask
# at wage=2; reuses the divorce-edge miniature from
# test_gated_edges_collective_solve.py::_make_divorce_regimes.
# ----------------------------------------------------------------------------------

_WAGE_3 = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)  # {1, 2, 3}


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
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_zero_collective, "utility_m": _u_zero_collective},
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
        states={"wage": _WAGE_3},
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
        states={"wage": _WAGE_3},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_zero_collective, "utility_m": _u_zero_collective},
    )
    single_f = Regime(
        transition={"single_f_terminal": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_single_f_ir},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wage": _WAGE_3},
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


def _solve_divorce():
    ages = AgeGrid(start=0, stop=3, step="Y")
    regimes_dict = _make_divorce_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
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
    solution, _sim_policies, divorce_flags = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return ages, regimes, regime_names_to_ids, flat_params, solution, divorce_flags


def test_divorce_edge_routes_primary_leg_to_own_single_regime():
    """D=True at wage=2 (slice-3 IR empties the mask there, see solve test).

    The married household's row is routed to the FIRST declared leg's
    fallback ("f" -> `single_f`) instead of `married_ir` — a real regime
    membership change, not merely a value-side fold. wage=1 and wage=3 stay
    married (`married_ir`, D=False there).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, divorce_flags = (
        _solve_divorce()
    )
    np.testing.assert_array_equal(
        np.asarray(divorce_flags[1]["married_ir"]), [False, True, False]
    )

    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [regime_names_to_ids["married"]] * 3, dtype=jnp.int32
            ),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_divorce_flags=divorce_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )

    married_ir = result.raw_results["married_ir"][1]
    single_f = result.raw_results["single_f"][1]
    single_m = result.raw_results["single_m"][1]
    np.testing.assert_array_equal(np.asarray(married_ir.in_regime), [True, False, True])
    np.testing.assert_array_equal(np.asarray(single_f.in_regime), [False, True, False])
    # single_m never becomes this row's OWN continuing membership — the
    # documented scope fence (primary-leg convention); see module docstring.
    np.testing.assert_array_equal(np.asarray(single_m.in_regime), [False, False, False])

    # The divorced household's (wage=2) value under its routed regime matches
    # the IR miniature's single fallback exactly (hand-computed in
    # test_gated_edges_collective_solve.py): V_single_f(wage=2) = 5.5.
    np.testing.assert_allclose(np.asarray(single_f.V_arr)[1], 5.5, rtol=1e-6)
    # The still-married households keep their married_ir value.
    np.testing.assert_allclose(np.asarray(married_ir.V_arr)[0], [2.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(married_ir.V_arr)[2], [6.0, 3.0], rtol=1e-6)


def test_divorce_edge_leg_projector_computes_the_non_primary_states_too():
    """Unit-level: the SECOND leg's ("m") own fallback projector is correct.

    Not exercised by the end-to-end assertion above (the primary-leg
    convention means `single_m` never becomes the row's own continuing
    membership in that scenario) — this directly calls the resolved
    projector the value router builds and consumes, confirming it maps the
    target-grid wage coordinate onto `single_m`'s own state coordinate
    exactly like the fallback the solve-side fold reads for the SAME leg.
    """
    _ages, regimes, _ids, _params, _solution, _divorce = _solve_divorce()
    married = regimes["married"]
    edge = married.gated_edges["married_ir"]
    leg_names = [leg.source_stakeholder for leg in edge.legs]
    assert leg_names == ["f", "m"]
    m_projector = married.gated_edge_leg_projectors["married_ir"][1]
    projected = m_projector(wage=jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(np.asarray(projected["wage"]), [1.0, 2.0, 3.0])


# ----------------------------------------------------------------------------------
# Test 4: value-masked simulate — the simulated argmax never selects an action
# excluded by a slice-3 value constraint (reuses the divorce miniature's IR
# mask: married_ir's mask is empty at wage=2).
# ----------------------------------------------------------------------------------


def test_value_masked_simulate_reports_the_solved_masked_value():
    """A household evaluated directly IN married_ir at the D=True cell gets -inf.

    Initializing subjects directly into `married_ir` (bypassing the `married`
    source, so the value router's routing does not preempt it) confirms the
    simulate-side Q_and_F applies the IDENTICAL value-aware feasibility mask
    (E2) the solve phase used: the argmax's `V_arr` reports the same `-inf`
    sentinel `solution[1]["married_ir"]` carries at wage=2 (the empty mask;
    the household's real-world routing away from this cell is E3'/E4's
    separate concern, covered by the divorce test above).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, divorce_flags = (
        _solve_divorce()
    )
    np.testing.assert_allclose(
        np.asarray(solution[1]["married_ir"])[1], [-np.inf, -np.inf]
    )

    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([1.0, 1.0, 1.0]),
            "regime_id": jnp.array(
                [regime_names_to_ids["married_ir"]] * 3, dtype=jnp.int32
            ),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_divorce_flags=divorce_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )
    married_ir = result.raw_results["married_ir"][1]
    np.testing.assert_array_equal(np.asarray(married_ir.in_regime), [True, True, True])
    np.testing.assert_allclose(
        np.asarray(married_ir.V_arr), [[2.0, 1.0], [-np.inf, -np.inf], [6.0, 3.0]]
    )


# ----------------------------------------------------------------------------------
# Test 5: pin — a still-unsupported simulate construct raises clearly rather
# than crashing obscurely. `Model.simulate()` does not yet surface
# `period_to_regime_to_divorce_flags` through its public API (documented gap,
# see `simulate()`'s own docstring); calling the internal `simulate()` the
# same way (omitting it) for a divorce-gated model must fail with a clear
# message, not a bare `None > 0.5` TypeError.
# ----------------------------------------------------------------------------------


def test_gate_reading_d_target_without_divorce_flags_raises_clearly():
    ages, regimes, regime_names_to_ids, flat_params, solution, _divorce_flags = (
        _solve_divorce()
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([2.0]),
            "age": jnp.array([0.0]),
            "regime_id": jnp.array([regime_names_to_ids["married"]], dtype=jnp.int32),
        }
    )
    with pytest.raises(NotImplementedError, match="D_target"):
        simulate(
            flat_params=flat_params,
            initial_conditions=initial_conditions,
            regimes=regimes,
            regime_names_to_ids=regime_names_to_ids,
            logger=get_logger(log_level="off"),
            period_to_regime_to_V_arr=solution,
            # period_to_regime_to_divorce_flags omitted (defaults to empty).
            ages=ages,
            simulation_output_dtypes={},
            seed=0,
        )
