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

Dissolution (a COLLECTIVE source's multi-leg gated edge) is a documented SCOPE
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
import pandas as pd
import pytest

import lcm.model as model_module
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.simulation.simulate import simulate
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, LinSpacedGrid, Model, categorical, fixed_transition
from lcm.ages import AgeGrid
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
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
    solution, _sim_policies, dissolution_flags = solve(
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
        period_to_regime_to_dissolution_flags=dissolution_flags,
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
    solution, _sim_policies, dissolution_flags = solve(
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
        period_to_regime_to_dissolution_flags=dissolution_flags,
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
    solution, _sim_policies, dissolution_flags = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags


def test_consent_routing_simulate_matches_gate_exactly():
    """wage=1 clears mutual consent (marries); wage=2 does not (stays single).

    Gate at wage=1: (V_married_f=2 > V_single_f_ref=1.5) & (V_married_m=1 >
    V_single_m_ref=0.5) -> OPEN. Gate at wage=2: (4 > 3) & (2 > 3) -> the
    husband's outside option beats marriage -> CLOSED (unanimity).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
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
        period_to_regime_to_dissolution_flags=dissolution_flags,
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
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
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
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )
    single_m_term = result.raw_results["single_m_terminal"][1]
    np.testing.assert_array_equal(np.asarray(single_m_term.in_regime), [False, False])


# ----------------------------------------------------------------------------------
# Test 3: dissolution routing — a married cohort where slice-3 IR empties the mask
# at wage=2; reuses the dissolution-edge miniature from
# test_gated_edges_collective_solve.py::_make_dissolution_regimes.
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


def _no_dissolution_gate(D_target: BoolND) -> BoolND:
    return ~D_target


def _make_dissolution_regimes() -> dict[str, Regime]:
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
                gate=_no_dissolution_gate,
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


def _solve_dissolution():
    ages = AgeGrid(start=0, stop=3, step="Y")
    regimes_dict = _make_dissolution_regimes()
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
    solution, _sim_policies, dissolution_flags = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags


def test_dissolution_edge_routes_primary_leg_to_own_single_regime():
    """D=True at wage=2 (slice-3 IR empties the mask there, see solve test).

    The married household's row is routed to the FIRST declared leg's
    fallback ("f" -> `single_f`) instead of `married_ir` — a real regime
    membership change, not merely a value-side fold. wage=1 and wage=3 stay
    married (`married_ir`, D=False there).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_dissolution()
    )
    np.testing.assert_array_equal(
        np.asarray(dissolution_flags[1]["married_ir"]), [False, True, False]
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
        period_to_regime_to_dissolution_flags=dissolution_flags,
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

    # The dissolutiond household's (wage=2) value under its routed regime matches
    # the IR miniature's single fallback exactly (hand-computed in
    # test_gated_edges_collective_solve.py): V_single_f(wage=2) = 5.5.
    np.testing.assert_allclose(np.asarray(single_f.V_arr)[1], 5.5, rtol=1e-6)
    # The still-married households keep their married_ir value.
    np.testing.assert_allclose(np.asarray(married_ir.V_arr)[0], [2.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(married_ir.V_arr)[2], [6.0, 3.0], rtol=1e-6)


def test_dissolution_edge_leg_projector_computes_the_non_primary_states_too():
    """Unit-level: the SECOND leg's ("m") own fallback projector is correct.

    Not exercised by the end-to-end assertion above (the primary-leg
    convention means `single_m` never becomes the row's own continuing
    membership in that scenario) — this directly calls the resolved
    projector the value router builds and consumes, confirming it maps the
    target-grid wage coordinate onto `single_m`'s own state coordinate
    exactly like the fallback the solve-side fold reads for the SAME leg.
    """
    _ages, regimes, _ids, _params, _solution, _dissolution = _solve_dissolution()
    married = regimes["married"]
    edge = married.gated_edges["married_ir"]
    leg_names = [leg.source_stakeholder for leg in edge.legs]
    assert leg_names == ["f", "m"]
    m_projector = married.gated_edge_leg_projectors["married_ir"][1]
    projected = m_projector(wage=jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(np.asarray(projected["wage"]), [1.0, 2.0, 3.0])


# ----------------------------------------------------------------------------------
# Test 4: value-masked simulate — the simulated argmax never selects an action
# excluded by a slice-3 value constraint (reuses the dissolution miniature's IR
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
    separate concern, covered by the dissolution test above).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_dissolution()
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
        period_to_regime_to_dissolution_flags=dissolution_flags,
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
# `period_to_regime_to_dissolution_flags` through its public API (documented gap,
# see `simulate()`'s own docstring); calling the internal `simulate()` the
# same way (omitting it) for a dissolution-gated model must fail with a clear
# message, not a bare `None > 0.5` TypeError.
# ----------------------------------------------------------------------------------


def test_gate_reading_d_target_without_dissolution_flags_raises_clearly():
    ages, regimes, regime_names_to_ids, flat_params, solution, _dissolution_flags = (
        _solve_dissolution()
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
            # period_to_regime_to_dissolution_flags omitted (defaults to empty).
            ages=ages,
            simulation_output_dtypes={},
            seed=0,
        )


# ----------------------------------------------------------------------------------
# Test 6: regression — a gated edge's TARGET has a discrete state axis (in
# addition to a continuous one). `route_gated_edges` interpolates the fold's
# `gate` array at each subject's candidate target-state draw
# (`_lcm.simulation.gated_routing._call_vmapped_with_accepted_kwargs`); before
# that call was `vmap`-ped over subjects, a discrete target axis broke
# `jax.scipy.ndimage.map_coordinates`'s `len(coordinates) == input.ndim`
# invariant (advanced indexing on whole-population `(n_subjects,)` discrete
# indices collapses the discrete axes into a leading batch dimension while
# leaving the continuous axis trailing) — every OTHER gated-edge simulate test
# above uses a continuous-only target, where a `(n_subjects,)` coordinate
# array happens to broadcast correctly through `map_coordinates` by
# coincidence, so this gap went untested. Adapted from
# `_make_consent_regimes` (Test 2 above) by adding a discrete `educ` state,
# carried as-is (`fixed_transition`) and contributing 0 to every utility, so
# the gate arithmetic and expected routing/values are IDENTICAL to
# `test_consent_routing_simulate_matches_gate_exactly`.
# ----------------------------------------------------------------------------------


@categorical(ordered=True)
class Educ:
    low: ScalarInt  # code 0
    high: ScalarInt  # code 1


def _identity_educ(educ: DiscreteState) -> DiscreteState:
    return educ


def _u_single_f_educ(
    wage: ContinuousState, work: DiscreteAction, educ: DiscreteState
) -> FloatND:
    return wage * work + 0.0 * educ


def _u_single_f_terminal_educ(wage: ContinuousState, educ: DiscreteState) -> FloatND:
    return 1.5 * wage + 0.0 * educ  # {1.5, 3.0}


def _u_married_f_educ(
    wage: ContinuousState, work: DiscreteAction, educ: DiscreteState
) -> FloatND:
    return 2.0 * wage + 0.0 * work + 0.0 * educ  # {2, 4}


def _u_married_m_educ(
    wage: ContinuousState, work: DiscreteAction, educ: DiscreteState
) -> FloatND:
    return wage + 0.0 * work + 0.0 * educ  # {1, 2}


def _make_consent_regimes_with_discrete_target_axis() -> dict[str, Regime]:
    single_f = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        # ONLY difference vs. `_make_consent_regimes`: an added discrete
        # "educ" state, carried as-is (`fixed_transition`), on the source and
        # every regime the gated edge's fold/gate/fallback touch.
        states={"wage": _WAGE_2, "educ": DiscreteGrid(Educ)},
        state_transitions={
            "wage": fixed_transition("wage"),
            "educ": fixed_transition("educ"),
        },
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_single_f_educ},
        gated_edges={
            "married_terminal": GatedEdge(
                gate=_consent_gate,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="single_f_terminal",
                            projection={"wage": _identity_wage, "educ": _identity_educ},
                        ),
                    )
                },
                gate_refs={
                    "V_single_f_ref": SamePeriodRef(
                        regime="single_f_terminal",
                        projection={"wage": _identity_wage, "educ": _identity_educ},
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
        states={"wage": _WAGE_2, "educ": DiscreteGrid(Educ)},
        functions={"utility": _u_single_f_terminal_educ},
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
        states={"wage": _WAGE_2, "educ": DiscreteGrid(Educ)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_married_f_educ, "utility_m": _u_married_m_educ},
    )
    return {
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m_terminal": single_m_terminal,
        "married_terminal": married_terminal,
    }


def _solve_consent_discrete_axis():
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes_dict = _make_consent_regimes_with_discrete_target_axis()
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
    solution, _sim_policies, dissolution_flags = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags


def test_consent_routing_simulate_with_discrete_target_axis_routes_correctly():
    """Regression pin for the discrete-target-axis gate-interpolation crash.

    Four subjects cross wage in {1, 2} with educ in {low, high}, so the
    `vmap`-ped gate interpolator (`gated_routing._call_vmapped_with_accepted_kwargs`)
    is exercised at BOTH discrete indices, not just one — a discrete-axis
    lookup that silently read the wrong subject's index would flip a
    subject's routing here. Wage=1 clears mutual consent (marries)
    regardless of educ; wage=2 does not (stays single) regardless of educ —
    `educ` contributes 0.0 to every utility, so its exact numeric routing and
    values are identical to `test_consent_routing_simulate_matches_gate_exactly`.
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_consent_discrete_axis()
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 1.0, 2.0, 2.0]),
            "educ": jnp.array([0, 1, 0, 1], dtype=jnp.int32),
            "age": jnp.array([0.0, 0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [regime_names_to_ids["single_f"]] * 4, dtype=jnp.int32
            ),
        }
    )

    # Pre-fix, this raised: `ValueError: coordinates must be a sequence of
    # length input.ndim, but 1 != 2` (the un-vmapped gate interpolator).
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )

    # wage=1 (subjects 0, 1) -> married_terminal (gate open), regardless of
    # educ; wage=2 (subjects 2, 3) -> single_f_terminal (gate closed).
    married = result.raw_results["married_terminal"][1]
    single_f_term = result.raw_results["single_f_terminal"][1]
    np.testing.assert_array_equal(
        np.asarray(married.in_regime), [True, True, False, False]
    )
    np.testing.assert_array_equal(
        np.asarray(single_f_term.in_regime), [False, False, True, True]
    )

    # Every recorded value is finite (no stray -inf/NaN from a mis-shaped
    # interpolation), and the routed branches match the hand-computed values
    # exactly, unaffected by educ.
    married_V = np.asarray(married.V_arr)
    single_f_term_V = np.asarray(single_f_term.V_arr)
    assert np.all(np.isfinite(married_V[[0, 1]]))
    assert np.all(np.isfinite(single_f_term_V[[2, 3]]))
    np.testing.assert_allclose(married_V[0], [2.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(married_V[1], [2.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(single_f_term_V[2], 3.0, rtol=1e-6)
    np.testing.assert_allclose(single_f_term_V[3], 3.0, rtol=1e-6)

    # The discrete "educ" state itself was carried correctly into the routed
    # regime's own state slot (fixed_transition identity), for both branches.
    np.testing.assert_array_equal(np.asarray(married.states["educ"])[[0, 1]], [0, 1])
    np.testing.assert_array_equal(
        np.asarray(single_f_term.states["educ"])[[2, 3]], [0, 1]
    )


# ----------------------------------------------------------------------------------
# Test 7: public `Model.solve`/`Model.simulate` API — closes the gap Test 5 pins
# at the INTERNAL `simulate()` level. `Model.solve(return_dissolution_flags=True)`
# must surface the dissolution-flag mapping `backward_induction.solve` already
# computes (previously discarded), and `Model.simulate(...)` must accept it via
# `period_to_regime_to_dissolution_flags` and thread it down to the same dissolution
# routing exercised by Test 3 (`_make_dissolution_regimes`), reusing that fixture
# through the public API instead of the internal `process_regimes` harness.
# ----------------------------------------------------------------------------------


@categorical(ordered=False)
class DissolutionRegimeId:
    married: ScalarInt
    married_ir: ScalarInt
    married_terminal: ScalarInt
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m: ScalarInt
    single_m_terminal: ScalarInt


def _make_dissolution_model() -> Model:
    ages = AgeGrid(start=0, stop=3, step="Y")
    return Model(
        regimes=_make_dissolution_regimes(),
        ages=ages,
        regime_id_class=DissolutionRegimeId,
    )


_DISSOLUTION_PARAMS = {
    "discount_factor": _BETA,
    "delta_f": 0.5,
    "delta_m": 0.2,
}


def test_public_model_solve_return_dissolution_flags_matches_internal_solve():
    """`Model.solve(return_dissolution_flags=True)` surfaces the same `D` array.

    Same hand-computed cell as the internal-harness test
    (`test_dissolution_edge_routes_primary_leg_to_own_single_regime`): D=True only
    at wage=2 for `married_ir` in period 1.
    """
    model = _make_dissolution_model()
    solution, dissolution_flags = model.solve(
        params=_DISSOLUTION_PARAMS, log_level="off", return_dissolution_flags=True
    )
    np.testing.assert_array_equal(
        np.asarray(dissolution_flags[1]["married_ir"]), [False, True, False]
    )
    np.testing.assert_allclose(
        np.asarray(solution[1]["married_ir"])[1], [-np.inf, -np.inf]
    )


def test_public_model_solve_return_both_returns_three_tuple():
    """`return_simulation_policy=True` and `return_dissolution_flags=True` combine."""
    model = _make_dissolution_model()
    solution, sim_policy, dissolution_flags = model.solve(
        params=_DISSOLUTION_PARAMS,
        log_level="off",
        return_simulation_policy=True,
        return_dissolution_flags=True,
    )
    assert isinstance(solution, MappingProxyType)
    assert isinstance(sim_policy, MappingProxyType)
    np.testing.assert_array_equal(
        np.asarray(dissolution_flags[1]["married_ir"]), [False, True, False]
    )


def test_public_model_solve_default_return_shape_is_byte_identical():
    """Without either flag, `solve` still returns the bare value-function mapping."""
    model = _make_dissolution_model()
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
    assert isinstance(solution, MappingProxyType)
    assert set(solution) == {0, 1, 2, 3}


def test_public_model_simulate_routes_dissolution_edge_when_flags_supplied():
    """End-to-end: a dissolution `GatedEdge` whose gate reads `D_target` simulates.

    Reproduces `test_dissolution_edge_routes_primary_leg_to_own_single_regime`
    through the public `Model` API: wage=2 (D=True) is routed to `single_f`
    (the first declared leg's fallback) instead of `married_ir`; wage=1 and
    wage=3 (D=False) stay married.
    """
    model = _make_dissolution_model()
    solution, dissolution_flags = model.solve(
        params=_DISSOLUTION_PARAMS, log_level="off", return_dissolution_flags=True
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [model.regime_names_to_ids["married"]] * 3, dtype=jnp.int32
            ),
        }
    )
    result = model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        log_level="off",
        seed=0,
    )

    married_ir = result.raw_results["married_ir"][1]
    single_f = result.raw_results["single_f"][1]
    single_m = result.raw_results["single_m"][1]
    np.testing.assert_array_equal(np.asarray(married_ir.in_regime), [True, False, True])
    np.testing.assert_array_equal(np.asarray(single_f.in_regime), [False, True, False])
    np.testing.assert_array_equal(np.asarray(single_m.in_regime), [False, False, False])

    np.testing.assert_allclose(np.asarray(single_f.V_arr)[1], 5.5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(married_ir.V_arr)[0], [2.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(married_ir.V_arr)[2], [6.0, 3.0], rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(single_f.V_arr)[[0, 1, 2]]))
    assert np.all(np.isfinite(np.asarray(married_ir.V_arr)[[0, 2]]))


def test_public_model_simulate_runs_edge_fold_collision_guard_on_precomputed_values(
    monkeypatch,
):
    """simulate-round8 F1 (re-review): the edge-fold state/source-param collision
    guard must run on the SIMULATE entry, not only in `solve()`.

    The public `Model.simulate` accepts a precomputed / cached
    `period_to_regime_to_V_arr` and skips `solve()` entirely, so a guard installed
    only in `solve()` (the round-8 placement) would let the simulate gate and
    fallback-state projector read a colliding leaf unchecked. This asserts the
    guard is invoked on exactly that precomputed-value path — for a gated model,
    even though `solve()` never runs. Correctness of the guard itself (that it
    rejects a genuine collision) is pinned by the round-8 tests in
    `test_gated_edge_arg_provenance.py`; this pins its PLACEMENT.
    """
    model = _make_dissolution_model()
    solution, dissolution_flags = model.solve(
        params=_DISSOLUTION_PARAMS, log_level="off", return_dissolution_flags=True
    )

    calls: list[frozenset[str]] = []
    real = model_module._reject_edge_fold_state_param_collisions

    def _spy(**kwargs):
        calls.append(frozenset(kwargs))
        return real(**kwargs)

    monkeypatch.setattr(model_module, "_reject_edge_fold_state_param_collisions", _spy)

    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [model.regime_names_to_ids["married"]] * 3, dtype=jnp.int32
            ),
        }
    )
    model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,  # precomputed -> solve() is skipped
        period_to_regime_to_dissolution_flags=dissolution_flags,
        log_level="off",
        seed=0,
    )
    assert calls, (
        "the edge-fold collision guard was not invoked on the precomputed-value "
        "simulate path (it would run only inside the skipped solve())"
    )


def test_public_model_simulate_without_dissolution_flags_raises_clearly():
    """Omitting `period_to_regime_to_dissolution_flags` for a dissolution-gated model.

    `period_to_regime_to_dissolution_flags` defaults to `None`; the public
    `Model.simulate()` must surface the SAME clear `NotImplementedError` the
    internal `simulate()` raises (Test 5 above), not a bare `None`-arithmetic
    crash — confirming the default path change is opt-in only.
    """
    model = _make_dissolution_model()
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([2.0]),
            "age": jnp.array([0.0]),
            "regime_id": jnp.array(
                [model.regime_names_to_ids["married"]], dtype=jnp.int32
            ),
        }
    )
    with pytest.raises(NotImplementedError, match="D_target"):
        model.simulate(
            params=_DISSOLUTION_PARAMS,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=solution,
            # period_to_regime_to_dissolution_flags omitted (defaults to None).
            log_level="off",
            seed=0,
        )


# ----------------------------------------------------------------------------------
# Test 7: `to_dataframe()` flattens a collective regime's 2D per-stakeholder value
# into per-stakeholder columns, and leaves a singleton regime's 1D `value` column
# untouched — regression pin for the `pd.DataFrame` "must be 1-dimensional" crash
# fixed in `_lcm.simulation.result_dataframe._process_regime`.
# ----------------------------------------------------------------------------------


def test_to_dataframe_splits_collective_value_into_per_stakeholder_columns():
    """A mixed singleton+collective topology (consent routing) round-trips.

    Reuses the consent fixture (`test_consent_routing_simulate_matches_gate_exactly`):
    period 0 both households are in singleton `single_f` (own `value` column,
    hand-verified V=(2.9, 4.85)); period 1 the wage=1 household routes to
    collective `married_terminal` (own `value_f`/`value_m` columns, V=(2, 1))
    while the wage=2 household routes to singleton `single_f_terminal`
    (`value`=3.0). Each row must be finite in exactly the column(s) its own
    regime populates and NaN in the other regime kind's column(s).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
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
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )

    df = result.to_dataframe()

    assert {"value", "value_f", "value_m"} <= set(df.columns)
    # 2 subjects x 2 periods; single_m_terminal never routed to (0 rows).
    assert len(df) == 4

    period0 = df[df["period"] == 0].sort_values("subject_id")
    np.testing.assert_allclose(
        period0["value"].to_numpy(dtype=float), [2.9, 4.85], rtol=1e-6
    )
    assert period0["value_f"].isna().all()
    assert period0["value_m"].isna().all()

    married_row = df[df["regime_name"] == "married_terminal"]
    assert len(married_row) == 1
    np.testing.assert_allclose(married_row["value_f"].to_numpy(dtype=float), [2.0])
    np.testing.assert_allclose(married_row["value_m"].to_numpy(dtype=float), [1.0])
    assert married_row["value"].isna().all()

    single_term_row = df[df["regime_name"] == "single_f_terminal"]
    assert len(single_term_row) == 1
    np.testing.assert_allclose(single_term_row["value"].to_numpy(dtype=float), [3.0])
    assert single_term_row["value_f"].isna().all()
    assert single_term_row["value_m"].isna().all()


@categorical(ordered=False)
class SoloRegimeId:
    solo: ScalarInt
    solo_terminal: ScalarInt


def _next_solo_regime() -> ScalarInt:
    return SoloRegimeId.solo_terminal


def _u_solo(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work)


def _u_solo_terminal(wage: ContinuousState) -> FloatND:
    return wage


def _make_solo_regimes() -> dict[str, Regime]:
    """Singleton-only two-regime model, structurally identical to the couple
    fixture (`_make_couple_regimes`) minus `stakeholders` and the split
    `utility_f`/`utility_m` pair — isolates the singleton path from the
    collective one for the byte-identical regression check below."""
    solo = Regime(
        transition=_next_solo_regime,
        active=lambda age: age < 1,
        states={"wage": _WAGE_GRID_2},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_solo},
    )
    solo_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_GRID_2},
        functions={"utility": _u_solo_terminal},
    )
    return {"solo": solo, "solo_terminal": solo_terminal}


def test_to_dataframe_singleton_only_value_column_is_unchanged():
    """A singleton-only model's `to_dataframe()` keeps a single 1D `value` column.

    No regime here declares `stakeholders`, so `_process_regime` never enters
    the split branch added for collective regimes — this pins that the
    singleton path is untouched by the fix.
    """
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes_dict = _make_solo_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType(
        {
            "solo": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "solo_terminal": MappingProxyType({}),
        }
    )
    solution, _sim_policies, dissolution_flags = solve(
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
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )

    df = result.to_dataframe()

    assert "value" in df.columns
    assert not any(col.startswith("value_") for col in df.columns)
    assert len(df) == 4  # 2 subjects x 2 periods, no routing/dissolution.
    assert df["value"].notna().all()
    assert pd.api.types.is_float_dtype(df["value"])


# ----------------------------------------------------------------------------------
# Test 8: regression — a REPEATING, self-looping, value-gated edge past the
# source's own activity boundary. Every OTHER gated-edge test above has a
# source active for exactly ONE period (a "one-shot" edge): the target is
# always present at `period + 1`, so `gated_routing.py` never has to handle a
# period whose target regime is not itself solved. `src` here is active over
# TWO periods (ages 0 and 1) and declares a gated edge back to ITSELF — a
# genuinely repeating self-loop. At `src`'s own last active period (age 1,
# the activity boundary), the edge's target (`src` at age 2) does not exist:
# `period_to_regime_to_V_arr[2]` (the sparse per-period solve output) has no
# `"src"` entry. Pre-fix, `build_same_period_mapping_for_fold`'s unconditional
# `period_solution[edge.target]` raised `KeyError: 'src'` there.
# ----------------------------------------------------------------------------------

_REPEAT_GATE_THRESHOLD = 2.0


def _u_src_repeat(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage + 0.0 * work


def _u_src_exit(wage: ContinuousState) -> FloatND:
    return 0.5 * wage


def _u_src_fallback(wage: ContinuousState) -> FloatND:
    return 0.1 * wage


def _prob_stay(age: FloatND) -> FloatND:
    return jnp.where(age < 1.0, 1.0, 0.0)


def _prob_exit_boundary(age: FloatND) -> FloatND:
    return jnp.where(age < 1.0, 0.0, 1.0)


def _repeat_gate(V_target: FloatND) -> BoolND:
    return V_target > _REPEAT_GATE_THRESHOLD


def _make_repeating_self_loop_regimes() -> dict[str, Regime]:
    """A source active over TWO periods with a repeating self-loop `GatedEdge`.

    `src` is active for ages 0 and 1 (periods 0, 1) and declares a gated edge
    back to ITSELF (`gated_edges={"src": ...}`). At period 0 the edge fires
    normally — its target (`src` itself) is active at period 1. At period 1 —
    `src`'s own last active period, the activity boundary — the edge's target
    (`src` at period 2) does not exist: `src` is not active past age 1. The
    ordinary (ungated) regime transition routes a household past the
    boundary into `src_exit` instead (`_prob_stay` / `_prob_exit_boundary`,
    both keyed off age, sum to 1, and structurally declare BOTH `src` and
    `src_exit` as reachable — the gated edge's target must be one of the
    regime's declared transition targets).
    """
    src = Regime(
        transition={
            "src": MarkovTransition(_prob_stay),
            "src_exit": MarkovTransition(_prob_exit_boundary),
        },
        active=lambda age: age < 2,
        states={"wage": _WAGE_2},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src_repeat},
        gated_edges={
            "src": GatedEdge(
                gate=_repeat_gate,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="src_fallback",
                            projection={"wage": _identity_wage},
                        ),
                    )
                },
            )
        },
    )
    src_exit = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_2},
        functions={"utility": _u_src_exit},
    )
    src_fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_2},
        functions={"utility": _u_src_fallback},
    )
    return {"src": src, "src_exit": src_exit, "src_fallback": src_fallback}


def test_repeating_self_loop_gated_edge_simulates_past_activity_boundary():
    """E4 regression: a REPEATING self-loop edge must not `KeyError` at the
    source's own activity boundary (see module-level Test 8 comment).

    Hand computation. Period 1's `src` Bellman weights only the `src_exit`
    continuation (`_prob_stay(age=1)=0`, `_prob_exit_boundary(age=1)=1`):
    `V_1(wage) = wage + beta * 0.5 * wage = 1.475 * wage` (`beta=0.95`). The
    self-loop's gate (`V_target > 2.0`) reads exactly this `V_1`: OPEN at
    wage=2 (`V_1(2)=2.95`), CLOSED at wage=1 (`V_1(1)=1.475`).

    - wage=1: period-0 gate CLOSED -> routed to `src_fallback` for period 1
      (period-1 value `0.1 * 1 = 0.1`); period-0 own value
      `V_0(1) = 1 + beta * 0.1 = 1.095`.
    - wage=2: period-0 gate OPEN -> STAYS in `src` for period 1 — the
      genuine repeat, and exactly the household that raised `KeyError`
      pre-fix (period 1 is `src`'s own activity boundary, and period 2's
      solution has no `src` entry). Post-fix the edge is a no-op at period 1
      (its target is absent); the ordinary transition (100% `src_exit` at
      age=1) routes it to `src_exit` for period 2, with period-1 own value
      `V_1(2) = 2.95` and period-2 terminal value `0.5 * 2 = 1.0`. Period-0
      own value `V_0(2) = 2 + beta * 2.95 = 4.8025`.
    """
    ages = AgeGrid(start=0, stop=3, step="Y")
    regimes_dict = _make_repeating_self_loop_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "src_exit": MappingProxyType({}),
            "src_fallback": MappingProxyType({}),
        }
    )
    # Solve already tolerates a per-period-absent target (the SOLVE-side
    # `_roll_gated_edges` guard predates this fix); this is the control that
    # isolates the bug to the SIMULATE path below.
    solution, _sim_policies, dissolution_flags = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )

    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([regime_names_to_ids["src"]] * 2, dtype=jnp.int32),
        }
    )
    # Pre-fix, this raised `KeyError: 'src'` inside
    # `build_same_period_mapping_for_fold`, called from
    # `substitute_gated_edge_continuations` while simulating the wage=2
    # household's period-1 (boundary) step.
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )

    period_0 = result.raw_results["src"][0]
    np.testing.assert_allclose(np.asarray(period_0.V_arr), [1.095, 4.8025], rtol=1e-6)

    # wage=1 -> gate closed at period 0 -> `src_fallback`; wage=2 -> gate
    # open -> stays in `src` for the genuine repeat.
    fallback_1 = result.raw_results["src_fallback"][1]
    src_1 = result.raw_results["src"][1]
    np.testing.assert_array_equal(np.asarray(fallback_1.in_regime), [True, False])
    np.testing.assert_array_equal(np.asarray(src_1.in_regime), [False, True])
    np.testing.assert_allclose(np.asarray(fallback_1.V_arr)[0], 0.1, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(src_1.V_arr)[1], 2.95, rtol=1e-6)

    # Past the boundary: the wage=2 household's repeat ends at `src`'s own
    # activity boundary; the ordinary (not gated-edge) transition routes it
    # to `src_exit`, exactly as the ordinary per-target probabilities say.
    exit_2 = result.raw_results["src_exit"][2]
    np.testing.assert_array_equal(np.asarray(exit_2.in_regime), [False, True])
    np.testing.assert_allclose(np.asarray(exit_2.V_arr)[1], 1.0, rtol=1e-6)
