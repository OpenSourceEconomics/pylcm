"""Canonical collective policy, arrival, and own-role simulation fixtures.

The joint policy chooses one common action and publishes both member values.
Source-owned arrival decisions run after the scheduled landing and before the
final actor's action. Current participation exits retain threshold-only slack
and pay each member's full single payoff when the kept set is empty.

A fixed population keeps each original subject and its own current role;
settling a household does not create another subject for its other member.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pandas as pd

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
)
from _lcm.regime_law import RegimeLawDeclaration
from _lcm.simulation.simulate import simulate
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import (
    AgeRange,
    ArrivalDecision,
    ArrivalRead,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.exits import Exit, ExitTarget, RuleResult
from lcm.koopmans_aggregation import LinearAggregator
from lcm.result import SimulationResult
from lcm.solver_api import (
    ResultRetention,
    SolutionResult,
    ValueStore,
)
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    RegimeName,
    ScalarInt,
)
from tests.conftest import bind_laws, build_prepared_structure
from tests.simulation.test_runtime_helpers import bind_eager_simulation


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
    *,
    regimes_dict: dict[str, Regime],
    laws: dict[RegimeName, RegimeLawDeclaration | Transition],
    ages: AgeGrid,
    regime_names: list[RegimeName],
):
    """Shared build+solve harness (kernel-level, mirrors the solve-side tests).

    `laws` maps each regime to its law as `Model(edges=...)` would bind it;
    `None` is terminal.
    """
    # Regime codes follow `regimes_dict`, so the laws are bound in its order.
    bound_laws = bind_laws({name: laws[name] for name in regimes_dict})
    regime_names_to_ids = MappingProxyType(
        {name: jnp.int32(i) for i, name in enumerate(regime_names)}
    )
    regimes = process_regimes(
        prepared_structure=build_prepared_structure(
            user_regimes=finalize_regimes(
                user_regimes=regimes_dict,
                derived_categoricals={},
                koopmans_aggregator=LinearAggregator(),
                certainty_equivalent=LinearExpectation(),
                laws=bound_laws,
            ),
            ages=ages,
            laws=bound_laws,
        ),
        user_regimes=finalize_regimes(
            user_regimes=regimes_dict,
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
            laws=bound_laws,
        ),
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=False,
    )
    return regimes, regime_names_to_ids


# Test 1: a solved (non-gated) collective regime simulates the recomputed joint
# argmax over two periods, both stakeholders tracked (reuses the
# test_nonterminal_collective_solve.py couple model).


@categorical(ordered=False)
class CoupleRegimeId:
    couple: ScalarInt
    couple_terminal: ScalarInt


def _utility_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work)


def _utility_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    consumption = wage * work
    return 2.0 * consumption


def _next_wage(work: DiscreteAction) -> ContinuousState:
    return 40.0 * work + 8.0 * (1.0 - work)


def _next_couple_regime() -> ScalarInt:
    return CoupleRegimeId.couple_terminal


_WAGE_GRID_2 = LinSpacedGrid(start=8.0, stop=40.0, n_points=2)


def _couple_laws() -> dict[RegimeName, RegimeLawDeclaration | Transition]:
    """Return the law of every regime of `_make_couple_regimes`.

    `None` marks a terminal regime.
    """
    return {
        "couple": ByAge(
            cases={
                AgeRange(exclusive_stop=1): _SupportedDeterministicTransition(
                    func=_next_couple_regime, targets=("couple_terminal",)
                )
            }
        ),
        "couple_terminal": None,
    }


def _make_couple_regimes() -> dict[str, Regime]:
    couple = Regime(
        states={"wage": _WAGE_GRID_2},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
    )
    couple_terminal = Regime(
        states={"wage": _WAGE_GRID_2},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
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
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    regimes_dict = _make_couple_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict,
        laws=_couple_laws(),
        ages=ages,
        regime_names=list(regimes_dict),
    )
    flat_params = MappingProxyType(
        {
            "couple": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "couple_terminal": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        program_fingerprint="test_collective_regime_simulate",
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    dissolution_flags = _bi_result.dissolution_flags

    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([8.0, 40.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
            "own_stakeholder": jnp.full(
                2, regimes["couple"].stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=bind_eager_simulation(regimes=regimes),
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="debug"),
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

    The `validate_V` / NaN-diagnostic code paths compare `V_arr` against
    `in_regime`/`subject_ids_in_regime`, which for a collective regime's
    `(n_subjects, n_stakeholders)` V needs an explicit broadcast (a singleton
    regime's `V_arr` is already `(n_subjects,)`). Pinned at `log_level="debug"`,
    the strictest level, which raises on any validation failure.
    """
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    regimes_dict = _make_couple_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict,
        laws=_couple_laws(),
        ages=ages,
        regime_names=list(regimes_dict),
    )
    flat_params = MappingProxyType(
        {
            "couple": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "couple_terminal": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        program_fingerprint="test_collective_regime_simulate",
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    dissolution_flags = _bi_result.dissolution_flags
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([8.0, 40.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
            "own_stakeholder": jnp.full(
                2, regimes["couple"].stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=bind_eager_simulation(regimes=regimes),
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


# Test 2: consent routing — a singleton source (single_f) with a gated marriage
# edge; households whose realized wage clears mutual consent marry, others stay
# single. Exact per-seed routing (deterministic wage draws, no stochastic offer).

_WAGE_2 = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)  # {1.0, 2.0}


def _u_single_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage * work


def _u_single_f_terminal(wage: ContinuousState) -> FloatND:
    return 1.5 * wage  # {1.5, 3.0}


def _u_single_m_terminal(wage: ContinuousState) -> FloatND:
    return jnp.where(wage < 1.5, 0.5, 3.0)  # {0.5, 3.0}


def _u_married_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 2.0 * wage + 0.0 * work  # {2, 4}


def _u_married_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage + 0.0 * work  # {1, 2}


def _consent_gate(
    *,
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
) -> BoolND:
    return (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)


def _consent_laws() -> dict[RegimeName, RegimeLawDeclaration | Transition]:
    """Return the law of every regime of `_make_consent_regimes`.

    `None` marks a terminal regime.
    """
    return {
        "single_f": Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "married_terminal": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            arrivals={
                "married_terminal": ArrivalDecision(
                    predicate=_consent_gate,
                    routes={
                        "f": StakeholderRoute(
                            target_stakeholder="f",
                            fallback=ProjectedRegimeValue(
                                regime="single_f_terminal",
                                projection={"wage": _identity_wage},
                            ),
                        )
                    },
                    reads={
                        "V_target_f": ArrivalRead(
                            regime="married_terminal",
                            stakeholder="f",
                            kind="Participating",
                            projection={"wage": _identity_wage},
                        ),
                        "V_target_m": ArrivalRead(
                            regime="married_terminal",
                            stakeholder="m",
                            kind="Participating",
                            projection={"wage": _identity_wage},
                        ),
                        "V_single_f_ref": ArrivalRead(
                            kind="Raw",
                            regime="single_f_terminal",
                            projection={"wage": _identity_wage},
                        ),
                        "V_single_m_ref": ArrivalRead(
                            kind="Raw",
                            regime="single_m_terminal",
                            projection={"wage": _identity_wage},
                        ),
                    },
                )
            },
        ),
        "single_f_terminal": None,
        "single_m_terminal": None,
        "married_terminal": None,
    }


def _make_consent_regimes() -> dict[str, Regime]:
    single_f = Regime(
        states={"wage": _WAGE_2},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f},
    )
    single_f_terminal = Regime(
        states={"wage": _WAGE_2},
        functions={"utility": _u_single_f_terminal},
    )
    single_m_terminal = Regime(
        states={"wage": _WAGE_2},
        functions={"utility": _u_single_m_terminal},
    )
    married_terminal = Regime(
        states={"wage": _WAGE_2},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_married_f, "m": _u_married_m}
            )
        },
    )
    return {
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m_terminal": single_m_terminal,
        "married_terminal": married_terminal,
    }


def _solve_consent():
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    regimes_dict = _make_consent_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict,
        laws=_consent_laws(),
        ages=ages,
        regime_names=list(regimes_dict),
    )
    flat_params = MappingProxyType(
        {
            "single_f": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "single_f_terminal": MappingProxyType({}),
            "single_m_terminal": MappingProxyType({}),
            "married_terminal": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        program_fingerprint="test_collective_regime_simulate",
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    dissolution_flags = _bi_result.dissolution_flags
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
        regimes=bind_eager_simulation(regimes=regimes),
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="debug"),
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
        regimes=bind_eager_simulation(regimes=regimes),
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="debug"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )
    single_m_term = result.raw_results["single_m_terminal"][1]
    np.testing.assert_array_equal(np.asarray(single_m_term.in_regime), [False, False])


# Test 3: dissolution routing — a married cohort where the IR mask empties
# at wage=2; reuses the dissolution-edge miniature from
# test_gated_edges_collective_solve.py::_make_dissolution_regimes.

_WAGE_3 = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)  # {1, 2, 3}


def _u_married_ir_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 3.0 * (1.0 - work) + 2.0 * wage * work


def _u_married_ir_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.5 * (1.0 - work) + wage * work


def _u_single_f_ir(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    target = jnp.where((wage > 1.5) & (wage < 2.5), 5.5, 1.5)
    return target * work


def _u_single_m_ir(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 1.0 * work + 0.0 * wage


def _u_zero(wage: ContinuousState) -> FloatND:
    return 0.0 * wage


def _u_zero_collective(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.0 * wage * work


def _ir_f(*, Q_f: FloatND, V_single_f_ref: FloatND, delta_f: FloatND) -> BoolND:
    return Q_f >= V_single_f_ref - delta_f


def _ir_m(*, Q_m: FloatND, V_single_m_ref: FloatND, delta_m: FloatND) -> BoolND:
    return Q_m >= V_single_m_ref - delta_m


class _ParticipationRule:
    """One common action meets both thresholds; dissolution pays full outside value."""

    reads_outside_values: ClassVar[bool] = True

    def __call__(
        self,
        *,
        Q_f: FloatND,
        Q_m: FloatND,
        exit_payoffs: Mapping[str, Mapping[str, FloatND]],
        alternative_ids: IntND,
        pareto_weights: Mapping[str, FloatND],
        delta_f: FloatND,
        delta_m: FloatND,
    ) -> RuleResult:
        q_f, q_m = Q_f.reshape(-1), Q_m.reshape(-1)
        outside = exit_payoffs["dissolve"]
        kept = (
            jnp.isfinite(q_f)
            & jnp.isfinite(q_m)
            & _ir_f(Q_f=q_f, V_single_f_ref=outside["f"], delta_f=delta_f)
            & _ir_m(Q_m=q_m, V_single_m_ref=outside["m"], delta_m=delta_m)
        )
        objective = pareto_weights["f"] * jnp.where(
            pareto_weights["f"] == 0, 0.0, q_f
        ) + pareto_weights["m"] * jnp.where(pareto_weights["m"] == 0, 0.0, q_m)
        stay = jnp.argmax(jnp.where(kept, objective, -jnp.inf))
        return RuleResult(
            action=alternative_ids[jnp.where(jnp.any(kept), stay, q_f.size)]
        )


def _participation_support(*, target_support: BoolND) -> BoolND:
    return target_support


def _married_dissolution_transition() -> Transition:
    """Return the source-owned arrival into `married_ir` after age 0.

    The edge's stakeholder routes fall back to `single_f` / `single_m`, so
    `married` reaches all three at age 0.
    """
    return Transition(
        law=ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "married_ir": StochasticTransition(func=_prob_one)
                }
            }
        ),
        arrivals={
            "married_ir": ArrivalDecision(
                predicate=_participation_support,
                reads={
                    "target_support": ArrivalRead(
                        regime="married_ir",
                        stakeholder="f",
                        kind="Participating",
                        field="support",
                        projection={"wage": _identity_wage},
                    )
                },
                routes={
                    "f": StakeholderRoute(
                        target_stakeholder="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f",
                            projection={"wage": _identity_wage},
                        ),
                    ),
                    "m": StakeholderRoute(
                        target_stakeholder="m",
                        fallback=ProjectedRegimeValue(
                            regime="single_m",
                            projection={"wage": _identity_wage},
                        ),
                    ),
                },
            )
        },
    )


def _dissolution_laws() -> dict[RegimeName, RegimeLawDeclaration | Transition]:
    """Return the law of every regime of `_make_dissolution_regimes`.

    `None` marks a terminal regime. Each nonterminal law but `married`'s picks
    the single outgoing edge of its source age.
    """
    return {
        "married": _married_dissolution_transition(),
        "married_ir": ByAge(
            cases={
                AgeRange(start=1, exclusive_stop=2): {
                    "married_terminal": StochasticTransition(func=_prob_one)
                }
            }
        ),
        "married_terminal": None,
        "single_f": ByAge(
            cases={
                AgeRange(start=1, exclusive_stop=2): {
                    "single_f_terminal": StochasticTransition(func=_prob_one)
                }
            }
        ),
        "single_f_terminal": None,
        "single_m": {"single_m_terminal": StochasticTransition(func=_prob_one)},
        "single_m_terminal": None,
    }


def _make_dissolution_regimes() -> dict[str, Regime]:
    """Return the dissolution miniature's regimes, without their laws.

    A model declares them as `_make_dissolution_model` does; the kernel-level
    harness binds `_dissolution_laws` instead.
    """
    married = Regime(
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    married_ir = Regime(
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_married_ir_f, "m": _u_married_ir_m}
            )
        },
        exits={
            "dissolve": Exit(
                targets={
                    "f": ExitTarget(
                        regime="single_f", projection={"wage": _identity_wage}
                    ),
                    "m": ExitTarget(
                        regime="single_m", projection={"wage": _identity_wage}
                    ),
                },
                rule=_ParticipationRule(),
            )
        },
    )
    married_terminal = Regime(
        states={"wage": _WAGE_3},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    single_f = Regime(
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f_ir},
    )
    single_f_terminal = Regime(
        states={"wage": _WAGE_3},
        functions={"utility": _u_zero},
    )
    single_m = single_f.replace(functions={"utility": _u_single_m_ir})
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
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    regimes_dict = _make_dissolution_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict,
        laws=_dissolution_laws(),
        ages=ages,
        regime_names=list(regimes_dict),
    )
    flat_params = MappingProxyType(
        {
            "married": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "married_ir": MappingProxyType(
                {
                    "koopmans_aggregator__discount_factor": jnp.asarray(_BETA),
                    "exits__dissolve__rule__delta_f": jnp.asarray(0.5),
                    "exits__dissolve__rule__delta_m": jnp.asarray(0.2),
                }
            ),
            "married_terminal": MappingProxyType({}),
            "single_f": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "single_f_terminal": MappingProxyType({}),
            "single_m": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "single_m_terminal": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        program_fingerprint="test_collective_regime_simulate",
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    dissolution_flags = _bi_result.dissolution_flags
    return ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags


def test_dissolution_edge_routes_the_row_to_its_own_roles_single_regime():
    """D=True at wage=2 (the IR mask empties there, see solve test).

    The cohort declares itself the wives, so the dissolving household's row is
    routed to that role's leg ("f" -> `single_f`) instead of `married_ir` — a
    real regime membership change, not merely a value-side fold. wage=1 and
    wage=3 stay married (`married_ir`, D=False there).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_dissolution()
    )
    np.testing.assert_allclose(
        np.asarray(solution[1]["married_ir"])[1], [5.5, 1.0], rtol=1e-6
    )

    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [regime_names_to_ids["married"]] * 3, dtype=jnp.int32
            ),
            "own_stakeholder": jnp.full(
                3, regimes["married"].stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=bind_eager_simulation(regimes=regimes),
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="debug"),
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
    # single_m never becomes this row's OWN continuing membership — this cohort
    # tracks the wives, and the husband's leg is his own row's; see the module
    # docstring's scope fence.
    np.testing.assert_array_equal(np.asarray(single_m.in_regime), [False, False, False])

    # The dissolved household's (wage=2) value under its routed regime matches
    # the IR miniature's single fallback exactly (hand-computed in
    # test_gated_edges_collective_solve.py): V_single_f(wage=2) = 5.5.
    np.testing.assert_allclose(np.asarray(single_f.V_arr)[1], 5.5, rtol=1e-6)
    # The still-married households keep their married_ir value.
    np.testing.assert_allclose(np.asarray(married_ir.V_arr)[0], [2.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(married_ir.V_arr)[2], [6.0, 3.0], rtol=1e-6)


# Test 4: value-masked simulate — the simulated argmax never selects an action
# excluded by a value constraint (reuses the dissolution miniature's IR
# mask: married_ir's mask is empty at wage=2).


def _simulate_directly_in_married_ir(*, wages: list[float]) -> SimulationResult:
    """Simulate households placed directly IN married_ir at period 1.

    Initializing subjects directly into `married_ir` bypasses the `married`
    source, so the value router's routing does not preempt the masked cell.
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_dissolution()
    )
    n_subjects = len(wages)
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array(wages),
            "age": jnp.ones(n_subjects),
            "regime_id": jnp.full(
                n_subjects, regime_names_to_ids["married_ir"], dtype=jnp.int32
            ),
            "own_stakeholder": jnp.full(
                n_subjects,
                regimes["married_ir"].stakeholder_names_to_ids["f"],
                dtype=jnp.int32,
            ),
        }
    )
    return simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=bind_eager_simulation(regimes=regimes),
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="debug"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )


def test_empty_participation_publishes_the_original_exit_payoffs() -> None:
    """Full pays the original outside tuple where no common stay action participates."""
    _, _, _, _, values, _ = _solve_dissolution()
    np.testing.assert_allclose(values[1]["married_ir"][1], [5.5, 1.0], rtol=1e-6)


def test_initial_households_with_empty_participation_take_their_own_exit() -> None:
    """An initial wife exits the empty wage cell under her current policy."""
    result = _simulate_directly_in_married_ir(wages=[1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        result.raw_results["single_f"][1].in_regime, [False, True, False]
    )
    np.testing.assert_allclose(
        result.raw_results["single_f"][1].V_arr[1], 5.5, rtol=1e-6
    )


def test_value_masked_simulate_reports_the_solved_values_off_the_empty_cell():
    """Households on the cells the mask leaves feasible get the solved values."""
    married_ir = _simulate_directly_in_married_ir(wages=[1.0, 3.0]).raw_results[
        "married_ir"
    ][1]

    np.testing.assert_allclose(np.asarray(married_ir.V_arr), [[2.0, 1.0], [6.0, 3.0]])


# Test 6: regression — a gated edge's TARGET has a discrete state axis (in
# addition to a continuous one). `route_gated_edges` interpolates the fold's
# `gate` array at each subject's candidate target-state draw
# (`_lcm.simulation.gated_routing._call_vmapped_with_accepted_kwargs`), one
# subject at a time under `vmap`. Interpolating for the whole population at once
# instead would break `jax.scipy.ndimage.map_coordinates`'s
# `len(coordinates) == input.ndim` invariant on a discrete target axis (advanced
# indexing on whole-population `(n_subjects,)` discrete indices collapses the
# discrete axes into a leading batch dimension while leaving the continuous axis
# trailing) — every OTHER gated-edge simulate test above uses a continuous-only
# target, where a `(n_subjects,)` coordinate array happens to broadcast correctly
# through `map_coordinates` by coincidence, so this is the fixture that covers
# the discrete case. Adapted from
# `_make_consent_regimes` (Test 2 above) by adding a discrete `educ` state,
# carried as-is (`fixed_transition`) and contributing 0 to every utility, so
# the gate arithmetic and expected routing/values are IDENTICAL to
# `test_consent_routing_simulate_matches_gate_exactly`.


@categorical(ordered=True)
class Educ:
    low: ScalarInt  # code 0
    high: ScalarInt  # code 1


def _identity_educ(educ: DiscreteState) -> DiscreteState:
    return educ


def _u_single_f_educ(
    *, wage: ContinuousState, work: DiscreteAction, educ: DiscreteState
) -> FloatND:
    return wage * work + 0.0 * educ


def _u_single_f_terminal_educ(*, wage: ContinuousState, educ: DiscreteState) -> FloatND:
    return 1.5 * wage + 0.0 * educ  # {1.5, 3.0}


def _u_married_f_educ(
    *, wage: ContinuousState, work: DiscreteAction, educ: DiscreteState
) -> FloatND:
    return 2.0 * wage + 0.0 * work + 0.0 * educ  # {2, 4}


def _u_married_m_educ(
    *, wage: ContinuousState, work: DiscreteAction, educ: DiscreteState
) -> FloatND:
    return wage + 0.0 * work + 0.0 * educ  # {1, 2}


def _consent_discrete_axis_laws() -> dict[
    RegimeName, RegimeLawDeclaration | Transition
]:
    """Return the law of every regime of the discrete-target-axis consent miniature.

    Its regimes come from `_make_consent_regimes_with_discrete_target_axis`; `None`
    marks a terminal regime.
    """
    return {
        "single_f": Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "married_terminal": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            arrivals={
                "married_terminal": ArrivalDecision(
                    predicate=_consent_gate,
                    routes={
                        "f": StakeholderRoute(
                            target_stakeholder="f",
                            fallback=ProjectedRegimeValue(
                                regime="single_f_terminal",
                                projection={
                                    "wage": _identity_wage,
                                    "educ": _identity_educ,
                                },
                            ),
                        )
                    },
                    reads={
                        "V_target_f": ArrivalRead(
                            regime="married_terminal",
                            stakeholder="f",
                            kind="Participating",
                            projection={"wage": _identity_wage, "educ": _identity_educ},
                        ),
                        "V_target_m": ArrivalRead(
                            regime="married_terminal",
                            stakeholder="m",
                            kind="Participating",
                            projection={"wage": _identity_wage, "educ": _identity_educ},
                        ),
                        "V_single_f_ref": ArrivalRead(
                            kind="Raw",
                            regime="single_f_terminal",
                            projection={
                                "wage": _identity_wage,
                                "educ": _identity_educ,
                            },
                        ),
                        "V_single_m_ref": ArrivalRead(
                            kind="Raw",
                            regime="single_m_terminal",
                            projection={"wage": _identity_wage},
                        ),
                    },
                )
            },
        ),
        "single_f_terminal": None,
        "single_m_terminal": None,
        "married_terminal": None,
    }


def _make_consent_regimes_with_discrete_target_axis() -> dict[str, Regime]:
    single_f = Regime(
        # ONLY difference vs. `_make_consent_regimes`: an added discrete
        # "educ" state, carried as-is (`fixed_transition`), on the source and
        # every regime the gated edge's fold/gate/fallback touch.
        states={"wage": _WAGE_2, "educ": DiscreteGrid(category_class=Educ)},
        state_transitions={
            "wage": fixed_transition("wage"),
            "educ": fixed_transition("educ"),
        },
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f_educ},
    )
    single_f_terminal = Regime(
        states={"wage": _WAGE_2, "educ": DiscreteGrid(category_class=Educ)},
        functions={"utility": _u_single_f_terminal_educ},
    )
    single_m_terminal = Regime(
        states={"wage": _WAGE_2},
        functions={"utility": _u_single_m_terminal},
    )
    married_terminal = Regime(
        states={"wage": _WAGE_2, "educ": DiscreteGrid(category_class=Educ)},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_married_f_educ, "m": _u_married_m_educ}
            )
        },
    )
    return {
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m_terminal": single_m_terminal,
        "married_terminal": married_terminal,
    }


def _solve_consent_discrete_axis():
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    regimes_dict = _make_consent_regimes_with_discrete_target_axis()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict,
        laws=_consent_discrete_axis_laws(),
        ages=ages,
        regime_names=list(regimes_dict),
    )
    flat_params = MappingProxyType(
        {
            "single_f": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "single_f_terminal": MappingProxyType({}),
            "single_m_terminal": MappingProxyType({}),
            "married_terminal": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        program_fingerprint="test_collective_regime_simulate",
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    dissolution_flags = _bi_result.dissolution_flags
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

    # An un-vmapped gate interpolator raises here: `ValueError: coordinates
    # must be a sequence of length input.ndim, but 1 != 2`.
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=bind_eager_simulation(regimes=regimes),
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="debug"),
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


# Test 7: public `Model.solve`/`Model.simulate` API — the flags Test 5 pins at
# the INTERNAL `simulate()` level travel through the addressed public result too.
# The default `Model.solve()` result carries the dissolution artifact computed by
# `backward_induction.solve`, and `Model.simulate(solution=...)` threads it down to
# the same dissolution routing exercised by Test 3 (`_make_dissolution_regimes`).


@categorical(ordered=False)
class DissolutionRegimeId:
    married: ScalarInt
    married_ir: ScalarInt
    married_terminal: ScalarInt
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m: ScalarInt
    single_m_terminal: ScalarInt


def _make_dissolution_model(
    execution_config: ExecutionConfig = ExecutionConfig(),  # noqa: B008
) -> Model:
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    return Model(
        regimes=_make_dissolution_regimes(),
        ages=ages,
        regime_id_class=DissolutionRegimeId,
        initial_nodes={ages.exact_values[0]: "married"},
        edges={
            "married": _married_dissolution_transition(),
            "married_ir": {"married_terminal": 1},
            "single_f": {"single_f_terminal": 1},
            "single_m": {"single_m_terminal": (0, 1)},
        },
        execution_config=execution_config,
    )


_DISSOLUTION_PARAMS = {
    "discount_factor": _BETA,
    "delta_f": 0.5,
    "delta_m": 0.2,
}


def test_public_model_prices_the_actual_participation_routes() -> None:
    """Arrival continuation discounts the kept policy or full outside payoff once."""
    result = _make_dissolution_model().solve(
        params=_DISSOLUTION_PARAMS, log_level="off"
    )
    np.testing.assert_allclose(
        result.values[0]["married"],
        _BETA * np.array([[2.0, 1.0], [5.5, 1.0], [6.0, 3.0]]),
        rtol=1e-6,
    )


def test_public_model_solve_returns_labelled_values_for_every_demanded_period():
    """The public solve result labels exactly the periods holding a demanded node.

    The age-0 root reaches no node in the final period, so it has no key.
    """
    model = _make_dissolution_model()
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="debug")
    assert isinstance(solution, SolutionResult)
    assert isinstance(solution.values, ValueStore)
    assert set(solution.values) == {0, 1, 2}


def test_public_model_simulate_routes_dissolution_edge_when_flags_supplied():
    """End-to-end: a dissolution `GatedEdge` whose gate reads `D_target` simulates.

    Reproduces `test_dissolution_edge_routes_the_row_to_its_own_roles_single_regime`
    through the public `Model` API: wage=2 (D=True) is routed to `single_f`
    (the wife's leg, the role this cohort declares) instead of `married_ir`;
    wage=1 and wage=3 (D=False) stay married.
    """
    model = _make_dissolution_model()
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="debug")
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [model.regime_names_to_ids["married"]] * 3, dtype=jnp.int32
            ),
            "own_stakeholder": jnp.full(
                3, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        }
    )
    result = model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="debug",
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


def test_public_model_replays_participation_from_values_only() -> None:
    """Current participation replay needs values, without a dissolution artifact."""
    model = _make_dissolution_model()
    solution = model.solve(
        params=_DISSOLUTION_PARAMS,
        log_level="debug",
        retention=ResultRetention.VALUES,
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([2.0]),
            "age": jnp.array([0.0]),
            "regime_id": jnp.array(
                [model.regime_names_to_ids["married"]], dtype=jnp.int32
            ),
            "own_stakeholder": jnp.full(
                1, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        }
    )
    result = model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="debug",
        seed=0,
    )
    np.testing.assert_array_equal(result.raw_results["single_f"][1].in_regime, [True])
    np.testing.assert_allclose(
        result.raw_results["single_f"][1].V_arr, [5.5], rtol=1e-6
    )


# Test 7: `to_dataframe()` flattens a collective regime's 2D per-stakeholder value
# into per-stakeholder columns, and leaves a singleton regime's 1D `value` column
# as it is — regression pin against the `pd.DataFrame` "must be 1-dimensional"
# crash in `_lcm.simulation.result_dataframe._process_regime`.


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
        regimes=bind_eager_simulation(regimes=regimes),
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="debug"),
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


def _u_solo(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work)


def _u_solo_terminal(wage: ContinuousState) -> FloatND:
    return wage


def _solo_laws() -> dict[RegimeName, RegimeLawDeclaration | Transition]:
    """Return the law of every regime of `_make_solo_regimes`.

    `None` marks a terminal regime.
    """
    return {
        "solo": ByAge(
            cases={
                AgeRange(exclusive_stop=1): _SupportedDeterministicTransition(
                    func=_next_solo_regime, targets=("solo_terminal",)
                )
            }
        ),
        "solo_terminal": None,
    }


def _make_solo_regimes() -> dict[str, Regime]:
    """Singleton-only two-regime model, structurally identical to the couple
    fixture (`_make_couple_regimes`) minus `stakeholders` and the split
    `utility_f`/`utility_m` pair — isolates the singleton path from the
    collective one for the byte-identical regression check below."""
    solo = Regime(
        states={"wage": _WAGE_GRID_2},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_solo},
    )
    solo_terminal = Regime(
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
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    regimes_dict = _make_solo_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict,
        laws=_solo_laws(),
        ages=ages,
        regime_names=list(regimes_dict),
    )
    flat_params = MappingProxyType(
        {
            "solo": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "solo_terminal": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        program_fingerprint="test_collective_regime_simulate",
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    dissolution_flags = _bi_result.dissolution_flags
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
        regimes=bind_eager_simulation(regimes=regimes),
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="debug"),
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


# Test 8: regression — a REPEATING, self-looping, value-gated edge past the
# source's own activity boundary. Every OTHER gated-edge test above has a
# source active for exactly ONE period (a "one-shot" edge): the target is
# always present at `period + 1`, so `gated_routing.py` never has to handle a
# period whose target regime is not itself solved. `src` here is active over
# TWO periods (ages 0 and 1) and declares a gated edge back to ITSELF — a
# genuinely repeating self-loop. At `src`'s own last active period (age 1,
# the activity boundary), the edge's target (`src` at age 2) does not exist:
# `period_to_regime_to_V_arr[2]` (the sparse per-period solve output) has no
# `"src"` entry, so an unconditional `period_solution[edge.target]` in
# `build_same_period_mapping_for_fold` raises `KeyError: 'src'` there.

_REPEAT_GATE_THRESHOLD = 2.0


def _u_src_repeat(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
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


def _repeating_self_loop_laws() -> dict[RegimeName, RegimeLawDeclaration | Transition]:
    """Return the law of every regime of `_make_repeating_self_loop_regimes`.

    `None` marks a terminal regime.
    """
    exit_cell = StochasticTransition(func=_prob_exit_boundary)
    return {
        "src": Transition(
            law=ByAge.until(
                stop_age_exclusive=2,
                law={
                    "src": StochasticTransition(func=_prob_stay),
                    "src_exit": exit_cell,
                },
                then={"src_exit": exit_cell},
            ),
            arrivals={
                "src": ArrivalDecision(
                    predicate=_repeat_gate,
                    reads={
                        "V_target": ArrivalRead(
                            regime="src",
                            kind="IntegratedFull",
                            projection={"wage": _identity_wage},
                        )
                    },
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="src_fallback",
                                projection={"wage": _identity_wage},
                            ),
                        )
                    },
                )
            },
        ),
        "src_exit": None,
        "src_fallback": None,
    }


def _make_repeating_self_loop_regimes() -> dict[str, Regime]:
    """A source active over TWO periods with a repeating self-loop `GatedEdge`.

    `src` is active for ages 0 and 1 (periods 0, 1) and declares a gated edge
    back to ITSELF (`gated_edges={"src": ...}`). At period 0 the edge fires
    normally — its target (`src` itself) is active at period 1. At period 1 —
    `src`'s own last active period, the activity boundary — the edge's target
    (`src` at period 2) does not exist: `src` is not solved past age 1. The
    ordinary (ungated) regime transition routes a household past the
    boundary into `src_exit` instead (`_prob_stay` / `_prob_exit_boundary`,
    both keyed off age, sum to 1, and structurally declare BOTH `src` and
    `src_exit` as reachable — the gated edge's target must be one of the
    regime's declared transition targets).
    """
    src = Regime(
        states={"wage": _WAGE_2},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_src_repeat},
    )
    src_exit = Regime(
        states={"wage": _WAGE_2},
        functions={"utility": _u_src_exit},
    )
    src_fallback = Regime(
        states={"wage": _WAGE_2},
        functions={"utility": _u_src_fallback},
    )
    return {"src": src, "src_exit": src_exit, "src_fallback": src_fallback}
