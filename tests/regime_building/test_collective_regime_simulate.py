"""Canonical collective policy, formation, dissolution and own-role simulation fixtures.

The joint policy chooses one common action and publishes both member values.
A single forms the household through its own `Exit` under `Consent`; a household
dissolves through its own `Exit`, which retains threshold-only slack and pays each
member's single payoff net of the cost when no stay action is kept.

A fixed population keeps each original subject and its own current role;
settling a household does not create another subject for its other member.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.engine import Regime as EngineRegime
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
)
from _lcm.regime_law import RegimeLawDeclaration
from _lcm.simulation.simulate import simulate
from _lcm.solution.backward_induction import solve
from _lcm.typing import FlatRegimeParams, PeriodToRegimeToVArr, RegimeNamesToIds
from _lcm.utils.logging import get_logger
from lcm import (
    AgeRange,
    ByAge,
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    ExecutionConfig,
    Gate,
    LinSpacedGrid,
    MemberRoute,
    Model,
    ProjectedRegimeValue,
    Regime,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.exits import Exit, ExitTarget, Unilateral
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
    RegimeName,
    ScalarInt,
    UserFunction,
)
from tests.conftest import bind_laws, build_prepared_structure
from tests.simulation.test_runtime_helpers import bind_eager_simulation


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


_BETA = 0.95

type _Solved = tuple[
    AgeGrid,
    MappingProxyType[RegimeName, EngineRegime],
    RegimeNamesToIds,
    MappingProxyType[RegimeName, FlatRegimeParams],
    PeriodToRegimeToVArr,
]


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
) -> tuple[MappingProxyType[RegimeName, EngineRegime], RegimeNamesToIds]:
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
# argmax over two periods, both members tracked (reuses the
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


def test_couple_simulates_recomputed_joint_argmax_two_periods() -> None:
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

    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([8.0, 40.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
            "own_member": jnp.full(
                2, regimes["couple"].member_names_to_ids["f"], dtype=jnp.int32
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


def test_couple_simulate_with_runtime_validation_enabled() -> None:
    """The trailing member axis on `V_arr` broadcasts against `in_regime`.

    The `validate_V` / NaN-diagnostic code paths compare `V_arr` against
    `in_regime`/`subject_ids_in_regime`, which for a collective regime's
    `(n_subjects, n_members)` V needs an explicit broadcast (a singleton
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
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([8.0, 40.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
            "own_member": jnp.full(
                2, regimes["couple"].member_names_to_ids["f"], dtype=jnp.int32
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


def _consent_laws() -> dict[RegimeName, RegimeLawDeclaration | Transition]:
    """Return the law of every regime of `_make_consent_regimes`.

    `None` marks a terminal regime.
    """
    return {
        "single_f": _single_f_law(),
        "single_f_terminal": None,
        "single_m_terminal": None,
        "married_terminal": None,
    }


def _single_f_law() -> ByAge:
    """`single_f` lands in `single_f_terminal`, whose own exit decides the marriage."""
    return ByAge(
        cases={
            AgeRange(exclusive_stop=1): {
                "single_f_terminal": StochasticTransition(func=_prob_one)
            }
        }
    )


def _formation_exit(
    *,
    target_projection: dict[str, UserFunction],
    counterparty_projection: dict[str, UserFunction],
) -> Exit:
    """She marries only if both her and his married values beat their single ones."""
    return Exit(
        targets={
            "self": ExitTarget(
                regime="married_terminal", role="f", projection=target_projection
            )
        },
        rule=Consent(
            counterparty=Counterparty(
                regime="single_m_terminal",
                role="m",
                projection=counterparty_projection,
            )
        ),
    )


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
        exit=_formation_exit(
            target_projection={"wage": _identity_wage},
            counterparty_projection={"wage": _identity_wage},
        ),
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


def _solve_consent() -> _Solved:
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
    return ages, regimes, regime_names_to_ids, flat_params, solution


def test_consent_routing_simulate_matches_gate_exactly() -> None:
    """wage=1 clears mutual consent (marries); wage=2 does not (stays single).

    Consent at wage=1: (V_married_f=2 > V_single_f=1.5) & (V_married_m=1 >
    V_single_m=0.5) -> she moves. At wage=2: (4 > 3) & (2 > 3) -> the husband's
    single value beats marriage -> she stays (both must consent).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution = _solve_consent()
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
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )

    # Period-0 own value matches the hand-computed / solve-verified path
    # (test_gated_edges_collective_solve.py::test_consent_gate_routes_...).
    period_0 = result.raw_results["single_f"][0]
    np.testing.assert_allclose(np.asarray(period_0.V_arr), [2.9, 4.85], rtol=1e-6)

    # wage=1 -> married_terminal (both consent); wage=2 -> stays in
    # single_f_terminal (he refuses).
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


def test_consent_routing_never_populates_the_non_routed_target() -> None:
    """single_m_terminal has no source edge in this topology, so it stays empty."""
    ages, regimes, regime_names_to_ids, flat_params, solution = _solve_consent()
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


_COST_F = "exit__targets__f__utility_adjustment__delta_f"
_COST_M = "exit__targets__m__utility_adjustment__delta_m"


def _dissolution_cost_f(delta_f: FloatND) -> FloatND:
    return -delta_f


def _dissolution_cost_m(delta_m: FloatND) -> FloatND:
    return -delta_m


def _married_dissolution_transition() -> Transition:
    """Return `married`'s move into `married_ir` after age 0.

    Whether the household stays is decided by `married_ir`'s own `Exit`, whose
    member targets are `single_f` / `single_m`.
    """
    return Transition(
        law=ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "married_ir": StochasticTransition(func=_prob_one)
                }
            }
        )
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
        exit=Exit(
            targets={
                "f": ExitTarget(
                    regime="single_f",
                    projection={"wage": _identity_wage},
                    utility_adjustment=_dissolution_cost_f,
                ),
                "m": ExitTarget(
                    regime="single_m",
                    projection={"wage": _identity_wage},
                    utility_adjustment=_dissolution_cost_m,
                ),
            },
            rule=Unilateral(),
        ),
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


def _solve_dissolution() -> _Solved:
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
                    _COST_F: jnp.asarray(0.5),
                    _COST_M: jnp.asarray(0.2),
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
    return ages, regimes, regime_names_to_ids, flat_params, solution


def test_dissolution_edge_routes_the_row_to_its_own_roles_single_regime() -> None:
    """The household's own exit dissolves it at wage=2, where no stay action is kept.

    The cohort declares itself the wives, so the dissolving household's row is
    routed to that role's target ("f" -> `single_f`) instead of `married_ir` — a
    real regime membership change, not merely a value-side fold. wage=1 and
    wage=3 stay married (`married_ir`).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution = _solve_dissolution()
    np.testing.assert_allclose(
        np.asarray(solution[1]["married_ir"])[1], [5.0, 0.8], rtol=1e-6
    )

    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [regime_names_to_ids["married"]] * 3, dtype=jnp.int32
            ),
            "own_member": jnp.full(
                3, regimes["married"].member_names_to_ids["f"], dtype=jnp.int32
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

    # The dissolved household's (wage=2) wife reports her single value net of
    # the exit's cost: V_single_f(wage=2) - delta_f = 5.5 - 0.5 = 5.
    np.testing.assert_allclose(np.asarray(single_f.V_arr)[1], 5.0, rtol=1e-6)
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
    ages, regimes, regime_names_to_ids, flat_params, solution = _solve_dissolution()
    n_subjects = len(wages)
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array(wages),
            "age": jnp.ones(n_subjects),
            "regime_id": jnp.full(
                n_subjects, regime_names_to_ids["married_ir"], dtype=jnp.int32
            ),
            "own_member": jnp.full(
                n_subjects,
                regimes["married_ir"].member_names_to_ids["f"],
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
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )


def test_an_empty_stay_set_publishes_the_exit_payoffs_net_of_costs() -> None:
    """Without a kept stay action each member gets the outside value minus the cost."""
    _, _, _, _, values = _solve_dissolution()
    np.testing.assert_allclose(values[1]["married_ir"][1], [5.0, 0.8], rtol=1e-6)


def test_initial_households_with_no_kept_stay_action_take_their_own_exit() -> None:
    """An initial wife exits the empty wage cell under her current policy."""
    result = _simulate_directly_in_married_ir(wages=[1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        result.raw_results["single_f"][1].in_regime, [False, True, False]
    )
    np.testing.assert_allclose(
        result.raw_results["single_f"][1].V_arr[1], 5.0, rtol=1e-6
    )


def test_value_masked_simulate_reports_the_solved_values_off_the_empty_cell() -> None:
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
# carried as-is (`fixed_transition`) and contributing 0 to every utility, and by
# routing with a state-only gate that is open exactly where the consent
# miniature's spouses both marry, so the expected routing/values are IDENTICAL
# to `test_consent_routing_simulate_matches_gate_exactly`.


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


def _low_wage(wage: ContinuousState) -> BoolND:
    """Open the gate where the consent miniature's spouses both marry: wage=1."""
    return wage < 1.5


def _consent_discrete_axis_laws() -> dict[
    RegimeName, RegimeLawDeclaration | Transition
]:
    """Return the law of every regime of the discrete-target-axis gate miniature.

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
            gates={
                "married_terminal": Gate(
                    predicate=_low_wage,
                    routes={
                        "f": MemberRoute(
                            target_member="f",
                            fallback=ProjectedRegimeValue(
                                regime="single_f_terminal",
                                projection={
                                    "wage": _identity_wage,
                                    "educ": _identity_educ,
                                },
                            ),
                        )
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


def _solve_consent_discrete_axis() -> _Solved:
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
    return ages, regimes, regime_names_to_ids, flat_params, solution


def test_consent_routing_simulate_with_discrete_target_axis_routes_correctly() -> None:
    """Regression pin for the discrete-target-axis gate-interpolation crash.

    Four subjects cross wage in {1, 2} with educ in {low, high}, so the
    `vmap`-ped gate interpolator (`gated_routing._call_vmapped_with_accepted_kwargs`)
    is exercised at BOTH discrete indices, not just one — a discrete-axis
    lookup that silently read the wrong subject's index would flip a
    subject's routing here. Wage=1 opens the gate (marries) regardless of
    educ; wage=2 closes it (stays single) regardless of educ —
    `educ` contributes 0.0 to every utility, so its exact numeric routing and
    values are identical to `test_consent_routing_simulate_matches_gate_exactly`.
    """
    ages, regimes, regime_names_to_ids, flat_params, solution = (
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


def test_public_model_prices_the_households_own_exit() -> None:
    """`married` discounts the kept policy or the exit payoffs net of costs once."""
    result = _make_dissolution_model().solve(
        params=_DISSOLUTION_PARAMS, log_level="off"
    )
    np.testing.assert_allclose(
        result.values[0]["married"],
        _BETA * np.array([[2.0, 1.0], [5.0, 0.8], [6.0, 3.0]]),
        rtol=1e-6,
    )


def test_public_model_solve_returns_labelled_values_for_every_demanded_period() -> None:
    """The public solve result labels exactly the periods holding a demanded node.

    The age-0 root reaches no node in the final period, so it has no key.
    """
    model = _make_dissolution_model()
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="debug")
    assert isinstance(solution, SolutionResult)
    assert isinstance(solution.values, ValueStore)
    assert set(solution.values) == {0, 1, 2}


def test_public_model_simulate_routes_dissolution_edge_when_flags_supplied() -> None:
    """End-to-end: the household's own exit dissolves it after landing.

    Reproduces `test_dissolution_edge_routes_the_row_to_its_own_roles_single_regime`
    through the public `Model` API: wage=2 (no kept stay action) is routed to
    `single_f` (the wife's target, the role this cohort declares) instead of
    `married_ir`; wage=1 and wage=3 (both members stay) stay married.
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
            "own_member": jnp.full(3, model.member_names_to_ids["f"], dtype=jnp.int32),
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

    np.testing.assert_allclose(np.asarray(single_f.V_arr)[1], 5.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(married_ir.V_arr)[0], [2.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(married_ir.V_arr)[2], [6.0, 3.0], rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(single_f.V_arr)[[0, 1, 2]]))
    assert np.all(np.isfinite(np.asarray(married_ir.V_arr)[[0, 2]]))


def test_public_model_replays_the_households_exit_from_values_only() -> None:
    """The household's exit replays from values, without a dissolution artifact."""
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
            "own_member": jnp.full(1, model.member_names_to_ids["f"], dtype=jnp.int32),
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
        result.raw_results["single_f"][1].V_arr, [5.0], rtol=1e-6
    )


# Test 7: `to_dataframe()` flattens a collective regime's 2D per-member value
# into per-member columns, and leaves a singleton regime's 1D `value` column
# as it is — regression pin against the `pd.DataFrame` "must be 1-dimensional"
# crash in `_lcm.simulation.result_dataframe._process_regime`.


def test_to_dataframe_splits_collective_value_into_per_member_columns() -> None:
    """A mixed singleton+collective topology (consent routing) round-trips.

    Reuses the consent fixture (`test_consent_routing_simulate_matches_gate_exactly`):
    period 0 both households are in singleton `single_f` (own `value` column,
    hand-verified V=(2.9, 4.85)); period 1 the wage=1 household routes to
    collective `married_terminal` (own `value_f`/`value_m` columns, V=(2, 1))
    while the wage=2 household routes to singleton `single_f_terminal`
    (`value`=3.0). Each row must be finite in exactly the column(s) its own
    regime populates and NaN in the other regime kind's column(s).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution = _solve_consent()
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
    fixture (`_make_couple_regimes`) minus `members` and the split
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


def test_to_dataframe_singleton_only_value_column_is_unchanged() -> None:
    """A singleton-only model's `to_dataframe()` keeps a single 1D `value` column.

    No regime here declares `members`, so `_process_regime` never enters
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
