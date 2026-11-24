"""Integration tests for moves between singleton and collective regimes.

Two moves matter for EKL 2019: the singles->married CONSENT decision (a singleton
reaches a collective regime only by mutual consent, eq. 27), declared as the
single's own `Exit` under a `Consent` rule, and the married DISSOLUTION, the
household's own current-period `Exit` routing each member to her single regime.
A state-only `Gate` beside the law of a `Transition` routes a landing row into its
target or into that member's fallback::

    Wbar^s(x) = jnp.where(gate(x), V_target^{route_s}(x), V_fallback^s(pi_s(x)))
"""

from collections.abc import Mapping
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.gated_edge import GatedEdge
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import (
    _fail_if_gated_edge_references_inactive,
    process_regimes,
)
from _lcm.regime_law import RegimeLaws
from _lcm.solution.backward_induction import solve
from _lcm.typing import PeriodToRegimeToSimulationPolicy, PeriodToRegimeToVArr
from _lcm.user_regime_validation import validate_regime_law
from _lcm.utils.logging import get_logger
from lcm import (
    AgeRange,
    ByAge,
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
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
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.exits import Exit, ExitTarget, Unilateral
from lcm.koopmans_aggregation import LinearAggregator
from lcm.transition import AgeSelector, StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    RegimeName,
    ScalarInt,
)
from tests.conftest import bind_laws, build_prepared_structure
from tests.regime_building.test_collective_regime_simulate import (
    _dissolution_cost_f,
    _dissolution_cost_m,
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


# Consent edge: single_f (singleton) -> married_terminal (collective)


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


def _consent_source() -> Regime:
    """The consent source, which may move into the couple at age 0."""
    return Regime(
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f},
    )


def _consent_laws(edge: Gate | None) -> RegimeLaws:
    """The consent regimes' laws, the source's move into the couple under `edge`.

    `None` declares the move into the couple without a gate.
    """
    return bind_laws(
        {
            "single_f": Transition(
                law=ByAge(
                    cases={
                        AgeRange(exclusive_stop=1): {
                            "married_terminal": StochasticTransition(func=_prob_one)
                        }
                    }
                ),
                gates=({"married_terminal": edge} if edge is not None else {}),
            ),
            "single_f_terminal": None,
            "single_m_terminal": None,
            "married_terminal": None,
        }
    )


def _formation_laws() -> RegimeLaws:
    """`single_f` lands in `single_f_terminal`, whose own exit decides the marriage."""
    return bind_laws(
        {
            "single_f": ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "single_f_terminal": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            "single_f_terminal": None,
            "single_m_terminal": None,
            "married_terminal": None,
        }
    )


def _formation_exit() -> Exit:
    """EKL eq. 27: she marries only if both partners strictly gain.

    Her couple value is compared with her own single value, his with his single
    value as the counterparty.
    """
    return Exit(
        targets={
            "self": ExitTarget(
                regime="married_terminal",
                role="f",
                projection={"wage": _identity_wage},
            )
        },
        rule=Consent(
            counterparty=Counterparty(
                regime="single_m_terminal",
                role="m",
                projection={"wage": _identity_wage},
            )
        ),
    )


def _open_where_both_marry(wage: ContinuousState) -> BoolND:
    """The consent decision's outcome stated on the landing wage: open at wage 1."""
    return wage < 1.5


def _gate_edge(
    *, fallback_regime: RegimeName = "single_f_terminal", target_member: str = "f"
) -> Gate:
    """A gate into the couple, open where consent holds, routing the wife."""
    return Gate(
        predicate=_open_where_both_marry,
        routes={
            "f": MemberRoute(
                target_member=target_member,
                fallback=ProjectedRegimeValue(
                    regime=fallback_regime,
                    projection={"wage": _identity_wage},
                ),
            )
        },
    )


def _make_consent_regimes() -> dict[str, Regime]:
    single_f = _consent_source()
    single_f_terminal = Regime(
        states={"wage": _WAGE},
        functions={"utility": _u_single_f_terminal},
        exit=_formation_exit(),
    )
    single_m_terminal = Regime(
        states={"wage": _WAGE},
        functions={"utility": _u_single_m_terminal},
    )
    married_terminal = Regime(
        states={"wage": _WAGE},
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


def _solve_consent(
    *, enable_jit: bool = False
) -> tuple[PeriodToRegimeToVArr, PeriodToRegimeToSimulationPolicy]:
    laws = _formation_laws()
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    regimes = process_regimes(
        prepared_structure=build_prepared_structure(
            user_regimes=finalize_regimes(
                user_regimes=_make_consent_regimes(),
                derived_categoricals={},
                koopmans_aggregator=LinearAggregator(),
                certainty_equivalent=LinearExpectation(),
                laws=laws,
            ),
            ages=ages,
            laws=laws,
        ),
        user_regimes=finalize_regimes(
            user_regimes=_make_consent_regimes(),
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
            laws=laws,
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
            "single_f": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "single_f_terminal": MappingProxyType({}),
            "single_m_terminal": MappingProxyType({}),
            "married_terminal": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        program_fingerprint="test_gated_edges_collective_solve",
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=enable_jit,
    )
    return _bi_result.value_functions, _bi_result.simulation_policies


def test_consent_routes_marriage_value_and_single_value() -> None:
    """UNANIMITY: both gain -> married value flows; one partner short -> she stays.

    wage=1: (V_married_f=2 > V_single_f=1.5) & (V_married_m=1 > V_single_m=0.5)
        -> she moves. Her period-1 value is V_married_f = 2.
    wage=2: (4 > 3) & (2 > 3) -> husband's single value beats marriage -> she stays
        (unanimity). Her period-1 value is her single value = 3.

    Wbar = [2, 3]; V_single_f(period 0) = wage + beta*Wbar(wage):
        wage=1: 1 + 0.95*2 = 2.9;  wage=2: 2 + 0.95*3 = 4.85.
    """
    solution, _sim = _solve_consent()
    V_single_f = np.asarray(solution[0]["single_f"])
    np.testing.assert_allclose(V_single_f, np.array([2.9, 4.85]), rtol=1e-6)


def test_consent_matches_under_jit() -> None:
    """The jitted path prices the same exit."""
    solution, _sim = _solve_consent(enable_jit=True)
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"]), np.array([2.9, 4.85]), rtol=1e-6
    )


# The value-constraint IR miniature's regime payoffs, reused by the full topology.

_WAGE3 = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)  # {1, 2, 3}


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


# Scope fences and build-time validation (pins)


def test_raw_ungated_mixed_transition_still_rejected() -> None:
    """A singleton regime reaching a collective target WITHOUT an edge is rejected."""
    regimes = _make_consent_regimes()
    # Drop the gate but keep the singleton -> collective transition.
    laws = _consent_laws(None)
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    with pytest.raises(NotImplementedError, match="member_axis"):
        process_regimes(
            prepared_structure=build_prepared_structure(
                user_regimes=finalize_regimes(
                    user_regimes=regimes,
                    derived_categoricals={},
                    koopmans_aggregator=LinearAggregator(),
                    certainty_equivalent=LinearExpectation(),
                    laws=laws,
                ),
                ages=ages,
                laws=laws,
            ),
            user_regimes=finalize_regimes(
                user_regimes=regimes,
                derived_categoricals={},
                koopmans_aggregator=LinearAggregator(),
                certainty_equivalent=LinearExpectation(),
                laws=laws,
            ),
            ages=ages,
            regime_names_to_ids=MappingProxyType(
                {n: jnp.int32(i) for i, n in enumerate(regimes)}
            ),
            enable_jit=False,
        )


def test_probabilistic_gate_is_rejected() -> None:
    """A stochastic (StochasticTransition) gate is out of scope — boolean only."""
    with pytest.raises(RegimeInitializationError, match="boolean"):
        validate_regime_law(
            Regime(
                states={"wage": _WAGE},
                state_transitions={"wage": fixed_transition("wage")},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={"utility": _u_single_f},
            ),
            law=_consent_laws(
                Gate(
                    predicate=StochasticTransition(func=_prob_one),
                    routes={
                        "f": MemberRoute(
                            target_member="f",
                            fallback=ProjectedRegimeValue(
                                regime="single_f_terminal",
                                projection={"wage": _identity_wage},
                            ),
                        )
                    },
                )
            )["single_f"],
        )


@categorical(ordered=False)
class ConsentRegimeId:
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt
    married_terminal: ScalarInt


def _consent_model(edge: Gate) -> Model:
    """The consent regimes as a public model, the move into the couple under `edge`."""
    return Model(
        regimes=_make_consent_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=ConsentRegimeId,
        initial_nodes={0: "single_f"},
        edges={
            "single_f": Transition(
                law=ByAge(
                    cases={
                        AgeRange(exclusive_stop=1): {
                            "married_terminal": StochasticTransition(func=_prob_one)
                        }
                    }
                ),
                gates={"married_terminal": edge},
            )
        },
    )


def test_consent_model_builds_with_valid_routes() -> None:
    """Control for the refusals below: the well-formed gate builds and solves."""
    solution = _consent_model(_gate_edge()).solve(
        params={"discount_factor": _BETA}, log_level="off"
    )
    np.testing.assert_allclose(
        np.asarray(solution.values[0]["single_f"]), np.array([2.9, 4.85]), rtol=1e-6
    )


def test_edge_fallback_to_unknown_regime_is_rejected() -> None:
    """A gate whose fallback names a missing regime is rejected."""
    with pytest.raises(ModelInitializationError, match="no_such_regime"):
        _consent_model(_gate_edge(fallback_regime="no_such_regime"))


def test_edge_leg_naming_a_missing_target_member_is_rejected() -> None:
    """A gate route naming a member the target lacks is rejected."""
    with pytest.raises(
        ModelInitializationError,
        match=r"target_member 'not_a_member', which is not one of the target's",
    ):
        _consent_model(_gate_edge(target_member="not_a_member"))


# Full mini-EKL topology in ONE model via the public API.
#
# `_make_full_topology_regimes` / `_full_topology_transitions` /
# `_with_full_topology_laws` are the topology tests/solution imports (fold reads,
# liveness pins, period replay/waves, release log): state-gated consent edges into
# `married`. The test here forms `married` through each single's `Consent` exit
# and dissolves it through the household's own current-period exit instead.


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
    """The mini-EKL regimes: consent edges into `married`, a plain edge out.

    The regimes carry no law. The sources with several outgoing edges at an age
    (`single_f`, `single_m`, `married`) take theirs from
    `_full_topology_transitions`; `_with_full_topology_laws` wraps them into a
    model's edges.
    """
    single_f = Regime(
        states={"wage": _WAGE3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f},
    )
    single_m = single_f.replace()
    single_f_p1 = Regime(
        states={"wage": _WAGE3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f_ir},
    )
    single_m_p1 = single_f_p1.replace(functions={"utility": _u_single_m_ir})
    married = Regime(
        states={"wage": _WAGE3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_married_ir_f, "m": _u_married_ir_m}
            )
        },
    )
    married_terminal = Regime(
        states={"wage": _WAGE3},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    single_f_terminal = Regime(
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


def _consent_where_marriage_pays_both(wage: ContinuousState) -> BoolND:
    """Mutual consent of the mini-EKL topology, stated on the landing wage.

    On the wage grid {1, 2, 3} marriage pays both partners strictly more than
    their single references (1.5, 1), (5.5, 1), (1.5, 1) only at wage 3, where
    it pays (6, 3): the husband gains nothing at wage 1, the wife nothing at
    wage 2. A gate routes on states, so the comparison is written as the wage
    set where it holds.
    """
    return wage > 2.5


def _full_topology_transitions() -> dict[str, Transition]:
    """The transitions of the mini-EKL sources with several outgoing edges at an age.

    - `single_f` / `single_m`: at age 0 a mutual-consent edge into `married`,
      each single's own leg falling back to her `*_p1` regime;
    - `married`: at age 1 a plain move into `married_terminal`, whose actions are
      all feasible, so no member ever needs a fallback there.
    """

    def _consent_leg(
        *, fallback_regime: RegimeName, member: str
    ) -> dict[str, MemberRoute]:
        return {
            member: MemberRoute(
                target_member=member,
                fallback=ProjectedRegimeValue(
                    regime=fallback_regime, projection={"wage": _identity_wage}
                ),
            )
        }

    def _consent_transition(*, fallback_regime: RegimeName, member: str) -> Transition:
        return Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "married": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "married": Gate(
                    predicate=_consent_where_marriage_pays_both,
                    routes=_consent_leg(fallback_regime=fallback_regime, member=member),
                )
            },
        )

    return {
        "single_f": _consent_transition(fallback_regime="single_f_p1", member="f"),
        "single_m": _consent_transition(fallback_regime="single_m_p1", member="m"),
        "married": Transition(
            law=ByAge(
                cases={
                    AgeRange(start=1, exclusive_stop=2): {
                        "married_terminal": StochasticTransition(func=_prob_one)
                    }
                }
            )
        ),
    }


def _with_full_topology_laws(
    edges: Mapping[str, Mapping[str, AgeSelector]],
) -> dict[str, Mapping[str, AgeSelector] | Transition]:
    """Wrap each source of `edges` that `_full_topology_transitions` covers.

    The source's `targets` are the ones `edges` declares for it; its law and
    gates come from `_full_topology_transitions`.
    """
    transitions = _full_topology_transitions()
    return {
        source: Transition(
            targets=targets,
            law=transitions[source].law,
            gates=transitions[source].gates,
        )
        if source in transitions
        else targets
        for source, targets in edges.items()
    }


def _single_formation_exit(*, member: str, spouse: str) -> Exit:
    """A single's mutual-consent exit into `married` at her own landing period."""
    return Exit(
        targets={
            "self": ExitTarget(
                regime="married", role=member, projection={"wage": _identity_wage}
            )
        },
        rule=Consent(
            counterparty=Counterparty(
                regime=f"single_{spouse}_p1",
                role=spouse,
                projection={"wage": _identity_wage},
            )
        ),
    )


def _exit_topology_regimes() -> dict[str, Regime]:
    """The mini-EKL regimes, formed and dissolved through current-period exits."""
    regimes = _make_full_topology_regimes()
    regimes["married"] = regimes["married"].replace(
        constraints={},
        exit=Exit(
            targets={
                "f": ExitTarget(
                    regime="single_f_p1",
                    projection={"wage": _identity_wage},
                    utility_adjustment=_dissolution_cost_f,
                ),
                "m": ExitTarget(
                    regime="single_m_p1",
                    projection={"wage": _identity_wage},
                    utility_adjustment=_dissolution_cost_m,
                ),
            },
            rule=Unilateral(),
        ),
    )
    for member, spouse in (("f", "m"), ("m", "f")):
        regimes[f"single_{member}_p1"] = regimes[f"single_{member}_p1"].replace(
            exit=_single_formation_exit(member=member, spouse=spouse)
        )
    return regimes


def test_full_ekl_topology_via_public_model_api() -> None:
    """single_f/single_m + married with consent and dissolution exits, one model.

    married (period 1) dissolves at wage=2, where no stay action is kept. Its
    continuation into the (zero-valued) terminal couple is zero, so its V is the
    felicity readout where both members stay. At wage=2 no common action meets
    both thresholds and the exit pays each member's single payoff net of the cost.
    single_f lands in single_f_p1 (period 1), whose CONSENT exit moves her into
    married:

    consent(wage) = (V_married_f > V_single_f) & (V_married_m > V_single_m):
      w=1: (2>1.5)&(1>1)  -> closed (husband indifferent, strict).
      w=2: (5.5>5.5)&...  -> closed (strict; formerly -inf, closed as well).
      w=3: (6>1.5)&(3>1)  -> OPEN.
    Wbar_f = [1.5, 5.5, 6]; V_single_f(0) = wage + 0.95*Wbar_f = [2.425, 7.225, 8.7].
    """
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    model = Model(
        regimes=_exit_topology_regimes(),
        edges={
            "single_f": {"single_f_p1": 0},
            "single_m": {"single_m_p1": 0},
            "single_f_p1": {"single_f_terminal": 1},
            "single_m_p1": {"single_m_terminal": (0, 1)},
            "married": {"married_terminal": 1},
        },
        ages=ages,
        regime_id_class=EKLRegimeId,
        # A couple starting at period 1 stores `married`'s value there; a
        # consent exit alone reads its problem without storing it.
        initial_nodes={
            ages.exact_values[0]: "single_f",
            ages.exact_values[1]: "married",
        },
    )
    solution = model.solve(
        params={"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2},
        log_level="off",
    ).values
    # married period-1: continuation zero, the empty cell paid by the
    # dissolution exit.
    np.testing.assert_allclose(
        np.asarray(solution[1]["married"]),
        np.array([[2.0, 1.0], [5.0, 0.8], [6.0, 3.0]]),
        rtol=1e-6,
    )
    # single_f period-0: continuation through her single_f_p1 consent exit.
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"]),
        np.array([2.425, 7.225, 8.7]),
        rtol=1e-6,
    )


def test_singleton_source_with_two_legs_is_rejected() -> None:
    """A singleton source must declare exactly one route."""
    edge = _gate_edge()
    two_routes = Gate(
        predicate=edge.predicate,
        routes={
            **edge.routes,
            "extra": MemberRoute(
                target_member="m",
                fallback=ProjectedRegimeValue(
                    regime="single_m_terminal",
                    projection={"wage": _identity_wage},
                ),
            ),
        },
    )
    with pytest.raises(RegimeInitializationError, match="exactly one leg"):
        _consent_model(two_routes)


def _always_open_gate(wage: ContinuousState) -> BoolND:
    return jnp.ones_like(wage, dtype=bool)


def _edge_with_refs(*, fallback_regime: RegimeName) -> GatedEdge:
    """A one-leg gated edge whose fallback names the given regime.

    The co-activity guard reads only `.legs[*].fallback.regime`, so everything
    else here is filler that keeps the edge a well-formed `GatedEdge`: the guard
    runs before any regime graph exists and never evaluates the gate or the
    projections.
    """
    return GatedEdge(
        gate=_always_open_gate,
        legs=MappingProxyType(
            {
                "f": MemberRoute(
                    fallback=ProjectedRegimeValue(
                        regime=fallback_regime,
                        projection={"wage": _identity_wage},
                    )
                )
            }
        ),
    )


def test_gated_edge_reference_uncovered_in_consumed_period_is_rejected() -> None:
    """A fallback uncovered in a period whose ``Wbar`` is
    actually CONSUMED (target covered at t, source covered at t-1) must be rejected
    at construction, so the solve-side `_roll_gated_edges` never feeds the source
    a stale later-period ``Wbar``.
    """
    edge = _edge_with_refs(fallback_regime="single_m_terminal")
    regimes_to_active_periods = MappingProxyType(
        {
            "single_f": (0,),  # source solved in period 0
            "married_terminal": (1,),  # target solved in period 1 -> consumed
            "single_m_terminal": (2,),  # not solved in consumed period 1
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"single_m_terminal.*solved.*\[1\]",
    ):
        _fail_if_gated_edge_references_inactive(
            prefix=(
                "Regime 'single_f', value-dependent transition into "
                "'married_terminal': "
            ),
            source_name="single_f",
            target_name="married_terminal",
            edge=edge,
            regimes_to_active_periods=regimes_to_active_periods,
            gated_periods=(0,),
        )


def test_gated_edge_references_covered_in_every_consumed_period_pass() -> None:
    """A fallback covered in every consumed period passes the guard."""
    edge = _edge_with_refs(fallback_regime="single_m_terminal")
    regimes_to_active_periods = MappingProxyType(
        {
            "single_f": (0,),
            "married_terminal": (1,),
            "single_m_terminal": (1,),
        }
    )
    # No raise: every referenced regime covers the consumed period {1}.
    _fail_if_gated_edge_references_inactive(
        prefix=(
            "Regime 'single_f', value-dependent transition into 'married_terminal': "
        ),
        source_name="single_f",
        target_name="married_terminal",
        edge=edge,
        regimes_to_active_periods=regimes_to_active_periods,
        gated_periods=(0,),
    )


def test_gated_edge_reference_uncovered_at_unconsumed_boundary_passes() -> None:
    """Over-reach guard: a self-loop edge's reference may be uncovered in a
    target-covered period whose ``Wbar`` is NEVER consumed — the target's earliest
    covered period, where no source exists one period earlier. Requiring
    joint coverage across ALL target-covered periods would wrongly reject the
    legitimate repeating self-loop boundary.
    """
    # Self-loop: source == target == "src", solved in periods {0, 1}; the
    # fallback is solved in {1, 2}, i.e. not in target-solved period 0 — but
    # period 0's Wbar is unconsumed (no source at period -1), so it is legal.
    edge = _edge_with_refs(fallback_regime="src_fallback")
    regimes_to_active_periods = MappingProxyType(
        {
            "src": (0, 1),
            "src_fallback": (1, 2),
        }
    )
    # Consumed = {t in {0,1} : t-1 in {0,1}} = {1}; fallback solved {1,2} covers
    # it. No raise, even though the fallback is absent in target-solved period 0.
    _fail_if_gated_edge_references_inactive(
        prefix="Regime 'src', value-dependent transition into 'src': ",
        source_name="src",
        target_name="src",
        edge=edge,
        regimes_to_active_periods=regimes_to_active_periods,
        gated_periods=(0, 1),
    )
