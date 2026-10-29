"""Integration tests for value decisions on mixed singleton/collective edges.

A source regime declares an `ArrivalDecision` (or a state-only `Gate`) beside the
law of its `Transition`, keyed by TARGET regime name. After the scheduled landing
and before the target acts, the decision compares its typed reads, per source
stakeholder ``s``, and routes the row into the target or into that stakeholder's
fallback (EKL 2019 eqs. 9/12/27)::

    Wbar^s(x) = jnp.where(decision(x), V_target^{route_s}(x), V_fallback^s(pi_s(x)))

Two moves matter for EKL: the singles->married CONSENT decision (a singleton
source reaches a collective target only by mutual consent, eq. 27) and the
married DISSOLUTION, a current-period participation exit routing each member to
her single regime.
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
from _lcm.user_regime_validation import validate_regime_law
from _lcm.utils.logging import get_logger
from lcm import (
    AgeRange,
    ArrivalDecision,
    ArrivalRead,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    ValueDependentConstraint,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.exits import Exit, ExitTarget
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
from tests.regime_building.test_collective_regime_simulate import _ParticipationRule


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


def _consent_gate(
    *,
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
) -> BoolND:
    # EKL eq. 27: strict, unanimous mutual consent.
    return (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)


def _ir_f(*, Q_f: FloatND, V_single_f_ref: FloatND, delta_f: FloatND) -> BoolND:
    return Q_f >= V_single_f_ref - delta_f


def _ir_m(*, Q_m: FloatND, V_single_m_ref: FloatND, delta_m: FloatND) -> BoolND:
    return Q_m >= V_single_m_ref - delta_m


def _no_dissolution_gate(D_target: BoolND) -> BoolND:
    return ~D_target


def _consent_source() -> Regime:
    """The consent source, which may move into the couple at age 0."""
    return Regime(
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f},
    )


def _consent_laws(edge: Gate | ArrivalDecision | None) -> RegimeLaws:
    """The consent regimes' laws, the source's move into the couple under `edge`.

    `None` declares the move into the couple without a gate or arrival decision.
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
                gates=({"married_terminal": edge} if isinstance(edge, Gate) else {}),
                arrivals=(
                    {"married_terminal": edge}
                    if isinstance(edge, ArrivalDecision)
                    else {}
                ),
            ),
            "single_f_terminal": None,
            "single_m_terminal": None,
            "married_terminal": None,
        }
    )


def _consent_reads(
    *, target: RegimeName, f_reference: RegimeName, m_reference: RegimeName
) -> dict[str, ArrivalRead]:
    """Both partners' values in the couple and each one's single reference."""
    return {
        "V_target_f": ArrivalRead(
            kind="IntegratedFull",
            regime=target,
            stakeholder="f",
            projection={"wage": _identity_wage},
        ),
        "V_target_m": ArrivalRead(
            kind="IntegratedFull",
            regime=target,
            stakeholder="m",
            projection={"wage": _identity_wage},
        ),
        "V_single_f_ref": ArrivalRead(
            kind="IntegratedFull",
            regime=f_reference,
            projection={"wage": _identity_wage},
        ),
        "V_single_m_ref": ArrivalRead(
            kind="IntegratedFull",
            regime=m_reference,
            projection={"wage": _identity_wage},
        ),
    }


def _consent_edge(
    *, fallback_regime: RegimeName = "single_f_terminal", target_stakeholder: str = "f"
) -> ArrivalDecision:
    """The consent decision into the couple, routing the wife to her own fallback."""
    return ArrivalDecision(
        predicate=_consent_gate,
        routes={
            "f": StakeholderRoute(
                target_stakeholder=target_stakeholder,
                fallback=ProjectedRegimeValue(
                    regime=fallback_regime,
                    projection={"wage": _identity_wage},
                ),
            )
        },
        reads=_consent_reads(
            target="married_terminal",
            f_reference="single_f_terminal",
            m_reference="single_m_terminal",
        ),
    )


def _make_consent_regimes() -> dict[str, Regime]:
    single_f = _consent_source()
    single_f_terminal = Regime(
        states={"wage": _WAGE},
        functions={"utility": _u_single_f_terminal},
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


def _solve_consent(*, enable_jit: bool = False):
    laws = _consent_laws(_consent_edge())
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
    return (
        _bi_result.value_functions,
        _bi_result.simulation_policies,
        _bi_result.dissolution_flags,
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
    solution, _sim, _dissolution = _solve_consent()
    V_single_f = np.asarray(solution[0]["single_f"])
    np.testing.assert_allclose(V_single_f, np.array([2.9, 4.85]), rtol=1e-6)


def test_consent_gate_matches_under_jit():
    """The public (jitted) path folds the same edge."""
    solution, _sim, _dissolution = _solve_consent(enable_jit=True)
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


def test_raw_ungated_mixed_transition_still_rejected():
    """A singleton regime reaching a collective target WITHOUT an edge is rejected."""
    regimes = _make_consent_regimes()
    # Drop the gate but keep the singleton -> collective transition.
    laws = _consent_laws(None)
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    with pytest.raises(NotImplementedError, match="stakeholders"):
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


def test_probabilistic_gate_is_rejected():
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
                        "f": StakeholderRoute(
                            target_stakeholder="f",
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


def _consent_model(edge: ArrivalDecision) -> Model:
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
                arrivals={"married_terminal": edge},
            )
        },
    )


def test_consent_model_builds_with_valid_routes():
    """Control for the refusals below: the well-formed decision builds and solves."""
    solution = _consent_model(_consent_edge()).solve(
        params={"discount_factor": _BETA}, log_level="off"
    )
    np.testing.assert_allclose(
        np.asarray(solution.values[0]["single_f"]), np.array([2.9, 4.85]), rtol=1e-6
    )


def test_edge_fallback_to_unknown_regime_is_rejected():
    """An arrival decision whose fallback names a missing regime is rejected."""
    with pytest.raises(ModelInitializationError, match="no_such_regime"):
        _consent_model(_consent_edge(fallback_regime="no_such_regime"))


def test_edge_leg_naming_a_missing_target_stakeholder_is_rejected():
    """An arrival route naming a target stakeholder the target lacks is rejected."""
    with pytest.raises(RegimeInitializationError, match="exact role"):
        _consent_model(_consent_edge(target_stakeholder="not_a_stakeholder"))


# Full mini-EKL topology in ONE model via the public API.
#
# `_make_full_topology_regimes` / `_full_topology_transitions` /
# `_with_full_topology_laws` are the legacy value-gate fixture still imported by
# tests/solution (fold reads, liveness pins, period replay/waves, release log).
# They retire with those consumers. The test here uses the migrated topology:
# consent arrival decisions and a current-period participation exit.


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
    """The mini-EKL regimes: consent edges into `married`, a dissolution edge out.

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
        constraints={
            "ir_f": ValueDependentConstraint(
                predicate=_ir_f,
                references={
                    "V_single_f_ref": ProjectedRegimeValue(
                        regime="single_f_p1", projection={"wage": _identity_wage}
                    )
                },
            ),
            "ir_m": ValueDependentConstraint(
                predicate=_ir_m,
                references={
                    "V_single_m_ref": ProjectedRegimeValue(
                        regime="single_m_p1", projection={"wage": _identity_wage}
                    )
                },
            ),
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


def _full_topology_transitions() -> dict[str, Transition]:
    """The transitions of the mini-EKL sources with several outgoing edges at an age.

    - `single_f` / `single_m`: at age 0 a mutual-consent edge into `married`,
      each single's own leg falling back to her `*_p1` regime;
    - `married`: at age 1 a dissolution edge into `married_terminal`, each
      stakeholder falling back to her own single terminal regime.
    """

    def _consent_leg(
        *, fallback_regime: RegimeName, stakeholder: str
    ) -> dict[str, StakeholderRoute]:
        return {
            stakeholder: StakeholderRoute(
                target_stakeholder=stakeholder,
                fallback=ProjectedRegimeValue(
                    regime=fallback_regime, projection={"wage": _identity_wage}
                ),
            )
        }

    consent_gate_refs = {
        "V_single_f_ref": ProjectedRegimeValue(
            regime="single_f_p1", projection={"wage": _identity_wage}
        ),
        "V_single_m_ref": ProjectedRegimeValue(
            regime="single_m_p1", projection={"wage": _identity_wage}
        ),
    }

    def _consent_transition(
        *, fallback_regime: RegimeName, stakeholder: str
    ) -> Transition:
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
                    predicate=_consent_gate,
                    routes=_consent_leg(
                        fallback_regime=fallback_regime, stakeholder=stakeholder
                    ),
                    references=consent_gate_refs,
                )
            },
        )

    return {
        "single_f": _consent_transition(fallback_regime="single_f_p1", stakeholder="f"),
        "single_m": _consent_transition(fallback_regime="single_m_p1", stakeholder="m"),
        "married": Transition(
            law=ByAge(
                cases={
                    AgeRange(start=1, exclusive_stop=2): {
                        "married_terminal": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "married_terminal": Gate(
                    predicate=_no_dissolution_gate,
                    routes={
                        "f": StakeholderRoute(
                            target_stakeholder="f",
                            fallback=ProjectedRegimeValue(
                                regime="single_f_terminal",
                                projection={"wage": _identity_wage},
                            ),
                        ),
                        "m": StakeholderRoute(
                            target_stakeholder="m",
                            fallback=ProjectedRegimeValue(
                                regime="single_m_terminal",
                                projection={"wage": _identity_wage},
                            ),
                        ),
                    },
                )
            },
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


def _participation_topology_regimes() -> dict[str, Regime]:
    """The mini-EKL regimes, `married` dissolving through a participation exit."""
    regimes = _make_full_topology_regimes()
    regimes["married"] = regimes["married"].replace(
        constraints={},
        exits={
            "dissolve": Exit(
                targets={
                    "f": ExitTarget(
                        regime="single_f_p1", projection={"wage": _identity_wage}
                    ),
                    "m": ExitTarget(
                        regime="single_m_p1", projection={"wage": _identity_wage}
                    ),
                },
                rule=_ParticipationRule(),
            )
        },
    )
    return regimes


def _consent_arrival(*, fallback_regime: RegimeName, stakeholder: str) -> Transition:
    """A single's mutual-consent arrival into `married` at age 0."""
    return Transition(
        targets={"married": 0, fallback_regime: 0},
        law=ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "married": StochasticTransition(func=_prob_one)
                }
            }
        ),
        arrivals={
            "married": ArrivalDecision(
                predicate=_consent_gate,
                routes={
                    stakeholder: StakeholderRoute(
                        target_stakeholder=stakeholder,
                        fallback=ProjectedRegimeValue(
                            regime=fallback_regime,
                            projection={"wage": _identity_wage},
                        ),
                    )
                },
                reads={
                    **{
                        f"V_target_{member}": ArrivalRead(
                            kind="Participating",
                            regime="married",
                            stakeholder=member,
                            projection={"wage": _identity_wage},
                        )
                        for member in ("f", "m")
                    },
                    **{
                        f"V_single_{member}_ref": ArrivalRead(
                            kind="IntegratedFull",
                            regime=f"single_{member}_p1",
                            projection={"wage": _identity_wage},
                        )
                        for member in ("f", "m")
                    },
                },
            )
        },
    )


def test_full_ekl_topology_via_public_model_api():
    """single_f/single_m + married with consent arrival + dissolution exit, one model.

    married (period 1) is the participation miniature (empty participation at
    wage=2). Its continuation into the (zero-valued) terminal couple is zero, so
    its V is the felicity readout where both members participate. At wage=2 no
    common action meets both thresholds and the exit pays each member's full
    single payoff (5.5, 1.0), where the retired dissolution flag left `-inf`.
    single_f (period 0) reaches married through the CONSENT arrival:

    decision(wage) = (V_married_f > V_single_f) & (V_married_m > V_single_m):
      w=1: (2>1.5)&(1>1)  -> closed (husband indifferent, strict).
      w=2: (5.5>5.5)&...  -> closed (strict; formerly -inf, closed as well).
      w=3: (6>1.5)&(3>1)  -> OPEN.
    Wbar_f = [1.5, 5.5, 6]; V_single_f(0) = wage + 0.95*Wbar_f = [2.425, 7.225, 8.7].
    """
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    model = Model(
        regimes=_participation_topology_regimes(),
        edges={
            "single_f": _consent_arrival(
                fallback_regime="single_f_p1", stakeholder="f"
            ),
            "single_m": _consent_arrival(
                fallback_regime="single_m_p1", stakeholder="m"
            ),
            "single_f_p1": {"single_f_terminal": 1},
            "single_m_p1": {"single_m_terminal": (0, 1)},
            "married": {"married_terminal": 1},
        },
        ages=ages,
        regime_id_class=EKLRegimeId,
        initial_nodes={ages.exact_values[0]: "single_f"},
    )
    solution = model.solve(
        params={"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2},
        log_level="off",
    ).values
    # married period-1: participation miniature (continuation zero, the empty
    # cell paid by the dissolution exit).
    np.testing.assert_allclose(
        np.asarray(solution[1]["married"]),
        np.array([[2.0, 1.0], [5.5, 1.0], [6.0, 3.0]]),
        rtol=1e-6,
    )
    # single_f period-0: mutual-consent continuation.
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"]),
        np.array([2.425, 7.225, 8.7]),
        rtol=1e-6,
    )


def test_singleton_source_with_two_legs_is_rejected():
    """A singleton source must declare exactly one route."""
    edge = _consent_edge()
    two_routes = ArrivalDecision(
        predicate=edge.predicate,
        reads=edge.reads,
        routes={
            **edge.routes,
            "extra": StakeholderRoute(
                target_stakeholder="m",
                fallback=ProjectedRegimeValue(
                    regime="single_m_terminal",
                    projection={"wage": _identity_wage},
                ),
            ),
        },
    )
    with pytest.raises(RegimeInitializationError, match="single source route"):
        _consent_model(two_routes)


def _always_open_gate(V_target_f: FloatND) -> BoolND:
    return jnp.ones_like(V_target_f, dtype=bool)


def _edge_with_refs(
    *, fallback_regime: RegimeName, gate_ref_regime: RegimeName
) -> GatedEdge:
    """A one-leg gated edge whose fallback and gate reference name given regimes.

    The co-activity guard reads only `.legs[*].fallback.regime` and
    `.gate_refs[*].regime`, so everything else here is filler that keeps the edge
    a well-formed `GatedEdge`: the guard runs before any regime graph exists and
    never evaluates the gate or the projections.
    """
    return GatedEdge(
        gate=_always_open_gate,
        legs=MappingProxyType(
            {
                "f": StakeholderRoute(
                    fallback=ProjectedRegimeValue(
                        regime=fallback_regime,
                        projection={"wage": _identity_wage},
                    )
                )
            }
        ),
        gate_refs=MappingProxyType(
            {
                "g": ProjectedRegimeValue(
                    regime=gate_ref_regime,
                    projection={"wage": _identity_wage},
                )
            }
        ),
    )


def test_gated_edge_reference_uncovered_in_consumed_period_is_rejected():
    """A gate reference (or fallback) uncovered in a period whose ``Wbar`` is
    actually CONSUMED (target covered at t, source covered at t-1) must be rejected
    at construction, so the solve-side `_roll_gated_edges` never feeds the source
    a stale later-period ``Wbar``.
    """
    edge = _edge_with_refs(
        fallback_regime="single_f_terminal", gate_ref_regime="single_m_terminal"
    )
    regimes_to_active_periods = MappingProxyType(
        {
            "single_f": (0,),  # source solved in period 0
            "married_terminal": (1,),  # target solved in period 1 -> consumed
            "single_f_terminal": (1,),  # also solved in period 1
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


def test_gated_edge_references_covered_in_every_consumed_period_pass():
    """References covered in every consumed period pass the guard."""
    edge = _edge_with_refs(
        fallback_regime="single_f_terminal", gate_ref_regime="single_m_terminal"
    )
    regimes_to_active_periods = MappingProxyType(
        {
            "single_f": (0,),
            "married_terminal": (1,),
            "single_f_terminal": (1,),
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


def test_gated_edge_reference_uncovered_at_unconsumed_boundary_passes():
    """Over-reach guard: a self-loop edge's reference may be uncovered in a
    target-covered period whose ``Wbar`` is NEVER consumed — the target's earliest
    covered period, where no source exists one period earlier. Requiring
    joint coverage across ALL target-covered periods would wrongly reject the
    legitimate repeating self-loop boundary. The public occupied-boundary
    counterpart is
    `test_repeating_arrival_boundary.py`,
    `test_public_repeating_arrival_stops_at_the_sources_last_supported_date`;
    this kernel control separately checks reference coverage.
    """
    # Self-loop: source == target == "src", solved in periods {0, 1}; the
    # fallback is solved in {1, 2}, i.e. not in target-solved period 0 — but
    # period 0's Wbar is unconsumed (no source at period -1), so it is legal.
    edge = _edge_with_refs(
        fallback_regime="src_fallback", gate_ref_regime="src_fallback"
    )
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


@pytest.mark.parametrize("field_name", ["legs", "gate_refs"])
def test_gated_edge_refuses_a_plain_dict_for_its_mappings(field_name: str) -> None:
    """A gated edge holds read-only mappings; a plain dict is refused."""
    edge = _edge_with_refs(fallback_regime="single", gate_ref_regime="single")
    arguments = {
        "legs": edge.legs,
        "gate_refs": edge.gate_refs,
        field_name: dict(getattr(edge, field_name)),
    }
    with pytest.raises(RegimeInitializationError, match=field_name):
        GatedEdge(gate=_always_open_gate, **arguments)  # ty: ignore[invalid-argument-type]
