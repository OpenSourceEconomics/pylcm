"""The pruning walk and the usage check read one definition of a root.

Broadcast pruning asks which variables a regime's computations reach; the
variable-usage check asks which declared variables are reached by them. Both
questions are answered from the same root set, so a slot that is a read for one
is a read for the other. This module pins that agreement by recording what each
consumer asks for while a model exercising every root slot is built.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

import jax.numpy as jnp
import pytest

from _lcm import model_processing
from _lcm.regime_building import broadcast
from lcm import (
    AgeGrid,
    DiscreteGrid,
    EdgeLeg,
    GatedEdge,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    SamePeriodRef,
    categorical,
    fixed_transition,
)
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    IntND,
    RegimeName,
    ScalarInt,
    UserFunction,
)

# Which regime is asked for its roots, in which phase.
type RootCallKey = tuple[str, str]


@categorical(ordered=False)
class RegimeId:
    """Regime ids of the every-slot model."""

    couple: ScalarInt  # code 0
    couple_ir: ScalarInt  # code 1
    couple_terminal: ScalarInt  # code 2
    single_f: ScalarInt  # code 3
    single_m: ScalarInt  # code 4
    single_terminal: ScalarInt  # code 5


@categorical(ordered=True)
class Work:
    """The binary action every regime of the every-slot model offers."""

    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=True)
class WageGroup:
    """The derived categorical `wage_group` indexes."""

    low: ScalarInt  # code 0
    high: ScalarInt  # code 1


_AGES = AgeGrid(start=0, stop=3, step="Y")

_WAGE_GRID = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)

# The model-level state nothing reads. Its only role is to give the pruning
# walk a candidate in every regime, so the walk visits them all.
_BONUS_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)

# Every regime, in both phases: the roots the model build must ask for.
_EXPECTED_CALL_KEYS = frozenset(
    (regime_name, phase)
    for regime_name in (
        "couple",
        "couple_ir",
        "couple_terminal",
        "single_f",
        "single_m",
        "single_terminal",
    )
    for phase in ("solve", "simulate")
)


def test_both_consumers_ask_for_the_same_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pruning and the usage check see the same root names in every regime."""
    recorded = _record_root_calls(monkeypatch)

    assert recorded["pruning"] == recorded["usage"]


def test_every_regime_and_phase_is_asked_for_its_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both consumers cover every regime in both phases, so agreement is not vacuous."""
    recorded = _record_root_calls(monkeypatch)

    assert set(recorded["pruning"]) == set(recorded["usage"]) == _EXPECTED_CALL_KEYS


def test_the_root_set_names_every_slot_of_the_regime_that_carries_it() -> None:
    """`couple_ir`'s roots name its value constraints, references and incoming gate.

    An incoming fallback is rooted once per phase, because the value an agent
    expects and the state a settlement realizes may be declared separately.
    """
    roots = broadcast.root_functions(
        regime_name="couple_ir",
        regime=_make_regimes()["couple_ir"],
        all_regimes=_make_regimes(),
        phase="solve",
    )

    assert set(roots) == {
        "__utility__utility_f",
        "__utility__utility_m",
        "__next_regime__couple_terminal",
        "__value_constraint__ir_f",
        "__value_constraint__ir_m",
        "__same_period_ref__V_single_f_ref__wage",
        "__same_period_ref__V_single_m_ref__wage",
        "__incoming_gate__couple",
        "__incoming_gate_ref__couple__V_single_ref__wage",
        "__incoming_fallback__couple__f__solve__wage",
        "__incoming_fallback__couple__f__simulate__wage",
        "__incoming_fallback__couple__m__solve__wage",
        "__incoming_fallback__couple__m__simulate__wage",
    }


def _record_root_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> Mapping[str, dict[RootCallKey, frozenset[str]]]:
    """Build the every-slot model, recording each consumer's root names.

    Returns a mapping of consumer name (`"pruning"` for the broadcast walk,
    `"usage"` for the variable-usage check) to the root names that consumer
    asked for, per regime and phase.
    """
    original = broadcast.root_functions
    recorded: dict[str, dict[RootCallKey, frozenset[str]]] = {
        "pruning": {},
        "usage": {},
    }

    def make_spy(consumer: str) -> object:
        def spy(
            *,
            regime_name: RegimeName,
            regime: Regime,
            all_regimes: Mapping[RegimeName, Regime],
            phase: Literal["solve", "simulate"],
            koopmans_aggregator: UserFunction | None = None,
        ) -> MappingProxyType[str, UserFunction]:
            roots = original(
                regime_name=regime_name,
                regime=regime,
                all_regimes=all_regimes,
                phase=phase,
                koopmans_aggregator=koopmans_aggregator,
            )
            recorded[consumer][(regime_name, phase)] = frozenset(roots)
            return roots

        return spy

    monkeypatch.setattr(broadcast, "root_functions", make_spy("pruning"))
    monkeypatch.setattr(model_processing, "root_functions", make_spy("usage"))

    Model(
        regimes=_make_regimes(),
        ages=_AGES,
        regime_id_class=RegimeId,
        states={"bonus": _BONUS_GRID},
        state_transitions={"bonus": fixed_transition("bonus")},
    )
    return recorded


def _make_regimes() -> dict[str, Regime]:
    """Build a six-regime model exercising every root slot.

    A collective `couple` reaches a collective `couple_ir` through both a raw
    per-target regime transition and a gated edge whose gate reads the target's
    dissolution flag and a same-period reference to `single`. `couple_ir`
    carries the value constraints and same-period references that flag drives.
    """
    couple = Regime(
        transition={"couple_ir": MarkovTransition(_probability_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _utility_f_with_group,
            "utility_m": Phased(solve=_utility_m_solve, simulate=_utility_m_simulate),
            "wage_group": _wage_group,
        },
        derived_categoricals={"wage_group": DiscreteGrid(WageGroup)},
        constraints={"work_pays": _work_pays},
        gated_edges={
            "couple_ir": GatedEdge(
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
                gate_refs={
                    "V_single_ref": SamePeriodRef(
                        regime="single_f", projection={"wage": _identity_wage}
                    )
                },
            )
        },
    )
    couple_ir = Regime(
        transition={"couple_terminal": MarkovTransition(_probability_one)},
        active=lambda age: (age >= 1) & (age < 2),
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_collective, "utility_m": _utility_collective},
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
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_collective, "utility_m": _utility_collective},
    )
    single_f = Regime(
        transition={"single_terminal": MarkovTransition(_probability_one)},
        active=lambda age: (age >= 1) & (age < 2),
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_single_f},
    )
    single_m = single_f.replace(functions={"utility": _utility_single_m})
    single_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wage": _WAGE_GRID},
        functions={"utility": _utility_terminal_single},
    )
    return {
        "couple": couple,
        "couple_ir": couple_ir,
        "couple_terminal": couple_terminal,
        "single_f": single_f,
        "single_m": single_m,
        "single_terminal": single_terminal,
    }


def _probability_one(age: FloatND) -> FloatND:
    """Regime transition: the declared target is reached with probability one."""
    return jnp.ones_like(age, dtype=float)


def _wage_group(wage: ContinuousState) -> IntND:
    """Split the wage grid into a low and a high group."""
    return jnp.int32(wage > 2.0)


def _utility_f_with_group(
    wage: ContinuousState,
    work: DiscreteAction,
    wage_group: IntND,
    group_bonus: FloatND,
) -> FloatND:
    """The wife's payoff: her wage when working, plus her group's bonus."""
    return wage * work + group_bonus[wage_group]


def _utility_m_solve(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """The husband's payoff during backward induction."""
    return 0.5 * wage * work


def _utility_m_simulate(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """The husband's payoff as simulation records it."""
    return 0.5 * wage * work


def _utility_collective(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """A stakeholder's payoff: the wage when working, nothing otherwise."""
    return wage * work


def _utility_single_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """The single wife's payoff: twice the wage when working."""
    return 2.0 * wage * work


def _utility_single_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """The single husband's payoff: the wage when working."""
    return wage * work


def _utility_terminal_single(wage: ContinuousState) -> FloatND:
    """The single's terminal payoff."""
    return 0.5 * wage


def _work_pays(wage: ContinuousState, work: DiscreteAction) -> BoolND:
    """Working is only feasible above the lowest wage node."""
    return (work == 0) | (wage > 1.0)


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    """Read the reference regime at the same wage the declaring cell sits on."""
    return wage


def _no_dissolution_gate(D_target: BoolND) -> BoolND:
    """The edge is open wherever the target does not dissolve."""
    return ~D_target


def _ir_f(Q_f: FloatND, V_single_f_ref: FloatND, delta_f: FloatND) -> BoolND:
    """The wife participates while the match beats her outside option."""
    return Q_f >= V_single_f_ref - delta_f


def _ir_m(Q_m: FloatND, V_single_m_ref: FloatND, delta_m: FloatND) -> BoolND:
    """The husband participates while the match beats his outside option."""
    return Q_m >= V_single_m_ref - delta_m
