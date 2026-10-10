"""Lower age-indexed regime declarations into demand, support and engine laws.

A regime's law (bound from `Model(edges=...)` and kept on the model graph)
declares where a local problem is available and where it may go.
This module is the single place that reads it for that purpose:

- `None` (no outgoing edges) is terminal and available at every age;
- a plain nonterminal law — a regime name, a `DeterministicTransition`, a vector
  `StochasticTransition` with `targets`, or a per-target mapping — is available at
  every non-final age;
- a `ByAge` schedule is available at the non-final ages its cases select. A law
  it selects at the last age is unused: no continuation exists there. A schedule
  may select no available age at all; it is an error only once a required
  problem needs it.

Availability alone solves nothing. `resolve_demand` expands the declared starts
into two sets of `(period, regime)` pairs:

- H, the visited pairs: those a subject can physically visit;
- S, the valued pairs: H and every pair whose value a problem in S reads.

S is the coverage every later stage reads. Support comes only from the
declarations. A bare callable or a vector `StochasticTransition` without `targets`
names no support and is rejected.

The engine downstream reads one law per regime and phase, lowered by
`lower_demanded_transitions` from the laws demand selects, only. A schedule
whose demanded periods select different laws is lowered into one equivalent
law whose cells dispatch on the `period` context argument; a cell whose target
is outside the selected law's support at a period evaluates to exactly zero
there. Laws shared across cases are reused unchanged, so no per-age closure is
created.
"""

import dataclasses
import inspect
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NoReturn, cast, no_type_check

import jax.numpy as jnp

from _lcm.gated_edge import GatedEdge
from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
    _SupportedStochasticTransition,
)
from _lcm.time import TimeAxis, coordinate_kind
from _lcm.typing import AnnotationForm, EconFunctionArg, RegimeName
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.initial_nodes import InitialNodes, UserInitialNodes
from lcm.phased import Phased
from lcm.transition import (
    AgeRange,
    ByAge,
    DeterministicTransition,
    PeriodRange,
    Periods,
    StochasticTransition,
    _fail_if_invalid_age_selector,
    _select_periods,
)
from lcm.typing import FloatND, IntND, Period, UserAge, UserFunction

if TYPE_CHECKING:
    from _lcm.regime_law import RegimeLaw
else:
    # `_lcm.regime_law` imports this module, so `RegimeLaw` is not importable here
    # at runtime; the law's own constructor checks its fields.
    type RegimeLaw = object

type PhaseKey = str
type Side = Literal["solve", "simulate"]
# One per-target probability cell, as declared or bound.
type ProbabilityCell = StochasticTransition | UserFunction | Phased
# One phase's law at one source age: a regime name, a function or
# `DeterministicTransition` returning a regime code, a vector
# `StochasticTransition`, or a per-target mapping of probability cells.
type PhaseLaw = (
    RegimeName
    | DeterministicTransition
    | StochasticTransition
    | UserFunction
    | Mapping[RegimeName, ProbabilityCell]
)
# A law at one source age: one law for both phases, or a `Phased` pair.
type CaseLaw = PhaseLaw | Phased[PhaseLaw, PhaseLaw]
# A source's whole regime-transition law: a case law, an age schedule of case
# laws, or `None` for a terminal regime.
type RegimeTransitionLaw = CaseLaw | ByAge | None
# One phase's law in the engine vocabulary: a callable returning a regime code or
# a probability vector, a vector `StochasticTransition`, or a per-target mapping
# of probability cells.
type EnginePhaseLaw = (
    UserFunction | StochasticTransition | Mapping[RegimeName, ProbabilityCell]
)
# A law in the engine vocabulary: one law for both phases, or a `Phased` pair.
type EngineLaw = EnginePhaseLaw | Phased[EnginePhaseLaw, EnginePhaseLaw]


_PHASES: tuple[PhaseKey, PhaseKey] = ("solution", "simulation")
_SIDES: tuple[Side, Side] = ("solve", "simulate")


@dataclass(frozen=True, kw_only=True)
class RegimeSchedules:
    """Coverage and per-period support of every regime, from its declaration."""

    coverage_by_regime: MappingProxyType[RegimeName, tuple[int, ...]]
    """Periods at which each regime is solved: its available periods until
    `resolve_demand` restricts them to the demanded ones."""

    support_by_phase: MappingProxyType[
        PhaseKey, MappingProxyType[RegimeName, MappingProxyType[int, tuple[str, ...]]]
    ]
    """Per phase, per source regime, the declared targets at each covered period."""

    value_reads_by_phase: MappingProxyType[
        PhaseKey, MappingProxyType[RegimeName, MappingProxyType[int, tuple[str, ...]]]
    ] = MappingProxyType({})
    """Per phase, per source regime, the regimes whose next-age value a gated
    edge reads at each period: gate references, plus the solve fallbacks in
    the solution phase."""

    landings_by_regime: MappingProxyType[
        RegimeName, MappingProxyType[int, tuple[str, ...]]
    ] = MappingProxyType({})
    """Per source regime, the simulate fallbacks a gated edge may land a row in
    at the next age, at each period."""

    valued_nodes: frozenset[tuple[int, RegimeName]] = frozenset()
    """The valued pairs S."""

    visited_nodes: frozenset[tuple[int, RegimeName]] = frozenset()
    """The visited pairs H."""

    law_by_period_by_regime: MappingProxyType[
        RegimeName, MappingProxyType[int, CaseLaw]
    ] = MappingProxyType({})
    """Per nonterminal regime, the declared law at each available period."""

    gated_edges_by_regime: MappingProxyType[
        RegimeName, Mapping[RegimeName, GatedEdge]
    ] = MappingProxyType({})
    """Per source regime, its gated edges by gated target."""

    @property
    def targets_by_regime(self) -> MappingProxyType[RegimeName, frozenset[RegimeName]]:
        """Per source regime, every target its support names in either phase."""
        return MappingProxyType(
            {
                name: frozenset(
                    target
                    for by_regime in self.support_by_phase.values()
                    for targets in by_regime.get(name, {}).values()
                    for target in targets
                )
                for name in self.coverage_by_regime
            }
        )

    @property
    def visited_periods_by_regime(
        self,
    ) -> MappingProxyType[RegimeName, tuple[int, ...]]:
        """Per regime, the periods of its visited pairs, ascending."""
        return MappingProxyType(
            {
                name: tuple(sorted(p for p, n in self.visited_nodes if n == name))
                for name in self.coverage_by_regime
            }
        )


def uses_declaration_vocabulary(transition: RegimeTransitionLaw) -> bool:
    """Whether a regime transition has a form the engine cannot read directly."""
    if isinstance(transition, ByAge | DeterministicTransition | str):
        return True
    if isinstance(transition, StochasticTransition):
        return isinstance(transition, _SupportedStochasticTransition)
    if isinstance(transition, Phased):
        return any(
            uses_declaration_vocabulary(side)
            for side in (transition.solve, transition.simulate)
        )
    return False


def declaration_view(transition: RegimeTransitionLaw) -> EngineLaw | None:
    """Return a period-independent law in the engine vocabulary, for validation.

    A `ByAge` is replaced by the union of its laws: every cell each case declares
    for a target. A regime name and a `DeterministicTransition` become probability
    cells when they meet a stochastic law, and deterministic otherwise. Cells
    are not masked by age, so the view is only for construction-time checks of
    the declared grammar, gates and state handoffs, never for evaluation.
    """
    if isinstance(transition, ByAge):
        return _union_law(
            laws=cast("tuple[CaseLaw, ...]", transition.laws),
            code_by_name=None,
            mask=None,
        )
    if isinstance(transition, Phased):
        return Phased(
            solve=_plain_law(law=transition.solve, code_by_name=None),
            simulate=_plain_law(law=transition.simulate, code_by_name=None),
        )
    if transition is None:
        return None
    return _plain_law(law=transition, code_by_name=None)


def resolve_regime_schedules(
    *,
    laws: Mapping[RegimeName, RegimeLaw],
    ages: TimeAxis,
    source_ages_by_phase: Mapping[
        PhaseKey, Mapping[RegimeName, Mapping[RegimeName, frozenset[UserAge]]]
    ]
    | None = None,
) -> RegimeSchedules:
    """Resolve every regime's available periods, support and value reads.

    Nothing is lowered here: `lower_demanded_transitions` lowers only the laws
    demand selects. `laws` maps each regime to its graph-bound `RegimeLaw`.
    """
    all_periods = tuple(range(ages.n_periods))
    coverage: dict[RegimeName, tuple[int, ...]] = {}
    support: dict[
        PhaseKey, dict[RegimeName, MappingProxyType[int, tuple[str, ...]]]
    ] = {phase: {} for phase in _PHASES}
    value_reads: dict[
        PhaseKey, dict[RegimeName, MappingProxyType[int, tuple[str, ...]]]
    ] = {phase: {} for phase in _PHASES}
    landings: dict[RegimeName, MappingProxyType[int, tuple[str, ...]]] = {}
    law_by_period_by_regime: dict[RegimeName, MappingProxyType[int, CaseLaw]] = {}
    for name, regime_law in laws.items():
        transition = regime_law.transition
        if transition is None:
            coverage[name] = all_periods
            continue
        if isinstance(transition, ByAge):
            # A schedule's cases are case laws; `ByAge` itself refuses any other.
            resolved = cast(
                "Mapping[int, CaseLaw]", transition.resolve(ages=ages).law_by_period
            )
            law_by_period = {
                period: law
                for period, law in resolved.items()
                if period < ages.n_periods - 1
            }
        else:
            law_by_period = dict.fromkeys(all_periods[:-1], transition)
        if source_ages_by_phase is not None:
            # A source supplies a local problem only at ages with an edge.
            edge_ages = {
                age
                for by_source in source_ages_by_phase.values()
                for selected in by_source.get(name, {}).values()
                for age in selected
            }
            law_by_period = {
                period: law
                for period, law in law_by_period.items()
                if ages.exact_values[period] in edge_ages
            }
        coverage[name] = tuple(law_by_period)
        side_by_phase = {
            phase: {
                period: _phase_side(law=law, side=side)
                for period, law in law_by_period.items()
            }
            for phase, side in zip(_PHASES, ("solve", "simulate"), strict=True)
        }
        for phase, side_by_period in side_by_phase.items():
            support[phase][name] = MappingProxyType(
                {
                    period: _edge_support(
                        law=law,
                        source=name,
                        period=period,
                        ages=ages,
                        regime_names=tuple(laws),
                        source_ages=None
                        if source_ages_by_phase is None
                        else source_ages_by_phase[phase],
                    )
                    for period, law in side_by_period.items()
                }
            )
        gated = regime_law.gated_edges
        solution_support = support["solution"][name]
        simulation_support = support["simulation"][name]
        value_reads["solution"][name] = MappingProxyType(
            {
                period: (
                    *_gate_references(gated_edges=gated, targets=targets),
                    *_fallbacks(gated_edges=gated, targets=targets, side="solve"),
                )
                for period, targets in solution_support.items()
            }
        )
        value_reads["simulation"][name] = MappingProxyType(
            {
                period: _gate_references(gated_edges=gated, targets=targets)
                for period, targets in simulation_support.items()
            }
        )
        landings[name] = MappingProxyType(
            {
                period: _fallbacks(gated_edges=gated, targets=targets, side="simulate")
                for period, targets in simulation_support.items()
            }
        )
        law_by_period_by_regime[name] = MappingProxyType(law_by_period)

    return RegimeSchedules(
        coverage_by_regime=MappingProxyType(coverage),
        support_by_phase=MappingProxyType(
            {phase: MappingProxyType(by_regime) for phase, by_regime in support.items()}
        ),
        value_reads_by_phase=MappingProxyType(
            {
                phase: MappingProxyType(by_regime)
                for phase, by_regime in value_reads.items()
            }
        ),
        landings_by_regime=MappingProxyType(landings),
        law_by_period_by_regime=MappingProxyType(law_by_period_by_regime),
        gated_edges_by_regime=MappingProxyType(
            {name: regime_law.gated_edges for name, regime_law in laws.items()}
        ),
    )


def lower_demanded_transitions(
    *,
    schedules: RegimeSchedules,
    declared_transitions: Mapping[RegimeName, object],
    code_by_name: Mapping[str, int],
) -> MappingProxyType[RegimeName, EngineLaw | None]:
    """Lower each regime's law over the periods demand requires, only.

    The solve side is lowered over the regime's valued periods and the
    simulate side over its visited periods. A regime that is valued but never
    visited lowers to its solve side alone, so its realized routing requires
    no runtime argument. A case selected only at undemanded ages contributes no
    cell and no runtime argument. A regime without demanded periods keeps one
    declared law unchanged — its first available one, or its first declared one
    if none is available — so it stays inspectable without a period-dispatched
    union; it is never executed and gets no transition program. The free
    parameters of every declared case keep their slots regardless, since the
    `edges` parameter template is read off the declarations, not off the
    lowered laws.
    """
    lowered: dict[RegimeName, EngineLaw | None] = {}
    visited_periods = schedules.visited_periods_by_regime
    for name, declared in declared_transitions.items():
        if declared is None:
            lowered[name] = None
            continue
        law_by_period = schedules.law_by_period_by_regime[name]
        solve_periods = schedules.coverage_by_regime[name]
        if not solve_periods:
            if not law_by_period:
                # A declared nonterminal law is a case law or a schedule of them.
                law_by_period = {
                    0: cast(
                        "CaseLaw",
                        declared.laws[0] if isinstance(declared, ByAge) else declared,
                    )
                }
            solve_periods = (min(law_by_period),)
        else:
            _fail_if_conflicting_annotations(
                regime_name=name,
                laws=tuple(
                    _phase_side(law=law_by_period[period], side=side)
                    for side, periods in zip(
                        _SIDES,
                        (solve_periods, visited_periods[name]),
                        strict=True,
                    )
                    for period in periods
                ),
            )
        lowered[name] = _lower(
            law_by_period=law_by_period,
            solve_periods=solve_periods,
            simulate_periods=visited_periods[name],
            code_by_name=code_by_name,
        )
    return MappingProxyType(lowered)


def _fail_if_conflicting_annotations(
    *, regime_name: RegimeName, laws: tuple[PhaseLaw, ...]
) -> None:
    """Refuse one argument name annotated differently by two required laws."""
    annotations: dict[str, AnnotationForm] = {}
    for func in _distinct(func for law in laws for func in _law_functions(law)):
        for name, parameter in inspect.signature(func).parameters.items():
            annotation = parameter.annotation
            if annotation is inspect.Parameter.empty:
                continue
            existing = annotations.setdefault(name, annotation)
            if existing != annotation:
                raise ModelInitializationError(
                    f"Argument '{name}' of regime '{regime_name}' is annotated as "
                    f"{_annotation_name(existing)} by one required law and as "
                    f"{_annotation_name(annotation)} by another. An argument name "
                    "has one schema across the laws a regime requires."
                )


def _law_functions(law: PhaseLaw | ProbabilityCell) -> tuple[UserFunction, ...]:
    """The user callables a regime law evaluates to select targets."""
    if isinstance(law, Phased):
        return (*_law_functions(law.solve), *_law_functions(law.simulate))
    if isinstance(law, Mapping):
        return tuple(func for cell in law.values() for func in _law_functions(cell))
    if isinstance(law, DeterministicTransition | StochasticTransition):
        return (law.func,)
    return ()


def _annotation_name(annotation: AnnotationForm) -> str:
    return getattr(annotation, "__name__", repr(annotation))


def _lower(
    *,
    law_by_period: Mapping[int, CaseLaw],
    solve_periods: tuple[int, ...],
    simulate_periods: tuple[int, ...],
    code_by_name: Mapping[str, int],
) -> EngineLaw:
    """One engine law from the per-period laws at the given periods."""
    solve_side = {
        period: _phase_side(law=law_by_period[period], side="solve")
        for period in solve_periods
    }
    simulate_side = {
        period: _phase_side(law=law_by_period[period], side="simulate")
        for period in simulate_periods
    }
    solve_law = _lower_side(law_by_period=solve_side, code_by_name=code_by_name)
    # A schedule without phase variation lowers to one shared engine law.
    if all(
        law is _phase_side(law=law_by_period[period], side="solve")
        for period, law in simulate_side.items()
    ):
        return solve_law
    return Phased(
        solve=solve_law,
        simulate=_lower_side(law_by_period=simulate_side, code_by_name=code_by_name),
    )


def resolve_demand(
    *,
    schedules: RegimeSchedules,
    initial_nodes: frozenset[tuple[object, RegimeName]],
    same_period_refs_by_regime: Mapping[RegimeName, tuple[RegimeName, ...]],
    terminal_regimes: frozenset[RegimeName],
    ages: TimeAxis,
) -> RegimeSchedules:
    """Restrict `schedules` to the problems the starts require.

    Two roles are expanded to a fixed point from the starts:

    - a visit (H) requires the pair's value, its realized successors and gated
      landings as visits, and the values the realized routing reads;
    - a value (S) requires the values its backward problem reads: perceived
      continuation targets, gate references, solve fallbacks and same-period
      references. Realized routes out of a value-only pair are not followed.

    Every required pair must be an available problem: a terminal regime, or a
    nonterminal one with a law at a non-final age. The returned schedules
    cover exactly S; solution support is kept for S and simulation support for
    H.

    Raises:
        ModelInitializationError: If a start or a required read names a pair
            where no problem is available.

    """
    period_by_age: dict[object, int] = {
        age: period for period, age in enumerate(ages.exact_values)
    }
    available = {
        name: frozenset(periods)
        for name, periods in schedules.coverage_by_regime.items()
    }
    work: deque[tuple[bool, int, RegimeName, str]] = deque(
        (True, period_by_age[age], name, "`initial_nodes`")
        for age, name in sorted(initial_nodes, key=repr)
    )
    visited: set[tuple[int, RegimeName]] = set()
    valued: set[tuple[int, RegimeName]] = set()
    support = schedules.support_by_phase
    value_reads = schedules.value_reads_by_phase
    while work:
        physical, period, name, requester = work.popleft()
        done = visited if physical else valued
        if (period, name) in done:
            continue
        required_phase = "simulation" if physical else "solution"
        if name not in terminal_regimes and (
            period not in available[name]
            or not support[required_phase].get(name, {}).get(period, ())
        ):
            raise ModelInitializationError(
                _unavailable_message(
                    requester=requester,
                    name=name,
                    period=period,
                    side=(
                        ""
                        if period not in available[name]
                        else ("simulate " if physical else "solve ")
                    ),
                    ages=ages,
                )
            )
        done.add((period, name))
        here = f"the transition out of ({ages.exact_values[period]}, '{name}')"
        if physical:
            work.append((False, period, name, requester))
            work.extend(
                (True, period + 1, target, here)
                for target in (
                    *support["simulation"].get(name, {}).get(period, ()),
                    *schedules.landings_by_regime.get(name, {}).get(period, ()),
                )
            )
            reads = value_reads["simulation"].get(name, {}).get(period, ())
        else:
            reads = (
                *support["solution"].get(name, {}).get(period, ()),
                *value_reads["solution"].get(name, {}).get(period, ()),
            )
            work.extend(
                (
                    False,
                    period,
                    reference,
                    (
                        f"a same-period reference of ({ages.exact_values[period]}, "
                        f"'{name}')"
                    ),
                )
                for reference in same_period_refs_by_regime.get(name, ())
            )
        targets = (
            support["simulation" if physical else "solution"]
            .get(name, {})
            .get(period, ())
        )
        pair = f"({ages.exact_values[period]}, '{name}')"
        kinds = {
            **dict.fromkeys(reads, f"a fallback of {pair}"),
            **dict.fromkeys(
                _gate_references(
                    gated_edges=schedules.gated_edges_by_regime.get(name, {}),
                    targets=targets,
                ),
                f"a gate reference of {pair}",
            ),
            **dict.fromkeys(targets, here),
        }
        work.extend((False, period + 1, target, kinds[target]) for target in reads)

    return dataclasses.replace(
        schedules,
        coverage_by_regime=MappingProxyType(
            {
                name: tuple(sorted(p for p, n in valued if n == name))
                for name in schedules.coverage_by_regime
            }
        ),
        support_by_phase=MappingProxyType(
            {
                "solution": _restricted(by_regime=support["solution"], keep=valued),
                "simulation": _restricted(
                    by_regime=support["simulation"], keep=visited
                ),
            }
        ),
        valued_nodes=frozenset(valued),
        visited_nodes=frozenset(visited),
    )


def gated_source_periods(
    *, schedules: RegimeSchedules
) -> MappingProxyType[tuple[RegimeName, RegimeName], tuple[int, ...]]:
    """Per `(source, target)` edge, the demanded source periods that use a gate.

    A gate holds wherever its target is reached: solve gates count at valued
    nodes and simulation gates at visited nodes, provided the selected phase
    graph contains the gated edge. A gated edge folds and reads its references
    only after those periods.
    """
    periods: dict[tuple[RegimeName, RegimeName], set[int]] = {}
    for source, gated in schedules.gated_edges_by_regime.items():
        for period in schedules.coverage_by_regime[source]:
            for phase in ("solution", "simulation"):
                selected = (
                    schedules.support_by_phase[phase].get(source, {}).get(period, ())
                )
                for target in gated:
                    if target in selected:
                        periods.setdefault((source, target), set()).add(period)
    return MappingProxyType(
        {edge: tuple(sorted(by_edge)) for edge, by_edge in periods.items()}
    )


def _restricted(
    *,
    by_regime: Mapping[RegimeName, Mapping[int, tuple[str, ...]]],
    keep: set[tuple[int, RegimeName]],
) -> MappingProxyType[RegimeName, MappingProxyType[int, tuple[str, ...]]]:
    """Keep each regime's per-period entries only at the pairs in `keep`."""
    return MappingProxyType(
        {
            name: MappingProxyType(
                {p: t for p, t in by_period.items() if (p, name) in keep}
            )
            for name, by_period in by_regime.items()
        }
    )


def _unavailable_message(
    *, requester: str, name: RegimeName, period: int, side: str, ages: TimeAxis
) -> str:
    """Name why a required pair has no problem.

    A law is bound only at the source ages its edges select, so a nonterminal
    pair before the last age lacks a problem exactly where `edges` declares no
    edge out of it: in either phase when `side` is empty, else in that phase.
    """
    age = ages.exact_values[period]
    reason = (
        "which is nonterminal at the last age: no next age exists"
        if period == ages.n_periods - 1
        else f"where `edges` declares no {side}edge out of '{name}' at that age"
    )
    return f"{requester} requires '{name}' at age {age}, {reason}."


def _gate_references(
    *, gated_edges: Mapping[RegimeName, GatedEdge], targets: tuple[RegimeName, ...]
) -> tuple[RegimeName, ...]:
    """The regimes whose value the gates of the reached `targets` read."""
    return tuple(
        reference.regime
        for target, edge in gated_edges.items()
        if target in targets
        for reference in edge.gate_refs.values()
    )


def _fallbacks(
    *,
    gated_edges: Mapping[RegimeName, GatedEdge],
    targets: tuple[RegimeName, ...],
    side: Side,
) -> tuple[RegimeName, ...]:
    """The gate-closed regimes of the reached gated `targets`, for one phase side."""
    return tuple(
        (route.solve_fallback if side == "solve" else route.simulate_fallback).regime
        for target, edge in gated_edges.items()
        if target in targets
        for route in edge.legs.values()
    )


def resolve_initial_nodes(
    *,
    initial_nodes: UserInitialNodes,
    regime_names: Sequence[RegimeName],
    ages: TimeAxis,
) -> frozenset[tuple[UserAge, RegimeName]]:
    """Normalize `Model(initial_nodes=...)` to the exact admissible start pairs.

    `InitialNodes` explicitly names the model's age or period coordinate.
    Legacy exact pairs and bare selector mappings remain age-only inputs.
    Selector rules contribute the Cartesian product of coordinates and names; all pairs
    are unioned. The result depends on the declaration and clock, never on
    solve coverage.
    """
    entries = _initial_node_entries(
        initial_nodes=initial_nodes, kind=coordinate_kind(ages)
    )
    period_by_age: dict[object, int] = {
        age: period for period, age in enumerate(ages.exact_values)
    }
    permitted: set[tuple[UserAge, RegimeName]] = set()
    for selector, value in entries:
        names = _entry_names(value)
        _fail_if_unknown_entry_regimes(names=names, regime_names=regime_names)
        try:
            _fail_if_invalid_age_selector(selector)
            periods = _select_periods(
                selector=selector,
                ages=ages,
                period_by_age=period_by_age,
            )
        except RegimeInitializationError as error:
            raise ModelInitializationError(str(error)) from error
        if not periods:
            raise ModelInitializationError(
                f"The `initial_nodes` selector {selector!r} selects no "
                f"{coordinate_kind(ages)} of "
                "the model."
            )
        permitted |= {
            (ages.exact_values[period], name) for period in periods for name in names
        }
    return frozenset(permitted)


_INITIAL_NODE_ARITY = 2


def _initial_node_entries(
    *, initial_nodes: UserInitialNodes, kind: str = "age"
) -> Sequence[tuple[object, str | Sequence[str]]]:
    """Normalize exact-pair or selector-mapping entries before grid selection."""
    if isinstance(initial_nodes, InitialNodes):
        selected = initial_nodes.by_age if kind == "age" else initial_nodes.by_period
        if selected is None:
            raise ModelInitializationError(
                f"This model requires InitialNodes(by_{kind}=...)."
            )
        return list(selected.items())
    if kind == "period":
        raise ModelInitializationError(
            "Period models require InitialNodes(by_period=...)."
        )
    if isinstance(initial_nodes, Sequence | AbstractSet) and not isinstance(
        initial_nodes, str
    ):
        if not initial_nodes:
            raise ModelInitializationError(
                "`initial_nodes` must name at least one starting pair."
            )
        entries = [_initial_node_pair(pair) for pair in initial_nodes]
    elif isinstance(initial_nodes, Mapping) and initial_nodes:
        entries = list(initial_nodes.items())
        if any(isinstance(selector, PeriodRange | Periods) for selector, _ in entries):
            raise ModelInitializationError(
                "Age initial-node mappings cannot use period selectors."
            )
    else:
        raise ModelInitializationError(
            "`initial_nodes` must be a nonempty mapping from age selectors to "
            f"regime names, e.g. `{{25: ('single', 'couple')}}`; got "
            f"{initial_nodes!r}."
        )
    return entries


def _initial_node_pair(pair: object) -> tuple[object, RegimeName]:
    """Normalize one legacy exact age pair."""
    if (
        not isinstance(pair, Sequence)
        or isinstance(pair, str)
        or len(pair) != _INITIAL_NODE_ARITY
    ):
        raise ModelInitializationError(
            f"`initial_nodes` must contain (age, regime) pairs; got {pair!r}."
        )
    age, name = pair
    if isinstance(age, AgeRange | PeriodRange | Periods | tuple | range):
        raise ModelInitializationError(
            f"`initial_nodes` pairs require an exact age; got {age!r}."
        )
    if not isinstance(name, str):
        raise ModelInitializationError(
            f"`initial_nodes` pair must name one regime; got {pair!r}."
        )
    return age, name


def _entry_names(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if (
        isinstance(value, Sequence)
        and value
        and all(isinstance(name, str) for name in value)
    ):
        return tuple(value)
    raise ModelInitializationError(
        "`initial_nodes` names regimes by a name or a nonempty sequence of "
        f"names; got {value!r}."
    )


def _fail_if_unknown_entry_regimes(
    *, names: tuple[str, ...], regime_names: Sequence[RegimeName]
) -> None:
    unknown = sorted(set(names) - set(regime_names))
    if unknown:
        raise ModelInitializationError(
            f"`initial_nodes` names unknown regime(s) {unknown}; regimes are "
            f"{sorted(regime_names)}."
        )


# keyword-only-exempt: primary-argument=func
def _with_signature(
    func: UserFunction,
    *,
    names: tuple[str, ...],
    sources: tuple[UserFunction, ...],
) -> UserFunction:
    """Expose `names` as keyword-only arguments annotated as in `sources`.

    The DAG machinery requires one annotation per argument name across a
    model, so each argument keeps the annotation of the law that reads it, and
    `period` is annotated as the period context argument.

    The `sources` are kept on the wrapper so parameter-indexing inspection
    reads the laws the user wrote rather than the wrapper's body.
    """
    annotations: dict[str, AnnotationForm] = {"period": Period}
    for source in sources:
        for name, parameter in inspect.signature(source).parameters.items():
            if parameter.annotation is not inspect.Parameter.empty:
                annotations.setdefault(name, parameter.annotation)
    signature = inspect.Signature(
        [
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=annotations.get(name, inspect.Parameter.empty),
            )
            for name in names
        ]
    )
    # The laws are frozen callable instances, so the attributes go through
    # `object.__setattr__`.
    object.__setattr__(func, "__signature__", signature)
    object.__setattr__(
        func,
        "__annotations__",
        {name: annotations[name] for name in names if name in annotations},
    )
    object.__setattr__(func, "__lcm_sources__", sources)
    if not hasattr(func, "__name__"):
        name = type(func).__name__.lstrip("_").lower()
        object.__setattr__(func, "__name__", name)
        object.__setattr__(func, "__qualname__", name)
    return func


# The lowered laws below are frozen callable instances rather than nested
# functions: the beartype claw memoizes every function it decorates for the rest
# of the process, so a function defined per model build would pin that model's
# laws, and everything they close over, after the model is dropped.


@dataclass(frozen=True, eq=False)
class _Constant:
    """Return `value` as an array; read nothing."""

    value: float | tuple[float, ...]
    """The fixed value or probability vector."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "__signature__", inspect.Signature())
        object.__setattr__(self, "__annotations__", {})
        object.__setattr__(self, "__name__", "constant")

    @no_type_check
    def __call__(self) -> FloatND | IntND:
        return jnp.asarray(self.value)


@dataclass(frozen=True, eq=False)
class _DeclaredExit:
    """A regime-name exit in a declaration view: its target, never evaluated.

    A declaration view is built before regime codes exist, so it carries the
    target's name rather than a code, and evaluating it is an error.
    """

    target: RegimeName
    """The regime the exit names."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "__signature__", inspect.Signature())
        object.__setattr__(self, "__annotations__", {})
        object.__setattr__(self, "__name__", "declared_exit")

    @no_type_check
    def __call__(self) -> NoReturn:
        raise RuntimeError(
            f"The declaration view of an exit into {self.target!r} is for "
            "construction-time checks only and is never evaluated."
        )


def _indicator(
    *, selector: UserFunction, code: int | None, names: tuple[str, ...]
) -> UserFunction:
    """Probability one where a deterministic selector returns `code`."""
    return _with_signature(
        _Indicator(selector=selector, code=code, names=names),
        names=names,
        sources=(selector,),
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _Indicator:
    """Probability one where `selector` returns `code`, zero elsewhere."""

    selector: UserFunction
    """The deterministic regime selector."""
    code: int | None
    """The regime code whose selection has probability one, or `None` in a
    declaration view, which is never evaluated."""
    names: tuple[str, ...]
    """The arguments `selector` reads."""

    @no_type_check
    def __call__(self, **kwargs: EconFunctionArg) -> FloatND:
        if self.code is None:
            raise RuntimeError(
                "A declaration view is for construction-time checks only and is "
                "never evaluated."
            )
        selected = self.selector(**{name: kwargs[name] for name in self.names})
        return jnp.asarray(selected == self.code, dtype=float)


def _period_masked(
    *, cell: UserFunction, periods: tuple[int, ...], names: tuple[str, ...]
) -> UserFunction:
    """A probability cell that is exactly zero outside `periods`."""
    return _with_signature(
        _PeriodMasked(cell=cell, periods=periods, names=names),
        names=_with_period(names),
        sources=(cell,),
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _PeriodMasked:
    """`cell` at `periods`, exactly zero at every other period."""

    cell: UserFunction
    """The probability cell."""
    periods: tuple[int, ...]
    """The periods at which the cell applies."""
    names: tuple[str, ...]
    """The arguments `cell` reads."""

    @no_type_check
    def __call__(self, **kwargs: EconFunctionArg) -> FloatND:
        value = self.cell(**{name: kwargs[name] for name in self.names})
        selected = jnp.isin(kwargs["period"], jnp.asarray(self.periods))
        return jnp.where(selected, value, jnp.zeros_like(value))


def _period_dispatch(
    *,
    cases: tuple[UserFunction, ...],
    case_names: tuple[tuple[str, ...], ...],
    case_by_period: tuple[int, ...],
) -> UserFunction:
    """Evaluate the case callable selected for the current period."""
    names = tuple(sorted({name for names in case_names for name in names}))
    return _with_signature(
        _PeriodDispatch(
            cases=cases, case_names=case_names, case_by_period=case_by_period
        ),
        names=_with_period(names),
        sources=cases,
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _PeriodDispatch:
    """Evaluate the case `case_by_period` selects for the current period."""

    cases: tuple[UserFunction, ...]
    """The case callables."""
    case_names: tuple[tuple[str, ...], ...]
    """The arguments each case reads."""
    case_by_period: tuple[int, ...]
    """The position in `cases` selected at each period."""

    @no_type_check
    def __call__(self, **kwargs: EconFunctionArg) -> FloatND | IntND:
        # Every case is evaluated and the selected one kept. Each case is a
        # declared law of this regime evaluated on the same arguments, so an
        # unselected case can at worst produce a value that is discarded. The
        # regime law is never differentiated, so a discarded non-finite value
        # cannot reach a gradient.
        values = [
            jnp.asarray(case(**{name: kwargs[name] for name in names}))
            for case, names in zip(self.cases, self.case_names, strict=True)
        ]
        index = jnp.asarray(self.case_by_period)[kwargs["period"]]
        result = values[-1]
        for position in range(len(values) - 2, -1, -1):
            result = jnp.where(index == position, values[position], result)
        return result


def _with_period(names: tuple[str, ...]) -> tuple[str, ...]:
    return ("period", *(name for name in names if name != "period"))


def _argument_names(func: UserFunction) -> tuple[str, ...]:
    return tuple(inspect.signature(func).parameters)


def _phase_side[L](*, law: L | Phased[L, L], side: Side) -> L:
    if not isinstance(law, Phased):
        return law
    return law.solve if side == "solve" else law.simulate


def _edge_support(
    *,
    # Any value: the graph binder passes laws a `ByAge` resolves, which
    # `ResolvedSchedule` types as `object`.
    law: object,
    source: RegimeName,
    period: int,
    ages: TimeAxis,
    regime_names: tuple[RegimeName, ...],
    source_ages: Mapping[RegimeName, Mapping[RegimeName, frozenset[UserAge]]] | None,
) -> tuple[str, ...]:
    """The targets one phase side of a law declares that an edge admits.

    `source_ages` maps source to target to the source ages its edges select;
    `None` admits every declared target.
    """
    return tuple(
        target
        for target in _declared_support(law=law, regime_names=regime_names)
        if source_ages is None
        or ages.exact_values[period]
        in source_ages.get(source, {}).get(target, frozenset())
    )


def _declared_support(
    *, law: object, regime_names: tuple[RegimeName, ...]
) -> tuple[str, ...]:
    """The targets one nonterminal law declares."""
    if isinstance(law, str):
        support: tuple[str, ...] = (law,)
    elif isinstance(
        law, _SupportedDeterministicTransition | _SupportedStochasticTransition
    ):
        support = tuple(law.targets)
    elif isinstance(law, Mapping):
        support = tuple(law)
    else:
        support = ()
    _fail_if_unknown_targets(targets=support, regime_names=regime_names)
    return support


def _fail_if_unknown_targets(
    *, targets: tuple[RegimeName, ...], regime_names: tuple[RegimeName, ...]
) -> None:
    unknown = sorted(set(targets) - set(regime_names))
    if unknown:
        raise ModelInitializationError(
            f"A regime transition names unknown target regime(s) {unknown}; "
            f"regimes are {sorted(regime_names)}."
        )


def _lower_side(
    *, law_by_period: Mapping[int, PhaseLaw], code_by_name: Mapping[str, int]
) -> EnginePhaseLaw:
    """One phase's law as a single period-independent engine law."""
    distinct = _distinct(law_by_period.values())
    if len(distinct) == 1:
        return _plain_law(law=distinct[0], code_by_name=code_by_name)
    return _union_phase_law(
        laws=distinct,
        code_by_name=code_by_name,
        mask={
            id(law): tuple(p for p, x in law_by_period.items() if x is law)
            for law in distinct
        },
        law_by_period=law_by_period,
    )


def _plain_law(
    *, law: PhaseLaw, code_by_name: Mapping[str, int] | None
) -> EnginePhaseLaw:
    """A single nonterminal law in the engine vocabulary.

    Without `code_by_name` (a declaration view) a regime name becomes an exit
    that names its target and is never evaluated.
    """
    if isinstance(law, str):
        if code_by_name is None:
            return _DeclaredExit(target=law)
        return _Constant(value=_code(name=law, code_by_name=code_by_name))
    if isinstance(law, DeterministicTransition):
        return law.func
    return law


def _union_law(
    *,
    laws: tuple[CaseLaw, ...],
    code_by_name: Mapping[str, int] | None,
    mask: Mapping[int, tuple[int, ...]] | None,
) -> EngineLaw:
    """Several laws as one, each phase side separately where any is `Phased`."""
    phase_laws = tuple(law for law in laws if not isinstance(law, Phased))
    if len(phase_laws) == len(laws):
        return _union_phase_law(laws=phase_laws, code_by_name=code_by_name, mask=mask)
    solve = _union_phase_law(
        laws=_distinct(_phase_side(law=law, side="solve") for law in laws),
        code_by_name=code_by_name,
        mask=None,
    )
    simulate = _union_phase_law(
        laws=_distinct(_phase_side(law=law, side="simulate") for law in laws),
        code_by_name=code_by_name,
        mask=None,
    )
    return Phased(solve=solve, simulate=simulate)


def _union_phase_law(
    *,
    laws: tuple[PhaseLaw, ...],
    code_by_name: Mapping[str, int] | None,
    mask: Mapping[int, tuple[int, ...]] | None,
    law_by_period: Mapping[int, PhaseLaw] | None = None,
) -> EnginePhaseLaw:
    """One phase's laws as one: dispatch deterministic laws, merge stochastic cells."""
    stochastic = [
        law for law in laws if isinstance(law, Mapping | StochasticTransition)
    ]
    if not stochastic:
        return _deterministic_dispatch(
            laws=laws, code_by_name=code_by_name, law_by_period=law_by_period
        )
    if any(isinstance(law, StochasticTransition) for law in stochastic):
        return _vector_union(
            laws=laws, code_by_name=code_by_name, law_by_period=law_by_period
        )
    return _mapping_union(laws=laws, code_by_name=code_by_name, mask=mask)


def _deterministic_dispatch(
    *,
    laws: tuple[PhaseLaw, ...],
    code_by_name: Mapping[str, int] | None,
    law_by_period: Mapping[int, PhaseLaw] | None,
) -> EnginePhaseLaw:
    cases = tuple(_plain_law(law=law, code_by_name=code_by_name) for law in laws)
    if law_by_period is None:
        return cases[0]
    # A deterministic law lowers to a callable: a regime name to a constant code,
    # a `DeterministicTransition` to its function.
    return _dispatch(
        cases=cast("tuple[UserFunction, ...]", cases),
        laws=laws,
        law_by_period=law_by_period,
    )


def _dispatch(
    *,
    cases: tuple[UserFunction, ...],
    laws: tuple[PhaseLaw, ...],
    law_by_period: Mapping[int, PhaseLaw],
) -> UserFunction:
    n_periods = max(law_by_period) + 2
    position = {id(law): index for index, law in enumerate(laws)}
    return _period_dispatch(
        cases=cases,
        case_names=tuple(_argument_names(case) for case in cases),
        case_by_period=tuple(
            position[id(law_by_period[period])] if period in law_by_period else 0
            for period in range(n_periods)
        ),
    )


def _vector_union(
    *,
    laws: tuple[PhaseLaw, ...],
    code_by_name: Mapping[str, int] | None,
    law_by_period: Mapping[int, PhaseLaw] | None,
) -> StochasticTransition:
    """Several vector laws and exits as one vector `StochasticTransition`."""
    if any(isinstance(law, Mapping) for law in laws):
        raise ModelInitializationError(
            "A schedule mixes a vector `StochasticTransition` with a per-target "
            "mapping. Use one form for every case: write the vector law's "
            "targets as a per-target mapping, or every case as a vector law."
        )
    targets = tuple(
        dict.fromkeys(target for law in laws for target in _law_targets(law))
    )
    if law_by_period is None or code_by_name is None:
        return next(law for law in laws if isinstance(law, StochasticTransition))
    n_regimes = max(code_by_name.values()) + 1
    cases = tuple(
        _vector_case(law=law, n_regimes=n_regimes, code_by_name=code_by_name)
        for law in laws
    )
    return _SupportedStochasticTransition(
        func=_dispatch(cases=cases, laws=laws, law_by_period=law_by_period),
        targets=targets,
    )


def _vector_case(
    *, law: PhaseLaw, n_regimes: int, code_by_name: Mapping[str, int]
) -> UserFunction:
    if isinstance(law, StochasticTransition):
        return law.func
    if not isinstance(law, str):
        raise ModelInitializationError(
            f"A schedule mixes a vector `StochasticTransition` with {law!r}. Pair a "
            "vector law only with regime-name exits, or write every case as a "
            "per-target mapping."
        )
    one_hot = [0.0] * n_regimes
    one_hot[_code(name=law, code_by_name=code_by_name)] = 1.0
    return _Constant(value=tuple(one_hot))


def _mapping_union(
    *,
    laws: tuple[PhaseLaw, ...],
    code_by_name: Mapping[str, int] | None,
    mask: Mapping[int, tuple[int, ...]] | None,
) -> MappingProxyType[RegimeName, ProbabilityCell]:
    """Per-target mappings and exits merged; each target keeps one cell."""
    cells_by_target: dict[RegimeName, list[tuple[PhaseLaw, ProbabilityCell]]] = {}
    for law in laws:
        for target, cell in _law_cells(law=law, code_by_name=code_by_name).items():
            cells_by_target.setdefault(target, []).append((law, cell))
    merged: dict[RegimeName, ProbabilityCell] = {}
    all_covered = (
        None
        if mask is None
        else tuple(sorted(period for law in laws for period in mask[id(law)]))
    )
    for target, entries in cells_by_target.items():
        distinct_cells = _distinct(cell for _, cell in entries)
        if len(distinct_cells) > 1:
            if mask is None:
                merged[target] = distinct_cells[0]
                continue
            periods_by_cell = {
                id(cell): tuple(
                    period
                    for law, same in entries
                    if same is cell
                    for period in mask[id(law)]
                )
                for cell in distinct_cells
            }
            merged[target] = StochasticTransition(
                func=_period_sum(
                    tuple(
                        _masked(cell=cell, periods=periods_by_cell[id(cell)])
                        for cell in distinct_cells
                    )
                )
            )
            continue
        cell = distinct_cells[0]
        if mask is None:
            merged[target] = cell
            continue
        periods = tuple(
            sorted(period for law, _ in entries for period in mask[id(law)])
        )
        merged[target] = (
            cell if periods == all_covered else _masked_cell(cell=cell, periods=periods)
        )
    return MappingProxyType(merged)


def _masked(*, cell: ProbabilityCell, periods: tuple[int, ...]) -> UserFunction:
    # Validation admits `Phased` only around a whole law, never as one target's cell.
    func = (
        cell.func
        if isinstance(cell, StochasticTransition)
        else cast("UserFunction", cell)
    )
    return _period_masked(
        cell=func, periods=tuple(sorted(periods)), names=_argument_names(func)
    )


def _period_sum(parts: tuple[UserFunction, ...]) -> UserFunction:
    """Sum of period-masked cells whose periods never overlap."""
    part_names = tuple(_argument_names(part) for part in parts)
    names = tuple(sorted({name for names in part_names for name in names}))
    return _with_signature(
        _PeriodSum(parts=parts, part_names=part_names),
        names=_with_period(names),
        sources=parts,
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _PeriodSum:
    """Sum of `parts`, each called with the arguments it reads."""

    parts: tuple[UserFunction, ...]
    """The period-masked cells."""
    part_names: tuple[tuple[str, ...], ...]
    """The arguments each part reads."""

    @no_type_check
    def __call__(self, **kwargs: EconFunctionArg) -> FloatND:
        values = [
            part(**{name: kwargs[name] for name in names})
            for part, names in zip(self.parts, self.part_names, strict=True)
        ]
        return sum(values[1:], values[0])


def _masked_cell(
    *, cell: ProbabilityCell, periods: tuple[int, ...]
) -> StochasticTransition:
    """Zero a cell outside `periods`."""
    return StochasticTransition(func=_masked(cell=cell, periods=periods))


def _law_cells(
    *, law: PhaseLaw, code_by_name: Mapping[str, int] | None
) -> Mapping[RegimeName, ProbabilityCell]:
    if isinstance(law, Mapping):
        return law
    if isinstance(law, str):
        return {law: StochasticTransition(func=_Constant(value=1.0))}
    if isinstance(law, DeterministicTransition):
        names = _argument_names(law.func)
        return {
            target: StochasticTransition(
                func=_indicator(
                    selector=law.func,
                    code=(
                        None
                        if code_by_name is None
                        else _code(name=target, code_by_name=code_by_name)
                    ),
                    names=names,
                )
            )
            for target in _law_targets(law)
        }
    raise ModelInitializationError(
        f"A schedule case {law!r} is not a nonterminal regime law. Use a regime "
        "name, a `DeterministicTransition` or a per-target mapping, and declare "
        "its destinations in `Model(edges=...)`."
    )


def _law_targets(law: PhaseLaw) -> tuple[str, ...]:
    if isinstance(law, str):
        return (law,)
    if isinstance(
        law, _SupportedDeterministicTransition | _SupportedStochasticTransition
    ):
        return tuple(law.targets)
    return ()


def _code(*, name: str, code_by_name: Mapping[str, int]) -> int:
    if name not in code_by_name:
        raise ModelInitializationError(
            f"A regime transition names unknown target regime {name!r}; regimes "
            f"are {sorted(code_by_name)}."
        )
    return code_by_name[name]


def _distinct[T](values: Iterable[T]) -> tuple[T, ...]:
    """Distinct objects by identity, in first-seen order."""
    return tuple({id(value): value for value in values}.values())
