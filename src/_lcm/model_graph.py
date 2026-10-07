"""Resolve explicit model topology and bind it to the numerical transition laws."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from _lcm.reachability import ModelReachability, PhaseReachability
from _lcm.regime_building.fixed_regime_support import prune_fixed_regime_support
from _lcm.regime_building.schedules import (
    RegimeSchedules,
    _Constant,
    _edge_support,
    _fallbacks,
    _phase_side,
    resolve_demand,
    resolve_regime_schedules,
)
from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
    _SupportedStochasticTransition,
)
from _lcm.regime_law import RegimeLaw, RegimeLaws, bind_regime_law
from _lcm.user_regime_validation import validate_regimes
from lcm.ages import AgeGrid
from lcm.collective import ValueDependentTransition
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.phased import Phased
from lcm.regime import Regime
from lcm.transition import (
    AgeSelector,
    ByAge,
    DeterministicTransition,
    StochasticTransition,
    Transition,
    _fail_if_invalid_age_selector,
    _select_periods,
)
from lcm.typing import RegimeName, UserAge, UserFunction, UserParams

type ResolvedEdges = MappingProxyType[
    RegimeName, MappingProxyType[RegimeName, frozenset[UserAge]]
]
type Edge = tuple[object, RegimeName, RegimeName]


@dataclass(frozen=True, kw_only=True)
class GraphEdges:
    """Store immutable destination-to-source-age edges for each phase."""

    solve: ResolvedEdges
    """Declared perceived edges, resolved to exact source ages."""
    simulate: ResolvedEdges
    """Declared realized edges, resolved to exact source ages."""


@dataclass(frozen=True, kw_only=True)
class ModelGraph:
    """Inspect declared topology, effective support and demand without rebuilding."""

    edges: GraphEdges
    """All declared edges, including edges removed by fixed-zero proofs."""
    initial_nodes: frozenset[tuple[object, RegimeName]]
    """Admissible exact age-regime starting pairs."""
    reachability: ModelReachability
    """Effective phase graphs and their demanded nodes."""
    pruned_edges: MappingProxyType[str, MappingProxyType[Edge, str]]
    """Per phase, fixed-zero edges and the reason they were removed."""
    laws: RegimeLaws
    """Each source regime's bound law, by regime name.

    The law the solver and simulator evaluate: graph-bound, pruned of
    fixed-zero cells and lowered to the demanded periods. A regime is terminal
    (`laws[name].terminal`) when it has no outgoing edges, and the gated edges
    of a source are those its law's `ValueDependentTransition` cells declare
    (`laws[name].gated_edges`).
    """

    @property
    def solution(self) -> PhaseReachability:
        """Return the effective perceived graph."""
        return self.reachability.solution

    @property
    def simulation(self) -> PhaseReachability:
        """Return the effective realized graph."""
        return self.reachability.simulation

    @property
    def nodes(self) -> frozenset[tuple[object, RegimeName]]:
        """Return every valued age-regime pair."""
        return self.reachability.nodes

    @property
    def visited_nodes(self) -> frozenset[tuple[object, RegimeName]]:
        """Return every physically reachable age-regime pair."""
        return self.reachability.visited_nodes


@dataclass(frozen=True, kw_only=True)
class DroppedCells:
    """Targets a source's law names at one age without a declared edge."""

    age: object
    """The source age."""
    targets: tuple[RegimeName, ...]
    """The law's targets that have no edge out of the source at that age."""


# Keyed by `(source, period)`.
CellsWithoutEdges = MappingProxyType[tuple[RegimeName, int], DroppedCells]


@dataclass(frozen=True, kw_only=True)
class GraphPreparation:
    """Keep the graph proof and its numerical declarations together."""

    regimes: MappingProxyType[RegimeName, Regime]
    """Regimes without the declarations toward fixed-zero pruned edges."""
    laws: RegimeLaws
    """Graph-bound laws after fixed-zero probability pruning."""
    schedules: RegimeSchedules
    """Graph support restricted to physical and valued demand."""
    declarations: MappingProxyType[RegimeName, object]
    """Graph-bound kernels before fixed-zero pruning, for dormant inspection."""
    consumed_param_keys: frozenset[str]
    """Fixed leaves consumed by a removed zero cell."""
    removed_edge_reads: MappingProxyType[
        RegimeName, MappingProxyType[str, tuple[RegimeName, ...]]
    ]
    """Per regime, each variable read across a removed zero edge and the edges'
    targets, for error messages only."""
    pruned_edges: MappingProxyType[str, MappingProxyType[Edge, str]]
    """Fixed-zero primary edges and their proof reason."""
    cells_without_edges: CellsWithoutEdges
    """Law cells binding dropped because the graph declares no edge for them."""


@contextmanager
def naming_cells_without_edges(
    cells_without_edges: CellsWithoutEdges,
) -> Iterator[None]:
    """Name the missing edges when a law's mass falls short because of them.

    A law cell toward a target the graph gives no edge at that age is dropped when
    the law is bound, so its probability mass is lost. The unit-mass check then
    fails on the law, while the cause is the graph. When the failing source and
    period have dropped cells, the error names each `(age, source -> target)` cell;
    otherwise it is raised unchanged.
    """
    try:
        yield
    except InvalidRegimeTransitionProbabilitiesError as error:
        dropped = cells_without_edges.get(
            getattr(error, "unit_mass_violation", None)  # ty: ignore[invalid-argument-type]
        )
        if dropped is None:
            raise
        source = error.unit_mass_violation[0]  # ty: ignore[unresolved-attribute]
        cells = ", ".join(
            f"(age {dropped.age}, '{source}' -> '{target}')"
            for target in dropped.targets
        )
        msg = (
            f"{error.mass_detail}\n"  # ty: ignore[unresolved-attribute]
            f"The regime law of '{source}' at age {dropped.age} has cells for "
            f"{cells}, but the graph declares no edge for them. Those cells are "
            "dropped, so their probability mass is missing. Declare these edges "
            "in `Model(edges=...)`, or give these targets no cell at that age."
        )
        raise InvalidRegimeTransitionProbabilitiesError(msg) from error


def prepare_graph(
    *,
    regimes: Mapping[RegimeName, Regime],
    laws: RegimeLaws,
    edges: GraphEdges,
    ages: AgeGrid,
    initial_nodes: frozenset[tuple[object, RegimeName]],
    fixed_params: UserParams,
) -> GraphPreparation:
    """Bind laws, prove fixed zeros, and close physical and value demand."""
    bound, cells_without_edges = bind_graph_support(laws=laws, edges=edges, ages=ages)
    declarations = MappingProxyType(
        {name: law.transition for name, law in bound.items()}
    )
    source_ages = {"solution": edges.solve, "simulation": edges.simulate}
    fixed_support = prune_fixed_regime_support(
        user_regimes=regimes, laws=bound, fixed_params=fixed_params
    )
    after = resolve_regime_schedules(
        laws=fixed_support.laws,
        ages=ages,
        source_ages_by_phase=source_ages,
    )
    schedules = resolve_demand(
        schedules=after,
        initial_nodes=initial_nodes,
        same_period_refs_by_regime={
            name: tuple(ref.regime for ref in regime.same_period_refs.values())
            for name, regime in fixed_support.user_regimes.items()
        },
        terminal_regimes=frozenset(name for name, law in laws.items() if law.terminal),
        ages=ages,
    )
    return GraphPreparation(
        regimes=fixed_support.user_regimes,
        laws=fixed_support.laws,
        schedules=schedules,
        declarations=declarations,
        consumed_param_keys=fixed_support.consumed_param_keys,
        removed_edge_reads=fixed_support.removed_edge_reads,
        pruned_edges=fixed_zero_edge_reasons(
            bound=bound, edges=edges, after=after, ages=ages
        ),
        cells_without_edges=cells_without_edges,
    )


def bind_edge_laws(
    *, edges: object, regimes: Mapping[RegimeName, Regime], ages: AgeGrid
) -> tuple[RegimeLaws, GraphEdges]:
    """Bind each regime's law from `Model(edges=...)` and resolve its support.

    Per phase, a source's law at each source age with outgoing edges is:

    - the only destination, where the age has exactly one outgoing edge and the
      source is a plain `{target: selector}` mapping or its `Transition` law
      selects nothing there, as a probability-one cell where the other phase's
      law at that age is a per-target probability mapping;
    - the `Transition` law (its `ByAge` case, its `Phased` side), otherwise.

    A source age with several outgoing edges and no law is rejected, and so is
    a `Transition` on a source whose every age has at most one outgoing edge,
    unless its law carries a `ValueDependentTransition` gate. A regime with no
    outgoing edge in either phase is terminal. Each regime is validated against
    its bound law.

    Returns:
        Each regime's bound law, and both phases' edges with their selectors
        snapshotted as exact source ages.
    """
    declared = (
        {"solve": edges.solve, "simulate": edges.simulate}
        if isinstance(edges, Phased)
        else {"solve": edges, "simulate": edges}
    )
    structural = {
        side: (
            {
                source: (decl.targets if isinstance(decl, Transition) else decl)
                for source, decl in phase.items()
            }
            if isinstance(phase, Mapping)
            else phase
        )
        for side, phase in declared.items()
    }
    resolved = {
        side: _resolve_edges(edges=structural[side], regimes=regimes, ages=ages)
        for side in ("solve", "simulate")
    }
    bound: dict[RegimeName, object] = {}
    for name in regimes:
        laws_by_side = {}
        for side in ("solve", "simulate"):
            declaration = cast("Mapping[str, object]", declared[side]).get(name)
            if isinstance(declaration, Transition):
                laws_by_side[side] = _transition_laws(
                    source=name,
                    transition=declaration,
                    resolved=resolved[side].get(name, {}),
                    ages=ages,
                    side=side,
                )
            else:
                laws_by_side[side] = _single_destination_laws(
                    source=name,
                    resolved=resolved[side].get(name, {}),
                    ages=ages,
                    side=side,
                )
        bound[name] = _combined_law(laws_by_side=laws_by_side, ages=ages)
    laws = MappingProxyType({name: bind_regime_law(bound[name]) for name in regimes})
    validate_regimes(regimes=regimes, laws=laws)
    return laws, GraphEdges(solve=resolved["solve"], simulate=resolved["simulate"])


def _targets_by_period(
    *, resolved: Mapping[RegimeName, frozenset[UserAge]], ages: AgeGrid
) -> dict[int, tuple[RegimeName, ...]]:
    """Return the destinations of each non-final source period that has any."""
    targets = {
        period: tuple(
            target for target, selected in resolved.items() if age in selected
        )
        for period, age in enumerate(ages.exact_values[:-1])
    }
    return {period: names for period, names in targets.items() if names}


def _single_destination_laws(
    *,
    source: RegimeName,
    resolved: Mapping[RegimeName, frozenset[UserAge]],
    ages: AgeGrid,
    side: str,
) -> dict[int, object]:
    """Read a law-free source's law off its edges: the only destination per age."""
    laws: dict[int, object] = {}
    for period, targets in _targets_by_period(resolved=resolved, ages=ages).items():
        if len(targets) > 1:
            raise ModelInitializationError(
                f"'{source}' has {len(targets)} outgoing {side} edges at age "
                f"{ages.exact_values[period]} ({', '.join(targets)}) and no law to "
                f"choose among them. Declare `edges['{source}']` as "
                "`Transition(targets=..., law=...)`."
            )
        laws[period] = targets[0]
    return laws


def _transition_laws(
    *,
    source: RegimeName,
    transition: Transition,
    resolved: Mapping[RegimeName, frozenset[UserAge]],
    ages: AgeGrid,
    side: str,
) -> dict[int, object]:
    """Select a `Transition`'s law at each source period with outgoing edges."""
    targets_by_period = _targets_by_period(resolved=resolved, ages=ages)
    law = transition.law
    if isinstance(law, Phased):
        law = getattr(law, side)
    if all(len(targets) <= 1 for targets in targets_by_period.values()) and not (
        _carries_gate(law)
    ):
        raise ModelInitializationError(
            f"Every source age of '{source}' has at most one outgoing {side} edge, "
            f"so the graph is its law. Declare `edges['{source}']` as the plain "
            "`{target: source_ages}` mapping instead of a `Transition`."
        )
    selected = (
        law.resolve(ages).law_by_period
        if isinstance(law, ByAge)
        else dict.fromkeys(targets_by_period, law)
    )
    laws: dict[int, object] = {}
    for period, targets in targets_by_period.items():
        if period in selected:
            case = selected[period]
            laws[period] = getattr(case, side) if isinstance(case, Phased) else case
        elif len(targets) == 1:
            laws[period] = targets[0]
        else:
            raise ModelInitializationError(
                f"The law of '{source}' selects nothing at age "
                f"{ages.exact_values[period]}, where its {side} edges lead to "
                f"{', '.join(targets)}. Give the `Transition` law a case there."
            )
    return laws


def _carries_gate(law: object) -> bool:
    """Whether a law declares a `ValueDependentTransition` cell at any age."""
    laws = law.laws if isinstance(law, ByAge) else (law,)
    return any(
        isinstance(cell, ValueDependentTransition)
        for one in laws
        for side in ((one.solve, one.simulate) if isinstance(one, Phased) else (one,))
        if isinstance(side, Mapping)
        for cell in side.values()
    )


def _combined_law(
    *, laws_by_side: Mapping[str, Mapping[int, object]], ages: AgeGrid
) -> object:
    """Combine per-phase, per-period laws into one regime law.

    `None` when neither phase has an outgoing edge (terminal). One law object
    shared by every period is returned as is; otherwise a `ByAge` with one case
    per source age, `Phased` where the two phases differ.
    """
    solve, simulate = laws_by_side["solve"], laws_by_side["simulate"]
    periods = sorted(set(solve) | set(simulate))
    if not periods:
        return None
    pairs: dict[tuple[int, int], object] = {}
    combined: dict[int, object] = {}
    for period in periods:
        solve_law = solve.get(period, simulate.get(period))
        simulate_law = simulate.get(period, solve_law)
        if solve_law is simulate_law or (
            isinstance(solve_law, str) and solve_law == simulate_law
        ):
            combined[period] = solve_law
            continue
        key = (id(solve_law), id(simulate_law))
        if key not in pairs:
            pairs[key] = Phased(
                solve=_lottery_if_paired(law=solve_law, other=simulate_law),
                simulate=_lottery_if_paired(law=simulate_law, other=solve_law),
            )
        combined[period] = pairs[key]
    laws = list(combined.values())
    first = laws[0]
    if all(law is first or (isinstance(first, str) and law == first) for law in laws):
        return first
    return ByAge(
        cases={ages.exact_values[period]: law for period, law in combined.items()}
    )


def _lottery_if_paired(*, law: object, other: object) -> object:
    """A lone edge as a probability-one cell when the other phase has a mapping.

    The graph is the law of a lone edge; across phases it takes the form of the
    other phase's per-target probability mapping so the two sides match.
    """
    if isinstance(law, str) and isinstance(other, Mapping):
        return MappingProxyType({law: StochasticTransition(func=_Constant(value=1.0))})
    return law


def bind_graph_support(
    *, laws: RegimeLaws, edges: GraphEdges, ages: AgeGrid
) -> tuple[RegimeLaws, CellsWithoutEdges]:
    """Bind graph-selected probability cells and selector support per source age.

    Numerical kernels supply no support. The private tags used by scheduling
    and lowering are derived exclusively from the resolved graph.

    Returns:
        The graph-bound laws, and per `(source, period)` the targets whose law
        cells binding dropped because no edge leads to them at that age.
    """
    regime_names = tuple(laws)
    result: dict[RegimeName, RegimeLaw] = {}
    dropped: dict[tuple[RegimeName, int], DroppedCells] = {}
    for source, source_law in laws.items():
        if source_law.terminal:
            result[source] = source_law
            continue
        transition = source_law.transition
        kernels = (
            transition.resolve(ages).law_by_period
            if isinstance(transition, ByAge)
            else dict.fromkeys(range(ages.n_periods), transition)
        )
        _fail_if_kernel_extends_graph(kernels=kernels, source=source, edges=edges)
        cases: list[tuple[AgeSelector, object]] = []
        cache: dict[tuple[int, tuple[str, ...], str], object] = {}
        for period, age in enumerate(ages.exact_values[:-1]):
            targets = {
                side: tuple(
                    target
                    for target in regime_names
                    if age in getattr(edges, side).get(source, {}).get(target, ())
                )
                for side in ("solve", "simulate")
            }
            if not any(targets.values()):
                continue
            if period not in kernels:
                raise ModelInitializationError(
                    f"Graph edges out of ({age}, '{source}') have no transition "
                    "kernel selected by `ByAge`."
                )
            kernel = kernels[period]
            sides: dict[str, object] = {}
            for side in ("solve", "simulate"):
                law = getattr(kernel, side) if isinstance(kernel, Phased) else kernel
                # Ordinary broadcast kernels must retain identity across phases.
                # Gated cells additionally resolve phase-specific fallbacks.
                phase_key = (
                    side
                    if isinstance(law, Mapping)
                    and any(
                        isinstance(cell, ValueDependentTransition)
                        for cell in law.values()
                    )
                    else "shared"
                )
                key = (id(law), targets[side], phase_key)
                if key not in cache:
                    cache[key] = _bind_law(
                        law=law,
                        targets=targets[side],
                        source=source,
                        age=age,
                        side=side,
                        regime_names=regime_names,
                    )
                sides[side] = cache[key]
                if isinstance(law, Mapping) and targets[side]:
                    missing = tuple(
                        target
                        for target in regime_names
                        if target in law and target not in targets[side]
                    )
                    if missing:
                        previous = dropped.get((source, period))
                        dropped[(source, period)] = DroppedCells(
                            age=age,
                            targets=tuple(
                                dict.fromkeys(
                                    (*(previous.targets if previous else ()), *missing)
                                )
                            ),
                        )
            cases.append(
                (
                    age,
                    sides["solve"]
                    if sides["solve"] is sides["simulate"]
                    else Phased(solve=sides["solve"], simulate=sides["simulate"]),
                )
            )
        # An undemanded source remains inspectable but supplies no local problem.
        bound = (
            ByAge(cases=dict(cases))
            if cases
            else _SupportedStochasticTransition(
                func=_Constant(value=(0.0,) * len(regime_names)),
                targets=(),
            )
        )
        result[source] = bind_regime_law(bound)
    return MappingProxyType(result), MappingProxyType(dropped)


def fixed_zero_edge_reasons(
    *,
    bound: RegimeLaws,
    edges: GraphEdges,
    after: RegimeSchedules,
    ages: AgeGrid,
) -> MappingProxyType[str, MappingProxyType[Edge, str]]:
    """Record only support removed by the fixed-probability proof stage.

    `bound` holds the graph-bound laws before the proof and `after` the
    schedules resolved from the pruned laws. A bound law is a `ByAge` with one
    case per source age that has an edge, so each case's edge support is
    compared directly with what survives in `after`.
    """
    names = tuple(bound)
    reasons: dict[str, MappingProxyType[Edge, str]] = {}
    for side, phase in (("solve", "solution"), ("simulate", "simulation")):
        removed: dict[Edge, str] = {}
        for source, source_law in bound.items():
            transition = source_law.transition
            if not isinstance(transition, ByAge):
                continue
            kept = after.support_by_phase[phase].get(source, {})
            for period, law in transition.resolve(ages).law_by_period.items():
                removed |= {
                    (ages.exact_values[period], source, target): (
                        "fixed_zero_probability"
                    )
                    for target in _edge_support(
                        law=_phase_side(law=law, side=side),
                        source=source,
                        period=period,
                        ages=ages,
                        regime_names=names,
                        source_ages=getattr(edges, side),
                    )
                    if target not in kept.get(period, ())
                }
        reasons[side] = MappingProxyType(removed)
    return MappingProxyType(reasons)


def _resolve_edges(
    *, edges: object, regimes: Mapping[RegimeName, Regime], ages: AgeGrid
) -> ResolvedEdges:
    if not isinstance(edges, Mapping):
        raise ModelInitializationError(
            "`edges` must map source regimes to destination-to-source-age "
            f"selectors, or be a `Phased` of those mappings; got {edges!r}."
        )
    period_by_age: dict[object, int] = {
        age: period for period, age in enumerate(ages.exact_values)
    }
    resolved: dict[str, MappingProxyType[str, frozenset[UserAge]]] = {}
    for source, destinations in edges.items():
        if not isinstance(source, str) or source not in regimes:
            raise ModelInitializationError(
                f"Graph names unknown source regime {source!r}."
            )
        if not isinstance(destinations, Mapping):
            raise ModelInitializationError(
                f"Graph source '{source}' must map destinations "
                "to source-age selectors."
            )
        selected: dict[str, frozenset[UserAge]] = {}
        for target, selector in destinations.items():
            if not isinstance(target, str) or target not in regimes:
                raise ModelInitializationError(
                    f"Graph names unknown target regime {target!r}."
                )
            selected[target] = _selected_source_ages(
                selector=selector,
                edge=f"'{source}' → '{target}'",
                ages=ages,
                period_by_age=period_by_age,
            )
        resolved[source] = MappingProxyType(selected)
    return MappingProxyType(resolved)


def _selected_source_ages(
    *,
    selector: object,
    edge: str,
    ages: AgeGrid,
    period_by_age: Mapping[object, int],
) -> frozenset[UserAge]:
    """Resolve one edge's selector to the source ages at which it can fire."""
    try:
        _fail_if_invalid_age_selector(selector)
        periods = _select_periods(
            selector=selector, ages=ages, period_by_age=period_by_age
        )
    except RegimeInitializationError as error:
        raise ModelInitializationError(str(error)) from error
    if not periods:
        raise ModelInitializationError(
            f"Graph selector {selector!r} for {edge} selects no model age."
        )
    if set(periods) == {ages.n_periods - 1}:
        raise ModelInitializationError(
            f"Graph selector {selector!r} for {edge} selects only the final age "
            f"{ages.exact_values[-1]}, where no transition happens."
        )
    return frozenset(ages.exact_values[period] for period in periods)


def _bind_law(
    *,
    law: object,
    targets: tuple[str, ...],
    source: RegimeName,
    age: object,
    side: Literal["solve", "simulate"],
    regime_names: tuple[RegimeName, ...],
) -> object:
    if not targets:
        return _absent_law(law=law, n_regimes=len(regime_names))
    if isinstance(law, Mapping):
        return _bind_cells(
            law=law,
            targets=targets,
            source=source,
            age=age,
            side=side,
            regime_names=regime_names,
        )
    if isinstance(law, StochasticTransition):
        return _SupportedStochasticTransition(func=law.func, targets=targets)
    if isinstance(law, DeterministicTransition) or callable(law):
        return _SupportedDeterministicTransition(
            func=law.func
            if isinstance(law, DeterministicTransition)
            else cast("UserFunction", law),
            targets=targets,
        )
    if isinstance(law, str):
        if law not in targets:
            raise ModelInitializationError(
                f"Transition kernel out of ({age}, '{source}') names '{law}' "
                f"outside its {side} graph edges."
            )
        return law
    raise ModelInitializationError(
        f"Invalid transition kernel {law!r} out of ({age}, '{source}')."
    )


def _fail_if_kernel_extends_graph(
    *, kernels: Mapping[int, object], source: RegimeName, edges: GraphEdges
) -> None:
    """Reject scalar probability destinations outside the declared phase graph."""
    for kernel in kernels.values():
        for side in ("solve", "simulate"):
            law = getattr(kernel, side) if isinstance(kernel, Phased) else kernel
            if isinstance(law, Mapping):
                allowed = set(getattr(edges, side).get(source, {}))
                if not allowed:
                    continue
                if not isinstance(kernel, Phased):
                    allowed |= set(edges.solve.get(source, {})) | set(
                        edges.simulate.get(source, {})
                    )
                extra = set(law) - allowed
                if extra:
                    raise ModelInitializationError(
                        f"Transition kernel of '{source}' in {side} names "
                        f"destinations {sorted(extra)} outside its declared graph."
                    )


def _absent_law(*, law: object, n_regimes: int) -> object:
    """Preserve kernel grammar for a phase with no source-age support."""
    if isinstance(law, Mapping | str):
        return law
    if isinstance(law, StochasticTransition):
        return _SupportedStochasticTransition(
            func=_Constant(value=(0.0,) * n_regimes), targets=()
        )
    return _SupportedDeterministicTransition(func=_Constant(value=0), targets=())


def _bind_cells(
    *,
    law: Mapping[str, object],
    targets: tuple[str, ...],
    source: RegimeName,
    age: object,
    side: Literal["solve", "simulate"],
    regime_names: tuple[RegimeName, ...],
) -> MappingProxyType[str, object]:
    """Select scalar cells and represent omitted probabilities as exact zeros."""
    unknown = set(law) - set(regime_names)
    if unknown:
        raise ModelInitializationError(
            f"Transition kernel of '{source}' names unknown regimes {sorted(unknown)}."
        )
    cells = {target: law[target] for target in targets if target in law}
    fallbacks = _fallbacks(law=cells, side=side)
    cells.update(
        {
            target: StochasticTransition(func=_Constant(value=0.0))
            for target in targets
            if target not in cells and target not in fallbacks
        }
    )
    undeclared_fallbacks = set(fallbacks) - set(targets)
    if undeclared_fallbacks:
        raise ModelInitializationError(
            f"Graph edges out of ({age}, '{source}') in {side} omit gated "
            f"fallback destinations {sorted(undeclared_fallbacks)}."
        )
    return MappingProxyType(cells)
