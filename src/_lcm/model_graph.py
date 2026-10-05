"""Resolve explicit model topology and bind it to the numerical transition laws."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from _lcm.reachability import ModelReachability, PhaseReachability
from _lcm.regime_building.fixed_regime_support import prune_fixed_regime_support
from _lcm.regime_building.schedules import (
    RegimeSchedules,
    _Constant,
    _fallbacks,
    resolve_demand,
    resolve_regime_schedules,
)
from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
    _SupportedStochasticTransition,
)
from lcm.ages import AgeGrid
from lcm.collective import ValueDependentTransition
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.phased import Phased
from lcm.regime import Regime
from lcm.transition import (
    AgeSelector,
    ByAge,
    DeterministicTransition,
    StochasticTransition,
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
class GraphPreparation:
    """Keep the graph proof and its numerical declarations together."""

    regimes: MappingProxyType[RegimeName, Regime]
    """Graph-bound regimes after fixed-zero probability pruning."""
    schedules: RegimeSchedules
    """Graph support restricted to physical and valued demand."""
    declarations: MappingProxyType[RegimeName, object]
    """Graph-bound kernels before fixed-zero pruning, for dormant inspection."""
    consumed_param_keys: frozenset[str]
    """Fixed leaves consumed by a removed zero cell."""
    pruned_edges: MappingProxyType[str, MappingProxyType[Edge, str]]
    """Fixed-zero primary edges and their proof reason."""


def prepare_graph(
    *,
    regimes: Mapping[RegimeName, Regime],
    edges: GraphEdges,
    ages: AgeGrid,
    initial_nodes: frozenset[tuple[object, RegimeName]],
    fixed_params: UserParams,
) -> GraphPreparation:
    """Bind laws, prove fixed zeros, and close physical and value demand."""
    bound = bind_graph_support(regimes=regimes, edges=edges, ages=ages)
    declarations = MappingProxyType(
        {name: regime.regime_transitions for name, regime in bound.items()}
    )
    source_ages = {"solution": edges.solve, "simulation": edges.simulate}
    before = resolve_regime_schedules(
        user_regimes=bound, ages=ages, source_ages_by_phase=source_ages
    )
    fixed_support = prune_fixed_regime_support(
        user_regimes=bound, fixed_params=fixed_params
    )
    after = resolve_regime_schedules(
        user_regimes=fixed_support.user_regimes,
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
        terminal_regimes=frozenset(
            name for name, regime in regimes.items() if regime.terminal
        ),
        ages=ages,
    )
    return GraphPreparation(
        regimes=fixed_support.user_regimes,
        schedules=schedules,
        declarations=declarations,
        consumed_param_keys=fixed_support.consumed_param_keys,
        pruned_edges=fixed_zero_edge_reasons(before=before, after=after, ages=ages),
    )


def resolve_graph_edges(
    *, edges: object, regimes: Mapping[RegimeName, Regime], ages: AgeGrid
) -> GraphEdges:
    """Validate both phases and snapshot selectors as exact source ages."""
    solve, simulate = (
        (edges.solve, edges.simulate)
        if isinstance(edges, Phased | GraphEdges)
        else (edges, edges)
    )
    if isinstance(edges, GraphEdges):
        solve, simulate = (
            {
                source: {
                    target: tuple(sorted(selected))
                    for target, selected in destinations.items()
                }
                for source, destinations in phase.items()
            }
            for phase in (edges.solve, edges.simulate)
        )
    return GraphEdges(
        solve=_resolve_edges(edges=solve, regimes=regimes, ages=ages),
        simulate=_resolve_edges(edges=simulate, regimes=regimes, ages=ages),
    )


def bind_graph_support(
    *, regimes: Mapping[RegimeName, Regime], edges: GraphEdges, ages: AgeGrid
) -> MappingProxyType[RegimeName, Regime]:
    """Bind graph-selected probability cells and selector support per source age.

    Numerical kernels supply no support. The private tags used by scheduling
    and lowering are derived exclusively from the resolved graph.
    """
    result: dict[RegimeName, Regime] = {}
    for source, regime in regimes.items():
        if regime.terminal:
            result[source] = regime
            continue
        transition = regime.regime_transitions
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
                    for target in regimes
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
                        regime_names=tuple(regimes),
                    )
                sides[side] = cache[key]
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
                func=_Constant(value=(0.0,) * len(regimes)),
                targets=MappingProxyType({}),
            )
        )
        result[source] = regime.replace(regime_transitions=bound)
    return MappingProxyType(result)


def fixed_zero_edge_reasons(
    *, before: RegimeSchedules, after: RegimeSchedules, ages: AgeGrid
) -> MappingProxyType[str, MappingProxyType[Edge, str]]:
    """Record only support removed by the fixed-probability proof stage."""
    return MappingProxyType(
        {
            side: MappingProxyType(
                {
                    (
                        ages.exact_values[period],
                        source,
                        target,
                    ): "fixed_zero_probability"
                    for source, by_period in before.support_by_phase[phase].items()
                    for period, targets in by_period.items()
                    for target in targets
                    if target
                    not in after.support_by_phase[phase].get(source, {}).get(period, ())
                }
            )
            for side, phase in (("solve", "solution"), ("simulate", "simulation"))
        }
    )


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
            if regimes[source].terminal:
                raise ModelInitializationError(
                    f"Terminal regime '{source}' cannot declare outgoing graph edges."
                )
            try:
                _fail_if_invalid_age_selector(selector)
                periods = _select_periods(
                    selector=selector, ages=ages, period_by_age=period_by_age
                )
            except RegimeInitializationError as error:
                raise ModelInitializationError(str(error)) from error
            if not periods:
                raise ModelInitializationError(
                    f"Graph selector {selector!r} for '{source}' → '{target}' "
                    "selects no model age."
                )
            selected[target] = frozenset(
                ages.exact_values[period] for period in periods
            )
        resolved[source] = MappingProxyType(selected)
    return MappingProxyType(resolved)


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
