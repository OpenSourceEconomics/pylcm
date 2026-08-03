"""Construction-time, solver-independent temporal regime reachability.

This module owns the model graph. Solver code may derive layouts from the graph,
but may not infer which regime pairs are reachable.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Literal

from _lcm.typing import RegimeName

type PhaseName = Literal["solution", "simulation"]


class EdgeStatus(IntEnum):
    """Static classification of a declared one-period regime edge.

    No current declaration proves unconditional positive probability
    independently of state, action, and free runtime parameters — a
    per-target mapping with one key is still not such a proof. Every
    retained edge is therefore `CONDITIONAL`; there is no `TRUE` status to
    infer from declaration shape alone.
    """

    FALSE = 0
    CONDITIONAL = 2


@dataclass(frozen=True, kw_only=True)
class PhaseReachability:
    """One phase's immutable period-indexed regime graph.

    ``CONDITIONAL`` is retained in ``targets_by_period``. It is provenance, not a
    deferred Boolean; there is deliberately no runtime ``resolve`` method.
    """

    n_periods: int
    active_regimes_by_period: tuple[frozenset[RegimeName], ...]
    candidate_targets_by_source: MappingProxyType[RegimeName, tuple[RegimeName, ...]]
    targets_by_period: tuple[MappingProxyType[RegimeName, tuple[RegimeName, ...]], ...]
    edge_status_by_period: tuple[
        MappingProxyType[tuple[RegimeName, RegimeName], EdgeStatus], ...
    ]

    def targets(self, *, period: int, source: RegimeName) -> tuple[RegimeName, ...]:
        """Return retained targets for the edge from ``period`` to ``period + 1``."""
        if not 0 <= period < self.n_periods - 1:
            raise IndexError(period)
        return self.targets_by_period[period].get(source, ())

    def has_edge(self, *, period: int, source: RegimeName, target: RegimeName) -> bool:
        """Return whether the static graph contains this period-specific edge."""
        return (
            self.edge_status(period=period, source=source, target=target)
            != EdgeStatus.FALSE
        )

    def edge_status(
        self, *, period: int, source: RegimeName, target: RegimeName
    ) -> EdgeStatus:
        """Return the construction-time status of a candidate edge."""
        if not 0 <= period < self.n_periods - 1:
            raise IndexError(period)
        return self.edge_status_by_period[period].get(
            (source, target), EdgeStatus.FALSE
        )

    def periods_for_edge(
        self, *, source: RegimeName, target: RegimeName
    ) -> tuple[int, ...]:
        """Return all source periods in which the edge is retained."""
        return tuple(
            period
            for period in range(self.n_periods - 1)
            if self.has_edge(period=period, source=source, target=target)
        )

    def union_targets(self, *, source: RegimeName) -> tuple[RegimeName, ...]:
        """Return retained targets over all periods for build-time consumers."""
        return tuple(
            sorted(
                {
                    target
                    for period in range(self.n_periods - 1)
                    for target in self.targets(period=period, source=source)
                }
            )
        )

    def reachable_from(
        self, initial_regimes: Collection[RegimeName]
    ) -> tuple[frozenset[RegimeName], ...]:
        """Return the forward closure over the already-built static graph."""
        reachable = [frozenset(initial_regimes) & self.active_regimes_by_period[0]]
        for period in range(self.n_periods - 1):
            targets = {
                target
                for source in reachable[-1]
                for target in self.targets(period=period, source=source)
            }
            reachable.append(
                frozenset(targets) & self.active_regimes_by_period[period + 1]
            )
        return tuple(reachable)


@dataclass(frozen=True, kw_only=True)
class ModelReachability:
    """The model's static solve and simulate graphs."""

    solution: PhaseReachability
    simulation: PhaseReachability

    def for_phase(self, phase: PhaseName) -> PhaseReachability:
        """Select one phase without reconstructing anything."""
        return self.solution if phase == "solution" else self.simulation


def candidate_targets_from_transition(
    *, transition: object, all_regime_names: Collection[RegimeName]
) -> tuple[RegimeName, ...]:
    """Return the static candidate universe declared by one transition.

    * ``None``: terminal, no targets.
    * per-target mapping: its keys are the declared candidate universe.
    * coarse callable / Markov transition: all regimes are candidates.

    The coarse default is intentionally conservative. After the temporal activity
    intersection, every retained coarse edge is validated. A narrower coarse
    transition must expose static support metadata in a future API or use the
    existing per-target form; pylcm does not infer structural zeros by executing a
    transition at selected states or parameter values.
    """
    if transition is None:
        return ()
    if isinstance(transition, Mapping):
        return tuple(sorted(transition))
    return tuple(sorted(all_regime_names))


def build_phase_reachability(
    *,
    n_periods: int,
    active_periods_by_regime: Mapping[RegimeName, Collection[int]],
    candidate_targets_by_source: Mapping[RegimeName, Collection[RegimeName]],
    terminal_regimes: Collection[RegimeName] = (),
) -> PhaseReachability:
    """Build one static graph; every retained edge is `CONDITIONAL`."""
    if n_periods < 1:
        raise ValueError("n_periods must be positive")

    regimes = frozenset(active_periods_by_regime)
    unknown_sources = frozenset(candidate_targets_by_source) - regimes
    unknown_targets = {
        target
        for targets in candidate_targets_by_source.values()
        for target in targets
        if target not in regimes
    }
    if unknown_sources or unknown_targets:
        raise ValueError(
            "Candidate support contains unknown regimes: "
            f"sources={sorted(unknown_sources)}, targets={sorted(unknown_targets)}"
        )

    active = {
        regime: frozenset(periods)
        for regime, periods in active_periods_by_regime.items()
    }
    terminal = frozenset(terminal_regimes)
    candidates = MappingProxyType(
        {
            source: tuple(sorted(set(targets)))
            for source, targets in candidate_targets_by_source.items()
        }
    )
    active_by_period = tuple(
        frozenset(regime for regime, periods in active.items() if period in periods)
        for period in range(n_periods)
    )

    target_maps: list[MappingProxyType[RegimeName, tuple[RegimeName, ...]]] = []
    status_maps: list[MappingProxyType[tuple[RegimeName, RegimeName], EdgeStatus]] = []
    for period in range(n_periods - 1):
        period_targets: dict[RegimeName, tuple[RegimeName, ...]] = {}
        period_status: dict[tuple[RegimeName, RegimeName], EdgeStatus] = {}
        for source in sorted(regimes):
            retained: list[RegimeName] = []
            for target in candidates.get(source, ()):
                if (
                    source in terminal
                    or period not in active[source]
                    or period + 1 not in active[target]
                ):
                    status = EdgeStatus.FALSE
                else:
                    status = EdgeStatus.CONDITIONAL
                period_status[(source, target)] = status
                if status != EdgeStatus.FALSE:
                    retained.append(target)
            if retained:
                period_targets[source] = tuple(retained)
        target_maps.append(MappingProxyType(period_targets))
        status_maps.append(MappingProxyType(period_status))

    return PhaseReachability(
        n_periods=n_periods,
        active_regimes_by_period=active_by_period,
        candidate_targets_by_source=candidates,
        targets_by_period=tuple(target_maps),
        edge_status_by_period=tuple(status_maps),
    )


def build_model_reachability(
    *,
    n_periods: int,
    active_periods_by_regime: Mapping[RegimeName, Collection[int]],
    transitions_by_phase: Mapping[PhaseName, Mapping[RegimeName, object]],
    terminal_regimes: Collection[RegimeName] = (),
) -> ModelReachability:
    """Build solve and simulate graphs from the same construction-time semantics.

    `active_periods_by_regime` must be the single canonical activity mapping
    computed once at model preparation (via `AgeGrid.get_periods_where`) —
    this function does not evaluate `Regime.active` itself.
    """
    all_regime_names = frozenset(active_periods_by_regime)

    def build(phase: PhaseName) -> PhaseReachability:
        transitions = transitions_by_phase[phase]
        candidates = {
            source: candidate_targets_from_transition(
                transition=transitions.get(source),
                all_regime_names=all_regime_names,
            )
            for source in all_regime_names
        }
        return build_phase_reachability(
            n_periods=n_periods,
            active_periods_by_regime=active_periods_by_regime,
            candidate_targets_by_source=candidates,
            terminal_regimes=terminal_regimes,
        )

    return ModelReachability(solution=build("solution"), simulation=build("simulation"))
