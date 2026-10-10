"""Construction-time, solver-independent temporal regime reachability.

This module materializes effective period-indexed graphs from the canonical
coverage and graph-bound support prepared by `_lcm.model_graph.prepare_graph`.
`Model.edges` owns declared connectivity; numerical regime transitions and
`ByAge` supply the available laws. There is no runtime topology pass: the graph
never changes after construction, and runtime probabilities cannot alter it.

Every retained edge in `targets_by_period` is
`EdgeStatus.CONDITIONAL`: no declaration proves unconditional positive probability
independently of states, actions and free runtime parameters. Construction-fixed
exact-zero scalar probability proofs may remove effective edges. Every retained
edge requires a valid state handoff at model construction.

The solve and simulate phases build independent graphs (`ModelReachability.solution`
/ `.simulation`) from graph-declared phase support and canonical demand. They may
retain different edges at the same source period when `Model.edges` is phased.

Solver and simulation runtime code consume this graph (`PhaseReachability.targets`,
`.union_targets`, `.edge_status`, ...) but never infers reachability itself — it does
not call an activity predicate, inspect a declared transition's raw mapping keys, or
derive continuation targets from state-law-bundle keys.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Literal

from _lcm.typing import RegimeName
from lcm.transition import StochasticTransition
from lcm.typing import UserAge

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

    `CONDITIONAL` is retained in `targets_by_period`. It is provenance, not a
    deferred Boolean; there is deliberately no runtime `resolve` method.
    """

    n_periods: int
    active_regimes_by_period: tuple[frozenset[RegimeName], ...]
    candidate_targets_by_source: MappingProxyType[RegimeName, tuple[RegimeName, ...]]
    targets_by_period: tuple[MappingProxyType[RegimeName, tuple[RegimeName, ...]], ...]

    def __hash__(self) -> int:
        """Return a structural hash independent of mapping insertion order."""
        return hash(
            (
                self.n_periods,
                self.active_regimes_by_period,
                tuple(sorted(self.candidate_targets_by_source.items())),
                tuple(
                    tuple(sorted(targets_by_source.items()))
                    for targets_by_source in self.targets_by_period
                ),
            )
        )

    def targets(self, *, period: int, source: RegimeName) -> tuple[RegimeName, ...]:
        """Return retained targets for the edge from `period` to `period + 1`."""
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
        """Return the construction-time status of a candidate edge.

        A retained edge is `CONDITIONAL`; every other pair is `FALSE`.
        """
        if not 0 <= period < self.n_periods - 1:
            raise IndexError(period)
        if target in self.targets_by_period[period].get(source, ()):
            return EdgeStatus.CONDITIONAL
        return EdgeStatus.FALSE

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
        self, initial_regime_names: Collection[RegimeName]
    ) -> tuple[frozenset[RegimeName], ...]:
        """Return the forward closure over the already-built static graph."""
        reachable = [frozenset(initial_regime_names) & self.active_regimes_by_period[0]]
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
    nodes: frozenset[tuple[UserAge, RegimeName]] = frozenset()
    """Exact `(age, regime)` pairs of every solved problem: each pair the
    declared starts can visit, and each pair whose value a solved problem reads."""

    visited_nodes: frozenset[tuple[UserAge, RegimeName]] = frozenset()
    """Exact `(age, regime)` pairs a subject starting at a declared start can
    physically visit; a subset of `nodes`."""

    def for_phase(self, phase: PhaseName) -> PhaseReachability:
        """Select one phase without reconstructing anything."""
        return self.solution if phase == "solution" else self.simulation


def candidate_targets_from_transition(
    *, transition: object, all_regime_names: Collection[RegimeName]
) -> tuple[RegimeName, ...]:
    """Return the static candidate universe declared by one transition.

    * `None`: terminal, no targets.
    * per-target mapping: its keys are the declared candidate universe.
    * vector `StochasticTransition` with `targets`: its declared targets.
    * any other lowered callable or Markov transition: all regimes are
      candidates.

    This reads an engine law, whose period-specific support lives in the model
    graph; pylcm does not infer structural zeros by executing a transition at
    selected states or parameter values.
    """
    if transition is None:
        return ()
    if isinstance(transition, Mapping):
        # A mapping is the per-target form, whose keys are regime names by
        # construction.
        return tuple(sorted(transition))
    targets = getattr(transition, "targets", None)
    if isinstance(transition, StochasticTransition) and targets is not None:
        return tuple(sorted(targets))
    return tuple(sorted(all_regime_names))


def build_phase_reachability(
    *,
    n_periods: int,
    active_periods_by_regime: Mapping[RegimeName, Collection[int]],
    support_by_period: Mapping[RegimeName, Mapping[int, Collection[RegimeName]]],
    terminal_regimes: Collection[RegimeName] = (),
) -> PhaseReachability:
    """Build one static graph; every retained edge is `CONDITIONAL`.

    A source's retained targets at a period are exactly its declared support
    there. Every declared target must be covered at the next period; the graph
    never drops a declared target.
    """
    if n_periods < 1:
        raise ValueError("n_periods must be positive")

    regimes = frozenset(active_periods_by_regime)
    active = {
        regime: frozenset(periods)
        for regime, periods in active_periods_by_regime.items()
    }
    unknown_sources = frozenset(support_by_period) - regimes
    unknown_targets = {
        target
        for by_period in support_by_period.values()
        for targets in by_period.values()
        for target in targets
        if target not in regimes
    }
    if unknown_sources or unknown_targets:
        raise ValueError(
            "Declared support contains unknown regimes: "
            f"sources={sorted(unknown_sources)}, targets={sorted(unknown_targets)}"
        )
    uncovered = sorted(
        (source, period, target)
        for source, by_period in support_by_period.items()
        for period, targets in by_period.items()
        for target in targets
        if period + 1 not in active[target]
    )
    if uncovered:
        raise ValueError(
            "Declared targets must be covered at the next period; "
            f"(source, period, target) = {uncovered}"
        )

    terminal = frozenset(terminal_regimes)
    candidates = MappingProxyType(
        {
            source: tuple(
                sorted({target for targets in by_period.values() for target in targets})
            )
            for source, by_period in support_by_period.items()
        }
    )
    active_by_period = tuple(
        frozenset(regime for regime, periods in active.items() if period in periods)
        for period in range(n_periods)
    )

    target_maps: list[MappingProxyType[RegimeName, tuple[RegimeName, ...]]] = []
    for period in range(n_periods - 1):
        period_targets: dict[RegimeName, tuple[RegimeName, ...]] = {}
        for source in sorted(regimes):
            declared = (
                ()
                if source in terminal or period not in active[source]
                else tuple(support_by_period.get(source, {}).get(period, ()))
            )
            retained = tuple(
                target for target in candidates.get(source, ()) if target in declared
            )
            if retained:
                period_targets[source] = retained
        target_maps.append(MappingProxyType(period_targets))

    return PhaseReachability(
        n_periods=n_periods,
        active_regimes_by_period=active_by_period,
        candidate_targets_by_source=candidates,
        targets_by_period=tuple(target_maps),
    )


def build_model_reachability(
    *,
    n_periods: int,
    active_periods_by_regime: Mapping[RegimeName, Collection[int]],
    support_by_phase: Mapping[
        str, Mapping[RegimeName, Mapping[int, Collection[RegimeName]]]
    ],
    terminal_regimes: Collection[RegimeName] = (),
    visited_periods_by_regime: Mapping[RegimeName, Collection[int]] | None = None,
) -> ModelReachability:
    """Build solve and simulate graphs from the declared per-period support.

    `active_periods_by_regime` must be the single canonical coverage mapping
    computed once at model preparation from the declarations. The simulate
    graph is active only where a subject can be: `visited_periods_by_regime`,
    which defaults to the coverage.
    """
    return ModelReachability(
        solution=build_phase_reachability(
            n_periods=n_periods,
            active_periods_by_regime=active_periods_by_regime,
            support_by_period=support_by_phase["solution"],
            terminal_regimes=terminal_regimes,
        ),
        simulation=build_phase_reachability(
            n_periods=n_periods,
            active_periods_by_regime=(
                active_periods_by_regime
                if visited_periods_by_regime is None
                else visited_periods_by_regime
            ),
            support_by_period=support_by_phase["simulation"],
            terminal_regimes=terminal_regimes,
        ),
    )
