"""Placement-independent analysis of invariant state coordinates.

A state is a candidate when some declaration gives it the identity law
(`fixed_transition`, including the group state generated for a declared
`fixed_component`). Each candidate is analysed separately per phase over the
model's static regime graph. It is eligible in a phase only when the
declarations themselves establish that its value never changes and that no
value read crosses its codes:

- every retained edge between two regimes carrying the state declares the
  identity law for it;
- an edge from a carrier into a regime without the state reads one shared,
  type-free value, recorded as a shared dependency;
- no edge enters a carrier from a regime without the state, since the source
  would read every code's value (entry, or re-entry after a drop, without an
  established binding);
- every carrier holds the state on one discrete grid, the canonical code
  mapping;
- no same-period reference, gated edge or edge reference touches a carrier, and
  no simulate-phase carrier replays a retained solve payload; those channels
  have no supported per-code binding.

Invariance is never inferred from names, simulated paths, zero probabilities or
evaluations of user functions. The analysis reads no execution configuration,
so it does not depend on which states are sharded.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from _lcm.engine import GridRecomputationRoute, Regime
from _lcm.grids import DiscreteGrid
from _lcm.identity_transition import _IdentityTransition
from _lcm.reachability import ModelReachability, PhaseReachability
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.fixed_components import FixedComponentSplit
from _lcm.regime_building.phases import (
    PhasedRegimeSpec,
    RegimePhaseSpec,
    normalize_regime_phases,
)
from _lcm.typing import RegimeName, StateName
from lcm.ages import AgeGrid
from lcm.exceptions import ExecutionPlanningError

type Phase = Literal["solve", "simulate"]


@dataclass(frozen=True, kw_only=True)
class RegimeEdge:
    """One retained regime edge from a source period to the next period."""

    period: int
    """Period of the source regime."""

    source: RegimeName
    """Regime the edge leaves."""

    target: RegimeName
    """Regime the edge enters at `period + 1`."""


@dataclass(frozen=True, kw_only=True)
class InvariantPhaseAnalysis:
    """Whether one phase preserves a candidate coordinate, and on which edges."""

    phase: Phase
    """The analysed phase."""

    carrying_periods: MappingProxyType[RegimeName, tuple[int, ...]]
    """Per regime carrying the state in this phase, its active periods."""

    preserving_edges: tuple[RegimeEdge, ...]
    """Edges between carriers whose declared law is the identity."""

    shared_dependencies: tuple[RegimeEdge, ...]
    """Edges from a carrier into a regime that does not carry the state."""

    refusals: tuple[str, ...]
    """Every failed condition; empty when the declarations establish preservation."""

    @property
    def eligible(self) -> bool:
        """Whether the coordinate has carriers and no failed condition."""
        return bool(self.carrying_periods) and not self.refusals


@dataclass(frozen=True, kw_only=True)
class InvariantComponent:
    """A candidate coordinate, its canonical code mapping and per-phase analyses."""

    state_name: StateName
    """Canonical state name."""

    codes: tuple[int, ...]
    """Codes of the state grid; empty for a state without a discrete grid."""

    labels: tuple[str, ...]
    """Label of each code, in code order."""

    original_state_name: StateName | None
    """Declared state a generated fixed-component group state was split from."""

    original_codes_by_code: tuple[tuple[int, ...], ...]
    """For a generated group state, the declared codes of each group; else empty."""

    initial_nodes: tuple[tuple[int, RegimeName], ...]
    """Admissible `(period, regime)` starts whose regime carries the state."""

    solve: InvariantPhaseAnalysis
    """Backward-induction analysis."""

    simulate: InvariantPhaseAnalysis
    """Forward-simulation analysis."""


def analyze_invariant_components(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    regimes: Mapping[RegimeName, Regime],
    reachability: ModelReachability,
    initial_nodes: frozenset[tuple[object, RegimeName]],
    ages: AgeGrid,
    fixed_component_splits: Mapping[StateName, FixedComponentSplit],
) -> MappingProxyType[StateName, InvariantComponent]:
    """Analyse every identity-law state of a built model.

    Args:
        user_regimes: Mapping of regime names to the finalized regimes the
            engine was built from, after broadcast pruning.
        regimes: Mapping of regime names to the canonical engine regimes, the
            authority for value-read channels besides ordinary continuation.
        reachability: The model's static solve and simulate regime graphs.
        initial_nodes: The admissible `(age, regime)` starts.
        ages: The model's age grid.
        fixed_component_splits: Mapping of each declared fixed-component state
            to its group split.

    Returns:
        Immutable mapping of each candidate state, in name order, to its analysis.

    """
    specs = {
        name: normalize_regime_phases(regime) for name, regime in user_regimes.items()
    }
    period_of_age: dict[object, int] = {
        age: period for period, age in enumerate(ages.exact_values)
    }
    generated = {
        f"{name}_fixed": (name, split) for name, split in fixed_component_splits.items()
    }
    components: dict[StateName, InvariantComponent] = {}
    for state_name in sorted(_identity_law_states(specs)):
        analyses = {
            phase: _analyze_phase(
                phase=phase,
                state_name=state_name,
                specs=specs,
                regimes=regimes,
                reachability=reachability.solution
                if phase == "solve"
                else reachability.simulation,
            )
            for phase in ("solve", "simulate")
        }
        grid = _first_grid(specs=specs, state_name=state_name)
        source = generated.get(state_name)
        components[state_name] = InvariantComponent(
            state_name=state_name,
            codes=tuple(grid.codes) if isinstance(grid, DiscreteGrid) else (),
            labels=tuple(grid.categories) if isinstance(grid, DiscreteGrid) else (),
            original_state_name=None if source is None else source[0],
            original_codes_by_code=()
            if source is None
            else _original_codes_by_group(source[1]),
            initial_nodes=tuple(
                sorted(
                    (period_of_age[age], regime_name)
                    for age, regime_name in initial_nodes
                    if regime_name in analyses["simulate"].carrying_periods
                )
            ),
            solve=analyses["solve"],
            simulate=analyses["simulate"],
        )
    return MappingProxyType(components)


def fail_if_invariant_blocking_is_unsafe(
    *,
    components: Mapping[StateName, InvariantComponent],
    block_widths: Mapping[StateName, int],
    phase: Phase,
) -> None:
    """Refuse an explicit invariant blocking request the analysis does not support.

    Args:
        components: Mapping of candidate states to their analyses, from
            `analyze_invariant_components`.
        block_widths: Mapping of each state to block to the number of its codes
            evaluated at once.
        phase: The phase the blocking would execute in.

    Raises:
        ExecutionPlanningError: If more than one state is blocked, a named state
            has no identity law, a width is not a positive integer no larger
            than the number of codes, or the phase is not eligible. The message
            names every failed condition and the remedies.

    """
    problems: list[str] = []
    if len(block_widths) > 1:
        problems.append(
            f"only one invariant state can be blocked, got {sorted(block_widths)}"
        )
    for state_name, width in block_widths.items():
        component = components.get(state_name)
        if component is None:
            problems.append(
                f"{state_name!r} has no identity law (`fixed_transition`), so it "
                "is not an invariant candidate"
            )
            continue
        if type(width) is not int or width < 1:
            problems.append(
                f"the block width of {state_name!r} must be a positive integer, "
                f"got {width!r}"
            )
        elif width > len(component.codes):
            problems.append(
                f"the block width {width} of {state_name!r} exceeds its "
                f"{len(component.codes)} codes"
            )
        analysis = component.solve if phase == "solve" else component.simulate
        if not analysis.carrying_periods:
            problems.append(f"no regime carries {state_name!r} in the {phase} phase")
        problems += [
            f"{state_name!r} is not invariant in the {phase} phase: {refusal}"
            for refusal in analysis.refusals
        ]
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        msg = (
            f"Invariant blocking {dict(block_widths)} is unsafe:\n{details}\n"
            "Remove the state from the blocking request to run the unblocked "
            "route, or declare the state with `fixed_transition` on every "
            "reachable edge between regimes that carry it and remove value reads "
            "across its codes."
        )
        raise ExecutionPlanningError(msg)


def _identity_law_states(specs: Mapping[RegimeName, PhasedRegimeSpec]) -> set[str]:
    """Return every state some phase slice gives the identity law toward a target."""
    return {
        state_name
        for spec in specs.values()
        for phase_slice in (spec.solution, spec.simulation)
        for state_name, law in phase_slice.state_transitions.items()
        if any(
            isinstance(leaf, _IdentityTransition)
            for leaf in (law.values() if isinstance(law, Mapping) else (law,))
        )
    }


def _analyze_phase(
    *,
    phase: Phase,
    state_name: StateName,
    specs: Mapping[RegimeName, PhasedRegimeSpec],
    regimes: Mapping[RegimeName, Regime],
    reachability: PhaseReachability,
) -> InvariantPhaseAnalysis:
    """Classify every retained edge and value channel of one phase."""
    slices = {
        name: spec.solution if phase == "solve" else spec.simulation
        for name, spec in specs.items()
    }
    carriers = frozenset(
        name
        for name, phase_slice in slices.items()
        if state_name in phase_slice.grid_states
    )
    refusals = _grid_refusals(state_name=state_name, slices=slices, carriers=carriers)
    preserving: list[RegimeEdge] = []
    shared: list[RegimeEdge] = []
    for period in range(reachability.n_periods - 1):
        for source in sorted(reachability.active_regimes_by_period[period]):
            for target in reachability.targets(period=period, source=source):
                edge = RegimeEdge(period=period, source=source, target=target)
                if source in carriers and target in carriers:
                    law = _law_toward(
                        phase_slice=slices[source], state_name=state_name, target=target
                    )
                    if isinstance(law, _IdentityTransition):
                        preserving.append(edge)
                    else:
                        refusals.append(
                            f"edge {source} -> {target} at period {period} gives "
                            f"{state_name!r} the law {_describe(law)}, not the identity"
                        )
                elif source in carriers:
                    shared.append(edge)
                elif target in carriers:
                    refusals.append(
                        f"edge {source} -> {target} at period {period} enters a "
                        f"carrier of {state_name!r} from a regime without it, so the "
                        "source reads every code's value"
                    )
    refusals += _channel_refusals(
        phase=phase, state_name=state_name, regimes=regimes, carriers=carriers
    )
    return InvariantPhaseAnalysis(
        phase=phase,
        carrying_periods=MappingProxyType(
            {
                name: tuple(
                    period
                    for period, active in enumerate(
                        reachability.active_regimes_by_period
                    )
                    if name in active
                )
                for name in sorted(carriers)
            }
        ),
        preserving_edges=tuple(preserving),
        shared_dependencies=tuple(shared),
        refusals=tuple(refusals),
    )


def _grid_refusals(
    *,
    state_name: StateName,
    slices: Mapping[RegimeName, RegimePhaseSpec],
    carriers: frozenset[RegimeName],
) -> list[str]:
    """Require one discrete grid, the canonical code mapping, across all carriers."""
    grids = {name: slices[name].grid_states[state_name] for name in sorted(carriers)}
    refusals = [
        f"regime {name} holds {state_name!r} on {type(grid).__name__}, not a discrete "
        "grid, so its codes cannot be blocked"
        for name, grid in grids.items()
        if not isinstance(grid, DiscreteGrid)
    ]
    domains = {
        (tuple(grid.categories), tuple(grid.codes))
        for grid in grids.values()
        if isinstance(grid, DiscreteGrid)
    }
    if len(domains) > 1:
        refusals.append(
            f"carriers of {state_name!r} declare different categories "
            f"{sorted(domains)}, so no single code mapping exists"
        )
    return refusals


def _law_toward(
    *, phase_slice: RegimePhaseSpec, state_name: StateName, target: RegimeName
) -> object:
    """Return the law a slice declares for a state toward one target.

    A joint kernel that outputs the state toward the target owns that cell; a
    per-target mapping names the target's law; any other law applies to every
    target, as canonicalization broadcasts it.
    """
    if any(
        state_name in kernel.outputs
        for kernel in phase_slice.joint_transitions.get(target, {}).values()
    ):
        return "a joint transition"
    law = phase_slice.state_transitions.get(state_name)
    return law.get(target) if isinstance(law, Mapping) else law


def _channel_refusals(
    *,
    phase: Phase,
    state_name: StateName,
    regimes: Mapping[RegimeName, Regime],
    carriers: frozenset[RegimeName],
) -> list[str]:
    """Refuse every non-continuation value channel that touches a carrier."""
    refusals = [
        f"same-period reference {name} -> {reference} touches a carrier of "
        f"{state_name!r}; no per-code binding is supported for it"
        for name, regime in sorted(regimes.items())
        for reference in regime.same_period_ref_regimes
        if carriers & {name, reference}
    ]
    refusals += [
        f"gated edge {name} -> {target} touches a carrier of {state_name!r}; no "
        "per-code binding is supported for it"
        for name, regime in sorted(regimes.items())
        for target in regime.gated_edges
        if carriers & {name, target, *regime.edge_reference_regimes}
    ]
    if phase == "simulate":
        refusals += [
            f"regime {name} replays a retained solve payload; no per-code binding "
            "is supported for it"
            for name in sorted(carriers)
            if regimes[name].simulation.replay_unsupported
            or not isinstance(
                regimes[name].simulation.external_replay_route,
                GridRecomputationRoute | None,
            )
        ]
    return refusals


def _first_grid(
    *, specs: Mapping[RegimeName, PhasedRegimeSpec], state_name: StateName
) -> object:
    """Return the state's grid in the first carrier, solve phase before simulate."""
    return next(
        (
            phase_slice.grid_states[state_name]
            for phase_slice in (
                *(spec.solution for spec in specs.values()),
                *(spec.simulation for spec in specs.values()),
            )
            if state_name in phase_slice.grid_states
        ),
        None,
    )


def _original_codes_by_group(split: FixedComponentSplit) -> tuple[tuple[int, ...], ...]:
    """Return the declared codes of each fixed-component group, in code order."""
    return tuple(
        tuple(
            code
            for code, group in enumerate(split.fixed_of_code)
            if group == group_code
        )
        for group_code in range(len(split.fixed_grid.codes))
    )


def _describe(law: object) -> str:
    """Name a declared law for a refusal message."""
    if law is None:
        return "none"
    if isinstance(law, str):
        return law
    return f"{type(law).__name__} {getattr(law, '__name__', '')}".rstrip()
