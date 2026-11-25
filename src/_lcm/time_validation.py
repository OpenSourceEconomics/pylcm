"""Reject ambiguous time declarations before DAG compilation or specialization."""

from collections.abc import Iterator, Mapping

from _lcm.model_graph import GraphEdges
from _lcm.params.temporal import temporal_parameter_names
from _lcm.regime_building.schedules import RegimeTransitionLaw
from _lcm.regime_law import RegimeLaws
from _lcm.time import ModelTime, TimeAxis, coordinate_kind
from _lcm.utils.functools import is_user_function
from lcm.consumption_savings_regime import _EGMFamilyRegime
from lcm.exceptions import ModelInitializationError
from lcm.exits import Consent
from lcm.phased import Phased
from lcm.regime import (
    ActionEntry,
    ConstraintEntry,
    FunctionEntry,
    Regime,
    StateEntry,
    StateTransitionEntry,
)
from lcm.transition import (
    AgeSpecializedFunction,
    AgeSpecializedGrid,
    ByAge,
    PeriodSpecializedFunction,
    PeriodSpecializedGrid,
)
from lcm.typing import RegimeName

# A declaration the walk visits: a regime slot value or a regime law, or a
# mapping of them.
type _Declaration = (
    FunctionEntry
    | ConstraintEntry
    | StateEntry
    | ActionEntry
    | StateTransitionEntry
    | RegimeTransitionLaw
    | Mapping[str, _Declaration]
)


def _declarations(value: _Declaration) -> Iterator[_Declaration]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _declarations(item)
    elif isinstance(value, Phased):
        yield from _declarations(value.solve)
        yield from _declarations(value.simulate)
    elif isinstance(value, ByAge):
        yield value
        for item in value.laws:
            yield from _declarations(item)
    else:
        yield value


def validate_time_declarations(
    *, regimes: Mapping[str, Regime], laws: RegimeLaws, ages: TimeAxis
) -> None:
    """Keep period and age markers distinct, and temporal slots as parameters."""
    from lcm.transition import ByPeriod  # noqa: PLC0415

    for name, regime in regimes.items():
        functions = {**regime.decomposed_functions, **regime.decomposed_constraints}
        wired = (
            set(functions)
            | set(regime.states)
            | set(regime.actions)
            | {f"next_{state}" for state in regime.state_transitions}
            | {"age", "period", "next_regime"}
        )
        # Broadcast states without a law can still be pruned; inspect declarations
        # without collecting a complete transition DAG at this early boundary.
        for declaration in _declarations(
            {
                "functions": functions,
                "states": regime.states,
                "actions": regime.actions,
                "transitions": regime.state_transitions,
                "law": laws[name].transition,
            }
        ):
            if isinstance(
                declaration, (AgeSpecializedFunction, AgeSpecializedGrid, ByAge)
            ):
                is_period = isinstance(
                    declaration,
                    (PeriodSpecializedFunction, PeriodSpecializedGrid, ByPeriod),
                )
                if is_period != (coordinate_kind(ages) == "period"):
                    raise ModelInitializationError(
                        f"Regime {name!r}: {type(declaration).__name__} has the "
                        "wrong coordinate kind for a "
                        f"{coordinate_kind(ages)} model."
                    )
            if is_user_function(declaration):
                conflicts = temporal_parameter_names(declaration) & wired
                if conflicts:
                    raise ModelInitializationError(
                        f"Regime {name!r}: temporal parameter(s) "
                        f"{sorted(conflicts)} name a state or DAG node; "
                        "only parameter arguments may be declared."
                    )


def validate_stages(
    *,
    regimes: Mapping[RegimeName, Regime],
    laws: RegimeLaws,
    edges: GraphEdges,
    time: ModelTime,
) -> None:
    """Refuse declarations that break the order of the stages within a period.

    With its pass-throughs, every edge lands in the next stage; a gate's fallback
    lands there too. An exit enters a regime of its own stage or of a later stage
    of its period. A stage before the last links to the next one without
    discounting, so its regimes declare no Koopmans aggregator and use no
    endogenous-grid solver.
    """
    if not time.stages:
        return
    stage_of = time.stage_of
    names = [name for name, _ in time.stages]
    problems = [
        f"the edge '{source}' -> '{target}' lands in stage "
        f"'{names[stage_of[target]]}', not in the next stage "
        f"'{names[(stage_of[source] + 1) % time.n_stages]}'"
        for phase_edges in (edges.solve, edges.simulate)
        for source, targets in phase_edges.items()
        for target in targets
        if stage_of[target] != (stage_of[source] + 1) % time.n_stages
    ]
    for name, regime in regimes.items():
        exit_ = regime.exit
        destinations = (
            ()
            if exit_ is None
            else (
                *(target.regime for target in exit_.targets.values()),
                *(
                    (exit_.rule.counterparty.regime,)
                    if isinstance(exit_.rule, Consent)
                    else ()
                ),
            )
        )
        problems.extend(
            f"the exit of '{name}' in stage '{names[stage_of[name]]}' reads "
            f"'{destination}' of the earlier stage '{names[stage_of[destination]]}'"
            for destination in destinations
            if stage_of[destination] < stage_of[name]
        )
        if stage_of[name] == time.n_stages - 1 or laws[name].terminal:
            continue
        if regime.koopmans_aggregator is not None:
            problems.append(
                f"'{name}' declares a Koopmans aggregator in stage "
                f"'{names[stage_of[name]]}'; a stage before the last links to the "
                "next stage of its period without discounting"
            )
        if isinstance(regime, _EGMFamilyRegime):
            problems.append(
                f"'{name}' uses an endogenous-grid solver in stage "
                f"'{names[stage_of[name]]}'; only the last stage of a period may"
            )
    if problems:
        raise ModelInitializationError(
            "The model's stages are inconsistent: "
            + "; ".join(dict.fromkeys(problems))
            + "."
        )
