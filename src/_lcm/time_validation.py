"""Reject ambiguous time declarations before DAG compilation or specialization."""

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import cast

from _lcm.model_graph import GraphEdges
from _lcm.params.temporal import temporal_parameter_names
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.schedules import RegimeTransitionLaw
from _lcm.regime_law import RegimeLaws
from _lcm.time import ModelTime, TimeAxis, coordinate_kind, format_age
from _lcm.utils.functools import is_user_function
from lcm.consumption_savings_regime import _EGMFamilyRegime
from lcm.exceptions import ModelInitializationError
from lcm.exits import Consent, Exit
from lcm.phased import Phased
from lcm.regime import (
    ActionEntry,
    ConstraintEntry,
    FunctionEntry,
    Regime,
    StateEntry,
    StateTransitionEntry,
)
from lcm.shocks import ShockGroup, ShockGroupName
from lcm.transition import (
    AgeSpecializedFunction,
    AgeSpecializedGrid,
    ByAge,
    PeriodSpecializedFunction,
    PeriodSpecializedGrid,
)
from lcm.typing import RegimeName, UserAge

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


def validate_stages(  # noqa: C901, PLR0912 - one pass over the regimes' edges and exits
    *,
    regimes: Mapping[RegimeName, Regime],
    laws: RegimeLaws,
    edges: GraphEdges,
    time: ModelTime,
    shocks: Mapping[ShockGroupName, ShockGroup],
) -> MappingProxyType[RegimeName, frozenset[int]]:
    """Refuse declarations that break the order of the stages within a period.

    An edge lands at its target's next occurrence: in a later stage of the
    source's period without discounting, else in the next period with one
    discount. At each age, a source's edges all land in its period or all cross
    into the next one; one whose edges stay in its period at some age declares
    no Koopmans aggregator and uses no endogenous-grid solver. A gated edge and
    its fallbacks land in the next stage, and an edge skips no stage of a target
    with continuous stochastic processes. An exit enters a regime of its own
    stage or of a later stage of its period, and observes no shock that such a
    regime draws.

    Returns:
        The regimes whose edges land in their own period at some age, each with
        the periods in which its edges cross into the next period instead.

    """
    if not time.stages:
        return MappingProxyType({})
    stage_of = time.stage_of
    names = [name for name, _ in time.stages]
    problems: list[str] = []
    linked: dict[RegimeName, frozenset[int]] = {}
    for name, regime in regimes.items():
        stage = stage_of[name]
        targets = sorted(
            {
                target
                for phase_edges in (edges.solve, edges.simulate)
                for target in phase_edges.get(name, {})
            }
        )
        by_age: dict[UserAge, set[RegimeName]] = {}
        for phase_edges in (edges.solve, edges.simulate):
            for target, ages in phase_edges.get(name, {}).items():
                for age in ages:
                    by_age.setdefault(age, set()).add(target)
        within_ages = set()
        for age, at_age in sorted(by_age.items()):
            within = sorted(target for target in at_age if stage_of[target] > stage)
            if within:
                within_ages.add(age)
            if within and len(within) < len(at_age):
                problems.append(
                    f"at {coordinate_kind(time)} {format_age(float(age))}, the edges "
                    f"of '{name}' in stage '{names[stage]}' land in its period, "
                    f"into {within}, and in the next period, into "
                    f"{sorted(at_age - set(within))}; "
                    "a source's edges at one age either all stay in its period or "
                    "all cross into the next one"
                )
        skipped = {
            target: names[stage + 1 :] + names[: stage_of[target]]
            if stage_of[target] <= stage
            else names[stage + 1 : stage_of[target]]
            for target in targets
        }
        for target in laws[name].gated_edges:
            gated = (
                target,
                *(
                    route.regime
                    for leg in laws[name].gated_edges[target].legs.values()
                    for route in (leg.solve_fallback, leg.simulate_fallback)
                ),
            )
            problems.extend(
                f"the gated edge '{name}' -> '{target}' lands in '{landing}' of "
                f"stage '{names[stage_of[landing]]}'; a gated edge and its "
                "fallbacks land in the next stage"
                for landing in gated
                if stage_of[landing] != (stage + 1) % time.n_stages
            )
        for target, sat_out in skipped.items():
            processes = sorted(
                state
                for state, grid in regimes[target].states.items()
                if isinstance(grid, _ContinuousStochasticProcess)
            )
            if sat_out and processes:
                problems.append(
                    f"'{target}' sits out stage '{sat_out[0]}' on the edge from "
                    f"'{name}', so its continuous stochastic processes "
                    f"{processes} are not drawn there; declare a regime of its "
                    f"own in stage '{sat_out[0]}'"
                )
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
        for destination in destinations:
            if stage_of[destination] < stage:
                problems.append(
                    f"the exit of '{name}' in stage '{names[stage]}' reads "
                    f"'{destination}' of the earlier stage "
                    f"'{names[stage_of[destination]]}'"
                )
            drawn = sorted(
                set(cast("Exit", exit_).observes)
                & {
                    shock
                    for group in regimes[destination].shocks
                    if group in shocks
                    for shock in shocks[group].shocks
                }
            )
            if stage_of[destination] > stage and drawn:
                problems.append(
                    f"the exit of '{name}' in stage '{names[stage]}' observes "
                    f"{drawn} of '{destination}', drawn in its later stage "
                    f"'{names[stage_of[destination]]}'; draw the shock in stage "
                    f"'{names[stage]}' and carry it as a state"
                )
        if not within_ages or laws[name].terminal:
            continue
        crossing = frozenset(by_age) - within_ages
        linked[name] = frozenset(
            step // time.n_stages
            for step, label in enumerate(time.exact_values)
            if label in crossing
        )
        if regime.koopmans_aggregator is not None:
            problems.append(
                f"'{name}' declares a Koopmans aggregator in stage "
                f"'{names[stage]}'; its edges land in its period, which they "
                "reach without discounting"
            )
        if isinstance(regime, _EGMFamilyRegime):
            problems.append(
                f"'{name}' uses an endogenous-grid solver in stage "
                f"'{names[stage]}'; only a regime whose edges cross into the next "
                "period may"
            )
    if problems:
        raise ModelInitializationError(
            "The model's stages are inconsistent: "
            + "; ".join(dict.fromkeys(problems))
            + "."
        )
    return MappingProxyType(linked)
