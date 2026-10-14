"""Reject ambiguous time declarations before DAG compilation or specialization."""

from collections.abc import Iterator, Mapping

from _lcm.params.temporal import temporal_parameter_names
from _lcm.regime_building.schedules import RegimeTransitionLaw
from _lcm.regime_law import RegimeLaws
from _lcm.time import TimeAxis, coordinate_kind
from _lcm.utils.functools import is_user_function
from lcm.exceptions import ModelInitializationError
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
