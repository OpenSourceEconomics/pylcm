"""Read construction-time model declarations and their declared DAG inputs."""

import dataclasses
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from _lcm.constraints.processed import ProcessedConstraint, normalize_constraints
from _lcm.exits import ExitFunctionDemand
from _lcm.model_graph import ModelGraph
from _lcm.regime_building.age_normalization import (
    PeriodizedEconFunction,
    PeriodizedUserFunction,
)
from _lcm.regime_building.phases import RegimePhaseSpec
from _lcm.regime_building.processing import PreparedModelStructure
from lcm.regime import Regime
from lcm.typing import Phase, ReferenceName, RegimeName, UserFunction


@dataclass(frozen=True, kw_only=True)
class FunctionDeclaration:
    """Describe a callable through the same advertised signature its DAG reads."""

    function: UserFunction
    """The original callable; inspection never evaluates it."""
    signature: inspect.Signature | None
    """Its declared signature, or `None` when Python cannot inspect it.

    This follows `__wrapped__` and explicit signatures, as model construction
    does. It declares inputs; it does not prove a callback's implementation.
    """

    @property
    def arg_names(self) -> tuple[ReferenceName, ...] | None:
        """Return finite declared inputs, or `None` for an unknown interface."""
        if self.signature is None or any(
            parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            for parameter in self.signature.parameters.values()
        ):
            return None
        return tuple(self.signature.parameters)


@dataclass(frozen=True, kw_only=True)
class ModelDeclarations:
    """Inspect the model's resolved declarations without preparing execution.

    Containers are read-only. Grids, callables and their captured data remain
    the original objects: this view neither copies numerical inputs nor asserts
    semantic equivalence of graph transformations.
    """

    graph: ModelGraph
    """The model's declared and effective phase graphs, with bound laws."""
    regimes_by_period: MappingProxyType[int, MappingProxyType[RegimeName, Regime]]
    """Finalized regimes with construction-time age/period factories resolved."""
    phases: MappingProxyType[tuple[int, RegimeName, Phase], RegimePhaseSpec]
    """Resolved phase slices for demanded period/regime contexts; grids stay full-sized.

    Phase-invariant actions and ownership declarations remain in
    `regimes_by_period`. Phase slices carry functions, constraints, grid states,
    state and joint laws, continuation aggregators, and same-period exits.
    """
    functions: tuple[FunctionDeclaration, ...]
    """Unique callable identities and the input signatures construction accepts."""
    original_periods_by_regime: Mapping[RegimeName, tuple[int, ...]]
    """Physical entry cohorts before same-period exits."""
    value_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None
    """Periods with stored full values, or `None` when not separately recorded."""
    query_periods_by_phase: Mapping[str, Mapping[RegimeName, tuple[int, ...]]]
    """Query-only decisions keyed by `solution` and `simulation`, including exits."""
    exit_function_demand: ExitFunctionDemand | None
    """Declared callback paths demanded by same-period exits."""
    schema_version: int = 1
    """Version of this declaration-view contract."""


def _get_model_declarations(
    *,
    structure: PreparedModelStructure,
    graph: ModelGraph,
    n_periods: int,
    declared_regimes: Mapping[RegimeName, Regime],
) -> ModelDeclarations:
    regimes = MappingProxyType(
        {
            period: (
                structure.period_regimes.by_period.get(
                    period, structure.representative_user_regimes
                )
                if structure.period_regimes is not None
                else structure.representative_user_regimes
            )
            for period in range(n_periods)
        }
    )
    phases: dict[tuple[int, RegimeName, Phase], RegimePhaseSpec] = {}
    for regime_name, periods in structure.active_periods_by_regime.items():
        for period in periods:
            spec = structure.phased_specs[regime_name]
            for phase, phase_spec in (
                ("solve", spec.solution),
                ("simulate", spec.simulation),
            ):
                if phase == "simulate" and (
                    regime_name
                    not in structure.reachability.simulation.active_regimes_by_period[
                        period
                    ]
                    and period
                    not in structure.query_periods_by_phase.get("simulation", {}).get(
                        regime_name, ()
                    )
                ):
                    continue
                resolved = dataclasses.replace(
                    phase_spec,
                    **{
                        field.name: _resolve_value(
                            value=getattr(phase_spec, field.name), period=period
                        )
                        for field in dataclasses.fields(phase_spec)
                    },
                )
                if structure.grid_schedule is not None:
                    grids = structure.grid_schedule.by_period.get(period, {}).get(
                        regime_name, {}
                    )
                    resolved = dataclasses.replace(
                        resolved,
                        grid_states=MappingProxyType(
                            {
                                name: grids[name].grid if name in grids else grid
                                for name, grid in resolved.grid_states.items()
                            }
                        ),
                    )
                phases[period, regime_name, phase] = resolved
    return ModelDeclarations(
        graph=graph,
        regimes_by_period=regimes,
        phases=MappingProxyType(phases),
        functions=_get_function_declarations(
            roots=(declared_regimes, regimes, phases, graph.laws)
        ),
        original_periods_by_regime=structure.original_periods_by_regime,
        value_periods_by_regime=structure.value_periods_by_regime,
        query_periods_by_phase=structure.query_periods_by_phase,
        exit_function_demand=structure.exit_function_demand,
    )


def _get_function_declarations(
    *,
    roots: tuple[object, ...],  # noqa: PAN001 - heterogeneous canonical declaration roots
) -> tuple[FunctionDeclaration, ...]:
    functions: list[FunctionDeclaration] = []
    seen: set[int] = set()
    pending = list(roots)
    while pending:
        value = pending.pop()
        if id(value) in seen or isinstance(value, type):
            continue
        seen.add(id(value))
        if callable(value):
            try:
                signature = inspect.signature(value)
            except ValueError, TypeError:
                signature = None
            functions.append(
                FunctionDeclaration(
                    function=cast("UserFunction", value), signature=signature
                )
            )
        if isinstance(value, Mapping):
            pending.extend(value.values())
        elif dataclasses.is_dataclass(value):
            pending.extend(
                getattr(value, field.name) for field in dataclasses.fields(value)
            )
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
    return tuple(functions)


def _resolve_value[T](*, value: T, period: int) -> T:
    """Read cached period families without executing their user factories."""
    if isinstance(value, (PeriodizedUserFunction, PeriodizedEconFunction)):
        return cast("T", value.resolve(period))
    if isinstance(value, ProcessedConstraint):
        declaration = _resolve_value(value=value.declaration, period=period)
        if declaration is value.declaration:
            return value
        return cast(
            "T",
            normalize_constraints(constraints={value.name: declaration})[value.name],
        )
    if isinstance(value, Mapping):
        return cast(
            "T",
            MappingProxyType(
                {
                    name: _resolve_value(value=item, period=period)
                    for name, item in value.items()
                }
            ),
        )
    return value
