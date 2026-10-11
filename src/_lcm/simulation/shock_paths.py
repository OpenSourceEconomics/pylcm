"""Validate given realisations and bind them to call-local transition programs."""

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

import jax
import numpy as np

from _lcm.dtypes import canonical_float_dtype
from _lcm.engine import Regime
from _lcm.grids import DiscreteGrid, Grid
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.processes.iid import LogNormalIIDProcess, UniformIIDProcess, _IIDProcess
from _lcm.regime_building.processing import _build_next_state_vmapped
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.simulation.subject_groups import SubjectRows
from _lcm.typing import FlatParams, ReferenceName, RegimeName
from lcm.exceptions import InvalidSimulationInputError
from lcm.typing import StateName, UserInitialConditions


def canonicalize_shock_paths(  # noqa: C901, PLR0912
    *,
    paths: Mapping[StateName, jax.Array | np.ndarray] | None,
    regimes: Mapping[RegimeName, Regime],
    draw_only: Mapping[RegimeName, Mapping[StateName, _IIDProcess]],
    params: FlatParams,
    n_subjects: int,
) -> MappingProxyType[StateName, np.ndarray]:
    """Check shape, coverage and support in the simulation phase's active periods.

    Continuous processes accept values in their continuous distribution's support,
    including values outside the solve nodes. Discrete shocks accept exact codes.
    Transition-local draws are dated at the destination period, like carried draws.
    Values in periods where the state is not declared active are ignored.
    """
    result: dict[StateName, np.ndarray] = {}
    for name, path in (paths or {}).items():
        declarations: list[tuple[RegimeName, Grid, tuple[int, ...]]] = []
        for regime_name, regime in regimes.items():
            grid = regime.simulation.grids.get(name)
            is_lottery = any(
                f"next_{name}" in plan.lotteries
                for source in regimes.values()
                for plan in source.simulation.transition_plans.values()
            )
            if isinstance(grid, _ContinuousStochasticProcess) or (
                isinstance(grid, DiscreteGrid) and is_lottery
            ):
                declarations.append((regime_name, grid, regime.active_periods))
            local_grid = draw_only.get(regime_name, {}).get(name, grid)
            local_periods = tuple(
                period + 1
                for period in regime.active_periods
                if period < regime.simulation.reachability.n_periods - 1
                and any(
                    f"next_{name}"
                    in regime.simulation.transition_plans[target].lotteries
                    for target in regime.simulation.reachability.targets(
                        period=period, source=regime_name
                    )
                    if target in regime.simulation.transition_plans
                    and name not in regimes[target].simulation.grids
                )
            )
            if local_grid is not None and local_periods:
                declarations.append((regime_name, local_grid, local_periods))
        if not declarations:
            raise InvalidSimulationInputError(
                f"shock_paths state '{name}' is not a simulation shock."
            )
        values = np.asarray(path)
        if values.ndim not in (1, 2) or values.dtype.kind not in "fiu":
            raise InvalidSimulationInputError(
                f"shock_paths '{name}' requires a real array of shape (periods,) "
                "or (subjects, periods)."
            )
        if values.ndim != 1 and values.shape[0] != n_subjects:
            raise InvalidSimulationInputError(
                f"shock_paths '{name}' has {values.shape[0]} subjects; "
                f"expected {n_subjects}."
            )
        discrete = all(isinstance(grid, DiscreteGrid) for _, grid, _ in declarations)
        if discrete != any(
            isinstance(grid, DiscreteGrid) for _, grid, _ in declarations
        ):
            raise InvalidSimulationInputError(
                f"shock_paths '{name}' changes state type."
            )
        for regime_name, grid, periods in declarations:
            for period in periods:
                if period >= values.shape[-1]:
                    raise InvalidSimulationInputError(
                        f"shock_paths '{name}' does not cover period {period} "
                        f"in regime '{regime_name}'."
                    )
                column = values[..., period]
                valid = np.isfinite(column)
                if isinstance(grid, DiscreteGrid):
                    valid &= np.isin(column, np.asarray(grid.to_jax()))
                elif isinstance(grid, LogNormalIIDProcess):
                    valid &= column > 0
                elif isinstance(grid, UniformIIDProcess):
                    start = (
                        grid.start
                        if grid.start is not None
                        else params[regime_name][f"{name}__start"]
                    )
                    stop = (
                        grid.stop
                        if grid.stop is not None
                        else params[regime_name][f"{name}__stop"]
                    )
                    valid &= (column >= start) & (column <= stop)
                if not np.all(valid):
                    raise InvalidSimulationInputError(
                        f"shock_paths '{name}' has an inadmissible value at period "
                        f"{period} in regime '{regime_name}'."
                    )
        # Check after narrowing too: overflow/underflow cannot manufacture an
        # admissible path in the format simulation actually consumes.
        dtype = np.dtype(np.int32 if discrete else canonical_float_dtype())
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            narrowed = values.astype(dtype)
        for regime_name, grid, periods in declarations:
            for period in periods:
                column = narrowed[..., period]
                valid = np.isfinite(column)
                if discrete:
                    valid &= values[..., period] == column
                elif isinstance(grid, LogNormalIIDProcess):
                    valid &= column > 0
                if not np.all(valid):
                    raise InvalidSimulationInputError(
                        f"shock_paths '{name}' is not admissible in the simulation "
                        f"dtype at period {period} in regime '{regime_name}'."
                    )
        result[name] = narrowed
    return MappingProxyType(result)


def seed_shock_paths(
    *,
    initial: UserInitialConditions,
    paths: Mapping[StateName, np.ndarray],
    age_values: jax.Array,
    regimes: Mapping[RegimeName, Regime],
) -> UserInitialConditions:
    """A subject's entry value comes from its starting computational period."""
    if not paths:
        return initial
    starts = np.argmax(
        np.asarray(initial["age"])[:, None] == np.asarray(age_values), axis=1
    )
    state_names = {
        state for regime in regimes.values() for state in regime.simulation.state_names
    }
    supplied = {}
    for name, path in paths.items():
        if name not in state_names:
            continue
        periods = np.minimum(starts, path.shape[-1] - 1)
        supplied[name] = (
            path[periods] if path.ndim == 1 else path[np.arange(len(starts)), periods]
        )
    return dict(initial) | supplied


def bind_shock_path_programs(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    paths: Mapping[StateName, np.ndarray],
    params: FlatParams,
) -> MappingProxyType[RegimeName, Regime]:
    """Replace realised lotteries while retaining solve and decision programs."""
    if not paths:
        return regimes
    all_grids = MappingProxyType(
        {name: regime.simulation.grids for name, regime in regimes.items()}
    )
    bound = {}
    for name, regime in regimes.items():
        built = _build_next_state_vmapped(
            active_periods=regime.active_periods,
            phase_reachability=regime.simulation.reachability,
            source_regime_name=name,
            functions=regime.simulation.functions,
            transitions=regime.simulation.transitions,
            transition_plans=regime.simulation.transition_plans,
            all_grids=all_grids,
            flat_param_names=frozenset(params[name]),
            enable_jit=False,
            supplied_states=frozenset(paths),
        )
        transitions = {
            period: dataclasses.replace(
                program,
                function=_SubjectTiled(
                    func=built.per_subject_by_period[period].function,
                    subject_arg_names=built.per_subject_by_period[
                        period
                    ].subject_arg_names,
                ),
                argument_builder=_ArgumentsBoundAtDispatch(
                    program_name=program.name,
                    subject_arg_names=built.per_subject_by_period[
                        period
                    ].subject_arg_names,
                ),
            )
            for period, program in regime.simulation.programs.transition.items()
        }
        bound[name] = dataclasses.replace(
            regime,
            simulation=dataclasses.replace(
                regime.simulation,
                programs=dataclasses.replace(
                    regime.simulation.programs, transition=MappingProxyType(transitions)
                ),
            ),
        )
    return MappingProxyType(bound)


def shock_path_values(
    *,
    paths: Mapping[StateName, np.ndarray],
    argument_names: tuple[ReferenceName, ...],
    period: int,
    subject_slice: slice | SubjectRows,
    n_subjects: int,
) -> dict[ReferenceName, np.ndarray]:
    """Gather original rows, repeating the last real row for inert padding."""
    result = {}
    for argument in argument_names:
        if not argument.startswith("supplied_shock__"):
            continue
        path = paths[argument.removeprefix("supplied_shock__")]
        if path.ndim == 1:
            result[argument] = np.full(n_subjects, path[period], dtype=path.dtype)
        else:
            rows = (
                subject_slice.rows
                if isinstance(subject_slice, SubjectRows)
                else np.arange(subject_slice.start or 0, subject_slice.stop)
            )
            result[argument] = path[np.minimum(rows, path.shape[0] - 1), period]
    return result


def supplied_shock_argument_names(
    *, regime: Regime, period: int
) -> tuple[ReferenceName, ...]:
    """Read the exact bound transition's subject argument declaration."""
    program = regime.simulation.programs.transition.get(period)
    if program is None:
        return ()
    builder = program.argument_builder
    if not isinstance(builder, _ArgumentsBoundAtDispatch):
        raise TypeError("Simulation transition arguments must be bound at dispatch.")
    return builder.subject_arg_names
