"""Validate given realisations and bind them to call-local transition programs."""

import dataclasses
import functools
from collections.abc import Mapping
from types import MappingProxyType

import jax
import numpy as np

from _lcm.dtypes import canonical_float_dtype
from _lcm.engine import Regime
from _lcm.grids import DiscreteGrid, Grid
from _lcm.params.edges import regime_kernel_params
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.processes.iid import LogNormalIIDProcess, UniformIIDProcess, _IIDProcess
from _lcm.regime_building.age_normalization import group_periods_by_key
from _lcm.regime_building.processing import _build_next_state_vmapped
from _lcm.simulation.initial_conditions import resolve_initial_periods
from _lcm.simulation.operand_placement import SubjectArgumentNames
from _lcm.simulation.program_types import _PerSubjectFunction
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.simulation.subject_groups import SubjectRows
from _lcm.time import TimeAxis
from _lcm.typing import (
    FlatParams,
    QualifiedName,
    ReferenceName,
    RegimeName,
    RegimeNamesToIds,
)
from _lcm.utils.containers import invert_regime_ids
from lcm.exceptions import InvalidSimulationInputError
from lcm.typing import StateName, UserInitialConditions


def resolve_shock_paths(  # noqa: C901, PLR0912
    *,
    paths: Mapping[StateName, jax.Array | np.ndarray] | None,
    initial: UserInitialConditions,
    regimes: Mapping[RegimeName, Regime],
    draw_only: Mapping[RegimeName, Mapping[StateName, _IIDProcess]],
    params: FlatParams,
    ages: TimeAxis,
    regime_names_to_ids: RegimeNamesToIds,
) -> tuple[MappingProxyType[StateName, np.ndarray], UserInitialConditions]:
    """Check given realisations where a subject may read them; seed entry values.

    A cell `(subject, period)` is checked against a regime's support only when the
    subject may occupy the reading regime then: its starting regime at its starting
    period, and afterwards every regime the simulation graph reaches from there.
    Regime membership after entry is drawn during simulation, so a value is checked
    against every regime the subject can reach in that period; a cell no regime
    can read, such as one before the subject enters, is not checked.

    Continuous processes accept values in their continuous distribution's support,
    including values outside the solve nodes. Discrete shocks accept exact codes.
    Transition-local draws are dated at the destination period, like carried draws.

    A subject's entry value is replaced by its starting period's column only when
    its starting regime draws the state.

    Returns:
        The paths in the simulation dtype and the seeded initial conditions.

    """
    if not paths:
        return MappingProxyType({}), initial
    if "age" not in initial or "regime_id" not in initial:
        # Nothing can be read without them; the entry validator reports the gap.
        return MappingProxyType({}), initial
    regime_ids = np.asarray(initial["regime_id"])
    occupancy = _subject_occupancy(
        initial_ages=initial["age"],
        regime_ids=regime_ids,
        ages=ages,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
    )
    starts, on_grid = resolve_initial_periods(ages=ages, initial_ages=initial["age"])
    starts = np.where(on_grid, starts, 0)
    result: dict[StateName, np.ndarray] = {}
    seeded = dict(initial)
    for name, path in paths.items():
        declarations = _shock_declarations(
            name=name, regimes=regimes, draw_only=draw_only
        )
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
        if values.ndim != 1 and values.shape[0] != len(regime_ids):
            raise InvalidSimulationInputError(
                f"shock_paths '{name}' has {values.shape[0]} subjects; "
                f"expected {len(regime_ids)}."
            )
        discrete = all(isinstance(grid, DiscreteGrid) for _, grid, _, _ in declarations)
        if discrete != any(
            isinstance(grid, DiscreteGrid) for _, grid, _, _ in declarations
        ):
            raise InvalidSimulationInputError(
                f"shock_paths '{name}' changes state type."
            )
        dtype = np.dtype(np.int32 if discrete else canonical_float_dtype())
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            narrowed = values.astype(dtype)
        for regime_name, grid, periods, lag in declarations:
            for period in periods:
                readers = occupancy[regime_name][:, period - lag]
                if not readers.any():
                    continue
                if period >= values.shape[-1]:
                    raise InvalidSimulationInputError(
                        f"shock_paths '{name}' does not cover period {period} "
                        f"in regime '{regime_name}'."
                    )
                original, column = (
                    (values[period : period + 1], narrowed[period : period + 1])
                    if values.ndim == 1
                    else (values[readers, period], narrowed[readers, period])
                )
                if not np.all(
                    _admissible(
                        column=original,
                        grid=grid,
                        params=params,
                        regime_name=regime_name,
                        name=name,
                    )
                ):
                    raise InvalidSimulationInputError(
                        f"shock_paths '{name}' has an inadmissible value at period "
                        f"{period} in regime '{regime_name}'."
                    )
                # Check after narrowing too: overflow/underflow cannot manufacture
                # an admissible path in the format simulation actually consumes.
                valid = np.isfinite(column)
                if discrete:
                    valid &= original == column
                elif isinstance(grid, LogNormalIIDProcess):
                    valid &= column > 0
                if not np.all(valid):
                    raise InvalidSimulationInputError(
                        f"shock_paths '{name}' is not admissible in the simulation "
                        f"dtype at period {period} in regime '{regime_name}'."
                    )
        result[name] = narrowed
        seeded |= _seed_entry_values(
            name=name,
            path=narrowed,
            initial=initial,
            draws_at_entry=np.logical_or.reduce(
                [
                    occupancy[regime_name][np.arange(len(starts)), starts]
                    & np.isin(starts, periods)
                    for regime_name, _, periods, lag in declarations
                    if lag == 0
                ]
                + [np.zeros(len(starts), dtype=bool)]
            ),
            starts=starts,
        )
    return MappingProxyType(result), seeded


def _seed_entry_values(
    *,
    name: StateName,
    path: np.ndarray,
    initial: UserInitialConditions,
    draws_at_entry: np.ndarray,
    starts: np.ndarray,
) -> dict[StateName, np.ndarray]:
    """Take a drawing subject's entry value from its starting period's column.

    Other subjects keep their supplied entry value. When none is supplied and
    some subject's starting regime does not draw the state, nothing is seeded,
    so the entry validator reports the missing state.
    """
    if not draws_at_entry.any():
        return {}
    columns = np.minimum(starts, path.shape[-1] - 1)
    entry = path[columns] if path.ndim == 1 else path[np.arange(len(starts)), columns]
    if name in initial:
        return {name: np.where(draws_at_entry, entry, np.asarray(initial[name]))}
    return {name: entry} if draws_at_entry.all() else {}


def _shock_declarations(
    *,
    name: StateName,
    regimes: Mapping[RegimeName, Regime],
    draw_only: Mapping[RegimeName, Mapping[StateName, _IIDProcess]],
) -> list[tuple[RegimeName, Grid, tuple[int, ...], int]]:
    """List `(regime, support, periods, lag)` for every regime reading `name`.

    A state is read in the regime holding it; a transition-local draw dated at
    period `p` is read by the source regime in period `p - lag`.
    """
    declarations: list[tuple[RegimeName, Grid, tuple[int, ...], int]] = []
    for regime_name, regime in regimes.items():
        simulation_periods = tuple(regime.simulation.feasibility_pools_by_period)
        grid = regime.simulation.grids.get(name)
        if (
            grid is not None
            and simulation_periods
            and _draws_state(name=name, regime_name=regime_name, regimes=regimes)
        ):
            declarations.append((regime_name, grid, simulation_periods, 0))
        local_grid = draw_only.get(regime_name, {}).get(name, grid)
        local_periods = tuple(
            period + 1
            for period in simulation_periods
            if period < regime.simulation.reachability.n_periods - 1
            and any(
                f"next_{name}" in regime.simulation.transition_plans[target].lotteries
                for target in regime.simulation.reachability.targets(
                    period=period, source=regime_name
                )
                if target in regime.simulation.transition_plans
                and name not in regimes[target].simulation.grids
            )
        )
        if local_grid is not None and local_periods:
            declarations.append((regime_name, local_grid, local_periods, 1))
    return declarations


def _draws_state(
    *, name: StateName, regime_name: RegimeName, regimes: Mapping[RegimeName, Regime]
) -> bool:
    """Whether `regime_name` holds `name` as a stochastic process or a lottery.

    A discrete state is a lottery in a regime that draws it on leaving, or that
    some regime enters with a draw of it.
    """
    grid = regimes[regime_name].simulation.grids.get(name)
    if isinstance(grid, _ContinuousStochasticProcess):
        return True
    if not isinstance(grid, DiscreteGrid):
        return False
    next_name = f"next_{name}"
    return any(
        next_name in plan.lotteries
        for plan in regimes[regime_name].simulation.transition_plans.values()
    ) or any(
        next_name in source.simulation.transition_plans[regime_name].lotteries
        for source in regimes.values()
        if regime_name in source.simulation.transition_plans
    )


def _admissible(
    *,
    column: np.ndarray,
    grid: Grid,
    params: FlatParams,
    regime_name: RegimeName,
    name: StateName,
) -> np.ndarray:
    """Return which values of `column` lie in `grid`'s support."""
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
            grid.stop if grid.stop is not None else params[regime_name][f"{name}__stop"]
        )
        valid &= (column >= start) & (column <= stop)
    return valid


def _subject_occupancy(
    *,
    initial_ages: jax.Array | np.ndarray,
    regime_ids: np.ndarray,
    ages: TimeAxis,
    regimes: Mapping[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
) -> dict[RegimeName, np.ndarray]:
    """Mark, per regime, the `(subject, period)` cells the subject may occupy it in.

    A subject occupies its starting regime in its starting period and may occupy
    every regime the simulation graph reaches from there. Subjects with an
    off-grid age or an unknown regime id occupy nothing; the entry validator
    rejects them.
    """
    starts, on_grid = resolve_initial_periods(ages=ages, initial_ages=initial_ages)
    names = invert_regime_ids(regime_names_to_ids)
    n_periods = ages.n_periods
    occupancy = {
        name: np.zeros((len(regime_ids), n_periods), dtype=bool) for name in regimes
    }
    for start, regime_id in {
        (int(start), int(regime_id))
        for start, regime_id in zip(starts[on_grid], regime_ids[on_grid], strict=True)
    }:
        rows = on_grid & (starts == start) & (regime_ids == regime_id)
        current = {names[regime_id]} & set(regimes)
        for period in range(start, n_periods):
            for name in current:
                occupancy[name][rows, period] = True
            if period + 1 < n_periods:
                current = {
                    target
                    for source in current
                    if period in regimes[source].simulation.feasibility_pools_by_period
                    for target in regimes[source].simulation.reachability.targets(
                        period=period, source=source
                    )
                }
    return occupancy


def kernel_param_names(
    *, regimes: Mapping[RegimeName, Regime], params: FlatParams
) -> MappingProxyType[RegimeName, frozenset[QualifiedName]]:
    """Name the parameters each regime's transition kernels bind by name."""
    return MappingProxyType(
        {
            name: frozenset(regime_kernel_params(params, regime_name=name))
            | frozenset(regime.resolved_fixed_params)
            for name, regime in regimes.items()
        }
    )


def bind_shock_path_programs(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    supplied: frozenset[StateName],
    param_names: Mapping[RegimeName, frozenset[QualifiedName]],
) -> MappingProxyType[RegimeName, Regime]:
    """Replace realised lotteries while retaining solve and decision programs.

    The bound programs depend only on which shocks are supplied and on the
    parameter names; values arrive at dispatch. A regime drawing none of the
    supplied shocks keeps its programs.
    """
    all_grids = MappingProxyType(
        {name: regime.simulation.grids for name, regime in regimes.items()}
    )
    bound = {}
    for name, regime in regimes.items():
        if not any(
            f"next_{state}" in plan.lotteries
            for plan in regime.simulation.transition_plans.values()
            for state in supplied
        ):
            bound[name] = regime
            continue
        pools = regime.simulation.feasibility_pools_by_period
        groups = group_periods_by_key(
            active_periods=tuple(pools),
            key=lambda period: id(pools[period][0]),
        )
        built_by_period: dict[int, _PerSubjectFunction] = {}
        for periods in groups.values():
            built = _build_next_state_vmapped(
                active_periods=tuple(periods),
                phase_reachability=regime.simulation.reachability,
                source_regime_name=name,
                functions=pools[periods[0]][0],
                transitions=regime.simulation.transitions,
                transition_plans=regime.simulation.transition_plans,
                all_grids=all_grids,
                flat_param_names=param_names[name],
                enable_jit=False,
                supplied_states=supplied,
            )
            built_by_period.update(built.per_subject_by_period)
        transitions = {
            period: dataclasses.replace(
                program,
                function=functools.partial(
                    _SubjectTiled(
                        func=built_by_period[period].function,
                        subject_arg_names=built_by_period[period].subject_arg_names,
                    ),
                    **regime.resolved_fixed_params,
                ),
                argument_builder=_ArgumentsBoundAtDispatch(
                    program_name=program.name,
                    subject_arg_names=built_by_period[period].subject_arg_names,
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
    if not isinstance(builder, SubjectArgumentNames):
        return ()
    return builder.subject_arg_names
