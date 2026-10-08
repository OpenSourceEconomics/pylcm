"""Admission of an `ExecutionConfig.invariant_block_widths` request.

Blocking a state solves every regime carrying it one code at a time on the
ordinary hard-max `GridSearch` route. A request is checked twice, both times
before anything is lowered:

- `fail_if_invariant_blocking_route_is_unsupported` reads only the
  declarations, before the regimes are built, and refuses what the route
  does not serve;
- `admit_invariant_blocking` runs the placement-independent invariant analysis
  on the built model and refuses a state whose declarations do not establish
  that its value never changes and that no value read crosses its codes.

When the forward phase preserves the state too, forward simulation groups its
subjects by the state's code; otherwise it keeps the ungrouped route.
"""

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

from _lcm.engine import Regime
from _lcm.grids import DiscreteGrid
from _lcm.reachability import ModelReachability
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.fixed_components import FixedComponentSplit
from _lcm.regime_building.invariant_components import (
    analyze_invariant_components,
    fail_if_invariant_blocking_is_unsafe,
)
from _lcm.regime_law import RegimeLaw, RegimeLaws
from _lcm.simulation.subject_groups import SubjectGroupingRoute
from _lcm.solution.backward_induction import _value_axis_names
from _lcm.solution.grid_search import GridSearch
from _lcm.time import TimeAxis
from _lcm.typing import RegimeName, StateName
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import InvariantBlockSchedule

_REMEDY = (
    "Remove the state from ExecutionConfig.invariant_block_widths to solve it "
    "unblocked."
)


def bound_state_names(
    *,
    user_regime: FinalizedUserRegime,
    law: RegimeLaw,
    block_widths: Mapping[StateName, int],
    state_names: tuple[StateName, ...],
) -> tuple[StateName, ...]:
    """Return the blocked states one regime evaluates one code at a time.

    A non-terminal regime binds every blocked state it carries on a discrete
    grid. A terminal regime reads no continuation and stays unblocked.
    """
    if law.terminal:
        return ()
    return tuple(
        name
        for name in state_names
        if name in block_widths
        and isinstance(user_regime.states.get(name), DiscreteGrid)
    )


def fail_if_invariant_blocking_route_is_unsupported(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    laws: RegimeLaws,
    block_widths: Mapping[StateName, int],
    sharded_states: frozenset[StateName],
    schedule: InvariantBlockSchedule = InvariantBlockSchedule.PERIOD_MAJOR,
) -> None:
    """Refuse a request the blocked GridSearch route does not serve.

    Every failed condition is named at once:

    - more than one blocked state;
    - a width other than one, since each block holds a single code;
    - a blocked state that is also sharded;
    - a state no regime declares;
    - a non-terminal regime carrying the state that is solved by another
      solver, or declares taste shocks, stakeholders, gated edges or
      same-period references, or carries the state on a non-discrete grid;
    - under the block-major schedule, no blocked state at all, or a regime
      that does not carry the blocked state on a discrete grid: each code is
      solved as one component, and a regime without the state would be solved
      once per code.

    Folded processes and edge-reference reads are refused once the regimes are
    built, where they are resolved.

    Raises:
        ExecutionPlanningError: The request is not served.

    """
    if schedule is InvariantBlockSchedule.BLOCK_MAJOR and not block_widths:
        msg = (
            "ExecutionConfig.invariant_block_schedule=BLOCK_MAJOR solves one code "
            "of a blocked state at a time, but invariant_block_widths names no "
            "state. Name the state in invariant_block_widths, or keep the default "
            "InvariantBlockSchedule.PERIOD_MAJOR."
        )
        raise ExecutionPlanningError(msg)
    if not block_widths:
        return
    failures = [
        *_request_failures(
            user_regimes=user_regimes,
            block_widths=block_widths,
            sharded_states=sharded_states,
        ),
        *(
            failure
            for regime_name, regime in user_regimes.items()
            # A terminal regime reads no continuation and is solved unblocked.
            if not laws[regime_name].terminal
            for failure in _regime_failures(
                regime_name=regime_name,
                regime=regime,
                law=laws[regime_name],
                carried=tuple(name for name in block_widths if name in regime.states),
            )
        ),
        *(
            _block_major_failures(
                user_regimes=user_regimes, laws=laws, block_widths=block_widths
            )
            if schedule is InvariantBlockSchedule.BLOCK_MAJOR
            else ()
        ),
    ]
    if failures:
        msg = (
            "ExecutionConfig.invariant_block_widths cannot be served: "
            + "; ".join(failures)
            + f". {_REMEDY}"
        )
        raise ExecutionPlanningError(msg)


def _request_failures(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    block_widths: Mapping[StateName, int],
    sharded_states: frozenset[StateName],
) -> list[str]:
    """Name what the request itself asks that the route does not serve."""
    failures: list[str] = []
    if len(block_widths) > 1:
        failures.append(
            f"{len(block_widths)} states are blocked; only one state can be blocked"
        )
    for name, width in block_widths.items():
        if width != 1 or type(width) is not int:
            failures.append(
                f"state {name!r} has block width {width!r}; only width 1 is supported"
            )
        if name in sharded_states:
            failures.append(
                f"state {name!r} is also sharded; a sharded state cannot be blocked"
            )
        if not any(name in regime.states for regime in user_regimes.values()):
            failures.append(f"no regime declares a state {name!r}")
    return failures


def _block_major_failures(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    laws: RegimeLaws,
    block_widths: Mapping[StateName, int],
) -> list[str]:
    """Name every regime the block-major schedule cannot take components of."""
    return [
        (
            f"regime {regime_name!r} does not carry {name!r}, which the "
            "block-major schedule requires of every regime"
            if name not in regime.states
            else f"regime {regime_name!r} carries {name!r} on a non-discrete grid"
        )
        for name in block_widths
        for regime_name, regime in user_regimes.items()
        if not isinstance(regime.states.get(name), DiscreteGrid)
        # A non-terminal carrier's grid is already checked by the route itself.
        and not (regime.states.get(name) is not None and not laws[regime_name].terminal)
    ]


def _regime_failures(
    *,
    regime_name: RegimeName,
    regime: FinalizedUserRegime,
    law: RegimeLaw,
    carried: tuple[StateName, ...],
) -> list[str]:
    """Name what one non-terminal regime carrying a blocked state declares."""
    if not carried:
        return []
    checks = (
        (
            not isinstance(regime.solver, GridSearch),
            f"is solved by {type(regime.solver).__name__}, not GridSearch",
        ),
        (regime.taste_shocks is not None, "declares taste shocks"),
        (regime.stakeholders is not None, "is a collective regime"),
        (bool(law.gated_edges), "declares gated edges"),
        (bool(regime.same_period_refs), "declares same-period references"),
        *(
            (
                not isinstance(regime.states[name], DiscreteGrid),
                f"carries {name!r} on a non-discrete grid",
            )
            for name in carried
        ),
    )
    return [f"regime {regime_name!r} {reason}" for failed, reason in checks if failed]


def admit_invariant_blocking(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    laws: RegimeLaws,
    regimes: Mapping[RegimeName, Regime],
    reachability: ModelReachability,
    initial_nodes: frozenset[tuple[object, RegimeName]],
    ages: TimeAxis,
    fixed_component_splits: Mapping[StateName, FixedComponentSplit],
    block_widths: Mapping[StateName, int],
    schedule: InvariantBlockSchedule = InvariantBlockSchedule.PERIOD_MAJOR,
) -> MappingProxyType[RegimeName, Regime]:
    """Refuse an unsafe solve-phase request and group simulation where certified.

    Nothing is analysed when no state is blocked. Forward simulation groups
    subjects by the blocked state only when the simulate phase preserves it as
    well, and every regime draws through the ordinary grid decision: no regime
    declares taste shocks or gated edges, or replays a solve payload. Any other
    model keeps the ungrouped forward route.

    Returns:
        The regimes, whose forward programs carry the grouping route when
        grouping applies.

    Raises:
        ExecutionPlanningError: The solve-phase request is unsafe.

    """
    if not block_widths:
        return MappingProxyType(dict(regimes))
    if schedule is InvariantBlockSchedule.BLOCK_MAJOR:
        _fail_if_a_regime_drops_the_component_state(
            regimes=regimes, block_widths=block_widths
        )
    components = analyze_invariant_components(
        user_regimes=user_regimes,
        regimes=regimes,
        laws=laws,
        reachability=reachability,
        initial_nodes=initial_nodes,
        ages=ages,
        fixed_component_splits=fixed_component_splits,
    )
    fail_if_invariant_blocking_is_unsafe(
        components=components, block_widths=block_widths, phase="solve"
    )
    (state_name,) = block_widths
    component = components[state_name]
    if not component.simulate.eligible or any(
        regime.has_taste_shocks
        or regime.gated_edges
        or regime.simulation.replay_route.policy_applicable
        or regime.simulation.external_replay_route is not None
        for regime in regimes.values()
    ):
        return MappingProxyType(dict(regimes))
    route = SubjectGroupingRoute(
        state_name=state_name,
        codes=component.codes,
        value_axis_names=_value_axis_names(regimes=regimes),
    )
    return MappingProxyType(
        {
            name: dataclasses.replace(
                regime,
                simulation=dataclasses.replace(
                    regime.simulation,
                    programs=dataclasses.replace(
                        regime.simulation.programs, grouping=route
                    ),
                ),
            )
            for name, regime in regimes.items()
        }
    )


def _fail_if_a_regime_drops_the_component_state(
    *,
    regimes: Mapping[RegimeName, Regime],
    block_widths: Mapping[StateName, int],
) -> None:
    """Refuse the block-major schedule when a built regime solves without the state.

    A regime may declare the state and still never read it, in which case the
    built regime carries no axis for it. Its value is then shared by every
    code, and the component schedule would solve it once per code.

    Raises:
        ExecutionPlanningError: A built regime's value has no axis for the state.

    """
    (state_name,) = block_widths
    dropping = [
        name
        for name, regime in regimes.items()
        if state_name not in regime.solution.state_names
    ]
    if dropping:
        msg = (
            "ExecutionConfig.invariant_block_schedule=BLOCK_MAJOR requires every "
            f"regime's value to carry {state_name!r}, but regimes "
            f"{dropping!r} solve without it. Keep the default "
            "InvariantBlockSchedule.PERIOD_MAJOR for this model."
        )
        raise ExecutionPlanningError(msg)
