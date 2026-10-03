"""Admission of an `ExecutionConfig.invariant_block_widths` request.

Blocking a state solves every regime carrying it one code at a time on the
ordinary hard-max `GridSearch` route. A request is checked twice, both times
before anything is lowered:

- `fail_if_invariant_blocking_route_is_unsupported` reads only the
  declarations, before the regimes are built, and refuses what the route
  does not serve;
- `fail_if_invariant_blocking_is_unsafe_for_model` runs the placement-independent
  invariant analysis on the built model and refuses a state whose
  declarations do not establish that its value never changes and that no
  value read crosses its codes.
"""

from collections.abc import Mapping

from _lcm.engine import Regime
from _lcm.grids import DiscreteGrid
from _lcm.reachability import ModelReachability
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.fixed_components import FixedComponentSplit
from _lcm.regime_building.invariant_components import (
    analyze_invariant_components,
    fail_if_invariant_blocking_is_unsafe,
)
from _lcm.solution.grid_search import GridSearch
from _lcm.typing import RegimeName, StateName
from lcm.ages import AgeGrid
from lcm.exceptions import ExecutionPlanningError

_REMEDY = (
    "Remove the state from ExecutionConfig.invariant_block_widths to solve it "
    "unblocked."
)


def bound_state_names(
    *,
    user_regime: FinalizedUserRegime,
    block_widths: Mapping[StateName, int],
    state_names: tuple[StateName, ...],
) -> tuple[StateName, ...]:
    """Return the blocked states one regime evaluates one code at a time.

    A non-terminal regime binds every blocked state it carries on a discrete
    grid. A terminal regime reads no continuation and stays unblocked.
    """
    if user_regime.terminal:
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
    block_widths: Mapping[StateName, int],
    sharded_states: frozenset[StateName],
) -> None:
    """Refuse a request the blocked GridSearch route does not serve.

    Every failed condition is named at once:

    - more than one blocked state;
    - a width other than one, since each block holds a single code;
    - a blocked state that is also sharded;
    - a state no regime declares;
    - a non-terminal regime carrying the state that is solved by another
      solver, or declares taste shocks, stakeholders, gated edges or
      same-period references, or carries the state on a non-discrete grid.

    Folded processes and edge-reference reads are refused once the regimes are
    built, where they are resolved.

    Raises:
        ExecutionPlanningError: The request is not served.

    """
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
            if not regime.terminal
            for failure in _regime_failures(
                regime_name=regime_name,
                regime=regime,
                carried=tuple(name for name in block_widths if name in regime.states),
            )
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


def _regime_failures(
    *,
    regime_name: RegimeName,
    regime: FinalizedUserRegime,
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
        (bool(regime.gated_edges), "declares gated edges"),
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


def fail_if_invariant_blocking_is_unsafe_for_model(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    regimes: Mapping[RegimeName, Regime],
    reachability: ModelReachability,
    initial_nodes: frozenset[tuple[object, RegimeName]],
    ages: AgeGrid,
    fixed_component_splits: Mapping[StateName, FixedComponentSplit],
    block_widths: Mapping[StateName, int],
) -> None:
    """Analyse the built model and refuse an unsafe solve-phase request.

    Nothing is analysed when no state is blocked.
    """
    if not block_widths:
        return
    components = analyze_invariant_components(
        user_regimes=user_regimes,
        regimes=regimes,
        reachability=reachability,
        initial_nodes=initial_nodes,
        ages=ages,
        fixed_component_splits=fixed_component_splits,
    )
    fail_if_invariant_blocking_is_unsafe(
        components=components, block_widths=block_widths, phase="solve"
    )
