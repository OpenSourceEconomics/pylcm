"""Admission of an `ExecutionConfig.action_partitions` request.

Sharing a regime's action product over several devices solves it on the
ordinary streamed hard-max `GridSearch` route, with each device reducing its own
run of whole action blocks and the devices merging compact accumulators. The
request is checked once, from the declarations, before any regime is built or
lowered. Every failed condition is named at once, so a request is either served
as asked or refused; an explicit device count is never reduced.
"""

import math
from collections.abc import Mapping
from types import MappingProxyType

from _lcm.processes.iid import _IIDProcess
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.processing import _declared_extent
from _lcm.solution.grid_search import ACTION_PRODUCT_AXIS, GridSearch
from _lcm.typing import RegimeName, StateName
from lcm.exceptions import ExecutionPlanningError

_REMEDY = (
    "Remove the regime from ExecutionConfig.action_partitions, or lower its count, "
    "to solve it on the ordinary route."
)


def fail_if_action_partition_route_is_unsupported(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    action_partitions: Mapping[RegimeName, int],
    sharded_states: frozenset[StateName],
    continuous_sharded_state: StateName | None,
    n_devices: int,
    fixed_widths_by_regime: Mapping[RegimeName, Mapping[str, int]],
) -> None:
    """Refuse a request the action-partitioned GridSearch route does not serve.

    A count of one asks for the ordinary route and is only checked for naming a
    regime. Above one, every failed condition is named:

    - a regime the model does not declare;
    - more devices than the model may use;
    - a terminal regime, a regime solved by another solver, or one declaring
      taste shocks, stakeholders, gated edges or same-period references;
    - a folded process or a discrete sharded state;
    - an action product with fewer than two actions, or fewer actions than
      devices;
    - a fixed action width leaving a device without a block.

    Raises:
        ExecutionPlanningError: The request is not served.

    """
    failures = [
        failure
        for regime_name, count in action_partitions.items()
        for failure in _request_failures(
            regime_name=regime_name,
            count=count,
            user_regimes=user_regimes,
            sharded_states=sharded_states,
            continuous_sharded_state=continuous_sharded_state,
            n_devices=n_devices,
            fixed_widths=fixed_widths_by_regime.get(regime_name, MappingProxyType({})),
        )
    ]
    if failures:
        msg = (
            "ExecutionConfig.action_partitions cannot be served: "
            + "; ".join(failures)
            + f". {_REMEDY}"
        )
        raise ExecutionPlanningError(msg)


def action_partition_width_ceilings(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    action_partitions: Mapping[RegimeName, int],
    fixed_widths_by_regime: Mapping[RegimeName, Mapping[str, int]],
) -> MappingProxyType[RegimeName, MappingProxyType[str, int]]:
    """Return the action-block ceiling each partitioned regime is planned under.

    At most `n_actions // count` identities per block cut the product into at
    least `count` blocks, so every device owns at least one. A regime whose
    action width is fixed keeps that width, which admission has checked, and
    a count of one adds no ceiling.
    """
    return MappingProxyType(
        {
            regime_name: MappingProxyType(
                {
                    ACTION_PRODUCT_AXIS: _n_actions(regime=user_regimes[regime_name])
                    // count
                }
            )
            for regime_name, count in action_partitions.items()
            if count > 1
            and ACTION_PRODUCT_AXIS
            not in fixed_widths_by_regime.get(regime_name, MappingProxyType({}))
        }
    )


def _request_failures(
    *,
    regime_name: RegimeName,
    count: int,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    sharded_states: frozenset[StateName],
    continuous_sharded_state: StateName | None,
    n_devices: int,
    fixed_widths: Mapping[str, int],
) -> list[str]:
    """Name what one regime's entry asks that the route does not serve."""
    if regime_name not in user_regimes:
        return [
            (
                f"no regime is named {regime_name!r}; declared regimes are "
                f"{sorted(user_regimes)!r}"
            )
        ]
    if count == 1:
        return []
    regime = user_regimes[regime_name]
    n_actions = _n_actions(regime=regime)
    discrete_sharded = sorted(
        name
        for name in regime.states
        if name in sharded_states and name != continuous_sharded_state
    )
    folded = sorted(
        name
        for name, grid in regime.states.items()
        if isinstance(grid, _IIDProcess) and grid.fold
    )
    fixed_width = fixed_widths.get(ACTION_PRODUCT_AXIS)
    checks = (
        (
            count > n_devices,
            f"needs {count} devices, but the model may use {n_devices}",
        ),
        (regime.terminal, "is terminal and has no action product to share"),
        (
            not isinstance(regime.solver, GridSearch),
            f"is solved by {type(regime.solver).__name__}, not GridSearch",
        ),
        (regime.taste_shocks is not None, "declares taste shocks"),
        (regime.stakeholders is not None, "is a collective regime"),
        (bool(regime.gated_edges), "declares gated edges"),
        (bool(regime.same_period_refs), "declares same-period references"),
        (bool(folded), f"folds the processes {folded!r}"),
        (
            bool(discrete_sharded),
            f"carries the discrete sharded states {discrete_sharded!r}",
        ),
        (
            not regime.terminal and n_actions < 2,  # noqa: PLR2004
            f"has {n_actions} action, so there is nothing to share",
        ),
        (
            not regime.terminal and 2 <= n_actions < count,  # noqa: PLR2004
            f"has {n_actions} actions, fewer than its {count} devices",
        ),
        (
            fixed_width is not None
            and n_actions >= count
            and math.ceil(n_actions / fixed_width) < count,
            (
                f"fixes {ACTION_PRODUCT_AXIS!r} at {fixed_width}, which cuts its "
                f"{n_actions} actions into fewer blocks than its {count} devices; "
                f"use at most {n_actions // count}"
            ),
        ),
    )
    return [
        f"regime {regime_name!r} ({count} partitions) {reason}"
        for failed, reason in checks
        if failed
    ]


def _n_actions(*, regime: FinalizedUserRegime) -> int:
    """Return the size of the regime's canonical action product."""
    return math.prod(
        _declared_extent(grid=grid)
        for grid in regime.actions.values()
        # A broadcast action a regime masks is declared `None` and has no axis.
        if grid is not None
    )
