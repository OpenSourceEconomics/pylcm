"""Resolve a period's continuation reads when a target's grids vary by age.

A period-`t` EGM kernel interpolates each continuation target's `V_{t+1}` on the
target's own period-`t+1` grid. With an `AgeSpecializedGrid` those nodes move from
age to age, which puts two obligations on any solver that builds per-period kernels:

- read the interpolation info at `period + 1`, not the representative age's;
- keep periods whose continuation grids differ in different kernel-sharing groups,
  since one compiled program can only carry one set of nodes.

Both helpers below collapse to the age-invariant answer when the schedule is `None`,
so a model without an age-specialized state pays nothing and traces identically.
"""

from collections.abc import Hashable
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, cast

from _lcm.typing import RegimeName

if TYPE_CHECKING:
    from _lcm.regime_building.V import VInterpolationInfo

    RegimeToVInterpolationInfo: TypeAlias = MappingProxyType[  # noqa: UP040
        RegimeName, VInterpolationInfo
    ]
else:
    # `VInterpolationInfo`'s module imports `lcm.regime`, which closes a cycle
    # through the `lcm.solvers` façade and the solvers that call these helpers.
    # ty reads the precise element type above; at runtime the wider mapping is
    # all the annotation has to resolve to.
    RegimeToVInterpolationInfo = MappingProxyType


def continuation_v_interpolation_info(
    *,
    period: int,
    regime_to_v_interpolation_info: RegimeToVInterpolationInfo,
    period_to_regime_v_interp: (
        MappingProxyType[int, RegimeToVInterpolationInfo] | None
    ),
) -> RegimeToVInterpolationInfo:
    """Get all-regime interpolation info for `period`'s continuation `V_{t+1}`.

    Args:
        period: The period whose kernel does the reading.
        regime_to_v_interpolation_info: The representative-age info, which is the
            whole answer for every regime without an age-specialized state.
        period_to_regime_v_interp: `SolverBuildContext.period_to_regime_v_interp`,
            or `None` for an age-invariant model.

    Returns:
        Immutable mapping of regime name to the info the reader must use, with each
        age-specialized target resolved at `period + 1`.

    """
    if period_to_regime_v_interp is None:
        return regime_to_v_interpolation_info
    at_target_period = period_to_regime_v_interp.get(
        period + 1, cast("RegimeToVInterpolationInfo", MappingProxyType({}))
    )
    return MappingProxyType(
        {
            target: at_target_period.get(target, info)
            for target, info in regime_to_v_interpolation_info.items()
        }
    )


def continuation_grid_signature(
    *,
    period: int,
    targets: tuple[RegimeName, ...],
    period_to_regime_grid_signature: (
        MappingProxyType[int, MappingProxyType[RegimeName, Hashable]] | None
    ),
) -> Hashable:
    """Get the continuation targets' declared grid signatures at `period + 1`.

    Args:
        period: The period whose kernel does the reading.
        targets: The continuation targets the kernel reads.
        period_to_regime_grid_signature:
            `SolverBuildContext.period_to_regime_grid_signature`, or `None` for an
            age-invariant model.

    Returns:
        A hashable to fold into a kernel-sharing group key; empty when no target
        has an age-specialized grid, so the grouping is unchanged.

    """
    if period_to_regime_grid_signature is None:
        return ()
    at_target_period = period_to_regime_grid_signature.get(period + 1, {})
    return tuple(
        (target, at_target_period[target])
        for target in targets
        if target in at_target_period
    )
