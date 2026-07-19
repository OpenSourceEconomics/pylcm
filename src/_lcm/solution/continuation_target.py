"""Deterministic-lifecycle target resolution and cross-regime params union.

Shared by the endogenous-grid solvers that thread a single deterministic
continuation target per period (`OneAssetEGM`, `TwoDimEGM`): resolve which
target regime each active period continues into, and bind or admit the union
of the source's and target's flat params in the kernel build.
"""

from types import MappingProxyType

from _lcm.solution.contract import (
    SolverBuildContext,
)
from _lcm.typing import (
    FlatParams,
    RegimeName,
)
from lcm.exceptions import RegimeInitializationError


def _period_to_continuation_target(
    *, context: SolverBuildContext
) -> dict[int, RegimeName]:
    """Resolve each active period's single deterministic continuation target.

    The model's deterministic lifecycle transition reaches exactly one target
    next period: the regime among this regime's transition targets that is
    active at `period + 1`. The last active period continues into the target
    active at the period beyond the horizon's interior (the terminal regime).
    """
    own_active = set(context.regimes_to_active_periods[context.regime_name])
    target_active = {
        target: set(context.regimes_to_active_periods[target])
        for target in context.transitions
    }
    result: dict[int, RegimeName] = {}
    for period in sorted(own_active):
        reached = [
            target for target, active in target_active.items() if (period + 1) in active
        ]
        if len(reached) != 1:
            msg = (
                f"Regime '{context.regime_name}' does not reach exactly one "
                f"active target at period {period + 1}: candidates {reached}. "
                "The endogenous-grid solvers require a deterministic "
                "lifecycle transition (one active target per period)."
            )
            raise RegimeInitializationError(msg)
        result[period] = reached[0]
    return result


def _union_free_params(
    *,
    flat_params: FlatParams,
    regime_name: RegimeName,
    transition_target_names: tuple[RegimeName, ...],
) -> dict[str, object]:
    """Union the regime's free params with its transition targets' free params.

    The boundary step evaluates the target regime's transition params (e.g. the
    pension payout factor the source never reads), so the core needs the union;
    captured functions read only the keys they need.
    """
    params: dict[str, object] = dict(flat_params[regime_name])
    for target_name in transition_target_names:
        for key, value in flat_params.get(target_name, MappingProxyType({})).items():
            params.setdefault(key, value)
    return params


def _union_fixed_params(
    *,
    fixed_flat_params: FlatParams,
    regime_name: RegimeName,
    transition_target_names: tuple[RegimeName, ...],
) -> dict[str, object]:
    """Union the regime's and its targets' fixed params for core binding."""
    bound = dict(fixed_flat_params.get(regime_name, MappingProxyType({})))
    for target_name in transition_target_names:
        for key, value in fixed_flat_params.get(
            target_name, MappingProxyType({})
        ).items():
            bound.setdefault(key, value)
    return bound
