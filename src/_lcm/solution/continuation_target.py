"""Deterministic-lifecycle target resolution and cross-regime params union.

Shared by the endogenous-grid solvers that thread a single deterministic
continuation target per period (`EGM`, `TwoAssetEGM`): resolve which
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
from lcm.typing import Float1D, StateName


def _period_to_continuation_target(
    *, context: SolverBuildContext
) -> dict[int, RegimeName]:
    """Resolve each active period's single deterministic continuation target.

    The canonical solution graph must retain exactly one target from every
    active source period represented in backward induction.
    """
    result: dict[int, RegimeName] = {}
    for period, active_regimes in enumerate(
        context.solution_reachability.active_regimes_by_period[:-1]
    ):
        if context.regime_name not in active_regimes:
            continue
        reached = context.solution_reachability.targets(
            period=period, source=context.regime_name
        )
        if len(reached) != 1:
            msg = (
                f"Regime '{context.regime_name}' does not reach exactly one "
                f"active target at period {period + 1}: candidates {list(reached)}. "
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


def target_period_grid(
    *,
    context: SolverBuildContext,
    period: int,
    target: RegimeName,
    target_state_name: StateName,
) -> Float1D:
    """The nodes `target` tabulates `target_state_name` on at `period + 1`.

    A period-`t` kernel reads `V_{t+1}` and its marginal on the *target's*
    grid, which differs from this regime's on two independent axes: the target
    may be a different regime with its own grid, and the state may be
    age-specialized so the same regime's nodes move between periods.

    Args:
        context: The solver build context of the regime doing the reading.
        period: The period whose kernel reads the continuation, so the grid
            wanted is the target's at `period + 1`.
        target: The regime the continuation is read from.
        target_state_name: That regime's own name for the state the
            continuation is tabulated on.

    Returns:
        The target's nodes for that state at `period + 1`.

    Raises:
        RegimeInitializationError: If the target does not carry the state at
            all, so no continuation for it exists to read.

    """
    representative = context.regime_to_v_interpolation_info[target].continuous_states
    if target_state_name not in representative:
        msg = (
            f"Regime '{context.regime_name}' reads its continuation from target "
            f"regime '{target}', which does not carry the state "
            f"'{target_state_name}' (its continuous states are "
            f"{sorted(representative)}). There is no continuation for that state "
            f"to read."
        )
        raise RegimeInitializationError(msg)

    per_period = context.period_to_regime_v_interp
    if per_period is not None:
        info = per_period.get(period + 1, {}).get(target)
        if info is not None and target_state_name in info.continuous_states:
            return info.continuous_states[target_state_name].to_jax()
    # No age-specialized state anywhere in the model: the target's
    # representative grid is its grid in every period. It is still the
    # target's, though, never this regime's.
    return representative[target_state_name].to_jax()
