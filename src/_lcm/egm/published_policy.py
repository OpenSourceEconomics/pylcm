"""Published continuous-action policy for off-grid DC-EGM forward simulation.

The DC-EGM solve recovers an exact, off-grid consumption function: Euler
inversion plus the upper envelope give the optimal continuous action on the
endogenous (resources-space) grid. `EGMSimPolicy` is the per-period snapshot of
that function the solve hands to simulation, so a simulated subject's continuous
action is the policy *interpolated* at its resources, not an argmax snapped to
the action grid.

Unlike the rolling `EGMCarry` (the cross-period continuation channel, overwritten
each period), this is saved for *every* solved period and travels to `simulate`
alongside the value-function arrays. Its `endog_grid` rows are shared with the
period's carry; only the `policy` row is additional state.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax

from lcm.typing import FloatND


@dataclass(frozen=True, kw_only=True)
class EGMSimPolicy:
    """Per-regime refined continuous-action policy on the resources grid.

    Leading axes match the regime's combo layout (discrete states, then passive
    states, then discrete actions, as in `EGMCarry`); the trailing axis is the
    static refined-grid length. Both rows are NaN-padded in lockstep in the tail.
    """

    endog_grid: FloatND
    """Endogenous grid in resources space, NaN-padded in the tail.

    Shared with the period's `EGMCarry.endog_grid`; weakly ascending per row,
    with envelope-kink abscissae duplicated.
    """

    policy: FloatND
    """Optimal continuous action at `endog_grid` (NaN on padding slots)."""


_EGM_SIM_POLICY_FIELDS = ("endog_grid", "policy")


def _flatten_egm_sim_policy(policy: EGMSimPolicy) -> tuple[tuple[Any, ...], None]:
    return tuple(getattr(policy, name) for name in _EGM_SIM_POLICY_FIELDS), None


def _unflatten_egm_sim_policy(_aux: None, children: Sequence[Any]) -> EGMSimPolicy:
    policy = object.__new__(EGMSimPolicy)
    for name, child in zip(_EGM_SIM_POLICY_FIELDS, children, strict=True):
        object.__setattr__(policy, name, child)
    return policy


jax.tree_util.register_pytree_node(
    EGMSimPolicy, _flatten_egm_sim_policy, _unflatten_egm_sim_policy
)
