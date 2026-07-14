"""Published continuous-action policy for off-grid DC-EGM forward simulation.

The DC-EGM solve recovers an exact, off-grid consumption function: Euler
inversion plus the upper envelope give the optimal continuous action on the
endogenous (resources-space) grid. `EGMSimPolicy` is the per-period snapshot of
that function — the off-grid policy a simulated subject's continuous action
*could* be interpolated from at its resources, rather than an argmax snapped to
the action grid.

It is produced and carried for *every* solved period alongside the
value-function arrays. Forward simulation consumes it where the regime
qualifies (`SimulationPhase.egm_policy_read`): the subject's row — indexed by
its discrete states and the chosen discrete action, with an off-grid passive
state blending the two bracketing rows — is interpolated at the subject's
resources, replacing the action-grid argmax value of the continuous action.

The gate exists because the stored policy is the **solve-phase** optimum. It
is valid at simulate time only when the solve-phase and simulate-phase
aggregators `H` coincide: with a phase-variant `H` (e.g. naive present bias —
solve under the exponential `δ`, simulate under `β̃β`) the stored policy
encodes the wrong continuous-action FOC. Such regimes — and regimes with EV1
taste shocks, whose realized draws perturb the decision — keep the
grid-argmax path, which re-applies the simulate-phase decision.

Unlike the rolling `EGMCarry` (the cross-period continuation channel, overwritten
each period), this is saved for every solved period and travels to `simulate`
alongside the value-function arrays. Its `endog_grid` rows are shared with the
period's carry; only the `policy` row is additional state.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax

from lcm.typing import ActionName, FloatND, StateName


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

    row_discrete_state_names: tuple[StateName, ...] = ()
    """Names of the leading discrete-state row axes, in axis order."""

    row_passive_state_names: tuple[StateName, ...] = ()
    """Names of the passive continuous-state row axes, after the discrete
    states."""

    row_discrete_action_names: tuple[ActionName, ...] = ()
    """Names of the discrete-action row axes, after the passive states."""


_EGM_SIM_POLICY_ARRAY_FIELDS = ("endog_grid", "policy")
_EGM_SIM_POLICY_STATIC_FIELDS = (
    "row_discrete_state_names",
    "row_passive_state_names",
    "row_discrete_action_names",
)


def _flatten_egm_sim_policy(
    policy: EGMSimPolicy,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    children = tuple(getattr(policy, name) for name in _EGM_SIM_POLICY_ARRAY_FIELDS)
    aux = tuple(getattr(policy, name) for name in _EGM_SIM_POLICY_STATIC_FIELDS)
    return children, aux


def _unflatten_egm_sim_policy(
    aux: tuple[Any, ...], children: Sequence[Any]
) -> EGMSimPolicy:
    policy = object.__new__(EGMSimPolicy)
    for name, child in zip(_EGM_SIM_POLICY_ARRAY_FIELDS, children, strict=True):
        object.__setattr__(policy, name, child)
    for name, value in zip(_EGM_SIM_POLICY_STATIC_FIELDS, aux, strict=True):
        object.__setattr__(policy, name, value)
    return policy


jax.tree_util.register_pytree_node(
    EGMSimPolicy, _flatten_egm_sim_policy, _unflatten_egm_sim_policy
)
