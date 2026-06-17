"""Published continuous-action policy for off-grid DC-EGM forward simulation.

The DC-EGM solve recovers an exact, off-grid consumption function: Euler
inversion plus the upper envelope give the optimal continuous action on the
endogenous (resources-space) grid. `EGMSimPolicy` is the per-period snapshot of
that function — the off-grid policy a simulated subject's continuous action
*could* be interpolated from at its resources, rather than an argmax snapped to
the action grid.

It is produced and carried for *every* solved period alongside the
value-function arrays, but it is **not consumed by `simulate` today**: forward
simulation re-optimizes the gridded continuous action via
`argmax_and_max_Q_over_a` (the grid-restricted path documented on `DCEGM`), so
simulated actions live on the action grid. `EGMSimPolicy` is built ahead of an
off-grid simulation path that is not yet wired in.

Invariant for whoever wires it in: the stored policy is the **solve-phase**
optimum. It may be used at simulate time only when the solve-phase and
simulate-phase aggregators `H` coincide. With a phase-variant `H` (e.g. naive
present bias — solve under the exponential `δ`, simulate under `β̃β`) the stored
policy encodes the wrong continuous-action FOC; such models must keep the
re-optimization path, which re-applies the simulate-phase decision. A regression
should assert that DC-EGM simulation of a present-bias model differs from the
exponential one in the expected direction before any off-grid lookup replaces
the re-optimization.

Unlike the rolling `EGMCarry` (the cross-period continuation channel, overwritten
each period), this is saved for every solved period and travels to `simulate`
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
