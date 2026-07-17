"""Published continuous-action policy for off-grid DC-EGM forward simulation.

The DC-EGM solve recovers the optimal continuous action off the action grid:
Euler inversion plus the upper envelope give it exactly at each node of the
endogenous (resources-space) grid. `EGMSimPolicy` is the per-period snapshot of
those nodes — a refined off-grid policy interpolant under the selected envelope
convention, which a simulated subject's continuous action *could* be read from
at its resources, rather than an argmax snapped to the action grid. Between
nodes the read carries the interpolation error of a finite row; the envelope
gate below buys branch faithfulness at the switches, not exactness within a
branch.

It is produced and carried for *every* solved period alongside the
value-function arrays. Forward simulation consumes it where the regime
qualifies (`SimulationPhase.egm_policy_read`): the subject's row — indexed by
its discrete states — is interpolated at the subject's resources, replacing the
action-grid argmax value of the continuous action, subject to a post-read
feasibility check (in-support, finite, positive, within the intrinsic budget).

Regimes with discrete actions publish one conditional row per discrete-action
combination — value and policy per branch, on that branch's own endogenous
grid. Simulation then *re-decides* the branch at the subject's state: each
branch's conditional value is interpolated at that branch's own resources
(discrete-only constraints mask infeasible branches to `-inf`), the feasible
branch of highest interpolated value wins, and only the winner's policy is
read. The value read uses the cubic Hermite interpolant with the
`marginal_utility` row supplied as the node-slope input (the economic marginal
at each node, Fritsch-Carlson-limited inside the interpolant) — the same
convention the solve uses to publish values — so the ranking the re-decision
sees is the ranking the solve convention implies.

The gate exists because the stored rows are the **solve-phase** optimum of
one conditional problem each, and a read is faithful only where the rows
carry the coordinates and branch topology they are interpolated over. Kept on
the grid-argmax path:
- regimes with any `Phased` declaration — a phase-variant utility, budget,
  transition, or state domain (not only `H`, e.g. naive present bias) makes
  the stored policy solve the wrong simulate-phase FOC or puts the policy
  rows on the wrong coordinates;
- regimes whose upper-envelope backend does not certify every crossing —
  only MSS enumerates the complete envelope-switch sequence (interior
  crossings, exact-node switches, value jumps; loud overflow past its
  budget); FUES decides segment identity by a slope-threshold heuristic (no
  labels from the kernel), so its row can silently bridge a missed switch,
  and RFC/LTM leave switches between retained nodes;
- regimes with a passive continuous state — each row is the envelope policy
  conditional on one passive node, so blending rows across a passive-dimension
  branch switch would read an action from neither branch;
- regimes with a continuous stochastic-process state — the process is a
  node-valued row axis, but its simulation transition draws an off-node
  continuous value that nearest-node row selection cannot resolve;
- asset-row DC-EGM regimes (a savings-stage function reads the Euler state) —
  the per-node solve publishes one point per exogenous asset node, not a
  crossing-complete row, so interpolating across nodes would mix branches;
- regimes with EV1 taste shocks, whose realized draws perturb the decision.
Publishing per-passive-node / per-process-node conditional values and
re-deciding across those axes the way the discrete-action axis already is
lifts the passive, process, and asset-row exclusions (the tracked follow-up).

Unlike the rolling `EGMCarry` (the cross-period continuation channel, overwritten
each period), this is saved for every solved period and travels to `simulate`
alongside the value-function arrays. Its rows are shared with the period's
carry; only the `policy` row is additional state.
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
    static refined-grid length. All rows are NaN-padded in lockstep in the tail.
    """

    endog_grid: FloatND
    """Endogenous grid in resources space, NaN-padded in the tail.

    Shared with the period's `EGMCarry.endog_grid`; weakly ascending per row.
    Under MSS the envelope-switch abscissae are duplicated with one-sided
    policy copies — the topology the off-grid read requires; FUES rows are
    not certified crossing-complete (segment identity is a slope-threshold
    heuristic) and RFC/LTM rows leave switches between retained nodes, so
    neither qualifies for the read.
    """

    policy: FloatND
    """Optimal continuous action at `endog_grid` (NaN on padding slots)."""

    value: FloatND
    """Conditional value at `endog_grid` (NaN on padding slots).

    Shared with the period's `EGMCarry.value`: the row's combo-conditional
    value function on the refined resources grid. Simulation compares the
    interpolated conditional values across discrete-action rows to re-decide
    the branch at the subject's state.
    """

    marginal_utility: FloatND
    """Marginal utility at `endog_grid` (NaN on padding slots).

    Shared with the period's `EGMCarry.marginal_utility`: the economic
    marginal `u'(c)` at each node — the value row's slope by the envelope
    theorem at solve nodes. Simulation passes it as the slope input of the
    cubic Hermite value read (Fritsch-Carlson-limited inside the
    interpolant), matching the interpolation convention the solve publishes
    values under.
    """

    row_discrete_state_names: tuple[StateName, ...] = ()
    """Names of the leading discrete-state row axes, in axis order."""

    row_passive_state_names: tuple[StateName, ...] = ()
    """Names of the passive continuous-state row axes, after the discrete
    states."""

    row_discrete_action_names: tuple[ActionName, ...] = ()
    """Names of the discrete-action row axes, after the passive states."""


_EGM_SIM_POLICY_ARRAY_FIELDS = ("endog_grid", "policy", "value", "marginal_utility")
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
