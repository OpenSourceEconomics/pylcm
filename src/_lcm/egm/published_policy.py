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

A requesting solve can retain it alongside the value-function arrays. Forward
simulation requests it only when a regime qualifies
(`SimulationPhase.egm_policy_read`): the subject's row — indexed by
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
  transition, or state domain (not only `W`, e.g. naive present bias) makes
  the stored policy solve the wrong simulate-phase FOC or puts the policy
  rows on the wrong coordinates;
- every currently shipped upper-envelope backend — none has yet passed the
  crossing- and support-completeness contract required by the read. MSS can
  omit an owner that wins only inside one candidate interval; FUES can merge
  branches when its slope heuristic misses a switch; RFC and LTM leave
  switches between retained nodes; and the exact backend does not yet publish
  a support verdict for compacted gaps;
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
each period), this is retained only when inspection or an eligible simulation
read requests it. Its rows are shared with the period's carry; only the `policy`
row is additional state.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax

from lcm.typing import ActionName, FloatND, IntND, StateName


@dataclass(frozen=True, kw_only=True)
class EGMSimPolicy:
    """Per-regime refined continuous-action policy on the resources grid.

    Leading axes match the regime's combo layout (discrete states, then passive
    states, then discrete actions, as in `EGMCarry`); the trailing axis is the
    static refined-grid length. All rows are NaN-padded in lockstep in the tail.

    A row whose upper envelope reports no read support (an interior coverage
    gap: NaN-dead candidates or a finite value decrease split the segment
    chain, so a linear read would bridge the gap with fabricated values) is
    published fully NaN — the simulation's acceptance check then rejects it
    and falls back to the grid-argmax decision. The solve-side carry keeps
    its compacted rows; only the published read fail-closes.
    """

    endog_grid: FloatND
    """Endogenous grid in resources space, NaN-padded in the tail.

    Shared with the period's `EGMCarry.endog_grid`; weakly ascending per row.
    The off-grid read requires duplicated one-sided records at every switch
    and an explicit support verdict for every compacted gap. No shipped
    envelope currently proves both properties, so every backend remains on the
    grid-argmax simulation path.
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


@dataclass(frozen=True, kw_only=True)
class NBEGMGridPolicy:
    """Inner NBEGM action on the regime state grid.

    This is an intermediate solve artifact. A nested outer solver stacks one
    instance per candidate before publishing the joint replay payload. A
    discrete inner envelope additionally retains every branch's conditional
    action and value; smooth solves leave those optional banks absent so their
    hot path and storage stay unchanged.
    """

    action: FloatND
    """Collapsed optimal inner action, aligned with ``state_names`` axes."""

    state_names: tuple[StateName, ...]
    """State-grid axis names in the array order of ``action``."""

    branch_inner_action: FloatND | None = None
    """Shape ``(n_discrete, *state_shape)`` before the branch upper envelope."""

    branch_value: FloatND | None = None
    """Conditional values on exactly the same branch/state axes."""

    branch_discrete_actions: IntND | None = None
    """Exact code matrix of shape ``(n_discrete, n_discrete_actions)``."""

    discrete_action_names: tuple[ActionName, ...] = ()
    """Columns of ``branch_discrete_actions`` in declared product order."""


@dataclass(frozen=True, kw_only=True)
class NNBEGMSimPolicy:
    """Candidate-aligned NNBEGM joint policies for forward replay.

    Candidate order is outer-major: the state-specific keeper first, then
    ``NNBEGM.outer_grid`` order; within each outer candidate, the inner
    solver's declared discrete Cartesian-product order. Continuous surfaces
    and represented solve values share that leading candidate axis. Discrete
    codes are exact metadata and are selected, never interpolated. Simulation
    retains this finite-set ordering and canonical-scores the complete selected
    tuple once.
    """

    candidate_inner_action: FloatND
    """Shape ``(n_candidates, *state_shape)`` in solve candidate order."""

    candidate_outer_action: FloatND
    """Outer action on exactly the same candidate/state axes."""

    candidate_value: FloatND
    """Conditional solve value on exactly the same candidate/state axes."""

    state_names: tuple[StateName, ...]
    """State-grid axis names following the leading candidate axis."""

    inner_action_name: ActionName
    outer_action_name: ActionName

    candidate_discrete_actions: IntND | None = None
    """Exact shape ``(n_candidates, n_discrete_actions)`` code metadata."""

    discrete_action_names: tuple[ActionName, ...] = ()
    """Columns of ``candidate_discrete_actions`` in declaration order."""


def _flatten_grid_policy(
    policy: NBEGMGridPolicy,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    aux = (policy.state_names, policy.discrete_action_names)
    return (
        policy.action,
        policy.branch_inner_action,
        policy.branch_value,
        policy.branch_discrete_actions,
    ), aux


def _unflatten_grid_policy(
    aux: tuple[Any, ...], children: Sequence[Any]
) -> NBEGMGridPolicy:
    state_names, discrete_action_names = aux
    return NBEGMGridPolicy(
        action=children[0],
        branch_inner_action=children[1],
        branch_value=children[2],
        branch_discrete_actions=children[3],
        state_names=state_names,
        discrete_action_names=discrete_action_names,
    )


def _flatten_nnbegm_policy(
    policy: NNBEGMSimPolicy,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    aux = (
        policy.state_names,
        policy.inner_action_name,
        policy.outer_action_name,
        policy.discrete_action_names,
    )
    return (
        policy.candidate_inner_action,
        policy.candidate_outer_action,
        policy.candidate_value,
        policy.candidate_discrete_actions,
    ), aux


def _unflatten_nnbegm_policy(
    aux: tuple[Any, ...], children: Sequence[Any]
) -> NNBEGMSimPolicy:
    state_names, inner_action_name, outer_action_name, discrete_action_names = aux
    return NNBEGMSimPolicy(
        candidate_inner_action=children[0],
        candidate_outer_action=children[1],
        candidate_value=children[2],
        candidate_discrete_actions=children[3],
        state_names=state_names,
        inner_action_name=inner_action_name,
        outer_action_name=outer_action_name,
        discrete_action_names=discrete_action_names,
    )


jax.tree_util.register_pytree_node(
    NBEGMGridPolicy, _flatten_grid_policy, _unflatten_grid_policy
)
jax.tree_util.register_pytree_node(
    NNBEGMSimPolicy, _flatten_nnbegm_policy, _unflatten_nnbegm_policy
)
