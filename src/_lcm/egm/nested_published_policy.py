"""The nested (continuous-outer) simulation-policy payload.

A continuous-outer solve is only complete if simulation can *replay the same
continuous policy class*: re-decide keeper vs adjuster off-grid, re-run the
same interpolant/search across the outer axis, and read the inner
consumption policy at the subject's actual resources. Storing only the final
outer action on the state grid cannot do that — simulation arrives off the
liquid grid, and a different branch can win there than at the nearest grid
node. So the payload keeps the *conditional* ingredients:

- the keeper's inner `EGMSimPolicy`;
- one `EGMSimPolicy` per shared outer mesh node, stacked along a leading
  candidate axis (`OuterPolicyBank`) — each is the exact inner solve's
  published policy conditional on that outer action;
- the static names and search settings the reader needs to rebuild the
  outer-value interpolant and refine it per subject.

All rows are re-reads on the *shared liquid state grid* (the NB-EGM inner's
`carry_rows_share_state_grid` contract), so every branch is queried at the
same abscissa — the subject's liquid state. Each branch's conditional
resources shift (the keeper's held durable, each candidate's credited outer
move) is already inside its conditional solve, not applied at read time.

Both containers are registered as JAX pytrees (explicit
`register_pytree_node`, matching `EGMSimPolicy` / `EGMCarry`).
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.published_policy import EGMSimPolicy
from lcm.typing import ActionName, Float1D, FloatND, FunctionName, StateName


def derive_inner_sim_policy(
    *,
    carry: EGMCarry,
    state_grid_values: Float1D,
    inverse_marginal: Callable[..., FloatND] | None,
    row_discrete_state_names: tuple[StateName, ...],
    row_passive_state_names: tuple[StateName, ...],
    extra_leading_axes: int = 0,
) -> EGMSimPolicy | None:
    """Derive the published inner policy from a state-grid NB-EGM carry.

    An NB-EGM inner publishes its carry rows *re-read on the shared liquid
    state grid* (`carry_rows_share_state_grid`), so the row abscissae are the
    state grid itself and no consumption row survives the solve. On the
    smooth v1 scope the envelope theorem recovers it exactly at the nodes:
    the row's marginal is `u'(c)` (interior optimum and the credit-constrained
    corner alike, since consumption absorbs a marginal unit of the liquid
    state in both), so `c = inverse_marginal(marginal_continuation=mu)` —
    the same closed-form inverse the Euler inversion used. Nodes with a
    nonfinite value or a nonpositive marginal (the infeasible-row `0.0`
    convention) publish NaN, which the simulation read's acceptance check
    rejects.

    Fails closed (returns `None`, so the caller publishes no nested payload
    and simulation keeps the grid-argmax path) when the closed-form inverse
    is unavailable, the rows are not on the state grid, the carry keeps axes
    the given row names do not describe (`extra_leading_axes` covers a
    candidate-stacked carry's leading axis), or the rows carry declared
    topology (`breakpoints`).

    Works elementwise, so a candidate-stacked carry (leading axis `C`)
    yields the candidate-stacked policy.
    """
    expected_ndim = (
        extra_leading_axes
        + len(row_discrete_state_names)
        + len(row_passive_state_names)
        + 1
    )
    if (
        inverse_marginal is None
        or carry.value.shape[-1] != state_grid_values.shape[0]
        or carry.value.ndim != expected_ndim
        or carry.breakpoints is not None
    ):
        return None
    node_valid = jnp.isfinite(carry.value) & (carry.marginal_utility > 0.0)
    policy = jnp.where(
        node_valid,
        inverse_marginal(
            marginal_continuation=jnp.where(node_valid, carry.marginal_utility, 1.0)
        ),
        jnp.nan,
    )
    return EGMSimPolicy(
        endog_grid=carry.endog_grid,
        policy=policy,
        value=carry.value,
        marginal_utility=carry.marginal_utility,
        row_discrete_state_names=row_discrete_state_names,
        row_passive_state_names=row_passive_state_names,
        row_discrete_action_names=(),
    )


@dataclass(frozen=True, kw_only=True)
class OuterPolicyBank:
    """Per-outer-node published inner policies on the shared mesh.

    `policies` is an `EGMSimPolicy` whose every array leaf carries a leading
    candidate axis of length `C = outer_nodes.shape[0]` (the stacked form
    `OuterCandidateBank` collects); the row-name metadata applies to the
    axes *after* that leading candidate axis.
    """

    outer_nodes: Float1D
    """Shared outer mesh nodes, shape `(C,)`, strictly increasing."""

    policies: EGMSimPolicy
    """The candidate-stacked published policies (leading axis `C`)."""

    @property
    def n_candidates(self) -> int:
        """Number of outer candidate nodes in the bank."""
        return int(self.outer_nodes.shape[0])


def _flatten_outer_policy_bank(
    bank: OuterPolicyBank,
) -> tuple[tuple[Float1D, EGMSimPolicy], None]:
    return (bank.outer_nodes, bank.policies), None


def _unflatten_outer_policy_bank(
    _aux: None, children: tuple[Float1D, EGMSimPolicy]
) -> OuterPolicyBank:
    bank = object.__new__(OuterPolicyBank)
    object.__setattr__(bank, "outer_nodes", children[0])
    object.__setattr__(bank, "policies", children[1])
    return bank


jax.tree_util.register_pytree_node(
    OuterPolicyBank, _flatten_outer_policy_bank, _unflatten_outer_policy_bank
)


@dataclass(frozen=True, kw_only=True)
class NestedEGMSimPolicy:
    """Keeper plus conditional adjuster policies for continuous replay.

    The simulation reader indexes both sides' rows at the subject's discrete
    and passive states, reads every branch's value at the subject's liquid
    state (the keeper holds the durable via the no-adjustment map; each
    adjuster candidate's solve already binds the outer post-decision to its
    node), rebuilds the outer-value interpolant from the conditional value
    reads, refines with the same safeguarded search the solve used, and
    compares against the keeper.
    """

    keeper: EGMSimPolicy
    """The keeper branch's ordinary published policy."""

    adjuster: OuterPolicyBank
    """The conditional adjuster policies on the shared outer mesh."""

    outer_action_name: ActionName
    """The regime's outer continuous action the reader replaces."""

    outer_post_decision_name: FunctionName
    """The outer post-decision the candidate nodes are values of."""

    inner_action_name: ActionName
    """The inner continuous action (consumption) the reader replaces."""

    resources_target_name: FunctionName
    """DAG function computing the resources each row is read at."""

    savings_lower_bound: float
    """Inner savings-grid lower bound for the intrinsic budget check."""

    golden_iterations: int
    """Static golden-section budget of the subject-level outer refinement —
    the same setting the solve's search used."""


_NESTED_STATIC_FIELDS = (
    "outer_action_name",
    "outer_post_decision_name",
    "inner_action_name",
    "resources_target_name",
    "savings_lower_bound",
    "golden_iterations",
)


def _flatten_nested_egm_sim_policy(
    policy: NestedEGMSimPolicy,
) -> tuple[tuple[EGMSimPolicy, OuterPolicyBank], tuple[object, ...]]:
    aux = tuple(getattr(policy, name) for name in _NESTED_STATIC_FIELDS)
    return (policy.keeper, policy.adjuster), aux


def _unflatten_nested_egm_sim_policy(
    aux: tuple[object, ...], children: tuple[EGMSimPolicy, OuterPolicyBank]
) -> NestedEGMSimPolicy:
    policy = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(policy, "keeper", children[0])
    object.__setattr__(policy, "adjuster", children[1])
    for name, value in zip(_NESTED_STATIC_FIELDS, aux, strict=True):
        object.__setattr__(policy, name, value)
    return policy


jax.tree_util.register_pytree_node(
    NestedEGMSimPolicy,
    _flatten_nested_egm_sim_policy,
    _unflatten_nested_egm_sim_policy,
)
