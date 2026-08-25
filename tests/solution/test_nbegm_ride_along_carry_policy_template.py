"""The standalone ride-along NBEGM continuation pytree carries its policy leaf.

On the continuous-only, jump-free ride-along path, `_assemble_ride_carry` populates
the exact-consumption `EGMCarry.policy` leaf, so the runtime carry
has `policy=array`. JAX treats `policy=None` and `policy=array` as DIFFERENT pytree
structures (`None` is an empty subtree, not a leaf), so the lowering template
(`_build_ride_along_carry_template`) must carry the same leaf under the same
predicate. Otherwise a *standalone* ride-along NBEGM regime — whose continuation is
rolled cross-period and lowered against the template — sees a template pytree that
differs from the runtime carry, and the compiled solve rejects it.

These tests pin the template's pytree against the real runtime producer
(`_assemble_ride_carry`) on both the continuous-only and discrete-branch
paths, so a regression to a policy-free continuous template fails structurally
without needing the full model solve.
"""

import jax
import jax.numpy as jnp
import pytest

from _lcm.dtypes import canonical_float_dtype
from _lcm.solution.nbegm import _assemble_ride_carry, _build_ride_along_carry_template

_RIDE_SHAPE = (4,)
_N_LIQUID = 6


def _runtime_carry(*, carry_policy: bool):
    """The carry `_assemble_ride_carry` builds on the jump-free ride-along path."""
    dtype = canonical_float_dtype()
    liquid = jnp.linspace(0.1, 30.0, _N_LIQUID, dtype=dtype)
    n_cell = _RIDE_SHAPE[0] * _N_LIQUID
    value_stack = jnp.zeros((n_cell,), dtype=dtype)
    marginal_stack = jnp.ones((n_cell,), dtype=dtype)
    policy_stack = jnp.ones((n_cell,), dtype=dtype)
    if carry_policy:
        n_action_branches = 0
        stacks = (value_stack, marginal_stack, policy_stack)
    else:
        n_action_branches = 2
        branch_shape = (n_cell * n_action_branches,)
        stacks = (
            value_stack,
            marginal_stack,
            policy_stack,
            jnp.zeros(branch_shape, dtype=dtype),
            jnp.ones(branch_shape, dtype=dtype),
        )
    _, carry, *_ = _assemble_ride_carry(
        stacks=stacks,
        n_jumps=0,
        n_action_branches=n_action_branches,
        liquid=liquid,
        ride_shape=_RIDE_SHAPE,
        liquid_axis_pos=0,
        dtype=dtype,
    )
    return carry


def _template(*, carry_policy: bool):
    dtype = canonical_float_dtype()
    return _build_ride_along_carry_template(
        liquid_grid=jnp.linspace(0.1, 30.0, _N_LIQUID, dtype=dtype),
        ride_shape=_RIDE_SHAPE,
        n_breakpoints=0,
        carry_policy=carry_policy,
    )


@pytest.mark.parametrize("carry_policy", [True, False])
def test_ride_along_template_pytree_matches_runtime_carry(carry_policy):
    """The template pytree structure equals the runtime carry's on both paths."""
    template = _template(carry_policy=carry_policy)
    runtime = _runtime_carry(carry_policy=carry_policy)
    assert jax.tree.structure(template) == jax.tree.structure(runtime)
    # And the discriminating leaf itself: policy present iff carry_policy.
    assert (template.policy is None) == (not carry_policy)
    assert (runtime.policy is None) == (not carry_policy)


def test_carry_policy_toggles_the_pytree_structure():
    """The policy leaf's presence is what distinguishes the two structures.

    Guards against a regression where the template drops the policy leaf on the
    continuous-only path: the two carry_policy structures must differ,
    so a policy-free template could not match the policy-carrying runtime carry.
    """
    assert jax.tree.structure(_template(carry_policy=True)) != jax.tree.structure(
        _template(carry_policy=False)
    )
