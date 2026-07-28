"""The DC-EGM carry holds the rows the parent's Euler step reads, plus an
optional exact-policy row.

Backward induction threads the child's value, marginal utility, and endogenous
grid (plus the taste-shock scale) back to the parent. The parent's Euler step
never reads the child's optimal continuous action, so `policy` is `None` for
every ordinary carry and costs no pytree leaf there — the rolling
`next_regime_to_continuation` is the dominant device resident at scale, and a
write-only row would be wasted memory.

It is published in exactly one place: the continuous-only, jump-free ride-along
NB-EGM core, whose per-cell EGM step has already computed the optimal
consumption. The continuous-outer simulation replay reads it instead of
re-inverting `marginal_utility`, which is only valid at unit budget slope
(round-3 audit F2). So the field exists, and its *presence* is what varies.
"""

import jax

from _lcm.egm.carry import _EGM_CARRY_FIELDS, EGMCarry, build_template_egm_carry


def test_egm_carry_fields_include_the_optional_policy_row():
    """The carry's fields are the Euler-step rows plus the optional policy row.

    `policy` must appear in `_EGM_CARRY_FIELDS`: that tuple defines the pytree
    flatten/unflatten order, so omitting it would silently drop the row on every
    JAX transform round-trip.
    """
    assert set(_EGM_CARRY_FIELDS) == {
        "endog_grid",
        "value",
        "marginal_utility",
        "taste_shock_scale",
        "breakpoints",
        "policy",
    }
    field_names = {f.name for f in EGMCarry.__dataclass_fields__.values()}
    assert "policy" in field_names
    # Optional, and absent by default: `None` is a non-leaf empty subtree, so an
    # unpublished policy costs nothing and changes the tree structure when set.
    assert EGMCarry.__dataclass_fields__["policy"].default is None


def test_template_carry_has_no_policy_leaf():
    """The template carry exposes one leaf per kept field, none for policy."""
    template = build_template_egm_carry(n_rows=8, leading_shape=(3,))
    leaves = jax.tree_util.tree_leaves(template)
    # endog_grid, value, marginal_utility, taste_shock_scale.
    assert len(leaves) == 4
