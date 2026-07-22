"""The DC-EGM carry holds only the rows the parent's Euler step reads.

Backward induction threads the child's value, marginal utility, and endogenous
grid (plus the taste-shock scale) back to the parent; the parent never reads the
child's optimal continuous action. The carry therefore does not retain a policy
row — the rolling `next_regime_to_continuation` is the dominant device resident at
scale, so a write-only row would be pure wasted memory.
"""

import jax

from _lcm.egm.carry import _EGM_CARRY_FIELDS, EGMCarry, build_template_egm_carry


def test_egm_carry_fields_exclude_policy():
    """The carry's fields are exactly the rows the parent Euler step consumes."""
    assert set(_EGM_CARRY_FIELDS) == {
        "endog_grid",
        "value",
        "marginal_utility",
        "taste_shock_scale",
    }
    assert not hasattr(EGMCarry, "policy")
    field_names = {f.name for f in EGMCarry.__dataclass_fields__.values()}
    assert "policy" not in field_names


def test_template_carry_has_no_policy_leaf():
    """The template carry exposes one leaf per kept field, none for policy."""
    template = build_template_egm_carry(n_rows=8, leading_shape=(3,))
    leaves = jax.tree_util.tree_leaves(template)
    # endog_grid, value, marginal_utility, taste_shock_scale.
    assert len(leaves) == 4
