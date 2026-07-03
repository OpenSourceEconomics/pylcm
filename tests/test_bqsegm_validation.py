"""Smoothness gate for the declared functions of a BQSEGM case split.

BQSEGM only works when each piece is smooth within its case, so hidden branching
in a declared formula — a Python `if`, a comparison, a `jnp.where` buried in a
helper — must be caught at build time. The gate runs on the boundary predicates
(AST-only boundary mode) and the declared `when`/`otherwise` pieces (AST plus
JAXPR); a reviewed numerical helper opts out with `lcm.smooth_helper`.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.bqsegm_validation import (
    find_ast_violations,
    find_jaxpr_violations,
    is_smooth_helper,
)
from lcm.case_piece import smooth_helper


def smooth_oop(medical_expense, coinsurance):
    return coinsurance * medical_expense


def test_ast_gate_accepts_a_smooth_formula():
    """An arithmetic-only formula raises no AST violation."""
    assert find_ast_violations(smooth_oop, mode="smooth_user") == []


def test_ast_gate_rejects_a_python_if_in_a_piece():
    """A Python `if` is a hidden case boundary the AST gate must reject."""

    def branchy(medical_expense, medicaid_asset_limit):
        if medical_expense > medicaid_asset_limit:
            return 0.0
        return medical_expense

    violations = find_ast_violations(branchy, mode="smooth_user")
    assert any("if" in message.lower() for message in violations)


def test_ast_gate_rejects_a_comparison_in_a_smooth_piece():
    """A bare comparison in smooth mode creates an undeclared boundary."""

    def comparing(assets, limit):
        return (assets < limit) * assets

    violations = find_ast_violations(comparing, mode="smooth_user")
    assert any("comparison" in message.lower() for message in violations)


def test_ast_gate_allows_a_comparison_in_a_boundary_predicate():
    """A case-boundary predicate is meant to compare; boundary mode permits it."""

    def predicate(assets, limit):
        return assets < limit

    assert find_ast_violations(predicate, mode="boundary") == []


def test_ast_gate_fails_loudly_on_uninspectable_source():
    """A function whose source cannot be read fails the gate rather than passing."""
    violations = find_ast_violations(len, mode="smooth_user")
    assert len(violations) == 1
    assert "source" in violations[0].lower()


def test_jaxpr_gate_catches_a_where_hidden_in_a_helper():
    """A `jnp.where` inside a called helper is invisible to AST but caught by JAXPR."""

    def hidden_where_helper(medical_expense):
        return jnp.where(medical_expense > 0.0, medical_expense, 0.0)

    def piece(medical_expense):
        return hidden_where_helper(medical_expense)

    violations = find_jaxpr_violations(
        piece, abstract_args=(jnp.array(1.0),), mode="smooth_user"
    )
    assert violations != []


def test_jaxpr_gate_walks_into_a_nested_cond():
    """Piecewise logic hidden in a `lax.cond` branch is caught via nested jaxprs."""

    def hidden_cond(medical_expense):
        return jax.lax.cond(
            medical_expense > 0.0,
            lambda value: value,
            lambda value: -value,
            medical_expense,
        )

    violations = find_jaxpr_violations(
        hidden_cond, abstract_args=(jnp.array(1.0),), mode="smooth_user"
    )
    assert violations != []


def test_jaxpr_gate_accepts_a_smooth_helper_chain():
    """A purely-arithmetic helper chain raises no JAXPR violation."""

    def helper(medical_expense):
        return medical_expense**0.5

    def piece(medical_expense):
        return 2.0 * helper(medical_expense)

    violations = find_jaxpr_violations(
        piece, abstract_args=(jnp.array(1.0),), mode="smooth_user"
    )
    assert violations == []


def test_smooth_helper_marks_a_node_as_trusted():
    """`@lcm.smooth_helper` attests a numerical (non-economic) `clip`/`max` use."""

    @smooth_helper
    def numerically_clipped(consumption):
        return jnp.clip(consumption, 1e-10, None)

    assert is_smooth_helper(numerically_clipped) is True
    assert is_smooth_helper(smooth_oop) is False
