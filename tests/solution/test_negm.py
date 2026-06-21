"""Spec for the `NEGM` solver configuration and its construction-time guards.

`NEGM` composes an inner `DCEGM` (the 1-D consumption-savings solve) with an
outer deterministic grid search over a durable/illiquid margin. Its
`__post_init__` guards reject — at construction, with a
`RegimeInitializationError` — an outer grid that is a stochastic process, an
outer action that coincides with the inner continuous action, and an outer
post-decision that coincides with the inner post-decision. Each case here
constructs an `NEGM`/`DCEGM` and asserts the right error; none solves a model.
"""

import dataclasses

import pytest

from _lcm.grids import ContinuousGrid
from lcm import DCEGM, NEGM, LinSpacedGrid, NormalIIDProcess
from lcm.exceptions import RegimeInitializationError

_INNER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="liquid_savings",
    savings_grid=LinSpacedGrid(start=0.0, stop=30.0, n_points=40),
)

_OUTER_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=20)


def _negm(
    *,
    inner: DCEGM = _INNER,
    outer_action: str = "illiquid_investment",
    outer_post_decision: str = "next_illiquid",
    outer_grid: ContinuousGrid = _OUTER_GRID,
    outer_no_adjustment_candidate: str | None = None,
) -> NEGM:
    return NEGM(
        inner=inner,
        outer_action=outer_action,
        outer_post_decision=outer_post_decision,
        outer_grid=outer_grid,
        outer_no_adjustment_candidate=outer_no_adjustment_candidate,
    )


def test_negm_with_valid_fields_constructs():
    """A `NEGM` with distinct margins and a deterministic outer grid constructs."""
    solver = _negm()
    assert solver.inner is _INNER
    assert solver.outer_action == "illiquid_investment"
    assert solver.outer_post_decision == "next_illiquid"
    assert solver.outer_no_adjustment_candidate is None


def test_negm_with_no_adjustment_candidate_constructs():
    """The optional state-specific no-adjustment candidate is accepted."""
    solver = _negm(outer_no_adjustment_candidate="keep_illiquid")
    assert solver.outer_no_adjustment_candidate == "keep_illiquid"


def test_negm_stochastic_outer_grid_is_rejected():
    """A stochastic process cannot serve as the deterministic outer search grid."""
    process = NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)
    with pytest.raises(RegimeInitializationError, match="stochastic process"):
        _negm(outer_grid=process)


def test_negm_outer_action_equal_to_inner_continuous_action_is_rejected():
    """The outer durable margin must differ from the inner consumption action."""
    with pytest.raises(RegimeInitializationError, match="coincides with the inner"):
        _negm(outer_action="consumption")


def test_negm_outer_post_decision_equal_to_inner_post_decision_is_rejected():
    """The outer post-decision must differ from the inner liquid post-decision."""
    with pytest.raises(RegimeInitializationError, match="coincides with"):
        _negm(outer_post_decision="liquid_savings")


def test_negm_invalid_inner_dcegm_is_rejected_by_inner_guards():
    """An invalid inner `DCEGM` is rejected by its own guards before NEGM's.

    The composition reuses `DCEGM.__post_init__` wholesale, so a stochastic
    inner savings grid fails when the inner config is constructed.
    """
    process = NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)
    with pytest.raises(RegimeInitializationError, match="savings_grid"):
        dataclasses.replace(_INNER, savings_grid=process)
