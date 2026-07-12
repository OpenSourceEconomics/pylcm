"""The nested-inner adapter normalizes 1-D EGM solver configs.

Outer-search solvers wrap a one-dimensional inner EGM solver (DCEGM or
NBEGM). The two inner configs name the same concepts differently (`resources`
vs `budget_target`) and NBEGM leaves two slots inferable (`continuous_state`,
`post_decision_function`); `get_nnbegm_inner_spec` maps either config onto one
explicit spec so the outer kernel code never dispatches on the inner type.
"""

import pytest

from _lcm.solution.solvers import get_nnbegm_inner_spec
from lcm import LinSpacedGrid
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import DCEGM, NBEGM, GridSearch

_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=8)


def _dcegm() -> DCEGM:
    return DCEGM(
        continuous_state="liquid",
        continuous_action="consumption",
        resources="cash_on_hand",
        post_decision_function="liquid_savings",
        savings_grid=_SAVINGS_GRID,
    )


def _nbegm(**overrides: object) -> NBEGM:
    config: dict[str, object] = {
        "continuous_state": "liquid",
        "post_decision_function": "liquid_savings",
        "budget_target": "cash_on_hand",
        "savings_grid": _SAVINGS_GRID,
    }
    config.update(overrides)
    return NBEGM(**config)  # ty: ignore[invalid-argument-type]


def test_dcegm_inner_spec_maps_resources_to_budget_target() -> None:
    """A DCEGM inner's `resources` slot fills the spec's `budget_target`."""
    spec = get_nnbegm_inner_spec(inner=_dcegm())
    assert spec.budget_target == "cash_on_hand"


def test_dcegm_inner_spec_carries_state_post_decision_and_grid() -> None:
    """The spec mirrors the DCEGM config's Euler-state slots verbatim."""
    spec = get_nnbegm_inner_spec(inner=_dcegm())
    assert (spec.continuous_state, spec.post_decision_function) == (
        "liquid",
        "liquid_savings",
    )
    assert spec.savings_grid is _SAVINGS_GRID


def test_nbegm_inner_spec_carries_explicit_slots() -> None:
    """An NBEGM inner with explicit slots maps onto the spec verbatim."""
    inner = _nbegm()
    spec = get_nnbegm_inner_spec(inner=inner)
    assert (
        spec.continuous_state,
        spec.post_decision_function,
        spec.budget_target,
    ) == ("liquid", "liquid_savings", "cash_on_hand")
    assert spec.solver is inner


def test_nbegm_inner_spec_requires_explicit_continuous_state() -> None:
    """An NBEGM inner leaving `continuous_state` to inference is rejected.

    Inside a nested solver the regime has two continuous states, so the
    single-continuous-state inference NBEGM applies standalone is ambiguous.
    """
    with pytest.raises(RegimeInitializationError, match="continuous_state"):
        get_nnbegm_inner_spec(inner=_nbegm(continuous_state=None))


def test_nbegm_inner_spec_requires_explicit_post_decision_function() -> None:
    """An NBEGM inner without `post_decision_function` is rejected."""
    with pytest.raises(RegimeInitializationError, match="post_decision_function"):
        get_nnbegm_inner_spec(inner=_nbegm(post_decision_function=None))


def test_non_egm_inner_is_rejected_with_the_offending_type() -> None:
    """A non-1-D-EGM inner raises `TypeError` naming the offending type."""
    with pytest.raises(TypeError, match="GridSearch"):
        get_nnbegm_inner_spec(inner=GridSearch())
