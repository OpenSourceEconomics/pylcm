"""Spec for what an NBEGM regime may declare in its `constraints` slot.

The NBEGM kernel recovers consumption by inverting the Euler equation at each
node of a savings grid: the action is produced first and the state falls out of
the budget identity afterwards. There is no point in that step at which a
general predicate over `(state, action)` is evaluable, and the published
candidates are never masked by one.

So a regime that declares such a predicate and is solved by NBEGM must be
refused when the model is built. Accepting it would solve a different problem
than the one declared, silently and with no diagnostic.
"""

from collections.abc import Callable, Mapping

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import LinSpacedGrid, LiquidMargin, Model, post_decision_lower_bound
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, ContinuousAction, ContinuousState
from tests.conftest import DECIMAL_PRECISION
from tests.test_models import n_nbegm_toy, nbegm_medicaid_toy
from tests.test_models.nbegm_common import (
    make_alive_dead_model,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_medicaid_toy import (
    medicaid_eligible,
    resources,
    subsidy,
    subsidy_medicaid,
    subsidy_private,
)


def rationing(consumption: ContinuousAction, liquid: ContinuousState) -> BoolND:
    """A feasibility predicate whose boundary is a root in no declared coordinate.

    Deliberately not a bound on the post-decision state: no savings-grid node
    and no case boundary locates it, so no EGM-family kernel can enforce it.
    """
    return jnp.square(consumption) + jnp.square(liquid) <= 400.0


def _build_model(
    *, variant: str, constraints: Mapping[str, Callable[..., object]]
) -> Model:
    """Assemble the Medicaid one-asset toy with an arbitrary constraint pool."""
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=10,
        liquid_max=20.0,
        n_consumption=12,
        alive_functions={
            "utility": utility,
            "medicaid_eligible": medicaid_eligible,
            "subsidy_medicaid": subsidy_medicaid,
            "subsidy_private": subsidy_private,
            "subsidy": subsidy,
            "resources": resources,
            "savings": savings,
        },
        liquid_law=next_liquid_from_savings,
        alive_solver=resolve_solver(
            variant,
            savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=10),
        ),
        constraints=constraints,
    )


def test_nbegm_refuses_a_constraint_its_kernel_cannot_evaluate():
    """An NBEGM regime declaring a general feasibility predicate fails to build."""
    with pytest.raises(ModelInitializationError, match="rationing"):
        _build_model(variant="nbegm", constraints={"rationing": rationing})


def test_grid_search_accepts_the_same_constraint():
    """The refusal is NBEGM's own: `GridSearch` evaluates the predicate and builds."""
    model = _build_model(variant="brute", constraints={"rationing": rationing})
    assert "rationing" in model._regimes["alive"].solution.constraints


def nested_rationing(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    """A general feasibility predicate over the nested toy's liquid margin."""
    return jnp.square(consumption) + jnp.square(wealth) <= 400.0


def test_nnbegm_refuses_a_constraint_its_kernel_cannot_evaluate():
    """A nested NB-EGM regime declaring a feasibility predicate fails to build."""
    with pytest.raises(ModelInitializationError, match="nested_rationing"):
        n_nbegm_toy.build_model(
            variant="n_nbegm",
            constraints={"nested_rationing": nested_rationing},
        )


def test_grid_search_accepts_the_same_nested_constraint():
    """The refusal is the solver's: `GridSearch` builds the same declaration."""
    model = n_nbegm_toy.build_model(
        variant="brute",
        constraints={"nested_rationing": nested_rationing},
    )
    assert "nested_rationing" in model._regimes["alive"].solution.constraints


_LIQUID_MARGIN = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)


def test_nbegm_accepts_a_lower_bound_its_savings_grid_already_enforces():
    """Declaring the limit the savings grid implies builds under NBEGM.

    The kernel inverts on that grid, so its lowest node *is* the borrowing
    limit. A declaration stating the same number adds no predicate the kernel
    would have to evaluate, and is admitted where a general one is refused.
    """
    model = _build_model(
        variant="nbegm",
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=_LIQUID_MARGIN, lower=0.0
            )
        },
    )

    assert "alive" in model.user_regimes


def test_nbegm_refuses_a_lower_bound_that_disagrees_with_its_savings_grid():
    """A declared limit the grid contradicts is refused, naming the grid's node.

    The grid's lowest node governs both the solve and the simulate-phase mask,
    so a declaration naming a different number would be overridden silently.
    """
    with pytest.raises(ModelInitializationError, match=r"savings grid starts at 0\.0"):
        _build_model(
            variant="nbegm",
            constraints={
                "borrowing_limit": post_decision_lower_bound(
                    margin=_LIQUID_MARGIN, lower=1.0
                )
            },
        )


_NESTED_LIQUID_MARGIN = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="liquid_resources",
    post_decision_state="liquid_savings",
)


def test_nnbegm_accepts_a_lower_bound_its_inner_savings_grid_already_enforces():
    """The nested solver admits the limit its inner savings grid implies."""
    model = n_nbegm_toy.build_model(
        variant="n_nbegm",
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=_NESTED_LIQUID_MARGIN, lower=n_nbegm_toy.SAVINGS_FLOOR
            )
        },
    )

    assert "alive" in model.user_regimes


def test_nnbegm_refuses_a_lower_bound_that_disagrees_with_its_inner_grid():
    """A nested declaration the inner grid contradicts is refused."""
    with pytest.raises(ModelInitializationError, match="borrowing_limit"):
        n_nbegm_toy.build_model(
            variant="n_nbegm",
            constraints={
                "borrowing_limit": post_decision_lower_bound(
                    margin=_NESTED_LIQUID_MARGIN,
                    lower=n_nbegm_toy.SAVINGS_FLOOR + 1.0,
                )
            },
        )


def test_declaring_the_bound_leaves_the_nbegm_solution_unchanged():
    """A proved declaration is inert: the savings grid already enforces it.

    Building is the weaker claim — it says the declaration was admitted, not
    that it was disposed of without effect. Solving both arms says the
    admitted declaration adds no mask, no candidate, and no shift in value.
    """
    params = nbegm_medicaid_toy.build_params(final_age_alive=3.0)
    declared = _build_model(
        variant="nbegm",
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=_LIQUID_MARGIN, lower=0.0
            )
        },
    )
    bare = _build_model(variant="nbegm", constraints={})

    with_declaration = declared.solve(params=params, log_level="off")
    without = bare.solve(params=params, log_level="off")

    for period, regime_to_V in without.items():
        for regime_name, V_arr in regime_to_V.items():
            aaae(
                with_declaration[period][regime_name],
                V_arr,
                decimal=DECIMAL_PRECISION,
            )
