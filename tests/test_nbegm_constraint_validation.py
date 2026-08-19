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

from lcm import LinSpacedGrid, Model
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, ContinuousAction, ContinuousState
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
