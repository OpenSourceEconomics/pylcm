"""Cross-solver acceptance spec for declared feasibility constraints.

One economic declaration, several solvers. A constraint means the same thing to
every one of them: it restricts the admissible candidates. What differs is how
a solver *discharges* it — by evaluating the predicate, by proving the
candidate family already satisfies it, by compiling its boundary into the
kernel's own partition, or by refusing the regime.

What no solver may do is accept a declaration and then ignore it. Each row
below therefore pins the same declaration against more than one solver, so a
solver that silently drops it fails here even when its own tests agree with
their own oracle.
"""

from collections.abc import Callable, Mapping

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import LinSpacedGrid, Model
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, ContinuousAction, ContinuousState
from tests.test_models import n_nbegm_toy
from tests.test_models import nbegm_medicaid_toy as toy
from tests.test_models.nbegm_common import (
    feasible,
)

SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=20)


def unsatisfiable(consumption: ContinuousAction) -> BoolND:
    """A predicate false at every node of the consumption grid."""
    return consumption < jnp.asarray(-1.0)


def nonlinear(*, consumption: ContinuousAction, liquid: ContinuousState) -> BoolND:
    """A general predicate whose boundary is a root in no declared coordinate."""
    return jnp.square(consumption) + jnp.square(liquid) <= 400.0


def _build(*, variant: str, constraints: Mapping[str, Callable[..., object]]) -> Model:
    """Build the Medicaid one-asset toy under `variant` with a given pool."""
    return toy.build_model(
        variant=variant,
        n_periods=4,
        n_liquid=10,
        n_consumption=12,
        n_savings=20,
        constraints=constraints,
    )


def _alive_values(model: Model) -> np.ndarray:
    """Solve and stack every alive period's value array."""
    solution = model.solve(params=toy.build_params(), log_level="off")
    return np.concatenate(
        [
            np.asarray(regimes["alive"]).ravel()
            for regimes in solution.values.values()
            if "alive" in regimes
        ]
    )


def test_grid_search_sends_every_state_to_minus_infinity_on_an_unsatisfiable_pool():
    """`GridSearch` leaves no admissible action when the predicate is never true."""
    values = _alive_values(_build(variant="brute", constraints={"u": unsatisfiable}))
    assert np.all(np.isneginf(values))


def test_grid_search_publishes_finite_values_under_a_satisfiable_pool():
    """The same model under the borrowing constraint is finite everywhere.

    The control for the row above: `-inf` there has to come from the predicate
    being unsatisfiable, not from declaring a constraint at all. The budget
    predicate is what keeps the dense grid off the region where the CRRA flow
    is undefined, so the unconstrained model is not the comparison — it is not
    finite either, for an unrelated reason.
    """
    values = _alive_values(_build(variant="brute", constraints={"f": feasible}))
    assert np.all(np.isfinite(values))


def test_nbegm_refuses_the_unsatisfiable_pool_rather_than_publishing_finite_values():
    """NBEGM refuses what it cannot enforce instead of solving past it."""
    with pytest.raises(ModelInitializationError, match="'u'"):
        _build(variant="nbegm", constraints={"u": unsatisfiable})


def test_grid_search_evaluates_a_general_nonlinear_predicate():
    """`GridSearch` admits an arbitrary callable and carries it into the solve."""
    model = _build(variant="brute", constraints={"nonlinear": nonlinear})
    assert "nonlinear" in model._regimes["alive"].solution.constraints


def test_nbegm_refuses_a_general_nonlinear_predicate():
    """No EGM-family kernel can locate an arbitrary predicate's boundary."""
    with pytest.raises(ModelInitializationError, match="nonlinear"):
        _build(variant="nbegm", constraints={"nonlinear": nonlinear})


def outer_financeable(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    """A predicate over the nested toy's liquid margin."""
    return jnp.square(consumption) + jnp.square(wealth) <= 400.0


def test_grid_search_evaluates_the_nested_declaration():
    """The nested model's declaration reaches the grid-search solve phase."""
    model = n_nbegm_toy.build_model(
        variant="brute", constraints={"outer_financeable": outer_financeable}
    )
    assert "outer_financeable" in model._regimes["alive"].solution.constraints


def test_nnbegm_refuses_the_nested_declaration_it_cannot_enforce():
    """The nested kernels refuse a predicate neither margin evaluates."""
    with pytest.raises(ModelInitializationError, match="outer_financeable"):
        n_nbegm_toy.build_model(
            variant="n_nbegm", constraints={"outer_financeable": outer_financeable}
        )
