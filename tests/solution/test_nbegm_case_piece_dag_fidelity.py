"""NB-EGM's single-liquid kernels refuse a regime whose economics they cannot run.

Those kernels are not composed from the regime's DAG: they solve a fixed
consumption-saving form — CRRA flow utility and an affine liquid law — reading the
coefficients under fixed qualified parameter names. A regime whose `utility` or
liquid law carries any structure beyond that form declares a different Bellman
problem from the one the kernel solves, so it is rejected at build rather than
solved as if the extra structure were not there.

Regimes needing a richer objective declare a `lcm.piecewise_affine` schedule with a
`post_decision_function`, which composes its budget and utility from the DAG.
"""

import copy

import pytest

from lcm import LinSpacedGrid
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.test_models import nbegm_medicaid_toy as toy
from tests.test_models.nbegm_common import (
    crra_utility,
    make_alive_dead_model,
    next_liquid,
    resolve_solver,
)


def scaled_utility(
    consumption: ContinuousAction, crra: float, util_scale: float
) -> FloatND:
    """CRRA consumption utility under an ordinary flat scale."""
    return util_scale * crra_utility(consumption, crra)


def scaled_next_liquid(
    resources: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    income: float,
    income_scale: float,
) -> ContinuousState:
    """Affine liquid law whose income is rescaled before it is added."""
    return (1.0 + return_liquid) * (resources - consumption) + income_scale * income


def _build(
    *,
    utility_func=None,
    liquid_law=None,
):
    """Assemble the Medicaid case-piece toy over a substituted economic node."""
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=20,
        liquid_max=30.0,
        n_consumption=20,
        alive_functions={
            "utility": utility_func if utility_func is not None else toy.utility,
            "medicaid_eligible": toy.medicaid_eligible,
            "subsidy_medicaid": toy.subsidy_medicaid,
            "subsidy_private": toy.subsidy_private,
            "subsidy": toy.subsidy,
            "resources": toy.resources,
        },
        liquid_law=liquid_law if liquid_law is not None else next_liquid,
        alive_solver=resolve_solver(
            "nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=22.0, n_points=30),
        ),
        constraints={},
        liquid_grid=LinSpacedGrid(start=0.1, stop=30.0, n_points=20),
    )


def _params(**extra: float) -> dict:
    """The toy's parameters, with any extra flat params merged into `utility`.

    Deep-copied because the toy caches its parameter tree.
    """
    params = copy.deepcopy(toy.build_params())
    params["alive"]["utility"].update(extra)
    return params


def test_a_scaled_utility_is_refused_by_the_single_liquid_kernels():
    """A flat scale on `utility` is rejected, not silently dropped from the objective.

    The kernel evaluates unscaled CRRA, so accepting the scale would solve a
    different problem than the regime declares.
    """
    with pytest.raises(RegimeInitializationError, match="util_scale"):
        _build(utility_func=scaled_utility)


def test_a_rescaled_liquid_law_is_refused_by_the_single_liquid_kernels():
    """An extra parameter in the liquid law is rejected rather than ignored.

    The kernel evaluates `(1 + return_liquid) * savings + income`, so a law that
    rescales income before adding it is a budget the kernel never applies.
    """
    with pytest.raises(RegimeInitializationError, match="income_scale"):
        _build(liquid_law=scaled_next_liquid)


def test_the_supported_fixed_form_still_builds():
    """The plain CRRA / affine-law regime the kernels do solve is still accepted."""
    model = _build()
    solution = model.solve(params=_params(), log_level="off")
    assert any("alive" in period for period in solution.values())
