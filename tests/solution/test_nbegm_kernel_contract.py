"""The single-liquid NB-EGM kernels state their fixed naming contract up front.

Those kernels are not DAG-composed: they read the liquid axis under the name
`liquid` and the CRRA coefficient, gross return, and income under fixed
qualified parameter names. A regime that names them otherwise is refused at
build with the offending name, instead of dying inside a traced kernel with a
missing-argument or missing-parameter error.
"""

import pytest

from lcm import LinSpacedGrid
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    crra_utility,
    make_alive_dead_model,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_medicaid_toy import (
    medicaid_eligible,
    subsidy_medicaid,
    subsidy_private,
)

SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=40)


def utility_with_gamma(consumption: ContinuousAction, gamma: float) -> FloatND:
    """CRRA consumption utility whose coefficient is named `gamma`."""
    return crra_utility(consumption, gamma)


def resources(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: liquid wealth plus the Medicaid-contingent subsidy."""
    return liquid + subsidy


def next_liquid_with_interest(
    savings: FloatND, interest: float, income: float
) -> ContinuousState:
    """Liquid law of motion whose gross-return parameter is named `interest`."""
    return (1.0 + interest) * savings + income


def _build(*, alive_functions, liquid_law):
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=20,
        liquid_max=20.0,
        n_consumption=20,
        alive_functions=alive_functions,
        liquid_law=liquid_law,
        alive_solver=resolve_solver(
            "nbegm", savings_grid=SAVINGS_GRID, post_decision_function="savings"
        ),
        constraints={},
    )


_PIECES = {
    "savings": savings,
    "medicaid_eligible": medicaid_eligible,
    "subsidy_medicaid": subsidy_medicaid,
    "subsidy_private": subsidy_private,
    "resources": resources,
}


def test_a_utility_without_a_crra_parameter_is_named_at_build() -> None:
    """The kernels read `utility__crra`, so a `gamma`-named coefficient is refused."""
    with pytest.raises(RegimeInitializationError, match=r"'crra'.*'utility'"):
        _build(
            alive_functions={"utility": utility_with_gamma, **_PIECES},
            liquid_law=next_liquid_from_savings,
        )


def test_a_liquid_law_without_a_return_liquid_parameter_is_named_at_build() -> None:
    """The kernels read `next_liquid__return_liquid`, so `interest` is refused."""
    with pytest.raises(
        RegimeInitializationError, match=r"'next_liquid'.*return_liquid"
    ):
        _build(
            alive_functions={"utility": utility, **_PIECES},
            liquid_law=next_liquid_with_interest,
        )
