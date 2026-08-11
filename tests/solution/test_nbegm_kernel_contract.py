"""The single-liquid NB-EGM kernels state their fixed naming contract up front.

The budget is not DAG-composed: the kernels read the liquid axis under the name
`liquid` and the gross return and income under fixed qualified parameter names.
A regime that names them otherwise is refused at build with the offending name,
instead of dying inside a traced kernel with a missing-argument or
missing-parameter error.

The felicity carries no such contract — it is the regime's own `utility` target,
solved as declared — so what a regime calls its curvature parameter is its own
business.
"""

import copy

import numpy as np
import pytest

from lcm import LinSpacedGrid
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.test_models import nbegm_tax_toy as tax_toy
from tests.test_models.nbegm_common import (
    crra_utility,
    make_alive_dead_model,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_medicaid_toy import (
    build_params,
    medicaid_eligible,
    subsidy,
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


# Ages run `0 .. 2`, so `alive` goes inactive at age 2.
_TOY_PARAMS = build_params(final_age_alive=2.0)

_PIECES = {
    "savings": savings,
    "medicaid_eligible": medicaid_eligible,
    "subsidy_medicaid": subsidy_medicaid,
    "subsidy_private": subsidy_private,
    "subsidy": subsidy,
    "resources": resources,
}


def test_a_utility_naming_its_coefficient_gamma_solves_the_same_problem() -> None:
    """Two spellings of one CRRA coefficient give one value function.

    The kernels evaluate the felicity the regime declares, so the name its
    curvature parameter carries is invisible to them.
    """
    crra_params = copy.deepcopy(_TOY_PARAMS)
    gamma_params = copy.deepcopy(_TOY_PARAMS)
    gamma_params["alive"]["utility"] = {
        "gamma": crra_params["alive"]["utility"]["crra"]
    }

    named_gamma = _build(
        alive_functions={"utility": utility_with_gamma, **_PIECES},
        liquid_law=next_liquid_from_savings,
    ).solve(params=gamma_params, log_level="debug")
    named_crra = _build(
        alive_functions={"utility": utility, **_PIECES},
        liquid_law=next_liquid_from_savings,
    ).solve(params=crra_params, log_level="debug")

    np.testing.assert_allclose(
        np.asarray(named_gamma[0]["alive"]), np.asarray(named_crra[0]["alive"])
    )


def test_a_budget_node_named_for_its_own_domain_still_satisfies_the_contract() -> None:
    """Renaming cash-on-hand renames it in the liquid law too, and that is fine.

    The law states savings as the budget node minus consumption, so a model
    naming that node `cash_on_hand` writes a law reading `cash_on_hand`. The
    contract is about the budget the law computes, not the words it computes it
    from, so the renamed model solves to the same value function as the default.
    """

    def solve(budget_name: str):
        model = tax_toy.build_model(
            variant="nbegm",
            budget_name=budget_name,
            n_liquid=20,
            liquid_max=30.0,
            n_savings=30,
            savings_max=28.0,
            n_consumption=20,
        )
        return model.solve(
            params=tax_toy.build_params(budget_name=budget_name), log_level="debug"
        )

    renamed = solve("cash_on_hand")
    default = solve("resources")

    np.testing.assert_allclose(
        np.asarray(renamed[0]["alive"]), np.asarray(default[0]["alive"])
    )


def _hand_written_liquid_law(
    savings: FloatND, return_liquid: float, income: float
) -> ContinuousState:
    """A user's own spelling of the fixed law, computing exactly the same value."""
    return (1.0 + return_liquid) * savings + income


def _rescaled_liquid_law(
    savings: FloatND, return_liquid: float, income: float
) -> ContinuousState:
    """The fixed law scaled by `1 + 9e-6`, a difference no probe set separates."""
    return (1.0 + 9e-6) * ((1.0 + return_liquid) * savings + income)


@pytest.mark.parametrize("law", [_hand_written_liquid_law, _rescaled_liquid_law])
def test_a_hand_written_liquid_law_is_refused_by_the_single_liquid_route(law) -> None:
    """The single-liquid route takes pylcm's own law object, not a user callable.

    The kernels apply a fixed budget rather than calling the declared law, and no
    finite check establishes that an arbitrary callable computes it — a global
    rescaling agrees at every sampled point and still moves every state's value.
    So the route accepts the law pylcm supplies, whose identity settles the
    question by construction, and refuses everything else with somewhere to go.
    """
    with pytest.raises(RegimeInitializationError, match="liquid_law_from_savings"):
        _build(
            alive_functions={"utility": utility, **_PIECES},
            liquid_law=law,
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
