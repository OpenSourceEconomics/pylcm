"""The single-liquid NB-EGM kernels state their structural contract up front.

The felicity and the accounting are the regime's own: the kernels evaluate the
declared `utility` target and read the declared liquid law, so what a regime
calls its curvature parameter, its return parameter, or its budget node is its
own business.

Two structural requirements remain, and a regime breaking either is refused at
build with the offending name rather than dying inside a traced kernel:

- the liquid law is stated as a function of a post-decision savings node, which
  is the axis the Euler inversion runs on;
- the case-piece route's budget node is `lcm.cash_on_hand_with_subsidy`, because
  those kernels form `liquid + subsidy` themselves instead of calling it.
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
    feasible,
    make_alive_dead_model,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_medicaid_toy import (
    build_params,
    medicaid_eligible,
    resources,
    subsidy,
    subsidy_medicaid,
    subsidy_private,
)

SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=40)


def utility_with_gamma(consumption: ContinuousAction, gamma: float) -> FloatND:
    """CRRA consumption utility whose coefficient is named `gamma`."""
    return crra_utility(consumption, gamma)


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
            "nbegm",
            savings_grid=SAVINGS_GRID,
            envelope_arithmetic="ordinary",
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
    """Renaming cash-on-hand renames it in the savings node too, and that is fine.

    Post-decision savings are the budget node net of consumption, so a model
    naming that node `cash_on_hand` writes a savings function reading
    `cash_on_hand`. What the Euler axis is called upstream is invisible to the
    kernels, so the renamed model solves to the same value function as the
    default.
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
            envelope_arithmetic="ordinary",
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
    """A user's own spelling of the conventional law, computing the same value."""
    return (1.0 + return_liquid) * savings + income


def test_a_hand_written_liquid_law_solves_the_same_problem() -> None:
    """A user's own spelling of the conventional law gives one value function.

    The kernels read the law the regime declares, so a law is accepted for the
    landing points it produces rather than for being the object pylcm supplies.
    """
    hand_written = _build(
        alive_functions={"utility": utility, **_PIECES},
        liquid_law=_hand_written_liquid_law,
    ).solve(params=copy.deepcopy(_TOY_PARAMS), log_level="debug")
    supplied = _build(
        alive_functions={"utility": utility, **_PIECES},
        liquid_law=next_liquid_from_savings,
    ).solve(params=copy.deepcopy(_TOY_PARAMS), log_level="debug")

    np.testing.assert_allclose(
        np.asarray(hand_written[0]["alive"]), np.asarray(supplied[0]["alive"])
    )


def _liquid_law_from_resources(
    resources: FloatND, consumption: ContinuousAction, return_liquid: float
) -> ContinuousState:
    """The conventional law written as a displacement of cash-on-hand."""
    return (1.0 + return_liquid) * (resources - consumption)


def test_a_liquid_law_in_displacement_form_is_refused_by_the_single_liquid_route() -> (
    None
):
    """The liquid law states where a level of post-decision savings lands.

    The Euler inversion runs on a grid of savings and reads the continuation off
    the landing points that grid reaches, so a law whose landing point still
    moves with cash-on-hand at fixed savings has no single continuation to read
    — even when it happens to depend on the difference alone.
    """
    with pytest.raises(RegimeInitializationError, match="savings"):
        _build(
            alive_functions={"utility": utility, **_PIECES},
            liquid_law=_liquid_law_from_resources,
        )


def test_a_regime_without_a_post_decision_savings_node_is_refused() -> None:
    """The regime declares the function computing post-decision savings.

    That node is what the liquid law is read against, so its absence is named at
    build rather than surfacing as a missing DAG input inside a traced kernel.
    """
    without_savings = {
        name: func for name, func in _PIECES.items() if name != "savings"
    }
    with pytest.raises(RegimeInitializationError, match="savings"):
        make_alive_dead_model(
            n_periods=3,
            n_liquid=20,
            liquid_max=20.0,
            n_consumption=20,
            alive_functions={"utility": utility, **without_savings},
            liquid_law=next_liquid_from_savings,
            alive_solver=resolve_solver(
                "nbegm",
                savings_grid=SAVINGS_GRID,
                envelope_arithmetic="ordinary",
            ),
            # The budget node reaches the liquid state only through `savings`, so
            # the constraint is what keeps `liquid` live long enough for the
            # solver to speak; without it the model reports an unused state first.
            constraints={"feasible": feasible},
        )


def _hand_written_budget(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """A user's own spelling of the fixed cash-on-hand, computing the same value."""
    return liquid + subsidy


def _rescaled_budget(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """The fixed cash-on-hand scaled by `1 + 9e-6`, invisible to any probe set."""
    return (1.0 + 9e-6) * (liquid + subsidy)


@pytest.mark.parametrize("budget_node", [_hand_written_budget, _rescaled_budget])
def test_a_hand_written_budget_node_is_refused_by_the_case_piece_route(
    budget_node,
) -> None:
    """The case-piece route takes pylcm's own cash-on-hand, not a user callable.

    The kernels add the split output to the liquid state themselves rather than
    calling the declared node, and no finite check establishes that an arbitrary
    callable computes the same thing — a global rescaling agrees at every sampled
    point and still moves every state's value. So the route accepts the node
    pylcm supplies, whose identity settles the question by construction, and
    refuses everything else with somewhere to go.
    """
    with pytest.raises(RegimeInitializationError, match="cash_on_hand_with_subsidy"):
        _build(
            alive_functions={
                "utility": utility,
                **_PIECES,
                "resources": budget_node,
            },
            liquid_law=next_liquid_from_savings,
        )


def test_a_liquid_law_naming_its_return_interest_solves_the_same_problem() -> None:
    """Two spellings of one gross-return parameter give one value function.

    The kernels evaluate the law the regime declares, so the name its return
    parameter carries is invisible to them.
    """
    return_liquid_params = copy.deepcopy(_TOY_PARAMS)
    interest_params = copy.deepcopy(_TOY_PARAMS)
    budget = interest_params["alive"]["alive"]["next_liquid"]
    renamed_budget = {"interest": budget["return_liquid"], "income": budget["income"]}
    for target in ("alive", "dead"):
        interest_params["alive"][target]["next_liquid"] = renamed_budget

    named_interest = _build(
        alive_functions={"utility": utility, **_PIECES},
        liquid_law=next_liquid_with_interest,
    ).solve(params=interest_params, log_level="debug")
    named_return_liquid = _build(
        alive_functions={"utility": utility, **_PIECES},
        liquid_law=next_liquid_from_savings,
    ).solve(params=return_liquid_params, log_level="debug")

    np.testing.assert_allclose(
        np.asarray(named_interest[0]["alive"]),
        np.asarray(named_return_liquid[0]["alive"]),
    )
