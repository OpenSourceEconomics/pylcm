"""NB-EGM's single-liquid kernels solve the felicity and the law a regime declares.

The felicity comes from the regime's own `utility` target and the accounting from
its own liquid law, read at each level of post-decision savings. So a scale on
utility, a rescaled income, a taxed return, an endowment, or a return compounded
over sub-periods is part of the problem solved, and each moves the value function
it produces.

The budget node is the one exception: the case-piece kernels form cash-on-hand as
`liquid + subsidy` themselves rather than calling the declared node, so a node
combining exactly those two differently states a problem they never solve and is
refused at build.
"""

import copy

import numpy as np
import pytest

from lcm import LinSpacedGrid
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.test_models import nbegm_medicaid_toy as toy
from tests.test_models.nbegm_common import (
    crra_utility,
    make_alive_dead_model,
    next_liquid_from_savings,
    resolve_solver,
    savings,
)

# Ages run `0 .. _N_PERIODS - 1`, and the last of them is the terminal age at
# which `alive` goes inactive. The survival law's `final_age_alive` has to name
# that same age, or the alive regime keeps sending mass to itself past the point
# where it can receive it.
_N_PERIODS = 3
_FINAL_AGE_ALIVE = float(_N_PERIODS - 1)


def scaled_utility(
    *, consumption: ContinuousAction, crra: float, util_scale: float
) -> FloatND:
    """CRRA consumption utility under an ordinary flat scale."""
    return util_scale * crra_utility(consumption=consumption, crra=crra)


def scaled_next_liquid(
    *,
    savings: FloatND,
    return_liquid: float,
    income: float,
    income_scale: float,
) -> ContinuousState:
    """Affine liquid law whose income is rescaled before it is added."""
    return (1.0 + return_liquid) * savings + income_scale * income


def doubled_income_next_liquid(
    *, savings: FloatND, return_liquid: float, income: float
) -> ContinuousState:
    """Affine liquid law crediting twice the income the parameter names."""
    return (1.0 + return_liquid) * savings + 2.0 * income


def taxed_return_next_liquid(
    *, savings: FloatND, return_liquid: float, income: float
) -> ContinuousState:
    """Affine liquid law crediting the liquid return net of a flat tax."""
    return (1.0 + 0.75 * return_liquid) * savings + income


def endowed_next_liquid(
    *, savings: FloatND, return_liquid: float, income: float
) -> ContinuousState:
    """Affine liquid law adding a literal endowment on top of income."""
    return (1.0 + return_liquid) * savings + income + 3.0


def compounded_next_liquid(
    *, savings: FloatND, return_liquid: float, income: float
) -> ContinuousState:
    """Liquid law compounding the return over two sub-periods."""
    return (1.0 + return_liquid) ** 2 * savings + income


def doubled_subsidy_resources(*, liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand crediting the subsidy twice over."""
    return liquid + 2.0 * subsidy


def fee_charging_resources(*, liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand net of a literal participation fee."""
    return liquid + subsidy - 0.5


def interest_bearing_resources(*, liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand crediting within-period interest on liquid wealth."""
    return 1.05 * liquid + subsidy


def _build(
    *,
    utility_func=None,
    liquid_law=None,
    budget_func=None,
):
    """Assemble the Medicaid case-piece toy over a substituted economic node."""
    return make_alive_dead_model(
        n_periods=_N_PERIODS,
        n_liquid=20,
        liquid_max=30.0,
        n_consumption=20,
        alive_functions={
            "utility": utility_func if utility_func is not None else toy.utility,
            "medicaid_eligible": toy.medicaid_eligible,
            "subsidy_medicaid": toy.subsidy_medicaid,
            "subsidy_private": toy.subsidy_private,
            "subsidy": toy.subsidy,
            "resources": budget_func if budget_func is not None else toy.resources,
            "savings": savings,
        },
        liquid_law=liquid_law if liquid_law is not None else next_liquid_from_savings,
        alive_solver=resolve_solver(
            variant="nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=22.0, n_points=30),
        ),
        constraints={},
        liquid_grid=LinSpacedGrid(start=0.1, stop=30.0, n_points=20),
    )


def _params(
    *, utility_extra: dict | None = None, law_extra: dict | None = None
) -> dict:
    """The toy's parameters, with any extra flat params merged into their function.

    Deep-copied because the toy caches its parameter tree.
    """
    params = copy.deepcopy(toy.build_params(final_age_alive=_FINAL_AGE_ALIVE))
    params["alive"]["utility"].update(utility_extra or {})
    for target in ("alive", "dead"):
        params["alive"][target]["next_liquid"].update(law_extra or {})
    return params


def _solve_alive(*, liquid_law=None, law_extra=None) -> np.ndarray:
    """Solve the case-piece toy under `liquid_law` and return the first-age value."""
    solution = _build(liquid_law=liquid_law).solve(
        params=_params(law_extra=law_extra), log_level="debug"
    )
    return np.asarray(solution[0]["alive"])


def test_a_scaled_utility_enters_the_solved_objective():
    """A flat scale on `utility` changes the value function it produces.

    The kernel evaluates the declared felicity, so a scale the regime writes is
    part of the problem solved rather than structure quietly dropped from it.
    """
    scaled = _build(utility_func=scaled_utility).solve(
        params=_params(utility_extra={"util_scale": 2.0}), log_level="debug"
    )
    plain = _build().solve(params=_params(), log_level="debug")
    assert not np.allclose(
        np.asarray(scaled[0]["alive"]), np.asarray(plain[0]["alive"])
    )


def test_a_rescaled_income_in_the_liquid_law_enters_the_solved_value():
    """An extra parameter in the liquid law is solved, not rejected.

    The kernels read the law the regime declares at each level of savings, so a
    law rescaling income before adding it is the budget solved — and the value
    it produces differs from the one the conventional law gives.
    """
    scaled = _solve_alive(
        liquid_law=scaled_next_liquid, law_extra={"income_scale": 2.0}
    )
    plain = _solve_alive()
    assert not np.allclose(scaled, plain)


@pytest.mark.parametrize(
    "liquid_law",
    [
        doubled_income_next_liquid,
        taxed_return_next_liquid,
        endowed_next_liquid,
        compounded_next_liquid,
    ],
)
def test_a_same_signature_liquid_law_enters_the_solved_value(liquid_law):
    """A law is solved for what it computes, not for which parameters it names.

    Each of these declares exactly the parameters the conventional law names and
    would pass any name-level check, yet none of them equals
    `(1 + return_liquid) * savings + income`. Each is the accounting the regime
    stated, so each is the accounting solved, and each moves the value function.
    """
    assert not np.allclose(_solve_alive(liquid_law=liquid_law), _solve_alive())


@pytest.mark.parametrize(
    "budget_func",
    [
        doubled_subsidy_resources,
        fee_charging_resources,
        interest_bearing_resources,
    ],
)
def test_a_same_signature_budget_node_is_refused_by_the_case_piece_kernels(budget_func):
    """The case-piece kernels solve `liquid + subsidy`, and say so at build.

    They form cash-on-hand from the liquid state and the case's own subsidy
    rather than calling the declared budget node, so a node reading exactly those
    two and combining them differently states a problem the kernels never solve.
    """
    with pytest.raises(RegimeInitializationError, match="resources"):
        _build(budget_func=budget_func)


def test_the_conventional_case_piece_regime_still_builds():
    """The plain CRRA / affine-law regime the kernels solve is still accepted."""
    model = _build()
    solution = model.solve(params=_params(), log_level="debug")
    assert any("alive" in period for period in solution.values())
