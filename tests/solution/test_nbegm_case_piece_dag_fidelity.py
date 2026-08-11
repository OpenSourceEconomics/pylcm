"""NB-EGM's single-liquid kernels solve the felicity a regime declares.

The felicity comes from the regime's own `utility` target, so a scale, a
subsistence level, or any other structure the modeller writes enters the solved
objective. The budget does not: the kernels apply a fixed affine liquid law,
`(1 + return_liquid) * savings + income`, reading its coefficients under fixed
qualified parameter names. A law carrying structure beyond that declares a
different Bellman problem from the one the kernel solves, so it is rejected at
build rather than solved as if the extra structure were not there.

Regimes needing a richer budget declare a `lcm.piecewise_affine` schedule with a
`post_decision_function`, which composes it from the DAG.
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
    next_liquid,
    resolve_solver,
)

# Ages run `0 .. _N_PERIODS - 1`, and the last of them is the terminal age at
# which `alive` goes inactive. The survival law's `final_age_alive` has to name
# that same age, or the alive regime keeps sending mass to itself past the point
# where it can receive it.
_N_PERIODS = 3
_FINAL_AGE_ALIVE = float(_N_PERIODS - 1)


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


def doubled_income_next_liquid(
    resources: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Affine liquid law crediting twice the income the parameter names."""
    return (1.0 + return_liquid) * (resources - consumption) + 2.0 * income


def taxed_return_next_liquid(
    resources: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Affine liquid law crediting the liquid return net of a flat tax."""
    return (1.0 + 0.75 * return_liquid) * (resources - consumption) + income


def endowed_next_liquid(
    resources: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Affine liquid law adding a literal endowment on top of income."""
    return (1.0 + return_liquid) * (resources - consumption) + income + 3.0


def compounded_next_liquid(
    resources: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Liquid law compounding the return over two sub-periods."""
    return (1.0 + return_liquid) ** 2 * (resources - consumption) + income


def doubled_subsidy_resources(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand crediting the subsidy twice over."""
    return liquid + 2.0 * subsidy


def fee_charging_resources(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand net of a literal participation fee."""
    return liquid + subsidy - 0.5


def interest_bearing_resources(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
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
    params = copy.deepcopy(toy.build_params(final_age_alive=_FINAL_AGE_ALIVE))
    params["alive"]["utility"].update(extra)
    return params


def test_a_scaled_utility_enters_the_solved_objective():
    """A flat scale on `utility` changes the value function it produces.

    The kernel evaluates the declared felicity, so a scale the regime writes is
    part of the problem solved rather than structure quietly dropped from it.
    """
    scaled = _build(utility_func=scaled_utility).solve(
        params=_params(util_scale=2.0), log_level="debug"
    )
    plain = _build().solve(params=_params(), log_level="debug")
    assert not np.allclose(
        np.asarray(scaled[0]["alive"]), np.asarray(plain[0]["alive"])
    )


def test_a_rescaled_liquid_law_is_refused_by_the_single_liquid_kernels():
    """An extra parameter in the liquid law is rejected rather than ignored.

    The kernel evaluates `(1 + return_liquid) * savings + income`, so a law that
    rescales income before adding it is a budget the kernel never applies.
    """
    with pytest.raises(RegimeInitializationError, match="income_scale"):
        _build(liquid_law=scaled_next_liquid)


@pytest.mark.parametrize(
    "liquid_law",
    [
        doubled_income_next_liquid,
        taxed_return_next_liquid,
        endowed_next_liquid,
        compounded_next_liquid,
    ],
)
def test_a_same_signature_liquid_law_is_refused_by_the_single_liquid_kernels(
    liquid_law,
):
    """A law is judged by what it computes, not by which parameters it names.

    Each of these declares exactly the parameters the kernels read and would pass
    any name-level check, yet none of them equals
    `(1 + return_liquid) * savings + income`. Accepting one would solve a budget
    the regime never declared and report it as the regime's own.
    """
    with pytest.raises(RegimeInitializationError, match="next_liquid"):
        _build(liquid_law=liquid_law)


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


def test_the_supported_fixed_form_still_builds():
    """The plain CRRA / affine-law regime the kernels do solve is still accepted."""
    model = _build()
    solution = model.solve(params=_params(), log_level="debug")
    assert any("alive" in period for period in solution.values())
