"""Spec for the `NEGM` solver configuration and its construction-time guards.

`NEGM` composes an inner `DCEGM` (the 1-D consumption-savings solve) with an
outer deterministic grid search over a durable/illiquid margin. Its
`__post_init__` guards reject — at construction, with a
`RegimeInitializationError` — an outer grid that is a stochastic process, an
outer action that coincides with the inner continuous action, and an outer
post-decision that coincides with the inner post-decision. The remaining case
builds a NEGM model and asserts its simulate phase carries the inner DC-EGM
budget constraint. Nothing here solves a model.
"""

import dataclasses
import inspect
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.budget import DCEGM_BUDGET_CONSTRAINT_NAME
from _lcm.grids import ContinuousGrid
from _lcm.solution.negm import (
    _with_no_adjustment_outer_function,
)
from _lcm.typing import EconFunction, EconFunctionsMapping
from lcm import (
    DCEGM,
    NEGM,
    AgeGrid,
    GridSearch,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    NormalIIDProcess,
    OuterContinuousMargin,
    outer_unchanged,
)
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousState, FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.test_models import negm_kinked_toy

_INNER = DCEGM(
    savings_grid=LinSpacedGrid(start=0.0, stop=30.0, n_points=40),
)

_OUTER_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=20)


def _negm(
    *,
    inner: DCEGM = _INNER,
    outer_grid: ContinuousGrid = _OUTER_GRID,
) -> NEGM:
    return NEGM(inner=inner, outer_grid=outer_grid)


def _nested_regime(
    *,
    outer_action: str = "illiquid_investment",
    outer_post_decision: str = "new_illiquid",
    outer_no_adjustment_candidate: str = outer_unchanged,
) -> NestedConsumptionSavingsRegime:
    functions = {
        "utility": lambda consumption: consumption,
        "resources": lambda wealth: wealth,
        "liquid_savings": lambda resources, consumption: resources - consumption,
        outer_post_decision: lambda illiquid, illiquid_investment: (
            illiquid + illiquid_investment
        ),
    }
    if outer_no_adjustment_candidate != outer_unchanged:
        functions[outer_no_adjustment_candidate] = lambda illiquid: illiquid
    return NestedConsumptionSavingsRegime(
        transition=lambda: 0,
        states={"wealth": _OUTER_GRID, "illiquid": _OUTER_GRID},
        state_transitions={
            "wealth": lambda liquid_savings: liquid_savings,
            "illiquid": lambda new_illiquid: new_illiquid,
        },
        actions={"consumption": _OUTER_GRID, outer_action: _OUTER_GRID},
        functions=functions,
        solver=_negm(),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="liquid_savings",
        ),
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action=outer_action,
            post_decision_state=outer_post_decision,
            no_adjustment=outer_no_adjustment_candidate,
        ),
    )


def test_negm_with_valid_fields_constructs():
    """A nested regime owns distinct liquid and outer margin declarations."""
    regime = _nested_regime()
    assert regime.liquid.state == "wealth"
    assert regime.outer_continuous.action == "illiquid_investment"
    assert regime.outer_continuous.post_decision_state == "new_illiquid"
    assert regime.outer_continuous.no_adjustment == outer_unchanged


def test_negm_with_no_adjustment_candidate_constructs():
    """The optional state-specific no-adjustment declaration is accepted."""
    regime = _nested_regime(outer_no_adjustment_candidate="keep_illiquid")
    assert regime.outer_continuous.no_adjustment == "keep_illiquid"


def test_negm_stochastic_outer_grid_is_rejected():
    """A stochastic process cannot serve as the deterministic outer search grid."""
    process = NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)
    with pytest.raises(RegimeInitializationError, match="stochastic process"):
        _negm(outer_grid=process)


def test_negm_outer_action_equal_to_inner_continuous_action_is_rejected():
    """The outer durable margin must differ from the inner consumption action."""
    with pytest.raises(RegimeInitializationError, match="must not collide"):
        _nested_regime(outer_action="consumption")


def test_negm_outer_post_decision_equal_to_inner_post_decision_is_rejected():
    """The outer post-decision must differ from the inner liquid post-decision."""
    with pytest.raises(RegimeInitializationError, match="must not collide"):
        _nested_regime(outer_post_decision="liquid_savings")


def test_negm_invalid_inner_dcegm_is_rejected_by_inner_guards():
    """An invalid inner `DCEGM` is rejected by its own guards before NEGM's.

    The composition reuses `DCEGM.__post_init__` wholesale, so a stochastic
    inner savings grid fails when the inner config is constructed.
    """
    process = NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)
    with pytest.raises(RegimeInitializationError, match="savings_grid"):
        dataclasses.replace(_INNER, savings_grid=process)


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_negm_simulate_phase_synthesizes_inner_budget_constraint():
    """A NEGM regime's simulate phase carries the inner DC-EGM budget mask.

    NEGM nests the same 1-D consumption-savings solve as `DCEGM`, so the
    forward-simulation grid argmax needs the inner liquid feasibility mask
    `consumption <= resources - borrowing_limit` exactly as a DC-EGM regime
    does. The mask is built from `solver.inner`. The solve phase is
    unaffected — the inner EGM kernels enforce the bound intrinsically and
    never see the synthesized constraint.
    """
    model = negm_kinked_toy.build_model()
    alive = model._regimes["alive"]
    assert DCEGM_BUDGET_CONSTRAINT_NAME in alive.simulation.constraints
    assert DCEGM_BUDGET_CONSTRAINT_NAME not in alive.solution.constraints


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_negm_configuration_does_not_change_reachability() -> None:
    """Nested EGM and grid search expose equal lifecycle graphs and hashes."""

    def next_wealth(
        wealth: ContinuousState, liquid_savings: FloatND
    ) -> ContinuousState:
        return negm_kinked_toy.next_wealth(liquid_savings) + 0.0 * wealth

    final_age_alive = 20 + (negm_kinked_toy.N_PERIODS - 2) * 5
    alive = negm_kinked_toy.build_alive_regime().replace(
        state_transitions={
            **negm_kinked_toy.build_alive_regime().state_transitions,
            "wealth": next_wealth,
        }
    )
    ages = AgeGrid(
        start=20,
        stop=20 + (negm_kinked_toy.N_PERIODS - 1) * 5,
        step="5Y",
    )
    negm_model = Model(
        regimes={"alive": alive, "dead": negm_kinked_toy.build_dead_regime()},
        regime_id_class=negm_kinked_toy.RegimeId,
        ages=ages,
        fixed_params={"final_age_alive": final_age_alive},
    )
    grid_search_model = Model(
        regimes={
            "alive": alive.replace(solver=GridSearch()),
            "dead": negm_kinked_toy.build_dead_regime(),
        },
        regime_id_class=negm_kinked_toy.RegimeId,
        ages=ages,
        fixed_params={"final_age_alive": final_age_alive},
    )

    assert negm_model.reachability == grid_search_model.reachability
    assert hash(negm_model.reachability) == hash(grid_search_model.reachability)
    assert (
        negm_model._regimes["alive"].solution.reachability
        is negm_model.reachability.solution
    )


def test_keeper_outer_function_threads_every_declared_argument() -> None:
    """The injected outer post-decision reads every argument the keeper map declares.

    A growth-deflating keeper `keep(car, growth)` feeds the econ-function DAG through
    `_with_no_adjustment_outer_function`, which must declare and thread both the
    durable stock and the growth node so concatenation wires the DAG's growth value
    in — not only the durable leaf.
    """

    def resources(next_car: ContinuousState) -> FloatND:
        return next_car

    def growth(perm_income: FloatND) -> FloatND:
        return perm_income

    def keep(car: ContinuousState, growth: FloatND) -> ContinuousState:
        return car * 0.9 / growth

    functions = cast(
        "EconFunctionsMapping",
        MappingProxyType({"resources": resources, "growth": growth}),
    )
    updated = _with_no_adjustment_outer_function(
        functions=functions,
        durable_state="car",
        outer_post_decision="new_car",
        no_adjustment_func=cast("EconFunction", keep),
    )
    injected = updated["new_car"]

    assert set(inspect.signature(injected).parameters) == {"car", "growth"}
    result = injected(car=jnp.asarray(100.0), growth=jnp.asarray(1.02))
    # Same arithmetic at the active float precision; see above.
    rtol = 64.0 * float(np.finfo(np.asarray(result).dtype).eps)
    np.testing.assert_allclose(np.asarray(result), 100.0 * 0.9 / 1.02, rtol=rtol)


def test_keeper_outer_function_identity_holds_the_durable_stock() -> None:
    """With no keeper map the injected outer post-decision holds the durable stock.

    A plain durable regime has `no_adjustment_func=None`, so `next_<durable>` is the
    identity on the durable leaf: it declares only the durable stock and returns it
    unchanged.
    """

    def resources(next_car: ContinuousState) -> FloatND:
        return next_car

    functions = cast("EconFunctionsMapping", MappingProxyType({"resources": resources}))
    updated = _with_no_adjustment_outer_function(
        functions=functions,
        durable_state="car",
        outer_post_decision="new_car",
        no_adjustment_func=None,
    )
    injected = updated["new_car"]

    assert set(inspect.signature(injected).parameters) == {"car"}
    result = injected(car=jnp.asarray(42.0))
    np.testing.assert_allclose(np.asarray(result), 42.0, rtol=1e-10)
