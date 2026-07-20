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
from _lcm.solution.solvers import (
    _durable_keeper_transition,
    _with_no_adjustment_outer_function,
)
from _lcm.typing import EconFunction, EconFunctionsMapping
from lcm import DCEGM, NEGM, LinSpacedGrid, NormalIIDProcess
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousState, FloatND
from tests.test_models import negm_kinked_toy

_INNER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="liquid_savings",
    savings_grid=LinSpacedGrid(start=0.0, stop=30.0, n_points=40),
)

_OUTER_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=20)


def _negm(
    *,
    inner: DCEGM = _INNER,
    outer_action: str = "illiquid_investment",
    outer_post_decision: str = "next_illiquid",
    outer_grid: ContinuousGrid = _OUTER_GRID,
    outer_no_adjustment_candidate: str | None = None,
) -> NEGM:
    return NEGM(
        inner=inner,
        outer_action=outer_action,
        outer_post_decision=outer_post_decision,
        outer_grid=outer_grid,
        outer_no_adjustment_candidate=outer_no_adjustment_candidate,
    )


def test_negm_with_valid_fields_constructs():
    """A `NEGM` with distinct margins and a deterministic outer grid constructs."""
    solver = _negm()
    assert solver.inner is _INNER
    assert solver.outer_action == "illiquid_investment"
    assert solver.outer_post_decision == "next_illiquid"
    assert solver.outer_no_adjustment_candidate is None


def test_negm_with_no_adjustment_candidate_constructs():
    """The optional state-specific no-adjustment candidate is accepted."""
    solver = _negm(outer_no_adjustment_candidate="keep_illiquid")
    assert solver.outer_no_adjustment_candidate == "keep_illiquid"


def test_negm_stochastic_outer_grid_is_rejected():
    """A stochastic process cannot serve as the deterministic outer search grid."""
    process = NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)
    with pytest.raises(RegimeInitializationError, match="stochastic process"):
        _negm(outer_grid=process)


def test_negm_outer_action_equal_to_inner_continuous_action_is_rejected():
    """The outer durable margin must differ from the inner consumption action."""
    with pytest.raises(RegimeInitializationError, match="coincides with the inner"):
        _negm(outer_action="consumption")


def test_negm_outer_post_decision_equal_to_inner_post_decision_is_rejected():
    """The outer post-decision must differ from the inner liquid post-decision."""
    with pytest.raises(RegimeInitializationError, match="coincides with"):
        _negm(outer_post_decision="liquid_savings")


def test_negm_invalid_inner_dcegm_is_rejected_by_inner_guards():
    """An invalid inner `DCEGM` is rejected by its own guards before NEGM's.

    The composition reuses `DCEGM.__post_init__` wholesale, so a stochastic
    inner savings grid fails when the inner config is constructed.
    """
    process = NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)
    with pytest.raises(RegimeInitializationError, match="savings_grid"):
        dataclasses.replace(_INNER, savings_grid=process)


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


def test_keeper_no_adjustment_map_threads_every_declared_argument() -> None:
    """A keeper map threads every argument it declares, not only the durable stock.

    A permanent-income deflator `keep(car, growth) = 0.9 * car / growth` reads the
    durable stock and a growth node; the keeper transition carries both arguments
    (copying the map's own annotations) and applies the map, so a stored-value
    normalization can divide the kept stock by the current growth factor.
    """

    def keep(car: ContinuousState, growth: FloatND) -> ContinuousState:
        return car * 0.9 / growth

    transition = _durable_keeper_transition(
        no_adjustment_func=cast("EconFunction", keep),
        durable_state="car",
        outer_post_decision="next_car",
    )

    assert set(inspect.signature(transition).parameters) == {"car", "growth"}
    result = transition(car=jnp.asarray(100.0), growth=jnp.asarray(1.02))
    # The composed transition reruns the same arithmetic at the active float
    # precision, so agreement holds to that dtype's roundoff.
    rtol = 64.0 * float(np.finfo(np.asarray(result).dtype).eps)
    np.testing.assert_allclose(np.asarray(result), 100.0 * 0.9 / 1.02, rtol=rtol)


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
        outer_post_decision="next_car",
        no_adjustment_func=cast("EconFunction", keep),
    )
    injected = updated["next_car"]

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
        outer_post_decision="next_car",
        no_adjustment_func=None,
    )
    injected = updated["next_car"]

    assert set(inspect.signature(injected).parameters) == {"car"}
    result = injected(car=jnp.asarray(42.0))
    np.testing.assert_allclose(np.asarray(result), 42.0, rtol=1e-10)
