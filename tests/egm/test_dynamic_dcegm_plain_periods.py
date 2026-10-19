"""Plain parents bind canonical DC child queries at the target period."""

import functools
from collections.abc import Callable
from typing import Literal, cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import core_program_graph
from _lcm.solution.egm import _EGMCore, _RegimeMassGuardedCore
from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LinearAggregator,
    LiquidMargin,
    Model,
    Phased,
    Regime,
    post_decision_lower_bound,
)
from lcm.exceptions import ModelInitializationError
from lcm.solvers import DCEGM, EGM
from lcm.typing import ContinuousState, FloatND, FunctionName, UserFunction
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import (
    RegimeId,
    _inverse,
    _lower,
    _ordinary_law,
    _parent_resources,
    _resources,
    _savings,
    _terminal,
    _to_child,
    _utility,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _resources_at_age(age: float) -> UserFunction:
    """Shift the resource intercept by one quarter per target period."""
    intercept = 1.0 + 0.25 * (age - 1.0)

    def resources(*, liquid: ContinuousState) -> FloatND:
        return 2.0 * liquid + intercept

    return resources


def _discount_at_age(age: float) -> UserFunction:
    """Return one half at age one and three quarters at age two."""
    beta = 0.5 + 0.25 * (age - 1.0)

    def discount_factor() -> FloatND:
        return jnp.asarray(beta)

    return discount_factor


def _linear_aggregator_at_age(age: float) -> UserFunction:  # noqa: ARG001
    """Build the default aggregator through a public specialization marker."""
    return LinearAggregator()


def _grouping_model(
    *,
    specialization: Literal["none", "resources", "discount"],
    child_aggregator: UserFunction | Phased[UserFunction, UserFunction] | None = None,
) -> Model:
    """Build a source at periods zero and one with a DC child one period later."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    functions: dict[FunctionName, UserFunction] = {
        "utility": _utility,
        "resources": _resources,
        "savings": _savings,
        "inverse_marginal_utility": _inverse,
    }
    if specialization == "resources":
        functions["resources"] = AgeSpecializedFunction(
            build=_resources_at_age, signature=lambda age: age
        )
    if specialization == "discount":
        functions["discount_factor"] = AgeSpecializedFunction(
            build=_discount_at_age, signature=lambda age: age
        )
    child = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=np.array([0.0, 0.125, 0.25, 0.5]))},
        actions={
            "consumption": IrregSpacedGrid(
                points=np.array([0.025, 0.05, 0.25, 0.5, 1.0])
            )
        },
        functions=functions,
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=_lower)
        },
        state_transitions={"liquid": _ordinary_law},
        koopmans_aggregator=child_aggregator,
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0])),
            n_constrained_points=4,
        ),
    )
    parent = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=np.array([0.225, 0.25]))},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.025, 0.05, 0.1]))},
        functions={
            "utility": _utility,
            "resources": _parent_resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse,
        },
        constraints={
            "parent_borrowing": post_decision_lower_bound(margin=margin, lower=0.2)
        },
        state_transitions={"liquid": _to_child},
        solver=EGM(savings_grid=IrregSpacedGrid(points=np.array([0.2, 0.3, 0.4]))),
        koopmans_aggregator=LinearAggregator()
        if child_aggregator is not None
        else None,
    )
    return Model(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={"liquid": IrregSpacedGrid(points=np.array([5.5, 7.0, 8.5]))},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        edges={"parent": {"child": (0, 1)}, "child": {"done": (1, 2)}},
        initial_nodes={0: "parent", 1: "parent"},
        regime_id_class=RegimeId,
    )


def _actual_plain_core(*, model: Model, period: int) -> _EGMCore:
    """Unwrap the installed guarded program without replacing its producer."""
    kernel = model._regimes["parent"].solution.period_kernels[period]
    function = core_program_graph(kernel=kernel)["main"].function
    seen: set[int] = set()
    while not isinstance(function, _EGMCore):
        assert id(function) not in seen
        seen.add(id(function))
        if isinstance(function, functools.partial):
            function = function.func
        elif isinstance(function, _RegimeMassGuardedCore):
            function = function.core
        else:
            assert hasattr(function, "__wrapped__")
            function = cast("Callable[..., FloatND]", function.__wrapped__)
    return function


@pytest.mark.parametrize("specialization", ["none", "resources", "discount"])
def test_plain_parent_groups_only_matching_actual_child_evaluators(
    *, specialization: Literal["none", "resources", "discount"]
) -> None:
    """Bind resource and discount evaluators at child periods one and two."""
    model = _grouping_model(specialization=specialization)
    cores = [_actual_plain_core(model=model, period=period) for period in (0, 1)]
    assert (cores[0] is cores[1]) == (specialization == "none")
    groups = model._regimes["parent"].solution.solver_period_group_keys
    assert (groups[0] == groups[1]) == (specialization == "none")
    for period, core in enumerate(cores):
        query = core.next_dc_query
        assert query is not None
        canonical = query.model
        assert canonical.functions["koopmans_aggregator"] is (
            canonical.koopmans_aggregator
        )
        assert canonical.authority.target_periods == (1, 2)
        actual_resource = query.resources(liquid=jnp.asarray(0.25))
        np.testing.assert_array_equal(
            actual_resource,
            1.5 + (0.25 * period if specialization == "resources" else 0.0),
        )
        if specialization == "discount":
            np.testing.assert_array_equal(
                query.read_discount_factor({}), 0.5 + 0.25 * period
            )


@pytest.mark.parametrize("phased", [False, True])
def test_plain_parent_dc_child_rejects_specialized_solve_aggregator(
    *, phased: bool
) -> None:
    """Reject a custom solve aggregator before constructing an Euler reader."""
    marker = AgeSpecializedFunction(
        build=_linear_aggregator_at_age, signature=lambda age: age
    )
    aggregator = Phased(solve=marker, simulate=LinearAggregator()) if phased else marker
    with pytest.raises(
        ModelInitializationError,
        match="Regime 'child' declares a custom solve-phase Koopmans aggregator",
    ):
        _grouping_model(specialization="none", child_aggregator=aggregator)
