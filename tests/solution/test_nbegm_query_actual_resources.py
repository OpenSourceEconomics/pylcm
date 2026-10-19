"""A finite NN parent's ordinary NB child query reports nonfinite budgets.

The parent's outer grid reaches the child's interior stock 1/2, between the
child's stored passive nodes 0 and 1. Every fixture's budget is finite at both
stored nodes; the nonfinite fixtures are NaN or infinite only at the reached
interior state. Such a budget is invalid query arithmetic and must reach the
host as `query_coverage_missing`, whatever the economic domain, winner or
diagnostic level. Finite budgets, including empty ones, stay economic.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_query import NBEGMQuery
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from _lcm.solution.nnbegm import _NNBEGMPeriodKernel
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
    Regime,
    categorical,
    outer_unchanged,
    post_decision_lower_bound,
)
from lcm.exceptions import InvalidValueFunctionError
from lcm.solvers import NBEGM, NNBEGM, FiniteOuterGrid
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserFunction,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution._nb_query_resource_oracle import (
    NONFINITE_KINDS,
    ResourceKind,
    fail_if_fixture_domain_wrong,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_LIQUID = IrregSpacedGrid(points=np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0]))
_STOCK = IrregSpacedGrid(points=np.array([0.0, 1.0]))
_OUTER = (0.0, 0.5, 1.0)
_SAVINGS = IrregSpacedGrid(points=np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0]))
_CONSUMPTION = IrregSpacedGrid(
    points=np.array([0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0])
)
_INVESTMENT = IrregSpacedGrid(points=np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
PARAMS = {
    "parent": {"koopmans_aggregator": {"discount_factor": 0.5}},
    "child": {"koopmans_aggregator": {"discount_factor": 0.0}},
}


@categorical(ordered=False)
class RegimeId:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


def new_stock(
    *, stock: ContinuousState, investment: ContinuousAction
) -> ContinuousState:
    return stock + investment


def parent_resources(
    *, liquid: ContinuousState, stock: ContinuousState, new_stock: ContinuousState
) -> FloatND:
    return liquid + 1.25 - (new_stock - stock) ** 2 / 2.0


def child_nan_resources(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
    return liquid + 1.25 + jnp.sqrt((stock - 0.5) ** 2 - 0.01)


def child_positive_infinity_resources(
    *, liquid: ContinuousState, stock: ContinuousState
) -> FloatND:
    return liquid + 1.25 + 1.0 / (stock - 0.5) ** 2


def child_negative_infinity_resources(
    *, liquid: ContinuousState, stock: ContinuousState
) -> FloatND:
    return liquid + 8.0 - 1.0 / (stock - 0.5) ** 2


def child_positive_resources(
    *, liquid: ContinuousState, stock: ContinuousState
) -> FloatND:
    return liquid + 1.25 + 4.0 * stock * (1.0 - stock)


def child_zero_resources(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
    return liquid + 1.25 - 5.0 * stock * (1.0 - stock)


def child_negative_resources(
    *, liquid: ContinuousState, stock: ContinuousState
) -> FloatND:
    return liquid + 1.25 - 64.0 * stock * (1.0 - stock)


_RESOURCES: dict[ResourceKind, UserFunction] = {
    ResourceKind.NAN: child_nan_resources,
    ResourceKind.POSITIVE_INFINITY: child_positive_infinity_resources,
    ResourceKind.NEGATIVE_INFINITY: child_negative_infinity_resources,
    ResourceKind.POSITIVE: child_positive_resources,
    ResourceKind.ZERO_AT_FLOOR: child_zero_resources,
    ResourceKind.NEGATIVE_FINITE: child_negative_resources,
}


def utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def inverse_marginal_utility(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def floor(*, stock: ContinuousState) -> FloatND:
    return 0.0 * stock


def next_liquid(*, savings: FloatND) -> ContinuousState:
    return savings


def next_parent_stock(*, new_stock: ContinuousState) -> ContinuousState:
    return new_stock


def next_child_stock(*, stock: ContinuousState) -> ContinuousState:
    return stock


def terminal(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
    return liquid + stock


def make_model(
    *,
    resources: UserFunction,
    outer_nodes: tuple[float, ...] = _OUTER,
) -> Model:
    """Build a finite NN parent whose outer grid reaches an ordinary NB child."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    common: dict[str, UserFunction] = {
        "utility": utility,
        "savings": savings,
        "inverse_marginal_utility": inverse_marginal_utility,
    }
    parent = NestedConsumptionSavingsRegime(
        states={"liquid": _LIQUID, "stock": _STOCK},
        actions={"consumption": _CONSUMPTION, "investment": _INVESTMENT},
        state_transitions={"liquid": next_liquid, "stock": next_parent_stock},
        functions={**common, "resources": parent_resources, "new_stock": new_stock},
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=floor)
        },
        liquid=margin,
        outer_continuous=OuterContinuousMargin(
            state="stock",
            action="investment",
            post_decision_state="new_stock",
            no_adjustment=outer_unchanged,
        ),
        solver=NNBEGM(
            inner=NBEGM(savings_grid=_SAVINGS),
            outer_search=FiniteOuterGrid(
                grid=IrregSpacedGrid(points=np.array(outer_nodes))
            ),
        ),
    )
    child = ConsumptionSavingsRegime(
        koopmans_aggregator=parent.koopmans_aggregator,
        certainty_equivalent=parent.certainty_equivalent,
        states={"liquid": _LIQUID, "stock": _STOCK},
        actions={"consumption": _CONSUMPTION},
        state_transitions={"liquid": next_liquid, "stock": next_child_stock},
        functions={**common, "resources": resources},
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=floor)
        },
        liquid=margin,
        solver=NBEGM(savings_grid=_SAVINGS),
    )
    return Model(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={"liquid": _LIQUID, "stock": _STOCK},
                functions={"utility": terminal},
            ),
        },
        ages=AgeGrid(exact_values=(0, 10, 20)),
        edges={"parent": {"child": (0,)}, "child": {"done": (10,)}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
    )


def fail_if_not_actual_nb_query_path(*, model: Model) -> None:
    """Fail unless both parent branches read the child through its NB query."""
    kernel = model._regimes["parent"].solution.period_kernels[0]
    assert isinstance(kernel, _NNBEGMPeriodKernel)
    child = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(child, _RideAlongNBEGMPeriodKernel)
    for inner in (kernel.keeper_kernel, kernel.adjuster_kernel):
        plan = inner.continuation_plan
        assert plan.query_targets == ("child",)
        reader = plan.child_reads["child"].query
        assert isinstance(reader, NBEGMQuery)
        authority = reader.model.authority
        assert authority.outer_mode == "none"
        assert authority.state_names == ("stock",)
        assert authority.target_periods == (1,)
        assert authority.interval_source_identity == ()
        assert authority.bound.kind == "dag"


def _qualified_model(*, kind: ResourceKind) -> Model:
    fail_if_fixture_domain_wrong(kind=kind)
    model = make_model(resources=_RESOURCES[kind])
    fail_if_not_actual_nb_query_path(model=model)
    return model


@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("kind", sorted(NONFINITE_KINDS, key=lambda kind: kind.value))
def test_nonfinite_child_budget_raises_query_coverage_missing(
    *, kind: ResourceKind, runtime_checks: bool
) -> None:
    """An invalid reached budget fails at the host even if its candidate loses."""
    model = _qualified_model(kind=kind)
    with pytest.raises(InvalidValueFunctionError, match="query_coverage_missing"):
        model.solve(params=PARAMS, log_level="off", runtime_checks=runtime_checks)


@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize(
    "kind",
    [ResourceKind.POSITIVE, ResourceKind.ZERO_AT_FLOOR, ResourceKind.NEGATIVE_FINITE],
)
def test_finite_child_budget_solves(
    *, kind: ResourceKind, runtime_checks: bool
) -> None:
    """Finite positive, empty and negative budgets keep their economic meaning."""
    model = _qualified_model(kind=kind)
    model.solve(params=PARAMS, log_level="off", runtime_checks=runtime_checks)
