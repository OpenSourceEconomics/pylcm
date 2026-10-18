"""Public declarations for finite query component records."""

import jax.numpy as jnp

from _lcm.egm.dcegm_query import DCEGMQuery
from _lcm.egm.step import _EGMStep
from _lcm.solution.continuation_target import union_free_params
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.negm import _NEGMPeriodKernel
from _lcm.typing import EconFunctionKwargs
from lcm import (
    AgeGrid,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    Regime,
    outer_unchanged,
    post_decision_lower_bound,
)
from lcm.solvers import DCEGM, NEGM
from lcm.typing import BoolND, ContinuousState, UserFunction
from tests.egm import _finite_germ_oracle as original
from tests.egm.test_dynamic_dcegm_activation import RegimeId


def _current_stock_is_negative(*, stock: ContinuousState) -> BoolND:
    return stock < 0.0


def make_canonical_model(
    *,
    false_predicate: bool = False,
    flow_utility: UserFunction | None = None,
    literal_child: bool = False,
    child_gross: UserFunction | None = None,
) -> Model:
    """Declare the finite nested model with its original grids and economic laws."""
    liquid = IrregSpacedGrid(points=(0.0, 0.5, 1.0, 2.0))
    stock = IrregSpacedGrid(points=(0.0, 1.0))
    action = IrregSpacedGrid(points=(0.125, 0.5, 1.0, 2.0))
    savings_grid = IrregSpacedGrid(points=(0.0, 0.25, 1.0, 2.0))
    nested_margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources=NetOfAdjustmentCost(
            output="resources", before_cost="gross", cost="cost"
        ),
        post_decision_state="savings",
    )
    return Model(
        regimes={
            "parent": NestedConsumptionSavingsRegime(
                states={"liquid": liquid, "stock": stock},
                actions={
                    "consumption": action,
                    "investment": IrregSpacedGrid(points=(-1.0, 0.0, 1.0, 2.0)),
                },
                functions={
                    "gross": original._gross,
                    "cost": original._cost,
                    "new_stock": original._new_stock,
                    "savings": original._savings,
                    "utility": original._parent_utility,
                },
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=nested_margin, lower=0.0
                    )
                },
                state_transitions={
                    "liquid": original._next_liquid,
                    "stock": original._same_stock,
                },
                liquid=nested_margin,
                outer_continuous=OuterContinuousMargin(
                    state="stock",
                    action="investment",
                    post_decision_state="new_stock",
                    no_adjustment=outer_unchanged,
                ),
                solver=NEGM(
                    inner=DCEGM(savings_grid=savings_grid),
                    outer_grid=IrregSpacedGrid(points=(1.0, 2.0)),
                ),
            ),
            "child": NestedConsumptionSavingsRegime(
                states={"liquid": liquid, "stock": stock},
                actions={
                    "consumption": action,
                    "investment": IrregSpacedGrid(points=(-1.0, 0.0, 1.0, 2.0)),
                },
                functions={
                    "gross": original._gross if child_gross is None else child_gross,
                    "cost": original._cost,
                    "new_stock": original._new_stock,
                    "savings": original._savings,
                    "utility": original._utility
                    if flow_utility is None
                    else flow_utility,
                },
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=nested_margin,
                        lower=0.0 if literal_child else original._floor,
                    ),
                    **(
                        {"current_stock": _current_stock_is_negative}
                        if false_predicate
                        else {}
                    ),
                },
                state_transitions={
                    "liquid": original._next_liquid,
                    "stock": original._next_stock,
                },
                liquid=nested_margin,
                outer_continuous=OuterContinuousMargin(
                    state="stock",
                    action="investment",
                    post_decision_state="new_stock",
                    no_adjustment=outer_unchanged,
                ),
                solver=NEGM(
                    inner=DCEGM(savings_grid=savings_grid),
                    outer_grid=IrregSpacedGrid(points=(1.0, 2.0)),
                ),
            ),
            "done": Regime(
                states={"liquid": liquid, "stock": stock},
                functions={"utility": original._terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
        enable_jit=False,
    )


def make_canonical_query(
    *, false_predicate: bool = False, flow_utility: UserFunction | None = None
) -> tuple[DCEGMQuery, EconFunctionKwargs]:
    """Install an actual nested target, optionally with a declared current predicate."""
    model = make_canonical_model(
        false_predicate=false_predicate, flow_utility=flow_utility
    )
    kernel = model._regimes["parent"].solution.period_kernels[0]
    assert isinstance(kernel, _NEGMPeriodKernel)
    keeper = kernel.keeper_kernel
    assert isinstance(keeper, _DCEGMPeriodKernel)
    function = keeper.core_programs()["replay"].function
    assert isinstance(function, _EGMStep)
    plan = function.pieces.continuation_plan
    assert plan.query_targets == ("child",)
    query = plan.child_reads["child"].query
    assert isinstance(query, DCEGMQuery)
    authority = query.model.authority
    assert authority is model._regimes["child"].solution.query_authority
    assert authority.outer_mode == "finite_max"
    assert authority.finite_outer is not None
    assert authority.finite_outer.outer_nodes == (1.0, 2.0)
    assert authority.bound.kind == "dag"
    assert authority.state_names == ("stock",)
    assert authority.action_names == ()
    assert authority.corner_coordinate_kind == "compact_interval"
    assert model.ages is not None
    projected = union_free_params(
        flat_params=model._process_params(
            {"discount_factor": 1.0}, runtime_checks=False
        ),
        regime_name="parent",
        transition_target_names=("child",),
    )
    pool = {
        **{
            name.removeprefix("child__"): value
            for name, value in projected.items()
            if name.startswith("child__")
        },
        "period": jnp.int32(1),
        "age": model.ages.values[1],
        "liquid": jnp.asarray(0.0),
        "stock": jnp.asarray(0.0),
    }
    return query, pool
