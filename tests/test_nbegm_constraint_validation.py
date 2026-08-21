"""Spec for what an NBEGM regime may declare in its `constraints` slot.

NBEGM compiles conjunctive, ordered comparisons on its current liquid state
into owner-aware feasibility boundaries. The candidate envelope, published
grid rows, and cross-period carry all enforce that geometry. Constraints that
need an action, post-decision value, another state, or unsupported Boolean
structure remain outside the route and are refused when the model is built.
"""

from collections.abc import Callable, Mapping
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import _lcm.solution.nbegm as nbegm_module
import lcm
from lcm import (
    GridBreakpoint,
    LinSpacedGrid,
    Model,
    PiecewiseLinSpacedGrid,
)
from lcm.consumption_savings_regime import (
    LiquidMargin,
    post_decision_lower_bound,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND
from tests.conftest import DECIMAL_PRECISION
from tests.test_models import (
    n_nbegm_toy,
    nbegm_floor_toy,
    nbegm_jump_schedule_toy,
    nbegm_medicaid_toy,
    nbegm_tax_toy,
)
from tests.test_models.nbegm_common import (
    make_alive_dead_model,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_medicaid_toy import (
    medicaid_eligible,
    resources,
    subsidy,
    subsidy_medicaid,
    subsidy_private,
)


def rationing(consumption: ContinuousAction, liquid: ContinuousState) -> BoolND:
    """A feasibility predicate whose boundary is a root in no declared coordinate.

    Deliberately not a bound on the post-decision state: no savings-grid node
    and no case boundary locates it, so no EGM-family kernel can enforce it.
    """
    return jnp.square(consumption) + jnp.square(liquid) <= 400.0


def _build_model(
    *, variant: str, constraints: Mapping[str, Callable[..., object]]
) -> Model:
    """Assemble the Medicaid one-asset toy with an arbitrary constraint pool."""
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=10,
        liquid_max=20.0,
        n_consumption=12,
        alive_functions={
            "utility": utility,
            "medicaid_eligible": medicaid_eligible,
            "subsidy_medicaid": subsidy_medicaid,
            "subsidy_private": subsidy_private,
            "subsidy": subsidy,
            "resources": resources,
            "savings": savings,
        },
        liquid_law=next_liquid_from_savings,
        alive_solver=resolve_solver(
            variant,
            savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=10),
        ),
        constraints=constraints,
    )


def _savings_from_liquid(
    liquid: ContinuousState, consumption: ContinuousAction
) -> FloatND:
    """Return the post-decision liquid state from direct resources."""
    return liquid - consumption


def _build_smooth_model(
    *,
    constraints: Mapping[str, Callable[..., object]],
    jump_read: str = "one_sided",
) -> Model:
    """Assemble a smooth direct-resources NBEGM model with an aligned threshold."""
    liquid_grid = PiecewiseLinSpacedGrid(
        start=0.1,
        stop=8.0,
        breakpoints=(GridBreakpoint(value=4.0, owner="right"),),
        points_per_segment=(5, 5),
    )
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=10,
        liquid_max=8.0,
        n_consumption=40,
        alive_functions={"utility": utility, "savings": _savings_from_liquid},
        liquid_law=next_liquid_from_savings,
        alive_solver=resolve_solver(
            "nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=8.0, n_points=30),
            jump_read=jump_read,
        ),
        constraints=constraints,
        liquid_grid=liquid_grid,
        liquid_resources="liquid",
    )


def _build_scheduled_model(
    *,
    functions: Mapping[str, Callable[..., object]],
    constraints: Mapping[str, Callable[..., object]],
) -> Model:
    """Assemble a one-asset NBEGM model around a declared budget schedule."""
    liquid_grid = PiecewiseLinSpacedGrid(
        start=0.1,
        stop=8.0,
        breakpoints=(GridBreakpoint(value=4.0, owner="right"),),
        points_per_segment=(5, 5),
    )
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=10,
        liquid_max=8.0,
        n_consumption=40,
        alive_functions={"utility": utility, "savings": savings, **functions},
        liquid_law=next_liquid_from_savings,
        alive_solver=resolve_solver(
            "nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=8.0, n_points=30),
        ),
        constraints=constraints,
        liquid_grid=liquid_grid,
    )


def _smooth_params(*, asset_limit: float | None = 4.0) -> dict:
    """Return the complete parameter tree for the smooth feasibility toy."""
    transition = {"return_liquid": 0.03, "income": 1.0}
    next_regime = {"final_age_alive": 2.0}
    constraint_params = {} if asset_limit is None else {"asset_limit": asset_limit}
    return {
        "alive": {
            "utility": {"crra": 2.0},
            "koopmans_aggregator": {"discount_factor": 0.95},
            "asset_test": constraint_params,
            "alive": {
                "next_liquid": transition,
                "next_regime": next_regime,
            },
            "dead": {
                "next_liquid": transition,
                "next_regime": next_regime,
            },
        },
        "dead": {"utility": {"crra": 2.0}},
    }


def test_nbegm_masks_a_compiled_parameter_interval_in_the_solved_value() -> None:
    """A compiled current-liquid half-space is enforced at every published row."""
    declaration = cast(
        "Callable[..., object]",
        lcm.ref("liquid") >= lcm.ref("asset_limit"),
    )
    model = _build_smooth_model(constraints={"asset_test": declaration})

    solution = model.solve(params=_smooth_params(), log_level="off")
    liquid = cast(
        "PiecewiseLinSpacedGrid", model.user_regimes["alive"].states["liquid"]
    ).to_jax()
    value = solution[0]["alive"]

    assert np.all(np.isneginf(value[liquid < 4.0]))
    assert np.all(np.isfinite(value[liquid >= 4.0]))


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        (lcm.ref("liquid") >= -1.0, lambda liquid: jnp.ones_like(liquid, dtype=bool)),
        (lcm.ref("liquid") > 9.0, lambda liquid: jnp.zeros_like(liquid, dtype=bool)),
        (
            (lcm.ref("liquid") >= 4.0) & (lcm.ref("liquid") < 7.0),
            lambda liquid: (liquid >= 4.0) & (liquid < 7.0),
        ),
        (
            (lcm.ref("liquid") >= 4.0) & (lcm.ref("liquid") <= 4.0),
            lambda liquid: liquid == 4.0,
        ),
    ],
)
def test_nbegm_enforces_full_empty_and_intersected_feasible_domains(
    declaration,
    expected,
) -> None:
    """Out-of-domain and conjunctive thresholds mask exactly their declared set."""
    model = _build_smooth_model(
        constraints={"asset_test": cast("Callable[..., object]", declaration)}
    )

    solution = model.solve(params=_smooth_params(asset_limit=None), log_level="off")
    liquid = cast(
        "PiecewiseLinSpacedGrid", model.user_regimes["alive"].states["liquid"]
    ).to_jax()
    value = solution[0]["alive"]
    feasible = expected(liquid)

    assert np.all(np.isfinite(value[feasible]))
    assert np.all(np.isneginf(value[~feasible]))


@pytest.mark.parametrize(
    ("functions", "budget_params"),
    [
        (
            {"tax": nbegm_tax_toy.tax, "resources": nbegm_tax_toy.resources},
            {
                "tax": {"tax_rate": 0.3, "tax_exemption": 3.0},
                "resources": {"base_income": 1.0},
            },
        ),
        (
            {
                "coh_floor": nbegm_floor_toy.coh_floor,
                "resources": nbegm_floor_toy.resources,
            },
            {
                "coh_floor": {"floor_asset": 3.0},
                "resources": {"base_income": 1.0},
            },
        ),
    ],
)
def test_nbegm_composes_feasibility_with_kinks_and_flat_budget_floors(
    functions,
    budget_params,
) -> None:
    """Supported budget refinements retain the compiled feasible half-space."""
    declaration = cast("Callable[..., object]", lcm.ref("liquid") >= 4.0)
    model = _build_scheduled_model(
        functions=functions,
        constraints={"asset_test": declaration},
    )
    params = _smooth_params(asset_limit=None)
    params["alive"].update(budget_params)

    solution = model.solve(params=params, log_level="off")
    liquid = cast(
        "PiecewiseLinSpacedGrid", model.user_regimes["alive"].states["liquid"]
    ).to_jax()
    value = solution[0]["alive"]

    assert np.all(np.isneginf(value[liquid < 4.0]))
    assert np.all(np.isfinite(value[liquid >= 4.0]))


def test_nbegm_rejects_feasibility_composed_with_a_finite_schedule_jump() -> None:
    """A value jump and a feasibility edge require a combined topology."""
    declaration = cast("Callable[..., object]", lcm.ref("liquid") >= 4.0)

    with pytest.raises(
        ModelInitializationError,
        match=r"feasibility.*finite-value schedule jump",
    ):
        _build_scheduled_model(
            functions={
                "subsidy": nbegm_jump_schedule_toy.subsidy,
                "resources": nbegm_jump_schedule_toy.resources,
            },
            constraints={"asset_test": declaration},
        )


def test_nbegm_refuses_a_constraint_its_kernel_cannot_evaluate():
    """The refusal identifies the opaque declaration NBEGM cannot compile."""
    with pytest.raises(
        ModelInitializationError,
        match=r"cannot compile constraint 'rationing'.*opaque",
    ):
        _build_model(variant="nbegm", constraints={"rationing": rationing})


def test_nbegm_builder_consumes_each_compiled_boundary_once(monkeypatch):
    """The builder pairs one planned boundary with its processed predicate."""
    consumed = []
    original = nbegm_module._consume_nbegm_feasibility_constraints

    def recording_consumer(*, context, solver_path):
        result = original(context=context, solver_path=solver_path)
        consumed.extend((context, item) for item in result)
        return result

    monkeypatch.setattr(
        nbegm_module,
        "_consume_nbegm_feasibility_constraints",
        recording_consumer,
    )
    lower = cast("Callable[..., object]", lcm.ref("liquid") >= 4.0)
    upper = cast("Callable[..., object]", lcm.ref("liquid") < 16.0)

    _build_smooth_model(
        constraints={"lower_asset_test": lower, "upper_asset_test": upper},
    )

    assert len(consumed) == 2
    assert [
        (
            item.program.constraint_name,
            item.predicate
            is context.constraint_functions[item.program.constraint_name],
        )
        for context, item in consumed
    ] == [("lower_asset_test", True), ("upper_asset_test", True)]


def test_nbegm_rejects_bridged_carry_with_a_feasibility_boundary() -> None:
    """Compiled feasibility requires a one-sided cross-period carry."""
    declaration = cast("Callable[..., object]", lcm.ref("liquid") >= 4.0)

    with pytest.raises(ModelInitializationError, match="bridged"):
        _build_smooth_model(
            constraints={"asset_test": declaration},
            jump_read="bridged",
        )


def test_nbegm_rejects_feasibility_composed_with_a_binary_case_piece() -> None:
    """A finite jump and an external feasibility boundary require one topology."""
    declaration = cast("Callable[..., object]", lcm.ref("liquid") >= 4.0)

    with pytest.raises(
        ModelInitializationError,
        match=r"feasibility.*case|case.*feasibility",
    ):
        _build_model(
            variant="nbegm",
            constraints={"asset_test": declaration},
        )


def test_nbegm_compiles_a_constraint_parameter_threshold():
    """A declared threshold parameter is resolved in the engine's flat namespace."""
    declaration = cast(
        "Callable[..., object]",
        lcm.ref("liquid") >= lcm.ref("asset_limit"),
    )

    model = _build_smooth_model(constraints={"asset_test": declaration})

    assert "alive" in model.user_regimes


def test_grid_search_accepts_the_same_constraint():
    """The refusal is NBEGM's own: `GridSearch` evaluates the predicate and builds."""
    model = _build_model(variant="brute", constraints={"rationing": rationing})
    assert "rationing" in model._regimes["alive"].solution.constraints


def nested_rationing(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    """A general feasibility predicate over the nested toy's liquid margin."""
    return jnp.square(consumption) + jnp.square(wealth) <= 400.0


def test_nnbegm_refuses_a_constraint_its_kernel_cannot_evaluate():
    """A nested NB-EGM regime declaring a feasibility predicate fails to build."""
    with pytest.raises(ModelInitializationError, match="nested_rationing"):
        n_nbegm_toy.build_model(
            variant="n_nbegm",
            constraints={"nested_rationing": nested_rationing},
        )


def test_grid_search_accepts_the_same_nested_constraint():
    """The refusal is the solver's: `GridSearch` builds the same declaration."""
    model = n_nbegm_toy.build_model(
        variant="brute",
        constraints={"nested_rationing": nested_rationing},
    )
    assert "nested_rationing" in model._regimes["alive"].solution.constraints


def test_nnbegm_filters_the_boundary_plan_for_each_inner_route(monkeypatch) -> None:
    """Adjuster and keeper inner builders receive only their own route ledger."""
    threshold = float(np.asarray(n_nbegm_toy.WEALTH_GRID.to_jax()[4]))
    declaration = cast(
        "Callable[..., object]",
        lcm.ref("wealth") >= threshold,
    )
    received_paths = []
    original = nbegm_module._consume_nbegm_feasibility_constraints

    def recording_consumer(*, context, **kwargs):
        assert context.constraint_plan is not None
        received_paths.append(
            {entry.route.solver_path for entry in context.constraint_plan.entries}
        )
        return original(context=context, **kwargs)

    monkeypatch.setattr(
        nbegm_module,
        "_consume_nbegm_feasibility_constraints",
        recording_consumer,
    )

    n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        constraints={"wealth_floor": declaration},
    )

    assert received_paths == [
        {("nnbegm", "adjuster")},
        {("nnbegm", "keeper")},
    ]


def test_nnbegm_enforces_a_compiled_wealth_boundary() -> None:
    """Keeper and adjuster together publish only feasible wealth rows."""
    wealth = n_nbegm_toy.WEALTH_GRID.to_jax()
    threshold = float(np.asarray(wealth[4]))
    declaration = cast(
        "Callable[..., object]",
        lcm.ref("wealth") >= threshold,
    )
    model = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        constraints={"wealth_floor": declaration},
    )

    solution = model.solve(params={"discount_factor": 0.95}, log_level="off")
    value = np.asarray(solution[0]["alive"])

    assert np.all(np.isneginf(value[wealth < threshold]))
    assert np.all(np.isfinite(value[wealth >= threshold]))


def test_nnbegm_carries_a_compiled_wealth_boundary_between_periods() -> None:
    """The outer envelope retains inner one-sided feasibility topology."""
    wealth = n_nbegm_toy.WEALTH_GRID.to_jax()
    threshold = float(np.asarray(wealth[4]))
    declaration = cast(
        "Callable[..., object]",
        lcm.ref("wealth") >= threshold,
    )
    model = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=3,
        constraints={"wealth_floor": declaration},
    )

    solution = model.solve(params={"discount_factor": 0.95}, log_level="off")
    first_period_value = np.asarray(solution[0]["alive"])

    assert np.all(np.isneginf(first_period_value[wealth < threshold]))
    assert np.all(np.isfinite(first_period_value[wealth >= threshold]))


_LIQUID_MARGIN = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)


def test_nbegm_accepts_a_lower_bound_its_savings_grid_already_enforces():
    """Declaring the limit the savings grid implies builds under NBEGM.

    The kernel inverts on that grid, so its lowest node *is* the borrowing
    limit. A declaration stating the same number adds no predicate the kernel
    would have to evaluate, and is admitted where a general one is refused.
    """
    model = _build_model(
        variant="nbegm",
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=_LIQUID_MARGIN, lower=0.0
            )
        },
    )

    assert "alive" in model.user_regimes


@pytest.mark.parametrize("declared", [-1.0, 1.0])
def test_nbegm_refuses_a_lower_bound_that_disagrees_with_its_savings_grid(declared):
    """A declared limit the grid contradicts is refused, naming the grid's node.

    The grid's lowest node governs both the solve and the simulate-phase mask,
    so a declaration naming a different number would be overridden silently.
    Both directions are checked: a bound above the grid's start claims a
    tighter limit than the kernel enforces, one below claims a looser one, and
    a check written for either alone would pass while the other went unnoticed.
    """
    with pytest.raises(ModelInitializationError, match=r"savings grid starts at 0\.0"):
        _build_model(
            variant="nbegm",
            constraints={
                "borrowing_limit": post_decision_lower_bound(
                    margin=_LIQUID_MARGIN, lower=declared
                )
            },
        )


_NESTED_LIQUID_MARGIN = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="liquid_resources",
    post_decision_state="liquid_savings",
)


def test_nnbegm_accepts_a_lower_bound_its_inner_savings_grid_already_enforces():
    """The nested solver admits the limit its inner savings grid implies."""
    model = n_nbegm_toy.build_model(
        variant="n_nbegm",
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=_NESTED_LIQUID_MARGIN, lower=n_nbegm_toy.SAVINGS_FLOOR
            )
        },
    )

    assert "alive" in model.user_regimes


@pytest.mark.parametrize("offset", [-1.0, 1.0])
def test_nnbegm_refuses_a_lower_bound_that_disagrees_with_its_inner_grid(offset):
    """A nested declaration the inner grid contradicts is refused, either way."""
    with pytest.raises(ModelInitializationError, match="borrowing_limit"):
        n_nbegm_toy.build_model(
            variant="n_nbegm",
            constraints={
                "borrowing_limit": post_decision_lower_bound(
                    margin=_NESTED_LIQUID_MARGIN,
                    lower=n_nbegm_toy.SAVINGS_FLOOR + offset,
                )
            },
        )


def test_declaring_the_bound_leaves_the_nbegm_solution_unchanged():
    """A proved declaration is inert: the savings grid already enforces it.

    Building is the weaker claim — it says the declaration was admitted, not
    that it was disposed of without effect. Solving both arms says the
    admitted declaration adds no mask, no candidate, and no shift in value.
    """
    params = nbegm_medicaid_toy.build_params(final_age_alive=3.0)
    declared = _build_model(
        variant="nbegm",
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=_LIQUID_MARGIN, lower=0.0
            )
        },
    )
    bare = _build_model(variant="nbegm", constraints={})

    with_declaration = declared.solve(params=params, log_level="off")
    without = bare.solve(params=params, log_level="off")

    for period, regime_to_V in without.items():
        for regime_name, V_arr in regime_to_V.items():
            aaae(
                with_declaration[period][regime_name],
                V_arr,
                decimal=DECIMAL_PRECISION,
            )
