"""Spec for NEGM build-time validation (the fail-loudly model contract).

A regime with `solver=NEGM(...)` must satisfy the nesting contract on top of the
inner DC-EGM contract. Every violation raises `ModelInitializationError` at model
build, naming the offending feature **and** the correct alternative solver. The
checks run on the user regimes directly (`validate_negm_regimes`), so each case
constructs regimes and asserts the rejection without building kernels or solving.

The cases mutate the valid kinked-toy NEGM regime one rule at a time:

1. no outer margin (outer action absent) → use `DCEGM`,
2. outer action equals the inner continuous action, or outer post-decision
   equals the inner post-decision → reject (distinct margins),
3. coupled-2-Euler: the outer post-decision enters the inner Euler-state
   transition → use the 2-D EGM foundation,
4. taste-shock ordering: a taste-shocked discrete choice exists → reject.
"""

import dataclasses
from typing import cast

import jax.numpy as jnp
import pytest

from _lcm.egm.negm_validation import (
    _fail_if_margins_not_distinct,
    validate_negm_regimes,
)
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.solution.negm import _BoundNEGM
from lcm import (
    DiscreteGrid,
    ExtremeValueTasteShocks,
    IrregSpacedGrid,
    LinearAggregator,
    LinearExpectation,
    Phased,
    categorical,
)
from lcm.certainty_equivalent import PowerMean
from lcm.consumption_savings_regime import (
    NetOfAdjustmentCost,
    post_decision_lower_bound,
)
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserFunction,
)
from tests.test_models import negm_kinked_toy

_VALID = negm_kinked_toy.build_alive_regime()


def _validate(regime: UserRegime) -> None:
    """Run the NEGM contract check on a single-regime mapping."""
    validate_negm_regimes(user_regimes={"alive": regime})


def test_valid_kinked_toy_negm_regime_passes_validation():
    """The kinked-toy NEGM regime satisfies the nesting contract."""
    _validate(_VALID)


def test_outer_action_absent_is_rejected_with_dcegm_pointer():
    """A regime with no outer continuous action is a pure 1-D problem.

    Dropping the durable action (and the durable state it moves) leaves a
    single continuous action; NEGM would silently run as plain DC-EGM, so it is
    rejected with a pointer to `DCEGM`.
    """
    regime = _VALID.replace(
        actions={"consumption": negm_kinked_toy.CONSUMPTION_GRID},
    )
    with pytest.raises(ModelInitializationError, match="use `DCEGM`"):
        _validate(regime)


def test_outer_post_decision_not_declared_is_rejected():
    """An outer post-decision that is neither a function nor a transition fails."""
    outer = dataclasses.replace(
        _VALID.outer_continuous, post_decision_state="not_a_function"
    )
    regime = _VALID.replace(outer_continuous=outer)
    with pytest.raises(ModelInitializationError, match="neither a declared function"):
        _validate(regime)


def test_margin_distinctness_recheck_rejects_outer_action_equal_to_inner_action():
    """The model-build re-check rejects an outer action equal to the inner one.

    `NEGM.__post_init__` enforces distinctness at construction (so a coincident
    `NEGM` cannot be built); the validator carries the same check as a single
    fail-loud model-build point, exercised here against an inner config whose
    continuous action matches the solver's outer action.
    """
    solver = cast("_BoundNEGM", _VALID.solver)
    inner_action_clashes = dataclasses.replace(
        solver.inner, continuous_action="illiquid_investment"
    )
    with pytest.raises(ModelInitializationError, match="coincides with the inner"):
        _fail_if_margins_not_distinct(
            regime_name="alive", solver=solver, inner=inner_action_clashes
        )


def test_margin_distinctness_recheck_rejects_outer_equal_to_inner_post_decision():
    """The model-build re-check rejects a coincident post-decision function."""
    solver = cast("_BoundNEGM", _VALID.solver)
    inner_post_clashes = dataclasses.replace(
        solver.inner, post_decision_function="new_durable"
    )
    with pytest.raises(ModelInitializationError, match="coincides with"):
        _fail_if_margins_not_distinct(
            regime_name="alive", solver=solver, inner=inner_post_clashes
        )


def _euler_law_reading_outer_margin(
    *, liquid_savings: FloatND, next_illiquid: ContinuousState
) -> ContinuousState:
    """A liquid Euler law that reads the outer post-decision (the pension shape).

    The next-period liquid wealth depends on the durable stock the outer choice
    sets, so the `c` and the outer FOCs invert on the same continuation.
    """
    rate = jnp.where(liquid_savings < 0.0, 0.12, 0.03)
    return (1.0 + rate) * liquid_savings + 0.01 * next_illiquid


def test_outer_margin_entering_inner_euler_law_is_rejected_with_2d_pointer():
    """The DS pension coupling fails fast with a pointer to the 2-D foundation.

    When the inner Euler-state transition reads the outer post-decision, the
    inner Euler inversion depends on the outer choice, so NEGM's deterministic
    outer max is invalid.
    """
    regime = _VALID.replace(
        state_transitions={
            "wealth": _euler_law_reading_outer_margin,
            "illiquid": negm_kinked_toy.durable_transition,
        },
    )
    with pytest.raises(ModelInitializationError, match="G2EGM / multidim-RFC"):
        _validate(regime)


def _utility_coupling_consumption_and_durable_move(
    *, consumption: ContinuousAction, new_durable: ContinuousState
) -> FloatND:
    """A utility that multiplies consumption by the outer post-decision.

    The cross-term makes the inner marginal utility depend on the outer choice,
    so the durable margin is not additively separable from consumption.
    """
    flow = consumption * (1.0 + 0.01 * new_durable)
    return flow ** (1.0 - 2.0) / (1.0 - 2.0)


def test_utility_coupling_the_two_margins_is_rejected_with_2d_pointer():
    """A non-additively-separable utility cross-term fails fast.

    NEGM treats the outer margin's utility term as a constant in the inner Euler
    inversion; a cross-term in `(consumption, new_durable)` breaks that.
    """
    regime = _VALID.replace(
        functions={
            **dict(_VALID.functions),
            "utility": _utility_coupling_consumption_and_durable_move,
        },
    )
    with pytest.raises(ModelInitializationError, match="G2EGM / multidim-RFC"):
        _validate(regime)


def _consumption_part(consumption: ContinuousAction) -> FloatND:
    return consumption


def _durable_part(new_durable: ContinuousState) -> FloatND:
    return 1.0 + 0.01 * new_durable


def _multiplicative_utility(
    *, _consumption_part: FloatND, _durable_part: FloatND
) -> FloatND:
    return _consumption_part * _durable_part


def test_utility_coupling_through_helper_branches_is_rejected() -> None:
    """Separability is checked after composing every helper in the utility DAG."""
    regime = _VALID.replace(
        functions={
            **dict(_VALID.functions),
            "_consumption_part": _consumption_part,
            "_durable_part": _durable_part,
            "utility": _multiplicative_utility,
        },
    )
    with pytest.raises(ModelInitializationError, match="not additively separable"):
        _validate(regime)


@categorical(ordered=False)
class _Work:
    work: ScalarInt
    rest: ScalarInt


def _is_working(labor_supply: DiscreteAction) -> FloatND:
    return labor_supply == _Work.work


def test_taste_shocked_discrete_choice_is_rejected_with_ordering_explanation():
    """A taste-shocked discrete choice violates the aggregation ordering.

    NEGM wraps its outer search around the inner solve (which performs the
    discrete `logsumexp`), so the outer max sits outside the taste-shock
    aggregation — the wrong order. The regime is rejected with the §2.3
    explanation.
    """
    regime = _VALID.replace(
        actions={
            **dict(_VALID.actions),
            "labor_supply": DiscreteGrid(category_class=_Work),
        },
        functions={
            **dict(_VALID.functions),
            "is_working": _is_working,
        },
        taste_shocks=ExtremeValueTasteShocks(),
    )
    with pytest.raises(ModelInitializationError, match="outermost aggregation"):
        _validate(regime)


def test_hard_discrete_action_is_rejected_with_carry_layout_explanation():
    """A hard (untaste-shocked) discrete action violates the carry layout.

    The stacked outer continuation carry places the candidate axis directly
    after the durable margin's passive axis; a discrete-action axis would sit
    between them and be mis-identified as the durable when the candidates are
    lifted. The regime is rejected with a pointer to `GridSearch`.
    """
    regime = _VALID.replace(
        actions={
            **dict(_VALID.actions),
            "labor_supply": DiscreteGrid(category_class=_Work),
        },
        functions={
            **dict(_VALID.functions),
            "is_working": _is_working,
        },
    )
    with pytest.raises(ModelInitializationError, match="stacked outer"):
        _validate(regime)


def test_passive_state_after_the_durable_is_order_independent():
    """A passive state may follow the durable in declaration order.

    State names determine the carry layout. The durable axis is located by name,
    so adding an otherwise passive ride-along state after it does not change
    whether the economic model is admissible.
    """
    regime = _VALID.replace(
        states={
            **dict(_VALID.states),
            "ride_along": negm_kinked_toy.ILLIQUID_GRID,
        },
    )
    assert _validate(regime) is None


def _credited_reading_the_euler_state(
    *,
    wealth: ContinuousState,
    illiquid: ContinuousState,
    next_illiquid: ContinuousState,
) -> FloatND:
    """A cost whose wedge scales with liquid wealth — no constant lift exists."""
    return (1.0 + 0.01 * wealth) * (next_illiquid - illiquid)


def test_outer_cost_reading_the_euler_state_is_rejected():
    """The declared outer cost may read only the durable, the target, and params.

    A cost that reads the liquid Euler state varies along the cash-on-hand axis,
    so no constant per-(durable, outer-node) translation exists and the stacked
    lift would place candidates on the wrong axis. The regime is rejected at
    model build.
    """
    regime = _VALID.replace(
        functions={
            **dict(_VALID.functions),
            "credited": _credited_reading_the_euler_state,
        },
    )
    with pytest.raises(ModelInitializationError, match="may read only the durable"):
        _validate(regime)


def _base_reading_the_outer_margin(
    *, wealth: ContinuousState, new_durable: ContinuousState
) -> FloatND:
    """A cost-free resources base that reads the outer margin directly."""
    return wealth + 5.0 + 0.01 * new_durable


def test_resources_base_reading_the_outer_margin_is_rejected():
    """The cost-free resources base must be independent of the outer margin.

    With a declared outer cost, pylcm composes `resources = base - cost`, so
    the base's only legitimate outer-margin channel is the subtracted cost
    itself; a base that reads the outer post-decision directly is rejected at
    model build.
    """
    regime = _VALID.replace(
        functions={
            **{
                name: func
                for name, func in _VALID.functions.items()
                if name != "resources"
            },
            "resources_before_outer_cost": _base_reading_the_outer_margin,
        },
    )
    with pytest.raises(
        ModelInitializationError, match="must not read the outer post-decision"
    ):
        _validate(regime)


def _resources_defined_by_the_user(
    *, wealth: ContinuousState, credited: FloatND
) -> FloatND:
    """A user-defined resources function alongside a declared outer cost."""
    return wealth + 5.0 - credited


def _cost_free_base(wealth: ContinuousState) -> FloatND:
    """A cost-free resources base for the composed-resources tests."""
    return wealth + 5.0


def test_negm_regime_rejects_nonlinear_certainty_equivalent():
    """NEGM, like every continuation-based solver, rejects a nonlinear CE.

    The rejection is keyed on the solver requiring a continuation, not on the
    concrete `DCEGM` type, so an `NEGM` regime declaring a nonlinear certainty
    equivalent is caught at model build with the same expected-utility pointer.
    """
    regime = _VALID.replace(certainty_equivalent=PowerMean())
    with pytest.raises(RegimeInitializationError, match="does not support a nonlinear"):
        finalize_regimes(
            user_regimes={"alive": regime},
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
        )


def test_user_defined_resources_with_a_declared_outer_cost_is_rejected():
    """With a declared outer cost the resources function is composed by pylcm.

    Affineness of resources in the cost holds by construction only when pylcm
    performs the subtraction itself; a user-defined resources function
    alongside a `NetOfAdjustmentCost` on `liquid.resources` is rejected at
    model build with a pointer to the cost-free base.
    """
    regime = _VALID.replace(
        functions={
            **dict(_VALID.functions),
            "resources": _resources_defined_by_the_user,
        },
    )
    with pytest.raises(ModelInitializationError, match="pylcm composes"):
        finalize_regimes(
            user_regimes={"alive": regime},
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
        )


def test_missing_resources_base_with_a_declared_outer_cost_is_rejected():
    """A declared outer cost requires the cost-free resources base function."""
    regime = _VALID.replace(
        functions={
            name: func
            for name, func in _VALID.functions.items()
            if name not in ("resources", "resources_before_outer_cost")
        },
    )
    with pytest.raises(ModelInitializationError, match="resources_before_outer_cost"):
        finalize_regimes(
            user_regimes={"alive": regime},
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
        )


def test_finalize_composes_resources_as_base_minus_outer_cost():
    """With a declared outer cost, `resources = base - cost` is injected.

    The finalized regime carries a synthesized resources function whose inputs
    are the cost-free base and the declared cost, and whose value is exactly
    their difference — affine in the cost by construction.
    """
    regime = _VALID.replace(
        functions={
            **{
                name: func
                for name, func in _VALID.functions.items()
                if name != "resources"
            },
            "resources_before_outer_cost": _cost_free_base,
        },
    )

    finalized = finalize_regimes(
        user_regimes={"alive": regime},
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )["alive"]
    composed = cast("UserFunction", finalized.functions["resources"])

    assert float(
        composed(
            resources_before_outer_cost=jnp.asarray(7.0), credited=jnp.asarray(2.0)
        )
    ) == pytest.approx(5.0)


def test_finalize_composes_resources_with_a_phased_base():
    """A `Phased` cost-free base still yields the composed resources function.

    The synthesized resources function reads the base and the cost by name, so
    phase resolution happens at the producer level: composition succeeds with a
    `Phased` base slot and the injected function is the plain difference of its
    two inputs in either phase.
    """
    regime = _VALID.replace(
        functions={
            **dict(_VALID.functions),
            "resources_before_outer_cost": Phased(
                solve=_cost_free_base, simulate=_cost_free_base
            ),
        },
    )

    finalized = finalize_regimes(
        user_regimes={"alive": regime},
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )["alive"]
    composed = cast("UserFunction", finalized.functions["resources"])

    assert float(
        composed(
            resources_before_outer_cost=jnp.asarray(7.0), credited=jnp.asarray(2.0)
        )
    ) == pytest.approx(5.0)


def test_missing_outer_cost_with_costful_resources_is_rejected():
    """A plain `liquid.resources` reading the outer margin is rejected.

    With `outer_cost=None` the user defines the resources function directly and
    the lift credits nothing, so a resources function that depends on the outer
    post-decision (here through the `credited` function it reads) is rejected
    with a pointer to `liquid.resources`.
    """
    liquid = dataclasses.replace(_VALID.liquid, resources="resources")
    regime = _VALID.replace(
        liquid=liquid,
        functions={
            **dict(_VALID.functions),
            "resources": _resources_defined_by_the_user,
        },
    )
    with pytest.raises(ModelInitializationError, match="declares no outer cost"):
        _validate(regime)


def test_undeclared_outer_cost_function_is_rejected():
    """An `outer_cost` name that is not a regime function fails at model build."""
    resources_spec = _VALID.liquid.resources
    assert isinstance(resources_spec, NetOfAdjustmentCost)
    resources = dataclasses.replace(resources_spec, cost="not_a_function")
    liquid = dataclasses.replace(_VALID.liquid, resources=resources)
    regime = _VALID.replace(liquid=liquid)
    with pytest.raises(ModelInitializationError, match="not a declared function"):
        _validate(regime)


def _keep_reading_the_euler_state(
    *, illiquid: ContinuousState, wealth: ContinuousState
) -> FloatND:
    """A no-adjustment candidate that reads more than the durable state."""
    return illiquid + 0.0 * wealth


def test_no_adjustment_candidate_with_extra_arguments_is_rejected():
    """The no-adjustment candidate must be a unary function of the durable.

    The keeper's no-adjustment level is evaluated as `keep(durable)` in both
    the credited-cost lift and the child-resources query map, so a candidate
    whose signature reads anything else cannot be bound there and is rejected
    at model build.
    """
    regime = _VALID.replace(
        outer_continuous=dataclasses.replace(
            _VALID.outer_continuous,
            no_adjustment="keep_illiquid",
        ),
        functions={
            **dict(_VALID.functions),
            "keep_illiquid": _keep_reading_the_euler_state,
        },
    )
    with pytest.raises(ModelInitializationError, match="unary function of the durable"):
        _validate(regime)


def _next_wealth_from_savings(liquid_savings):
    """An inner Euler law: next period's wealth is what was saved."""
    return liquid_savings


def _outer_law_reading_a_sibling_law(*, new_durable, next_wealth):
    """An outer law that reaches the inner margin through a sibling law."""
    return new_durable + 0.1 * next_wealth


def test_outer_law_coupled_through_another_transition_is_rejected():
    """An outer law reaching the inner margin via a sibling law is rejected.

    Chained state transitions are supported, so a law may read another law's
    output. That makes the chain a path to the inner margin: an outer law
    reading `next_wealth`, where `next_wealth` reads `liquid_savings`, carries
    a stock that depends on the consumption the inner Euler inversion solves
    for, exactly as a direct read would.
    """
    regime = _VALID.replace(
        state_transitions={
            "wealth": _next_wealth_from_savings,
            "illiquid": _outer_law_reading_a_sibling_law,
        },
    )
    with pytest.raises(ModelInitializationError, match="inner margin"):
        _validate(regime)


def _next_wealth_from_the_state_alone(wealth):
    """A sibling law that reads a state and nothing else."""
    return wealth


def _outer_law_reading_an_independent_sibling(*, new_durable, next_wealth):
    """An outer law chained only through a law that never sees the inner margin."""
    return new_durable + 0.1 * next_wealth


def test_outer_law_chained_through_an_independent_law_is_accepted():
    """A chain that never reaches the inner margin is legitimate and builds.

    Rejecting every chained law would forbid the supported pattern rather than
    the coupling, so the traversal must follow the chain and then find nothing.
    """
    regime = _VALID.replace(
        state_transitions={
            "wealth": _next_wealth_from_the_state_alone,
            "illiquid": _outer_law_reading_an_independent_sibling,
        },
    )
    _validate(regime)


def test_a_declared_lower_bound_matching_the_savings_grid_is_accepted():
    """Stating the limit the nested solve already enforces passes validation."""
    regime = _VALID.replace(
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=_VALID.liquid,
                lower=float(negm_kinked_toy.SAVINGS_GRID.start),
            )
        },
    )
    _validate(regime)


def test_a_declared_lower_bound_disagreeing_with_the_savings_grid_is_refused():
    """A NEGM regime's declared limit is checked against its own savings grid.

    The lowest node of the inner savings grid is the limit the nested solve
    enforces, so a declaration naming a different number is refused rather than
    dropped as something the grid already guarantees.
    """
    declared = float(negm_kinked_toy.SAVINGS_GRID.start) - 100.0
    regime = _VALID.replace(
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=_VALID.liquid, lower=declared
            )
        },
    )

    with pytest.raises(ModelInitializationError, match="lower bound"):
        _validate(regime)


def test_the_refusal_names_the_nested_savings_grids_lowest_node():
    """The message carries the number the NEGM savings grid actually enforces."""
    grid_start = float(negm_kinked_toy.SAVINGS_GRID.start)
    regime = _VALID.replace(
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=_VALID.liquid, lower=grid_start - 100.0
            )
        },
    )

    with pytest.raises(ModelInitializationError) as excinfo:
        _validate(regime)

    assert str(grid_start) in str(excinfo.value)


def test_runtime_outer_state_grid_is_refused_during_negm_validation() -> None:
    """The outer state grid must expose the nodes the continuation layout reads."""
    regime = _VALID.replace(
        states={
            **dict(_VALID.states),
            "illiquid": IrregSpacedGrid(
                n_points=int(negm_kinked_toy.ILLIQUID_GRID.n_points)
            ),
        }
    )

    with pytest.raises(ModelInitializationError, match=r"outer state grid.*runtime"):
        _validate(regime)


def test_runtime_inner_savings_grid_is_refused_during_negm_validation() -> None:
    """The nested inversion must know its savings nodes when kernels are built."""
    bound = cast("_BoundNEGM", _VALID.solver)
    solver = dataclasses.replace(
        bound,
        inner=dataclasses.replace(
            bound.inner,
            savings_grid=IrregSpacedGrid(
                n_points=int(negm_kinked_toy.SAVINGS_GRID.n_points)
            ),
        ),
    )

    with pytest.raises(ModelInitializationError, match=r"inner savings grid.*runtime"):
        _validate(_VALID.replace(solver=solver))


def test_runtime_outer_search_grid_is_refused_during_negm_validation() -> None:
    """The outer search must know the candidate nodes when kernels are built."""
    solver = dataclasses.replace(
        cast("_BoundNEGM", _VALID.solver),
        outer_grid=IrregSpacedGrid(n_points=int(negm_kinked_toy.OUTER_GRID.n_points)),
    )

    with pytest.raises(ModelInitializationError, match=r"outer search grid.*runtime"):
        _validate(_VALID.replace(solver=solver))
