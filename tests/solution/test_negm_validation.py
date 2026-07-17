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
from lcm import DiscreteGrid, ExtremeValueTasteShocks, categorical
from lcm.exceptions import ModelInitializationError
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
    solver = dataclasses.replace(
        negm_kinked_toy.NEGM_SOLVER, outer_post_decision="not_a_function"
    )
    regime = _VALID.replace(solver=solver)
    with pytest.raises(ModelInitializationError, match="neither a declared function"):
        _validate(regime)


def test_margin_distinctness_recheck_rejects_outer_action_equal_to_inner_action():
    """The model-build re-check rejects an outer action equal to the inner one.

    `NEGM.__post_init__` enforces distinctness at construction (so a coincident
    `NEGM` cannot be built); the validator carries the same check as a single
    fail-loud model-build point, exercised here against an inner config whose
    continuous action matches the solver's outer action.
    """
    solver = negm_kinked_toy.NEGM_SOLVER
    inner_action_clashes = dataclasses.replace(
        solver.inner, continuous_action="illiquid_investment"
    )
    with pytest.raises(ModelInitializationError, match="coincides with the inner"):
        _fail_if_margins_not_distinct(
            regime_name="alive", solver=solver, inner=inner_action_clashes
        )


def test_margin_distinctness_recheck_rejects_outer_equal_to_inner_post_decision():
    """The model-build re-check rejects a coincident post-decision function."""
    solver = negm_kinked_toy.NEGM_SOLVER
    inner_post_clashes = dataclasses.replace(
        solver.inner, post_decision_function="next_illiquid"
    )
    with pytest.raises(ModelInitializationError, match="coincides with"):
        _fail_if_margins_not_distinct(
            regime_name="alive", solver=solver, inner=inner_post_clashes
        )


def _euler_law_reading_outer_margin(
    liquid_savings: FloatND, next_illiquid: ContinuousState
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
    inner Euler inversion is no longer independent of the outer choice, so
    NEGM's deterministic outer max is invalid.
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
    consumption: ContinuousAction, next_illiquid: ContinuousState
) -> FloatND:
    """A utility that multiplies consumption by the outer post-decision.

    The cross-term makes the inner marginal utility depend on the outer choice,
    so the durable margin is not additively separable from consumption.
    """
    flow = consumption * (1.0 + 0.01 * next_illiquid)
    return flow ** (1.0 - 2.0) / (1.0 - 2.0)


def test_utility_coupling_the_two_margins_is_rejected_with_2d_pointer():
    """A non-additively-separable utility cross-term fails fast.

    NEGM treats the outer margin's utility term as a constant in the inner Euler
    inversion; a cross-term in `(consumption, next_illiquid)` breaks that.
    """
    regime = _VALID.replace(
        functions={
            **dict(_VALID.functions),
            "utility": _utility_coupling_consumption_and_durable_move,
        },
    )
    with pytest.raises(ModelInitializationError, match="G2EGM / multidim-RFC"):
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
            "labor_supply": DiscreteGrid(_Work),
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
            "labor_supply": DiscreteGrid(_Work),
        },
        functions={
            **dict(_VALID.functions),
            "is_working": _is_working,
        },
    )
    with pytest.raises(ModelInitializationError, match="stacked outer"):
        _validate(regime)


def test_passive_state_after_the_durable_is_rejected_with_layout_explanation():
    """A passive continuous state declared after the durable violates the layout.

    The stacked outer carry lifts each candidate by a per-durable-state credited
    cost, addressing the durable as the last passive axis; a passive state
    declared after it would occupy that axis instead. The regime is rejected
    with the required layout named.
    """
    regime = _VALID.replace(
        states={
            **dict(_VALID.states),
            "ride_along": negm_kinked_toy.ILLIQUID_GRID,
        },
    )
    with pytest.raises(ModelInitializationError, match="last"):
        _validate(regime)


def _credited_reading_the_euler_state(
    wealth: ContinuousState, illiquid: ContinuousState, next_illiquid: ContinuousState
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
    wealth: ContinuousState, next_illiquid: ContinuousState
) -> FloatND:
    """A cost-free resources base that reads the outer margin directly."""
    return wealth + 5.0 + 0.01 * next_illiquid


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
    wealth: ContinuousState, credited: FloatND
) -> FloatND:
    """A user-defined resources function alongside a declared outer cost."""
    return wealth + 5.0 - credited


def _cost_free_base(wealth: ContinuousState) -> FloatND:
    """A cost-free resources base for the composed-resources tests."""
    return wealth + 5.0


def test_user_defined_resources_with_a_declared_outer_cost_is_rejected():
    """With a declared outer cost the resources function is composed by pylcm.

    Affineness of resources in the cost holds by construction only when pylcm
    performs the subtraction itself; a user-defined resources function
    alongside a declared `NEGM.outer_cost` is rejected at model build with a
    pointer to the cost-free base.
    """
    regime = _VALID.replace(
        functions={
            **dict(_VALID.functions),
            "resources": _resources_defined_by_the_user,
        },
    )
    with pytest.raises(ModelInitializationError, match="composed by pylcm"):
        finalize_regimes(user_regimes={"alive": regime}, derived_categoricals={})


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
        finalize_regimes(user_regimes={"alive": regime}, derived_categoricals={})


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
        user_regimes={"alive": regime}, derived_categoricals={}
    )["alive"]
    composed = cast("UserFunction", finalized.functions["resources"])

    assert float(
        composed(
            resources_before_outer_cost=jnp.asarray(7.0), credited=jnp.asarray(2.0)
        )
    ) == pytest.approx(5.0)


def test_missing_outer_cost_with_costful_resources_is_rejected():
    """Omitting `NEGM.outer_cost` while resources reads the outer margin fails.

    With `outer_cost=None` the user defines the resources function directly and
    the lift credits nothing, so a resources function that depends on the outer
    post-decision (here through the `credited` function it reads) is rejected
    with a pointer to `NEGM.outer_cost`.
    """
    solver = dataclasses.replace(negm_kinked_toy.NEGM_SOLVER, outer_cost=None)
    regime = _VALID.replace(
        solver=solver,
        functions={
            **dict(_VALID.functions),
            "resources": _resources_defined_by_the_user,
        },
    )
    with pytest.raises(ModelInitializationError, match="declares no outer cost"):
        _validate(regime)


def test_undeclared_outer_cost_function_is_rejected():
    """An `outer_cost` name that is not a regime function fails at model build."""
    solver = dataclasses.replace(
        negm_kinked_toy.NEGM_SOLVER, outer_cost="not_a_function"
    )
    regime = _VALID.replace(solver=solver)
    with pytest.raises(ModelInitializationError, match="not a declared function"):
        _validate(regime)


def _keep_reading_the_euler_state(
    illiquid: ContinuousState, wealth: ContinuousState
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
        functions={
            **dict(_VALID.functions),
            "keep_illiquid": _keep_reading_the_euler_state,
        },
    )
    with pytest.raises(ModelInitializationError, match="unary function of the durable"):
        _validate(regime)
