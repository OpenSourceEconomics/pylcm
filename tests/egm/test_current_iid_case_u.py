"""Conditional DC candidates precede current-shock integration and full tangents."""

import functools
import math
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dags import with_signature
from jax.extend.core import ClosedJaxpr, Jaxpr

from _lcm.certainty_equivalent import PowerMean, QuasiArithmeticMean
from _lcm.current_iid import CurrentIIDPlan
from _lcm.egm.asset_row import (
    _continuation_of_euler_state,
    _publish_node_selection,
    _SolveOneNode,
    publish_node_from_bracket,
)
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import ContinuationPlan, _ChildRead
from _lcm.egm.current_iid_case_u import (
    CaseUCurrentNodes,
    CaseUPointSolver,
    realise_case_u_nodes,
)
from _lcm.egm.current_iid_ce import EconomicCEContinuation
from _lcm.egm.current_iid_economic import read_case_u_continuation
from _lcm.egm.current_iid_power_bank import _bank_coefficients
from _lcm.egm.current_iid_power_math import _materialize, _SignedLog, _sum_signed
from _lcm.egm.current_iid_power_mean import (
    PowerMeanNodeBank,
    reduce_power_mean_economic,
)
from _lcm.egm.step_core import ResourcesOfState, _EgmKernelPieces
from _lcm.egm.upper_envelope import get_bracket_finder, get_upper_envelope
from _lcm.egm.upper_envelope.fues import QueryBracket
from _lcm.shock_registry import ShockBinding
from _lcm.solution.feasibility import BoundedResult
from _lcm.typing import EconFunctionArg, EconFunctionKwargs
from lcm import IrregSpacedGrid, LinSpacedGrid, categorical
from lcm.shocks import Discrete, GaussHermite, Normal, Product, ShockGroup
from lcm.solvers import DCEGM, ExactEnvelope
from lcm.typing import (
    FloatND,
    ReferenceName,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    TransitionFunctionName,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON, X64_ENABLED

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class _Outcome:
    low: ScalarInt
    high: ScalarInt


def _probabilities(liquid: ScalarFloat) -> FloatND:
    high = liquid / 4.0
    return jnp.stack((1.0 - high, high))


def _location(liquid: ScalarFloat) -> ScalarFloat:
    return liquid / 4.0


def _resources(
    *, liquid: ScalarFloat, e: ScalarInt, eta: ScalarFloat, **_: EconFunctionArg
) -> ScalarFloat:
    return 1.0 + liquid + e + eta


def _utility(
    *, consumption: ScalarFloat, e: ScalarInt, eta: ScalarFloat, **_: EconFunctionArg
) -> ScalarFloat:
    return (1.0 + e + eta) * jnp.log(consumption)


def _inverse(
    *,
    marginal_continuation: ScalarFloat,
    e: ScalarInt,
    eta: ScalarFloat,
    **_: EconFunctionArg,
) -> ScalarFloat:
    return (1.0 + e + eta) / marginal_continuation


@with_signature(args=["savings"], enforce=False)
def _next(**kwargs: EconFunctionArg) -> Mapping[TransitionFunctionName, ScalarFloat]:
    return {"next_wealth": jnp.asarray(kwargs["savings"])}


def _child_resources(*, wealth: ScalarFloat, **_: EconFunctionArg) -> ScalarFloat:
    return wealth


@with_signature(args=[], enforce=False)
def _target_probs(
    *_args: EconFunctionArg, **_kwargs: EconFunctionArg
) -> MappingProxyType[RegimeName, FloatND]:
    return MappingProxyType({"done": jnp.asarray(1.0)})


def _discount(_: EconFunctionKwargs) -> dict[ReferenceName, EconFunctionArg]:
    return {"discount_factor": jnp.asarray(1.0)}


def _point_solver() -> CaseUPointSolver:
    """Bind genuine Euler inversion and ExactEnvelope to an original log carry."""
    saving = jnp.asarray([0.25, 0.5, 2.0 / 3.0, 0.8, 9.0 / 7.0, 7.0 / 5.0, 2.0, 3.0])
    child = EGMCarry(
        endog_grid=saving,
        value=jnp.log(saving),
        marginal_utility=1.0 / saving,
        taste_shock_scale=jnp.asarray(0.0),
    )
    read = _ChildRead(
        next_state_func=_next,
        next_state_key="next_wealth",
        euler_state_func=_next,
        euler_state_name="wealth",
        has_taste_shocks=False,
        resources_func=_child_resources,
        resources_arg_names=frozenset({"wealth"}),
        resources_param_names=frozenset(),
        age_values=jnp.asarray([0.0, 1.0]),
        resources_is_simple=True,
        discrete_state_names=(),
        stochastic_flags=(),
        stochastic_state_names=(),
        stochastic_node_values=(),
        process_grid_names=(),
        weight_keys=(),
        weights_func=None,
        passive_state_names=(),
        passive_grids=(),
        row_arg_names=(),
        row_values=(),
        row_block_shape=(),
        n_outer_candidates=0,
    )
    plan = ContinuationPlan(
        stateful_targets=("done",),
        scalar_targets=(),
        child_reads=MappingProxyType({"done": read}),
        compute_regime_transition_probs=_target_probs,
        post_decision_name="savings",
    )
    solver = DCEGM(
        savings_grid=IrregSpacedGrid(points=np.asarray(saving)),
        envelope=ExactEnvelope(),
    )
    pieces = _EgmKernelPieces(
        euler_state_name="liquid",
        action_name="consumption",
        savings_nodes=saving,
        borrowing_limit=saving[0],
        n_constrained=4,
        constrained_ratio=0.1,
        n_pad=64,
        n_carry_rows=64,
        combo_names=(),
        euler_axis_in_V=0,
        utility_func=_utility,
        inverse_marginal_utility_func=_inverse,
        own_resources_func=_resources,
        feasibility_func=None,
        build_W_kwargs=_discount,
        refine=get_upper_envelope(solver=solver, n_refined=64, publish_provenance=True),
        refine_to_bracket=get_bracket_finder(
            solver=solver, n_refined=64, publish_provenance=True
        ),
        continuation_plan=plan,
        check_query_construction=True,
    )
    return CaseUPointSolver(
        pieces=pieces,
        pool={"period": jnp.int32(0), "age": jnp.asarray(0.0)},
        children=MappingProxyType({"done": child}),
    )


def _law_nodes(liquid: ScalarFloat) -> CaseUCurrentNodes:
    group = ShockGroup(
        shocks={
            "e": Discrete(categories=_Outcome, probs=_probabilities),
            "eta": Normal(mu=_location, sigma=0.0),
        },
        nodes=Product(eta=GaussHermite(n=1)),
    )
    binding = ShockBinding(
        group=group,
        state_inputs=frozenset({"liquid"}),
        state_domains={"liquid": LinSpacedGrid(start=0.0, stop=3.0, n_points=3)},
        categorical_domains={},
        functions={},
        params_template={},
        callable_parameters=frozenset(),
    )
    plan = CurrentIIDPlan.create(
        bindings={"current": binding}, age=0, coordinate_kind="period"
    )
    assert plan is not None
    return realise_case_u_nodes(
        plan=plan,
        liquid=liquid,
        liquid_state="liquid",
        arguments={"period": jnp.int32(0), "age": jnp.asarray(0.0)},
        group_params={"current": {}},
        node_operands=plan.node_operands,
        indices=plan.indices,
    )


def _original_point(
    *, solver: CaseUPointSolver, liquid: ScalarFloat, physical: EconFunctionKwargs
) -> tuple[ScalarFloat, ScalarFloat]:
    pool = {**solver.pool, **physical}
    original = _SolveOneNode(
        pieces=solver.pieces,
        combo_pool=pool,
        discount_factor=jnp.asarray(1.0),
        next_regime_to_continuation=solver.children,
        dtype=liquid.dtype,
        stochastic_node_width=None,
        envelope_cell_width=1,
        resolved_process_grids=MappingProxyType({}),
        savings_point_width=None,
        own_resources_of_state=ResourcesOfState(
            resources_func=solver.pieces.own_resources_func,
            euler_state_name="liquid",
            bound=pool,
        ),
        continuation_of_euler_state=functools.partial(
            _continuation_of_euler_state,
            pieces=solver.pieces,
            combo_pool=pool,
            next_regime_to_continuation=solver.children,
            dtype=liquid.dtype,
            stochastic_node_width=None,
            resolved_process_grids=MappingProxyType({}),
        ),
    )(liquid)
    assert isinstance(original, BoundedResult)
    return original.output[:2]


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_conditional_candidates_and_full_economic_marginal() -> None:
    """Distinct original actions retain physical-law and probability derivatives."""
    solver = _point_solver()
    liquid = jnp.asarray(2.0)
    nodes = _law_nodes(liquid)
    np.testing.assert_array_equal(nodes.physical["e"], [0, 1])
    np.testing.assert_array_equal(nodes.physical["eta"], [0.5, 0.5])
    np.testing.assert_array_equal(nodes.physical_tangent["eta"], [0.25, 0.25])
    expected_saving = (Fraction(7, 5), Fraction(9, 7))
    expected_policy = (Fraction(21, 10), Fraction(45, 14))
    values = []
    for index in range(2):
        physical = {name: values[index] for name, values in nodes.physical.items()}
        tangent = {
            name: values[index] for name, values in nodes.physical_tangent.items()
        }
        selected = solver.node(
            liquid=liquid, physical=physical, physical_tangent=tangent
        )
        result = selected.output
        assert bool(result.valid & result.feasible)
        assert not bool(result.corner)
        assert int(result.native_owner) >= 0
        for status in selected.query_status or ():
            assert not any(bool(flag) for flag in status)
        alpha = 1.5 + index
        value = alpha * math.log(float(expected_policy[index])) + math.log(
            float(expected_saving[index])
        )
        values.append(value)
        np.testing.assert_array_almost_equal(
            [result.value, result.policy, result.savings],
            [value, float(expected_policy[index]), float(expected_saving[index])],
            decimal=DECIMAL_PRECISION,
        )
        original = _original_point(solver=solver, liquid=liquid, physical=physical)
        np.testing.assert_array_equal([result.value, result.policy], original)
    integrated = solver.integrate(liquid=liquid, nodes=nodes).output
    expected_marginal = (
        (values[1] - values[0]) / 4.0
        + 235.0 / 252.0
        + sum(math.log(float(c)) for c in expected_policy) / 8.0
    )
    np.testing.assert_array_almost_equal(
        [integrated.value, integrated.economic_marginal],
        [sum(values) / 2.0, expected_marginal],
        decimal=DECIMAL_PRECISION,
    )
    assert bool(integrated.valid)
    assert expected_saving[0] != expected_saving[1]
    assert expected_policy[0] != expected_policy[1]


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_zero_primal_probability_retains_its_boundary_value_derivative() -> None:
    """The one-sided null-event derivative keeps its finite original candidate."""
    solver = _point_solver()
    liquid = jnp.asarray(0.0)
    nodes = _law_nodes(liquid)
    assert bool(jnp.any(nodes.factors == 0.0))
    assert bool(jnp.any(nodes.factor_tangent != 0.0))
    result = solver.integrate(liquid=liquid, nodes=nodes).output
    low = 2.0 * math.log(0.5)
    high = 2.0 * math.log(4.0 / 3.0) + math.log(2.0 / 3.0)
    expected = 2.5 + math.log(0.5) / 4.0 + (high - low) / 4.0
    np.testing.assert_array_almost_equal(
        [result.value, result.economic_marginal],
        [low, expected],
        decimal=DECIMAL_PRECISION,
    )
    assert bool(result.valid)

    invalid = solver.integrate(
        liquid=liquid, nodes=replace(nodes, law_valid=jnp.asarray(a=False))
    ).output
    assert not bool(invalid.valid)
    poisoned = solver.integrate(
        liquid=liquid,
        nodes=replace(
            nodes, factor_tangent=jnp.full_like(nodes.factor_tangent, jnp.nan)
        ),
    ).output
    assert not bool(poisoned.valid)


def _zero_utility(consumption: ScalarFloat) -> ScalarFloat:
    return 0.0 * consumption


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_asset_publication_owns_corner_equality() -> None:
    """The shared source selector keeps the existing literal corner at equality."""
    bracket = QueryBracket(
        lower_grid=jnp.asarray(1.0),
        upper_grid=jnp.asarray(2.0),
        lower_policy=jnp.asarray(0.5),
        upper_policy=jnp.asarray(1.0),
        lower_value=jnp.asarray(0.0),
        upper_value=jnp.asarray(0.0),
        first_grid=jnp.asarray(1.0),
        n_kept=jnp.int32(2),
    )
    value, policy, corner = _publish_node_selection(
        bracket=bracket,
        n_pad=2,
        resources_at_node=jnp.asarray(1.5),
        borrowing_limit=jnp.asarray(0.25),
        utility_of_action=_zero_utility,
        discounted_expected_value_at_limit=jnp.asarray(0.0),
    )
    assert bool(corner)
    np.testing.assert_array_equal([value, policy], [0.0, 1.25])
    original = publish_node_from_bracket(
        bracket=bracket,
        n_pad=2,
        resources_at_node=jnp.asarray(1.5),
        borrowing_limit=jnp.asarray(0.25),
        utility_of_action=_zero_utility,
        discounted_expected_value_at_limit=jnp.asarray(0.0),
    )
    np.testing.assert_array_equal([value, policy], original)


@with_signature(args=["savings", "liquid"], enforce=False)
def _dependent_next(
    **kwargs: EconFunctionArg,
) -> Mapping[TransitionFunctionName, ScalarFloat]:
    return {
        "next_wealth": jnp.asarray(kwargs["savings"]) + jnp.asarray(kwargs["liquid"])
    }


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_source_dependent_future_preserves_genuine_candidate_envelope() -> None:
    """A source law contributes the child's separate economic marginal."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    read = replace(plan.child_reads["done"], euler_state_func=_dependent_next)
    solver = replace(
        original,
        pieces=replace(
            original.pieces,
            continuation_plan=replace(
                plan, child_reads=MappingProxyType({"done": read})
            ),
        ),
    )
    liquid = jnp.asarray(0.5)
    result = solver.node(
        liquid=liquid,
        physical={"e": jnp.int32(0), "eta": jnp.asarray(0.0)},
        physical_tangent={"eta": jnp.asarray(0.0)},
    ).output
    child = original.children["done"]
    assert isinstance(child, EGMCarry)
    economic = jnp.interp(
        result.savings + liquid, child.endog_grid, child.marginal_utility
    )
    np.testing.assert_array_almost_equal(
        result.marginal,
        1.0 / result.policy + economic,
        decimal=DECIMAL_PRECISION,
    )
    assert bool(result.valid)
    canonical = _original_point(
        solver=solver,
        liquid=liquid,
        physical={"e": jnp.int32(0), "eta": jnp.asarray(0.0)},
    )
    np.testing.assert_array_equal([result.value, result.policy], canonical)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_future_direction_uses_economic_row_instead_of_level_slope() -> None:
    """A declared marginal differing from the level slope remains authoritative."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    read = replace(plan.child_reads["done"], euler_state_func=_dependent_next)
    child = EGMCarry(
        endog_grid=jnp.asarray([1.0, 2.0]),
        value=jnp.asarray([1.0, 3.0]),
        marginal_utility=jnp.asarray([7.0, 7.0]),
        taste_shock_scale=jnp.asarray(0.0),
    )
    result = read_case_u_continuation(
        plan=replace(plan, child_reads=MappingProxyType({"done": read})),
        combo_pool={**original.pool, "liquid": jnp.asarray(0.25)},
        direction={"liquid": jnp.asarray(1.0)},
        children=MappingProxyType({"done": child}),
        savings=jnp.asarray(1.0),
        dtype=child.value.dtype,
    )
    np.testing.assert_array_equal(result.output[1], 7.0)


@with_signature(args=["liquid"], enforce=False)
def _future_probs(**kwargs: EconFunctionArg) -> MappingProxyType[RegimeName, FloatND]:
    high = jnp.asarray(kwargs["liquid"]) / 4.0
    return MappingProxyType({"done": 1.0 - high, "other": high})


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_zero_future_probability_preserves_weight_tangent() -> None:
    """The null edge's finite value contributes through its probability derivative."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    child = original.children["done"]
    assert isinstance(child, EGMCarry)
    first = replace(
        child,
        value=jnp.full_like(child.value, 2.0),
    )
    other = replace(first, value=jnp.full_like(first.value, 10.0))
    result = read_case_u_continuation(
        plan=replace(
            plan,
            stateful_targets=("done", "other"),
            child_reads=MappingProxyType(
                {
                    "done": plan.child_reads["done"],
                    "other": plan.child_reads["done"],
                }
            ),
            compute_regime_transition_probs=_future_probs,
        ),
        combo_pool={**original.pool, "liquid": jnp.asarray(0.0)},
        direction={"liquid": jnp.asarray(1.0)},
        children=MappingProxyType({"done": first, "other": other}),
        savings=jnp.asarray(1.0),
        dtype=first.value.dtype,
    )
    np.testing.assert_array_equal(result.output, [2.0, 2.0])


def _source_discount(pool: EconFunctionKwargs) -> dict[ReferenceName, EconFunctionArg]:
    return {"discount_factor": 1.0 + jnp.asarray(pool["liquid"]) / 8.0}


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_genuine_case_u_envelope_retains_discount_tangent() -> None:
    """The Koopmans discount contributes its derivative at original selected savings."""
    original = _point_solver()
    solver = replace(
        original, pieces=replace(original.pieces, build_W_kwargs=_source_discount)
    )
    liquid = jnp.asarray(0.5)
    selected = solver.node(
        liquid=liquid,
        physical={"e": jnp.int32(0), "eta": jnp.asarray(0.0)},
        physical_tangent={"eta": jnp.asarray(0.0)},
    ).output
    continuation = read_case_u_continuation(
        plan=solver.pieces.continuation_plan,
        combo_pool={**solver.pool, "liquid": liquid},
        direction={"liquid": jnp.asarray(1.0)},
        children=solver.children,
        savings=selected.savings,
        dtype=liquid.dtype,
    )
    np.testing.assert_array_almost_equal(
        selected.marginal,
        1.0 / selected.policy + continuation.output[0] / 8.0,
        decimal=DECIMAL_PRECISION,
    )
    assert bool(selected.valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_derivative_needed_future_corruption_survives_zero_probability() -> None:
    """A null edge cannot erase corrupt data needed by the probability derivative."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    child = original.children["done"]
    assert isinstance(child, EGMCarry)
    corrupt = replace(child, value=jnp.full_like(child.value, jnp.nan))
    result = read_case_u_continuation(
        plan=replace(
            plan,
            stateful_targets=("done", "other"),
            child_reads=MappingProxyType(
                {
                    "done": plan.child_reads["done"],
                    "other": plan.child_reads["done"],
                }
            ),
            compute_regime_transition_probs=_future_probs,
        ),
        combo_pool={**original.pool, "liquid": jnp.asarray(0.0)},
        direction={"liquid": jnp.asarray(1.0)},
        children=MappingProxyType({"done": child, "other": corrupt}),
        savings=jnp.asarray(1.0),
        dtype=child.value.dtype,
    )
    assert result.query_status is not None
    assert bool(result.query_status[-1].coverage_missing)


@with_signature(args=["savings", "next_z", "next_w"], enforce=False)
def _future_node_law(
    **kwargs: EconFunctionArg,
) -> Mapping[TransitionFunctionName, ScalarFloat]:
    return {
        "next_wealth": jnp.asarray(kwargs["savings"])
        + jnp.asarray(kwargs["next_z"])
        + jnp.asarray(kwargs["next_w"])
    }


@with_signature(args=["liquid"], enforce=False)
def _future_supports(**kwargs: EconFunctionArg) -> Mapping[str, FloatND]:
    liquid = jnp.asarray(kwargs["liquid"])
    return {
        "support_z": jnp.asarray([0.0, 1.0]) + liquid,
        "support_w": jnp.asarray([0.0, 2.0]) + 2.0 * liquid,
    }


@with_signature(args=["liquid"], enforce=False)
def _future_weights(**kwargs: EconFunctionArg) -> Mapping[str, FloatND]:
    high = jnp.asarray(kwargs["liquid"]) / 4.0
    return {
        "weight_z": jnp.stack((1.0 - high, high)),
        "weight_w": jnp.asarray([0.5, 0.5]),
    }


def _future_node_read(read: _ChildRead) -> _ChildRead:
    return replace(
        read,
        euler_state_func=_future_node_law,
        euler_draw_names=frozenset({"next_z", "next_w"}),
        local_draw_names=("next_z", "next_w"),
        local_draw_sizes=(2, 2),
        local_support_keys=("support_z", "support_w"),
        local_support_func=_future_supports,
        weight_keys=("weight_z", "weight_w"),
        weights_func=_future_weights,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("width", [None, 1, 2])
def test_future_node_support_and_factored_weight_economic_direction(
    width: int | None,
) -> None:
    """Support motion and a null product factor contribute independent terms."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    grid = jnp.arange(1.0, 9.0)
    child = EGMCarry(
        endog_grid=grid,
        value=2.0 * grid,
        marginal_utility=jnp.full_like(grid, 7.0),
        taste_shock_scale=jnp.asarray(0.0),
    )
    result = read_case_u_continuation(
        plan=replace(
            plan,
            child_reads=MappingProxyType(
                {"done": _future_node_read(plan.child_reads["done"])}
            ),
        ),
        combo_pool={**original.pool, "liquid": jnp.asarray(0.0)},
        direction={"liquid": jnp.asarray(1.0)},
        children=MappingProxyType({"done": child}),
        savings=jnp.asarray(1.0),
        dtype=grid.dtype,
        stochastic_node_width=width,
    )
    np.testing.assert_array_equal(result.output, [4.0, 21.5])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_genuine_case_u_future_mesh_preserves_original_candidate() -> None:
    """The actual future-node read feeds both original candidates and source tangent."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    solver = replace(
        original,
        pieces=replace(
            original.pieces,
            continuation_plan=replace(
                plan,
                child_reads=MappingProxyType(
                    {"done": _future_node_read(plan.child_reads["done"])}
                ),
            ),
        ),
    )
    liquid = jnp.asarray(0.0)
    physical = {"e": jnp.int32(0), "eta": jnp.asarray(0.0)}
    selected = solver.node(
        liquid=liquid, physical=physical, physical_tangent={"eta": jnp.asarray(0.0)}
    ).output
    child = original.children["done"]
    assert isinstance(child, EGMCarry)
    saving = selected.savings
    mu_low = jnp.interp(saving, child.endog_grid, child.marginal_utility)
    mu_high = jnp.interp(saving + 2.0, child.endog_grid, child.marginal_utility)
    # The support direction is three at every joint node. The probability
    # direction is a quarter times each z-high/low level difference.
    values = tuple(
        child.value_at(query=saving + offset) for offset in (0.0, 1.0, 2.0, 3.0)
    )
    expected = (
        1.0 / selected.policy
        + 1.5 * (mu_low + mu_high)
        + ((values[1] - values[0]) + (values[3] - values[2])) / 8.0
    )
    np.testing.assert_array_almost_equal(
        selected.marginal, expected, decimal=DECIMAL_PRECISION
    )
    assert bool(selected.valid)
    canonical = _original_point(solver=solver, liquid=liquid, physical=physical)
    np.testing.assert_array_equal([selected.value, selected.policy], canonical)


@with_signature(args=["liquid"], enforce=False)
def _invalid_future_weights(**kwargs: EconFunctionArg) -> Mapping[str, FloatND]:
    weights = _future_weights(**kwargs)
    return {**weights, "weight_w": 2.0 * weights["weight_w"]}


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_future_node_law_integrity_is_not_primal_activity() -> None:
    """An invalid future factor remains a source failure at a null-event boundary."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    read = replace(
        _future_node_read(plan.child_reads["done"]),
        weights_func=_invalid_future_weights,
    )
    result = read_case_u_continuation(
        plan=replace(plan, child_reads=MappingProxyType({"done": read})),
        combo_pool={**original.pool, "liquid": jnp.asarray(0.0)},
        direction={"liquid": jnp.asarray(1.0)},
        children=original.children,
        savings=jnp.asarray(1.0),
        dtype=jnp.asarray(0.0).dtype,
    )
    assert result.query_status is not None
    assert bool(result.query_status[-1].coverage_missing)


def _quadratic_transform(*, value: FloatND, theta: FloatND) -> FloatND:
    return value * value + theta * value


def _quadratic_inverse(*, value: FloatND, theta: FloatND) -> FloatND:
    return (jnp.sqrt(theta * theta + 4.0 * value) - theta) / 2.0


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("liquid", [0.0, 2.0])
def test_declared_ce_uses_economic_value_weight_and_parameter_directions(
    liquid: float,
) -> None:
    """The declared transform and inverse consume explicit economic tangents."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    read = replace(plan.child_reads["done"], euler_state_func=_dependent_next)
    grid = jnp.asarray([1.0, 5.0])
    first = EGMCarry(
        endog_grid=grid,
        value=jnp.full_like(grid, 2.0),
        marginal_utility=jnp.full_like(grid, 7.0),
        taste_shock_scale=jnp.asarray(0.0),
    )
    other = replace(first, value=jnp.full_like(grid, 4.0))
    declared = QuasiArithmeticMean(
        transform=_quadratic_transform, inverse=_quadratic_inverse
    )
    reader = EconomicCEContinuation(
        plan=replace(
            plan,
            stateful_targets=("done", "other"),
            child_reads=MappingProxyType({"done": read, "other": read}),
            compute_regime_transition_probs=_future_probs,
            certainty_equivalent=declared,
        ),
        pool={
            **original.pool,
            "liquid": jnp.asarray(liquid),
            "certainty_equivalent__theta": jnp.asarray(1.0),
        },
        direction={
            "liquid": jnp.asarray(1.0),
            "certainty_equivalent__theta": jnp.asarray(0.5),
        },
        children=MappingProxyType({"done": first, "other": other}),
        dtype=grid.dtype,
    )
    result = reader(jnp.asarray(1.0))
    certainty = (math.sqrt(53.0) - 1.0) / 2.0
    expected = (
        [2.0, 7.0, 7.7]
        if liquid == 0.0
        else [
            certainty,
            49.0 / math.sqrt(53.0),
            (54.0 - 0.5 * certainty) / math.sqrt(53.0),
        ]
    )
    np.testing.assert_array_almost_equal(
        result.output, expected, decimal=DECIMAL_PRECISION
    )
    assert result.query_status is not None
    assert not bool(result.query_status[0].coverage_missing)


@with_signature(args=[], enforce=False)
def _two_target_probs(**_: EconFunctionArg) -> MappingProxyType[RegimeName, FloatND]:
    return MappingProxyType({"done": jnp.asarray(0.5), "other": jnp.asarray(0.5)})


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_genuine_case_u_candidate_and_envelope_share_declared_ce() -> None:
    """Original Euler inversion consumes the same nonlinear economic CE reader."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    first = original.children["done"]
    assert isinstance(first, EGMCarry)
    first = replace(first, value=first.value + 2.0)
    other = replace(first, value=first.value + 1.0)
    read = replace(plan.child_reads["done"], euler_state_func=_dependent_next)
    declared = QuasiArithmeticMean(
        transform=_quadratic_transform, inverse=_quadratic_inverse
    )
    saving = 0.5
    low = 2.0 + math.log(saving)
    high = low + 1.0
    certainty = (
        math.sqrt(1.0 + 2.0 * (low * low + low + high * high + high)) - 1.0
    ) / 2.0
    economic = ((2.0 * low + 1.0) + (2.0 * high + 1.0)) / (
        2.0 * saving * (2.0 * certainty + 1.0)
    )
    alpha = saving * economic / (economic - 1.0)
    solver = replace(
        original,
        pool={**original.pool, "certainty_equivalent__theta": jnp.asarray(1.0)},
        children=MappingProxyType({"done": first, "other": other}),
        pieces=replace(
            original.pieces,
            continuation_plan=replace(
                plan,
                stateful_targets=("done", "other"),
                child_reads=MappingProxyType({"done": read, "other": read}),
                compute_regime_transition_probs=_two_target_probs,
                certainty_equivalent=declared,
            ),
        ),
    )
    result = solver.node(
        liquid=jnp.asarray(0.0),
        physical={"e": jnp.int32(0), "eta": jnp.asarray(alpha - 1.0)},
        physical_tangent={"eta": jnp.asarray(0.0)},
    ).output
    np.testing.assert_array_almost_equal(
        [result.savings, result.marginal],
        [saving, 2.0 * economic],
        decimal=DECIMAL_PRECISION,
    )
    assert bool(result.valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_declared_ce_transforms_original_future_nodes_before_joint_reduction() -> None:
    """The future lottery keeps support and null-factor derivatives inside g."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    grid = jnp.arange(1.0, 9.0)
    child = EGMCarry(
        endog_grid=grid,
        value=2.0 * grid,
        marginal_utility=jnp.full_like(grid, 7.0),
        taste_shock_scale=jnp.asarray(0.0),
    )
    declared = QuasiArithmeticMean(
        transform=_quadratic_transform, inverse=_quadratic_inverse
    )
    reader = EconomicCEContinuation(
        plan=replace(
            plan,
            certainty_equivalent=declared,
            child_reads=MappingProxyType(
                {"done": _future_node_read(plan.child_reads["done"])}
            ),
        ),
        pool={
            **original.pool,
            "liquid": jnp.asarray(0.0),
            "certainty_equivalent__theta": jnp.asarray(1.0),
        },
        direction={"liquid": jnp.asarray(1.0)},
        children=MappingProxyType({"done": child}),
        dtype=grid.dtype,
        stochastic_node_width=2,
    )
    result = reader(jnp.asarray(1.0))
    expected = [
        (math.sqrt(97.0) - 1.0) / 2.0,
        63.0 / math.sqrt(97.0),
        194.5 / math.sqrt(97.0),
    ]
    np.testing.assert_array_almost_equal(
        result.output, expected, decimal=DECIMAL_PRECISION
    )
    assert result.query_status is not None
    assert not bool(result.query_status[0].coverage_missing)


def _equation_count(jaxpr: Jaxpr) -> int:
    count = len(jaxpr.eqns)
    for eqn in jaxpr.eqns:
        for param in eqn.params.values():
            for inner in param if isinstance(param, tuple | list) else (param,):
                if isinstance(inner, ClosedJaxpr):
                    count += _equation_count(inner.jaxpr)
                elif isinstance(inner, Jaxpr):
                    count += _equation_count(inner)
    return count


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("stable", [False, True])
def test_declared_ce_traces_only_the_requested_economic_direction(
    *, stable: bool
) -> None:
    """Candidate and source reads do not trace the direction they discard."""
    original = _point_solver()
    grid = jnp.arange(1.0, 9.0)
    child = EGMCarry(
        endog_grid=grid,
        value=2.0 * grid,
        marginal_utility=jnp.full_like(grid, 7.0),
        taste_shock_scale=jnp.asarray(0.0),
    )
    parameter = (
        "certainty_equivalent__risk_aversion"
        if stable
        else "certainty_equivalent__theta"
    )
    plan = replace(
        original.pieces.continuation_plan,
        certainty_equivalent=PowerMean()
        if stable
        else QuasiArithmeticMean(
            transform=_quadratic_transform, inverse=_quadratic_inverse
        ),
        child_reads=MappingProxyType(
            {
                "done": _future_node_read(
                    original.pieces.continuation_plan.child_reads["done"]
                )
            }
        ),
    )
    reader = EconomicCEContinuation(
        power_mean_owner=replace(
            original,
            children=MappingProxyType({"done": child}),
            pieces=replace(original.pieces, continuation_plan=plan),
        ),
        plan=plan,
        pool={
            **original.pool,
            "liquid": jnp.asarray(0.0),
            parameter: jnp.asarray(1.0),
        },
        direction={"liquid": jnp.asarray(1.0), parameter: jnp.asarray(0.5)},
        children=MappingProxyType({"done": child}),
        dtype=grid.dtype,
        stochastic_node_width=2,
    )
    saving = jnp.asarray(1.0)
    both, candidate, source = (
        _equation_count(jax.make_jaxpr(read)(saving).jaxpr)
        for read in (reader, reader.savings, reader.source)
    )
    assert candidate < both
    assert source < both


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_declared_ce_transform_overflow_is_an_owned_source_failure() -> None:
    """Finite child values cannot turn transform overflow into an economic event."""
    original = _point_solver()
    child = original.children["done"]
    assert isinstance(child, EGMCarry)
    child = replace(
        child, value=jnp.full_like(child.value, jnp.finfo(child.value.dtype).max / 2.0)
    )
    declared = QuasiArithmeticMean(
        transform=_quadratic_transform, inverse=_quadratic_inverse
    )
    reader = EconomicCEContinuation(
        plan=replace(original.pieces.continuation_plan, certainty_equivalent=declared),
        pool={**original.pool, "certainty_equivalent__theta": jnp.asarray(1.0)},
        direction={},
        children=MappingProxyType({"done": child}),
        dtype=child.value.dtype,
    )
    result = reader(jnp.asarray(1.0))
    assert result.query_status is not None
    assert bool(result.query_status[0].coverage_missing)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("risk_direction", [False, True])
def test_stable_power_mean_literal_economic_and_unit_risk_directions(
    *,
    risk_direction: bool,
) -> None:
    """Unit-risk curvature and economic tangents differ from flat level slopes."""
    values = jnp.asarray([1.0, 4.0])
    factors = jnp.asarray([[0.5, 0.5]])
    coefficients, shifts, differentiated, derivative_shifts = _bank_coefficients(
        factors=factors, tangents=jnp.zeros_like(factors)
    )
    bank = PowerMeanNodeBank(
        values,
        jnp.asarray([3.0, 5.0]),
        jnp.ones_like(values),
        jnp.zeros_like(values) if risk_direction else jnp.asarray([2.0, -1.0]),
        coefficients,
        shifts,
        differentiated,
        derivative_shifts,
        jnp.ones(2, dtype=jnp.bool_),
        jnp.ones((), dtype=jnp.bool_),
    )
    result = reduce_power_mean_economic(
        declared=PowerMean(),
        params={"risk_aversion": jnp.asarray(1.0)},
        parameter_direction={"risk_aversion": jnp.asarray(float(risk_direction))},
        bank=bank,
    )
    np.testing.assert_array_almost_equal(
        result[:3],
        [2.0, 17.0 / 4.0, -(math.log(2.0) ** 2) if risk_direction else 19.0 / 4.0],
        decimal=DECIMAL_PRECISION,
    )
    assert bool(result[3])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("risk", [1.0, 2.0])
def test_stable_power_mean_derivative_only_event_uses_actual_unmasked_value(
    risk: float,
) -> None:
    """A null event outside the primal anchor still changes the normalized CE."""
    values = jnp.asarray([1.0, 4.0])
    factors = jnp.asarray([[1.0, 0.0]])
    coefficients, shifts, differentiated, derivative_shifts = _bank_coefficients(
        factors=factors, tangents=jnp.asarray([[0.0, 1.0]])
    )
    bank = PowerMeanNodeBank(
        values,
        jnp.zeros_like(values),
        jnp.zeros_like(values),
        jnp.zeros_like(values),
        coefficients,
        shifts,
        differentiated,
        derivative_shifts,
        jnp.ones(2, dtype=jnp.bool_),
        jnp.ones((), dtype=jnp.bool_),
    )
    result = reduce_power_mean_economic(
        declared=PowerMean(),
        params={"risk_aversion": jnp.asarray(risk)},
        parameter_direction={"risk_aversion": jnp.asarray(0.0)},
        bank=bank,
    )
    np.testing.assert_array_almost_equal(
        result[:3],
        [1.0, 0.0, math.log(4.0) if risk == 1.0 else 0.75],
        decimal=DECIMAL_PRECISION,
    )
    assert bool(result[3])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_genuine_case_u_candidate_and_envelope_share_stable_power_mean() -> None:
    """The real point's original Euler/FUES route consumes the stable CE sink."""
    original = _point_solver()
    plan = original.pieces.continuation_plan
    first = original.children["done"]
    assert isinstance(first, EGMCarry)
    first = replace(first, value=first.value + 2.0)
    other = replace(first, value=first.value + 1.0)
    read = replace(plan.child_reads["done"], euler_state_func=_dependent_next)
    saving = 0.5
    low = 2.0 + math.log(saving)
    high = low + 1.0
    certainty = math.sqrt(low * high)
    economic = certainty * (1.0 / low + 1.0 / high) / (2.0 * saving)
    alpha = saving * economic / (economic - 1.0)
    solver = replace(
        original,
        pool={**original.pool, "certainty_equivalent__risk_aversion": jnp.asarray(1.0)},
        children=MappingProxyType({"done": first, "other": other}),
        pieces=replace(
            original.pieces,
            continuation_plan=replace(
                plan,
                stateful_targets=("done", "other"),
                child_reads=MappingProxyType({"done": read, "other": read}),
                compute_regime_transition_probs=_two_target_probs,
                certainty_equivalent=PowerMean(),
            ),
        ),
    )
    result = solver.node(
        liquid=jnp.asarray(0.0),
        physical={"e": jnp.int32(0), "eta": jnp.asarray(alpha - 1.0)},
        physical_tangent={"eta": jnp.asarray(0.0)},
    ).output
    np.testing.assert_array_almost_equal(
        [result.savings, result.marginal],
        [saving, 2.0 * economic],
        decimal=DECIMAL_PRECISION,
    )
    assert bool(result.valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_stable_power_mean_invalid_derivative_only_node_is_not_discarded() -> None:
    """A zero primal weight cannot hide source arithmetic needed by its tangent."""
    factors = jnp.asarray([[1.0, 0.0]])
    coefficients, shifts, differentiated, derivative_shifts = _bank_coefficients(
        factors=factors, tangents=jnp.asarray([[0.0, 1.0]])
    )
    values = jnp.asarray([1.0, jnp.nan])
    bank = PowerMeanNodeBank(
        values,
        jnp.zeros_like(values),
        jnp.zeros_like(values),
        jnp.zeros_like(values),
        coefficients,
        shifts,
        differentiated,
        derivative_shifts,
        jnp.ones(2, dtype=jnp.bool_),
        jnp.ones((), dtype=jnp.bool_),
    )
    result = reduce_power_mean_economic(
        declared=PowerMean(),
        params={"risk_aversion": jnp.asarray(1.0)},
        parameter_direction={"risk_aversion": jnp.asarray(0.0)},
        bank=bank,
    )
    assert not bool(result[3])


@pytest.mark.coverage(backends=("cpu",), precisions="64")
@pytest.mark.skipif(not X64_ENABLED, reason="1e100 and 1e-310 are float64 operands")
def test_stable_power_mean_rare_variance_keeps_final_ce_scale() -> None:
    """A subnormal event still has an ordinary scaled economic contribution."""
    values = jnp.asarray([1e100, 4e100])
    factors = jnp.asarray([[1.0, 1e-310]])
    coefficients, shifts, differentiated, derivative_shifts = _bank_coefficients(
        factors=factors, tangents=jnp.zeros_like(factors)
    )
    bank = PowerMeanNodeBank(
        values,
        jnp.zeros_like(values),
        jnp.zeros_like(values),
        jnp.zeros_like(values),
        coefficients,
        shifts,
        differentiated,
        derivative_shifts,
        jnp.ones(2, dtype=jnp.bool_),
        jnp.ones((), dtype=jnp.bool_),
    )
    result = reduce_power_mean_economic(
        declared=PowerMean(),
        params={"risk_aversion": jnp.asarray(1.0)},
        parameter_direction={"risk_aversion": jnp.asarray(1.0)},
        bank=bank,
    )
    # This discriminator tests the variance scaling of the represented log
    # coordinates, separately from any backend logarithm accuracy claim.
    logs = np.asarray(jnp.log(values))
    expected = 0.5 * (logs[1] - logs[0]) ** 2
    observed = -float(result[2]) / (float(result[0]) * 1e-310)
    np.testing.assert_array_almost_equal(observed, expected, decimal=DECIMAL_PRECISION)
    assert bool(result[3])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_stable_power_mean_unproven_wide_cancellation_refuses() -> None:
    """The unfinished multi-scale class must not publish a false exact zero."""
    reduced = _sum_signed(
        terms=_SignedLog(jnp.asarray([0.0, 0.0, -200.0]), jnp.asarray([1.0, -1.0, 1.0]))
    )
    assert bool(jnp.isnan(_materialize(value=reduced)))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_stable_power_mean_nested_same_sign_then_opposition_refuses() -> None:
    """A same-sign inner reduction cannot hide its rare residual's scale."""
    inner = _sum_signed(
        terms=_SignedLog(jnp.asarray([0.0, -80.0]), jnp.asarray([1.0, 1.0]))
    )
    outer = _sum_signed(
        terms=_SignedLog(
            jnp.stack((inner.logabs, jnp.asarray(0.0))),
            jnp.stack((inner.sign, jnp.asarray(-1.0))),
        )
    )
    assert bool(jnp.isnan(_materialize(value=outer)))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_stable_power_mean_opposing_economic_channel_with_rare_variance_refuses() -> (
    None
):
    """Original economic bank joins parameter variance and opposing coordinates."""
    values = jnp.exp(jnp.asarray([0.0, 1.0, 2.0]))
    factors = jnp.asarray([[0.5, 0.5, math.exp(-80.0)]])
    coefficients, shifts, differentiated, derivative_shifts = _bank_coefficients(
        factors=factors, tangents=jnp.zeros_like(factors)
    )
    bank = PowerMeanNodeBank(
        values,
        values,
        jnp.zeros_like(values),
        jnp.full_like(values, -0.125),
        coefficients,
        shifts,
        differentiated,
        derivative_shifts,
        jnp.ones(3, dtype=jnp.bool_),
        jnp.ones((), dtype=jnp.bool_),
    )
    # For the dominant equal-weight [1,e] lottery, the unit-p exponent channel
    # is +C/8 while the coordinate channel is -C/8. The rare event contributes
    # additional variance. A rounded valid zero is not an admissible answer.
    result = reduce_power_mean_economic(
        declared=PowerMean(),
        params={"risk_aversion": jnp.asarray(1.0)},
        parameter_direction={"risk_aversion": jnp.asarray(-1.0)},
        bank=bank,
    )
    assert not bool(result[3])
