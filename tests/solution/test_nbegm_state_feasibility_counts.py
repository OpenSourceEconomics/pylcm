"""NB feasibility and economic infinity counts describe physical states."""

import math
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing_extensions import TypedDict

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import ContinuationPlan
from _lcm.solution.feasibility import BoundedResult, BoundInputStatus
from _lcm.solution.nbegm import (
    _build_nbegm_discrete_core,
    _NBEGMDiscreteSpec,
    _NBEGMFloor,
    _NBEGMOwnedSingleAxisCore,
)
from _lcm.typing import EconFunctionArg
from _lcm.utils.functools import allow_args
from lcm.typing import BoolND, Float1D, FloatND, RegimeName, ScalarFloat
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON


class _SourceInputs(TypedDict, closed=True):
    liquid: Float1D
    next_liquid_grid: Float1D
    next_value: Float1D
    next_marginal: Float1D
    next_liquid: Float1D
    marginal_return: Float1D
    effective_savings_grid: Float1D
    boundary_savings_targets: Float1D
    boundary_next_liquid: Float1D
    source_bound_status: BoundInputStatus
    permitted_job: FloatND
    koopmans_aggregator__discount_factor: FloatND


def _source_floor(*, floor__level: ScalarFloat) -> ScalarFloat:
    return floor__level


def _floor_and_nodes() -> tuple[ScalarFloat, Float1D, BoundInputStatus]:
    """Bind the source's qualified floor to the existing shared offsets."""
    floor = _NBEGMFloor(
        lower=_source_floor,
        lower_inputs=("floor__level",),
        offsets=jnp.asarray([0.0, 0.5, 1.0]),
    )
    actual_floor, nodes, status = floor.evaluate_floor(
        pool={"floor__level": jnp.asarray(0.5)}
    )
    np.testing.assert_array_equal(actual_floor, 0.5)
    np.testing.assert_array_equal(nodes, [0.5, 1.0, 1.5])
    np.testing.assert_array_equal(jnp.stack(status), [False, False])
    return actual_floor, nodes, status


def _resources(*, liquid: FloatND, job: FloatND) -> FloatND:
    return liquid + 0.0 * job


def _utility(*, consumption: FloatND) -> FloatND:
    return jnp.log(consumption)


def _inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _eligible_job(*, job: FloatND, permitted_job: FloatND) -> BoolND:
    return (permitted_job < 0) | (job == permitted_job)


def _empty_probabilities(
    *args: EconFunctionArg, **params: EconFunctionArg
) -> MappingProxyType[RegimeName, FloatND]:
    del args, params
    return MappingProxyType({})


def _no_query_plan() -> ContinuationPlan:
    """Declare no child query slots for the isolated current-state owner."""
    return ContinuationPlan(
        stateful_targets=(),
        scalar_targets=(),
        child_reads=MappingProxyType({}),
        compute_regime_transition_probs=_empty_probabilities,
        post_decision_name="savings",
    )


def _source_inputs(*, permitted_job: int) -> _SourceInputs:
    _floor, savings, status = _floor_and_nodes()
    landings = 3.0 * savings + 4.0
    return {
        "liquid": jnp.asarray([0.75, 1.5, 3.0]),
        "next_liquid_grid": landings,
        "next_value": jnp.log(landings),
        "next_marginal": 1.0 / landings,
        "next_liquid": landings,
        "marginal_return": jnp.full_like(savings, 3.0),
        "effective_savings_grid": savings,
        "boundary_savings_targets": jnp.asarray([], dtype=savings.dtype),
        "boundary_next_liquid": jnp.asarray([], dtype=savings.dtype),
        "source_bound_status": status,
        "permitted_job": jnp.asarray(permitted_job),
        "koopmans_aggregator__discount_factor": jnp.asarray(1.0),
    }


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize(
    ("permitted_job", "expected_empty"),
    [(-1, 0), (0, 0), (1, 0), (2, 3)],
    ids=["both-feasible", "only-first", "only-second", "neither-feasible"],
)
def test_single_axis_counts_jointly_empty_states(
    *, permitted_job: int, expected_empty: int
) -> None:
    """One admissible job suffices; two rejected jobs leave three empty states."""
    inputs = _source_inputs(permitted_job=permitted_job)
    spec = _NBEGMDiscreteSpec(
        coh_of_liquid_dag=_resources,
        coh_param_names=(),
        coh_action_names=("job",),
        liquid_state_name="liquid",
        discrete_actions=(("job", (0, 1)),),
    )
    inner = _build_nbegm_discrete_core(
        savings_grid=inputs["effective_savings_grid"],
        functions=MappingProxyType(
            {
                "utility": allow_args(_utility),
                "inverse_marginal_utility": allow_args(_inverse),
            }
        ),
        consumption_action="consumption",
        discrete_spec=spec,
        taste_shock_scale=0.0,
    )
    baseline = inner(**inputs)
    assert isinstance(baseline, BoundedResult)
    baseline_value, baseline_carry = baseline.output
    np.testing.assert_array_almost_equal(
        baseline_value[:2],
        [math.log(11.0 / 8.0), math.log(5.5)],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        baseline_carry.marginal_utility[:2], [4.0, 1.0], decimal=DECIMAL_PRECISION
    )
    core = _NBEGMOwnedSingleAxisCore(
        inner=inner,
        continuation_plan=_no_query_plan(),
        liquid_name="liquid",
        resources=_resources,
        predicates=(_eligible_job,),
        action_names=("job",),
        action_bindings=spec.branch_bindings,
        owns_query_rows=False,
    )
    actual = core(**inputs)
    value, carry, counts, invalid = actual.output
    np.testing.assert_array_equal(jnp.stack(counts), [expected_empty, 0])
    np.testing.assert_array_equal(invalid, desired=False)
    if expected_empty:
        np.testing.assert_array_equal(value, [-np.inf, -np.inf, -np.inf])
    else:
        np.testing.assert_array_equal(value, baseline_value)
    for observed, expected in zip(
        jax.tree.leaves(carry), jax.tree.leaves(baseline_carry), strict=True
    ):
        np.testing.assert_array_equal(observed, expected)
    assert actual.query_status is None
    assert len(actual.bound_status) == 1
    np.testing.assert_array_equal(jnp.stack(actual.bound_status[0]), [False, False])


@dataclass(frozen=True, kw_only=True)
class _AnalyticEconomicMinusInfinityRawCore:
    """A constant economic -inf utility with zero derivative on a legal budget.

    This analytic two-channel producer isolates the current-state count owner.
    It is not a native NB envelope producer or a claim of NB infinity admission.
    """

    def __call__(
        self,
        *,
        liquid: Float1D,
        source_bound_status: BoundInputStatus,
        **inputs: FloatND,
    ) -> BoundedResult[tuple[FloatND, EGMCarry]]:
        del inputs
        value = jnp.full_like(liquid, -jnp.inf)
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=jnp.zeros_like(liquid),
            taste_shock_scale=jnp.asarray(0.0, dtype=liquid.dtype),
        )
        return BoundedResult(output=(value, carry), bound_status=(source_bound_status,))


def test_economic_infinity_counts_states_once_at_the_wrapper_unit() -> None:
    """Two feasible jobs with value -inf contribute three economic states."""
    inputs = _source_inputs(permitted_job=-1)
    core = _NBEGMOwnedSingleAxisCore(
        inner=_AnalyticEconomicMinusInfinityRawCore(),
        continuation_plan=_no_query_plan(),
        liquid_name="liquid",
        resources=_resources,
        predicates=(_eligible_job,),
        action_names=("job",),
        action_bindings=(MappingProxyType({"job": 0}), MappingProxyType({"job": 1})),
        owns_query_rows=False,
    )
    actual = core(**inputs)
    value, carry, counts, invalid = actual.output
    np.testing.assert_array_equal(value, [-np.inf, -np.inf, -np.inf])
    np.testing.assert_array_equal(carry.endog_grid, [0.75, 1.5, 3.0])
    np.testing.assert_array_equal(carry.marginal_utility, [0.0, 0.0, 0.0])
    assert carry.query_data is None
    np.testing.assert_array_equal(jnp.stack(counts), [0, 3])
    np.testing.assert_array_equal(invalid, desired=False)
    assert actual.query_status is None
    assert len(actual.bound_status) == 1
    np.testing.assert_array_equal(jnp.stack(actual.bound_status[0]), [False, False])


@dataclass(frozen=True, kw_only=True)
class _AnalyticMixedConditionalRawCore:
    """Two action rows: log(liquid) and economic -inf, with exact derivatives.

    This analytic producer isolates the wrapper's conditional publication branch.
    It supplies no installed child-query authority or native NB admission witness.
    """

    def __call__(
        self,
        *,
        liquid: Float1D,
        source_bound_status: BoundInputStatus,
        **inputs: FloatND,
    ) -> BoundedResult[tuple[FloatND, EGMCarry]]:
        del inputs
        finite = jnp.log(liquid)
        rows = jnp.stack((finite, jnp.full_like(liquid, -jnp.inf)))
        carry = EGMCarry(
            endog_grid=jnp.broadcast_to(liquid, rows.shape),
            value=rows,
            marginal_utility=jnp.stack((1.0 / liquid, jnp.zeros_like(liquid))),
            taste_shock_scale=jnp.asarray(0.0, dtype=liquid.dtype),
        )
        return BoundedResult(
            output=(finite, carry), bound_status=(source_bound_status,)
        )


def test_finite_winner_excludes_infinite_feasible_loser_at_wrapper_unit() -> None:
    """A finite winning job leaves zero economic-infinite states."""
    inputs = _source_inputs(permitted_job=-1)
    spec = _NBEGMDiscreteSpec(
        coh_of_liquid_dag=_resources,
        coh_param_names=(),
        coh_action_names=("job",),
        liquid_state_name="liquid",
        discrete_actions=(("job", (0, 1)),),
    )
    assert tuple(dict(binding) for binding in spec.branch_bindings) == (
        {"job": 0},
        {"job": 1},
    )
    inner = _AnalyticMixedConditionalRawCore()
    raw = inner(**inputs)
    raw_value, raw_carry = raw.output
    expected = [math.log(0.75), math.log(1.5), math.log(3.0)]
    np.testing.assert_array_almost_equal(raw_value, expected, decimal=DECIMAL_PRECISION)
    np.testing.assert_array_equal(
        raw_carry.endog_grid, [[0.75, 1.5, 3.0], [0.75, 1.5, 3.0]]
    )
    np.testing.assert_array_almost_equal(
        raw_carry.value[0], expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(raw_carry.value[1], [-np.inf, -np.inf, -np.inf])
    np.testing.assert_array_almost_equal(
        raw_carry.marginal_utility[0],
        [4.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(raw_carry.marginal_utility[1], [0.0, 0.0, 0.0])
    assert raw_carry.query_data is None
    np.testing.assert_array_equal(
        _resources(liquid=inputs["liquid"], job=jnp.asarray(0)),
        inputs["liquid"],
    )
    np.testing.assert_array_equal(
        _resources(liquid=inputs["liquid"], job=jnp.asarray(1)),
        inputs["liquid"],
    )
    np.testing.assert_array_equal(
        inputs["liquid"] > inputs["effective_savings_grid"][0], [True, True, True]
    )
    for binding in spec.branch_bindings:
        np.testing.assert_array_equal(
            _eligible_job(
                job=jnp.asarray(binding["job"]),
                permitted_job=inputs["permitted_job"],
            ),
            desired=True,
        )
    core = _NBEGMOwnedSingleAxisCore(
        inner=inner,
        continuation_plan=_no_query_plan(),
        liquid_name="liquid",
        resources=_resources,
        predicates=(_eligible_job,),
        action_names=("job",),
        action_bindings=spec.branch_bindings,
        owns_query_rows=True,
    )
    actual = core(**inputs)
    value, carry, counts, invalid = actual.output
    np.testing.assert_array_almost_equal(value, expected, decimal=DECIMAL_PRECISION)
    np.testing.assert_array_equal(invalid, desired=False)
    for observed, reference in zip(
        jax.tree.leaves(carry), jax.tree.leaves(raw_carry), strict=True
    ):
        np.testing.assert_array_equal(observed, reference)
    assert actual.query_status is None
    assert len(actual.bound_status) == 1
    np.testing.assert_array_equal(jnp.stack(actual.bound_status[0]), [False, False])
    np.testing.assert_array_equal(jnp.stack(counts), [0, 0])
