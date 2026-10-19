"""Actual DC target domains govern associated interpolated row candidates."""

import importlib
import importlib.util
from dataclasses import replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import EGMQuerySpec, EGMRowTopologySpec
from _lcm.egm.dcegm_query import DCEGMQuery, DCEGMQueryResult
from _lcm.egm.dcegm_query_model import DCEGMQueryModel
from _lcm.egm.query_data import EGMQueryData
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.typing import EconFunction, EconFunctionsMapping
from lcm.typing import BoolND, FloatND, ScalarFloat
from tests.conftest import DECIMAL_PRECISION


def _lower(*, z: FloatND) -> FloatND:
    return z * (1.0 - z)


def _singular_lower(*, z: FloatND) -> FloatND:
    return 1.0 / (z - 0.5)


def _resources(*, wealth: FloatND) -> FloatND:
    return 2.0 * wealth + 1.0


def _utility(*, consumption: FloatND) -> FloatND:
    return jnp.log(consumption)


def _aggregator(*, utility: FloatND, CE: FloatND, discount_factor: FloatND) -> FloatND:
    return utility + discount_factor * CE


def _eligible(*, z: FloatND, eligibility: BoolND) -> BoolND:
    return (z < 1.0) & eligibility


def _adapter(*, singular: bool = False) -> DCEGMQuery:
    """Bind the independently supplied evaluator pool and exact row topology."""
    module_name = "_lcm.egm.dcegm_query"
    assert importlib.util.find_spec(module_name) is not None, (
        "A model-bound DC target query adapter must be installed."
    )
    module = importlib.import_module(module_name)
    lower = _singular_lower if singular else _lower
    bound = PostDecisionBoundSpec(
        post_decision="savings",
        liquid_state="wealth",
        action="consumption",
        resources_target="resources",
        kind="dag",
        literal_value=None,
        lower_target="lower",
        lower_inputs=("z",),
        dependency_identity=(((), lower),),
        declaration_names=("borrowing",),
    )
    authority = EGMQuerySpec(
        bound=bound,
        state_names=("z",),
        action_names=(),
        coordinate_kind="resources",
        marginal_convention="ordinary_linear",
        corner_coordinate_kind="compact_interval",
        row_capacity=2,
        interval_ids=(0,),
        interval_side_ids=(0,),
        evaluator_targets=(
            ("resources", "resources"),
            ("utility", "utility"),
            ("predicate:eligible", "eligible"),
        ),
        evaluator_inputs=(),
        evaluator_identity=(),
        target_periods=(1,),
        row_topology=EGMRowTopologySpec(kind="exact_monotone_run", owner_capacity=2),
    )
    functions = cast(
        "EconFunctionsMapping",
        MappingProxyType(
            {
                "lower": lower,
                "resources": _resources,
                "utility": _utility,
                "eligible": _eligible,
            }
        ),
    )
    aggregator = cast("EconFunction", _aggregator)
    model = DCEGMQueryModel(
        authority=authority,
        functions=functions,
        koopmans_aggregator=aggregator,
        offsets=jnp.array([0.0, 1.0]),
    )
    return module.build_dcegm_query(
        model=model, functions=functions, koopmans_aggregator=aggregator
    )


def _data() -> EGMQueryData:
    return EGMQueryData(
        savings=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        node_kind=jnp.ones((2, 2), dtype=jnp.int32),
        link_valid=jnp.ones((2, 1), dtype=jnp.bool_),
        owner_id=jnp.zeros((2, 2), dtype=jnp.int32),
        side_id=jnp.zeros((2, 2), dtype=jnp.int32),
        floor_used=jnp.array([0.0, 0.0]),
        corner_continuation=jnp.zeros((2, 1)),
        corner_kind=jnp.ones((2, 1), dtype=jnp.int32),
        corner_interval_id=jnp.zeros((2, 1), dtype=jnp.int32),
        corner_side_id=jnp.zeros((2, 1), dtype=jnp.int32),
    )


def _query(
    *,
    data: EGMQueryData,
    value: FloatND | None = None,
    marginal: FloatND | None = None,
    z: float = 0.5,
    resource_gradient: float = 10.0,
    singular: bool = False,
    eligibility: bool = True,
) -> DCEGMQueryResult:
    return _adapter(singular=singular)(
        grid=jnp.array([[0.5, 1.5], [0.5, 1.5]]),
        value=jnp.full((2, 2), 2.0) if value is None else value,
        marginal=jnp.ones((2, 2)) if marginal is None else marginal,
        data=data,
        row_queries=jnp.array([1.0, 1.0]),
        row_gradients=jnp.array([10.0, 10.0]),
        passive_values=(jnp.asarray(z),),
        passive_grids=(jnp.array([0.0, 1.0]),),
        pool={
            "wealth": jnp.asarray(0.0),
            "z": jnp.asarray(z),
            "discount_factor": jnp.asarray(1.0),
            "eligibility": jnp.asarray(eligibility),
        },
        resource_gradient=jnp.asarray(resource_gradient),
    )


@pytest.mark.parametrize("exact_zero", [False, True])
def test_two_axis_stencil_activity_survives_numeric_product_underflow(
    *, exact_zero: bool
) -> None:
    """Only an exact-zero factor neutralizes a corrupt two-axis stencil record."""
    adapter = _adapter()
    adapter = replace(
        adapter,
        model=replace(
            adapter.model,
            authority=replace(adapter.model.authority, state_names=("z", "w")),
        ),
    )
    epsilon = jnp.asarray(
        np.sqrt(np.finfo(np.asarray(jnp.asarray(1.0)).dtype).tiny) / 2.0
    )
    z = jnp.asarray(0.0) if exact_zero else epsilon
    data = EGMQueryData(
        savings=jnp.zeros((2, 2, 2)).at[1, 1].set(jnp.nan),
        node_kind=jnp.ones((2, 2, 2), dtype=jnp.int32),
        link_valid=jnp.ones((2, 2, 1), dtype=jnp.bool_),
        owner_id=jnp.zeros((2, 2, 2), dtype=jnp.int32),
        side_id=jnp.zeros((2, 2, 2), dtype=jnp.int32),
        floor_used=jnp.zeros((2, 2)),
        corner_continuation=jnp.zeros((2, 2, 1)),
        corner_kind=jnp.ones((2, 2, 1), dtype=jnp.int32),
        corner_interval_id=jnp.zeros((2, 2, 1), dtype=jnp.int32),
        corner_side_id=jnp.zeros((2, 2, 1), dtype=jnp.int32),
    )
    result = adapter(
        grid=jnp.broadcast_to(jnp.array([0.5, 1.5]), (2, 2, 2)),
        value=jnp.full((2, 2, 2), 2.0),
        marginal=jnp.ones((2, 2, 2)),
        data=data,
        row_queries=jnp.ones((2, 2)),
        row_gradients=jnp.full((2, 2), 10.0),
        passive_values=(z, epsilon),
        passive_grids=(jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])),
        pool={
            "wealth": jnp.asarray(0.0),
            "z": z,
            "w": epsilon,
            "discount_factor": jnp.asarray(1.0),
            "eligibility": jnp.asarray(a=True),
        },
        resource_gradient=jnp.asarray(10.0),
    )
    assert bool(result.coverage_missing) is (not exact_zero)
    np.testing.assert_allclose(
        [result.value, result.marginal],
        [2.0, 10.0] if exact_zero else [0.0, 10.0],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )


@pytest.mark.parametrize("resource_gradient", [1.0, 10.0])
def test_actual_nonlinear_floor_recovers_independent_corner(
    *, resource_gradient: float
) -> None:
    """An actual interior floor discards illegal saving and uses its own budget."""
    result = _query(data=_data(), resource_gradient=resource_gradient)
    np.testing.assert_allclose(
        [result.value, result.marginal, result.savings],
        [jnp.log(0.75), resource_gradient * 4.0 / 3.0, 0.25],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    assert not bool(result.coverage_missing)
    assert not bool(result.bound_status.nonfinite_floor)


def test_legal_owned_candidate_preserves_its_value_marginal_and_saving() -> None:
    """The selected associated tuple survives actual legality and corner ranking."""
    result = _query(data=replace(_data(), savings=jnp.full((2, 2), 0.5)))
    np.testing.assert_allclose(
        [result.value, result.marginal, result.savings], [2.0, 10.0, 0.5]
    )
    assert not bool(result.coverage_missing)


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_support_recovers_but_corrupt_active_data_remains_fatal(
    *, corrupt: bool
) -> None:
    """A finite corner recovers absent support without hiding corrupt active input."""
    data = replace(_data(), link_valid=jnp.zeros((2, 1), dtype=jnp.bool_))
    if corrupt:
        data = replace(data, savings=jnp.array([[0.0, jnp.nan], [0.0, 0.0]]))
    result = _query(data=data)
    np.testing.assert_allclose(
        [result.value, result.savings],
        [jnp.log(0.75), 0.25],
        rtol=10 ** (-DECIMAL_PRECISION),
    )
    assert bool(result.coverage_missing) is corrupt


def test_excluded_actual_predicate_has_no_coverage_requirement() -> None:
    """An excluded target branch carries no economic value or missing coverage."""
    data = replace(
        _data(),
        node_kind=jnp.zeros((2, 2), dtype=jnp.int32),
        corner_kind=jnp.zeros((2, 1), dtype=jnp.int32),
    )
    result = _query(data=data, z=1.25)
    assert jnp.isneginf(result.value)
    assert float(result.marginal) == 0.0
    assert not bool(result.coverage_missing)


@pytest.mark.parametrize("eligibility", [False, True])
def test_actual_floor_arithmetic_survives_target_predicate_exclusion(
    *, eligibility: bool
) -> None:
    """Actual bound arithmetic remains mandatory on excluded target branches."""
    result = _query(data=_data(), singular=True, eligibility=eligibility)
    assert bool(result.bound_status.nonfinite_floor)


@pytest.mark.parametrize("z", [0.0, 0.5])
def test_exact_zero_passive_neighbor_has_no_numeric_authority(*, z: float) -> None:
    """Corrupt source tuples matter exactly when their passive coefficient is active."""
    data = replace(_data(), savings=jnp.array([[0.0, 0.0], [jnp.nan, jnp.nan]]))
    result = _query(data=data, z=z)
    assert bool(result.coverage_missing) is (z != 0.0)
    expected = [2.0, 10.0, 0.0] if z == 0.0 else [jnp.log(0.75), 40.0 / 3.0, 0.25]
    np.testing.assert_allclose(
        [result.value, result.marginal, result.savings],
        np.asarray(expected),
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_signed_passive_stencil_preserves_association_and_active_numeric_status(
    *, corrupt: bool
) -> None:
    """Negative coefficients remain active without clipping or partial normalization."""
    data = _data()
    if corrupt:
        data = replace(data, savings=jnp.array([[0.0, 0.0], [jnp.nan, jnp.nan]]))
    result = _query(data=data, z=-0.5)
    expected = [jnp.log(1.75), 40.0 / 7.0, -0.75] if corrupt else [2.0, 10.0, 0.0]
    np.testing.assert_allclose(
        [result.value, result.marginal, result.savings],
        np.asarray(expected),
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    assert bool(result.coverage_missing) is corrupt


def test_negative_economic_passive_term_recovers_as_missing_support() -> None:
    """A negative economic minus-infinity term cannot create plus infinity."""
    data = replace(_data(), node_kind=jnp.array([[1, 1], [2, 2]], dtype=jnp.int32))
    result = _query(
        data=data,
        z=-0.5,
        value=jnp.array([[2.0, 2.0], [-jnp.inf, -jnp.inf]]),
        marginal=jnp.array([[1.0, 1.0], [0.0, 0.0]]),
    )
    np.testing.assert_allclose(
        [result.value, result.marginal, result.savings],
        [jnp.log(1.75), 40.0 / 7.0, -0.75],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    assert not bool(result.coverage_missing)


def test_complete_query_result_survives_jitted_mapping() -> None:
    """Mapped target queries preserve associated numbers and bounded status leaves."""
    adapter = _adapter()
    data = _data()

    def query(wealth: ScalarFloat) -> DCEGMQueryResult:
        resources = 2.0 * wealth + 1.0
        return adapter(
            grid=jnp.array([[0.5, 1.5], [0.5, 1.5]]),
            value=jnp.full((2, 2), 2.0),
            marginal=jnp.ones((2, 2)),
            data=data,
            row_queries=jnp.full((2,), resources),
            row_gradients=jnp.full((2,), 10.0),
            passive_values=(jnp.asarray(0.5),),
            passive_grids=(jnp.array([0.0, 1.0]),),
            pool={
                "wealth": wealth,
                "z": jnp.asarray(0.5),
                "discount_factor": jnp.asarray(1.0),
                "eligibility": jnp.asarray(a=True),
            },
            resource_gradient=jnp.asarray(10.0),
        )

    result = jax.jit(jax.vmap(query))(jnp.array([0.0, 0.25]))
    np.testing.assert_allclose(
        result.value,
        jnp.log(jnp.array([0.75, 1.25])),
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_allclose(
        result.marginal,
        [40.0 / 3.0, 8.0],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_array_equal(result.savings, [0.25, 0.25])
    np.testing.assert_array_equal(result.coverage_missing, [False, False])
    np.testing.assert_array_equal(result.bound_status.nonfinite_floor, [False, False])
    np.testing.assert_array_equal(
        result.bound_status.invalid_shifted_grid, [False, False]
    )
