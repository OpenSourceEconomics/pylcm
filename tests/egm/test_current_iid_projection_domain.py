"""Exact Case-R samples do not grant complete selector-topology admission."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import ProjectedContinuationSpec
from _lcm.egm.continuation import _validate_projected_child
from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluator
from _lcm.egm.current_iid_native import BoundResourcesMap
from _lcm.egm.current_iid_projection import _assess_cells, _read_projected
from _lcm.egm.current_iid_publication import (
    FiniteResourcesDemands,
    ProjectedResourcesReference,
    _assemble_resources_samples,
    publish_finite_resources_demands,
    publish_projected_resources,
)
from _lcm.egm.current_iid_topology import pullback_native_partition
from _lcm.egm.projected_child import ProjectedChildReader
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedChildAuthority,
    ProjectedEGMCarry,
    ProjectedRowRead,
)
from _lcm.egm.query_data import QueryNodeKind
from _lcm.egm.step_core import OwnedResourcesRow
from _lcm.egm.upper_envelope.segment_envelope import OwnedEnvelope
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.scheduler import BufferRegistry
from _lcm.params.mapping_leaf import MappingLeaf
from _lcm.params.sequence_leaf import SequenceLeaf
from _lcm.solution.continuation_reads import continuation_leaf_reads
from _lcm.solution.current_iid_case_r_build import _resources_parameter_argument
from _lcm.solution.feasibility import validate_query_input_status
from _lcm.solution.solve_inputs import (
    SolveInputMappings,
    register_rolled_inputs,
    substitute_artifact,
)
from _lcm.typing import ValueTemplate
from lcm.exceptions import InvalidValueFunctionError, UnsupportedOperationError
from lcm.shocks import ShockName
from lcm.typing import Float1D, FloatND, ScalarFloat
from tests.conftest import DECIMAL_PRECISION


def _resources(*, liquid: ScalarFloat, eps: ScalarFloat) -> ScalarFloat:
    return liquid + eps


def _scaled_resources(*, liquid: ScalarFloat, eps: ScalarFloat) -> ScalarFloat:
    return 4.0 * liquid + eps


def _corner_utility(consumption: ScalarFloat) -> ScalarFloat:
    # Corner consumption is R - 0.25, so the bump lifts the corner above the flat
    # row only near R = 1.37, between the 1.25/1.5/1.75 refinement probes.
    return 0.5 + jnp.exp(-jnp.square((consumption - 1.12) / 0.01))


def _unit_utility(consumption: ScalarFloat) -> ScalarFloat:
    return jnp.ones_like(consumption)


def _identity_query(savings: ScalarFloat) -> ScalarFloat:
    return savings


def _reference(*, liquid_grid: Float1D, capacity: int) -> ProjectedResourcesReference:
    """Bind a flat owned candidate and a narrow smooth competing corner.

    The row saves 0.5, strictly above the 0.25 borrowing limit, so it is an
    interior action distinct from the literal corner it competes with.
    """
    grid = jnp.asarray([1.0, 2.0], dtype=liquid_grid.dtype)
    row = OwnedResourcesRow(
        envelope=OwnedEnvelope(
            grid=grid,
            policy=grid - 0.5,
            value=jnp.ones_like(grid),
            n_kept=jnp.asarray(2, dtype=jnp.int32),
            savings=jnp.full_like(grid, 0.5),
            owner_id=jnp.zeros((2,), dtype=jnp.int32),
            side_id=jnp.zeros((2,), dtype=jnp.int32),
            link_valid=jnp.ones((1,), dtype=jnp.bool_),
            source_left_index=jnp.asarray([0, 0], dtype=jnp.int32),
            source_right_index=jnp.asarray([1, 1], dtype=jnp.int32),
        ),
        marginal=jnp.zeros_like(grid),
        node_kind=jnp.full((2,), QueryNodeKind.FINITE, dtype=jnp.int32),
        borrowing_limit=jnp.asarray(0.25, dtype=grid.dtype),
        upper_savings=jnp.asarray(2.0, dtype=grid.dtype),
        expected_floor_value=jnp.asarray(0.0, dtype=grid.dtype),
        discount_factor=jnp.asarray(1.0, dtype=grid.dtype),
        discounted_floor_value=jnp.asarray(0.0, dtype=grid.dtype),
        resources_domain=grid,
        construction_invalid=jnp.asarray(False),  # noqa: FBT003
        floor_status=None,
        child_status=None,
        continuation_read_counts=None,
    )
    resources = BoundResourcesMap(
        function=_resources,
        liquid_state="liquid",
        pool=MappingProxyType({}),
        names=("liquid", "eps"),
    )
    physical = MappingProxyType({"eps": jnp.zeros((1,), dtype=grid.dtype)})
    evaluator = CurrentResourcesEvaluator(
        row=row,
        resources=resources,
        resources_slope=jnp.asarray(1.0, dtype=grid.dtype),
        utility_of_action=_corner_utility,
        owner_capacity=1,
    )
    partition = pullback_native_partition(
        row=row,
        resources=resources,
        physical=physical,
        live=jnp.ones((1,), dtype=jnp.bool_),
        liquid_grid=liquid_grid,
        demanded_queries=jnp.empty((0,), dtype=liquid_grid.dtype),
        capacity=capacity,
    )
    return ProjectedResourcesReference(
        evaluator=evaluator,
        physical=physical,
        factors=jnp.ones((1, 1), dtype=grid.dtype),
        law_valid=jnp.asarray(True),  # noqa: FBT003
        partition=partition,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_irregular_declared_node_keeps_exact_integrated_channels() -> None:
    """An irregular non-midpoint survives union and exact unsided sample reads."""
    liquid_grid = jnp.asarray([1.0, 1.37, 2.0])
    reference = _reference(liquid_grid=liquid_grid, capacity=8)
    samples = _assemble_resources_samples(reference=reference, capacity=8)
    retained = np.asarray(samples.grid)[: int(samples.valid_length)]
    assert np.isfinite(retained).all()
    assert np.count_nonzero(retained == np.asarray(liquid_grid[1])) == 1
    assert bool(samples.valid)
    direct = reference(liquid_grid[1])
    exact = _read_projected(carry=samples, query=liquid_grid[1])
    for actual, expected in zip(exact, direct, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert bool(direct.valid)
    assert bool(direct.value > 1.0)
    assert not bool(jnp.any(samples.link_valid))
    assert bool(jnp.all(samples.side_id == 0))

    exhausted = _reference(liquid_grid=liquid_grid, capacity=liquid_grid.size)
    assert not bool(exhausted.partition.valid)
    assert not bool(
        _assemble_resources_samples(
            reference=exhausted, capacity=liquid_grid.size
        ).valid
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_hidden_corner_crossing_requires_explicit_topology_refusal() -> None:
    """Probe agreement cannot authorize a missed source-candidate crossing."""
    reference = _reference(liquid_grid=jnp.asarray([1.0, 2.0]), capacity=8)
    draw: Mapping[ShockName, ScalarFloat] = {
        "eps": jnp.asarray(0.0, dtype=reference.partition.points.dtype)
    }
    # These are the declared quarter/midpoint/three-quarter refinement probes.
    for coordinate in (1.25, 1.5, 1.75):
        node = reference.evaluator.node(liquid=jnp.asarray(coordinate), physical=draw)
        assert bool(node.valid)
        assert not bool(node.corner)
        np.testing.assert_array_equal(node.value, jnp.asarray(1.0))
    hidden = reference.evaluator.node(liquid=jnp.asarray(1.37), physical=draw)
    assert bool(hidden.valid)
    assert bool(hidden.corner)
    assert bool(hidden.value > 1.0)
    assert int(hidden.native_owner) == -1
    with pytest.raises(
        UnsupportedOperationError,
        match="complete selected-candidate row/corner topology",
    ):
        publish_projected_resources(reference=reference, capacity=8)


def _finite_demands(
    *,
    reference: ProjectedResourcesReference,
    liquid_grid: Float1D,
    demanded_queries: Float1D,
    capacity: int,
) -> FiniteResourcesDemands:
    return publish_finite_resources_demands(
        authority=ProjectedChildAuthority(
            producer="saving", period=0, liquid_state="liquid", capacity=capacity
        ),
        evaluator=reference.evaluator,
        physical=reference.physical,
        factors=reference.factors,
        law_valid=reference.law_valid,
        liquid_grid=liquid_grid,
        demanded_queries=demanded_queries,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_finite_demand_preserves_original_selection_and_refuses_holes() -> None:
    """Exact off-grid channels and rosters coexist with refused open intervals."""
    liquid_grid = jnp.asarray([1.0, 2.0])
    reference = _reference(liquid_grid=liquid_grid, capacity=8)
    reference = replace(
        reference,
        physical=MappingProxyType({"eps": jnp.asarray([0.0, 0.1])}),
        factors=jnp.asarray([[1.0, 0.0]]),
    )
    demanded_queries = jnp.asarray([1.375, 1.25, 1.375])
    result = _finite_demands(
        reference=reference,
        liquid_grid=liquid_grid,
        demanded_queries=demanded_queries,
        capacity=8,
    )
    assert result.reference.evaluator is reference.evaluator
    assert result.reference.physical is reference.physical
    assert result.reference.factors is reference.factors
    assert result.demanded_queries is demanded_queries
    assert bool(result.samples.valid)
    np.testing.assert_array_equal(result.live, np.array([True, False]))
    assert not bool(jnp.any(result.selected.corner[:, 1]))
    active_grid = np.asarray(result.samples.grid)[: int(result.samples.valid_length)]
    for query, expected_corner in zip(
        demanded_queries, (True, False, True), strict=True
    ):
        positions = np.flatnonzero(active_grid == np.asarray(query))
        assert positions.size == 1
        index = int(positions[0])
        direct = reference.evaluator.node(
            liquid=query,
            physical={"eps": jnp.asarray(0.0, dtype=liquid_grid.dtype)},
        )
        for name in ("native_owner", "side_id", "corner", "feasible", "valid"):
            np.testing.assert_array_equal(
                getattr(result.selected, name)[index, 0], getattr(direct, name)
            )
        assert bool(result.selected.corner[index, 0]) is expected_corner
        np.testing.assert_allclose(
            result.samples.value[index],
            result.selected.value[index, 0],
            rtol=10 ** (-DECIMAL_PRECISION),
            atol=10 ** (-DECIMAL_PRECISION),
        )
        np.testing.assert_allclose(
            result.samples.economic_marginal[index],
            result.selected.marginal[index, 0],
            rtol=10 ** (-DECIMAL_PRECISION),
            atol=10 ** (-DECIMAL_PRECISION),
        )
    corner_index = int(
        np.flatnonzero(active_grid == np.asarray(demanded_queries[0]))[0]
    )
    assert bool(result.samples.economic_marginal[corner_index] < -1.0)
    reader = ProjectedChildReader(
        authority=result.authority,
        carry=result.samples,
        next_liquid=_identity_query,
    )
    exact = reader(demanded_queries[0])
    assert exact.query_status is not None
    assert not bool(exact.query_status[0].coverage_missing)
    hole = reader(jnp.asarray(1.4, dtype=liquid_grid.dtype))
    assert hole.query_status is not None
    assert bool(hole.query_status[0].coverage_missing)
    assert not bool(jnp.any(result.samples.link_valid))
    with pytest.raises(
        InvalidValueFunctionError, match="Invalid projected continuation"
    ):
        validate_query_input_status(
            statuses=hole.query_status,
            authorities=(("saving", 0, result.authority),),
        )

    tied = _finite_demands(
        reference=replace(
            reference,
            evaluator=replace(reference.evaluator, utility_of_action=_unit_utility),
        ),
        liquid_grid=liquid_grid,
        demanded_queries=demanded_queries,
        capacity=8,
    )
    assert bool(tied.samples.valid)
    np.testing.assert_array_equal(
        tied.selected.value[corner_index, 0], jnp.asarray(1.0)
    )
    assert not bool(tied.selected.corner[corner_index, 0])
    assert int(tied.selected.native_owner[corner_index, 0]) == 0


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("defect", ["domain", "nonfinite", "capacity", "law", "source"])
def test_finite_demand_integrity_is_mandatory(*, defect: str) -> None:
    """Rejected demands and source corruption cannot shrink to a valid bank."""
    liquid_grid = jnp.asarray([1.0, 2.0])
    reference = _reference(liquid_grid=liquid_grid, capacity=8)
    demanded_queries = jnp.asarray([1.375])
    capacity = 8
    if defect == "domain":
        demanded_queries = jnp.asarray([3.0])
    elif defect == "nonfinite":
        demanded_queries = jnp.asarray([jnp.nan])
    elif defect == "capacity":
        capacity = 2
    elif defect == "law":
        reference = replace(reference, law_valid=jnp.asarray(False))  # noqa: FBT003
    else:
        reference = replace(
            reference,
            evaluator=replace(
                reference.evaluator,
                row=reference.evaluator.row._replace(
                    construction_invalid=jnp.asarray(True)  # noqa: FBT003
                ),
            ),
        )
    result = _finite_demands(
        reference=reference,
        liquid_grid=liquid_grid,
        demanded_queries=demanded_queries,
        capacity=capacity,
    )
    assert not bool(result.samples.valid)
    assert not bool(_read_projected(carry=result.samples, query=liquid_grid[0]).valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_integrated_marginal_overflow_invalidates_the_entire_bank(
    *, sign: float
) -> None:
    """Finite original marginals cannot authorize an overflowed chain factor."""
    liquid_grid = jnp.asarray([0.25, 0.5])
    reference = _reference(liquid_grid=liquid_grid, capacity=8)
    half_max = jnp.asarray(jnp.finfo(liquid_grid.dtype).max / 2.0)
    source = reference.evaluator.row._replace(
        marginal=jnp.stack((sign * half_max, jnp.zeros_like(half_max)))
    )
    reference = replace(
        reference,
        evaluator=replace(
            reference.evaluator,
            row=source,
            resources=BoundResourcesMap(
                function=_scaled_resources,
                liquid_state="liquid",
                pool=MappingProxyType({}),
                names=("liquid", "eps"),
            ),
            resources_slope=jnp.asarray(4.0, dtype=liquid_grid.dtype),
        ),
    )
    evaluated, selected = reference.evaluator.integrate_with_selection(
        liquid=liquid_grid[0],
        physical=reference.physical,
        factors=reference.factors,
        law_valid=reference.law_valid,
    )
    assert bool(jnp.all(selected.valid & selected.feasible))
    assert bool(jnp.all(jnp.isfinite(selected.value) & jnp.isfinite(selected.marginal)))
    assert not bool(jnp.isfinite(evaluated.economic_marginal))
    assert not bool(evaluated.valid)
    assert bool(
        reference.evaluator.integrate(
            liquid=liquid_grid[-1],
            physical=reference.physical,
            factors=reference.factors,
            law_valid=reference.law_valid,
        ).valid
    )
    bank = _finite_demands(
        reference=reference,
        liquid_grid=liquid_grid,
        demanded_queries=jnp.empty((0,), dtype=liquid_grid.dtype),
        capacity=8,
    )
    assert not bool(bank.samples.valid)
    assert not bool(_read_projected(carry=bank.samples, query=liquid_grid[-1]).valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_finite_value_accumulation_overflow_is_not_economic_infinity(
    *, sign: float
) -> None:
    """Overflow of finite selected scores never supplies economic provenance."""
    liquid_grid = jnp.asarray([1.0, 2.0])
    reference = _reference(liquid_grid=liquid_grid, capacity=8)
    half_max = jnp.asarray(jnp.finfo(liquid_grid.dtype).max / 2.0)
    source = reference.evaluator.row._replace(
        envelope=reference.evaluator.row.envelope._replace(
            value=jnp.full_like(liquid_grid, sign * half_max * 1.5)
        ),
        expected_floor_value=-half_max * 1.75,
        discounted_floor_value=-half_max * 1.75,
    )
    # Reduction fault injection: these duplicate unit factors are deliberately
    # unnormalised, not a claim of an admitted probability law.
    reference = replace(
        reference,
        evaluator=replace(reference.evaluator, row=source),
        physical=MappingProxyType({"eps": jnp.zeros((2,), dtype=liquid_grid.dtype)}),
        factors=jnp.ones((1, 2), dtype=liquid_grid.dtype),
    )
    evaluated, selected = reference.evaluator.integrate_with_selection(
        liquid=liquid_grid[0],
        physical=reference.physical,
        factors=reference.factors,
        law_valid=reference.law_valid,
    )
    assert bool(jnp.all(selected.valid & selected.feasible))
    assert bool(jnp.all(jnp.isfinite(selected.value)))
    assert bool(jnp.isinf(evaluated.value))
    assert bool(jnp.isneginf(evaluated.value)) is (sign < 0.0)
    assert not bool(evaluated.valid)
    bank = _finite_demands(
        reference=reference,
        liquid_grid=liquid_grid,
        demanded_queries=jnp.empty((0,), dtype=liquid_grid.dtype),
        capacity=8,
    )
    assert not bool(bank.samples.valid)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("economic_source", ["empty", "feasible_row"])
def test_original_economic_minus_infinity_retains_zero_marginal(
    *, economic_source: Literal["empty", "feasible_row"]
) -> None:
    """Original economic infinity remains valid independently of feasibility."""
    liquid_grid = jnp.asarray(
        [0.125, 0.25] if economic_source == "empty" else [1.0, 2.0]
    )
    reference = _reference(liquid_grid=liquid_grid, capacity=8)
    if economic_source == "feasible_row":
        source = reference.evaluator.row._replace(
            envelope=reference.evaluator.row.envelope._replace(
                value=jnp.asarray([-jnp.inf, 1.0])
            ),
            marginal=jnp.asarray([0.0, 2.0]),
            node_kind=jnp.asarray(
                [QueryNodeKind.ECONOMIC_MINUS_INF, QueryNodeKind.FINITE],
                dtype=jnp.int32,
            ),
            expected_floor_value=jnp.asarray(-jnp.inf),
            discounted_floor_value=jnp.asarray(-jnp.inf),
        )
        reference = replace(
            reference,
            evaluator=replace(reference.evaluator, row=source),
            physical=MappingProxyType({"eps": jnp.asarray([0.0, 1.0])}),
            factors=jnp.asarray([[0.5, 0.5]]),
        )
    evaluated, selected = reference.evaluator.integrate_with_selection(
        liquid=liquid_grid[0],
        physical=reference.physical,
        factors=reference.factors,
        law_valid=reference.law_valid,
    )
    assert bool(jnp.all(selected.valid))
    assert bool(jnp.any(jnp.isneginf(selected.value)))
    assert bool(jnp.all(selected.feasible)) is (economic_source == "feasible_row")
    assert bool(evaluated.empty) is (economic_source == "empty")
    assert bool(evaluated.valid)
    assert bool(jnp.isneginf(evaluated.value))
    np.testing.assert_array_equal(evaluated.economic_marginal, jnp.asarray(0.0))
    bank = _finite_demands(
        reference=reference,
        liquid_grid=liquid_grid,
        demanded_queries=jnp.empty((0,), dtype=liquid_grid.dtype),
        capacity=8,
    )
    assert bool(bank.samples.valid)
    assert bool(_read_projected(carry=bank.samples, query=liquid_grid[0]).valid)
    for invalid in (
        replace(reference, law_valid=jnp.asarray(False)),  # noqa: FBT003
        replace(
            reference,
            evaluator=replace(
                reference.evaluator,
                row=reference.evaluator.row._replace(
                    construction_invalid=jnp.asarray(a=True)
                ),
            ),
        ),
    ):
        assert not bool(
            _finite_demands(
                reference=invalid,
                liquid_grid=liquid_grid,
                demanded_queries=jnp.empty((0,), dtype=liquid_grid.dtype),
                capacity=8,
            ).samples.valid
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_projected_child_date_guard_returns_the_actual_producer_date() -> None:
    """Missing source dates fail before projected producer-date arithmetic."""
    grid = jnp.asarray([1.0, 2.0])
    samples = _assemble_resources_samples(
        reference=_reference(liquid_grid=grid, capacity=8), capacity=8
    )
    spec = ProjectedContinuationSpec(
        template=samples,
        templates_by_period={1: samples},
        producer="saving",
        liquid_state="liquid",
    )
    with pytest.raises(TypeError, match="actual source period"):
        _validate_projected_child(
            spec=spec, source_period=None, euler_state_name="liquid", query=None
        )
    assert (
        _validate_projected_child(
            spec=spec, source_period=0, euler_state_name="liquid", query=None
        )
        == 1
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_resources_parameter_schema_refuses_grouped_leaves() -> None:
    """The scalar/array proof preserves numeric shape and refuses grouped inputs."""
    numeric = jnp.asarray([0.5, 1.0])
    abstract = _resources_parameter_argument(name="rates", value=numeric)
    assert abstract.shape == numeric.shape
    assert abstract.dtype == numeric.dtype
    for grouped in (MappingLeaf({"rate": numeric}), SequenceLeaf((numeric,))):
        with pytest.raises(UnsupportedOperationError, match="grouped parameter leaves"):
            _resources_parameter_argument(name="rates", value=grouped)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_projected_abstract_template_retains_original_leaf_read_schema() -> None:
    """Actual eval_shape descriptors retain original projected read addresses."""
    grid = jnp.asarray([1.0, 2.0])
    carry = ProjectedEGMCarry(
        grid=grid,
        value=grid,
        economic_marginal=grid,
        owner_id=jnp.zeros((2,), dtype=jnp.int32),
        side_id=jnp.zeros((2,), dtype=jnp.int32),
        node_kind=jnp.full((2,), QueryNodeKind.FINITE, dtype=jnp.int32),
        link_valid=jnp.ones((1,), dtype=jnp.bool_),
        valid_length=jnp.asarray(2, dtype=jnp.int32),
        domain_lower=jnp.asarray(1.0),
        domain_upper=jnp.asarray(2.0),
        valid=jnp.asarray(True),  # noqa: FBT003
    )
    template = jax.eval_shape(lambda value: value, carry)
    original = carry.leaves()
    abstract = template.leaves()
    assert tuple(abstract) == tuple(original)
    for path, leaf in abstract.items():
        assert isinstance(leaf, jax.ShapeDtypeStruct)
        assert leaf is getattr(template, path[0])
        assert leaf.shape == original[path].shape
        assert leaf.dtype == original[path].dtype
    reads = continuation_leaf_reads(
        template=template,
        artifact_key=PROJECTED_EGM_CONTINUATION,
        target="child",
        source_regime="parent",
        source_period=0,
        core_key="original_parent",
    )
    assert tuple(read.target.leaf_path for read in reads) == tuple(abstract)
    assert tuple(read.source.path for read in reads) == tuple(
        ("child", *path) for path in abstract
    )
    inputs = SolveInputMappings[FloatND](
        next_regime_to_V_arr=MappingProxyType({}),
        next_regime_to_continuation=MappingProxyType({"child": template}),
        next_edge_to_V_arr=MappingProxyType({}),
    )
    registry = BufferRegistry()
    register_rolled_inputs(
        inputs=inputs,
        next_period=1,
        ledger=PlannedInputLiveness(dispatch_accesses={}),
        registry=registry,
    )
    assert not registry._keys_by_shard
    ledger = PlannedInputLiveness(
        dispatch_accesses={(0, "parent"): tuple(read.target for read in reads)}
    )
    with pytest.raises(
        TypeError, match=r"^Runtime continuation storage requires actual arrays[.]$"
    ):
        register_rolled_inputs(
            inputs=inputs,
            next_period=1,
            ledger=ledger,
            registry=registry,
        )
    concrete = SolveInputMappings[FloatND](
        next_regime_to_V_arr=MappingProxyType({}),
        next_regime_to_continuation=MappingProxyType({"child": carry}),
        next_edge_to_V_arr=MappingProxyType({}),
    )
    artifact = next(
        read.target for read in reads if read.target.leaf_path == ("node_kind",)
    )
    templates = SolveInputMappings[ValueTemplate](
        next_regime_to_V_arr=MappingProxyType({}),
        next_regime_to_continuation=MappingProxyType({"child": template}),
        next_edge_to_V_arr=MappingProxyType({}),
    )
    replaced = substitute_artifact(
        inputs=concrete, templates=templates, artifact=artifact
    )
    restored = replaced.next_regime_to_continuation["child"]
    assert isinstance(restored, ProjectedEGMCarry)
    for path, leaf in restored.leaves().items():
        assert leaf is (abstract[path] if path == ("node_kind",) else original[path])
    with pytest.raises(TypeError, match="no concrete runtime leaf"):
        substitute_artifact(inputs=inputs, templates=templates, artifact=artifact)


def _affine_projection_row(
    *, capacity: int, valid_length: int = 2
) -> ProjectedEGMCarry:
    """Publish an affine value and its independent constant economic derivative."""
    grid = jnp.full((capacity,), jnp.nan).at[:2].set(jnp.asarray([1.0, 2.0]))
    return ProjectedEGMCarry(
        grid=grid,
        value=jnp.full_like(grid, jnp.nan).at[:2].set(jnp.asarray([5.0, 7.0])),
        economic_marginal=jnp.full_like(grid, jnp.nan).at[:2].set(2.0),
        owner_id=jnp.full((capacity,), -1, dtype=jnp.int32).at[:2].set(11),
        side_id=jnp.full((capacity,), -1, dtype=jnp.int32).at[:2].set(0),
        node_kind=jnp.full((capacity,), QueryNodeKind.PAD, dtype=jnp.int32)
        .at[:2]
        .set(QueryNodeKind.FINITE),
        link_valid=jnp.zeros((capacity - 1,), dtype=jnp.bool_).at[0].set(True),
        valid_length=jnp.asarray(valid_length, dtype=jnp.int32),
        domain_lower=jnp.asarray(1.0),
        domain_upper=jnp.asarray(2.0),
        valid=jnp.asarray(a=True),
    )


def _affine_projection_reference(query: ScalarFloat) -> ProjectedRowRead:
    """Read the literal affine equation without interpolating stored channels."""
    return ProjectedRowRead(
        value=2.0 * query + 3.0,
        economic_marginal=jnp.asarray(2.0, dtype=query.dtype),
        owner_id=jnp.asarray(11, dtype=jnp.int32),
        side_id=jnp.asarray(0, dtype=jnp.int32),
        valid=jnp.asarray(a=True),
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("capacity", [2, 8])
def test_projection_assessment_padding_preserves_affine_channels(
    *, compiled: bool, capacity: int
) -> None:
    """Inactive poisoned storage leaves a genuine active affine interval unchanged."""
    carry = _affine_projection_row(capacity=capacity)
    function = (
        jax.jit(_assess_cells, static_argnames=("reference", "accuracy"))
        if compiled
        else _assess_cells
    )
    hard, failed = function(
        carry=carry, reference=_affine_projection_reference, accuracy="balanced"
    )
    assert not bool(hard)
    np.testing.assert_array_equal(failed, np.zeros((capacity - 1,), dtype=bool))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("defect", ["invalid_source", "owner", "marginal"])
def test_projection_assessment_active_integrity_and_marginal_are_mandatory(
    *, compiled: bool, defect: str
) -> None:
    """An active source or owner defect fails despite identical economic values."""
    carry = _affine_projection_row(capacity=8)

    def reference(query: ScalarFloat) -> ProjectedRowRead:
        read = _affine_projection_reference(query)
        middle = query == 1.5
        return read._replace(
            valid=read.valid & ~(middle & (defect == "invalid_source")),
            owner_id=jnp.where(middle & (defect == "owner"), 12, read.owner_id),
            economic_marginal=jnp.where(
                middle & (defect == "marginal"), 4.0, read.economic_marginal
            ),
        )

    function = (
        jax.jit(_assess_cells, static_argnames=("reference", "accuracy"))
        if compiled
        else _assess_cells
    )
    hard, failed = function(carry=carry, reference=reference, accuracy="balanced")
    assert bool(hard) is (defect != "marginal")
    np.testing.assert_array_equal(
        failed,
        np.asarray([defect == "marginal", False, False, False, False, False, False]),
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("compiled", [False, True])
def test_projection_assessment_inactive_reference_invalidity_is_ignored(
    *, compiled: bool
) -> None:
    """The same invalid reference matters only when an interval is active."""

    def reference(query: ScalarFloat) -> ProjectedRowRead:
        return _affine_projection_reference(query)._replace(valid=jnp.asarray(a=False))

    function = (
        jax.jit(_assess_cells, static_argnames=("reference", "accuracy"))
        if compiled
        else _assess_cells
    )
    inactive = _affine_projection_row(capacity=8, valid_length=1)
    hard, failed = function(carry=inactive, reference=reference, accuracy="balanced")
    assert not bool(hard)
    np.testing.assert_array_equal(failed, np.zeros((7,), dtype=bool))
    active = replace(inactive, valid_length=jnp.asarray(2, dtype=jnp.int32))
    hard, failed = function(carry=active, reference=reference, accuracy="balanced")
    assert bool(hard)
    np.testing.assert_array_equal(failed, np.zeros((7,), dtype=bool))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("defect", ["missing_link", "reversed", "nonfinite"])
def test_projection_assessment_active_geometry_and_holes_still_refuse(
    *, defect: str
) -> None:
    """Inactive capacity cannot hide active geometry or certified topology failure."""
    carry = _affine_projection_row(capacity=8)
    if defect == "missing_link":
        carry = replace(carry, link_valid=carry.link_valid.at[0].set(False))
    elif defect == "reversed":
        carry = replace(carry, grid=carry.grid.at[1].set(0.0))
    else:
        carry = replace(carry, grid=carry.grid.at[1].set(jnp.nan))
    hard, _ = jax.jit(_assess_cells, static_argnames=("reference", "accuracy"))(
        carry=carry, reference=_affine_projection_reference, accuracy="certified"
    )
    assert bool(hard)
