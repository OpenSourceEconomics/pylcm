"""Model-owned query semantics survive declared reads and artifact persistence."""

import inspect
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import continuation
from _lcm.egm.budget import DCEGM_BUDGET_CONSTRAINT_NAME, _IntrinsicBudgetConstraint
from _lcm.egm.carry import (
    EGMCarry,
    build_template_egm_carry,
    build_template_egm_query_data,
)
from _lcm.engine import EGMPolicyRead
from _lcm.grids import LinSpacedGrid
from _lcm.solution.continuation_reads import continuation_leaf_reads
from _lcm.solution.egm import _EGMCore
from _lcm.solution.fingerprint import fingerprint_model_structure
from _lcm.solution.model_authority import SolutionAuthority, build_solution_authority
from lcm import AgeGrid, Model, post_decision_lower_bound
from lcm.exceptions import UnsupportedOperationError
from lcm.persistence import load_solution, save_solution
from lcm.solver_api import (
    EGM_CONTINUATION,
    ArtifactRef,
    ArtifactStore,
    LoadState,
    ResultRetention,
    SolutionMetadata,
    SolutionResult,
    ValueArraySchema,
    _snapshot_artifact_template_once,
)
from lcm.typing import FloatND
from tests.constraints.test_dynamic_lower_bound import LIQUID, RegimeId, _model, lower


def _outer_target(*, adjustment: FloatND) -> FloatND:
    return adjustment


def _outer_utility(
    *,
    consumption: FloatND,
    adjustment: FloatND,
    outer_target: FloatND,
    outer_cost: FloatND,
) -> FloatND:
    return consumption - adjustment + 0.0 * (outer_target + outer_cost)


def _outer_cost(*, adjustment: FloatND, income: FloatND) -> FloatND:
    return adjustment + income


# A finite nested operator over the two-node `adjustment` grid; slot zero is the
# keeper, so its capacity is three.
_FINITE_OUTER = continuation.FiniteOuterQuerySpec(
    outer_state="income",
    outer_action="adjustment",
    outer_post_decision="outer_target",
    keeper_target=None,
    cost_target="outer_cost",
    outer_nodes=(0.0, 1.0),
)


def _feature_model(
    *,
    liquid_corner: bool = False,
    outer_mode: Literal["none", "finite_max"] = "none",
    multiple_periods: bool = False,
) -> Model:
    """Install transport-only query declarations on a canonical ordinary model."""
    has_outer = outer_mode != "none"
    model = _model(
        declaration=post_decision_lower_bound(margin=LIQUID, lower=lower),
        functions={
            "utility": _outer_utility,
            "outer_target": _outer_target,
            "outer_cost": _outer_cost,
        }
        if has_outer
        else None,
        extra_actions={"adjustment": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)}
        if has_outer
        else None,
    )
    if multiple_periods:
        model = Model(
            regimes={"done": model.user_regimes["done"]},
            edges={},
            ages=AgeGrid(exact_values=(0, 1)),
            regime_id_class=RegimeId,
            initial_nodes={0: "done", 1: "done"},
        )
    regime = model._regimes["done"]
    bound = regime.solution.post_decision_bounds[0]
    outer_capacity = 3 if has_outer else 0
    finite = _FINITE_OUTER if outer_mode == "finite_max" else None
    query = build_template_egm_query_data(
        outer_capacity=outer_capacity,
        state_shape=(2,),
        action_shape=(),
        row_capacity=5,
        interval_capacity=1,
        liquid_corner=liquid_corner,
    )
    template = build_template_egm_carry(
        n_rows=5,
        leading_shape=(2, outer_capacity) if has_outer else (2,),
        query_data=query,
    )
    builder = getattr(continuation, "build_query_model_authority", None)
    assert builder is not None, "Query semantics need canonical evaluator construction."
    authority = builder(
        bound=bound,
        outer_mode=outer_mode,
        outer_capacity=outer_capacity,
        finite_outer=finite,
        state_names=("income",),
        action_names=(),
        coordinate_kind="resources" if finite is None else "nested_common_resources",
        marginal_convention="ordinary_linear",
        corner_coordinate_kind="liquid_row" if liquid_corner else "compact_interval",
        row_capacity=5,
        interval_ids=(0,),
        interval_side_ids=(0,),
        evaluator_targets=(("resources", "resources"), ("utility", "utility")),
        functions=regime.solution.continuation_functions,
        functions_by_period=MappingProxyType(
            {
                period: MappingProxyType({**functions, **constraints})
                for period, (
                    functions,
                    constraints,
                ) in regime.solution.feasibility_pools_by_period.items()
            }
        ),
        target_periods=regime.active_periods,
    )
    spec = continuation.EGMContinuationSpec(
        template=template,
        layout=continuation.EGMContinuationLayout(
            query_spec=authority, n_stacked_candidates=outer_capacity
        ),
    )
    solution_phase = replace(
        regime.solution, continuation_spec=spec, query_authority=authority
    )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            {
                "done": replace(regime, solution=solution_phase),
            }
        ),
    )
    return model


def _authority(model: Model) -> SolutionAuthority:
    """Build artifact declarations from canonical model inputs without a solve."""
    assert model.ages is not None
    return build_solution_authority(
        regimes=model._regimes,
        flat_params=model._process_params({"floor_scale": 0.0}),
        ages=model.ages,
    )


@pytest.mark.parametrize("liquid_corner", [False, True])
def test_query_artifact_axes_bind_floor_links_and_intervals(
    *, liquid_corner: bool
) -> None:
    """Each numeric field has its own independent model-labelled dimensions."""
    model = _feature_model(liquid_corner=liquid_corner)
    result = _authority(model)
    ref = ArtifactRef(period=0, regime="done", key=EGM_CONTINUATION)
    descriptor = result.artifact_descriptors[ref]
    by_path = {leaf.path: leaf for leaf in descriptor.leaf_descriptors}
    assert by_path[("attribute:query_data", "attribute:floor_used")].axis_names == (
        "income",
    )
    assert by_path[("attribute:query_data", "attribute:link_valid")].axis_names == (
        "income",
        "pylcm:egm:link",
    )
    corner = (
        (
            "attribute:query_data",
            "attribute:liquid_corner_data",
            "attribute:continuation",
        )
        if liquid_corner
        else (
            "attribute:query_data",
            "attribute:corner_continuation",
        )
    )
    assert by_path[corner].axis_names == (
        ("income", "pylcm:egm:node", "pylcm:egm:interval")
        if liquid_corner
        else ("income", "pylcm:egm:interval")
    )


@pytest.mark.parametrize(
    "change",
    ["evaluator", "inputs", "period", "coordinate", "marginal", "interval", "lower"],
)
def test_query_artifact_rejects_semantic_substitution(*, change: str) -> None:
    """Matching shapes never authenticate substituted query semantics."""
    model = _feature_model()
    regime = model._regimes["done"]
    spec = cast("continuation.EGMContinuationSpec", regime.solution.continuation_spec)
    query_spec = spec.layout.query_spec
    assert query_spec is not None
    changes = {
        "evaluator": {"evaluator_identity": (("resources", "foreign"),)},
        "inputs": {"evaluator_inputs": (("resources", ("consumption",)),)},
        "period": {"target_periods": (1,)},
        "coordinate": {"coordinate_kind": "liquid"},
        "marginal": {"marginal_convention": "paired_value_derivative"},
        "interval": {"interval_ids": (9,)},
        "lower": {"bound": replace(query_spec.bound, lower_target="foreign")},
    }
    tampered = replace(
        spec,
        layout=replace(spec.layout, query_spec=replace(query_spec, **changes[change])),
    )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            {
                "done": replace(
                    regime,
                    solution=replace(regime.solution, continuation_spec=tampered),
                ),
            }
        ),
    )
    with pytest.raises(TypeError, match=r"Query|query"):
        _authority(model)


def test_filtered_read_cannot_drop_query_leaf() -> None:
    """Flattened query inputs declare every numeric field they transport."""
    template = build_template_egm_carry(
        n_rows=5,
        query_data=build_template_egm_query_data(
            state_shape=(),
            action_shape=(),
            row_capacity=5,
            interval_capacity=1,
        ),
    )
    with pytest.raises(TypeError, match=r"query.*read|Query.*read"):
        continuation_leaf_reads(
            template=template,
            artifact_key=EGM_CONTINUATION,
            target="done",
            source_regime="parent",
            source_period=0,
            core_key="main",
            argument_by_leaf={("endog_grid",): "next_liquid_grid"},
        )


@pytest.mark.parametrize("liquid_corner", [False, True])
def test_query_container_snapshot_has_inert_static_metadata(
    *, liquid_corner: bool
) -> None:
    """Nested numeric records can enter canonical persistence snapshots."""

    template = build_template_egm_carry(
        n_rows=5,
        query_data=build_template_egm_query_data(
            state_shape=(),
            action_shape=(),
            row_capacity=5,
            interval_capacity=1,
            liquid_corner=liquid_corner,
        ),
    )
    snapshot, containers = _snapshot_artifact_template_once(
        template=template,
        payload_runtime_type=type(template),
    )
    assert ("attribute:query_data", "attribute:savings") in snapshot.leaf_paths
    assert containers[("attribute:query_data",)] is type(template.query_data)


def _query_solution(model: Model) -> SolutionResult:
    """Build a numeric transport fixture from independent canonical artifact schemas."""
    authority = _authority(model)
    ref = ArtifactRef(period=0, regime="done", key=EGM_CONTINUATION)
    value = authority.values[0, "done"]
    spec = cast(
        "continuation.EGMContinuationSpec",
        model._regimes["done"].solution.continuation_spec,
    )
    result = SolutionResult(
        values={0: {"done": jnp.zeros(value.shape)}},
        retained_continuations=ArtifactStore({ref: spec.template}),
        metadata=SolutionMetadata(
            retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
            n_periods=1,
            regime_names=("done",),
            solver_types={"done": "GridSearch"},
            model_instance_id="query-transport-fixture",
            params_fingerprint="0" * 64,
            value_schemas={
                (0, "done"): ValueArraySchema(
                    shape=value.shape, dtype=value.dtype, axis_names=value.axis_names
                )
            },
            artifact_descriptors={ref: authority.artifact_descriptors[ref]},
        ),
    )
    object.__setattr__(
        result, "_artifact_authority", MappingProxyType({ref: authority.artifacts[ref]})
    )
    return result


@pytest.mark.parametrize("liquid_corner", [False, True])
def test_query_numeric_archive_roundtrip_uses_fresh_model_template(
    *,
    tmp_path: Path,
    liquid_corner: bool,
) -> None:
    """Independent lazy entries restore exact query containers, paths and dtypes."""
    model = _feature_model(liquid_corner=liquid_corner)
    result = _query_solution(model)
    ref = ArtifactRef(period=0, regime="done", key=EGM_CONTINUATION)
    restored = load_solution(
        path=save_solution(solution=result, path=tmp_path / "query.lcm")
    )
    assert restored.retained_continuations.load_state(ref) is LoadState.UNLOADED
    fresh = _feature_model(liquid_corner=liquid_corner)
    template = cast(
        "continuation.EGMContinuationSpec",
        fresh._regimes["done"].solution.continuation_spec,
    ).template
    actual = cast(
        "EGMCarry", restored.retained_continuations.materialize(ref, template=template)
    )
    expected = cast("EGMCarry", result.retained_continuations[ref])
    assert type(actual.query_data) is type(expected.query_data)
    assert tuple(actual.leaves()) == tuple(expected.leaves())
    for path, leaf in actual.leaves().items():
        assert leaf.dtype == expected.leaves()[path].dtype
        np.testing.assert_array_equal(leaf, expected.leaves()[path])
    assert restored.retained_continuations.load_state(ref) is LoadState.LOADED


@pytest.mark.parametrize("change", ["missing", "roles", "function", "input"])
def test_query_authority_authenticates_canonical_context(*, change: str) -> None:
    """Model-owned roles and callable dependencies authenticate the descriptor."""
    model = _feature_model()
    regime = model._regimes["done"]
    phase = regime.solution
    expected = phase.query_authority
    assert expected is not None
    original_spec = cast("continuation.EGMContinuationSpec", phase.continuation_spec)
    if change == "missing":
        phase = replace(phase, query_authority=None)
    elif change == "roles":
        substitute = replace(expected, state_names=("wealth",))
        phase = replace(
            phase,
            query_authority=substitute,
            continuation_spec=replace(
                original_spec,
                layout=replace(original_spec.layout, query_spec=substitute),
            ),
        )
    else:

        def foreign_resources(*, wealth: FloatND) -> FloatND:
            return 2.0 * wealth

        def foreign_input(*, consumption: FloatND) -> FloatND:
            return consumption

        phase = replace(
            phase,
            _continuation_functions=MappingProxyType(
                {
                    **phase.continuation_functions,
                    "resources": foreign_input
                    if change == "input"
                    else foreign_resources,
                }
            ),
        )
    object.__setattr__(
        model, "_regimes", MappingProxyType({"done": replace(regime, solution=phase)})
    )
    with pytest.raises(TypeError, match=r"Query|query"):
        _authority(model)


def test_grouped_query_reads_address_original_nested_operands() -> None:
    """A query operand names every original numeric leaf at its nested locator."""
    template = build_template_egm_carry(
        n_rows=5,
        query_data=build_template_egm_query_data(
            state_shape=(),
            action_shape=(),
            row_capacity=5,
            interval_capacity=1,
        ),
    )
    reads = continuation_leaf_reads(
        template=template,
        artifact_key=EGM_CONTINUATION,
        target="done",
        source_regime="parent",
        source_period=0,
        core_key="main",
        argument_by_leaf={("query_data",): "next_query_data"},
    )
    expected = {path for path in template.leaves() if path[0] == "query_data"}
    assert {read.target.leaf_path for read in reads} == expected
    assert tuple((read.source.argument, read.source.path) for read in reads) == tuple(
        ("next_query_data", read.target.leaf_path[1:]) for read in reads
    )


def test_query_authority_authenticates_period_evaluator_pool() -> None:
    """A period-specific evaluator substitution invalidates saved row semantics."""
    model = _feature_model()
    regime = model._regimes["done"]
    functions, constraints = regime.solution.feasibility_pools_by_period[0]

    def foreign_resources(*, wealth: FloatND) -> FloatND:
        return 2.0 * wealth

    phase = replace(
        regime.solution,
        feasibility_pools_by_period=MappingProxyType(
            {
                0: (
                    MappingProxyType({**functions, "resources": foreign_resources}),
                    constraints,
                ),
            }
        ),
    )
    object.__setattr__(
        model, "_regimes", MappingProxyType({"done": replace(regime, solution=phase)})
    )
    with pytest.raises(TypeError, match=r"Query|query"):
        _authority(model)


def _structure_fingerprint(model: Model) -> str:
    """Hash canonical model structure with its original callable declarations."""
    return fingerprint_model_structure(
        ages=model._time,
        regimes=model._regimes,
        user_regimes=model.user_regimes,
        laws=model._graph.laws,
        regime_names_to_ids=model.regime_names_to_ids,
    )


@pytest.mark.parametrize("change", ["coordinate", "interval", "outer", "source"])
def test_query_structure_fingerprint_seals_semantics(*, change: str) -> None:
    """Seal conventions, interval/outer labels and evaluator bindings durably."""
    model = _feature_model()
    expected = _structure_fingerprint(model)
    regime = model._regimes["done"]
    spec = regime.solution.query_authority
    assert spec is not None
    changes = {
        "coordinate": {"marginal_convention": "paired_value_derivative"},
        "interval": {"interval_ids": (9,)},
        "outer": {
            "outer_mode": "finite_max",
            "outer_capacity": 3,
            "finite_outer": _FINITE_OUTER,
            "coordinate_kind": "nested_common_resources",
        },
        "source": {
            "evaluator_targets": (("resources", "savings"), ("utility", "utility"))
        },
    }
    phase = replace(regime.solution, query_authority=replace(spec, **changes[change]))
    object.__setattr__(
        model, "_regimes", MappingProxyType({"done": replace(regime, solution=phase)})
    )
    assert _structure_fingerprint(model) != expected


def test_query_resource_evaluator_matches_installed_liquid_role() -> None:
    """A utility node cannot impersonate the installed resource evaluator."""
    model = _feature_model()
    phase = model._regimes["done"].solution
    spec = phase.query_authority
    assert spec is not None
    with pytest.raises(TypeError, match=r"Query|query"):
        continuation.build_query_model_authority(
            bound=spec.bound,
            state_names=spec.state_names,
            action_names=spec.action_names,
            coordinate_kind=spec.coordinate_kind,
            marginal_convention=spec.marginal_convention,
            corner_coordinate_kind=spec.corner_coordinate_kind,
            row_capacity=spec.row_capacity,
            interval_ids=spec.interval_ids,
            interval_side_ids=spec.interval_side_ids,
            evaluator_targets=(("resources", "utility"), ("utility", "utility")),
            functions=phase.continuation_functions,
            target_periods=spec.target_periods,
        )


def _shifted_lower(*, floor_helper: FloatND) -> FloatND:
    return floor_helper + 1.0


def test_feature_free_fingerprint_is_reproducible_and_seals_its_floor() -> None:
    """Equivalent floors round-trip; changing the floor changes identity."""
    model = _model(declaration=post_decision_lower_bound(margin=LIQUID, lower=lower))
    observed = _structure_fingerprint(model)
    assert observed == model._model_structure_fingerprint
    rebuilt = _model(declaration=post_decision_lower_bound(margin=LIQUID, lower=lower))
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    assert observed == _structure_fingerprint(rebuilt)
    assert observed == restored._model_structure_fingerprint
    restored._check_identity_runtime()
    changed = _model(
        declaration=post_decision_lower_bound(margin=LIQUID, lower=_shifted_lower)
    )
    assert observed != _structure_fingerprint(changed)


def test_one_row_query_transport_has_explicit_model_adapter_refusal() -> None:
    """Query data never enters a one-row core as preference parameters."""

    assert "next_query_data" in inspect.signature(_EGMCore.__call__).parameters, (
        "A one-row consumer needs a typed query operand before reading feature data."
    )
    query = build_template_egm_query_data(
        state_shape=(), action_shape=(), row_capacity=5, interval_capacity=1
    )
    core = _EGMCore(
        declared_law=lambda **_: pytest.fail("Query data reached the declared law."),
        egm_one_asset_step=lambda **_: pytest.fail(
            "Query data reached the legacy step."
        ),
        build_preferences=lambda _: pytest.fail("Query data reached preferences."),
        read_discount_factor=lambda _: pytest.fail("Query data reached preferences."),
    )
    nodes = jnp.arange(5.0)
    with pytest.raises(UnsupportedOperationError, match="model-bound query adapter"):
        core(
            liquid=nodes,
            next_liquid_grid=nodes,
            next_value=nodes,
            next_marginal=nodes,
            next_liquid=nodes,
            marginal_return=nodes,
            boundary_savings_targets=nodes,
            boundary_next_liquid=nodes,
            effective_savings_grid=nodes,
            next_query_data=query,
        )


@pytest.mark.parametrize("kind", ["policy", "intrinsic"])
def test_query_authority_refuses_scalar_only_budget_descriptors(*, kind: str) -> None:
    """A scalar budget descriptor cannot supply callable query authority."""
    model = _feature_model()
    regime = model._regimes["done"]
    if kind == "policy":
        read = EGMPolicyRead(
            action_name="consumption",
            resources_target="resources",
            savings_lower_bound=0.0,
            row_discrete_state_names=(),
            row_passive_state_names=("income",),
            row_discrete_action_names=(),
            row_axis_lengths_by_period=MappingProxyType({0: (2,)}),
            float_dtype=str(jnp.asarray(0.0).dtype),
        )
        simulation = replace(
            regime.simulation, egm_policy_read=read, replay_unsupported=True
        )
        assert not simulation.replay_route.policy_applicable
    else:
        scalar = _IntrinsicBudgetConstraint(
            action_name="consumption",
            resources_name="resources",
            borrowing_limit=0.0,
        )
        simulation = replace(
            regime.simulation,
            constraints=MappingProxyType(
                {
                    **regime.simulation.constraints,
                    DCEGM_BUDGET_CONSTRAINT_NAME: scalar,
                }
            ),
        )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType({"done": replace(regime, simulation=simulation)}),
    )
    with pytest.raises(TypeError, match=r"scalar.*budget|budget.*scalar"):
        _authority(model)


@pytest.mark.parametrize("kind", ["policy", "intrinsic"])
def test_literal_budget_descriptors_keep_feature_free_authority(*, kind: str) -> None:
    """Literal budget descriptors preserve ordinary continuation authority."""
    model = _model(
        declaration=post_decision_lower_bound(margin=LIQUID, lower=0.0), helper=None
    )
    regime = model._regimes["done"]
    phase = replace(
        regime.solution,
        continuation_spec=continuation.EGMContinuationSpec(
            template=build_template_egm_carry(n_rows=5, leading_shape=(2,)),
        ),
    )
    if kind == "policy":
        read = EGMPolicyRead(
            action_name="consumption",
            resources_target="resources",
            savings_lower_bound=0.0,
            row_discrete_state_names=(),
            row_passive_state_names=("income",),
            row_discrete_action_names=(),
            row_axis_lengths_by_period=MappingProxyType({0: (2,)}),
            float_dtype=str(jnp.asarray(0.0).dtype),
        )
        simulation = replace(
            regime.simulation, egm_policy_read=read, replay_unsupported=True
        )
    else:
        scalar = _IntrinsicBudgetConstraint(
            action_name="consumption", resources_name="resources", borrowing_limit=0.0
        )
        simulation = replace(
            regime.simulation,
            constraints=MappingProxyType(
                {
                    **regime.simulation.constraints,
                    DCEGM_BUDGET_CONSTRAINT_NAME: scalar,
                }
            ),
        )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            {
                "done": replace(regime, solution=phase, simulation=simulation),
            }
        ),
    )
    assert model.ages is not None
    authority = build_solution_authority(
        regimes=model._regimes,
        flat_params=model._process_params({}),
        ages=model.ages,
    )
    ref = ArtifactRef(period=0, regime="done", key=EGM_CONTINUATION)
    assert cast("EGMCarry", authority.artifacts[ref].template).query_data is None


def test_outer_query_artifact_axes_follow_the_declared_operator() -> None:
    """Finite candidate banks keep fixed-cell floors apart from candidate rows."""
    model = _feature_model(outer_mode="finite_max")
    authority = _authority(model)
    ref = ArtifactRef(period=0, regime="done", key=EGM_CONTINUATION)
    paths = {
        leaf.path: leaf.axis_names
        for leaf in authority.artifact_descriptors[ref].leaf_descriptors
    }
    assert paths[("attribute:query_data", "attribute:floor_used")] == ("income",)
    assert paths[("attribute:query_data", "attribute:savings")] == (
        "income",
        "pylcm:egm:candidate",
        "pylcm:egm:node",
    )


@pytest.mark.parametrize("change", ["lower", "ancestor"])
def test_query_authority_authenticates_period_lower_dependencies(
    *, change: str
) -> None:
    """Every period's installed floor DAG belongs to canonical query authority."""
    model = _feature_model()
    regime = model._regimes["done"]
    bound = regime.solution.post_decision_bounds[0]
    functions, constraints = regime.solution.feasibility_pools_by_period[0]
    target = bound.lower_target if change == "lower" else "floor_helper"
    assert target is not None

    def foreign_floor(*, income: FloatND) -> FloatND:
        return 3.0 * income

    assert target in functions
    phase = replace(
        regime.solution,
        feasibility_pools_by_period=MappingProxyType(
            {
                0: (
                    MappingProxyType({**functions, target: foreign_floor}),
                    constraints,
                ),
            }
        ),
    )
    object.__setattr__(
        model, "_regimes", MappingProxyType({"done": replace(regime, solution=phase)})
    )
    with pytest.raises(TypeError, match=r"Query|query"):
        _authority(model)


@pytest.mark.parametrize("change", ["lower", "ancestor"])
@pytest.mark.parametrize("pool", ["base", "last-period"])
def test_query_authority_authenticates_lower_dependencies_in_every_context(
    *, change: str, pool: str
) -> None:
    """Base and later-period floor dependencies cannot change independently."""
    model = _feature_model(multiple_periods=True)
    regime = model._regimes["done"]
    assert regime.active_periods == (0, 1)
    expected = regime.solution.query_authority
    assert expected is not None
    assert expected.target_periods == (0, 1)
    accepted = _authority(model)
    assert {
        ref.period
        for ref in accepted.artifact_descriptors
        if ref.key == EGM_CONTINUATION
    } == {0, 1}
    bound = regime.solution.post_decision_bounds[0]
    target = bound.lower_target if change == "lower" else "floor_helper"
    assert target is not None

    def foreign_floor(*, income: FloatND) -> FloatND:
        return 3.0 * income

    if pool == "base":
        functions = regime.solution.continuation_functions
        assert target in functions
        phase = replace(
            regime.solution,
            _continuation_functions=MappingProxyType(
                {**functions, target: foreign_floor}
            ),
        )
    else:
        functions, constraints = regime.solution.feasibility_pools_by_period[1]
        assert target in functions
        phase = replace(
            regime.solution,
            feasibility_pools_by_period=MappingProxyType(
                {
                    **regime.solution.feasibility_pools_by_period,
                    1: (
                        MappingProxyType({**functions, target: foreign_floor}),
                        constraints,
                    ),
                }
            ),
        )
    assert phase.query_authority is expected
    assert phase.post_decision_bounds is regime.solution.post_decision_bounds
    object.__setattr__(
        model, "_regimes", MappingProxyType({"done": replace(regime, solution=phase)})
    )
    with pytest.raises(TypeError, match=r"Query|query"):
        _authority(model)
