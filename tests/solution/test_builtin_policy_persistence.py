"""Persistence contracts for built-in finite and adaptive replay policies."""

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.solution.contract import BackwardInductionResult
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from lcm import Model
from lcm.exceptions import InvalidSimulationInputError
from lcm.persistence import load_solution, save_solution
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    SOLVER_DIAGNOSTICS,
    ArtifactAuthority,
    ArtifactChannel,
    ArtifactDescriptor,
    ArtifactRef,
    ArtifactStore,
    AxisAuthority,
    AxisDescriptor,
    AxisRole,
    LeafAuthority,
    LeafDescriptor,
    LoadState,
    OmissionReason,
    PersistencePolicy,
    ResultRetention,
    SolutionMetadata,
    SolutionResult,
    ValueArraySchema,
    ValueStore,
)
from lcm.solvers import EGMContinuationSpec
from tests.collective_fixtures import (
    ParamsDict,
    make_couple_initial_conditions,
    make_two_stakeholder_model,
)
from tests.simulation.test_nnbegm_split_workflow_parity import (
    _INITIAL,
    _PARAMS,
    _SEED,
    _build,
)
from tests.solution.test_egm_published_policy import _two_period_bequest_model
from tests.test_models.deterministic.dcegm_variants import (
    get_retirement_only_params,
)


def _policy_refs(solution: SolutionResult) -> tuple[ArtifactRef, ...]:
    """Return every retained built-in simulation-policy address."""
    return tuple(
        ref for ref in solution.replay_artifacts if ref.key == SIMULATION_POLICY
    )


def _artifact_descriptors_by_ref(
    solution: SolutionResult,
) -> Mapping[ArtifactRef, ArtifactDescriptor]:
    """View descriptive schemas at their period/regime/artifact addresses.

    Persistence is a cell-level property: one model can use a finite route in one
    regime and an adaptive route in another while both share the public
    ``SIMULATION_POLICY`` key.
    """
    return solution.metadata.artifact_descriptors


def _solve_without_optional_dissolution_flags(
    *, monkeypatch: pytest.MonkeyPatch
) -> tuple[Model, ParamsDict, SolutionResult, tuple[ArtifactRef, ...]]:
    """Solve after suppressing optional flags at the engine-result boundary."""
    model, params = make_two_stakeholder_model()
    original_solve = model._solve_compiled

    def solve_without_dissolution_flags(**kwargs: Any) -> BackwardInductionResult:
        internal_result = original_solve(**kwargs)
        return dataclasses.replace(
            internal_result,
            dissolution_flags=MappingProxyType(
                {
                    period: MappingProxyType({})
                    for period in internal_result.dissolution_flags
                }
            ),
        )

    monkeypatch.setattr(model, "_solve_compiled", solve_without_dissolution_flags)

    solution = model.solve(
        params=params,
        log_level="off",
        retention=ResultRetention.VALUES_AND_REPLAY,
    )
    omitted_refs = tuple(
        ref for ref in solution.omissions if ref.key == DISSOLUTION_FLAG
    )
    return model, params, solution, omitted_refs


def _replace_manifest_omission_reason(
    *, path: Path, ref: ArtifactRef, reason: OmissionReason
) -> None:
    """Mutate one omission reason while preserving the manifest checksum."""
    with h5py.File(path, "r+") as archive:
        manifest = cast(
            "dict[str, object]",
            json.loads(bytes(archive["manifest"][()])),
        )
        omissions = cast("list[dict[str, object]]", manifest["omissions"])
        entry = next(
            entry
            for entry in omissions
            if entry["period"] == ref.period
            and entry["regime"] == ref.regime
            and entry["type_id"] == ref.key.type_id
            and entry["schema_version"] == ref.key.schema_version
        )
        entry["reason"] = reason.value
        manifest_bytes = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        del archive["manifest"]
        dataset = archive.create_dataset(
            "manifest",
            data=np.frombuffer(manifest_bytes, dtype=np.uint8),
        )
        dataset.attrs["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()


def test_missing_applicable_optional_artifact_is_recorded_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-memory omission ledger distinguishes absence from inapplicability."""
    _model, _params, solution, omitted_refs = _solve_without_optional_dissolution_flags(
        monkeypatch=monkeypatch
    )

    assert omitted_refs
    assert all(
        solution._artifact_authority[ref].applicable
        and not solution._artifact_authority[ref].required
        for ref in omitted_refs
    )
    assert all(
        solution.omissions[ref] is OmissionReason.UNSUPPORTED for ref in omitted_refs
    )


def test_simulation_leaves_unconsumed_optional_dissolution_flags_unloaded(
    tmp_path: Path,
) -> None:
    """Replay preflight materializes only flags a declared gate consumes."""
    source_model, params = make_two_stakeholder_model()
    restored = load_solution(
        path=save_solution(
            solution=source_model.solve(params=params, log_level="off"),
            path=tmp_path / "solution.lcm",
        )
    )
    flag_refs = tuple(
        ref for ref in restored.replay_artifacts if ref.key == DISSOLUTION_FLAG
    )
    assert flag_refs
    assert all(
        restored.replay_artifacts.load_state(ref) is LoadState.UNLOADED
        for ref in flag_refs
    )

    fresh_model, _fresh_params = make_two_stakeholder_model()
    fresh_model.simulate(
        params=params,
        initial_conditions=make_couple_initial_conditions(n_subjects=1),
        solution=restored,
        log_level="off",
        seed=0,
    )

    assert all(
        restored.replay_artifacts.load_state(ref) is LoadState.UNLOADED
        for ref in flag_refs
    )


def test_model_rejects_persisted_not_applicable_lie_before_materialization(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh model authority rejects an applicable archive cell's false omission."""
    _source_model, params, solution, omitted_refs = (
        _solve_without_optional_dissolution_flags(monkeypatch=monkeypatch)
    )
    assert omitted_refs
    tampered_ref = omitted_refs[0]
    path = save_solution(solution=solution, path=tmp_path / "solution.lcm")
    _replace_manifest_omission_reason(
        path=path,
        ref=tampered_ref,
        reason=OmissionReason.NOT_APPLICABLE,
    )

    restored = load_solution(path=path)
    assert isinstance(restored.values, ValueStore)
    model, _fresh_params = make_two_stakeholder_model()
    assert all(
        restored.values.load_state(period=period, regime=regime) is LoadState.UNLOADED
        for period in restored.values
        for regime in restored.values[period]
    )

    with pytest.raises(
        InvalidSimulationInputError,
        match=r"omissions.*not_applicable.*unsupported",
    ):
        model.simulate(
            params=params,
            initial_conditions=make_couple_initial_conditions(n_subjects=1),
            solution=restored,
            log_level="off",
            seed=0,
        )

    assert all(
        restored.values.load_state(period=period, regime=regime) is LoadState.UNLOADED
        for period in restored.values
        for regime in restored.values[period]
    )


def test_builtin_egm_continuation_roundtrips_as_an_independent_lazy_entry(
    tmp_path: Path,
) -> None:
    """All-persistable retention is broader than replay-only for EGM carries."""
    model = _two_period_bequest_model()
    solution = model.solve(
        params=get_retirement_only_params(n_periods=2, discount_factor=0.98),
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    ref = ArtifactRef(period=0, regime="retirement", key=EGM_CONTINUATION)
    path = save_solution(solution=solution, path=tmp_path / "egm-solution.lcm")

    restored = load_solution(path=path)

    assert ref in restored.retained_continuations
    assert restored.retained_continuations.load_state(ref) is LoadState.UNLOADED
    expected = jax.tree.leaves(solution.retained_continuations[ref])
    template = solution._artifact_authority[ref].template
    actual = jax.tree.leaves(
        restored.retained_continuations.materialize(ref, template=template)
    )
    assert len(actual) == len(expected)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    assert restored.retained_continuations.load_state(ref) is LoadState.LOADED


def test_finite_nnbegm_continuation_declares_its_stacked_candidate_axis() -> None:
    """The published spec describes the candidate axis added to its template."""
    model = _build("finite")
    spec = model._regimes["alive"].solution.continuation_spec

    assert isinstance(spec, EGMContinuationSpec)
    assert spec.layout.n_stacked_candidates > 0
    assert spec.template.endog_grid.shape[-2] == spec.layout.n_stacked_candidates


def test_finite_nnbegm_policy_roundtrips_lazily_into_a_fresh_model(
    tmp_path: Path,
) -> None:
    """A finite candidate bank is model-verifiable without solve-private facts."""
    source_model = _build("finite")
    solution = source_model.solve(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    policy_refs = _policy_refs(solution)
    descriptors = _artifact_descriptors_by_ref(solution)

    assert policy_refs
    assert all(
        type(solution.replay_artifacts[ref]) is NNBEGMSimPolicy for ref in policy_refs
    )
    assert all(
        descriptors[ref].persistence is PersistencePolicy.MODEL_VERIFIABLE
        for ref in policy_refs
    )

    expected = source_model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=solution,
        log_level="off",
        seed=_SEED,
    )
    path = save_solution(solution=solution, path=tmp_path / "finite-solution.lcm")
    restored = load_solution(path=path)

    assert set(restored.replay_artifacts) == set(policy_refs)
    assert all(
        restored.replay_artifacts.load_state(ref) is LoadState.UNLOADED
        for ref in policy_refs
    )

    # This model has never solved and therefore owns no solve-side replay cache. Its
    # declarations alone must reconstruct the finite policy's static PyTree fields.
    fresh_model = _build("finite")
    actual = fresh_model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=restored,
        log_level="off",
        seed=_SEED,
    )

    assert_frame_equal(actual.to_dataframe(), expected.to_dataframe())
    assert all(
        restored.replay_artifacts.load_state(ref) is LoadState.LOADED
        for ref in policy_refs
    )


def _candidate_axis(descriptor: ArtifactDescriptor) -> AxisDescriptor:
    """The axis along which a nested policy lists its outer candidate nodes."""
    return next(
        axis for axis in descriptor.named_axes if axis.role is AxisRole.CANDIDATE
    )


def _rewrite_manifest(*, path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Apply one in-place manifest edit while keeping the manifest checksum valid."""
    with h5py.File(path, "r+") as archive:
        manifest = cast(
            "dict[str, Any]",
            json.loads(bytes(archive["manifest"][()])),
        )
        mutate(manifest)
        manifest_bytes = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        del archive["manifest"]
        dataset = archive.create_dataset(
            "manifest",
            data=np.frombuffer(manifest_bytes, dtype=np.uint8),
        )
        dataset.attrs["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()


def _rewrite_candidate_axis(
    *, path: Path, ref: ArtifactRef, coordinates: list[object]
) -> None:
    """Replace one nested policy's declared outer nodes inside an archive."""

    def mutate(manifest: dict[str, Any]) -> None:
        descriptor = next(
            entry
            for entry in manifest["metadata"]["artifact_descriptors"]
            if entry["period"] == ref.period
            and entry["regime"] == ref.regime
            and entry["type_id"] == ref.key.type_id
        )
        axis = next(
            axis
            for axis in descriptor["named_axes"]
            if axis["role"] == AxisRole.CANDIDATE.value
        )
        axis["coordinates"] = coordinates
        axis["length"] = len(coordinates)

    _rewrite_manifest(path=path, mutate=mutate)


def test_adaptive_nnbegm_policy_declares_its_outer_nodes_in_its_descriptor() -> None:
    """The nodes an adaptive solve settled on are published as the candidate axis
    of the nested policy, and they are the nodes the policy's adjuster bank uses."""
    solution = _build("adaptive").solve(params=_PARAMS, log_level="off")
    descriptors = _artifact_descriptors_by_ref(solution)
    policy_refs = _policy_refs(solution)

    assert policy_refs
    for ref in policy_refs:
        assert descriptors[ref].persistence is PersistencePolicy.MODEL_VERIFIABLE
        policy = cast("NestedEGMSimPolicy", solution.replay_artifacts[ref])
        assert _candidate_axis(descriptors[ref]).coordinates == tuple(
            float(node) for node in np.asarray(policy.adjuster.outer_nodes)
        )


def test_adaptive_nnbegm_policy_roundtrips_into_a_fresh_model(
    tmp_path: Path,
) -> None:
    """A fresh equivalent model replays a restored adaptive result exactly as
    the producing model replays the in-memory one."""
    source_model = _build("adaptive")
    solution = source_model.solve(params=_PARAMS, log_level="off")
    policy_refs = _policy_refs(solution)
    path = save_solution(solution=solution, path=tmp_path / "adaptive-solution.lcm")
    restored = load_solution(path=path)
    assert set(policy_refs) <= set(restored.replay_artifacts)

    from_restored = _build("adaptive").simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=restored,
        log_level="off",
        seed=_SEED,
    )
    from_source = source_model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=solution,
        log_level="off",
        seed=_SEED,
    )

    assert_frame_equal(from_restored.to_dataframe(), from_source.to_dataframe())


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        (lambda nodes: list(reversed(nodes)), "strictly increasing"),
        (lambda nodes: [node + 1.0e6 for node in nodes], "outer state's domain"),
        (lambda nodes: [round(node) for node in nodes], "exact finite floats"),
        (lambda _nodes: [float(index) for index in range(600)], "node budget"),
    ],
    ids=["reversed", "outside_domain", "integer_typed", "over_budget"],
)
def test_declared_outer_nodes_that_no_mesh_could_produce_are_refused(
    *,
    tmp_path: Path,
    tamper: Callable[[list[float]], list[object]],
    match: str,
) -> None:
    """A consumer admits a result's outer nodes only when they are what a
    shared adaptive mesh can be: exact finite floats, strictly increasing,
    within the search's node budget, and inside the outer state's domain."""
    solution = _build("adaptive").solve(params=_PARAMS, log_level="off")
    ref = _policy_refs(solution)[0]
    nodes = [
        float(node)
        for node in _candidate_axis(
            _artifact_descriptors_by_ref(solution)[ref]
        ).coordinates
    ]
    path = save_solution(solution=solution, path=tmp_path / "adaptive-solution.lcm")
    _rewrite_candidate_axis(path=path, ref=ref, coordinates=tamper(nodes))
    restored = load_solution(path=path)

    with pytest.raises(InvalidSimulationInputError, match=match):
        _build("adaptive").simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=restored,
            log_level="off",
            seed=_SEED,
        )


def test_mixed_route_persistence_is_addressed_by_artifact_ref(
    tmp_path: Path,
) -> None:
    """A dynamic cell cannot overwrite a static sibling sharing its key.

    The actual finite and adaptive policies are exercised above. This smaller
    transport witness isolates the descriptor address: both cells deliberately use
    ``SIMULATION_POLICY``, while only the finite cell is persistable.
    """
    finite_ref = ArtifactRef(
        period=0,
        regime="finite",
        key=SIMULATION_POLICY,
    )
    adaptive_ref = ArtifactRef(
        period=0,
        regime="adaptive",
        key=SIMULATION_POLICY,
    )
    finite_descriptor = ArtifactDescriptor(
        key=SIMULATION_POLICY,
        channel=ArtifactChannel.REPLAY,
        persistence=PersistencePolicy.MODEL_VERIFIABLE,
        payload_type_id="jax.Array",
        leaf_descriptors=(
            LeafDescriptor(
                path=(),
                shape=(1,),
                dtype="float32",
                axis_names=("candidate",),
            ),
        ),
        named_axes=(
            AxisDescriptor(
                name="candidate",
                length=1,
                role=AxisRole.CANDIDATE,
                coordinates=(0,),
            ),
        ),
        required=True,
    )
    adaptive_descriptor = ArtifactDescriptor(
        key=SIMULATION_POLICY,
        channel=ArtifactChannel.REPLAY,
        persistence=PersistencePolicy.NOT_PERSISTED,
        payload_type_id="jax.Array",
        leaf_descriptors=(
            LeafDescriptor(
                path=(),
                shape=(1,),
                dtype="float32",
                axis_names=("adaptive_node",),
            ),
        ),
        named_axes=(
            AxisDescriptor(
                name="adaptive_node",
                length=1,
                role=AxisRole.CANDIDATE,
                coordinates=(0,),
            ),
        ),
        required=True,
    )
    values = {
        0: {
            "finite": jnp.asarray([1.0], dtype=jnp.float32),
            "adaptive": jnp.asarray([2.0], dtype=jnp.float32),
        }
    }
    solution = SolutionResult(
        values=values,
        replay_artifacts=ArtifactStore(
            {
                finite_ref: jnp.asarray([10.0], dtype=jnp.float32),
                adaptive_ref: jnp.asarray([20.0], dtype=jnp.float32),
            }
        ),
        metadata=SolutionMetadata(
            retention=ResultRetention.VALUES_AND_REPLAY,
            n_periods=1,
            regime_names=("finite", "adaptive"),
            solver_types={
                "finite": "example.FiniteNNBEGM",
                "adaptive": "example.AdaptiveNNBEGM",
            },
            model_instance_id="mixed-route-model",
            params_fingerprint="0" * 64,
            value_schemas={
                (0, regime): ValueArraySchema(
                    shape=(1,),
                    dtype="float32",
                    axis_names=("wealth",),
                )
                for regime in ("finite", "adaptive")
            },
            artifact_descriptors=cast(
                "Mapping",
                {
                    finite_ref: finite_descriptor,
                    adaptive_ref: adaptive_descriptor,
                },
            ),
        ),
    )
    object.__setattr__(
        solution,
        "_artifact_authority",
        MappingProxyType(
            {
                finite_ref: ArtifactAuthority(
                    descriptor=finite_descriptor,
                    payload_runtime_type=jax.Array,
                    template=jnp.zeros((1,), dtype=jnp.float32),
                    leaves={
                        (): LeafAuthority(
                            path=(),
                            runtime_type=jax.Array,
                            shape=(1,),
                            dtype="float32",
                            axis_names=("candidate",),
                        )
                    },
                    axes=(
                        AxisAuthority(
                            name="candidate",
                            length=1,
                            role=AxisRole.CANDIDATE,
                            coordinates=(0,),
                        ),
                    ),
                    required=True,
                ),
                adaptive_ref: ArtifactAuthority(
                    descriptor=adaptive_descriptor,
                    payload_runtime_type=jax.Array,
                    template=jnp.zeros((1,), dtype=jnp.float32),
                    leaves={
                        (): LeafAuthority(
                            path=(),
                            runtime_type=jax.Array,
                            shape=(1,),
                            dtype="float32",
                            axis_names=("adaptive_node",),
                        )
                    },
                    axes=(
                        AxisAuthority(
                            name="adaptive_node",
                            length=1,
                            role=AxisRole.CANDIDATE,
                            coordinates=(0,),
                        ),
                    ),
                    required=True,
                ),
            }
        ),
    )

    path = save_solution(solution=solution, path=tmp_path / "mixed-solution.lcm")
    restored = load_solution(path=path)
    restored_descriptors = _artifact_descriptors_by_ref(restored)

    assert set(restored_descriptors) == {finite_ref, adaptive_ref}
    assert set(restored.replay_artifacts) == {finite_ref}
    assert adaptive_ref not in restored.replay_artifacts
    assert restored.omissions[adaptive_ref] is OmissionReason.NOT_PERSISTED


def _diagnostics_refs(solution: SolutionResult) -> tuple[ArtifactRef, ...]:
    """Return every retained solver-diagnostics address."""
    return tuple(ref for ref in solution.diagnostics if ref.key == SOLVER_DIAGNOSTICS)


def _assert_same_diagnostics(*, actual: object, expected: object) -> None:
    """Compare two diagnostics payloads field by field, `None` included."""
    assert type(actual) is SolverDiagnostics
    assert type(expected) is SolverDiagnostics
    for field in dataclasses.fields(SolverDiagnostics):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if expected_value is None:
            assert actual_value is None, field.name
            continue
        assert actual_value is not None, field.name
        assert actual_value.dtype == expected_value.dtype, field.name
        np.testing.assert_array_equal(
            np.asarray(actual_value), np.asarray(expected_value), err_msg=field.name
        )


def test_solver_diagnostics_are_described_as_model_verifiable_artifacts() -> None:
    """A solve that keeps diagnostics describes each payload it retains."""
    solution = _build("adaptive").solve(params=_PARAMS, log_level="warning")
    refs = _diagnostics_refs(solution)
    assert refs
    for ref in refs:
        descriptor = solution.metadata.artifact_descriptors[ref]
        assert descriptor.channel is ArtifactChannel.DIAGNOSTIC
        assert descriptor.persistence is PersistencePolicy.MODEL_VERIFIABLE
        assert {leaf.path[0] for leaf in descriptor.leaf_descriptors} <= {
            f"attribute:{field.name}" for field in dataclasses.fields(SolverDiagnostics)
        }


def test_solver_diagnostics_survive_an_archive_roundtrip(tmp_path: Path) -> None:
    """Restored diagnostics read back as the arrays the solve published."""
    solution = _build("adaptive").solve(params=_PARAMS, log_level="warning")
    restored = load_solution(
        path=save_solution(solution=solution, path=tmp_path / "diagnosed.lcm")
    )

    assert set(_diagnostics_refs(restored)) == set(_diagnostics_refs(solution))
    for ref in _diagnostics_refs(solution):
        _assert_same_diagnostics(
            actual=restored.diagnostics[ref], expected=solution.diagnostics[ref]
        )


def test_a_restored_result_with_diagnostics_replays_in_a_fresh_model(
    tmp_path: Path,
) -> None:
    """Diagnostics in a restored result neither block nor alter its replay."""
    source_model = _build("adaptive")
    solution = source_model.solve(params=_PARAMS, log_level="warning")
    restored = load_solution(
        path=save_solution(solution=solution, path=tmp_path / "diagnosed.lcm")
    )

    from_restored = _build("adaptive").simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=restored,
        log_level="off",
        seed=_SEED,
    )
    from_source = source_model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=solution,
        log_level="off",
        seed=_SEED,
    )

    assert_frame_equal(from_restored.to_dataframe(), from_source.to_dataframe())


def _with_diagnostics_leaves(
    *,
    solution: SolutionResult,
    ref: ArtifactRef,
    rewrite: Callable[[tuple[LeafDescriptor, ...]], tuple[LeafDescriptor, ...]],
) -> SolutionResult:
    """Return a copy whose diagnostics descriptor at `ref` has rewritten leaves."""
    descriptor = solution.metadata.artifact_descriptors[ref]
    rewritten = dataclasses.replace(
        descriptor, leaf_descriptors=rewrite(descriptor.leaf_descriptors)
    )
    return dataclasses.replace(
        solution,
        metadata=dataclasses.replace(
            solution.metadata,
            artifact_descriptors={
                **solution.metadata.artifact_descriptors,
                ref: rewritten,
            },
        ),
    )


def _mask_leaf_as_float(
    leaves: tuple[LeafDescriptor, ...],
) -> tuple[LeafDescriptor, ...]:
    return tuple(
        dataclasses.replace(leaf, dtype="float64")
        if leaf.path == ("attribute:unresolved_mask",)
        else leaf
        for leaf in leaves
    )


def _drop_required_leaf(
    leaves: tuple[LeafDescriptor, ...],
) -> tuple[LeafDescriptor, ...]:
    return tuple(
        leaf for leaf in leaves if leaf.path != ("attribute:outer_nodes_used",)
    )


def _add_unknown_leaf(
    leaves: tuple[LeafDescriptor, ...],
) -> tuple[LeafDescriptor, ...]:
    return (
        *leaves,
        LeafDescriptor(
            path=("attribute:regret",), shape=(), dtype="float64", axis_names=()
        ),
    )


@pytest.mark.parametrize(
    ("rewrite", "message"),
    [
        pytest.param(_mask_leaf_as_float, "dtype", id="mask_declared_as_float"),
        pytest.param(_drop_required_leaf, "missing", id="required_field_absent"),
        pytest.param(_add_unknown_leaf, "unknown", id="unknown_field"),
    ],
)
def test_diagnostics_that_no_solver_could_publish_are_refused(
    *,
    rewrite: Callable[[tuple[LeafDescriptor, ...]], tuple[LeafDescriptor, ...]],
    message: str,
) -> None:
    """A diagnostics descriptor outside the published schema is refused."""
    model = _build("adaptive")
    solution = model.solve(params=_PARAMS, log_level="warning")
    ref = _diagnostics_refs(solution)[0]
    tampered = _with_diagnostics_leaves(solution=solution, ref=ref, rewrite=rewrite)

    with pytest.raises(InvalidSimulationInputError, match=message):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=tampered,
            log_level="off",
            seed=_SEED,
        )


def test_a_restored_result_saves_to_an_equivalent_archive(tmp_path: Path) -> None:
    """Saving a restored result writes an archive that reads and replays alike."""
    solution = _build("adaptive").solve(
        params=_PARAMS,
        log_level="warning",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    first = load_solution(
        path=save_solution(solution=solution, path=tmp_path / "first.lcm")
    )
    second = load_solution(path=first.save(path=tmp_path / "second.lcm"))

    assert second.metadata == first.metadata
    assert second.omissions == first.omissions
    assert set(second.retained_continuations) == set(first.retained_continuations)
    assert set(second.replay_artifacts) == set(first.replay_artifacts)
    assert set(second.auxiliary_artifacts) == set(first.auxiliary_artifacts)
    assert set(second.diagnostics) == set(first.diagnostics)
    for period, regime_to_value in first.values.items():
        for regime, value in regime_to_value.items():
            np.testing.assert_array_equal(
                np.asarray(second.values[period][regime]), np.asarray(value)
            )
    for ref in _diagnostics_refs(first):
        _assert_same_diagnostics(
            actual=second.diagnostics[ref], expected=first.diagnostics[ref]
        )
    from_second = _build("adaptive").simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=second,
        log_level="off",
        seed=_SEED,
    )
    from_first = _build("adaptive").simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=first,
        log_level="off",
        seed=_SEED,
    )
    assert_frame_equal(from_second.to_dataframe(), from_first.to_dataframe())
