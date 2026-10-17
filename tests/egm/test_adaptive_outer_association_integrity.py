"""Completed adaptive meshes retain their canonical outer associations."""

from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.adaptive_mesh_authority import (
    ACTIVE_CONTINUATION_MESH_AXIS,
    validate_continuation_mesh,
)
from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import EGMAdaptiveOuterData
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.solution.model_authority import SolutionAuthority
from lcm import Model
from lcm.exceptions import InvalidSimulationInputError
from lcm.persistence import load_solution
from lcm.solver_api import (
    EGM_CONTINUATION,
    ArtifactRef,
    ArtifactStore,
    ResultRetention,
    SolutionResult,
)
from tests.egm.test_adaptive_discrete_profiles import _adaptive_model
from tests.egm.test_adaptive_raw_action_records import _record_producers

type Mutation = Literal["active_owner", "active_side", "inactive_owner"]


@dataclass(frozen=True, kw_only=True)
class _Produced:
    model: Model
    solution: SolutionResult
    restored: SolutionResult
    authority: SolutionAuthority
    ref: ArtifactRef
    carry: EGMCarry
    nodes: tuple[float, ...]
    capacity: int


def _mesh(*, carry: EGMCarry) -> EGMAdaptiveOuterData:
    assert carry.query_data is not None
    mesh = carry.query_data.adaptive_outer_data
    assert mesh is not None
    return mesh


@pytest.fixture(scope="module")
def produced(tmp_path_factory: pytest.TempPathFactory) -> _Produced:
    """One real public solve supplies all immutable mutation controls."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        records = _record_producers(monkeypatch=monkeypatch)
        model = _adaptive_model(callable_floor=True)
        solution = model.solve(
            params={"discount_factor": 0.95, "alive": {"premium": 1.0}},
            log_level="off",
            retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
        )
    assert set(records.outputs) == {0}
    assert len(records.banks) == 1
    ref = ArtifactRef(period=0, regime="alive", key=EGM_CONTINUATION)
    carry = records.outputs[0].continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    retained = solution.retained_continuations[ref]
    assert isinstance(retained, EGMCarry)
    for path, original in carry.leaves().items():
        np.testing.assert_array_equal(retained.leaves()[path], original)
    engine_view = solution._engine_view
    assert isinstance(engine_view, OwnedSolutionView)
    authority = engine_view.authority
    descriptor = authority.artifacts[ref]
    axes = tuple(
        axis for axis in descriptor.axes if axis.name == ACTIVE_CONTINUATION_MESH_AXIS
    )
    assert len(axes) == 1
    coordinates = axes[0].coordinates
    assert all(type(node) is float for node in coordinates)
    nodes = cast("tuple[float, ...]", coordinates)
    mesh = _mesh(carry=carry)
    count = int(mesh.outer_count)
    capacity = mesh.outer_nodes.size
    query_authority = model._regimes["alive"].solution.query_authority
    assert query_authority is not None
    assert query_authority.outer_mode == "adaptive_continuous"
    assert query_authority.action_names == ("buy_private",)
    assert query_authority.state_names == ("illiquid",)
    assert query_authority.adaptive_outer is not None
    assert capacity == query_authority.adaptive_outer.adjuster_capacity == 513
    assert 2 <= count <= capacity
    assert count == len(nodes) == records.banks[0].n_candidates
    np.testing.assert_array_equal(records.banks[0].outer_nodes, nodes)
    expected_owner = np.where(np.arange(capacity) < count, np.arange(capacity), -1)
    np.testing.assert_array_equal(
        mesh.outer_owner_id,
        np.broadcast_to(expected_owner, mesh.outer_owner_id.shape),
    )
    np.testing.assert_array_equal(
        mesh.outer_side_id, np.zeros(mesh.outer_side_id.shape, dtype=np.int32)
    )
    validate_continuation_mesh(carry=carry, nodes=nodes, capacity=capacity)
    archive: Path = solution.save(
        path=tmp_path_factory.mktemp("adaptive-association") / "genuine.lcm"
    )
    restored = load_solution(path=archive)
    assert restored._engine_view is None
    restored_carry = restored.retained_continuations.materialize(ref, template=carry)
    assert isinstance(restored_carry, EGMCarry)
    for path, original in carry.leaves().items():
        np.testing.assert_array_equal(restored_carry.leaves()[path], original)
    return _Produced(
        model=model,
        solution=solution,
        restored=restored,
        authority=authority,
        ref=ref,
        carry=carry,
        nodes=nodes,
        capacity=capacity,
    )


def _mutated(*, produced: _Produced, mutation: Mutation) -> EGMCarry:
    mesh = _mesh(carry=produced.carry)
    count = int(mesh.outer_count)
    if mutation == "inactive_owner":
        assert count < produced.capacity, (
            "Inactive-owner qualification requires genuine padding."
        )
        index = (0,) * (mesh.outer_owner_id.ndim - 1) + (count,)
        owner = mesh.outer_owner_id.at[index].set(jnp.int32(0))
        changed = replace(mesh, outer_owner_id=owner)
    elif mutation == "active_owner":
        index = (0,) * mesh.outer_owner_id.ndim
        owner = mesh.outer_owner_id.at[index].set(jnp.int32(1))
        changed = replace(mesh, outer_owner_id=owner)
    else:
        index = (0,) * mesh.outer_side_id.ndim
        side = mesh.outer_side_id.at[index].set(jnp.int32(1))
        changed = replace(mesh, outer_side_id=side)
    assert produced.carry.query_data is not None
    result = replace(
        produced.carry,
        query_data=replace(produced.carry.query_data, adaptive_outer_data=changed),
    )
    changed_name = "outer_side_id" if mutation == "active_side" else "outer_owner_id"
    changed_path = ("query_data", "adaptive_outer_data", changed_name)
    for path, original in produced.carry.leaves().items():
        actual = result.leaves()[path]
        assert actual.shape == original.shape
        assert actual.dtype == original.dtype
        if path == changed_path:
            assert np.count_nonzero(np.asarray(actual) != np.asarray(original)) == 1
        else:
            np.testing.assert_array_equal(actual, original)
    return result


@pytest.mark.parametrize("mutation", ["active_owner", "active_side", "inactive_owner"])
def test_publication_validator_rejects_changed_outer_association(
    *,
    produced: _Produced,
    mutation: Mutation,
) -> None:
    """Native outer labels must agree with the independently admitted mesh."""
    validate_continuation_mesh(
        carry=produced.carry,
        nodes=produced.nodes,
        capacity=produced.capacity,
    )
    changed = _mutated(produced=produced, mutation=mutation)
    with pytest.raises(ValueError, match="Adaptive continuation"):
        validate_continuation_mesh(
            carry=changed,
            nodes=produced.nodes,
            capacity=produced.capacity,
        )


@pytest.mark.parametrize("mutation", ["active_owner", "active_side", "inactive_owner"])
def test_restored_preflight_rejects_changed_outer_association(
    *,
    produced: _Produced,
    mutation: Mutation,
) -> None:
    """Archive-backed continuation labels retain the producer's canonical identities."""
    produced.model._check_solution_result_artifacts(
        solution=produced.restored,
        authority=produced.authority,
        values=MappingProxyType(
            {
                period: MappingProxyType(dict(rows))
                for period, rows in produced.restored.values.items()
            }
        ),
    )
    changed = _mutated(produced=produced, mutation=mutation)
    entries = {
        ref: produced.restored.retained_continuations.materialize(
            ref, template=produced.solution.retained_continuations[ref]
        )
        for ref in produced.restored.retained_continuations
    }
    entries[produced.ref] = changed
    substituted = replace(
        produced.restored,
        retained_continuations=ArtifactStore(entries),
    )
    assert substituted._engine_view is None
    assert substituted.metadata is produced.restored.metadata
    with pytest.raises(
        InvalidSimulationInputError, match="mismatched continuation mesh"
    ):
        produced.model._check_solution_result_artifacts(
            solution=substituted,
            authority=produced.authority,
            values=MappingProxyType(
                {
                    period: MappingProxyType(dict(rows))
                    for period, rows in substituted.values.items()
                }
            ),
        )


@pytest.mark.parametrize("field", ["outer_owner_id", "outer_side_id"])
@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_outer_association_metadata_matches_its_actual_carry_schema(
    *,
    produced: _Produced,
    field: str,
    change: str,
) -> None:
    """Canonical outer labels retain exact shape and int32 dtype beside their bank."""
    mesh = _mesh(carry=produced.carry)
    original = getattr(mesh, field)
    changed = original[..., :-1] if change == "shape" else original.astype(jnp.int16)
    assert changed.shape != original.shape or changed.dtype != original.dtype
    if change == "shape":
        assert changed.dtype == original.dtype
        np.testing.assert_array_equal(changed, original[..., :-1])
    else:
        assert changed.shape == original.shape
        np.testing.assert_array_equal(changed, original)
    if change == "dtype":
        # Corrupt a decoded payload after its constructor validation.
        changed_mesh = copy(mesh)
        object.__setattr__(changed_mesh, field, changed)
    else:
        changed_mesh = replace(mesh, **{field: changed})
    assert produced.carry.query_data is not None
    changed_carry = replace(
        produced.carry,
        query_data=replace(produced.carry.query_data, adaptive_outer_data=changed_mesh),
    )
    with pytest.raises(ValueError, match="Adaptive continuation"):
        validate_continuation_mesh(
            carry=changed_carry,
            nodes=produced.nodes,
            capacity=produced.capacity,
        )


def test_query_only_adaptive_carry_needs_no_undeclared_exact_policy(
    *,
    produced: _Produced,
) -> None:
    """A restored conditional carry is valid when its model declares no exact replay."""
    from lcm.solver_api import SIMULATION_POLICY  # noqa: PLC0415

    ref = ArtifactRef(
        period=produced.ref.period, regime=produced.ref.regime, key=SIMULATION_POLICY
    )
    route = produced.authority.replay[ref]
    assert route.applicable is False
    assert route.required is False
    assert route.payload_type is None
    assert ref not in produced.authority.artifacts
    assert ref not in produced.restored.metadata.artifact_descriptors
    assert ref not in produced.restored.replay_artifacts
    assert ref not in produced.restored.omissions
    produced.model._check_solution_result_artifacts(
        solution=produced.restored,
        authority=produced.authority,
        values=MappingProxyType(
            {
                period: MappingProxyType(dict(rows))
                for period, rows in produced.restored.values.items()
            }
        ),
    )
