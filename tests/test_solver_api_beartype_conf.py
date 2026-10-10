from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.typing import ArtifactPayload
from lcm._solver_api.beartype_conf import SOLVER_API_CONF
from lcm._solver_api.contract import (
    _same_inert_pytree_metadata,
    _snapshot_inert_pytree_metadata,
)
from lcm._solver_api.identity import ArtifactValue, InertMetadata
from lcm.exceptions import SolverAPITypeError
from lcm.solver_api import (
    ArtifactAuthority,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    LoadState,
    ReplayModelContext,
    ReplayRouteSnapshot,
    SimulationBuildContext,
    ValueStore,
)
from lcm.typing import FloatND, IntND
from tests.solution.test_public_mapping_admission import _metadata


@beartype(conf=SOLVER_API_CONF)
def _echo_int(*, value: int) -> int:
    return value


@beartype(conf=SOLVER_API_CONF)
def _misreport_int(*, value: int) -> int:
    return str(value)  # ty: ignore[invalid-return-type]


def test_solver_api_conf_reports_a_wrong_argument_as_a_type_error() -> None:
    """A wrongly typed argument at a solver-API boundary is a `TypeError`."""
    with pytest.raises(TypeError, match="value"):
        _echo_int(value="1")  # ty: ignore[invalid-argument-type]


def test_solver_api_conf_reports_a_wrong_return_as_the_solver_api_error() -> None:
    """A wrongly typed return at a solver-API boundary is a `SolverAPITypeError`."""
    with pytest.raises(SolverAPITypeError):
        _misreport_int(value=1)


# One store read; each read in these tests raises before it returns.
type _StoreRead = Callable[
    [ValueStore, ArtifactStore],
    FloatND
    | LoadState
    | Mapping[str, FloatND]
    | Mapping[int, Mapping[str, ArtifactPayload]],
]


def _stores() -> tuple[ValueStore, ArtifactStore, ArtifactRef]:
    ref = ArtifactRef(
        period=0, regime="alive", key=ArtifactKey(type_id="example.policy")
    )
    values = ValueStore({(0, "alive"): jnp.asarray([1.0])})
    artifacts = ArtifactStore({ref: cast("ArtifactPayload", object())})
    return values, artifacts, ref


@pytest.mark.parametrize(
    ("read", "parameter"),
    [
        (lambda values, _artifacts: values["0"], "period"),
        (lambda values, _artifacts: values[0][1], "regime"),
        (lambda _values, artifacts: artifacts.load_state("ref"), "ref"),
        (lambda _values, artifacts: artifacts.project("key"), "key"),
    ],
    ids=["value-period", "value-regime", "artifact-ref", "artifact-key"],
)
def test_store_reads_report_a_wrongly_typed_coordinate_as_the_solver_api_error(
    *,
    read: _StoreRead,
    parameter: str,
) -> None:
    """A store read with a wrongly typed coordinate names it in a `TypeError`."""
    values, artifacts, _ = _stores()
    with pytest.raises(SolverAPITypeError, match=rf"parameter {parameter}="):
        read(values, artifacts)


_WRONGLY_TYPED_REF_FIELDS = [("period", "0"), ("regime", 0), ("key", "example")]


@pytest.mark.parametrize(("field", "value"), _WRONGLY_TYPED_REF_FIELDS)
def test_artifact_store_load_state_names_a_wrongly_typed_ref_field(
    *, field: str, value: str | int
) -> None:
    """An exact `ArtifactRef` holding a wrongly typed field is refused by name."""
    _, artifacts, ref = _stores()
    hostile = replace(ref)
    object.__setattr__(hostile, field, value)
    with pytest.raises(SolverAPITypeError, match=rf"parameter {field}="):
        artifacts.load_state(hostile)


@pytest.mark.parametrize(("field", "value"), _WRONGLY_TYPED_REF_FIELDS)
def test_artifact_store_membership_is_false_for_a_wrongly_typed_ref_field(
    *, field: str, value: str | int
) -> None:
    """Membership of an exact `ArtifactRef` holding a wrongly typed field is False."""
    _, artifacts, ref = _stores()
    hostile = replace(ref)
    object.__setattr__(hostile, field, value)
    assert hostile not in artifacts


@dataclass(frozen=True)
class _UnregisteredMetadata:
    label: str


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (cast("ArtifactValue", object()), 1),
        (cast("ArtifactValue", [1]), (1,)),
        (jax.tree_util.tree_structure({"a": 1}).node_data()[1], ("a",)),
        (1.0, 1),
    ],
    ids=["plugin-object", "list-for-tuple", "pytree-node-data", "float-for-int"],
)
def test_inert_metadata_comparison_is_false_for_a_value_of_another_exact_type(
    *, actual: ArtifactValue, expected: InertMetadata
) -> None:
    """A static value of another exact type compares unequal to the metadata."""
    assert _same_inert_pytree_metadata(actual=actual, expected=expected) is False


def test_inert_metadata_comparison_refuses_an_unregistered_record_as_a_type_error() -> (
    None
):
    """An unregistered record of the expected class raises a plain `TypeError`."""
    with pytest.raises(TypeError, match="escaped validation"):
        _same_inert_pytree_metadata(
            actual=cast("ArtifactValue", _UnregisteredMetadata(label="a")),
            expected=cast("InertMetadata", _UnregisteredMetadata(label="a")),
        )


def test_claw_checks_the_artifact_contract_module() -> None:
    """The package claw refuses a wrongly typed argument in the contract module."""
    with pytest.raises(BeartypeCallHintParamViolation):
        _snapshot_inert_pytree_metadata(
            value=1, active_ids=cast("set[int]", "not a set")
        )


_NODES: MappingProxyType[str, FloatND | IntND] = MappingProxyType(
    {"wealth": jnp.asarray([0.0, 1.0])}
)
_NO_ARTIFACTS: MappingProxyType[ArtifactKey, ArtifactPayload] = MappingProxyType({})
_NO_AUTHORITIES: MappingProxyType[ArtifactKey, ArtifactAuthority] = MappingProxyType({})


def test_replay_route_snapshot_refuses_a_plain_dict_of_artifacts() -> None:
    """The engine hands a route its snapshot artifacts as a read-only mapping."""
    with pytest.raises(BeartypeCallHintParamViolation, match="parameter artifacts="):
        ReplayRouteSnapshot(
            artifacts={},  # ty: ignore[invalid-argument-type]
            authorities=_NO_AUTHORITIES,
            metadata=_metadata(),
        )


def test_replay_route_snapshot_refuses_a_plain_dict_of_authorities() -> None:
    """The engine hands a route its snapshot authorities as a read-only mapping."""
    with pytest.raises(BeartypeCallHintParamViolation, match="parameter authorities="):
        ReplayRouteSnapshot(
            artifacts=_NO_ARTIFACTS,
            authorities={},  # ty: ignore[invalid-argument-type]
            metadata=_metadata(),
        )


def test_replay_model_context_refuses_a_plain_dict_of_state_nodes() -> None:
    """The engine hands a route its state nodes as a read-only mapping."""
    with pytest.raises(BeartypeCallHintParamViolation, match="parameter state_nodes="):
        ReplayModelContext(
            regime_name="alive",
            period=0,
            state_names=("wealth",),
            action_names=(),
            state_nodes=dict(_NODES),  # ty: ignore[invalid-argument-type]
            action_nodes=MappingProxyType({}),
        )


def test_replay_model_context_refuses_a_plain_dict_of_action_nodes() -> None:
    """The engine hands a route its action nodes as a read-only mapping."""
    with pytest.raises(BeartypeCallHintParamViolation, match="parameter action_nodes="):
        ReplayModelContext(
            regime_name="alive",
            period=0,
            state_names=(),
            action_names=("wealth",),
            state_nodes=MappingProxyType({}),
            action_nodes=dict(_NODES),  # ty: ignore[invalid-argument-type]
        )


def test_simulation_build_context_refuses_a_plain_dict_of_state_nodes() -> None:
    """The engine hands a reader builder its state nodes as a read-only mapping."""
    with pytest.raises(BeartypeCallHintParamViolation, match="parameter state_nodes="):
        SimulationBuildContext(
            period=0,
            regime_name="alive",
            state_names=("wealth",),
            action_names=(),
            state_nodes=dict(_NODES),  # ty: ignore[invalid-argument-type]
            action_nodes=MappingProxyType({}),
        )


def test_simulation_build_context_refuses_a_plain_dict_of_action_nodes() -> None:
    """The engine hands a reader builder its action nodes as a read-only mapping."""
    with pytest.raises(BeartypeCallHintParamViolation, match="parameter action_nodes="):
        SimulationBuildContext(
            period=0,
            regime_name="alive",
            state_names=(),
            action_names=("wealth",),
            state_nodes=MappingProxyType({}),
            action_nodes=dict(_NODES),  # ty: ignore[invalid-argument-type]
        )


def test_placed_build_context_nodes_stay_read_only_mappings() -> None:
    """Device placement of a context's nodes returns the read-only mapping type."""
    placed = place_simulation_arguments(
        arguments={"state_nodes": _NODES},
        subject_arg_names=(),
        value_reads=(),
        devices=(jax.devices()[0],),
    )

    assert type(placed["state_nodes"]) is MappingProxyType
