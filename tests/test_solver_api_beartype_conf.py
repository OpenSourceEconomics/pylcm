from dataclasses import replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.typing import ArtifactPayload
from lcm._solver_api.contract import _same_inert_pytree_metadata
from lcm._solver_api.identity import ArtifactValue, InertMetadata
from lcm.exceptions import SolverAPITypeError
from lcm.solver_api import (
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    ValueStore,
)
from lcm.typing import FloatND, IntND


def _stores() -> tuple[ValueStore, ArtifactStore, ArtifactRef]:
    ref = ArtifactRef(
        period=0, regime="alive", key=ArtifactKey(type_id="example.policy")
    )
    values = ValueStore({(0, "alive"): jnp.asarray([1.0])})
    artifacts = ArtifactStore({ref: cast("ArtifactPayload", object())})
    return values, artifacts, ref


def test_a_wrongly_typed_store_coordinate_is_the_solver_api_error() -> None:
    """A store read with a wrongly typed coordinate raises `SolverAPITypeError`."""
    values, _, _ = _stores()
    with pytest.raises(SolverAPITypeError, match=r"parameter period="):
        _ = values[cast("int", "0")]


_WRONGLY_TYPED_REF_FIELDS = [("period", "0"), ("regime", 0), ("key", "example")]


@pytest.mark.parametrize(("field", "value"), _WRONGLY_TYPED_REF_FIELDS)
def test_artifact_store_membership_is_false_for_a_wrongly_typed_ref_field(
    *, field: str, value: str | int
) -> None:
    """Membership of an exact `ArtifactRef` holding a wrongly typed field is False."""
    _, artifacts, ref = _stores()
    hostile = replace(ref)
    object.__setattr__(hostile, field, value)
    assert hostile not in artifacts


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


_NODES: MappingProxyType[str, FloatND | IntND] = MappingProxyType(
    {"wealth": jnp.asarray([0.0, 1.0])}
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
