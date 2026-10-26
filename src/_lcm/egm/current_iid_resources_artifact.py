"""Closed numerical-leaf storage for the original model-owned resources row."""

import jax

from _lcm.egm.step_core import OwnedResourcesRow
from _lcm.execution.core_program import ValueRead
from _lcm.simulation.replay_inputs import _payload_read
from _lcm.typing import ArtifactPayload
from lcm.exceptions import InvalidSimulationInputError
from lcm.solver_api import ArtifactKey
from lcm.typing import BoolND, FloatND, IntND, RegimeName

type ResourcesRowLeaf = FloatND | IntND | BoolND


def resources_row_payload(*, row: OwnedResourcesRow) -> tuple[ResourcesRowLeaf, ...]:
    """Keep original array identities and canonical numerical leaf order."""
    leaves = jax.tree.leaves(row)
    if any(not isinstance(leaf, jax.Array) for leaf in leaves):
        raise TypeError("Original resources rows require numerical array leaves.")
    return tuple(leaves)


def resources_row_leaf_specs(
    *, row: OwnedResourcesRow
) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Retain only the original model-owned shape and dtype declarations."""
    leaves = jax.tree.leaves(row)
    if any(not isinstance(leaf, jax.Array | jax.ShapeDtypeStruct) for leaf in leaves):
        raise TypeError("Original row schemas require actual array descriptors.")
    return tuple(jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for leaf in leaves)


def restore_resources_row(
    *,
    payload: ArtifactPayload,
    tree: jax.tree_util.PyTreeDef,
    leaf_specs: tuple[jax.ShapeDtypeStruct, ...],
) -> OwnedResourcesRow:
    """Restore the genuine row only under its original model-owned schema."""
    if (
        type(payload) is not tuple
        or len(payload) != len(leaf_specs)
        or tree.num_leaves != len(leaf_specs)
        or tree.node_data() is None
        or tree.node_data()[0] is not OwnedResourcesRow
    ):
        raise InvalidSimulationInputError("Case-R replay has a foreign row schema.")
    for leaf, spec in zip(payload, leaf_specs, strict=True):
        if (
            not isinstance(leaf, jax.Array)
            or leaf.shape != spec.shape
            or leaf.dtype != spec.dtype
        ):
            raise InvalidSimulationInputError(
                "Case-R replay has a foreign row leaf shape or dtype."
            )
    row = jax.tree.unflatten(tree, payload)
    if type(row) is not OwnedResourcesRow or any(
        original is not restored
        for original, restored in zip(
            payload, resources_row_payload(row=row), strict=True
        )
    ):
        raise InvalidSimulationInputError("Case-R replay lost original row buffers.")
    return row


def resources_row_reads(
    *,
    row: OwnedResourcesRow,
    key: ArtifactKey,
    period: int,
    regime: RegimeName,
    core: str,
) -> tuple[ValueRead, ...]:
    """Address the actual stored leaf tuple passed through compiler transfer."""
    payload = tuple(jax.tree.leaves(row))
    reads = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(payload)[0]:
        if not isinstance(leaf, jax.Array | jax.ShapeDtypeStruct):
            raise TypeError("Case-R leaf addresses require original numerical leaves.")
        reads.append(
            _payload_read(
                key=key,
                period=period,
                regime=regime,
                core=core,
                path=path,
                argument="row",
            )
        )
    return tuple(reads)
