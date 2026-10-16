"""Bind completed continuation meshes independently of simulation replay."""

import itertools
import math
from typing import cast

import numpy as np

from _lcm.continuation import EGMQuerySpec
from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import QueryNodeKind
from lcm.solver_api import ArtifactAuthority

ACTIVE_CONTINUATION_MESH_AXIS = "pylcm:egm:active_outer_mesh"
_MINIMUM_MESH_NODES = 2


def admit_continuation_mesh(
    *, nodes: tuple[bool | int | float | str, ...], spec: EGMQuerySpec
) -> tuple[float, ...]:
    """Check declared solution-owned coordinates against canonical mesh limits."""
    adaptive = spec.adaptive_outer
    if adaptive is None or spec.outer_mode != "adaptive_continuous":
        raise TypeError("A continuation mesh requires its canonical adaptive operator.")
    if type(nodes) is not tuple or any(
        type(node) is not float or not math.isfinite(node) for node in nodes
    ):
        raise TypeError("Continuation mesh coordinates must be exact finite floats.")
    admitted = cast("tuple[float, ...]", nodes)
    if not _MINIMUM_MESH_NODES <= len(admitted) <= adaptive.adjuster_capacity:
        raise ValueError(
            "Continuation mesh coordinates exceed the canonical active prefix."
        )
    if any(right <= left for left, right in itertools.pairwise(admitted)):
        raise ValueError("Continuation mesh coordinates must be strictly increasing.")
    identity = adaptive.mesh_authority_identity
    if type(identity) is not tuple or not identity or identity[0] != "adaptive_nbegm":
        raise TypeError("Continuation mesh lacks its canonical search declaration.")
    initial = identity[-1]
    if (
        type(initial) is not tuple
        or not initial
        or any(type(node) is not float for node in initial)
    ):
        raise TypeError("Continuation mesh declaration lacks its initial domain.")
    if nodes[0] != initial[0] or nodes[-1] != initial[-1]:
        raise ValueError(
            "Continuation refinement must preserve the initial mesh endpoints."
        )
    if not set(initial).issubset(nodes):
        raise ValueError(
            "Continuation refinement cannot discard an initial exact node."
        )
    return admitted


def validate_continuation_mesh(
    *,
    carry: EGMCarry,
    nodes: tuple[float, ...],
    capacity: int,
) -> None:
    """Compare numeric metadata to separately admitted generated coordinates."""
    data = carry.query_data
    mesh = None if data is None else data.adaptive_outer_data
    if mesh is None:
        raise TypeError("An adaptive continuation lacks its completed mesh metadata.")
    actual_nodes = np.asarray(mesh.outer_nodes)
    count = np.asarray(mesh.outer_count)
    if (
        count.shape != ()
        or count.dtype != np.dtype("int32")
        or int(count) != len(nodes)
    ):
        raise ValueError(
            "Adaptive continuation count differs from its bound mesh authority."
        )
    expected = np.asarray(
        (*nodes, *((nodes[0],) * (capacity - len(nodes)))), dtype=actual_nodes.dtype
    )
    if actual_nodes.shape != (capacity,) or not np.array_equal(actual_nodes, expected):
        raise ValueError(
            "Adaptive continuation coordinates differ from bound mesh authority."
        )
    kinds = np.asarray(mesh.outer_node_kind)
    expected_kinds = np.where(
        np.arange(capacity) < len(nodes), QueryNodeKind.FINITE, QueryNodeKind.PAD
    )
    if not np.array_equal(kinds, expected_kinds):
        raise ValueError(
            "Adaptive continuation activity differs from its bound mesh authority."
        )
    association_shape = (*carry.endog_grid.shape[:-2], capacity)
    owners = np.asarray(mesh.outer_owner_id)
    expected_owners = np.where(
        np.arange(capacity) < len(nodes),
        np.arange(capacity),
        -1,
    ).astype(np.int32)
    if (
        owners.shape != association_shape
        or owners.dtype != np.dtype("int32")
        or not np.array_equal(
            owners, np.broadcast_to(expected_owners, association_shape)
        )
    ):
        raise ValueError(
            "Adaptive continuation owners differ from its canonical outer bank."
        )
    sides = np.asarray(mesh.outer_side_id)
    if (
        sides.shape != association_shape
        or sides.dtype != np.dtype("int32")
        or np.any(sides != 0)
    ):
        raise ValueError(
            "Adaptive continuation sides differ from its canonical outer bank."
        )
    if np.any(np.asarray(mesh.outer_link_valid)[..., len(nodes) - 1 :]):
        raise ValueError(
            "An adaptive continuation links through inactive outer padding."
        )


def validate_bound_continuation_mesh(
    *, carry: EGMCarry, authority: ArtifactAuthority
) -> None:
    """Validate a restored carry using the model-admitted descriptor axis."""
    axes = tuple(
        axis for axis in authority.axes if axis.name == ACTIVE_CONTINUATION_MESH_AXIS
    )
    data = carry.query_data
    adaptive = None if data is None else data.adaptive_outer_data
    if adaptive is None:
        if axes:
            raise TypeError("A bound adaptive descriptor has no matching numeric mesh.")
        return
    if len(axes) != 1:
        raise TypeError(
            "An adaptive continuation lacks independent completed mesh authority."
        )
    nodes = axes[0].coordinates
    if any(type(node) is not float for node in nodes):
        raise TypeError("A completed continuation axis has nonfloat coordinates.")
    validate_continuation_mesh(
        carry=carry,
        nodes=cast("tuple[float, ...]", nodes),
        capacity=adaptive.outer_nodes.size,
    )
