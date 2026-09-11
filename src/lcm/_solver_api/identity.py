"""Versions, enumerations, and the identity and descriptor spine of the solver API."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    TypeAlias,
)

import numpy as np

from lcm.version import __version__

_SHA256_HEX_LENGTH = 64
PYLCM_VERSION = __version__

SOLVER_API_VERSION = 3
# Version of the public solver/plugin protocol implemented by this release.
# Version 2 covers typed internal outputs between core programs, the host-driven
# core-execution disposition, and compilation keys formed from program identity.

SOLUTION_SCHEMA_VERSION = 2
# Version of the labelled in-memory solution schema.

SOLUTION_FORMAT_VERSION = 2
# Version of the durable solution archive format.


class LoadState(StrEnum):
    """Whether one independently addressed persisted entry is materialized."""

    UNLOADED = "unloaded"
    LOADED = "loaded"


class ArtifactChannel(StrEnum):
    """Semantic channel on which a solver publishes an artifact."""

    CONTINUATION = "continuation"
    REPLAY = "replay"
    AUXILIARY = "auxiliary"
    DIAGNOSTIC = "diagnostic"


class PersistencePolicy(StrEnum):
    """Whether a model can independently authorize an artifact on restoration."""

    MODEL_VERIFIABLE = "model_verifiable"
    NOT_PERSISTED = "not_persisted"


class DeclaredReplay(StrEnum):
    """How simulation obtains an external solver's decision without a plugin route.

    An external solver publishes no engine-owned replay adapter, so it states
    what simulation may do with its stored solution:

    - `GRID_RECOMPUTATION`: the solve's decision is exactly the argmax over the
      regime's declared action grids at the realized state, so simulation
      recomputes it there from the stored values.
    - `UNSUPPORTED`: the decision cannot be reproduced from the stored solution;
      simulating the regime is refused with the reason named.

    A solver whose decision is neither implements `ExecutableReplayRoute`.
    """

    GRID_RECOMPUTATION = "grid_recomputation"
    UNSUPPORTED = "unsupported"


class SolutionSource(StrEnum):
    """Origin of the current result container."""

    IN_MEMORY = "in_memory"
    PERSISTED = "persisted"


class ResultRetention(StrEnum):
    """Artifacts a caller asks to keep after backward induction."""

    VALUES = "values"
    VALUES_AND_REPLAY = "values_and_replay"
    ALL_PERSISTABLE_ARTIFACTS = "all_persistable_artifacts"

    @property
    def retains_replay(self) -> bool:
        """Whether replay artifacts remain available after the solve."""
        return self is not ResultRetention.VALUES


@dataclass(frozen=True, order=True, kw_only=True)
class ArtifactKey:
    """Versioned identity of one artifact payload schema.

    ``type_id`` is a qualified, globally meaningful name such as
    ``"pylcm.simulation.policy"`` or ``"example_solver.euler_residuals"``.
    Changing a payload's interpretation requires a new ``schema_version``.
    """

    type_id: str
    """Qualified, globally meaningful name of the payload schema."""
    schema_version: int = 1
    """Version of the payload's interpretation; a new one whenever it changes."""

    def __post_init__(self) -> None:
        if type(self.type_id) is not str:
            raise TypeError("ArtifactKey.type_id must be an exact str.")
        if not self.type_id:
            raise ValueError("ArtifactKey.type_id must not be empty.")
        if type(self.schema_version) is not int:
            raise TypeError("ArtifactKey.schema_version must be an exact int.")
        if self.schema_version < 1:
            raise ValueError("ArtifactKey.schema_version must be at least 1.")


@dataclass(frozen=True, order=True, kw_only=True)
class SolverIdentity:
    """Durable identity and compatibility version of an installed solver plugin."""

    plugin_id: str
    """Qualified name of the installed solver package."""
    plugin_version: str
    """Release version of that package."""
    solver_api_version: int = SOLVER_API_VERSION
    """Public solver API version the package targets; it must match this release."""

    def __post_init__(self) -> None:
        if type(self.plugin_id) is not str:
            raise TypeError("SolverIdentity.plugin_id must be an exact str.")
        if not self.plugin_id:
            raise ValueError("SolverIdentity.plugin_id must not be empty.")
        if type(self.plugin_version) is not str:
            raise TypeError("SolverIdentity.plugin_version must be an exact str.")
        if not self.plugin_version:
            raise ValueError("SolverIdentity.plugin_version must not be empty.")
        if type(self.solver_api_version) is not int:
            raise TypeError("SolverIdentity.solver_api_version must be an exact int.")
        if self.solver_api_version != SOLVER_API_VERSION:
            raise ValueError(
                "SolverIdentity.solver_api_version is incompatible with this pylcm "
                f"release: got {self.solver_api_version}, expected "
                f"{SOLVER_API_VERSION}."
            )


@dataclass(frozen=True, order=True, kw_only=True)
class ReplayRouteIdentity:
    """Durable identity and schema version of one replay implementation."""

    route_id: str
    """Qualified name of the replay implementation."""
    route_version: int
    """Version of the payload schema the implementation reads."""

    def __post_init__(self) -> None:
        if type(self.route_id) is not str:
            raise TypeError("ReplayRouteIdentity.route_id must be an exact str.")
        if not self.route_id:
            raise ValueError("ReplayRouteIdentity.route_id must not be empty.")
        if type(self.route_version) is not int:
            raise TypeError("ReplayRouteIdentity.route_version must be an exact int.")
        if self.route_version < 1:
            raise ValueError("ReplayRouteIdentity.route_version must be at least 1.")


type TreePath = tuple[str, ...]
# Stable path to one container or numerical leaf in a public artifact PyTree. Each
# component records both the JAX key kind and value, for example ``"attribute:values"``
# or ``"flattened:0"``. The root path is ``()``.


class AxisRole(StrEnum):
    """Mathematical role of one named artifact axis."""

    STATE = "state"
    ACTION = "action"
    CANDIDATE = "candidate"
    STOCHASTIC = "stochastic"
    STAKEHOLDER = "stakeholder"
    OTHER = "other"


@dataclass(frozen=True, order=True, kw_only=True)
class CategoryDomain:
    """Exact labels, integer codes, and ordering of one categorical role."""

    labels: tuple[str, ...]
    """Category labels in code order."""
    codes: tuple[int, ...]
    """Integer code of each label, in the same order."""
    ordered: bool
    """Whether the categories carry a meaningful order."""

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        codes = tuple(self.codes)
        if not labels or len(labels) != len(codes):
            raise ValueError(
                "CategoryDomain labels and codes must be nonempty and have equal "
                "length."
            )
        if any(type(label) is not str or not label for label in labels):
            raise TypeError("CategoryDomain labels must be nonempty exact strs.")
        if any(type(code) is not int for code in codes):
            raise TypeError("CategoryDomain codes must be exact ints.")
        if type(self.ordered) is not bool:
            raise TypeError("CategoryDomain.ordered must be an exact bool.")
        if len(set(labels)) != len(labels) or len(set(codes)) != len(codes):
            raise ValueError("CategoryDomain labels and codes must each be unique.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "codes", codes)


@dataclass(frozen=True, order=True, kw_only=True)
class AxisDescriptor:
    """Descriptive name, length, and mathematical role of one artifact axis."""

    name: str
    """Name of the axis, unique within one artifact."""
    length: int
    """Number of entries along the axis."""
    role: AxisRole
    """Mathematical role of the axis."""
    coordinates: tuple[bool | int | float | str, ...] = ()
    """Coordinate of each entry, or empty when the axis is positional only."""

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TypeError("AxisDescriptor.name must be a nonempty exact str.")
        if type(self.length) is not int or self.length < 0:
            raise TypeError("AxisDescriptor.length must be a nonnegative exact int.")
        if type(self.role) is not AxisRole:
            raise TypeError("AxisDescriptor.role must be an exact AxisRole.")
        coordinates = tuple(self.coordinates)
        if coordinates and len(coordinates) != self.length:
            raise ValueError(
                f"AxisDescriptor {self.name!r} has {len(coordinates)} coordinates; "
                f"expected {self.length}."
            )
        if (
            not coordinates
            and self.length
            and self.role
            not in {
                AxisRole.STATE,
                AxisRole.ACTION,
            }
        ):
            raise ValueError(
                "Only a model state or action axis may defer descriptive coordinates."
            )
        if any(
            not any(type(value) is allowed for allowed in (bool, int, float, str))
            for value in coordinates
        ):
            raise TypeError(
                "AxisDescriptor coordinates must use exact JSON scalar types."
            )
        if any(
            type(value) is float and not np.isfinite(value) for value in coordinates
        ):
            raise ValueError("AxisDescriptor coordinates must be finite.")
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True, order=True, kw_only=True)
class AxisAuthority:
    """Model-owned axis description plus its exact canonical coordinates."""

    name: str
    """Name of the axis, unique within one artifact."""
    length: int
    """Number of entries along the axis."""
    role: AxisRole
    """Mathematical role of the axis."""
    coordinates: tuple[bool | int | float | str, ...] = ()
    """Exact canonical coordinate of each entry, or empty when positional only."""

    def __post_init__(self) -> None:
        descriptor = AxisDescriptor(
            name=self.name,
            length=self.length,
            role=self.role,
            coordinates=self.coordinates,
        )
        coordinates = tuple(self.coordinates)
        if coordinates and len(coordinates) != self.length:
            raise ValueError(
                f"AxisAuthority {self.name!r} has {len(coordinates)} coordinates; "
                f"expected {self.length}."
            )
        if (
            not coordinates
            and self.length
            and self.role
            not in {
                AxisRole.STATE,
                AxisRole.ACTION,
            }
        ):
            raise ValueError(
                "Only a model state or action axis may defer coordinates until model "
                "authority is bound."
            )
        if any(
            not any(type(value) is allowed for allowed in (bool, int, float, str))
            for value in coordinates
        ):
            raise TypeError(
                "AxisAuthority coordinates must use exact JSON scalar types."
            )
        object.__setattr__(self, "name", descriptor.name)
        object.__setattr__(self, "length", descriptor.length)
        object.__setattr__(self, "role", descriptor.role)
        object.__setattr__(self, "coordinates", coordinates)

    @property
    def descriptor(self) -> AxisDescriptor:
        """Return the transport-safe description of this authoritative axis."""
        return AxisDescriptor(
            name=self.name,
            length=self.length,
            role=self.role,
            coordinates=self.coordinates,
        )


@dataclass(frozen=True, order=True, kw_only=True)
class LeafDescriptor:
    """Transport-safe schema of one numerical artifact leaf."""

    path: TreePath
    """Stable path of the leaf inside the artifact PyTree; `()` for a root array."""
    shape: tuple[int, ...]
    """Exact array shape."""
    dtype: str
    """NumPy dtype name."""
    axis_names: tuple[str, ...]
    """Name of the artifact axis behind each array dimension, in order."""

    def __post_init__(self) -> None:
        path = tuple(self.path)
        shape = tuple(self.shape)
        axis_names = tuple(self.axis_names)
        if any(type(component) is not str or not component for component in path):
            raise TypeError(
                "LeafDescriptor.path components must be nonempty exact strs."
            )
        if any(type(size) is not int or size < 0 for size in shape):
            raise TypeError("LeafDescriptor.shape must contain nonnegative exact ints.")
        if type(self.dtype) is not str or not self.dtype:
            raise TypeError("LeafDescriptor.dtype must be a nonempty exact str.")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise TypeError("LeafDescriptor.dtype must name a NumPy dtype.") from error
        if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            raise TypeError("LeafDescriptor.dtype must be numerical or Boolean.")
        if len(axis_names) != len(shape) or any(
            type(name) is not str or not name for name in axis_names
        ):
            raise ValueError(
                "LeafDescriptor.axis_names must name every dimension exactly once."
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", str(dtype))
        object.__setattr__(self, "axis_names", axis_names)


@dataclass(frozen=True, kw_only=True)
class LeafAuthority:
    """Exact runtime type and schema of one model-authoritative PyTree leaf."""

    path: TreePath
    """Stable path of the leaf inside the artifact PyTree; `()` for a root array."""
    runtime_type: type[object]
    """Exact runtime class of the leaf."""
    shape: tuple[int, ...]
    """Exact array shape."""
    dtype: str
    """NumPy dtype name."""
    axis_names: tuple[str, ...]
    """Name of the artifact axis behind each array dimension, in order."""

    def __post_init__(self) -> None:
        descriptor = LeafDescriptor(
            path=self.path,
            shape=self.shape,
            dtype=self.dtype,
            axis_names=self.axis_names,
        )
        if not isinstance(self.runtime_type, type):
            raise TypeError("LeafAuthority.runtime_type must be a type.")
        object.__setattr__(self, "path", descriptor.path)
        object.__setattr__(self, "shape", descriptor.shape)
        object.__setattr__(self, "dtype", descriptor.dtype)
        object.__setattr__(self, "axis_names", descriptor.axis_names)

    @property
    def descriptor(self) -> LeafDescriptor:
        """Return the transport-safe description of this authoritative leaf."""
        return LeafDescriptor(
            path=self.path,
            shape=self.shape,
            dtype=self.dtype,
            axis_names=self.axis_names,
        )


if TYPE_CHECKING:
    _CategoricalDomainsBoundary: TypeAlias = Mapping[  # noqa: UP040
        str, CategoryDomain
    ]
    _ContainerRuntimeTypesBoundary: TypeAlias = Mapping[  # noqa: UP040
        TreePath, type[object]
    ]
    _LeafAuthoritiesBoundary: TypeAlias = Mapping[  # noqa: UP040
        TreePath, LeafAuthority
    ]
else:
    # These public constructors own exact, single-traversal mapping validation.
    # Runtime annotation sampling must not observe a stateful mapping first.
    _CategoricalDomainsBoundary = object
    _ContainerRuntimeTypesBoundary = object
    _LeafAuthoritiesBoundary = object
