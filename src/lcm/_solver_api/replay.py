"""Replay routes, their snapshots and build contexts, and kernel output."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Protocol,
    runtime_checkable,
)

import numpy as np
from jaxtyping import Float

from lcm._solver_api.authority import (
    ArtifactAuthority,
)
from lcm._solver_api.contract import (
    ReplayMode,
    SolutionMetadata,
)
from lcm._solver_api.identity import (
    ArtifactKey,
    ReplayRouteIdentity,
    SolverIdentity,
)
from lcm.typing import FloatND, IntND, RegimeName, StateName


@dataclass(frozen=True, kw_only=True)
class ReplayRouteSnapshot:
    """One immutable, preflighted cell passed unchanged to a replay route."""

    artifacts: Mapping[ArtifactKey, object]
    """Materialized payloads of the cell, keyed by artifact key."""
    authorities: Mapping[ArtifactKey, ArtifactAuthority]
    """Model-built authority of each payload."""
    metadata: SolutionMetadata
    """Descriptive metadata of the consumed result."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))
        object.__setattr__(
            self, "authorities", MappingProxyType(dict(self.authorities))
        )


@dataclass(frozen=True, kw_only=True)
class ReplayModelContext:
    """Narrow solve-grid view from which a route declares its dependencies.

    The names and node mappings describe the canonical, period-specific solution
    state/action space on which replay artifacts are defined. Simulation-only carried
    states are deliberately absent because they are not solution axes.
    """

    regime_name: RegimeName
    """Name of the regime being replayed."""
    period: int
    """Period of the solution cell."""
    state_names: tuple[str, ...]
    """Solution-state names in canonical product-map order."""

    action_names: tuple[str, ...]
    """Solution-action names in canonical product-map order."""

    state_nodes: Mapping[str, FloatND | IntND]
    """Period-specific grid nodes keyed exactly by ``state_names``."""

    action_nodes: Mapping[str, FloatND | IntND]
    """Period-specific grid nodes keyed exactly by ``action_names``."""

    def __post_init__(self) -> None:
        if type(self.regime_name) is not str or not self.regime_name:
            raise TypeError("ReplayModelContext.regime_name must be a nonempty str.")
        if type(self.period) is not int or self.period < 0:
            raise TypeError("ReplayModelContext.period must be a nonnegative int.")
        object.__setattr__(self, "state_names", tuple(self.state_names))
        object.__setattr__(self, "action_names", tuple(self.action_names))
        object.__setattr__(
            self, "state_nodes", MappingProxyType(dict(self.state_nodes))
        )
        object.__setattr__(
            self, "action_nodes", MappingProxyType(dict(self.action_nodes))
        )


@dataclass(frozen=True, kw_only=True)
class ReplayRouteRequirements:
    """Exact artifact keys one external route requires in every active cell."""

    required_artifacts: frozenset[ArtifactKey]
    """Artifact keys the route reads in every active cell."""

    def __post_init__(self) -> None:
        artifacts = frozenset(self.required_artifacts)
        if any(type(key) is not ArtifactKey for key in artifacts):
            raise TypeError(
                "ReplayRouteRequirements.required_artifacts must contain exact "
                "ArtifactKeys."
            )
        object.__setattr__(self, "required_artifacts", artifacts)


@dataclass(frozen=True, kw_only=True)
class SimulationBuildContext:
    """Public, model-owned solve-grid facts for constructing one replay reader.

    This carries the same state/action view used to declare route requirements.
    Simulation-only carried states are supplied later in ``ReplayReader.states`` but
    are not artifact axes and therefore do not appear here.
    """

    period: int
    """Period of the solution cell the reader is built for."""
    regime_name: RegimeName
    """Name of the regime the reader is built for."""
    state_names: tuple[str, ...]
    """Solution-state names in canonical product-map order."""

    action_names: tuple[str, ...]
    """Solution-action names in canonical product-map order."""

    state_nodes: Mapping[str, FloatND | IntND]
    """Period-specific grid nodes keyed exactly by ``state_names``."""

    action_nodes: Mapping[str, FloatND | IntND]
    """Period-specific grid nodes keyed exactly by ``action_names``."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_names", tuple(self.state_names))
        object.__setattr__(self, "action_names", tuple(self.action_names))
        object.__setattr__(
            self, "state_nodes", MappingProxyType(dict(self.state_nodes))
        )
        object.__setattr__(
            self, "action_nodes", MappingProxyType(dict(self.action_nodes))
        )


@dataclass(frozen=True, kw_only=True)
class ActionOutput:
    """Named action arrays returned by an external replay reader."""

    actions: Mapping[str, object]
    """Immutable mapping of action names to their per-subject values."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))


@runtime_checkable
class ReplayReader(Protocol):
    """JAX-transformable reader built from a validated replay snapshot."""

    def __call__(
        self, *, states: Mapping[str, object], fallback_actions: Mapping[str, object]
    ) -> ActionOutput:
        """Return each named action as a scalar or per-subject-broadcastable array.

        ``states`` contains the regime's full per-subject simulation state, including
        any carried-only states omitted from ``SimulationBuildContext``.
        """
        ...


# Built-in identities live on the public transport boundary. The engine imports
# these same singleton objects rather than defining private lookalikes, so an
# installed solver and pylcm always address the same schema.
EGM_CONTINUATION = ArtifactKey(type_id="pylcm.egm.continuation", schema_version=1)
SIMULATION_POLICY = ArtifactKey(type_id="pylcm.simulation.policy", schema_version=1)
DISSOLUTION_FLAG = ArtifactKey(
    type_id="pylcm.collective.dissolution_flag", schema_version=1
)
SOLVER_DIAGNOSTICS = ArtifactKey(type_id="pylcm.solver.diagnostics", schema_version=1)

# The coordinate an EGM carry's rows are tabulated on. It is also the state such
# a carry's marginal is taken with respect to, so a solver demanding that
# marginal and a payload publishing it name one string.
EGM_ENDOGENOUS_COORDINATE: StateName = "resources"


@runtime_checkable
class ReplayRoute(Protocol):
    """How one regime's simulated decision is obtained, declared by the model.

    Forward simulation and the pre-simulation payload check both dispatch on
    this object rather than on the concrete payload class, so a regime that
    retains no replay payload and one that retains an unfamiliar payload are
    described in the same vocabulary.
    """

    @property
    def replay_mode(self) -> ReplayMode:
        """How the decision is obtained under this route."""
        ...

    @property
    def payload_type(self) -> type[object] | None:
        """Exact class of the retained payload, `None` when none is kept."""
        ...

    @property
    def policy_applicable(self) -> bool:
        """Whether this route structurally publishes a replay payload."""
        ...

    @property
    def policy_required(self) -> bool:
        """Whether every successful solve must retain that payload."""
        ...

    @property
    def consumer_route(self) -> str | None:
        """Name of the reader consuming the payload, `None` without one."""
        ...


def _replay_route_identity(route: ReplayRoute) -> ReplayRouteIdentity:
    """Return the durable identity of a trusted built-in or executable route."""
    declared_identity = getattr(route, "identity", None)
    if type(declared_identity) is ReplayRouteIdentity:
        return declared_identity
    route_ids: dict[tuple[ReplayMode, str | None], str] = {
        (ReplayMode.VALID_RECOMPUTATION, None): "pylcm.grid_recomputation",
        (ReplayMode.UNSUPPORTED, None): "pylcm.replay_unsupported",
        (ReplayMode.EXACT_REPLAY, "egm_off_grid"): "pylcm.egm_off_grid",
        (ReplayMode.EXACT_REPLAY, "nnbegm_finite"): "pylcm.nnbegm_finite",
        (ReplayMode.EXACT_REPLAY, "nnbegm_nested"): "pylcm.nnbegm_nested",
        (ReplayMode.VALID_RECOMPUTATION, "nnbegm_finite"): "pylcm.nnbegm_finite",
        (ReplayMode.VALID_RECOMPUTATION, "nnbegm_nested"): "pylcm.nnbegm_nested",
        (ReplayMode.UNSUPPORTED, "nnbegm_finite"): "pylcm.nnbegm_finite",
        (ReplayMode.UNSUPPORTED, "nnbegm_nested"): "pylcm.nnbegm_nested",
    }
    try:
        route_id = route_ids[(route.replay_mode, route.consumer_route)]
    except KeyError as error:
        raise TypeError(
            "A replay route has no durable built-in or plugin identity."
        ) from error
    return ReplayRouteIdentity(route_id=route_id, route_version=1)


@runtime_checkable
class ExecutableReplayRoute(ReplayRoute, Protocol):
    """Installed plugin route that validates artifacts and builds its own reader."""

    @property
    def identity(self) -> ReplayRouteIdentity:
        """Return the route's durable identity and exact compatibility version."""
        ...

    @property
    def plugin_identity(self) -> SolverIdentity:
        """Return the installed plugin identity implementing this route."""
        ...

    def requirements(self, *, context: ReplayModelContext) -> ReplayRouteRequirements:
        """Declare the exact artifact dependencies for this model view."""
        ...

    def validate(
        self,
        *,
        snapshot: ReplayRouteSnapshot,
        context: SimulationBuildContext,
    ) -> None:
        """Check solver-specific mathematical invariants before simulation."""
        ...

    def build_reader(
        self,
        *,
        snapshot: ReplayRouteSnapshot,
        context: SimulationBuildContext,
    ) -> ReplayReader:
        """Build a JAX-transformable reader from the validated snapshot."""
        ...


@runtime_checkable
class ContinuationArtifact(Protocol):
    """A payload a period kernel publishes for the previous period's kernels.

    The engine stores and rolls it opaquely under its `artifact_key`; only a
    parent solver that declares the same key reads its fields.
    """

    @property
    def artifact_key(self) -> ArtifactKey:
        """Versioned identity of the payload's schema."""
        ...


@dataclass(frozen=True, kw_only=True)
class ContinuationCapabilities:
    """What a continuation payload can answer about itself."""

    value: bool = False
    """Whether the payload can return a continuation value at a query."""

    marginal_states: frozenset[StateName] = frozenset()
    """States the payload can differentiate its value with respect to."""

    exact_candidate_identity: bool = False
    """Whether the payload names which candidate owns a query point."""

    discontinuities: bool = False
    """Whether the payload locates its own one-sided boundaries."""


@runtime_checkable
class ContinuationReader(Protocol):
    """What a parent may ask its target's published continuation.

    A reader answers at a query rather than exposing its storage, so a parent
    that needs a value or a marginal is independent of how the target tabulated
    it. `leaves()` is the addressable content of the payload: the transfer
    catalogue plans one transfer per leaf a consumer declares.
    """

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Return what this payload can answer."""
        ...

    def value_at(self, *, query: FloatND) -> FloatND:
        """Return the continuation value at `query`."""
        ...

    def marginal_at(self, *, query: FloatND, state: StateName) -> FloatND:
        """Return the marginal of the continuation in `state` at `query`."""
        ...

    def leaves(self) -> Mapping[tuple[str, ...], FloatND]:
        """Return every published array by its pytree path."""
        ...


@dataclass(frozen=True, kw_only=True)
class KernelOutput:
    """One solver kernel's value and explicitly typed artifact channels.

    This is the dependency-safe producer envelope for solver extensions. Artifact
    identity is carried by :class:`ArtifactKey`, while the engine decides which
    declared artifacts it understands and consumes. The mappings are copied at
    construction and exposed as immutable views so a producer cannot mutate a
    published kernel result after returning it.

    Numerical diagnostics are kept outside this producer envelope. In-tree solvers
    that publish ``SolverDiagnostics`` use the engine-private result representation
    until that payload is represented as a public artifact.
    """

    value: FloatND | Float[np.ndarray, "*shape"]
    """The regime's value-function array on its exogenous state grid."""

    continuations: Mapping[ArtifactKey, object] = field(default_factory=dict)
    """Cross-period artifacts required while backward induction is running."""

    solve_time_artifacts: Mapping[ArtifactKey, object] = field(default_factory=dict)
    """Other artifacts consumed by the solve before the period rolls."""

    replay: Mapping[ArtifactKey, object] = field(default_factory=dict)
    """Artifacts a later simulation or policy replay may consume."""

    auxiliary: Mapping[ArtifactKey, object] = field(default_factory=dict)
    """Optional, solver-defined artifacts for inspection or persistence."""

    def __post_init__(self) -> None:
        if not hasattr(self.value, "shape") or not hasattr(self.value, "dtype"):
            raise TypeError(
                "KernelOutput.value must be one floating JAX or NumPy array leaf."
            )
        try:
            value_dtype = np.dtype(self.value.dtype)
        except TypeError as error:
            raise TypeError(
                "KernelOutput.value must be one floating JAX or NumPy array leaf."
            ) from error
        if not np.issubdtype(value_dtype, np.floating):
            raise TypeError(
                "KernelOutput.value must be one floating JAX or NumPy array leaf; "
                f"got dtype {value_dtype}."
            )

        key_to_channel: dict[ArtifactKey, str] = {}
        for field_name in (
            "continuations",
            "solve_time_artifacts",
            "replay",
            "auxiliary",
        ):
            entries = dict(getattr(self, field_name))
            if not all(type(key) is ArtifactKey for key in entries):
                raise TypeError(f"KernelOutput.{field_name} keys must be ArtifactKey.")
            for key in entries:
                if previous_channel := key_to_channel.get(key):
                    raise ValueError(
                        f"Artifact '{key.type_id}' version "
                        f"{key.schema_version} appears "
                        f"in both KernelOutput.{previous_channel} and "
                        f"KernelOutput.{field_name}; one artifact identity must belong "
                        "to exactly one semantic channel."
                    )
                key_to_channel[key] = field_name
            object.__setattr__(self, field_name, MappingProxyType(entries))
