"""Public, lightweight types for labelled solver artifacts and solutions.

This module defines pylcm's dependency-safe solver extension boundary. Its public
definitions cover result identity and retention without referring to engine-private
``_lcm`` types or concrete built-in solver payloads.
"""

from lcm._solver_api.authority import (
    ArtifactAuthority,
    _artifact_authority_from_template_snapshot,  # noqa: F401
    _artifact_authority_template_snapshot,  # noqa: F401
    _artifact_leaf_values_from_plan,  # noqa: F401
    _ArtifactLeafSlot,  # noqa: F401
    _ArtifactTuplePlan,  # noqa: F401
    _CanonicalArtifactPayload,  # noqa: F401
    _CanonicalArtifactTemplate,  # noqa: F401
    _canonicalize_artifact_payload,  # noqa: F401
    _canonicalize_artifact_payload_snapshot,  # noqa: F401
    _container_types_from_tree,  # noqa: F401
    _copy_artifact_array_leaf,  # noqa: F401
    _normalize_jax_tree_path,  # noqa: F401
    _reconstruct_artifact_from_plan,  # noqa: F401
    _reconstruct_artifact_from_template_snapshot,  # noqa: F401
    _snapshot_artifact_template_once,  # noqa: F401
    _validate_axes_and_leaves,  # noqa: F401
)
from lcm._solver_api.contract import (
    ArtifactRef,
    OmissionReason,
    ReplayMode,
    SolutionMetadata,
    ValueArraySchema,
    _register_artifact_static_metadata_dataclass,  # noqa: F401
    _same_exact_artifact_contract,  # noqa: F401
)
from lcm._solver_api.descriptors import (
    ArtifactDescriptor,
)
from lcm._solver_api.entries import (
    _canonical_artifact_entry_from_authority,  # noqa: F401
    _CanonicalArtifactEntry,  # noqa: F401
    _CanonicalValueEntry,  # noqa: F401
    _LazyEntry,  # noqa: F401
)
from lcm._solver_api.identity import (
    PYLCM_VERSION,
    SOLUTION_FORMAT_VERSION,
    SOLUTION_SCHEMA_VERSION,
    SOLVER_API_VERSION,
    ArtifactChannel,
    ArtifactKey,
    AxisAuthority,
    AxisDescriptor,
    AxisRole,
    CategoryDomain,
    DeclaredReplay,
    LeafAuthority,
    LeafDescriptor,
    LoadState,
    PersistencePolicy,
    ReplayRouteIdentity,
    ResultRetention,
    SolutionSource,
    SolverIdentity,
    TreePath,
)
from lcm._solver_api.replay import (
    DISSOLUTION_FLAG,
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    SOLVER_DIAGNOSTICS,
    ActionOutput,
    ContinuationArtifact,
    ExecutableReplayRoute,
    KernelOutput,
    ReplayModelContext,
    ReplayReader,
    ReplayRoute,
    ReplayRouteRequirements,
    ReplayRouteSnapshot,
    SimulationBuildContext,
    _replay_route_identity,  # noqa: F401
)
from lcm._solver_api.result import (
    SolutionResult,
)
from lcm._solver_api.stores import (
    ArtifactStore,
    ValueStore,
)

__all__ = [
    "DISSOLUTION_FLAG",
    "EGM_CONTINUATION",
    "PYLCM_VERSION",
    "SIMULATION_POLICY",
    "SOLUTION_FORMAT_VERSION",
    "SOLUTION_SCHEMA_VERSION",
    "SOLVER_API_VERSION",
    "SOLVER_DIAGNOSTICS",
    "ActionOutput",
    "ArtifactAuthority",
    "ArtifactChannel",
    "ArtifactDescriptor",
    "ArtifactKey",
    "ArtifactRef",
    "ArtifactStore",
    "AxisAuthority",
    "AxisDescriptor",
    "AxisRole",
    "CategoryDomain",
    "ContinuationArtifact",
    "DeclaredReplay",
    "ExecutableReplayRoute",
    "KernelOutput",
    "LeafAuthority",
    "LeafDescriptor",
    "LoadState",
    "OmissionReason",
    "PersistencePolicy",
    "ReplayMode",
    "ReplayModelContext",
    "ReplayReader",
    "ReplayRoute",
    "ReplayRouteIdentity",
    "ReplayRouteRequirements",
    "ReplayRouteSnapshot",
    "ResultRetention",
    "SimulationBuildContext",
    "SolutionMetadata",
    "SolutionResult",
    "SolutionSource",
    "SolverIdentity",
    "TreePath",
    "ValueArraySchema",
    "ValueStore",
]
