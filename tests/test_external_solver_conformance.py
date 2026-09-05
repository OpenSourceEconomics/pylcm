"""Acceptance tests for a solver living entirely outside pylcm's engine package."""

import ast
import dataclasses
import functools
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm.model as model_module
import lcm.solver_api as solver_api_module
from _lcm.solution import fingerprint as fingerprint_module
from _lcm.solution import period_replay as period_replay_module
from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.persistence import load_solution, replay_period
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    SOLVER_DIAGNOSTICS,
    ActionOutput,
    ArtifactAuthority,
    ArtifactChannel,
    ArtifactDescriptor,
    ArtifactKey,
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
    ReplayModelContext,
    ReplayRouteRequirements,
    ReplayRouteSnapshot,
    ResultRetention,
    SimulationBuildContext,
    SolutionMetadata,
    SolutionResult,
    SolutionSource,
    SolverIdentity,
    ValueStore,
)
from lcm.solvers import (
    CoreExecutionDisposition,
    GridSearch,
    ProgramScope,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    StateAxesLeading,
)
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarFloat,
    ScalarInt,
)
from tests.conformance_solver import (
    COUNTER_KEY,
    OPTIONAL_REPLAY_KEY,
    POLICY_KEY,
    SCRATCH_KEY,
    Counter,
    Policy,
    ReferenceReplayRoute,
    ReferenceSolver,
    TerminalCounterSolver,
)

_N_PERIODS = 3
_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)
_PRODUCTIVITY_GRID = LinSpacedGrid(start=0.5, stop=1.0, n_points=2)
_ACTION_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=3)
_PENSION_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=3)
_PARAMS = {"discount_factor": 1.0}


@categorical(ordered=False)
class _ConsumptionChoice:
    low: ScalarInt
    high: ScalarInt


_DISCRETE_ACTION_GRID = DiscreteGrid(_ConsumptionChoice)


class _HostileEquality:
    """Fail if a trust boundary executes attacker-defined equality."""

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("hostile equality escaped")


class _HostileComparisonInt(int):
    """Fail if a trust boundary compares a noncanonical shape entry."""

    __hash__ = int.__hash__

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("hostile integer equality escaped")

    def __lt__(self, other: object) -> bool:
        raise RuntimeError("hostile integer ordering escaped")


class _ChangingLazyEntry(solver_api_module._LazyEntry):
    """Return trusted input once, then expose a different unvalidated object."""

    def __init__(self, *, first: object, subsequent: object) -> None:
        self.first = first
        self.subsequent = subsequent
        self.materialization_count = 0

    @property
    def load_state(self) -> LoadState:
        """Report whether the adversarial entry has been asked for already."""
        return (
            LoadState.UNLOADED if self.materialization_count == 0 else LoadState.LOADED
        )

    def materialize(self, *, template: object | None = None) -> object:  # noqa: ARG002
        """Change the returned object after the first materialization."""
        self.materialization_count += 1
        return self.first if self.materialization_count == 1 else self.subsequent


class _FailingLazyEntry(solver_api_module._LazyEntry):
    """Expose a decoder failure at the private lazy-entry boundary."""

    def __init__(self) -> None:
        self.materialization_count = 0

    @property
    def load_state(self) -> LoadState:
        """Remain unloaded because materialization never succeeds."""
        return (
            LoadState.UNLOADED if self.materialization_count == 0 else LoadState.LOADED
        )

    def materialize(self, *, template: object | None = None) -> object:  # noqa: ARG002
        """Raise the low-level decoder error simulation must normalize."""
        self.materialization_count += 1
        raise TypeError("hostile lazy decoder")


class _MetadataMutatingLazyEntry(solver_api_module._LazyEntry):
    """Mutate caller metadata while returning an otherwise valid lazy value."""

    def __init__(
        self,
        *,
        value: object,
        metadata: SolutionMetadata,
        solver_identity: SolverIdentity,
    ) -> None:
        self._value = value
        self._metadata = metadata
        self._solver_identity = solver_identity

    @property
    def load_state(self) -> LoadState:
        """Report that the adversarial entry has not materialized."""
        return LoadState.UNLOADED

    def materialize(self, *, template: object | None = None) -> object:  # noqa: ARG002
        """Replace a mapping and mutate one nested frozen identity wrapper."""
        object.__setattr__(
            self._solver_identity,
            "plugin_version",
            "hostile-replacement",
        )
        object.__setattr__(
            self._metadata,
            "solver_types",
            MappingProxyType({"hostile": "replacement"}),
        )
        return self._value


class _CachedAuthorityMutatingLazyEntry(solver_api_module._LazyEntry):
    """Mutate model-cached authority while returning a valid lazy value."""

    def __init__(
        self,
        *,
        value: object,
        model: Model,
        fingerprint: str,
        ref: ArtifactRef,
    ) -> None:
        self._value = value
        self._model = model
        self._fingerprint = fingerprint
        self._ref = ref
        self.materialization_count = 0

    @property
    def load_state(self) -> LoadState:
        """Report whether the adversarial value was consumed."""
        return (
            LoadState.UNLOADED if self.materialization_count == 0 else LoadState.LOADED
        )

    def materialize(self, *, template: object | None = None) -> object:  # noqa: ARG002
        """Corrupt a cached nested authority wrapper after preflight copied it."""
        self.materialization_count += 1
        authority = self._model._solution_authorities[self._fingerprint]
        object.__setattr__(
            authority.artifacts[self._ref],
            "payload_runtime_type",
            tuple,
        )
        return self._value


class _TemplateMutatingLazyEntry(solver_api_module._LazyEntry):
    """Mutate the callback template before returning a valid replay payload."""

    def __init__(self, *, payload: Policy) -> None:
        self._payload = payload
        self.materialization_count = 0

    @property
    def load_state(self) -> LoadState:
        """Report whether the adversarial replay payload was consumed."""
        return (
            LoadState.UNLOADED if self.materialization_count == 0 else LoadState.LOADED
        )

    def materialize(self, *, template: object | None = None) -> object:
        """Corrupt only the decoder-facing copy of a frozen plugin template."""
        self.materialization_count += 1
        if type(template) is not Policy:
            raise AssertionError(
                "The replay callback did not receive its Policy template."
            )
        object.__setattr__(
            template,
            "values",
            jnp.zeros((1,), dtype=template.values.dtype),
        )
        return self._payload


class _EnvelopeMutatingLazyEntry(solver_api_module._LazyEntry):
    """Swap caller-owned result channels while returning a valid value."""

    def __init__(
        self,
        *,
        value: object,
        replacement_replay: ArtifactStore,
        replacement_omissions: MappingProxyType[ArtifactRef, OmissionReason],
    ) -> None:
        self._value = value
        self._replacement_replay = replacement_replay
        self._replacement_omissions = replacement_omissions
        self.target: SolutionResult | None = None

    @property
    def load_state(self) -> LoadState:
        """Report that the adversarial entry has not materialized."""
        return LoadState.UNLOADED

    def materialize(self, *, template: object | None = None) -> object:  # noqa: ARG002
        """Replace result channels after preflight has begun."""
        if self.target is None:
            raise AssertionError("The adversarial lazy entry has no target result.")
        object.__setattr__(self.target, "replay_artifacts", self._replacement_replay)
        object.__setattr__(self.target, "omissions", self._replacement_omissions)
        return self._value


def _moving_wealth_grid(age: float) -> LinSpacedGrid:
    """Move the state nodes with age while preserving exact shape and dtype."""
    return LinSpacedGrid(start=1.0 + age, stop=3.0 + age, n_points=3)


def _moving_wealth_signature(age: float) -> float:
    """Give each distinct concrete grid its own stable build signature."""
    return age


@categorical(ordered=False)
class _RegimeId:
    active: ScalarInt
    retired: ScalarInt


def _utility(
    *,
    wealth: ContinuousState,
    productivity: ContinuousState,
    consumption: ContinuousAction,
) -> FloatND:
    """Make every action an exact tie while preserving a state-dependent value."""
    return wealth + 0.0 * productivity + 0.0 * consumption


def _discrete_utility(
    *,
    wealth: ContinuousState,
    productivity: ContinuousState,
    consumption: DiscreteAction,
) -> FloatND:
    """Keep the reference tie while declaring a categorical action correctly."""
    return wealth + 0.0 * productivity + 0.0 * consumption


def _next_wealth(*, wealth: ContinuousState) -> ContinuousState:
    """Keep wealth fixed across the short reference lifecycle."""
    return wealth


def _next_productivity(*, productivity: ContinuousState) -> ContinuousState:
    """Keep the second state fixed across the reference lifecycle."""
    return productivity


def _impute_pension_wealth(*, wealth: ContinuousState) -> ContinuousState:
    """Give the solve a pension value without adding a solution-grid axis."""
    return wealth


def _next_pension_wealth(*, pension_wealth: ContinuousState) -> ContinuousState:
    """Carry the true per-subject pension value through simulation."""
    return pension_wealth


def _stay_active(*, age: ScalarFloat) -> ScalarFloat:
    """Keep the active regime through the first two decision transitions."""
    return jnp.where(age < _N_PERIODS - 1, 1.0, 0.0)


def _enter_retirement(*, age: ScalarFloat) -> ScalarFloat:
    """Enter the terminal regime after the third active decision period."""
    return jnp.where(age >= _N_PERIODS - 1, 1.0, 0.0)


def _retired_utility() -> ScalarFloat:
    """Supply the disconnected terminal regime's scalar value."""
    return jnp.asarray(0.0)


def _model(
    *,
    solver: Solver,
    wealth_grid: LinSpacedGrid | AgeSpecializedGrid = _WEALTH_GRID,
    carried_state: bool = False,
    action_grid: LinSpacedGrid | DiscreteGrid = _ACTION_GRID,
    utility: Callable[..., object] = _utility,
) -> Model:
    """Build the same tiny lifecycle around any solver under comparison."""
    return Model(
        regimes={
            "active": Regime(
                transition={
                    "active": MarkovTransition(_stay_active),
                    "retired": MarkovTransition(_enter_retirement),
                },
                active=lambda age: age < _N_PERIODS,
                states={
                    "wealth": wealth_grid,
                    "productivity": _PRODUCTIVITY_GRID,
                    **(
                        {
                            "pension_wealth": Phased(
                                solve=_impute_pension_wealth,
                                simulate=_PENSION_GRID,
                            )
                        }
                        if carried_state
                        else {}
                    ),
                },
                state_transitions={
                    "wealth": _next_wealth,
                    "productivity": _next_productivity,
                    **(
                        {"pension_wealth": _next_pension_wealth}
                        if carried_state
                        else {}
                    ),
                },
                actions={"consumption": action_grid},
                functions={"utility": utility},
                solver=solver,
            ),
            "retired": Regime(
                transition=None,
                active=lambda age: age >= _N_PERIODS,
                functions={"utility": _retired_utility},
                solver=TerminalCounterSolver(),
            ),
        },
        ages=AgeGrid(start=0, stop=_N_PERIODS, step="Y"),
        regime_id_class=_RegimeId,
    )


def _initial_conditions() -> dict[str, jax.Array]:
    """Return two subjects at exact policy-grid nodes."""
    return {
        "wealth": jnp.asarray([1.0, 3.0]),
        "productivity": jnp.asarray([0.5, 1.0]),
        "age": jnp.zeros(2),
        "regime_id": jnp.full(2, _RegimeId.active, dtype=jnp.int32),
    }


def _solve(*, model: Model) -> SolutionResult:
    """Solve with the retention mode that exercises persistence decisions."""
    return model.solve(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )


def _refs(*, key: ArtifactKey) -> tuple[ArtifactRef, ...]:
    """Return the active-regime addresses for one artifact key."""
    return tuple(
        ArtifactRef(period=period, regime="active", key=key)
        for period in range(_N_PERIODS)
    )


_OPAQUE_SHIFT_MARKER = object()


def _opaque_shifted_values_core(
    *,
    wealth: FloatND,
    productivity: FloatND,
    consumption: FloatND,  # noqa: ARG001
    next_count: FloatND,
    candidate_width: int,  # noqa: ARG001
) -> tuple[FloatND, Counter]:
    """Expose the distinct core selected only by an opaque solver token."""
    count = next_count + 1.0
    value = count * (wealth[:, None] + 0.0 * productivity[None, :])
    return value + 7.0, Counter(count=count)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _OpaqueMarkerSolver(ReferenceSolver):
    """Accepted solver whose opaque instance token selects executable semantics."""

    marker: object

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        kernels = super().build_period_kernels(context=context)
        if self.marker is not _OPAQUE_SHIFT_MARKER:
            return kernels
        period_kernels = {}
        for period, kernel in kernels.period_kernels.items():
            programs = dict(cast("Any", kernel).core_programs())
            programs["values"] = dataclasses.replace(
                programs["values"],
                function=_opaque_shifted_values_core,
            )
            period_kernels[period] = dataclasses.replace(
                cast("Any", kernel),
                programs=MappingProxyType(programs),
            )
        return dataclasses.replace(
            kernels,
            period_kernels=MappingProxyType(period_kernels),
        )


class _BadRoleSolver(ReferenceSolver):
    """Reference solver whose authority names an action absent from the model."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Mutate only a semantic role, leaving the declared payload unchanged."""
        kernels = super().build_period_kernels(context=context)
        policy = kernels.artifact_authorities[POLICY_KEY]
        descriptor = dataclasses.replace(
            policy.descriptor,
            action_roles=("missing_action",),
        )
        invalid_policy = dataclasses.replace(
            policy,
            descriptor=descriptor,
            action_roles=("missing_action",),
        )
        return dataclasses.replace(
            kernels,
            artifact_authorities=MappingProxyType(
                dict(kernels.artifact_authorities) | {POLICY_KEY: invalid_policy}
            ),
        )


class _LaunderedAxisSolver(ReferenceSolver):
    """Reference solver that labels a model state axis as solver-owned OTHER."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Try to retain plugin-chosen coordinates under a canonical state name."""
        kernels = super().build_period_kernels(context=context)
        policy = kernels.artifact_authorities[POLICY_KEY]
        laundered_axis = dataclasses.replace(
            policy.axes[0],
            role=AxisRole.OTHER,
            coordinates=(10.0, 20.0, 30.0),
        )
        descriptor = dataclasses.replace(
            policy.descriptor,
            named_axes=(laundered_axis.descriptor, policy.axes[1].descriptor),
            state_roles=("productivity",),
        )
        invalid_policy = dataclasses.replace(
            policy,
            descriptor=descriptor,
            axes=(laundered_axis, policy.axes[1]),
            state_roles=("productivity",),
        )
        return dataclasses.replace(
            kernels,
            artifact_authorities=MappingProxyType(
                dict(kernels.artifact_authorities) | {POLICY_KEY: invalid_policy}
            ),
        )


class _DiagnosticAuthoritySolver(ReferenceSolver):
    """Reference solver attempting to use an unsupported custom diagnostic channel."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Relabel the optional replay authority without adding an output channel."""
        kernels = super().build_period_kernels(context=context)
        artifact = kernels.artifact_authorities[OPTIONAL_REPLAY_KEY]
        descriptor = dataclasses.replace(
            artifact.descriptor,
            channel=ArtifactChannel.DIAGNOSTIC,
        )
        invalid_artifact = dataclasses.replace(artifact, descriptor=descriptor)
        return dataclasses.replace(
            kernels,
            artifact_authorities=MappingProxyType(
                dict(kernels.artifact_authorities)
                | {OPTIONAL_REPLAY_KEY: invalid_artifact}
            ),
        )


class _ReservedArtifactAuthoritySolver(ReferenceSolver):
    """Reference solver attempting to reuse an engine-owned artifact identity."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Relabel a custom replay authority with the dissolution type id."""
        kernels = super().build_period_kernels(context=context)
        artifact = kernels.artifact_authorities[OPTIONAL_REPLAY_KEY]
        reserved_key = ArtifactKey(
            type_id=DISSOLUTION_FLAG.type_id,
            schema_version=2,
        )
        descriptor = dataclasses.replace(
            artifact.descriptor,
            key=reserved_key,
        )
        invalid_artifact = dataclasses.replace(artifact, descriptor=descriptor)
        return dataclasses.replace(
            kernels,
            artifact_authorities=MappingProxyType({reserved_key: invalid_artifact}),
        )


class _ConflictingProducerTypeSolver(ReferenceSolver):
    """Reference solver whose producer type contradicts its model authority."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Change only the producer declaration for the required policy."""
        kernels = super().build_period_kernels(context=context)
        period_kernels = {}
        for period, kernel in kernels.period_kernels.items():
            programs = dict(cast("Any", kernel).core_programs())
            replay = programs["replay"]
            programs["replay"] = dataclasses.replace(
                replay,
                retained_artifact_payload_types=(
                    dict(replay.retained_artifact_payload_types) | {POLICY_KEY: tuple}
                ),
            )
            period_kernels[period] = dataclasses.replace(
                cast("Any", kernel),
                programs=MappingProxyType(programs),
            )
        return dataclasses.replace(
            kernels,
            period_kernels=MappingProxyType(period_kernels),
        )


class _MissingRequiredProducerSolver(ReferenceSolver):
    """Reference solver whose required replay artifact has no producer."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Retain only the optional replay artifact from the replay program."""
        kernels = super().build_period_kernels(context=context)
        period_kernels = {}
        for period, kernel in kernels.period_kernels.items():
            programs = dict(cast("Any", kernel).core_programs())
            replay = programs["replay"]
            programs["replay"] = dataclasses.replace(
                replay,
                retained_artifact_keys=(OPTIONAL_REPLAY_KEY,),
                retained_artifact_payload_types={OPTIONAL_REPLAY_KEY: jax.Array},
            )
            period_kernels[period] = dataclasses.replace(
                cast("Any", kernel),
                programs=MappingProxyType(programs),
            )
        return dataclasses.replace(
            kernels,
            period_kernels=MappingProxyType(period_kernels),
        )


class _MissingRequiredAuxiliaryProducerSolver(ReferenceSolver):
    """Reference solver whose required auxiliary artifact has no producer."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Remove the additive program while retaining its required authority."""
        kernels = super().build_period_kernels(context=context)
        period_kernels = {}
        for period, kernel in kernels.period_kernels.items():
            programs = dict(cast("Any", kernel).core_programs())
            del programs["scratch"]
            period_kernels[period] = dataclasses.replace(
                cast("Any", kernel),
                programs=MappingProxyType(programs),
            )
        return dataclasses.replace(
            kernels,
            period_kernels=MappingProxyType(period_kernels),
        )


class _InapplicablePublishedArtifactSolver(ReferenceSolver):
    """Reference solver that emits a custom artifact declared inapplicable."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Disable only the optional replay authority, not its producer."""
        kernels = super().build_period_kernels(context=context)
        authority = kernels.artifact_authorities[OPTIONAL_REPLAY_KEY]
        return dataclasses.replace(
            kernels,
            artifact_authorities=MappingProxyType(
                dict(kernels.artifact_authorities)
                | {
                    OPTIONAL_REPLAY_KEY: dataclasses.replace(
                        authority,
                        applicable=False,
                    )
                }
            ),
        )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["count"],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True, kw_only=True)
class _WrongCounter:
    """Structurally similar continuation with a deliberately different type."""

    count: FloatND

    @property
    def artifact_key(self) -> ArtifactKey:
        """Reuse the key so only the producer/template type distinguishes it."""
        return COUNTER_KEY


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ContradictoryContinuationAuthoritySolver(ReferenceSolver):
    """Reference solver whose custom authority contradicts ContinuationSpec."""

    defect: str

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Change one continuation-authority fact while leaving the spec intact."""
        kernels = super().build_period_kernels(context=context)
        authority = kernels.artifact_authorities[COUNTER_KEY]
        if self.defect == "channel":
            invalid = dataclasses.replace(
                authority,
                descriptor=dataclasses.replace(
                    authority.descriptor,
                    channel=ArtifactChannel.AUXILIARY,
                ),
            )
        elif self.defect == "applicable":
            invalid = dataclasses.replace(authority, applicable=False)
        elif self.defect == "required":
            invalid = dataclasses.replace(
                authority,
                descriptor=dataclasses.replace(authority.descriptor, required=False),
                required=False,
            )
        elif self.defect == "payload_type":
            invalid = dataclasses.replace(
                authority,
                descriptor=dataclasses.replace(
                    authority.descriptor,
                    payload_type_id="tests._WrongCounter",
                ),
                payload_runtime_type=_WrongCounter,
                template=_WrongCounter(count=cast("Counter", authority.template).count),
                container_runtime_types={(): _WrongCounter},
            )
        elif self.defect == "template_dtype":
            path, leaf = next(iter(authority.leaves.items()))
            incompatible_leaf = dataclasses.replace(leaf, dtype="int32")
            invalid = dataclasses.replace(
                authority,
                descriptor=dataclasses.replace(
                    authority.descriptor,
                    leaf_descriptors=(incompatible_leaf.descriptor,),
                ),
                template=Counter(count=jnp.asarray(0, dtype=jnp.int32)),
                leaves={path: incompatible_leaf},
            )
        else:
            raise AssertionError(
                f"Unknown continuation-authority defect {self.defect!r}."
            )
        return dataclasses.replace(
            kernels,
            artifact_authorities=MappingProxyType(
                dict(kernels.artifact_authorities) | {COUNTER_KEY: invalid}
            ),
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ScalarActionReader:
    """Return one scalar action for broadcast by the public boundary."""

    kind: str

    def __call__(
        self,
        *,
        states: Mapping[str, object],  # noqa: ARG002
        fallback_actions: Mapping[str, object],  # noqa: ARG002
    ) -> ActionOutput:
        """Return the selected scalar without Python data dependence."""
        values = {
            "finite": jnp.asarray(0.5),
            "nan": jnp.asarray(jnp.nan),
            "positive_inf": jnp.asarray(jnp.inf),
            "negative_inf": jnp.asarray(-jnp.inf),
        }
        return ActionOutput(actions={"consumption": values[self.kind]})


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ScalarActionRoute(ReferenceReplayRoute):
    """Build a JAX-transformable reader returning one scalar action."""

    kind: str = "nan"

    def build_reader(
        self,
        *,
        snapshot: ReplayRouteSnapshot,
        context: SimulationBuildContext,
    ) -> _ScalarActionReader:
        """Preserve route checks, then substitute the adversarial reader."""
        super().build_reader(snapshot=snapshot, context=context)
        return _ScalarActionReader(kind=self.kind)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ScalarActionSolver(ReferenceSolver):
    """Reference solver whose replay route emits one selected scalar."""

    kind: str

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Replace only the replay reader while keeping valid solve artifacts."""
        kernels = super().build_period_kernels(context=context)
        route = cast("ReferenceReplayRoute", kernels.replay_route)
        return dataclasses.replace(
            kernels,
            replay_route=_ScalarActionRoute(
                selected_action=route.selected_action,
                kind=self.kind,
            ),
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _OutOfDomainActionReader:
    """Return one scalar categorical code absent from the declared action domain."""

    def __call__(
        self,
        *,
        states: Mapping[str, object],  # noqa: ARG002
        fallback_actions: Mapping[str, object],  # noqa: ARG002
    ) -> ActionOutput:
        """Return a broadcastable but invalid int32 categorical code."""
        return ActionOutput(actions={"consumption": jnp.asarray(17, dtype=jnp.int32)})


@dataclasses.dataclass(frozen=True, kw_only=True)
class _OutOfDomainActionRoute(ReferenceReplayRoute):
    """Build a JAX-transformable reader with an out-of-domain categorical code."""

    def build_reader(
        self,
        *,
        snapshot: ReplayRouteSnapshot,
        context: SimulationBuildContext,
    ) -> _OutOfDomainActionReader:
        """Preserve route checks, then substitute the adversarial reader."""
        super().build_reader(snapshot=snapshot, context=context)
        return _OutOfDomainActionReader()


class _OutOfDomainActionSolver(ReferenceSolver):
    """Reference solver whose replay route emits an invalid categorical code."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Replace only the replay reader while keeping valid solve artifacts."""
        kernels = super().build_period_kernels(context=context)
        route = cast("ReferenceReplayRoute", kernels.replay_route)
        return dataclasses.replace(
            kernels,
            replay_route=_OutOfDomainActionRoute(
                selected_action=route.selected_action,
            ),
        )


def test_reference_solver_imports_only_the_public_extension_modules() -> None:
    """The plugin package has no import-time dependency on engine internals."""
    source = Path(__file__).with_name("conformance_solver") / "solver.py"
    tree = ast.parse(source.read_text())
    allowed = {"lcm.solvers", "lcm.solver_api", "lcm.typing"}
    imported_lcm_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "lcm" or node.module.startswith("lcm."))
    }
    direct_lcm_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "lcm" or alias.name.startswith("lcm.")
    }
    private_imports = [
        node
        for node in ast.walk(tree)
        if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("_lcm"))
        or (
            isinstance(node, ast.Import)
            and any(alias.name.startswith("_lcm") for alias in node.names)
        )
    ]

    assert imported_lcm_modules == allowed
    assert direct_lcm_imports == set()
    assert private_imports == []


def test_reference_solver_scopes_planned_cores_to_requested_retention() -> None:
    """VALUES skips replay assembly while ALL selects persistable replay."""
    model = _model(solver=ReferenceSolver())
    kernel = model._regimes["active"].solution.period_kernels[0]
    programs = cast("Any", kernel).core_programs()
    values_program = programs["values"]
    replay_program = programs["replay"]
    axis = values_program.requirements.streamable_axes[0]

    scratch_program = programs["scratch"]
    assert tuple(programs) == ("values", "replay", "scratch")
    assert values_program.scope is ProgramScope.VALUES_ONLY
    assert replay_program.scope is ProgramScope.REPLAY
    assert scratch_program.scope is ProgramScope.ARTIFACT
    assert replay_program.retained_artifact_keys == (
        POLICY_KEY,
        OPTIONAL_REPLAY_KEY,
    )
    assert replay_program.retained_artifact_payload_types == {
        POLICY_KEY: Policy,
        OPTIONAL_REPLAY_KEY: jax.Array,
    }
    assert replay_program.replaces_program == "values"
    assert scratch_program.retained_artifact_keys == (SCRATCH_KEY,)
    assert scratch_program.retained_artifact_payload_types == {SCRATCH_KEY: jax.Array}
    assert values_program.disposition is CoreExecutionDisposition.PLANNED
    assert replay_program.disposition is CoreExecutionDisposition.PLANNED
    assert axis.name == "candidate"
    assert axis.coordinate_names == ("consumption",)
    assert axis.coordinate_extents == (3,)
    assert axis.reduction.semantic_key == ("middle-tie-maximum", 1)
    value_role_leaves = jax.tree.leaves(values_program.output_roles)
    replay_role_leaves = jax.tree.leaves(replay_program.output_roles)
    assert len(value_role_leaves) == 2
    scratch_role_leaves = jax.tree.leaves(scratch_program.output_roles)
    assert len(replay_role_leaves) == 4
    assert len(scratch_role_leaves) == 2
    assert all(isinstance(role, StateAxesLeading) for role in replay_role_leaves[1:])

    values_only = model.solve(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.VALUES,
    )
    policy_refs = _refs(key=POLICY_KEY)
    scratch_refs = _refs(key=SCRATCH_KEY)
    assert all(ref not in values_only.replay_artifacts for ref in policy_refs)
    assert all(
        values_only.omissions[ref] is OmissionReason.NOT_REQUESTED
        for ref in (*policy_refs, *scratch_refs)
    )

    solution = _solve(model=model)
    assert all(ref in solution.replay_artifacts for ref in policy_refs)
    assert all(ref in solution.auxiliary_artifacts for ref in scratch_refs)
    wealth = np.asarray(_WEALTH_GRID.to_jax())[:, None]
    for period in range(_N_PERIODS):
        remaining = _N_PERIODS - period
        np.testing.assert_array_equal(
            solution.value(period=period, regime="active"),
            np.broadcast_to(
                remaining * wealth,
                (len(_WEALTH_GRID.to_jax()), len(_PRODUCTIVITY_GRID.to_jax())),
            ),
        )


def test_opaque_solver_marker_cannot_hide_a_distinct_accepted_core() -> None:
    """Instance identity tokens fail closed before two semantics share a digest."""
    plain = _model(solver=_OpaqueMarkerSolver(marker=object()))
    shifted = _model(solver=_OpaqueMarkerSolver(marker=_OPAQUE_SHIFT_MARKER))
    arguments = {
        "wealth": jnp.asarray([1.0, 2.0]),
        "productivity": jnp.asarray([0.5]),
        "consumption": jnp.asarray([0.0, 0.5, 1.0]),
        "next_count": jnp.asarray(0.0),
        "candidate_width": 3,
    }
    plain_program = cast(
        "Any", plain._regimes["active"].solution.period_kernels[0]
    ).core_programs()["values"]
    shifted_program = cast(
        "Any", shifted._regimes["active"].solution.period_kernels[0]
    ).core_programs()["values"]
    plain_value = cast("tuple[FloatND, object]", plain_program.function(**arguments))[0]
    shifted_value = cast(
        "tuple[FloatND, object]", shifted_program.function(**arguments)
    )[0]
    assert not np.array_equal(np.asarray(plain_value), np.asarray(shifted_value))

    for model in (plain, shifted):
        flat_params = model._process_params(_PARAMS)
        with pytest.raises(TypeError, match="opaque semantic value"):
            fingerprint_module.fingerprint_model(
                ages=model.ages,
                regimes=model._regimes,
                user_regimes=model.user_regimes,
                regime_names_to_ids=model.regime_names_to_ids,
                flat_params=flat_params,
            )

    stateless = _model(solver=ReferenceSolver())
    stateless_flat_params = stateless._process_params(_PARAMS)
    assert fingerprint_module.fingerprint_model(
        ages=stateless.ages,
        regimes=stateless._regimes,
        user_regimes=stateless.user_regimes,
        regime_names_to_ids=stateless.regime_names_to_ids,
        flat_params=stateless_flat_params,
    )


def test_period_replay_preserves_exact_artifact_selection(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A captured ALL solve reselects both its replay and additive programs."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "active@0")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    solution = _solve(model=_model(solver=ReferenceSolver()))
    observed: list[tuple[bool, frozenset[ArtifactKey]]] = []
    real_select = period_replay_module.select_programs

    def record_select(**kwargs: Any) -> object:
        observed.append(
            (
                kwargs["retain_replay"],
                cast("frozenset[ArtifactKey]", kwargs["selected_artifact_keys"]),
            )
        )
        return real_select(**kwargs)

    monkeypatch.setattr(period_replay_module, "select_programs", record_select)
    replay = replay_period(directory=tmp_path / "active@0")

    assert observed == [
        (False, frozenset({POLICY_KEY, OPTIONAL_REPLAY_KEY, SCRATCH_KEY}))
    ]
    assert set(replay.output.replay) == {POLICY_KEY, OPTIONAL_REPLAY_KEY}
    assert set(replay.output.auxiliary) == {SCRATCH_KEY}
    np.testing.assert_array_equal(
        np.asarray(replay.output.value),
        np.asarray(solution.value(period=0, regime="active")),
    )


def test_solution_retains_custom_replay_and_records_nonpersisted_artifacts() -> None:
    """Retention follows each custom artifact's model-built authority."""
    solution = _solve(model=_model(solver=ReferenceSolver()))
    policy_refs = _refs(key=POLICY_KEY)
    counter_refs = _refs(key=COUNTER_KEY)
    optional_replay_refs = _refs(key=OPTIONAL_REPLAY_KEY)
    scratch_refs = _refs(key=SCRATCH_KEY)

    assert set(solution.replay_artifacts) >= set(policy_refs)
    assert set(solution.replay_artifacts) >= set(optional_replay_refs)
    assert all(ref not in solution.retained_continuations for ref in counter_refs)
    assert all(ref in solution.auxiliary_artifacts for ref in scratch_refs)
    assert all(
        solution.omissions[ref] is OmissionReason.NOT_PERSISTED for ref in counter_refs
    )
    for ref in policy_refs:
        policy = cast("Policy", solution.replay_artifacts[ref])
        np.testing.assert_array_equal(
            policy.values,
            np.full((3, 2), 0.5),
        )
    for ref in scratch_refs:
        np.testing.assert_array_equal(
            solution.auxiliary_artifacts[ref],
            np.zeros((3, 2)),
        )


def test_custom_route_changes_the_grid_search_tie_decision() -> None:
    """Simulation invokes the plugin reader instead of the built-in grid argmax."""
    model = _model(solver=ReferenceSolver())
    custom = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        solution=_solve(model=model),
        log_level="off",
    ).to_dataframe()
    baseline_model = _model(solver=GridSearch())
    baseline = baseline_model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        log_level="off",
    ).to_dataframe()

    np.testing.assert_array_equal(
        custom.query("regime_name == 'active'")["consumption"],
        np.full(2 * _N_PERIODS, 0.5),
    )
    np.testing.assert_array_equal(
        baseline.query("regime_name == 'active'")["consumption"],
        np.zeros(2 * _N_PERIODS),
    )
    route = model._regimes["active"].simulation.external_replay_route
    assert isinstance(route, ReferenceReplayRoute)
    assert route.audit.validation_artifact_keys == [(POLICY_KEY,)] * _N_PERIODS
    assert route.audit.reader_artifact_keys == route.audit.validation_artifact_keys


@pytest.mark.parametrize("kind", ["nan", "positive_inf", "negative_inf"])
def test_external_reader_rejects_non_finite_actions_with_validation_off(
    *, kind: str
) -> None:
    """Reader actions are finite independently of optional runtime diagnostics."""
    model = _model(solver=_ScalarActionSolver(kind=kind))
    solution = _solve(model=model)

    with pytest.raises(
        InvalidSimulationInputError,
        match=r"non-finite action.*'consumption'",
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=solution,
            log_level="off",
        )


def test_external_reader_broadcasts_a_valid_scalar_action() -> None:
    """Validation preserves the public scalar-to-subject broadcasting contract."""
    model = _model(solver=_ScalarActionSolver(kind="finite"))

    panel = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        solution=_solve(model=model),
        log_level="off",
    ).to_dataframe()

    np.testing.assert_array_equal(
        panel.query("regime_name == 'active'")["consumption"],
        np.full(2 * _N_PERIODS, 0.5),
    )


def test_external_reader_rejects_out_of_domain_categorical_action() -> None:
    """A valid int32 code still has to belong to its declared action domain."""
    model = _model(
        solver=_OutOfDomainActionSolver(),
        action_grid=_DISCRETE_ACTION_GRID,
        utility=_discrete_utility,
    )
    solution = _solve(model=model)

    with pytest.raises(
        InvalidSimulationInputError,
        match=r"outside.*categorical domain.*'consumption'",
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=solution,
            log_level="off",
        )


def test_custom_reader_is_jax_transformable() -> None:
    """The installed route builds an array-only reader accepted by JAX."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    route = model._regimes["active"].simulation.external_replay_route
    assert isinstance(route, ReferenceReplayRoute)
    ref = ArtifactRef(period=0, regime="active", key=POLICY_KEY)
    authority = solution._artifact_authority[ref]
    snapshot = ReplayRouteSnapshot(
        artifacts={POLICY_KEY: solution.replay_artifacts[ref]},
        authorities={POLICY_KEY: authority},
        metadata=solution.metadata,
    )
    context = SimulationBuildContext(
        period=0,
        regime_name="active",
        state_names=("wealth", "productivity"),
        action_names=("consumption",),
        state_nodes={
            "wealth": _WEALTH_GRID.to_jax(),
            "productivity": _PRODUCTIVITY_GRID.to_jax(),
        },
        action_nodes={"consumption": _ACTION_GRID.to_jax()},
    )
    requirements = route.requirements(
        context=ReplayModelContext(
            regime_name="active",
            period=0,
            state_names=context.state_names,
            action_names=context.action_names,
            state_nodes=context.state_nodes,
            action_nodes=context.action_nodes,
        )
    )
    assert type(requirements) is ReplayRouteRequirements
    assert requirements.required_artifacts == frozenset({POLICY_KEY})
    route.validate(snapshot=snapshot, context=context)
    reader = route.build_reader(snapshot=snapshot, context=context)

    @jax.jit
    def read(wealth: jax.Array) -> jax.Array:
        output = reader(
            states={
                "wealth": wealth,
                "productivity": jnp.ones_like(wealth),
            },
            fallback_actions={"consumption": jnp.zeros_like(wealth)},
        )
        assert isinstance(output, ActionOutput)
        return jnp.asarray(output.actions["consumption"])

    np.testing.assert_array_equal(
        read(jnp.asarray([1.0, 2.0, 3.0])),
        np.full(3, 0.5),
    )


def test_persisted_solution_loads_lazily_and_replays_in_a_fresh_model(
    tmp_path: Path,
) -> None:
    """Independent entries stay lazy until value inspection or replay preflight."""
    solution = _solve(model=_model(solver=ReferenceSolver()))
    restored = load_solution(path=solution.save(path=tmp_path / "solution.lcm"))
    policy_refs = _refs(key=POLICY_KEY)
    optional_refs = _refs(key=OPTIONAL_REPLAY_KEY)

    assert isinstance(restored.values, ValueStore)
    assert all(
        restored.values.load_state(period=period, regime="active") is LoadState.UNLOADED
        for period in range(_N_PERIODS)
    )
    assert all(
        restored.replay_artifacts.load_state(ref) is LoadState.UNLOADED
        for ref in (*policy_refs, *optional_refs)
    )

    restored.value(period=0, regime="active")

    assert restored.values.load_state(period=0, regime="active") is LoadState.LOADED
    assert all(
        restored.replay_artifacts.load_state(ref) is LoadState.UNLOADED
        for ref in policy_refs
    )

    fresh_model = _model(solver=ReferenceSolver())
    panel = fresh_model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        solution=restored,
        log_level="off",
    ).to_dataframe()

    assert all(
        restored.replay_artifacts.load_state(ref) is LoadState.LOADED
        for ref in policy_refs
    )
    assert all(
        restored.replay_artifacts.load_state(ref) is LoadState.UNLOADED
        for ref in optional_refs
    )
    np.testing.assert_array_equal(
        panel.query("regime_name == 'active'")["consumption"],
        np.full(2 * _N_PERIODS, 0.5),
    )

    route = fresh_model._regimes["active"].simulation.external_replay_route
    assert isinstance(route, ReferenceReplayRoute)
    assert route.audit.validated_snapshots == route.audit.reader_snapshots
    assert len(route.audit.validated_snapshots) == _N_PERIODS
    assert {context.period for context in route.audit.requirement_contexts} == set(
        range(_N_PERIODS)
    )
    assert route.audit.validation_artifact_keys == [(POLICY_KEY,)] * _N_PERIODS


def test_simulation_materializes_each_value_and_replay_artifact_once() -> None:
    """Preflight validates the exact immutable objects consumed by simulation."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    value_entries: dict[object, object] = {}
    changing_values: list[_ChangingLazyEntry] = []
    for period, regime_to_value in solution.values.items():
        for regime_name, value in regime_to_value.items():
            entry = _ChangingLazyEntry(
                first=value,
                subsequent=jnp.asarray([0.0], dtype=value.dtype),
            )
            value_entries[(period, regime_name)] = entry
            changing_values.append(entry)

    replay_entries: dict[ArtifactRef, object] = {}
    changing_artifacts: list[_ChangingLazyEntry] = []
    for ref, payload in solution.replay_artifacts.items():
        if ref.key == POLICY_KEY:
            policy = cast("Policy", payload)
            entry = _ChangingLazyEntry(
                first=policy,
                subsequent=Policy(values=jnp.zeros(1, dtype=policy.values.dtype)),
            )
            replay_entries[ref] = entry
            changing_artifacts.append(entry)
        else:
            replay_entries[ref] = payload

    stateful_solution = dataclasses.replace(
        solution,
        values=ValueStore(value_entries),
        replay_artifacts=ArtifactStore(replay_entries),
    )

    panel = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        solution=stateful_solution,
        log_level="off",
    ).to_dataframe()

    assert all(entry.materialization_count == 1 for entry in changing_values)
    assert all(entry.materialization_count == 1 for entry in changing_artifacts)
    np.testing.assert_array_equal(
        panel.query("regime_name == 'active'")["consumption"],
        np.full(2 * _N_PERIODS, 0.5),
    )


def test_lazy_value_cannot_mutate_cached_model_authority_used_for_replay() -> None:
    """Replay consumes a private authority snapshot, not the mutable cache wrapper."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    flat_params = model._process_params(_PARAMS)
    fingerprint = solution.metadata.params_fingerprint
    cached_authority = model_module.build_solution_authority(
        regimes=model._regimes,
        flat_params=flat_params,
        ages=model.ages,
    )
    model._solution_authorities[fingerprint] = cached_authority
    policy_ref = next(ref for ref in solution.replay_artifacts if ref.key == POLICY_KEY)
    value_entries: dict[object, object] = {
        (period, regime_name): value
        for period, regime_to_value in solution.values.items()
        for regime_name, value in regime_to_value.items()
    }
    coordinate = next(iter(value_entries))
    adversarial = _CachedAuthorityMutatingLazyEntry(
        value=value_entries[coordinate],
        model=model,
        fingerprint=fingerprint,
        ref=policy_ref,
    )
    value_entries[coordinate] = adversarial
    stateful_solution = dataclasses.replace(
        solution,
        values=ValueStore(value_entries),
    )

    panel = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        solution=stateful_solution,
        log_level="off",
    ).to_dataframe()

    assert adversarial.materialization_count == 1
    assert cached_authority.artifacts[policy_ref].payload_runtime_type is tuple
    np.testing.assert_array_equal(
        panel.query("regime_name == 'active'")["consumption"],
        np.full(2 * _N_PERIODS, 0.5),
    )


def test_mutated_cached_authority_is_normalized_before_forward_simulation(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a beartype-invalid cached descriptor before forward execution."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    flat_params = model._process_params(_PARAMS)
    fingerprint = solution.metadata.params_fingerprint
    cached_authority = model_module.build_solution_authority(
        regimes=model._regimes,
        flat_params=flat_params,
        ages=model.ages,
    )
    policy_ref = next(
        ref for ref in cached_authority.artifact_descriptors if ref.key == POLICY_KEY
    )
    object.__setattr__(
        cached_authority.artifact_descriptors[policy_ref],
        "channel",
        "replay",
    )
    model._solution_authorities[fingerprint] = cached_authority

    def fail_if_forward_simulation_starts(**_kwargs: object) -> None:
        raise AssertionError("forward simulation started")

    monkeypatch.setattr(model_module, "simulate", fail_if_forward_simulation_starts)

    with pytest.raises(InvalidSimulationInputError, match="channel"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=solution,
            log_level="off",
        )


def test_hostile_descriptor_key_is_rejected_before_value_materialization() -> None:
    """Never execute descriptor-key equality while checking metadata structure."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    ref = next(iter(solution.metadata.artifact_descriptors))
    descriptor = dataclasses.replace(solution.metadata.artifact_descriptors[ref])
    object.__setattr__(descriptor, "key", _HostileEquality())
    descriptors = dict(solution.metadata.artifact_descriptors)
    descriptors[ref] = descriptor
    metadata = dataclasses.replace(solution.metadata)
    object.__setattr__(
        metadata,
        "artifact_descriptors",
        MappingProxyType(descriptors),
    )
    value_entries: dict[object, object] = {
        (period, regime_name): value
        for period, regime_to_value in solution.values.items()
        for regime_name, value in regime_to_value.items()
    }
    coordinate = next(iter(value_entries))
    lazy_value = _ChangingLazyEntry(
        first=value_entries[coordinate],
        subsequent=value_entries[coordinate],
    )
    value_entries[coordinate] = lazy_value
    malformed = dataclasses.replace(
        solution,
        metadata=metadata,
        values=ValueStore(value_entries),
    )
    object.__setattr__(malformed, "_artifact_authority", solution._artifact_authority)

    with pytest.raises(
        InvalidSimulationInputError,
        match=r"envelope.*Artifact descriptor key",
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=malformed,
            log_level="off",
        )

    assert lazy_value.materialization_count == 0


def test_hostile_cached_value_shape_is_rejected_before_materialization() -> None:
    """Snapshot cached value authority before any lazy value can execute."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    flat_params = model._process_params(_PARAMS)
    fingerprint = solution.metadata.params_fingerprint
    cached = model_module.build_solution_authority(
        regimes=model._regimes,
        flat_params=flat_params,
        ages=model.ages,
    )
    model._solution_authorities[fingerprint] = cached
    authority_coordinate = next(iter(cached.values))
    value_descriptor = cached.values[authority_coordinate]
    object.__setattr__(
        value_descriptor,
        "shape",
        tuple(
            _HostileComparisonInt(size) if index == 0 else size
            for index, size in enumerate(value_descriptor.shape)
        ),
    )
    value_entries: dict[object, object] = {
        (period, regime_name): value
        for period, regime_to_value in solution.values.items()
        for regime_name, value in regime_to_value.items()
    }
    coordinate = next(iter(value_entries))
    lazy_value = _ChangingLazyEntry(
        first=value_entries[coordinate],
        subsequent=value_entries[coordinate],
    )
    value_entries[coordinate] = lazy_value
    malformed = dataclasses.replace(
        solution,
        values=ValueStore(value_entries),
    )
    object.__setattr__(malformed, "_artifact_authority", solution._artifact_authority)

    with pytest.raises(InvalidSimulationInputError, match=r"authority.*shape"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=malformed,
            log_level="off",
        )

    assert lazy_value.materialization_count == 0


def test_type_different_descriptor_axis_is_rejected_before_forward_simulation(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject False-versus-zero metadata before an external route can execute."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    flat_params = model._process_params(_PARAMS)
    fingerprint = solution.metadata.params_fingerprint
    cached = model_module.build_solution_authority(
        regimes=model._regimes,
        flat_params=flat_params,
        ages=model.ages,
    )
    model._solution_authorities[fingerprint] = cached
    ref = next(ref for ref in cached.artifacts if ref.key == POLICY_KEY)
    source_artifact = cached.artifacts[ref]
    coordinates = tuple(range(source_artifact.axes[0].length))

    expected_axis = dataclasses.replace(
        source_artifact.axes[0],
        coordinates=coordinates,
    )
    expected_private_descriptor = dataclasses.replace(
        source_artifact.descriptor,
        named_axes=(
            expected_axis.descriptor,
            *source_artifact.descriptor.named_axes[1:],
        ),
    )
    expected_artifact = dataclasses.replace(
        source_artifact,
        descriptor=expected_private_descriptor,
        axes=(expected_axis, *source_artifact.axes[1:]),
    )
    expected_public_descriptor = dataclasses.replace(
        cached.artifact_descriptors[ref],
        named_axes=(
            dataclasses.replace(
                cached.artifact_descriptors[ref].named_axes[0],
                coordinates=coordinates,
            ),
            *cached.artifact_descriptors[ref].named_axes[1:],
        ),
    )
    model._solution_authorities[fingerprint] = dataclasses.replace(
        cached,
        artifacts=MappingProxyType(dict(cached.artifacts) | {ref: expected_artifact}),
        artifact_descriptors=MappingProxyType(
            dict(cached.artifact_descriptors) | {ref: expected_public_descriptor}
        ),
    )

    supplied_descriptor = dataclasses.replace(
        expected_public_descriptor,
        named_axes=(
            dataclasses.replace(
                expected_public_descriptor.named_axes[0],
                coordinates=(False, *coordinates[1:]),
            ),
            *expected_public_descriptor.named_axes[1:],
        ),
    )
    assert supplied_descriptor == expected_public_descriptor
    metadata = dataclasses.replace(
        solution.metadata,
        artifact_descriptors=(
            dict(solution.metadata.artifact_descriptors) | {ref: supplied_descriptor}
        ),
    )
    malformed = dataclasses.replace(solution, metadata=metadata)
    object.__setattr__(malformed, "_artifact_authority", solution._artifact_authority)

    def fail_if_forward_simulation_starts(**_kwargs: object) -> None:
        raise AssertionError("forward simulation started")

    monkeypatch.setattr(model_module, "simulate", fail_if_forward_simulation_starts)

    with pytest.raises(InvalidSimulationInputError, match="descriptors differ"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=malformed,
            log_level="off",
        )


def test_lazy_artifact_cannot_mutate_authoritative_replay_template() -> None:
    """A decoder receives a disposable template copy and is consumed only once."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    policy_ref = next(ref for ref in solution.replay_artifacts if ref.key == POLICY_KEY)
    policy = cast("Policy", solution.replay_artifacts[policy_ref])
    adversarial = _TemplateMutatingLazyEntry(payload=policy)
    stateful_solution = dataclasses.replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {policy_ref: adversarial}
        ),
    )

    panel = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        solution=stateful_solution,
        log_level="off",
    ).to_dataframe()

    assert adversarial.materialization_count == 1
    np.testing.assert_array_equal(
        panel.query("regime_name == 'active'")["consumption"],
        np.full(2 * _N_PERIODS, 0.5),
    )


def test_lazy_value_cannot_replace_metadata_seen_by_external_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay receives recursively copied metadata validated before decoding."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    private_identity = dataclasses.replace(
        solution.metadata.solver_identities["active"]
    )
    supplied_metadata = dataclasses.replace(
        solution.metadata,
        solver_identities=(
            dict(solution.metadata.solver_identities) | {"active": private_identity}
        ),
    )
    expected_solver_types = dict(supplied_metadata.solver_types)
    expected_plugin_versions = {
        regime: identity.plugin_version
        for regime, identity in supplied_metadata.solver_identities.items()
    }
    value_entries: dict[object, object] = {
        (period, regime_name): value
        for period, regime_to_value in solution.values.items()
        for regime_name, value in regime_to_value.items()
    }
    coordinate = next(iter(value_entries))
    value_entries[coordinate] = _MetadataMutatingLazyEntry(
        value=value_entries[coordinate],
        metadata=supplied_metadata,
        solver_identity=private_identity,
    )
    stateful_solution = dataclasses.replace(
        solution,
        metadata=supplied_metadata,
        values=ValueStore(value_entries),
    )
    observed_solver_types: list[dict[str, str]] = []
    observed_plugin_versions: list[dict[str, str]] = []
    snapshot_type = model_module.ReplayRouteSnapshot

    def _recording_snapshot(**kwargs: Any) -> ReplayRouteSnapshot:
        metadata = cast("SolutionMetadata", kwargs["metadata"])
        observed_solver_types.append(dict(metadata.solver_types))
        observed_plugin_versions.append(
            {
                regime: identity.plugin_version
                for regime, identity in metadata.solver_identities.items()
            }
        )
        return snapshot_type(**kwargs)

    monkeypatch.setattr(model_module, "ReplayRouteSnapshot", _recording_snapshot)

    model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        solution=stateful_solution,
        log_level="off",
    )

    assert dict(supplied_metadata.solver_types) == {"hostile": "replacement"}
    assert private_identity.plugin_version == "hostile-replacement"
    assert observed_solver_types
    assert all(observed == expected_solver_types for observed in observed_solver_types)
    assert all(
        observed == expected_plugin_versions for observed in observed_plugin_versions
    )


def test_lazy_value_cannot_swap_result_channels_after_envelope_snapshot() -> None:
    """Every check and consumer reads stores owned before lazy decoding starts."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    policy_ref = next(ref for ref in solution.replay_artifacts if ref.key == POLICY_KEY)
    failing_replacement = _FailingLazyEntry()
    replacement_replay = ArtifactStore(
        dict(solution.replay_artifacts) | {policy_ref: failing_replacement}
    )
    replacement_omissions = MappingProxyType(
        dict(solution.omissions) | {policy_ref: OmissionReason.UNSUPPORTED}
    )
    value_entries: dict[object, object] = {
        (period, regime_name): value
        for period, regime_to_value in solution.values.items()
        for regime_name, value in regime_to_value.items()
    }
    coordinate = next(iter(value_entries))
    mutating_value = _EnvelopeMutatingLazyEntry(
        value=value_entries[coordinate],
        replacement_replay=replacement_replay,
        replacement_omissions=replacement_omissions,
    )
    value_entries[coordinate] = mutating_value
    stateful_solution = dataclasses.replace(
        solution,
        values=ValueStore(value_entries),
    )
    mutating_value.target = stateful_solution

    panel = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        solution=stateful_solution,
        log_level="off",
    ).to_dataframe()

    assert stateful_solution.replay_artifacts is replacement_replay
    assert stateful_solution.omissions is replacement_omissions
    assert failing_replacement.materialization_count == 0
    np.testing.assert_array_equal(
        panel.query("regime_name == 'active'")["consumption"],
        np.full(2 * _N_PERIODS, 0.5),
    )


def test_external_lazy_decoder_error_is_normalized_before_simulation() -> None:
    """Private decoder failures cross the public boundary as invalid input."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    ref = ArtifactRef(period=0, regime="active", key=POLICY_KEY)
    malformed = dataclasses.replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {ref: _FailingLazyEntry()}
        ),
    )

    with pytest.raises(InvalidSimulationInputError, match="hostile lazy decoder"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=malformed,
            log_level="off",
        )


def test_external_authority_and_route_context_use_each_periods_state_nodes() -> None:
    """Age-specialized coordinates come from each canonical solution cell."""
    model = _model(
        solver=ReferenceSolver(),
        wealth_grid=AgeSpecializedGrid(
            build=_moving_wealth_grid,
            signature=_moving_wealth_signature,
        ),
    )
    solution = _solve(model=model)
    route = model._regimes["active"].simulation.external_replay_route
    assert isinstance(route, ReferenceReplayRoute)

    for period in range(_N_PERIODS):
        expected = tuple(np.linspace(1.0 + period, 3.0 + period, 3))
        for key in (POLICY_KEY, SCRATCH_KEY):
            ref = ArtifactRef(period=period, regime="active", key=key)
            axes = {
                axis.name: axis
                for axis in solution.metadata.artifact_descriptors[ref].named_axes
            }
            assert axes["wealth"].coordinates == expected
        requirement = next(
            context
            for context in route.audit.requirement_contexts
            if context.period == period
        )
        np.testing.assert_array_equal(requirement.state_nodes["wealth"], expected)

    model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        solution=solution,
        log_level="off",
    )

    assert len(route.audit.validation_contexts) == len(route.audit.reader_contexts)
    assert all(
        validated is reader
        for validated, reader in zip(
            route.audit.validation_contexts,
            route.audit.reader_contexts,
            strict=True,
        )
    )
    for context in route.audit.validation_contexts:
        expected = np.linspace(1.0 + context.period, 3.0 + context.period, 3)
        np.testing.assert_array_equal(context.state_nodes["wealth"], expected)


def test_fresh_replay_context_uses_solve_axes_with_a_carried_state(
    tmp_path: Path,
) -> None:
    """Authority and replay share one solve-grid view of a phased carried state."""
    original = _model(solver=ReferenceSolver(), carried_state=True)
    restored = load_solution(
        path=_solve(model=original).save(path=tmp_path / "carried-solution.lcm")
    )
    fresh_model = _model(solver=ReferenceSolver(), carried_state=True)

    panel = fresh_model.simulate(
        params=_PARAMS,
        initial_conditions={
            **_initial_conditions(),
            "pension_wealth": jnp.asarray([2.0, 8.0]),
        },
        solution=restored,
        log_level="off",
    ).to_dataframe()

    regime = fresh_model._regimes["active"]
    assert regime.solution.state_names == ("wealth", "productivity")
    assert regime.simulation.state_names == (
        "wealth",
        "productivity",
        "pension_wealth",
    )
    route = regime.simulation.external_replay_route
    assert isinstance(route, ReferenceReplayRoute)
    contexts = (
        *route.audit.requirement_contexts,
        *route.audit.validation_contexts,
        *route.audit.reader_contexts,
    )
    assert contexts
    assert all(
        context.state_names == ("wealth", "productivity") for context in contexts
    )
    assert all(
        tuple(context.state_nodes) == ("wealth", "productivity") for context in contexts
    )
    pension = panel.query("regime_name == 'active'").pivot(
        index="period",
        columns="subject_id",
        values="pension_wealth",
    )
    np.testing.assert_array_equal(
        pension.to_numpy(),
        np.tile(np.asarray([[2.0, 8.0]]), (_N_PERIODS, 1)),
    )


def test_structurally_invalid_custom_artifact_is_rejected_before_simulation() -> None:
    """Generic preflight refuses a replay leaf outside its model authority."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    ref = ArtifactRef(period=0, regime="active", key=POLICY_KEY)
    policy = cast("Policy", solution.replay_artifacts[ref])
    malformed = dataclasses.replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts)
            | {ref: Policy(values=jnp.asarray([0.5], dtype=policy.values.dtype))}
        ),
    )

    with pytest.raises(InvalidSimulationInputError, match=r"shape|authority|invalid"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=malformed,
            log_level="off",
        )


def test_mathematically_invalid_custom_artifact_is_rejected_by_the_route() -> None:
    """Plugin validation runs during preflight, before a reader enters simulation."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    ref = ArtifactRef(period=0, regime="active", key=POLICY_KEY)
    policy = cast("Policy", solution.replay_artifacts[ref])
    wrong_tie_winner = Policy(values=jnp.zeros_like(policy.values))
    malformed = dataclasses.replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {ref: wrong_tie_winner}
        ),
    )

    with pytest.raises(InvalidSimulationInputError, match="middle-action tie"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=malformed,
            log_level="off",
        )


def test_mutable_plugin_payload_container_is_rejected_by_authority() -> None:
    """A plugin cannot declare a mutable container as an engine-owned snapshot."""

    @dataclasses.dataclass(kw_only=True)
    class MutablePolicy:
        values: jax.Array

    jax.tree_util.register_dataclass(
        MutablePolicy,
        data_fields=["values"],
        meta_fields=[],
    )
    template = jnp.zeros(3)
    leaf = LeafAuthority(
        path=("attribute:values",),
        runtime_type=jax.Array,
        shape=(3,),
        dtype=str(template.dtype),
        axis_names=("wealth",),
    )
    axis = AxisAuthority(
        name="wealth",
        length=3,
        role=AxisRole.STATE,
        coordinates=(1.0, 2.0, 3.0),
    )
    descriptor = ArtifactDescriptor(
        key=POLICY_KEY,
        channel=ArtifactChannel.REPLAY,
        persistence=PersistencePolicy.MODEL_VERIFIABLE,
        payload_type_id="tests.MutablePolicy",
        leaf_descriptors=(
            LeafDescriptor(
                path=leaf.path,
                shape=leaf.shape,
                dtype=leaf.dtype,
                axis_names=leaf.axis_names,
            ),
        ),
        named_axes=(axis.descriptor,),
        state_roles=("wealth",),
    )

    with pytest.raises(TypeError, match="closed dataclass record"):
        ArtifactAuthority(
            descriptor=descriptor,
            payload_runtime_type=MutablePolicy,
            template=MutablePolicy(values=template),
            container_runtime_types={(): MutablePolicy},
            leaves={leaf.path: leaf},
            axes=(axis,),
            state_roles=("wealth",),
        )


def test_artifact_authority_rejects_type_different_axis_coordinates() -> None:
    """Bool and int coordinates cannot satisfy one authoritative axis relation."""
    values = jnp.ones(1, dtype=jnp.float32)
    leaf = LeafAuthority(
        path=(),
        runtime_type=jax.Array,
        shape=(1,),
        dtype=str(values.dtype),
        axis_names=("candidate",),
    )
    descriptor = ArtifactDescriptor(
        key=POLICY_KEY,
        channel=ArtifactChannel.REPLAY,
        persistence=PersistencePolicy.MODEL_VERIFIABLE,
        payload_type_id="jax.Array",
        leaf_descriptors=(leaf.descriptor,),
        named_axes=(
            AxisDescriptor(
                name="candidate",
                length=1,
                role=AxisRole.CANDIDATE,
                coordinates=(False,),
            ),
        ),
    )
    axis = AxisAuthority(
        name="candidate",
        length=1,
        role=AxisRole.CANDIDATE,
        coordinates=(0,),
    )
    assert descriptor.named_axes == (axis.descriptor,)

    with pytest.raises(ValueError, match="descriptive axes"):
        ArtifactAuthority(
            descriptor=descriptor,
            payload_runtime_type=jax.Array,
            template=values,
            leaves={leaf.path: leaf},
            axes=(axis,),
        )


def test_artifact_canonicalization_rejects_type_different_static_metadata() -> None:
    """Reject weak equality and caller-owned custom static metadata."""

    @dataclasses.dataclass(frozen=True)
    class Meta:
        flag: int

    @dataclasses.dataclass(frozen=True)
    class TaggedPolicy:
        values: object
        tag: object

    jax.tree_util.register_dataclass(
        TaggedPolicy,
        data_fields=["values"],
        meta_fields=["tag"],
    )
    values = jnp.ones(1, dtype=jnp.float32)
    leaf = LeafAuthority(
        path=("attribute:values",),
        runtime_type=jax.Array,
        shape=(1,),
        dtype=str(values.dtype),
        axis_names=("x",),
    )
    axis = AxisAuthority(
        name="x",
        length=1,
        role=AxisRole.OTHER,
        coordinates=(0,),
    )
    descriptor = ArtifactDescriptor(
        key=POLICY_KEY,
        channel=ArtifactChannel.REPLAY,
        persistence=PersistencePolicy.MODEL_VERIFIABLE,
        payload_type_id="tests.TaggedPolicy",
        leaf_descriptors=(leaf.descriptor,),
        named_axes=(axis.descriptor,),
    )
    authority = ArtifactAuthority(
        descriptor=descriptor,
        payload_runtime_type=TaggedPolicy,
        template=TaggedPolicy(values=values, tag=1),
        container_runtime_types={(): TaggedPolicy},
        leaves={leaf.path: leaf},
        axes=(axis,),
    )
    supplied = TaggedPolicy(values=values, tag=True)
    assert jax.tree_util.tree_structure(supplied) == jax.tree_util.tree_structure(
        authority.template
    )

    with pytest.raises(TypeError, match="PyTree static metadata"):
        solver_api_module._canonicalize_artifact_payload(
            payload=supplied,
            authority=authority,
        )

    with pytest.raises(TypeError, match=r"static metadata.*Meta"):
        ArtifactAuthority(
            descriptor=descriptor,
            payload_runtime_type=TaggedPolicy,
            template=TaggedPolicy(values=values, tag=Meta(flag=1)),
            container_runtime_types={(): TaggedPolicy},
            leaves={leaf.path: leaf},
            axes=(axis,),
        )


def test_artifact_ledger_rejects_unaccounted_and_undeclared_standard_refs() -> None:
    """Require accounting for every descriptor; standard keys cannot bypass it."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    counter_ref = _refs(key=COUNTER_KEY)[0]
    missing_accounting = dataclasses.replace(
        solution,
        omissions={
            ref: reason
            for ref, reason in solution.omissions.items()
            if ref != counter_ref
        },
    )

    with pytest.raises(InvalidSimulationInputError, match="missing accounting"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=missing_accounting,
            log_level="off",
        )

    injected_ref = ArtifactRef(
        period=0,
        regime="active",
        key=ArtifactKey(type_id="pylcm.egm.continuation", schema_version=1),
    )
    undeclared = dataclasses.replace(
        solution,
        retained_continuations=ArtifactStore(
            dict(solution.retained_continuations)
            | {injected_ref: solution.value(period=0, regime="active")}
        ),
    )

    with pytest.raises(InvalidSimulationInputError, match="undeclared"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=undeclared,
            log_level="off",
        )


def test_exact_nonpersisted_diagnostic_omission_is_replay_safe() -> None:
    """A persisted diagnostic omission is outside the §11 artifact partition."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    diagnostic_ref = ArtifactRef(
        period=0,
        regime="active",
        key=SOLVER_DIAGNOSTICS,
    )
    restored_shape = dataclasses.replace(
        solution,
        omissions={
            **solution.omissions,
            diagnostic_ref: OmissionReason.NOT_PERSISTED,
        },
    )

    result = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        solution=restored_shape,
        log_level="off",
    )

    assert not result.to_dataframe().empty


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        (SOLVER_DIAGNOSTICS, OmissionReason.NOT_REQUESTED),
        (
            ArtifactKey(
                type_id=SOLVER_DIAGNOSTICS.type_id,
                schema_version=SOLVER_DIAGNOSTICS.schema_version + 1,
            ),
            OmissionReason.NOT_PERSISTED,
        ),
    ],
)
def test_only_exact_nonpersisted_diagnostic_omission_bypasses_descriptors(
    *, key: ArtifactKey, reason: OmissionReason
) -> None:
    """Near-miss diagnostic omissions remain fail-closed."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    ref = ArtifactRef(period=0, regime="active", key=key)
    malformed = dataclasses.replace(
        solution,
        omissions={**solution.omissions, ref: reason},
    )

    with pytest.raises(InvalidSimulationInputError, match=r"undeclared|schema"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=malformed,
            log_level="off",
        )


class _StringSubclass(str):
    """Equal string impostor used to exercise exact-type validation."""

    __slots__ = ()


class _IntegerSubclass(int):
    """Equal integer impostor used to exercise exact-type validation."""

    __slots__ = ()


class _ArtifactKeySubclass(
    ArtifactKey  # ty: ignore[subclass-of-dataclass-with-order]
):
    """Structurally equal key subclass that must not be accepted as an address."""

    __slots__ = ()


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: ArtifactKey(type_id=_StringSubclass("plugin.key")),
            id="artifact-key-type-id",
        ),
        pytest.param(
            lambda: ArtifactKey(
                type_id="plugin.key", schema_version=_IntegerSubclass(1)
            ),
            id="artifact-key-version",
        ),
        pytest.param(
            lambda: SolverIdentity(
                plugin_id=_StringSubclass("plugin"), plugin_version="1"
            ),
            id="solver-plugin-id",
        ),
        pytest.param(
            lambda: SolverIdentity(
                plugin_id="plugin", plugin_version=_StringSubclass("1")
            ),
            id="solver-plugin-version",
        ),
        pytest.param(
            lambda: SolverIdentity(
                plugin_id="plugin", plugin_version="1", solver_api_version=True
            ),
            id="solver-api-version-bool",
        ),
        pytest.param(
            lambda: ArtifactRef(period=True, regime="active", key=POLICY_KEY),
            id="artifact-ref-period-bool",
        ),
        pytest.param(
            lambda: ArtifactRef(
                period=0, regime=_StringSubclass("active"), key=POLICY_KEY
            ),
            id="artifact-ref-regime",
        ),
        pytest.param(
            lambda: ArtifactRef(
                period=0,
                regime="active",
                key=_ArtifactKeySubclass(type_id="plugin.key"),
            ),
            id="artifact-ref-key",
        ),
    ],
)
def test_public_identity_and_address_types_are_exact(
    *, factory: Callable[[], object]
) -> None:
    """Equal subclasses and bool-as-int values fail at the public boundary."""
    with pytest.raises(TypeError, match="exact"):
        factory()


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("axis_type", [AxisDescriptor, AxisAuthority])
def test_artifact_axis_coordinates_must_be_finite(
    *, coordinate: float, axis_type: Callable[..., object]
) -> None:
    """Neither descriptive nor authoritative axes admit non-finite coordinates."""
    with pytest.raises(ValueError, match="finite"):
        axis_type(
            name="wealth",
            length=1,
            role=AxisRole.STATE,
            coordinates=(coordinate,),
        )


def test_artifact_semantic_roles_must_exist_in_the_canonical_model() -> None:
    """A role with no corresponding axis still cannot evade model authority."""
    model = _model(solver=_BadRoleSolver())

    with pytest.raises(TypeError, match="semantic roles"):
        _solve(model=model)


def test_artifact_axis_cannot_launder_a_state_role_as_other() -> None:
    """A model role axis cannot evade canonical coordinate binding via OTHER."""
    with pytest.raises(ValueError, match="state role must have STATE"):
        ArtifactDescriptor(
            key=POLICY_KEY,
            channel=ArtifactChannel.REPLAY,
            persistence=PersistencePolicy.MODEL_VERIFIABLE,
            payload_type_id="jax.Array",
            named_axes=(
                AxisDescriptor(
                    name="wealth",
                    length=1,
                    role=AxisRole.OTHER,
                    coordinates=(0.0,),
                ),
            ),
            state_roles=("wealth",),
        )

    model = _model(solver=_LaunderedAxisSolver())
    with pytest.raises(TypeError, match="canonical model state"):
        _solve(model=model)


def test_custom_diagnostic_authority_is_rejected_without_a_public_channel() -> None:
    """Authorities cannot claim a channel KernelOutput cannot publish on."""
    with pytest.raises(TypeError, match="cannot declare custom DIAGNOSTIC"):
        _model(solver=_DiagnosticAuthoritySolver())


def test_custom_authority_cannot_reuse_an_engine_owned_artifact_type_id() -> None:
    """Version changes cannot turn a built-in artifact namespace into an extension."""
    with pytest.raises(ValueError, match="engine-owned artifact type ids"):
        _model(solver=_ReservedArtifactAuthoritySolver())


def test_producer_payload_type_must_match_the_solver_built_authority() -> None:
    """A producer declaration cannot contradict the model's consuming authority."""
    model = _model(solver=_ConflictingProducerTypeSolver())

    with pytest.raises(TypeError, match="disagree with solver-built"):
        _solve(model=model)


@pytest.mark.parametrize(
    "defect",
    ["channel", "applicable", "required", "payload_type", "template_dtype"],
)
def test_custom_continuation_authority_must_match_its_producer_spec(
    *, defect: str
) -> None:
    """A custom authority cannot silently replace contradictory producer facts."""
    model = _model(solver=_ContradictoryContinuationAuthoritySolver(defect=defect))

    with pytest.raises(TypeError, match="authority shadowing a ContinuationSpec"):
        _solve(model=model)


def test_required_external_replay_artifact_must_have_a_producer() -> None:
    """A route cannot require an authority-only artifact no core can publish."""
    model = _model(solver=_MissingRequiredProducerSolver())

    with pytest.raises(TypeError, match="no CoreProgram producer for required"):
        _solve(model=model)


def test_required_auxiliary_artifact_must_have_a_producer() -> None:
    """A required additive authority cannot exist without a producing program."""
    model = _model(solver=_MissingRequiredAuxiliaryProducerSolver())

    with pytest.raises(TypeError, match="no CoreProgram producer for required"):
        _solve(model=model)


def test_a_producer_cannot_publish_an_inapplicable_custom_artifact() -> None:
    """Default replay retention refuses a present, inapplicable artifact."""
    model = _model(solver=_InapplicablePublishedArtifactSolver())

    with pytest.raises(
        RuntimeError,
        match=(
            r"Regime 'active'.*period 2.*published replay artifact "
            r"'tests\.conformance_solver\.optional_replay'.*not applicable"
        ),
    ):
        model.solve(params=_PARAMS, log_level="off")


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("source", "persisted"),
        ("model_instance_id", 1),
        ("pylcm_version", 1),
        ("solver_api_version", True),
        ("solution_schema_version", True),
    ],
)
def test_mutated_metadata_identity_types_fail_before_simulation(
    *, field: str, invalid: object
) -> None:
    """Identity/version fields cannot use equal or branch-bypassing impostors."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    metadata = dataclasses.replace(solution.metadata, source=SolutionSource.PERSISTED)
    object.__setattr__(metadata, field, invalid)
    malformed = dataclasses.replace(solution, metadata=metadata)

    with pytest.raises(InvalidSimulationInputError, match=field):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions(),
            solution=malformed,
            log_level="off",
        )
