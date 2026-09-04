"""Small solver implemented exclusively against pylcm's public extension API."""

import dataclasses
import functools
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Self, cast

import jax
import jax.numpy as jnp
import numpy as np

from lcm.solver_api import (
    ActionOutput,
    ArtifactAuthority,
    ArtifactChannel,
    ArtifactDescriptor,
    ArtifactKey,
    AxisAuthority,
    AxisDescriptor,
    AxisRole,
    ExecutableReplayRoute,
    KernelOutput,
    LeafAuthority,
    LeafDescriptor,
    PersistencePolicy,
    ReplayMode,
    ReplayModelContext,
    ReplayReader,
    ReplayRouteIdentity,
    ReplayRouteRequirements,
    ReplayRouteSnapshot,
    SimulationBuildContext,
    SolverIdentity,
)
from lcm.solvers import (
    ContinuationSpec,
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    OutputRole,
    ProgramScope,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    StateActionSpace,
    StateAxesLeading,
    StreamableProductAxis,
)
from lcm.typing import Float1D, FloatND, RegimeName

COUNTER_KEY = ArtifactKey(
    type_id="tests.conformance_solver.counter",
    schema_version=1,
)
POLICY_KEY = ArtifactKey(
    type_id="tests.conformance_solver.middle_tie_policy",
    schema_version=1,
)
SCRATCH_KEY = ArtifactKey(
    type_id="tests.conformance_solver.scratch",
    schema_version=1,
)
OPTIONAL_REPLAY_KEY = ArtifactKey(
    type_id="tests.conformance_solver.optional_replay",
    schema_version=1,
)

_PLUGIN_IDENTITY = SolverIdentity(
    plugin_id="tests.conformance_solver",
    plugin_version="1.0.0",
)
_ROUTE_IDENTITY = ReplayRouteIdentity(
    route_id="tests.conformance_solver.middle_tie",
    route_version=1,
)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["count"],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True, kw_only=True)
class Counter:
    """Non-EGM continuation carrying the remaining-horizon count."""

    count: FloatND

    @property
    def artifact_key(self) -> ArtifactKey:
        """Return the continuation's versioned artifact identity."""
        return COUNTER_KEY


def _counter_authority(*, template: Counter) -> ArtifactAuthority:
    """Describe the shared scalar continuation through the public API."""
    dtype = str(template.count.dtype)
    leaf = LeafAuthority(
        path=("attribute:count",),
        runtime_type=jax.Array,
        shape=(),
        dtype=dtype,
        axis_names=(),
    )
    return ArtifactAuthority(
        descriptor=ArtifactDescriptor(
            key=COUNTER_KEY,
            channel=ArtifactChannel.CONTINUATION,
            persistence=PersistencePolicy.NOT_PERSISTED,
            payload_type_id="tests.conformance_solver.Counter",
            leaf_descriptors=(
                LeafDescriptor(
                    path=leaf.path,
                    shape=leaf.shape,
                    dtype=leaf.dtype,
                    axis_names=leaf.axis_names,
                ),
            ),
            required=True,
        ),
        payload_runtime_type=Counter,
        template=template,
        container_runtime_types={(): Counter},
        leaves={leaf.path: leaf},
        required=True,
    )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["values"],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True, kw_only=True)
class Policy:
    """Plugin-defined replay PyTree wrapping the selected action values."""

    values: FloatND


@dataclasses.dataclass(frozen=True)
class _MiddleTieReduction:
    """Planner identity for maximum-value reduction with a middle tie winner."""

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the stable numerical identity of this reduction."""
        return ("middle-tie-maximum", 1)


MIDDLE_TIE_REDUCTION = _MiddleTieReduction()


def _value_and_counter(
    *,
    wealth: Float1D,
    productivity: Float1D,
    next_count: FloatND,
) -> tuple[FloatND, Counter]:
    """Return the common value and continuation without replay assembly."""
    count = next_count + 1.0
    value = count * (wealth[:, None] + 0.0 * productivity[None, :])
    return value, Counter(count=count)


def _solve_values_core(
    *,
    wealth: Float1D,
    productivity: Float1D,
    consumption: Float1D,  # noqa: ARG001
    next_count: FloatND,
    candidate_width: int,  # noqa: ARG001
) -> tuple[FloatND, Counter]:
    """Solve values while deliberately constructing no replay payload."""
    return _value_and_counter(
        wealth=wealth,
        productivity=productivity,
        next_count=next_count,
    )


def _solve_replay_core(
    *,
    wealth: Float1D,
    productivity: Float1D,
    consumption: Float1D,
    next_count: FloatND,
    candidate_width: int,  # noqa: ARG001
) -> tuple[FloatND, Counter, Policy, FloatND]:
    """Solve values and assemble the selected replay artifact."""
    value, counter = _value_and_counter(
        wealth=wealth,
        productivity=productivity,
        next_count=next_count,
    )
    middle = consumption[consumption.shape[0] // 2]
    policy = Policy(
        values=jnp.full(
            (wealth.shape[0], productivity.shape[0]),
            middle,
            dtype=wealth.dtype,
        )
    )
    return value, counter, policy, jnp.zeros_like(value)


def _build_scratch_core(
    *,
    wealth: Float1D,
    productivity: Float1D,
    consumption: Float1D,  # noqa: ARG001
    next_count: FloatND,
    candidate_width: int,  # noqa: ARG001
) -> tuple[FloatND, FloatND]:
    """Build the independently selected model-verifiable auxiliary artifact."""
    value, _counter = _value_and_counter(
        wealth=wealth,
        productivity=productivity,
        next_count=next_count,
    )
    return value, jnp.zeros_like(value)


def _terminal_value_and_counter() -> tuple[FloatND, Counter]:
    """Publish the terminal scalar value and the counter's zero boundary."""
    zero = jnp.asarray(0.0)
    return zero, Counter(count=zero)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TerminalArgumentBuilder:
    """Build the empty dynamic argument map for the terminal scalar core."""

    def __call__(self, _context: CoreBuildContext) -> Mapping[str, object]:
        """Return the terminal core's public empty argument view."""
        return MappingProxyType({})


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ArgumentBuilder:
    """Build dynamic arguments for one reference-solver core."""

    regime_name: RegimeName

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Read only public state, action, and continuation views."""
        state_action_space = cast("StateActionSpace", context.state_action_space)
        continuation = cast(
            "Counter",
            context.next_regime_to_continuation[self.regime_name],
        )
        return MappingProxyType(
            {
                "wealth": state_action_space.states["wealth"],
                "productivity": state_action_space.states["productivity"],
                "consumption": state_action_space.actions["consumption"],
                "next_count": continuation.count,
            }
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PeriodKernel:
    """Uniform period adapter dispatching one retention-scoped core."""

    programs: Mapping[str, CoreProgram]

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return the core graph consumed by the common planner."""
        return self.programs

    def with_fixed_params(
        self,
        *,
        fixed_flat_params: object,  # noqa: ARG002
    ) -> Self:
        """Return this parameter-free kernel unchanged."""
        return self

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable[..., object]],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[str, object],
        next_regime_to_continuation: Mapping[str, object],
        flat_params: Mapping[str, object],
        period: int,
        ages: object,
        logger: object,  # noqa: ARG002
        **_unused: object,
    ) -> KernelOutput:
        """Execute only the planner-selected retention-scoped core."""
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        selected_name = "replay" if "replay" in compiled_cores else "values"
        arguments = self.programs[selected_name].argument_builder(context)
        raw_output = compiled_cores[selected_name](**arguments)
        if selected_name == "replay":
            value, continuation, policy, optional_replay = cast(
                "tuple[FloatND, Counter, Policy, FloatND]", raw_output
            )
            scratch = (
                cast(
                    "tuple[FloatND, FloatND]",
                    compiled_cores["scratch"](**arguments),
                )[1]
                if "scratch" in compiled_cores
                else None
            )
            return KernelOutput(
                value=value,
                continuations={COUNTER_KEY: continuation},
                replay={
                    POLICY_KEY: policy,
                    OPTIONAL_REPLAY_KEY: optional_replay,
                },
                auxiliary={} if scratch is None else {SCRATCH_KEY: scratch},
            )
        value, continuation = cast("tuple[FloatND, Counter]", raw_output)
        return KernelOutput(
            value=value,
            continuations={COUNTER_KEY: continuation},
        )


class TerminalCounterSolver(Solver):
    """Publish the scalar zero boundary required by ``ReferenceSolver``."""

    @property
    def identity(self) -> SolverIdentity:
        """Return the shared external plugin's durable identity."""
        return _PLUGIN_IDENTITY

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one dense terminal program and its counter authority."""
        zero = jnp.asarray(0.0)
        counter_template = Counter(count=zero)
        counter_authority = _counter_authority(template=counter_template)
        program = CoreProgram(
            name="values",
            function=_terminal_value_and_counter,
            argument_builder=_TerminalArgumentBuilder(),
            requirements=CoreExecutionRequirements(),
            output_roles=(
                OutputRole.VALUE,
                Counter(
                    count=StateAxesLeading(  # ty: ignore[invalid-argument-type]
                        state_names=(),
                        dtype=zero.dtype,
                        shape=(),
                    )
                ),
            ),
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason="scalar_terminal_counter",
            scope=ProgramScope.VALUES_ONLY,
        )
        kernels = MappingProxyType(
            {
                period: _PeriodKernel(programs=MappingProxyType({"values": program}))
                for period in context.regimes_to_active_periods[context.regime_name]
            }
        )
        return SolutionKernels(
            period_kernels=kernels,
            continuation_spec=ContinuationSpec(
                template=counter_template,
                artifact_key=COUNTER_KEY,
            ),
            artifact_authorities=MappingProxyType({COUNTER_KEY: counter_authority}),
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ReferenceReader:
    """JAX-transformable interpolation reader for one validated policy."""

    state_nodes: Float1D
    policy: Policy

    def __call__(
        self,
        *,
        states: Mapping[str, object],
        fallback_actions: Mapping[str, object],  # noqa: ARG002
    ) -> ActionOutput:
        """Read the stored policy at each simulated wealth value."""
        wealth = jnp.asarray(states["wealth"])
        return ActionOutput(
            actions={"consumption": jnp.full_like(wealth, self.policy.values[0, 0])}
        )


@dataclasses.dataclass(kw_only=True)
class _RouteAudit:
    """Record object identities without participating in compiled execution."""

    validated_snapshots: list[int] = dataclasses.field(default_factory=list)
    reader_snapshots: list[int] = dataclasses.field(default_factory=list)
    requirement_contexts: list[ReplayModelContext] = dataclasses.field(
        default_factory=list
    )
    validation_contexts: list[SimulationBuildContext] = dataclasses.field(
        default_factory=list
    )
    reader_contexts: list[SimulationBuildContext] = dataclasses.field(
        default_factory=list
    )
    validation_artifact_keys: list[tuple[ArtifactKey, ...]] = dataclasses.field(
        default_factory=list
    )
    reader_artifact_keys: list[tuple[ArtifactKey, ...]] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReferenceReplayRoute(ExecutableReplayRoute):
    """Exact replay of the solver's middle-action tie convention."""

    selected_action: FloatND
    audit: _RouteAudit = dataclasses.field(
        default_factory=_RouteAudit, compare=False, repr=False
    )

    @property
    def identity(self) -> ReplayRouteIdentity:
        """Return the stable route identity."""
        return _ROUTE_IDENTITY

    @property
    def plugin_identity(self) -> SolverIdentity:
        """Return the installed plugin identity."""
        return _PLUGIN_IDENTITY

    @property
    def replay_mode(self) -> ReplayMode:
        """Declare exact solve-time replay."""
        return ReplayMode.EXACT_REPLAY

    @property
    def payload_type(self) -> type[object]:
        """Return the exact plugin-defined policy PyTree type."""
        return Policy

    @property
    def policy_applicable(self) -> bool:
        """Declare that every active period publishes a policy."""
        return True

    @property
    def policy_required(self) -> bool:
        """Declare that simulation requires every policy cell."""
        return True

    @property
    def consumer_route(self) -> str:
        """Return the stable reader route identifier."""
        return self.identity.route_id

    def requirements(self, *, context: ReplayModelContext) -> ReplayRouteRequirements:
        """Require the policy artifact for the declared two-state model."""
        if context.state_names != ("wealth", "productivity"):
            raise ValueError("The reference route requires its two declared states.")
        if context.action_names != ("consumption",):
            raise ValueError("The reference route requires the consumption action.")
        self.audit.requirement_contexts.append(context)
        return ReplayRouteRequirements(required_artifacts=frozenset({POLICY_KEY}))

    def validate(
        self,
        *,
        snapshot: ReplayRouteSnapshot,
        context: SimulationBuildContext,
    ) -> None:
        """Require the policy to obey the solver's declared tie convention."""
        if context.state_names != ("wealth", "productivity"):
            raise ValueError("The replay validation context has wrong state roles.")
        self.audit.validated_snapshots.append(id(snapshot))
        self.audit.validation_contexts.append(context)
        self.audit.validation_artifact_keys.append(tuple(snapshot.artifacts))
        policy = cast("Policy", snapshot.artifacts[POLICY_KEY])
        values = np.asarray(policy.values)
        expected = np.full(values.shape, np.asarray(self.selected_action))
        if not np.array_equal(values, expected):
            raise ValueError(
                "The replay policy violates the middle-action tie convention."
            )

    def build_reader(
        self,
        *,
        snapshot: ReplayRouteSnapshot,
        context: SimulationBuildContext,
    ) -> ReplayReader:
        """Build the pure array reader from the validated immutable snapshot."""
        if context.state_names != ("wealth", "productivity"):
            raise ValueError("The reference route requires exactly its two states.")
        if context.action_names != ("consumption",):
            raise ValueError(
                "The reference route requires exactly the consumption action."
            )
        self.audit.reader_snapshots.append(id(snapshot))
        self.audit.reader_contexts.append(context)
        self.audit.reader_artifact_keys.append(tuple(snapshot.artifacts))
        return _ReferenceReader(
            state_nodes=context.state_nodes["wealth"],
            policy=cast("Policy", snapshot.artifacts[POLICY_KEY]),
        )


class ReferenceSolver(Solver):
    """Value solver with a custom continuation, replay route, and artifact ledger."""

    @property
    def identity(self) -> SolverIdentity:
        """Return this external solver's durable plugin identity."""
        return _PLUGIN_IDENTITY

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        """Require this solver's counter from each reachable target."""
        return frozenset({COUNTER_KEY})

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one planner-owned program and its model-derived authorities."""
        state_nodes = jnp.asarray(context.state_action_space.states["wealth"])
        productivity_nodes = jnp.asarray(
            context.state_action_space.states["productivity"]
        )
        action_nodes = jnp.asarray(context.state_action_space.actions["consumption"])
        selected_action = action_nodes[action_nodes.shape[0] // 2]
        dtype = str(state_nodes.dtype)
        state_shape = (state_nodes.shape[0], productivity_nodes.shape[0])
        wealth_axis = AxisAuthority(
            name="wealth",
            length=state_shape[0],
            role=AxisRole.STATE,
        )
        wealth_axis_descriptor = AxisDescriptor(
            name="wealth",
            length=state_shape[0],
            role=AxisRole.STATE,
        )
        productivity_axis = AxisAuthority(
            name="productivity",
            length=productivity_nodes.shape[0],
            role=AxisRole.STATE,
        )
        productivity_axis_descriptor = AxisDescriptor(
            name="productivity",
            length=productivity_nodes.shape[0],
            role=AxisRole.STATE,
        )
        policy_leaf = LeafAuthority(
            path=("attribute:values",),
            runtime_type=jax.Array,
            shape=state_shape,
            dtype=dtype,
            axis_names=("wealth", "productivity"),
        )
        policy_leaf_descriptor = LeafDescriptor(
            path=("attribute:values",),
            shape=state_shape,
            dtype=dtype,
            axis_names=("wealth", "productivity"),
        )
        scratch_leaf = LeafAuthority(
            path=(),
            runtime_type=jax.Array,
            shape=state_shape,
            dtype=dtype,
            axis_names=("wealth", "productivity"),
        )
        scratch_leaf_descriptor = LeafDescriptor(
            path=(),
            shape=state_shape,
            dtype=dtype,
            axis_names=("wealth", "productivity"),
        )

        counter_template = Counter(count=jnp.zeros((), dtype=state_nodes.dtype))
        counter_authority = _counter_authority(
            template=counter_template,
        )
        policy_authority = ArtifactAuthority(
            descriptor=ArtifactDescriptor(
                key=POLICY_KEY,
                channel=ArtifactChannel.REPLAY,
                persistence=PersistencePolicy.MODEL_VERIFIABLE,
                payload_type_id="tests.conformance_solver.Policy",
                leaf_descriptors=(policy_leaf_descriptor,),
                named_axes=(wealth_axis_descriptor, productivity_axis_descriptor),
                state_roles=("wealth", "productivity"),
                action_roles=("consumption",),
                required_for=frozenset({_ROUTE_IDENTITY}),
                required=True,
            ),
            payload_runtime_type=Policy,
            template=Policy(values=jnp.zeros(state_shape, dtype=state_nodes.dtype)),
            container_runtime_types={(): Policy},
            leaves={policy_leaf.path: policy_leaf},
            axes=(wealth_axis, productivity_axis),
            state_roles=("wealth", "productivity"),
            action_roles=("consumption",),
            consumer_route=_ROUTE_IDENTITY,
            required=True,
        )
        scratch_authority = ArtifactAuthority(
            descriptor=ArtifactDescriptor(
                key=SCRATCH_KEY,
                channel=ArtifactChannel.AUXILIARY,
                persistence=PersistencePolicy.MODEL_VERIFIABLE,
                payload_type_id="jax.Array",
                leaf_descriptors=(scratch_leaf_descriptor,),
                named_axes=(wealth_axis_descriptor, productivity_axis_descriptor),
                state_roles=("wealth", "productivity"),
                required=True,
            ),
            payload_runtime_type=jax.Array,
            template=jnp.zeros(state_shape, dtype=state_nodes.dtype),
            leaves={scratch_leaf.path: scratch_leaf},
            axes=(wealth_axis, productivity_axis),
            state_roles=("wealth", "productivity"),
            required=True,
        )
        optional_replay_authority = ArtifactAuthority(
            descriptor=ArtifactDescriptor(
                key=OPTIONAL_REPLAY_KEY,
                channel=ArtifactChannel.REPLAY,
                persistence=PersistencePolicy.MODEL_VERIFIABLE,
                payload_type_id="jax.Array",
                leaf_descriptors=(scratch_leaf_descriptor,),
                named_axes=(wealth_axis_descriptor, productivity_axis_descriptor),
                state_roles=("wealth", "productivity"),
            ),
            payload_runtime_type=jax.Array,
            template=jnp.zeros(state_shape, dtype=state_nodes.dtype),
            leaves={scratch_leaf.path: scratch_leaf},
            axes=(wealth_axis, productivity_axis),
            state_roles=("wealth", "productivity"),
        )
        route = ReferenceReplayRoute(
            selected_action=selected_action,
        )

        argument_builder = _ArgumentBuilder(regime_name=context.regime_name)
        requirements = CoreExecutionRequirements(
            streamable_axes=(
                StreamableProductAxis(
                    name="candidate",
                    coordinate_names=("consumption",),
                    coordinate_extents=(int(action_nodes.shape[0]),),
                    canonical_order="c",
                    reduction=MIDDLE_TIE_REDUCTION,
                    width_keyword="candidate_width",
                ),
            )
        )
        values_program = CoreProgram(
            name="values",
            function=_solve_values_core,
            argument_builder=argument_builder,
            requirements=requirements,
            output_roles=(
                OutputRole.VALUE,
                Counter(
                    count=StateAxesLeading(  # ty: ignore[invalid-argument-type]
                        state_names=(),
                        dtype=state_nodes.dtype,
                        shape=(),
                    )
                ),
            ),
            disposition=CoreExecutionDisposition.PLANNED,
            scope=ProgramScope.VALUES_ONLY,
        )
        replay_program = CoreProgram(
            name="replay",
            function=_solve_replay_core,
            argument_builder=argument_builder,
            requirements=requirements,
            output_roles=(
                OutputRole.VALUE,
                Counter(
                    count=StateAxesLeading(  # ty: ignore[invalid-argument-type]
                        state_names=(),
                        dtype=state_nodes.dtype,
                        shape=(),
                    )
                ),
                Policy(
                    values=StateAxesLeading(  # ty: ignore[invalid-argument-type]
                        state_names=("wealth", "productivity"),
                        dtype=state_nodes.dtype,
                        shape=state_shape,
                    )
                ),
                StateAxesLeading(
                    state_names=("wealth", "productivity"),
                    dtype=state_nodes.dtype,
                    shape=state_shape,
                ),
            ),
            disposition=CoreExecutionDisposition.PLANNED,
            scope=ProgramScope.REPLAY,
            retained_artifact_keys=(POLICY_KEY, OPTIONAL_REPLAY_KEY),
            retained_artifact_payload_types={
                POLICY_KEY: Policy,
                OPTIONAL_REPLAY_KEY: jax.Array,
            },
            replaces_program="values",
        )
        scratch_program = CoreProgram(
            name="scratch",
            function=_build_scratch_core,
            argument_builder=argument_builder,
            requirements=requirements,
            output_roles=(
                OutputRole.VALUE,
                StateAxesLeading(
                    state_names=("wealth", "productivity"),
                    dtype=state_nodes.dtype,
                    shape=state_shape,
                ),
            ),
            disposition=CoreExecutionDisposition.PLANNED,
            scope=ProgramScope.ARTIFACT,
            retained_artifact_keys=(SCRATCH_KEY,),
            retained_artifact_payload_types={SCRATCH_KEY: jax.Array},
        )
        kernels = MappingProxyType(
            {
                period: _PeriodKernel(
                    programs=MappingProxyType(
                        {
                            "values": values_program,
                            "replay": replay_program,
                            "scratch": scratch_program,
                        }
                    )
                )
                for period in context.regimes_to_active_periods[context.regime_name]
            }
        )
        return SolutionKernels(
            period_kernels=kernels,
            continuation_spec=ContinuationSpec(
                template=counter_template,
                artifact_key=COUNTER_KEY,
            ),
            replay_route=route,
            artifact_authorities=MappingProxyType(
                {
                    COUNTER_KEY: counter_authority,
                    POLICY_KEY: policy_authority,
                    OPTIONAL_REPLAY_KEY: optional_replay_authority,
                    SCRATCH_KEY: scratch_authority,
                }
            ),
        )


__all__ = [
    "COUNTER_KEY",
    "MIDDLE_TIE_REDUCTION",
    "OPTIONAL_REPLAY_KEY",
    "POLICY_KEY",
    "SCRATCH_KEY",
    "Counter",
    "Policy",
    "ReferenceReplayRoute",
    "ReferenceSolver",
    "TerminalCounterSolver",
]
