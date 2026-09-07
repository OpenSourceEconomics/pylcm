"""Buffer lifetime on four devices with a sharded state, read off the release log.

Every artifact is released after its final consumer and never before; retained
values are never released; a buffer two keys share survives until both keys
close; a donated argument is unreadable afterwards while every value is
unchanged.

Runs on a four-CPU-device topology pinned at import, so the file skips wholesale
in a process whose backend is already initialized and runs in its own process.
"""

import dataclasses
import functools
import logging
import operator
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.scheduler import buffer_identity
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueArtifactKind
from _lcm.grids import categorical
from _lcm.grids.discrete import DiscreteGrid
from _lcm.solution import backward_induction
from _lcm.solution.continuation_reads import continuation_leaf_reads
from lcm import AgeGrid, MarkovTransition, Model, Regime, fixed_transition
from lcm.solver_api import ArtifactKey, ContinuationCapabilities, KernelOutput
from lcm.solvers import (
    ContinuationSpec,
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    DeclaredReplay,
    OutputRole,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    StateAxesLeading,
)
from lcm.typing import FloatND, RegimeName, ScalarFloat, ScalarInt, StateName
from tests.conftest import assert_agrees_to_ulp

# The out-of-tree solver module builds arrays at import, which initializes a JAX
# backend; every name it supplies is therefore imported inside the function that
# needs it, after the topology pin below has run.

# Run these tests on a four-CPU-device topology. The pin only applies in a
# process whose JAX backends are not yet initialized; otherwise the tests skip.
# The device-count update is attempted FIRST because it is the one that raises
# after initialization, which keeps the pin atomic.
try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _PYTEST_PARALLEL = False
except RuntimeError:
    _PYTEST_PARALLEL = True

_skip_pytest_parallel = pytest.mark.skipif(
    _PYTEST_PARALLEL, reason="Can't set num cpus in pytest paralellel"
)

_COUNTER = ArtifactKey(type_id="tests.lifetime_counter", schema_version=1)
_N_PERIODS = 4
_PARAMS = {"discount_factor": 1.0}

#: A regime value of the model below, and a dispatch that runs in every solve.
#: Logged as a release by the one test that shows the release log reports a
#: regime value when one is released.
_A_REGIME_VALUE = ValueArtifactAddress(
    kind=ValueArtifactKind.REGIME_VALUE, period=0, regime="alive"
)
_A_DISPATCH = (0, "alive")

#: Every count leaf an argument builder handed to a reading program, build calls
#: included; the dispatch's own read is the last entry when its output is
#: consumed.
_READ_INPUTS: list[FloatND] = []


@categorical(ordered=True)
class _Type:
    """A four-valued preference type; its extent is the mesh size."""

    a: ScalarInt
    b: ScalarInt
    c: ScalarInt
    d: ScalarInt


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Carry:
    """A one-leaf continuation carrying a state-shaped array."""

    count: FloatND

    @property
    def artifact_key(self) -> ArtifactKey:
        """Name the versioned key this payload is published under."""
        return _COUNTER

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Report that the payload answers no query of its own."""
        return ContinuationCapabilities()

    def value_at(self, *, query: FloatND) -> FloatND:
        """Refuse a value query: this payload publishes leaves only."""
        raise NotImplementedError

    def marginal_at(self, *, query: FloatND, state: StateName) -> FloatND:
        """Refuse a marginal query: this payload publishes leaves only."""
        raise NotImplementedError

    def leaves(self) -> MappingProxyType[tuple[str, ...], FloatND]:
        """Return the one addressable array of this payload."""
        return MappingProxyType({("count",): self.count})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _TwoLeafCarry:
    """A two-leaf continuation whose leaves may name one array."""

    count: FloatND
    echo: FloatND

    @property
    def artifact_key(self) -> ArtifactKey:
        """Name the versioned key this payload is published under."""
        return _COUNTER

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Report that the payload answers no query of its own."""
        return ContinuationCapabilities()

    def value_at(self, *, query: FloatND) -> FloatND:
        """Refuse a value query: this payload publishes leaves only."""
        raise NotImplementedError

    def marginal_at(self, *, query: FloatND, state: StateName) -> FloatND:
        """Refuse a marginal query: this payload publishes leaves only."""
        raise NotImplementedError

    def leaves(self) -> MappingProxyType[tuple[str, ...], FloatND]:
        """Return the two addressable arrays of this payload."""
        return MappingProxyType({("count",): self.count, ("echo",): self.echo})


def _state_sum(*, states: tuple[FloatND, ...]) -> FloatND:
    """One value per state node: the sum of the node's own coordinates."""
    return functools.reduce(operator.add, jnp.meshgrid(*states, indexing="ij"))


def _value_and_fresh_carry(*, states: tuple[FloatND, ...]) -> tuple[FloatND, _Carry]:
    """Publish the state sum, and a leaf on a buffer of its own."""
    value = _state_sum(states=states)
    return value, _Carry(count=value + 1.0)


def _value_and_shared_leaf(*, states: tuple[FloatND, ...]) -> tuple[FloatND, FloatND]:
    """Publish the state sum, and one further array of its own."""
    value = _state_sum(states=states)
    return value, value + 1.0


def _value_from_count(*, states: tuple[FloatND, ...], count: FloatND) -> FloatND:
    """One value per state node: the state sum plus the target's count."""
    return _state_sum(states=states) + count


def _value_from_two_counts(
    *, states: tuple[FloatND, ...], count: FloatND, echo: FloatND
) -> FloatND:
    """One value per state node: the state sum plus both leaves read."""
    return _state_sum(states=states) + count + echo


def _terminal_arguments(build: Any) -> dict[str, Any]:
    """Feed the state grids to the terminal program."""
    return {"states": tuple(build.state_action_space.states.values())}


def _reading_arguments(build: Any) -> dict[str, Any]:
    """Feed the state grids and the target's published count to the program."""
    count = build.next_regime_to_continuation["dead"].count
    _READ_INPUTS.append(count)
    return {
        "states": tuple(build.state_action_space.states.values()),
        "count": count,
    }


def _two_leaf_reading_arguments(build: Any) -> dict[str, Any]:
    """Feed the state grids and both published leaves to the program."""
    payload = build.next_regime_to_continuation["dead"]
    _READ_INPUTS.append(payload.count)
    return {
        "states": tuple(build.state_action_space.states.values()),
        "count": payload.count,
        "echo": payload.echo,
    }


def _certain(age: ScalarFloat) -> ScalarFloat:  # noqa: ARG001
    """Probability of the one target the reading regime can reach."""
    return jnp.asarray(1.0)


def _placed_zeros(
    *,
    shape: tuple[int, ...],
    state_names: tuple[StateName, ...],
    context: SolverBuildContext,
) -> FloatND:
    """Zeros on the placement the regime's own value array runs on.

    Published leaves are put on their template's sharding, so a state-shaped
    template of a regime with a distributed state must carry that regime's
    mesh; the state nodes a solver is handed at build carry the unplaced
    layout, so the mesh is resolved from the regime's grids and placed devices
    the way the engine resolves it for its own carries.

    That resolution reaches into `_lcm.engine`, whose `_build_regime_sharding`
    is private, and this helper breaks if it is renamed. Nothing public offers
    it: `lcm.solvers` and `lcm.solver_api` publish no sharding, the build
    context carries only `grids` and `submesh_device_ids`, and
    `_lcm.egm.carry.shard_carry_template` is typed to the EGM carry, so a
    foreign payload cannot go through it.
    """
    from _lcm.engine import (  # noqa: PLC0415
        _build_regime_sharding,
        placed_devices_for_ids,
    )

    zeros = jnp.zeros(shape)
    plan = _build_regime_sharding(
        grids=context.grids,
        devices=placed_devices_for_ids(submesh_device_ids=context.submesh_device_ids),
    )
    if plan is None:
        return zeros
    return jax.device_put(zeros, plan.V_arr_sharding(state_names))


def _terminal_template(*, context: SolverBuildContext) -> _Carry:
    """The solve-lifetime payload of the publishing regime, on its own mesh."""
    states = tuple(context.state_action_space.states.values())
    return _Carry(
        count=_placed_zeros(
            shape=tuple(int(node.size) for node in states),
            state_names=tuple(context.state_action_space.states),
            context=context,
        )
    )


def _terminal_program(*, function: Any, output_roles: Any) -> CoreProgram:
    """One dense program over the publishing regime's state nodes."""
    return CoreProgram(
        name="main",
        function=function,
        argument_builder=_terminal_arguments,
        requirements=CoreExecutionRequirements(),
        output_roles=output_roles,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="one_row_per_state_node",
    )


class _TerminalCarrySolver(Solver):
    """Publishes a state-shaped value and a leaf on a buffer of its own."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Declare one dense program per active period, publishing the count."""
        from tests.test_solver_api_out_of_tree import _GraphKernel  # noqa: PLC0415

        program = _terminal_program(
            function=_value_and_fresh_carry,
            output_roles=(
                OutputRole.VALUE,
                _Carry(
                    count=StateAxesLeading(  # ty: ignore[invalid-argument-type]
                        state_names=tuple(context.state_action_space.states)
                    )
                ),
            ),
        )
        kernels = {
            period: _GraphKernel(
                programs=MappingProxyType({"main": program}),
                continuation_key=_COUNTER,
            )
            for period in context.regimes_to_active_periods[context.regime_name]
        }
        return SolutionKernels(
            period_kernels=MappingProxyType(kernels),
            continuation_spec=ContinuationSpec(
                template=_terminal_template(context=context), artifact_key=_COUNTER
            ),
            replay_route=DeclaredReplay.GRID_RECOMPUTATION,
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _AliasingKernel:
    """Publish this period's value array itself as its continuation payload."""

    programs: MappingProxyType[str, CoreProgram]

    def core_programs(self) -> MappingProxyType[str, CoreProgram]:
        """Return the core graph the planner consumes."""
        return self.programs

    def with_fixed_params(self, *, fixed_flat_params: Any) -> _AliasingKernel:  # noqa: ARG002
        """Return this parameter-free kernel unchanged."""
        return self

    def __call__(
        self,
        *,
        compiled_cores: Any,
        state_action_space: Any,
        next_regime_to_V_arr: Any,
        next_regime_to_continuation: Any,
        flat_params: Any,
        period: int,
        ages: Any,
        logger: Any,  # noqa: ARG002
        **_unused: Any,
    ) -> KernelOutput:
        """Dispatch the value program and publish that array on both channels."""
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        value = compiled_cores["main"](
            **self.programs["main"].argument_builder(context)
        )
        return KernelOutput(value=value, continuations={_COUNTER: _Carry(count=value)})


class _AliasingTerminalSolver(Solver):
    """Publishes the value array itself as the continuation leaf."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Declare one dense value program per active period."""
        program = _terminal_program(function=_state_sum, output_roles=OutputRole.VALUE)
        kernels = {
            period: _AliasingKernel(programs=MappingProxyType({"main": program}))
            for period in context.regimes_to_active_periods[context.regime_name]
        }
        return SolutionKernels(
            period_kernels=MappingProxyType(kernels),
            continuation_spec=ContinuationSpec(
                template=_terminal_template(context=context), artifact_key=_COUNTER
            ),
            replay_route=DeclaredReplay.GRID_RECOMPUTATION,
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _SharedLeafKernel:
    """Publish one array the executable produced under both leaf paths."""

    programs: MappingProxyType[str, CoreProgram]

    def core_programs(self) -> MappingProxyType[str, CoreProgram]:
        """Return the core graph the planner consumes."""
        return self.programs

    def with_fixed_params(self, *, fixed_flat_params: Any) -> _SharedLeafKernel:  # noqa: ARG002
        """Return this parameter-free kernel unchanged."""
        return self

    def __call__(
        self,
        *,
        compiled_cores: Any,
        state_action_space: Any,
        next_regime_to_V_arr: Any,
        next_regime_to_continuation: Any,
        flat_params: Any,
        period: int,
        ages: Any,
        logger: Any,  # noqa: ARG002
        **_unused: Any,
    ) -> KernelOutput:
        """Dispatch the program and put its second output under both paths."""
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        value, leaf = compiled_cores["main"](
            **self.programs["main"].argument_builder(context)
        )
        return KernelOutput(
            value=value,
            continuations={_COUNTER: _TwoLeafCarry(count=leaf, echo=leaf)},
        )


class _SharedLeafTerminalSolver(Solver):
    """Publishes one produced array under both leaf paths of its payload."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Declare one dense program per active period, publishing both paths."""
        program = _terminal_program(
            function=_value_and_shared_leaf,
            output_roles=(
                OutputRole.VALUE,
                StateAxesLeading(state_names=tuple(context.state_action_space.states)),
            ),
        )
        zeros = _terminal_template(context=context).count
        kernels = {
            period: _SharedLeafKernel(programs=MappingProxyType({"main": program}))
            for period in context.regimes_to_active_periods[context.regime_name]
        }
        return SolutionKernels(
            period_kernels=MappingProxyType(kernels),
            continuation_spec=ContinuationSpec(
                template=_TwoLeafCarry(count=zeros, echo=zeros),
                artifact_key=_COUNTER,
            ),
            replay_route=DeclaredReplay.GRID_RECOMPUTATION,
        )


def _reading_kernels(
    *,
    context: SolverBuildContext,
    function: Any,
    argument_builder: Any,
    template: Any,
    argument_by_leaf: Mapping[tuple[str, ...], str],
    donation_candidates: tuple[str, ...],
) -> SolutionKernels:
    """One dense reading program per active period, over the declared leaves.

    The template is read for its leaf paths alone, so the reader's own shapes
    say nothing about the arrays the target publishes.
    """
    from tests.test_solver_api_out_of_tree import _GraphKernel  # noqa: PLC0415

    kernels = {}
    for period in context.regimes_to_active_periods[context.regime_name]:
        program = CoreProgram(
            name="main",
            function=function,
            argument_builder=argument_builder,
            requirements=CoreExecutionRequirements(
                value_reads=continuation_leaf_reads(
                    template=template,
                    artifact_key=_COUNTER,
                    target="dead",
                    source_regime=context.regime_name,
                    source_period=period,
                    core_key="main",
                    argument_by_leaf=argument_by_leaf,
                )
            ),
            output_roles=OutputRole.VALUE,
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason="one_row_per_state_node",
            donation_candidates=donation_candidates,
        )
        kernels[period] = _GraphKernel(programs=MappingProxyType({"main": program}))
    return SolutionKernels(
        period_kernels=MappingProxyType(kernels),
        replay_route=DeclaredReplay.GRID_RECOMPUTATION,
    )


class _ReadingSolver(Solver):
    """Reads the terminal regime's published leaf into its own value."""

    donation_candidates: tuple[str, ...] = ()

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        """Demand the counter the terminal regime publishes."""
        return frozenset({_COUNTER})

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Declare one dense program per active period, reading the count leaf."""
        return _reading_kernels(
            context=context,
            function=_value_from_count,
            argument_builder=_reading_arguments,
            template=_Carry(count=jnp.zeros(())),
            argument_by_leaf={("count",): "count"},
            donation_candidates=self.donation_candidates,
        )


class _DonatingReadingSolver(_ReadingSolver):
    """Names the continuation leaf it reads as a donation candidate."""

    donation_candidates = ("count",)


class _TwoLeafReadingSolver(_ReadingSolver):
    """Reads both leaves of the target's payload into its own value."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Declare one dense program per active period, reading both leaves."""
        return _reading_kernels(
            context=context,
            function=_value_from_two_counts,
            argument_builder=_two_leaf_reading_arguments,
            template=_TwoLeafCarry(count=jnp.zeros(()), echo=jnp.zeros(())),
            argument_by_leaf={("count",): "count", ("echo",): "echo"},
            donation_candidates=self.donation_candidates,
        )


def _model(
    *, solver: Solver | None = None, terminal_solver: Solver | None = None
) -> Model:
    """A reading regime over a sharded type beside the terminal one it reads."""
    from tests.test_solver_api_out_of_tree import (  # noqa: PLC0415
        _WEALTH,
        RegimeId,
        next_wealth,
    )

    return Model(
        regimes={
            "alive": Regime(
                transition={"dead": MarkovTransition(_certain)},
                active=lambda age: age < _N_PERIODS - 1,
                states={"wealth": _WEALTH},
                state_transitions={"wealth": next_wealth},
                functions={"utility": lambda wealth, type1: wealth * (type1 + 1.0)},
                solver=solver if solver is not None else _ReadingSolver(),
            ),
            "dead": Regime(
                transition=None,
                states={"wealth": _WEALTH},
                functions={"utility": lambda wealth, type1: 0.0 * wealth * type1},
                solver=(
                    terminal_solver
                    if terminal_solver is not None
                    else _TerminalCarrySolver()
                ),
            ),
        },
        ages=AgeGrid(start=0, stop=_N_PERIODS - 1, step="Y"),
        regime_id_class=RegimeId,
        states={"type1": DiscreteGrid(category_class=_Type, distributed=True)},
        state_transitions={"type1": fixed_transition("type1")},
    )


class _Events(logging.Handler):
    """Collect the release and donation records a solve emits, in order."""

    def __init__(self) -> None:
        """Start with no record collected."""
        super().__init__(level=logging.DEBUG)
        self.events: list[tuple[str, Any, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep one donation or release record, tagged by which it is."""
        if hasattr(record, "artifact_key") and hasattr(record, "donating_dispatch"):
            self.events.append(
                ("donate", record.artifact_key, record.donating_dispatch)
            )
        elif hasattr(record, "artifact_key") and hasattr(record, "closing_dispatch"):
            self.events.append(
                ("release", record.artifact_key, record.closing_dispatch)
            )


class _DispatchRecorder:
    """Run one period's kernel, recording the dispatch that ran it."""

    def __init__(self, *, kernel: Any, events: list[tuple[str, Any, Any]]) -> None:
        """Keep the kernel to delegate to and the log to interleave into."""
        self._kernel = kernel
        self._events = events

    def __call__(self, **kwargs: Any) -> Any:
        """Record the dispatch, then run it."""
        self._events.append(
            ("dispatch", (kwargs["period"], kwargs["regime_name"]), None)
        )
        return self._kernel(**kwargs)


class _ReadObserver:
    """Consume a kernel's output, recording what each dispatch read and published.

    A read is recorded at the dispatch that ran it, and the published value and
    leaf are the arrays the solve itself holds — the result publishes copies of
    its values, so a claim about a buffer is a claim about these.
    """

    def __init__(
        self,
        *,
        consume: Any,
        reads: list[tuple[int, FloatND]],
        deleted_at_dispatch: list[tuple[int, bool]],
        published: dict[int, tuple[FloatND, FloatND]],
    ) -> None:
        """Keep the consumer to delegate to and the records to fill."""
        self._consume = consume
        self._reads = reads
        self._deleted_at_dispatch = deleted_at_dispatch
        self._published = published

    def __call__(self, **kwargs: Any) -> Any:
        """Consume the output, then record the dispatch's read or publication."""
        result = self._consume(**kwargs)
        period = kwargs["period"]
        if kwargs["regime_name"] == "alive":
            leaf = _READ_INPUTS[-1]
            self._reads.append((period, leaf))
            self._deleted_at_dispatch.append((period, leaf.is_deleted()))
        elif kwargs["regime_name"] == "dead":
            self._published[period] = (result.value, result.continuation.count)
        return result


class _CompileRecorder:
    """Compile the solve's programs, keeping the ledger they were lowered against."""

    def __init__(self, *, compile_programs: Any) -> None:
        """Keep the compiler to delegate to; nothing is recorded yet."""
        self._compile_programs = compile_programs
        self.compiled: list[Any] = []

    def __call__(self, **kwargs: Any) -> Any:
        """Compile, and keep the result the loop will commit to."""
        result = self._compile_programs(**kwargs)
        self.compiled.append(result)
        return result


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Observation:
    """One solve, with the lifetime records it emitted."""

    solution: Any
    """The solve result, whose values the retention keeps."""

    events: list[tuple[str, Any, Any]]
    """Dispatches, releases and donations, in the order they happened."""

    reads: list[tuple[int, FloatND]]
    """Per dispatch of the reading regime, the count leaf it was handed."""

    deleted_at_dispatch: list[tuple[int, bool]]
    """Per dispatch of the reading regime, whether that leaf was already freed."""

    published: dict[int, tuple[FloatND, FloatND]]
    """Per period, the value and continuation leaf the publishing regime stored."""


def _solve_with_events(
    *,
    solver: Solver | None = None,
    terminal_solver: Solver | None = None,
    monkeypatch: pytest.MonkeyPatch,
    compile_recorder: _CompileRecorder | None = None,
    seeded_release: ValueArtifactAddress | None = None,
) -> _Observation:
    """Solve the sharded model, interleaving its records with its dispatches.

    `seeded_release` is logged as a release of the run, in the record shape the
    engine uses, so a claim that the solve released no artifact of some kind can
    be paired with the same predicate reporting one that was.
    """
    _READ_INPUTS.clear()
    handler = _Events()
    reads: list[tuple[int, FloatND]] = []
    deleted_at_dispatch: list[tuple[int, bool]] = []
    published: dict[int, tuple[FloatND, FloatND]] = {}
    monkeypatch.setattr(
        backward_induction,
        "_run_period_kernel",
        _DispatchRecorder(
            kernel=backward_induction._run_period_kernel, events=handler.events
        ),
    )
    monkeypatch.setattr(
        backward_induction,
        "consume_kernel_output",
        _ReadObserver(
            consume=backward_induction.consume_kernel_output,
            reads=reads,
            deleted_at_dispatch=deleted_at_dispatch,
            published=published,
        ),
    )
    if compile_recorder is not None:
        monkeypatch.setattr(
            backward_induction, "_compile_all_functions", compile_recorder
        )
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        solution = _model(solver=solver, terminal_solver=terminal_solver).solve(
            params=_PARAMS, log_level="debug"
        )
        if seeded_release is not None:
            logger.debug(
                "released %r after dispatch %r",
                seeded_release,
                _A_DISPATCH,
                extra={
                    "artifact_key": seeded_release,
                    "closing_dispatch": _A_DISPATCH,
                },
            )
    finally:
        logger.removeHandler(handler)
    return _Observation(
        solution=solution,
        events=handler.events,
        reads=reads,
        deleted_at_dispatch=deleted_at_dispatch,
        published=published,
    )


def _released_leaf_periods(*, observed: _Observation, regime: RegimeName) -> list[int]:
    """The periods whose continuation leaf of one regime the solve released."""
    return sorted(
        artifact.period
        for kind, artifact, _ in observed.events
        if kind == "release"
        and artifact.kind is ValueArtifactKind.CONTINUATION_LEAF
        and artifact.regime == regime
    )


@_skip_pytest_parallel
def test_every_release_follows_the_dispatch_that_closed_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A buffer is released after, never before, its final consumer ran."""
    observed = _solve_with_events(monkeypatch=monkeypatch)
    dispatched_at = {
        payload: index
        for index, (kind, payload, _) in enumerate(observed.events)
        if kind == "dispatch"
    }
    violations = [
        (artifact, closer)
        for index, (kind, artifact, closer) in enumerate(observed.events)
        if kind == "release" and not dispatched_at[closer] < index
    ]

    assert not violations


@_skip_pytest_parallel
def test_no_later_dispatch_reads_a_released_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a release, no dispatch that still runs declares the artifact."""
    recorder = _CompileRecorder(
        compile_programs=backward_induction._compile_all_functions
    )
    observed = _solve_with_events(monkeypatch=monkeypatch, compile_recorder=recorder)
    (programs,) = recorder.compiled
    ledger = programs.input_liveness
    later_readers = [
        (artifact, dispatch)
        for index, (kind, artifact, _) in enumerate(observed.events)
        if kind == "release"
        for later_kind, dispatch, _ in observed.events[index + 1 :]
        if later_kind == "dispatch"
        and artifact in ledger.accesses_of(dispatch=dispatch)
    ]

    assert not later_readers


@_skip_pytest_parallel
def test_every_reading_dispatch_declares_the_leaf_it_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ledger names one leaf per reading dispatch, the one it was handed."""
    recorder = _CompileRecorder(
        compile_programs=backward_induction._compile_all_functions
    )
    observed = _solve_with_events(monkeypatch=monkeypatch, compile_recorder=recorder)
    (programs,) = recorder.compiled
    ledger = programs.input_liveness

    assert [
        [artifact.period for artifact in ledger.accesses_of(dispatch=(period, "alive"))]
        for period, _ in observed.reads
    ] == [[period + 1] for period, _ in observed.reads]


@_skip_pytest_parallel
def test_the_continuation_leaf_of_every_read_period_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each leaf a reading period consumes is released once that reader has run."""
    observed = _solve_with_events(monkeypatch=monkeypatch)

    assert _released_leaf_periods(observed=observed, regime="dead") == list(
        range(1, _N_PERIODS)
    )


@_skip_pytest_parallel
def test_no_regime_value_is_released(monkeypatch: pytest.MonkeyPatch) -> None:
    """No regime value appears in the release log of a retaining solve."""
    observed = _solve_with_events(monkeypatch=monkeypatch)

    assert not [
        artifact
        for kind, artifact, _ in observed.events
        if kind == "release" and artifact.kind is ValueArtifactKind.REGIME_VALUE
    ]


@_skip_pytest_parallel
def test_a_released_regime_value_is_reported_by_the_kind_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The filter that reports no released regime value does report one."""
    observed = _solve_with_events(
        monkeypatch=monkeypatch, seeded_release=_A_REGIME_VALUE
    )

    assert [
        artifact
        for kind, artifact, _ in observed.events
        if kind == "release" and artifact.kind is ValueArtifactKind.REGIME_VALUE
    ] == [_A_REGIME_VALUE]


@_skip_pytest_parallel
def test_every_retained_value_stays_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every value a period published is still readable after the solve."""
    observed = _solve_with_events(monkeypatch=monkeypatch)

    assert [
        value.is_deleted() for _, (value, _) in sorted(observed.published.items())
    ] == [False] * _N_PERIODS


@_skip_pytest_parallel
def test_a_leaf_sharing_the_retained_values_buffer_survives_its_own_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A buffer under two keys is kept while the other key is retained."""
    observed = _solve_with_events(
        terminal_solver=_AliasingTerminalSolver(), monkeypatch=monkeypatch
    )

    assert not [period for period, leaf in observed.reads if leaf.is_deleted()]


@_skip_pytest_parallel
def test_a_leaf_sharing_the_retained_values_buffer_holds_that_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each read leaf is the buffer of the value published beside it.

    The dispatch at period p reads the leaf the target published at p + 1.
    """
    observed = _solve_with_events(
        terminal_solver=_AliasingTerminalSolver(), monkeypatch=monkeypatch
    )

    assert [
        buffer_identity(array=leaf)
        == buffer_identity(array=observed.published[period + 1][0])
        for period, leaf in observed.reads
    ] == [True] * (_N_PERIODS - 1)


@_skip_pytest_parallel
def test_no_release_names_a_leaf_the_retained_value_shares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key whose buffer a retained value shares is never released."""
    observed = _solve_with_events(
        terminal_solver=_AliasingTerminalSolver(), monkeypatch=monkeypatch
    )

    assert _released_leaf_periods(observed=observed, regime="dead") == []


@_skip_pytest_parallel
def test_two_keys_on_one_produced_buffer_are_released_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One release frees a buffer two keys name, and names both on its records.

    Both keys address one array a compiled executable produced, so the release
    reaches the partner-eligibility rule rather than the not-produced skip: a
    second delete of the same buffer would report one key, not two.
    """
    observed = _solve_with_events(
        solver=_TwoLeafReadingSolver(),
        terminal_solver=_SharedLeafTerminalSolver(),
        monkeypatch=monkeypatch,
    )

    assert sorted(
        (artifact.period, artifact.leaf_path)
        for kind, artifact, _ in observed.events
        if kind == "release" and artifact.kind is ValueArtifactKind.CONTINUATION_LEAF
    ) == [
        (period, path)
        for period in range(1, _N_PERIODS)
        for path in (("count",), ("echo",))
    ]


@_skip_pytest_parallel
def test_a_fresh_leaf_is_unreadable_once_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leaf sharing no retained buffer is gone once the solve has closed it."""
    observed = _solve_with_events(monkeypatch=monkeypatch)

    assert [leaf.is_deleted() for _, leaf in observed.reads] == [True] * (
        _N_PERIODS - 1
    )


@_skip_pytest_parallel
def test_every_read_period_records_a_donation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatches reading a published leaf hand it to their own executable."""
    observed = _solve_with_events(
        solver=_DonatingReadingSolver(), monkeypatch=monkeypatch
    )

    assert (
        len([artifact for kind, artifact, _ in observed.events if kind == "donate"])
        == _N_PERIODS - 1
    )


@_skip_pytest_parallel
def test_a_donated_argument_is_unreadable_afterwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every donated read is unreadable once the solve has run."""
    observed = _solve_with_events(
        solver=_DonatingReadingSolver(), monkeypatch=monkeypatch
    )

    assert [leaf.is_deleted() for _, leaf in observed.reads] == [True] * (
        _N_PERIODS - 1
    )


@_skip_pytest_parallel
def test_a_donated_argument_is_unreadable_when_its_dispatch_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A donated leaf is gone at its own dispatch, not at a later release."""
    observed = _solve_with_events(
        solver=_DonatingReadingSolver(), monkeypatch=monkeypatch
    )

    assert observed.deleted_at_dispatch == [
        (period, True) for period in range(_N_PERIODS - 2, -1, -1)
    ]


@_skip_pytest_parallel
def test_a_read_no_solver_donates_survives_its_own_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a donation candidate every read outlives the dispatch that read it."""
    observed = _solve_with_events(monkeypatch=monkeypatch)

    assert observed.deleted_at_dispatch == [
        (period, False) for period in range(_N_PERIODS - 2, -1, -1)
    ]


@pytest.fixture(scope="module")
def donating_and_plain_values() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Per period, the values the donating and the releasing solver publish."""
    donating = _model(solver=_DonatingReadingSolver()).solve(
        params=_PARAMS, log_level="debug"
    )
    plain = _model().solve(params=_PARAMS, log_level="debug")
    return (
        {
            period: np.asarray(donating.values[period]["dead"])
            for period in range(_N_PERIODS)
        },
        {
            period: np.asarray(plain.values[period]["dead"])
            for period in range(_N_PERIODS)
        },
    )


@_skip_pytest_parallel
@pytest.mark.parametrize("period", range(_N_PERIODS))
def test_donation_leaves_every_terminal_value_unchanged(
    *,
    period: int,
    donating_and_plain_values: tuple[dict[int, np.ndarray], dict[int, np.ndarray]],
) -> None:
    """The donating and the releasing solve publish the same terminal values."""
    donating, plain = donating_and_plain_values

    assert_agrees_to_ulp(
        got=donating[period],
        expected=plain[period],
        n_ulp=8,
        err_msg=f"period {period}",
    )


@pytest.fixture(scope="module")
def donating_and_plain_reading_values() -> tuple[
    dict[int, np.ndarray], dict[int, np.ndarray]
]:
    """Per read period, the values the two reading solvers publish."""
    donating = _model(solver=_DonatingReadingSolver()).solve(
        params=_PARAMS, log_level="debug"
    )
    plain = _model().solve(params=_PARAMS, log_level="debug")
    return (
        {
            period: np.asarray(donating.values[period]["alive"])
            for period in range(_N_PERIODS - 1)
        },
        {
            period: np.asarray(plain.values[period]["alive"])
            for period in range(_N_PERIODS - 1)
        },
    )


@_skip_pytest_parallel
@pytest.mark.parametrize("period", range(_N_PERIODS - 1))
def test_donation_leaves_every_reading_value_unchanged(
    *,
    period: int,
    donating_and_plain_reading_values: tuple[
        dict[int, np.ndarray], dict[int, np.ndarray]
    ],
) -> None:
    """The donating and the releasing solve publish the same read values."""
    donating, plain = donating_and_plain_reading_values

    assert_agrees_to_ulp(
        got=donating[period],
        expected=plain[period],
        n_ulp=8,
        err_msg=f"period {period}",
    )


@_skip_pytest_parallel
def test_the_value_is_sharded_over_all_four_devices() -> None:
    """The model whose lifetimes are measured really is sharded on this topology."""
    solution = _model().solve(params=_PARAMS, log_level="debug")

    assert len(solution.values[0]["alive"].sharding.device_set) == 4
