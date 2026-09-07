"""A solver that names a donation candidate has it donated by the engine.

The engine donates the leaf whose last remaining reader is the dispatch it is
lowering, keeps the solve-lifetime template, and publishes exactly the values a
non-donating twin publishes.
"""

import dataclasses
import re
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.donation import DonatedBuffer, ResolvedDonation
from _lcm.execution.scheduler import BufferRegistry, DispatchUnit, shard_identities
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueArtifactKind
from _lcm.solution import backward_induction
from _lcm.solution.continuation_reads import continuation_leaf_reads
from _lcm.solution.kernel_output import ConsumedKernelOutput
from _lcm.solution.solve_inputs import SolveInputMappings
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import ArtifactKey, ContinuationCapabilities
from lcm.solvers import (
    ContinuationSpec,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    OutputRole,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    StateAxesLeading,
)
from lcm.typing import Float1D, FloatND, StateName
from tests.solution.test_compilation_identity import _lowering_key
from tests.test_solver_api_out_of_tree import (
    _N_PERIODS,
    _GraphKernel,
    _two_regime_model,
)

_COUNTER = ArtifactKey(type_id="tests.donated_counter", schema_version=1)
_TEMPLATES: list[_ReadableCounter] = []
_DISPATCHED: list[FloatND] = []


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _ReadableCounter:
    """A one-leaf continuation that answers the reader protocol."""

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


def _counting_value(
    *, wealth: Float1D, count: FloatND
) -> tuple[Float1D, _ReadableCounter]:
    """Publish wealth plus the running count, and the count incremented."""
    return wealth + count, _ReadableCounter(count=count + 1.0)


def _counting_arguments(build: object) -> dict[str, object]:
    """Feed the state grid and the target's published count to the program."""
    arguments = {
        "wealth": build.state_action_space.states["wealth"],  # ty: ignore[unresolved-attribute]
        "count": build.next_regime_to_continuation["alive"].count,  # ty: ignore[unresolved-attribute]
    }
    _DISPATCHED.append(arguments["count"])
    return arguments


class _CounterSolver(Solver):
    """Reads its own next-period count and republishes it incremented."""

    donation_candidates: tuple[str, ...] = ()

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        """Demand the counter its own regime publishes."""
        return frozenset({_COUNTER})

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Declare one dense program per active period, reading the count leaf."""
        template = _ReadableCounter(count=jnp.zeros(()))
        _TEMPLATES.append(template)
        kernels = {}
        for period in context.regimes_to_active_periods[context.regime_name]:
            program = CoreProgram(
                name="main",
                function=_counting_value,
                argument_builder=_counting_arguments,
                requirements=CoreExecutionRequirements(
                    value_reads=continuation_leaf_reads(
                        template=template,
                        artifact_key=_COUNTER,
                        target="alive",
                        source_regime="alive",
                        source_period=period,
                        core_key="main",
                        argument_by_leaf={("count",): "count"},
                    )
                ),
                output_roles=(
                    OutputRole.VALUE,
                    _ReadableCounter(
                        count=StateAxesLeading(state_names=(), shape=())  # ty: ignore[invalid-argument-type]
                    ),
                ),
                disposition=CoreExecutionDisposition.DENSE,
                disposition_reason="one_row_per_state_node",
                donation_candidates=self.donation_candidates,
            )
            kernels[period] = _GraphKernel(
                programs=MappingProxyType({"main": program}),
                continuation_key=_COUNTER,
            )
        return SolutionKernels(
            period_kernels=MappingProxyType(kernels),
            continuation_spec=ContinuationSpec(
                template=template, artifact_key=_COUNTER
            ),
        )


class _DonatingCounterSolver(_CounterSolver):
    """The same solver, naming its count argument as a donation candidate."""

    donation_candidates = ("count",)


def _solve(*, solver: Solver) -> object:
    """Solve the self-looping two-regime model with the given solver."""
    _TEMPLATES.clear()
    _DISPATCHED.clear()
    return _two_regime_model(solver=solver, self_looping=True).solve(
        params={"discount_factor": 1.0}, log_level="off"
    )


def _reads_deleted_at_dispatch(
    *, solver: Solver, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[int, bool]]:
    """Report, per period, whether the count read is unreadable once it returns."""
    observed: list[tuple[int, bool]] = []
    consume = backward_induction.consume_kernel_output

    def _observe(**kwargs: object) -> ConsumedKernelOutput:
        result = consume(**kwargs)  # ty: ignore[invalid-argument-type]
        if kwargs["regime_name"] == "alive":
            period = kwargs["period"]
            assert isinstance(period, int)
            observed.append((period, _DISPATCHED[-1].is_deleted()))
        return result

    monkeypatch.setattr(backward_induction, "consume_kernel_output", _observe)
    _solve(solver=solver)
    return observed


def test_a_donated_argument_is_unreadable_when_its_dispatch_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every period but the last hands its count leaf to its own executable."""
    assert _reads_deleted_at_dispatch(
        solver=_DonatingCounterSolver(), monkeypatch=monkeypatch
    ) == [(2, False), (1, True), (0, True)]


def test_a_non_donating_solver_leaves_every_read_readable_at_its_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a candidate nothing is donated, so every read survives its call."""
    assert _reads_deleted_at_dispatch(
        solver=_CounterSolver(), monkeypatch=monkeypatch
    ) == [(2, False), (1, False), (0, False)]


def test_the_solve_lifetime_template_survives_a_donating_solve() -> None:
    """The last period reads the template, which no release or donation frees."""
    _solve(solver=_DonatingCounterSolver())

    assert not _TEMPLATES[-1].count.is_deleted()


@pytest.fixture(scope="module")
def published_values() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Per period, the values the donating and the non-donating solver publish."""
    donating = _solve(solver=_DonatingCounterSolver())
    plain = _solve(solver=_CounterSolver())
    return (
        {
            period: np.asarray(donating.values[period]["alive"])  # ty: ignore[unresolved-attribute]
            for period in range(_N_PERIODS)
        },
        {
            period: np.asarray(plain.values[period]["alive"])  # ty: ignore[unresolved-attribute]
            for period in range(_N_PERIODS)
        },
    )


@pytest.mark.parametrize("period", range(_N_PERIODS))
def test_donation_changes_no_published_value(
    *,
    period: int,
    published_values: tuple[dict[int, np.ndarray], dict[int, np.ndarray]],
) -> None:
    """The donating and the non-donating solver publish identical values."""
    donating, plain = published_values

    np.testing.assert_array_equal(donating[period], plain[period])


_VALUE = ValueArtifactAddress(
    kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="alive"
)


def _donation(*, argument: str = "next_value") -> ResolvedDonation:
    """One resolved donation of the regime value the fixtures address."""
    return ResolvedDonation(
        argument=argument,
        artifacts=(_VALUE,),
        buffer=DonatedBuffer.STORED_ARTIFACT,
    )


def _mappings(*, value: FloatND) -> SolveInputMappings:
    """Rolling input mappings holding one regime value and nothing else."""
    return SolveInputMappings(
        next_regime_to_V_arr=MappingProxyType({"alive": value}),
        next_regime_to_continuation=MappingProxyType({}),
        next_edge_to_V_arr=MappingProxyType({}),
    )


def _locate_donated(
    *,
    donations: MappingProxyType[tuple[str, int, str], tuple[ResolvedDonation, ...]],
    programs: tuple[str, ...],
    value: FloatND,
    template: FloatND,
    registry: BufferRegistry,
) -> tuple[object, ...]:
    """Run the loop's pre-dispatch donation check over one dispatch unit."""
    return backward_induction._donated_input_arrays(
        donations=donations,
        unit=DispatchUnit(period=0, regime="alive", programs=programs),
        inputs=_mappings(value=value),
        templates=_mappings(value=template),
        registry=registry,
    )


def test_two_programs_of_one_unit_may_not_donate_one_artifact() -> None:
    """One buffer is handed over once, whichever program of the unit names it."""
    with pytest.raises(ExecutionPlanningError, match="lowered two programs to donate"):
        _locate_donated(
            donations=MappingProxyType(
                {
                    ("alive", 0, "main"): (_donation(),),
                    ("alive", 0, "second"): (_donation(argument="also_next"),),
                }
            ),
            programs=("main", "second"),
            value=jnp.arange(3.0),
            template=jnp.zeros(3),
            registry=BufferRegistry(),
        )


def test_an_artifact_standing_at_its_template_is_not_donated() -> None:
    """A key whose leaf is the solve-lifetime template is no input of its own."""
    template = jnp.zeros(3)

    with pytest.raises(ExecutionPlanningError, match="not a solve input of its own"):
        _locate_donated(
            donations=MappingProxyType({("alive", 0, "main"): (_donation(),)}),
            programs=("main",),
            value=template,
            template=template,
            registry=BufferRegistry(),
        )


def test_a_buffer_no_executable_produced_is_not_donated() -> None:
    """A buffer the model owns is refused however the ledger reads its key."""
    value = jnp.arange(3.0)
    registry = BufferRegistry()
    registry.declare_not_produced(tree=(value,))

    with pytest.raises(ExecutionPlanningError, match="no compiled executable produced"):
        _locate_donated(
            donations=MappingProxyType({("alive", 0, "main"): (_donation(),)}),
            programs=("main",),
            value=value,
            template=jnp.zeros(3),
            registry=registry,
        )


def test_a_buffer_a_second_key_still_names_is_not_donated() -> None:
    """A key the ledger did not close keeps the buffer its partner would donate."""
    value = jnp.arange(3.0)
    registry = BufferRegistry()
    registry.register(array=value, artifact="another_key")

    with pytest.raises(ExecutionPlanningError, match="still name"):
        _locate_donated(
            donations=MappingProxyType({("alive", 0, "main"): (_donation(),)}),
            programs=("main",),
            value=value,
            template=jnp.zeros(3),
            registry=registry,
        )


class _SnapshottingRegistry(BufferRegistry):
    """A registry that snapshots its declared set after each declaration."""

    def __init__(self) -> None:
        """Start with no snapshot and record this instance for the test."""
        super().__init__()
        self.snapshots: list[frozenset[tuple[int, int]]] = []
        _REGISTRIES.append(self)

    def declare_not_produced(self, *, tree: object) -> None:
        """Declare, then record the shards declared at that moment."""
        super().declare_not_produced(tree=tree)
        self.snapshots.append(self.declared_shards)


_REGISTRIES: list[_SnapshottingRegistry] = []


def test_the_input_templates_are_declared_before_the_first_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No solve-lifetime template buffer is a release candidate at any period."""
    _REGISTRIES.clear()
    monkeypatch.setattr(backward_induction, "BufferRegistry", _SnapshottingRegistry)
    _solve(solver=_DonatingCounterSolver())

    assert shard_identities(array=_TEMPLATES[-1].count) <= _REGISTRIES[0].snapshots[0]


def test_the_donation_set_is_part_of_the_lowering_key() -> None:
    """Two lowerings that differ only in what they donate are two executables."""
    identity = ("program", "model", "alive", "main", None, None)
    plain = _lowering_key(program_identity=identity, layout_key=("layout",))
    donating = _lowering_key(
        program_identity=identity,
        layout_key=("layout",),
        donated_arguments=("count",),
    )

    assert plain != donating


def _plan_donations(
    *,
    donations: MappingProxyType[
        tuple[tuple[str, int, str], tuple[tuple[str, int], ...]],
        tuple[ResolvedDonation, ...],
    ],
) -> None:
    """Run the plan-time donation check over a mapping of width candidates."""
    backward_induction._fail_if_a_unit_donates_one_artifact_twice(donations=donations)


def test_a_plan_donating_one_artifact_from_two_programs_is_refused() -> None:
    """A plan handing one buffer over twice names it before anything compiles."""
    with pytest.raises(ExecutionPlanningError, match=re.escape(repr(_VALUE))):
        _plan_donations(
            donations=MappingProxyType(
                {
                    (("alive", 0, "main"), ()): (_donation(),),
                    (("alive", 0, "second"), ()): (_donation(argument="also_next"),),
                }
            )
        )


def test_a_plan_donating_one_artifact_at_two_widths_of_one_program_is_accepted() -> (
    None
):
    """Two widths of one core are alternatives, so only one of them is dispatched."""
    assert (
        _plan_donations(
            donations=MappingProxyType(
                {
                    (("alive", 0, "main"), ()): (_donation(),),
                    (("alive", 0, "main"), (("consumption", 4),)): (_donation(),),
                }
            )
        )
        is None
    )
