"""Donation hands over a buffer only the donating argument still reads.

A dispatch unit may aim several declared input locators at one continuation
leaf — two arguments of one core, or a donor core beside a non-donating
sibling. The unit is one consumer of that leaf, but the executable receives it
as more than one argument, so an argument is donated only when it is the one
declared locator of the whole unit for every artifact it carries. Width
alternatives of one core repeat that core's locators and count once. Two cores
that each nominate the same artifact are a different matter: nobody can say
which of them should keep it, so the plan is refused instead.

The tests come in two layers, and only the first covers everything. The solves
run the production route end to end, planner wiring included. The synthetic
units below call the planner's stages directly on programs they construct, so
they pin the decision but not the wiring that assembles the census for it.
"""

import dataclasses
import itertools
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    ResolvedCoreProgram,
    ValueRead,
)
from _lcm.execution.donation import (
    DonatedBuffer,
    ResolvedDonation,
    resolve_donations,
    unit_input_readers,
    withhold_shared_donations,
)
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
)
from _lcm.solution import backward_induction
from _lcm.solution.continuation_reads import continuation_leaf_reads
from _lcm.solution.kernel_output import ConsumedKernelOutput, KernelOutput
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import ArtifactKey, ContinuationCapabilities
from lcm.solvers import (
    ContinuationSpec,
    CoreProgram,
    DeclaredReplay,
    OutputRole,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    StateAxesLeading,
)
from lcm.solvers import (
    CoreExecutionDisposition as PublicDisposition,
)
from lcm.solvers import (
    CoreExecutionRequirements as PublicRequirements,
)
from lcm.typing import Float1D, FloatND, StateName
from tests.test_solver_api_out_of_tree import _N_PERIODS, _two_regime_model

_KEY = ArtifactKey(type_id="tests.donation_read_multiplicity", schema_version=1)

# The count array each `alive` dispatch of the last solve was handed.
_DISPATCHED: list[FloatND] = []


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Counter:
    """A one-leaf continuation whose array reaches the dispatch by identity."""

    count: FloatND

    @property
    def artifact_key(self) -> ArtifactKey:
        """Name the versioned key this payload is published under."""
        return _KEY

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


def _value_from_one_read(
    *, wealth: Float1D, count: FloatND
) -> tuple[Float1D, _Counter]:
    """Publish wealth plus the running count, and the count incremented."""
    return wealth + count, _Counter(count=count + 1.0)


def _value_from_two_reads(
    *, wealth: Float1D, count: FloatND, other: FloatND
) -> tuple[Float1D, _Counter]:
    """Publish wealth plus both readings of the count, and it incremented."""
    return wealth + 0.5 * count + 0.5 * other, _Counter(count=count + 1.0)


def _sibling_sum(*, wealth: Float1D, other: FloatND) -> Float1D:
    """The non-donating sibling core's own reading of the same leaf."""
    return wealth + other


def _one_read_arguments(build: object) -> dict[str, object]:
    """Feed the state grid and the target's published count to the program."""
    payload = build.next_regime_to_continuation["alive"]  # ty: ignore[unresolved-attribute]
    _DISPATCHED.append(payload.count)
    return {
        "wealth": build.state_action_space.states["wealth"],  # ty: ignore[unresolved-attribute]
        "count": payload.count,
    }


def _two_read_arguments(build: object) -> dict[str, object]:
    """Feed one published count array to two declared arguments of one core."""
    payload = build.next_regime_to_continuation["alive"]  # ty: ignore[unresolved-attribute]
    _DISPATCHED.append(payload.count)
    return {
        "wealth": build.state_action_space.states["wealth"],  # ty: ignore[unresolved-attribute]
        "count": payload.count,
        "other": payload.count,
    }


def _sibling_arguments(build: object) -> dict[str, object]:
    """Feed the same published count array to the non-donating sibling core."""
    payload = build.next_regime_to_continuation["alive"]  # ty: ignore[unresolved-attribute]
    return {
        "wealth": build.state_action_space.states["wealth"],  # ty: ignore[unresolved-attribute]
        "other": payload.count,
    }


@dataclasses.dataclass(frozen=True, kw_only=True)
class _MainKernel:
    """A period kernel dispatching its value-producing core, and no other."""

    programs: MappingProxyType[str, CoreProgram]

    def core_programs(self) -> MappingProxyType[str, CoreProgram]:
        """Return the core graph the planner resolves."""
        return self.programs

    def with_fixed_params(self, *, fixed_flat_params: object) -> _MainKernel:  # noqa: ARG002
        """Return this parameter-free kernel unchanged."""
        return self

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable[..., object]],
        state_action_space: object,
        next_regime_to_V_arr: Mapping[str, object],
        next_regime_to_continuation: Mapping[str, object],
        flat_params: Mapping[str, object],
        period: int,
        ages: object,
        logger: object,  # noqa: ARG002
        **_unused: object,
    ) -> KernelOutput:
        """Dispatch every compiled core, publishing the main core's outputs."""
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        published: tuple[Float1D, _Counter] | None = None
        for name, core in compiled_cores.items():
            output = core(**self.programs[name].argument_builder(context))
            if name == "main":
                published = cast("tuple[Float1D, _Counter]", output)
        assert published is not None
        return KernelOutput(value=published[0], continuations={_KEY: published[1]})


class _LeafReadingSolver(Solver):
    """Reads its own next-period count leaf through the arguments it names."""

    arguments: tuple[str, ...] = ("count",)
    donation_candidates: tuple[str, ...] = ()
    sibling: bool = False
    sibling_donates: bool = False

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        """Demand the counter its own regime publishes."""
        return frozenset({_KEY})

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Declare the dense cores of every active period of this regime."""
        template = _Counter(count=jnp.zeros(()))
        kernels = {}
        for period in context.regimes_to_active_periods[context.regime_name]:
            programs = {"main": self._main_program(period=period, template=template)}
            if self.sibling:
                programs["sibling"] = self._sibling_program(
                    period=period, template=template
                )
            kernels[period] = _MainKernel(programs=MappingProxyType(programs))
        return SolutionKernels(
            period_kernels=MappingProxyType(kernels),
            continuation_spec=ContinuationSpec(template=template, artifact_key=_KEY),
            replay_route=DeclaredReplay.GRID_RECOMPUTATION,
        )

    def _main_program(self, *, period: int, template: _Counter) -> CoreProgram:
        """Build the value-producing core with one read per named argument."""
        return CoreProgram(
            name="main",
            function=(
                _value_from_two_reads
                if len(self.arguments) == 2
                else _value_from_one_read
            ),
            argument_builder=(
                _two_read_arguments if len(self.arguments) == 2 else _one_read_arguments
            ),
            requirements=PublicRequirements(
                value_reads=tuple(
                    read
                    for argument in self.arguments
                    for read in continuation_leaf_reads(
                        template=template,
                        artifact_key=_KEY,
                        target="alive",
                        source_regime="alive",
                        source_period=period,
                        core_key="main",
                        argument_by_leaf={("count",): argument},
                    )
                )
            ),
            output_roles=(
                OutputRole.VALUE,
                _Counter(count=StateAxesLeading(state_names=(), shape=())),  # ty: ignore[invalid-argument-type]
            ),
            disposition=PublicDisposition.DENSE,
            disposition_reason="one_row_per_state_node",
            donation_candidates=self.donation_candidates,
        )

    def _sibling_program(self, *, period: int, template: _Counter) -> CoreProgram:
        """Build the second core of the unit, reading the same leaf as the first."""
        return CoreProgram(
            name="sibling",
            function=_sibling_sum,
            argument_builder=_sibling_arguments,
            requirements=PublicRequirements(
                value_reads=continuation_leaf_reads(
                    template=template,
                    artifact_key=_KEY,
                    target="alive",
                    source_regime="alive",
                    source_period=period,
                    core_key="sibling",
                    argument_by_leaf={("count",): "other"},
                )
            ),
            output_roles=OutputRole.VALUE,
            disposition=PublicDisposition.DENSE,
            disposition_reason="one_row_per_state_node",
            donation_candidates=("other",) if self.sibling_donates else (),
        )


class _DonatingOneReadSolver(_LeafReadingSolver):
    """One declared locator for the leaf, named as a donation candidate."""

    donation_candidates = ("count",)


class _TwoReadSolver(_LeafReadingSolver):
    """Two declared locators of one core for the leaf, donating neither."""

    arguments = ("count", "other")


class _DonatingTwoReadSolver(_TwoReadSolver):
    """Two declared locators of one core, one of them a donation candidate."""

    donation_candidates = ("count",)


class _DonatingSiblingSolver(_LeafReadingSolver):
    """A donor core beside a non-donating sibling core reading the same leaf."""

    donation_candidates = ("count",)
    sibling = True


class _SiblingSolver(_LeafReadingSolver):
    """The same two cores, with neither of them donating."""

    sibling = True


class _TwoDonorSolver(_LeafReadingSolver):
    """Two cores of one unit, each nominating its own read of the same leaf."""

    donation_candidates = ("count",)
    sibling = True
    sibling_donates = True


def _solve(*, solver: Solver) -> object:
    """Solve the self-looping two-regime model with the given solver."""
    _DISPATCHED.clear()
    return _two_regime_model(solver=solver, self_looping=True).solve(
        params={"discount_factor": 1.0}, log_level="off"
    )


def published_values(*, solver: Solver) -> dict[int, np.ndarray]:
    """Return the `alive` value array one solver publishes in every period."""
    solution = _solve(solver=solver)
    return {
        period: np.asarray(solution.values[period]["alive"])  # ty: ignore[unresolved-attribute]
        for period in range(_N_PERIODS)
    }


def fanout_values_agree() -> bool:
    """Report whether the donating and plain two-locator solvers agree exactly."""
    donating = published_values(solver=_DonatingTwoReadSolver())
    plain = published_values(solver=_TwoReadSolver())
    return all(
        np.array_equal(donating[period], plain[period]) for period in range(_N_PERIODS)
    )


@pytest.fixture(scope="module")
def two_read_values() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """The values the donating and plain two-locator solvers publish."""
    return (
        published_values(solver=_DonatingTwoReadSolver()),
        published_values(solver=_TwoReadSolver()),
    )


@pytest.mark.parametrize("period", range(_N_PERIODS))
def test_a_core_reading_one_leaf_through_two_arguments_publishes_the_plain_values(
    *,
    period: int,
    two_read_values: tuple[dict[int, np.ndarray], dict[int, np.ndarray]],
) -> None:
    """Naming one of two locators a candidate changes no published value."""
    donating, plain = two_read_values

    np.testing.assert_array_equal(donating[period], plain[period])


@pytest.fixture(scope="module")
def sibling_values() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """The values the donor-plus-sibling and plain sibling solvers publish."""
    return (
        published_values(solver=_DonatingSiblingSolver()),
        published_values(solver=_SiblingSolver()),
    )


@pytest.mark.parametrize("period", range(_N_PERIODS))
def test_a_donor_core_beside_a_sibling_reading_one_leaf_publishes_the_plain_values(
    *,
    period: int,
    sibling_values: tuple[dict[int, np.ndarray], dict[int, np.ndarray]],
) -> None:
    """A second core reading the leaf keeps it, whatever the donor names."""
    donating, plain = sibling_values

    np.testing.assert_array_equal(donating[period], plain[period])


def test_two_cores_of_one_unit_nominating_one_leaf_are_refused() -> None:
    """A plan handing one buffer to two cores is named and stopped before lowering."""
    with pytest.raises(ExecutionPlanningError, match=r"leaf_path=\('count',\)"):
        _solve(solver=_TwoDonorSolver())


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


def test_a_sole_locator_candidate_is_still_donated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One declared locator for the leaf keeps donation enabled at every period."""
    assert _reads_deleted_at_dispatch(
        solver=_DonatingOneReadSolver(), monkeypatch=monkeypatch
    ) == [(2, False), (1, True), (0, True)]


def test_a_candidate_a_second_argument_reads_is_not_donated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leaf survives its dispatch while the other argument still reads it."""
    assert _reads_deleted_at_dispatch(
        solver=_DonatingTwoReadSolver(), monkeypatch=monkeypatch
    ) == [(2, False), (1, False), (0, False)]


_FOUR_DEVICE_PROGRAM = """
import jax

jax.config.update("jax_num_cpu_devices", 4)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", {x64})

from tests.solution.test_donation_read_multiplicity import fanout_values_agree

print(len(jax.devices()), fanout_values_agree())
"""


def test_two_arguments_reading_one_leaf_agree_on_a_four_device_topology() -> None:
    """The rule holds where the leaf is sharded: four CPU devices, own process."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            _FOUR_DEVICE_PROGRAM.format(x64=jax.config.read("jax_enable_x64")),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent.parent.parent,
    )

    assert result.stdout.split() == ["4", "True"], result.stderr


_REGIME = "alive"
_PERIOD = 2
_N_SOLVE_PERIODS = 4


def _artifact(*, label: str) -> ValueArtifactAddress:
    """The continuation leaf one label of a synthetic unit addresses."""
    return ValueArtifactAddress(
        kind=ValueArtifactKind.CONTINUATION_LEAF,
        period=_PERIOD + 1,
        regime=_REGIME,
        artifact_key=_KEY,
        leaf_path=(label,),
    )


def _locator(*, core: str, argument: str) -> ValueConsumerAddress:
    """The declared input locator one argument of one core of the unit names."""
    return ValueConsumerAddress(
        source_period=_PERIOD,
        source_regime=_REGIME,
        core_key=core,
        channel=ValueInputChannel.CONTINUATION_LEAF,
        argument=argument,
        path=(),
    )


def _identity(**arguments: jax.Array) -> jax.Array:
    """The synthetic program body: it returns what its first argument holds."""
    return next(iter(arguments.values()))


def _resolved_program(
    *,
    core: str,
    occurrences: Sequence[tuple[str, str, str]],
    candidates: tuple[str, ...],
    width: int,
) -> ResolvedCoreProgram:
    """Build one core of a synthetic unit from its declared occurrences."""
    own = [entry for entry in occurrences if entry[0] == core]
    reads = tuple(
        ValueRead(
            target=_artifact(label=label),
            source=_locator(core=core, argument=argument),
        )
        for _core, argument, label in own
    )
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    array = jnp.zeros(3)
    return ResolvedCoreProgram(
        name=core,
        function=_identity,
        arguments=MappingProxyType({argument: array for _c, argument, _l in own}),
        static_kwargs=MappingProxyType({}),
        requirements=CoreExecutionRequirements(value_reads=reads),
        output_roles=None,
        disposition=CoreExecutionDisposition.DENSE,
        donation_candidates=candidates,
        tile_widths=MappingProxyType({"consumption": width}),
        specialization_key=("synthetic",),
        input_transfer_plan=tuple(
            ResolvedValueTransfer(
                target=read.target,
                source=read.source,
                kind=ValueTransferKind.ALIGNED_LOCAL,
                stored_sharding=sharding,
                source_sharding=sharding,
                expected_shape=array.shape,
                expected_dtype=array.dtype,
            )
            for read in reads
        ),
        disposition_reason="synthetic",
    )


def _synthetic_ledger(*, labels: frozenset[str]) -> PlannedInputLiveness:
    """The plan for a unit that is the one dispatch consuming every label.

    This is the ledger the production planner builds: a dispatch is counted
    once per artifact however many of its locators name it.
    """
    return PlannedInputLiveness(
        dispatch_accesses={
            (_PERIOD, _REGIME): tuple(
                _artifact(label=label) for label in sorted(labels)
            )
        }
    )


def _decide(
    *,
    occurrences: Sequence[tuple[str, str, str]],
    core: str,
    candidates: tuple[str, ...],
    widths: tuple[int, ...],
) -> tuple[ResolvedDonation, ...]:
    """Resolve one core's donations against the whole synthetic unit's census."""
    cores = dict.fromkeys(entry[0] for entry in occurrences)
    programs = [
        _resolved_program(
            occurrences=occurrences,
            core=other,
            candidates=candidates if other == core else (),
            width=width,
        )
        for other in cores
        for width in widths
    ]
    nominated = next(
        program
        for program in programs
        if program.name == core and program.donation_candidates
    )
    # The census is assembled the way the planner assembles it: the unit's
    # programs are grouped under their dispatch and censused as a group.
    by_dispatch: dict[tuple[int, str], list[ResolvedCoreProgram]] = {}
    for program in programs:
        by_dispatch.setdefault((_PERIOD, _REGIME), []).append(program)
    readers_by_dispatch = {
        dispatch: unit_input_readers(programs=group)
        for dispatch, group in by_dispatch.items()
    }
    return withhold_shared_donations(
        program=nominated,
        donations=resolve_donations(
            program=nominated,
            dispatch=(_PERIOD, _REGIME),
            ledger=_synthetic_ledger(
                labels=frozenset(entry[2] for entry in occurrences)
            ),
            n_periods=_N_SOLVE_PERIODS,
        ),
        unit_readers=readers_by_dispatch[(_PERIOD, _REGIME)],
    )


def test_a_second_argument_of_one_core_withholds_the_donation() -> None:
    """One core reading a leaf twice donates neither of the two arguments."""
    decisions = _decide(
        occurrences=[("main", "count", "A"), ("main", "other", "A")],
        core="main",
        candidates=("count",),
        widths=(1,),
    )

    assert [decision.donated for decision in decisions] == [False]


def test_a_withheld_donation_names_the_second_locator() -> None:
    """The decision records which other locator keeps the buffer readable."""
    decisions = _decide(
        occurrences=[("main", "count", "A"), ("main", "other", "A")],
        core="main",
        candidates=("count",),
        widths=(1,),
    )

    assert decisions[0].withheld_by == _locator(core="main", argument="other")


def test_a_withheld_donation_still_names_the_buffer_it_would_have_handed_over() -> None:
    """Withholding is not a transfer: the argument still reaches its own buffer."""
    decisions = _decide(
        occurrences=[("main", "count", "A"), ("main", "other", "A")],
        core="main",
        candidates=("count",),
        widths=(1,),
    )

    assert decisions[0].buffer is DonatedBuffer.STORED_ARTIFACT


def test_a_sibling_core_reading_the_same_leaf_withholds_the_donation() -> None:
    """A non-donating core of the unit is a reader the donor may not invalidate."""
    decisions = _decide(
        occurrences=[("main", "count", "A"), ("sibling", "other", "A")],
        core="main",
        candidates=("count",),
        widths=(1,),
    )

    assert decisions[0].withheld_by == _locator(core="sibling", argument="other")


def test_a_sibling_core_reading_another_leaf_leaves_the_donation_standing() -> None:
    """Two cores reading two leaves are two exclusive owners, so both donate."""
    decisions = _decide(
        occurrences=[("main", "count", "A"), ("sibling", "other", "B")],
        core="main",
        candidates=("count",),
        widths=(1,),
    )

    assert [decision.donated for decision in decisions] == [True]


def test_three_width_alternatives_of_one_core_leave_the_donation_standing() -> None:
    """Width alternatives repeat one core's locators, so they count once."""
    decisions = _decide(
        occurrences=[("main", "count", "A")],
        core="main",
        candidates=("count",),
        widths=(1, 2, 3),
    )

    assert [decision.donated for decision in decisions] == [True]


def _exclusive_candidates(
    *,
    occurrences: Sequence[tuple[str, str, str]],
    candidates: Sequence[tuple[str, str]],
) -> set[tuple[str, str]]:
    """Return the candidates no other occurrence of the unit shares a label with.

    An independent census over the width-free semantic graph: it enumerates
    occurrence pairs directly and consults neither the ledger nor the planner's
    own bookkeeping.
    """
    exclusive: set[tuple[str, str]] = set()
    for core, argument in candidates:
        selected = [entry for entry in occurrences if entry[:2] == (core, argument)]
        if not selected:
            continue
        if all(
            not any(
                other[2] == entry[2] and other[:2] != entry[:2] for other in occurrences
            )
            for entry in selected
        ):
            exclusive.add((core, argument))
    return exclusive


def _occurrence_graphs() -> list[
    tuple[str, list[tuple[str, str, str]], tuple[int, ...]]
]:
    """Enumerate the small units the planner and the independent census share."""
    graphs = []
    for n_cores, fanout, shared, widths, reverse in itertools.product(
        (1, 2, 3), (1, 2, 3), (True, False), ((1,), (1, 2, 3)), (False, True)
    ):
        occurrences = [
            (
                f"core{core}",
                f"argument{argument}",
                "A" if shared else f"A{core}{argument}",
            )
            for core in range(n_cores)
            for argument in range(fanout)
        ]
        if reverse:
            occurrences.reverse()
        label = (
            f"cores{n_cores}-fanout{fanout}-"
            f"{'shared' if shared else 'distinct'}-"
            f"widths{len(widths)}-{'reversed' if reverse else 'forward'}"
        )
        graphs.append((label, occurrences, widths))
    return graphs


_GRAPHS = _occurrence_graphs()


@pytest.mark.parametrize(
    ("occurrences", "widths"),
    [(occurrences, widths) for _label, occurrences, widths in _GRAPHS],
    ids=[label for label, _occurrences, _widths in _GRAPHS],
)
def test_the_planner_donates_exactly_the_exclusive_occurrences(
    *,
    occurrences: list[tuple[str, str, str]],
    widths: tuple[int, ...],
) -> None:
    """The planner's decision matches a literal count of distinct locators."""
    decisions = _decide(
        occurrences=occurrences,
        core="core0",
        candidates=("argument0",),
        widths=widths,
    )

    assert [decision.donated for decision in decisions] == [
        ("core0", "argument0")
        in _exclusive_candidates(
            occurrences=occurrences, candidates=[("core0", "argument0")]
        )
    ]
