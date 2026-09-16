"""Two regimes acting in one period: the first one's donation reaches the second.

A donation replaces the three rolling input mappings rather than editing them,
so every later use in the same period has to read the replaced mappings. With
one acting regime per period nothing observes the difference: the period's only
dispatch is also its first. Here both regimes act in every period and both
donate, so the second dispatch of a period runs after a donation has already
replaced the mappings, and the next period reads what the second dispatch left
behind.

The twin is the same model solved with the same solver class with its
`donation_candidates` emptied: same programs, same arguments, same dispatch
order, nothing handed to the compiler. Its published values are the reference.
"""

import dataclasses
import functools
from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.scheduler import DispatchUnit
from _lcm.execution.value_transfer import ValueArtifactAddress
from _lcm.solution import backward_induction
from _lcm.solution.continuation_reads import continuation_leaf_reads
from _lcm.solution.solve_inputs import SolveInputMappings, locate_artifact
from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model, Regime, categorical
from lcm.solver_api import (
    ArtifactKey,
    ContinuationCapabilities,
    KernelOutput,
    SolverExecutionCapabilities,
)
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
from lcm.typing import Float1D, FloatND, ScalarFloat, ScalarInt, StateName

_N_PERIODS = 3
_WEALTH = LinSpacedGrid(start=1.0, stop=5.0, n_points=5)
_ACTING_REGIMES = ("early", "late")
_COUNTER = ArtifactKey(type_id="tests.two_regime_donated_counter", schema_version=1)


@categorical(ordered=False)
class RegimeId:
    """Two acting regimes that stay where they are, plus a terminal one."""

    early: ScalarInt
    late: ScalarInt
    dead: ScalarInt


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


def _utility(wealth: ScalarFloat) -> ScalarFloat:
    """Report flow utility equal to wealth, so the fixture stays arithmetic."""
    return wealth


def _next_wealth(wealth: ScalarFloat) -> ScalarFloat:
    """Keep wealth where it is, so every period sees the same grid."""
    return wealth


def _stay(age: ScalarFloat) -> ScalarFloat:  # noqa: ARG001
    """Send the whole mass back into the regime it came from."""
    return jnp.asarray(1.0)


def _counting_value(
    *, wealth: Float1D, count: FloatND
) -> tuple[Float1D, _ReadableCounter]:
    """Publish wealth plus the running count, and the count incremented."""
    return wealth + count, _ReadableCounter(count=count + 1.0)


# keyword-only-exempt: library-callback=lcm.solvers.CoreProgram.argument_builder
def _counting_arguments(
    build: CoreBuildContext, /, *, regime_name: str
) -> dict[str, object]:
    """Feed the state grid and the regime's own published count to the program."""
    return {
        "wealth": build.state_action_space.states["wealth"],  # ty: ignore[unresolved-attribute]
        "count": build.next_regime_to_continuation[regime_name].count,  # ty: ignore[unresolved-attribute]
    }


@dataclasses.dataclass(frozen=True, kw_only=True)
class _GraphKernel:
    """A period kernel dispatching its single declared program."""

    program: CoreProgram
    regime_name: str

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Expose the one program the engine compiles and dispatches."""
        return MappingProxyType({"main": self.program})

    def with_fixed_params(self, *, fixed_flat_params: object) -> _GraphKernel:  # noqa: ARG002
        """Return itself: the fixture reads no fixed parameter."""
        return self

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, object],
        state_action_space: object,
        next_regime_to_V_arr: Mapping[str, object],
        next_regime_to_continuation: Mapping[str, object],
        flat_params: Mapping[str, object],
        period: int,
        ages: object,
        logger: object,  # noqa: ARG002
        **_unused: object,
    ) -> KernelOutput:
        """Build the program's arguments and dispatch its compiled core."""
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        value, artifact = compiled_cores["main"](  # ty: ignore[call-non-callable]
            **self.program.argument_builder(context)
        )
        return KernelOutput(value=value, continuations={_COUNTER: artifact})


class _CounterSolver(Solver):
    """Reads its own next-period count and republishes it incremented."""

    donation_candidates: tuple[str, ...] = ()

    @property
    def capabilities(self) -> SolverExecutionCapabilities:
        """Describe the fixture's narrowly scoped reference computation."""
        return SolverExecutionCapabilities(
            required_declaration="Regime",
            problem_shape="Reference fixture computation",
            prerequisites="Fixture-specific model contract",
            main_tradeoff="Reference implementation for contract tests",
        )

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        """Demand the counter the regime itself publishes."""
        return frozenset({_COUNTER})

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Declare one dense program per active period, reading the count leaf."""
        regime_name = context.regime_name
        template = _ReadableCounter(count=jnp.zeros(()))
        kernels = {}
        for period in context.regimes_to_active_periods[regime_name]:
            program = CoreProgram(
                name="main",
                function=_counting_value,
                argument_builder=functools.partial(
                    _counting_arguments, regime_name=regime_name
                ),
                requirements=CoreExecutionRequirements(
                    value_reads=continuation_leaf_reads(
                        template=template,
                        artifact_key=_COUNTER,
                        target=regime_name,
                        source_regime=regime_name,
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
            kernels[period] = _GraphKernel(program=program, regime_name=regime_name)
        return SolutionKernels(
            period_kernels=MappingProxyType(kernels),
            continuation_spec=ContinuationSpec(
                template=template, artifact_key=_COUNTER
            ),
            replay_route=DeclaredReplay.GRID_RECOMPUTATION,
        )


class _DonatingCounterSolver(_CounterSolver):
    """The same solver, naming its count argument as a donation candidate."""

    donation_candidates = ("count",)


def _always(age: ScalarFloat) -> bool:  # noqa: ARG001
    """Keep the regime acting in every period of the age grid."""
    return True


def _model(*, solver_class: type[_CounterSolver]) -> Model:
    """Build the two-regime model, both regimes acting in every period."""
    return Model(
        regimes={
            name: Regime(
                transition={name: MarkovTransition(_stay)},
                active=_always,
                states={"wealth": _WEALTH},
                state_transitions={"wealth": _next_wealth},
                functions={"utility": _utility},
                solver=solver_class(),
            )
            for name in _ACTING_REGIMES
        }
        | {
            "dead": Regime(
                transition=None,
                states={"wealth": _WEALTH},
                functions={"utility": lambda wealth: 0.0 * wealth},
            )
        },
        ages=AgeGrid(start=0, stop=_N_PERIODS - 1, step="Y"),
        regime_id_class=RegimeId,
    )


def _published(
    *, solver_class: type[_CounterSolver]
) -> dict[tuple[int, str], np.ndarray]:
    """Solve and return every published value array, keyed by period and regime."""
    solution = _model(solver_class=solver_class).solve(
        params={"discount_factor": 1.0}, log_level="off"
    )
    return {
        (period, regime): np.asarray(solution.values[period][regime])
        for period in range(_N_PERIODS)
        for regime in _ACTING_REGIMES
    }


@pytest.fixture(scope="module")
def published_values() -> tuple[
    dict[tuple[int, str], np.ndarray], dict[tuple[int, str], np.ndarray]
]:
    """The values the donating solve and its non-donating twin publish."""
    return (
        _published(solver_class=_DonatingCounterSolver),
        _published(solver_class=_CounterSolver),
    )


@pytest.mark.parametrize(
    ("period", "regime"),
    [(period, regime) for period in range(_N_PERIODS) for regime in _ACTING_REGIMES],
)
def test_donation_by_the_first_of_two_acting_regimes_changes_no_value(
    *,
    period: int,
    regime: str,
    published_values: tuple[
        dict[tuple[int, str], np.ndarray], dict[tuple[int, str], np.ndarray]
    ],
) -> None:
    """Both regimes publish bit-identical values with and without donation."""
    donating, plain = published_values

    np.testing.assert_array_equal(donating[(period, regime)], plain[(period, regime)])


@dataclasses.dataclass(frozen=True, kw_only=True)
class _UnitInputs:
    """What one dispatch unit was handed, and what it was allowed to donate."""

    unit: DispatchUnit
    inputs: SolveInputMappings
    templates: SolveInputMappings
    donated: tuple[ValueArtifactAddress, ...]


def _record_unit_inputs(*, monkeypatch: pytest.MonkeyPatch) -> list[_UnitInputs]:
    """Solve the donating model, recording the mappings every unit was handed."""
    recorded: list[_UnitInputs] = []
    donate = backward_induction._donated_input_arrays

    def _observe(**kwargs: object) -> tuple[object, ...]:
        donated = donate(**kwargs)  # ty: ignore[invalid-argument-type]
        recorded.append(
            _UnitInputs(
                unit=kwargs["unit"],  # ty: ignore[invalid-argument-type]
                inputs=kwargs["inputs"],  # ty: ignore[invalid-argument-type]
                templates=kwargs["templates"],  # ty: ignore[invalid-argument-type]
                donated=tuple(item.artifact for item in donated),
            )
        )
        return donated

    monkeypatch.setattr(backward_induction, "_donated_input_arrays", _observe)
    _published(solver_class=_DonatingCounterSolver)
    return recorded


def test_the_first_regime_of_a_period_donates_before_the_second_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard the premise: 'early' hands a buffer over ahead of 'late' every period."""
    recorded = _record_unit_inputs(monkeypatch=monkeypatch)

    donating_periods = {record.unit.period for record in recorded if record.donated}
    assert donating_periods == {0, 1}
    for period in sorted(donating_periods):
        units = [
            record.unit.regime for record in recorded if record.unit.period == period
        ]
        assert units.index("early") < units.index("late")
        assert [
            record.donated
            for record in recorded
            if record.unit.period == period and record.unit.regime == "early"
        ] != [()]


def test_the_second_regime_of_a_period_reads_the_replaced_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A buffer 'early' donated stands at its template when 'late' is planned.

    A donation replaces the three mappings, so a `SolveInputMappings` built
    before it is stale. The second unit of the period must be handed the
    replaced mappings, not the snapshot the period opened with.
    """
    recorded = _record_unit_inputs(monkeypatch=monkeypatch)
    by_dispatch = {
        (record.unit.period, record.unit.regime): record for record in recorded
    }

    for period in (0, 1):
        for artifact in by_dispatch[(period, "early")].donated:
            later = by_dispatch[(period, "late")]
            assert locate_artifact(inputs=later.inputs, artifact=artifact) is (
                locate_artifact(inputs=later.templates, artifact=artifact)
            )
