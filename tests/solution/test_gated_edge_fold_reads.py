"""The engine's gated-edge fold is a declared consumer of the values it reads.

A fold reads the target's value and every reference regime's value at the period
it folds. Those reads are counted like a core's, so the ledger says who still
needs a value.
"""

import jax
import jax.numpy as jnp

from _lcm.execution.core_program import (
    ProgramScope,
    ValueRead,
    core_program_graph,
)
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactKind,
    ValueTransferKind,
)
from _lcm.solution.backward_induction import (
    _build_planned_input_liveness,
    _ProgramExecutionMetadata,
    gated_edge_fold_value_reads,
)
from lcm import AgeGrid, Model
from tests.regime_building.test_gated_edges_collective_solve import (
    EKLRegimeId,
    _make_full_topology_regimes,
)

_PERIOD = 1


def _model() -> Model:
    """The full gated-edge topology: a consent edge, a dissolution edge, IR."""
    return Model(
        regimes=_make_full_topology_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=EKLRegimeId,
    )


def _program_metadata(
    *, model: Model
) -> dict[tuple[str, int, str], _ProgramExecutionMetadata]:
    """Synthetic metadata for every declared program of every regime-period.

    Each program keeps its own declared requirements, paired with the aligned
    single-device transfer plan those reads resolve to on one device, so the
    core dispatches enter the ledger the way a solve enters them.
    """
    return {
        (regime_name, period, core_key): _ProgramExecutionMetadata(
            requirements=program.requirements,
            disposition=program.disposition,
            scope=ProgramScope.ANY,
            input_transfer_plan=tuple(
                _aligned_transfer(read=read)
                for read in program.requirements.value_reads
            ),
        )
        for regime_name, regime in model._regimes.items()
        for period, kernel in regime.solution.period_kernels.items()
        for core_key, program in core_program_graph(kernel=kernel).items()
    }


def _aligned_transfer(*, read: ValueRead) -> ResolvedValueTransfer:
    """Resolve one declared read as the aligned transfer of a one-device solve."""
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    return ResolvedValueTransfer(
        target=read.target,
        source=read.source,
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_sharding=sharding,
        source_sharding=sharding,
        expected_shape=(1,),
        expected_dtype=jnp.float64,
    )


def test_a_fold_declares_a_read_of_every_regime_it_evaluates() -> None:
    """A period's folds declare a read of exactly the regimes they evaluate on.

    Both consent edges fold onto `married` here, and their gate reads the two
    singleton regimes the partners fall back to. No other regime's value enters
    a fold at this period, so no other regime is declared.
    """
    reads = gated_edge_fold_value_reads(regimes=_model()._regimes, period=_PERIOD)

    assert {read.target.regime for read in reads} == {
        "married",
        "single_f_p1",
        "single_m_p1",
    }


def test_every_fold_read_names_a_regime_value_of_the_folded_period() -> None:
    """A fold reads raw same-period values, not continuations."""
    reads = gated_edge_fold_value_reads(regimes=_model()._regimes, period=_PERIOD)

    assert {(read.target.kind, read.target.period) for read in reads} == {
        (ValueArtifactKind.REGIME_VALUE, _PERIOD)
    }


def test_a_folded_value_has_the_fold_among_its_planned_consumers() -> None:
    """A value the fold reads is counted, so a scheduler can see it close."""
    model = _model()
    ledger = _build_planned_input_liveness(
        regimes=model._regimes,
        program_metadata=_program_metadata(model=model),
    )
    folded = gated_edge_fold_value_reads(regimes=model._regimes, period=_PERIOD)[
        0
    ].target

    # Two consent edges fold onto `married` at this period and no core reads
    # its raw value, so the folds are its two planned consumers.
    assert ledger.remaining_consumers(artifact=folded) == 2


def test_a_fold_dispatch_is_a_planned_dispatch_of_its_own() -> None:
    """The fold of one edge at one period is an identified dispatch."""
    model = _model()
    ledger = _build_planned_input_liveness(
        regimes=model._regimes,
        program_metadata=_program_metadata(model=model),
    )

    assert any(
        len(dispatch) == 3 and dispatch[0] == _PERIOD
        for dispatch in ledger.pending_dispatches
    )


def test_a_folded_value_is_pinned_by_retention_rather_than_by_the_fold() -> None:
    """A folded value's only unplanned pin is the retained public result.

    Every regime value of every active period is retained in the solve result,
    so a folded value stays ineligible for release even once every planned
    consumer of it — the folds among them — has committed.
    """
    model = _model()
    ledger = _build_planned_input_liveness(
        regimes=model._regimes,
        program_metadata=_program_metadata(model=model),
    )
    folded = gated_edge_fold_value_reads(regimes=model._regimes, period=_PERIOD)[
        0
    ].target
    for dispatch in tuple(ledger.pending_dispatches):
        ledger.commit_successful_dispatch(dispatch=dispatch)
    # Refuses unless every planned dispatch committed and every count reached
    # zero, so the assertion below reads a genuinely closed count.
    ledger.assert_solve_complete()

    assert not ledger.is_release_eligible(artifact=folded)
