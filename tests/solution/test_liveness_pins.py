"""The ledger counts every declared read and pins exactly the undeclared ones.

A regime value is retained by the solve result. A dense program's declared reads
are counted like a planned program's. Two reads remain undeclared — the EGM
family's host read of the target's `breakpoints` leaf, and every read of a
program that declares none — and those are pinned so no release can free them.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest

from _lcm.continuation import ContinuationSpec
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    ProgramScope,
    ValueRead,
    core_program_graph,
)
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueTransferKind,
)
from _lcm.solution.backward_induction import (
    _build_planned_input_liveness,
    _ProgramExecutionMetadata,
)
from _lcm.solution.continuation_reads import published_continuation_template
from _lcm.solution.undeclared_reads import (
    HOST_READ_CONTINUATION_LEAVES,
    undeclared_read_pins,
)
from _lcm.typing import RegimeName
from lcm import Model
from lcm.solver_api import EGM_CONTINUATION
from lcm.solvers import EGM, NNBEGM
from tests.solution.test_egm_solver import _SAVINGS_GRID
from tests.solution.test_egm_solver import _model as _egm_model
from tests.solution.test_gated_edge_fold_reads import _model as _gated_model
from tests.solution.test_gated_edge_fold_reads import _program_metadata
from tests.test_models import nbegm_jump_ride_along_toy as _jump_ride_along_toy
from tests.test_models.n_nbegm_toy import build_model as _build_nnbegm_model


def _metadata(*, model: Model) -> dict[tuple[str, int, str], _ProgramExecutionMetadata]:
    """Every declared program's metadata, with an aligned one-device plan."""
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    result: dict[tuple[str, int, str], _ProgramExecutionMetadata] = {}
    for regime_name, regime in model._regimes.items():
        for period, kernel in regime.solution.period_kernels.items():
            for core_key, program in core_program_graph(kernel=kernel).items():
                plan = tuple(
                    _aligned(read=read, sharding=sharding)
                    for read in program.requirements.value_reads
                )
                result[(regime_name, period, core_key)] = _ProgramExecutionMetadata(
                    requirements=program.requirements,
                    disposition=program.disposition,
                    scope=ProgramScope.ANY,
                    input_transfer_plan=(
                        plan
                        if program.disposition is CoreExecutionDisposition.PLANNED
                        else ()
                    ),
                )
    return result


def _aligned(
    *, read: ValueRead, sharding: jax.sharding.Sharding
) -> ResolvedValueTransfer:
    return ResolvedValueTransfer(
        target=read.target,
        source=read.source,
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_sharding=sharding,
        source_sharding=sharding,
        expected_shape=(1,),
        expected_dtype=jnp.float64,
    )


def test_every_regime_value_is_retained() -> None:
    """The solve result keeps every regime's value of every active period."""
    model = _egm_model(solver=EGM(savings_grid=_SAVINGS_GRID))
    ledger = _build_planned_input_liveness(
        regimes=model._regimes, program_metadata=_metadata(model=model)
    )

    assert ledger.retained_artifacts == frozenset(
        ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=period, regime=name
        )
        for name, regime in model._regimes.items()
        for period in regime.active_periods
    )


def test_a_dense_program_s_declared_leaf_read_is_a_counted_consumer() -> None:
    """A dense EGM program's continuation-leaf read counts, so it can close."""
    model = _egm_model(solver=EGM(savings_grid=_SAVINGS_GRID))
    metadata = _metadata(model=model)
    ledger = _build_planned_input_liveness(
        regimes=model._regimes, program_metadata=metadata
    )
    dense_reads = [
        read.target
        for entry in metadata.values()
        if entry.disposition is CoreExecutionDisposition.DENSE
        for read in entry.requirements.value_reads
    ]

    assert {ledger.remaining_consumers(artifact=target) for target in dense_reads} == {
        1
    }


def test_the_breakpoints_host_read_is_pinned_wherever_a_target_publishes_one() -> None:
    """The leaf the EGM builder reads on the host cannot be declared; it is pinned.

    `n_nbegm_toy` declares no case-piece or schedule boundary, so no variant of
    it ever publishes a `breakpoints` leaf; the ride-along jump-schedule toy
    does, and its `alive` regime is both the source and the self-target — a
    plain NB-EGM regime read by an EGM-family source.
    """
    model = _jump_ride_along_toy.build_model(
        variant="nbegm",
        n_liquid=8,
        liquid_max=30.0,
        n_savings=10,
        savings_max=28.0,
        n_consumption=6,
    )
    ledger = _build_planned_input_liveness(
        regimes=model._regimes, program_metadata=_metadata(model=model)
    )
    pinned = [
        artifact
        for artifact in ledger.remaining_counts
        if artifact.kind is ValueArtifactKind.CONTINUATION_LEAF
        and artifact.leaf_path == ("breakpoints",)
    ]

    assert (
        bool(pinned),
        all(not ledger.is_release_eligible(artifact=artifact) for artifact in pinned),
    ) == (True, True)


def test_host_read_leaves_name_exactly_the_egm_breakpoints_row() -> None:
    """The one host read the EGM family keeps is the target's `breakpoints`."""
    assert dict(HOST_READ_CONTINUATION_LEAVES) == {
        EGM_CONTINUATION: (("breakpoints",),)
    }


def test_a_program_declaring_no_reads_pins_every_reachable_continuation_leaf() -> None:
    """An undeclared reader pins its targets' values, continuations and leaves."""
    model = _build_nnbegm_model(variant="n_nbegm")
    regimes = model._regimes
    source = next(
        name
        for name, regime in model.user_regimes.items()
        if isinstance(regime.solver, NNBEGM)
    )
    period = regimes[source].active_periods[0]
    pins = undeclared_read_pins(
        regimes=regimes, regime_name=source, period=period, declares_no_reads=True
    )
    targets = regimes[source].solution.reachability.targets(
        period=period, source=source
    )
    leaf_pins = {
        (artifact.regime, artifact.leaf_path)
        for artifact in pins
        if artifact.kind is ValueArtifactKind.CONTINUATION_LEAF
    }
    continuation_specs: dict[RegimeName, ContinuationSpec] = {
        name: r.solution.continuation_spec
        for name, r in regimes.items()
        if r.solution.continuation_spec is not None
    }

    assert leaf_pins == {
        (target, path)
        for target in targets
        for path in _published_leaf_paths(
            continuation_specs=continuation_specs, target=target
        )
    }


def _published_leaf_paths(
    *,
    continuation_specs: Mapping[RegimeName, ContinuationSpec],
    target: RegimeName,
) -> tuple[tuple[str, ...], ...]:
    """Return the leaf paths `target`'s continuation publishes, or none."""
    template = published_continuation_template(
        continuation_specs=continuation_specs, target=target
    )
    return () if template is None else tuple(template.leaves())


@pytest.mark.parametrize("declares_no_reads", [True, False])
def test_the_last_period_pins_nothing_beyond_same_period_references(
    *, declares_no_reads: bool
) -> None:
    """At the last period there is no next period to read."""
    model = _egm_model(solver=EGM(savings_grid=_SAVINGS_GRID))
    regimes = model._regimes
    name, regime = next(iter(regimes.items()))
    last = regime.solution.reachability.n_periods - 1

    assert (
        undeclared_read_pins(
            regimes=regimes,
            regime_name=name,
            period=last,
            declares_no_reads=declares_no_reads,
        )
        == ()
    )


def test_a_rolled_gated_continuation_aliases_the_same_edge_one_period_later() -> None:
    """An edge that does not fold at a period keeps the later period's buffer."""
    model = _gated_model()
    ledger = _build_planned_input_liveness(
        regimes=model._regimes, program_metadata=_program_metadata(model=model)
    )

    assert all(
        source.kind is target.kind
        and target.period == source.period + 1
        and source.regime == target.regime
        and source.target_regime == target.target_regime
        for source, target in ledger.aliases.items()
    )
