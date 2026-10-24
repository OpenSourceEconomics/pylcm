"""Candidate-specific exclusions from one immutable schedule inventory."""

from collections.abc import Hashable, Mapping
from dataclasses import replace
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.arrival_continuation import (
    ARRIVAL_CONTINUATION,
    ArrivalContinuation,
    ArrivalContinuationSpec,
)
from _lcm.execution import footprint
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    InternalInputRef,
    ResolvedCoreProgram,
)
from _lcm.execution.footprint import ArtifactFootprint, ScheduledUnit
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueArtifactKind
from _lcm.solution import backward_induction
from _lcm.solution.fingerprint import fingerprint_model_structure
from _lcm.typing import PytreeValue
from tests.solution.test_policy_queries import _native_model


class _ReachedDatedInventoryError(Exception):
    """Stop at native admission before executing the test-owned publisher schema."""


def test_same_shaped_arrival_origins_change_the_native_model_identity() -> None:
    """Equal numerical schema cannot authorize a different stored-value owner."""
    model = _native_model(enable_jit=True, source_size=2)
    payload = ArrivalContinuation(
        regime="root",
        period=0,
        schema="read-owner-discriminator",
        operands={"stored": jnp.zeros((2, 2))},
    )
    digests = []
    for owner in ("root", "target"):
        spec = ArrivalContinuationSpec(
            template=payload,
            artifact_key=ARRIVAL_CONTINUATION,
            templates_by_period={0: payload},
            origins_by_period={
                0: {
                    "stored": ValueArtifactAddress(
                        kind=ValueArtifactKind.REGIME_VALUE,
                        period=0,
                        regime=owner,
                    )
                }
            },
        )
        root = model._regimes["root"]
        digests.append(
            fingerprint_model_structure(
                ages=model._time,
                regimes={
                    **model._regimes,
                    "root": replace(
                        root,
                        solution=replace(root.solution, continuation_spec=spec),
                    ),
                },
                user_regimes=model.user_regimes,
                laws=model._graph.laws,
                regime_names_to_ids=model.regime_names_to_ids,
            )
        )
    assert digests[0] != digests[1]


def test_native_residency_inventories_dated_extra_leaves_and_concrete_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual budget boundary sees a leaf absent from the representative."""
    model = _native_model(enable_jit=True, budget=1_000_000_000)
    representative = ArrivalContinuation(
        regime="root", period=0, schema="representative", operands={}
    )
    dated_leaf = jnp.zeros(7, dtype=jnp.int32)
    dated = ArrivalContinuation(
        regime="root", period=0, schema="actual-date", operands={"extra": dated_leaf}
    )
    root = model._regimes["root"]
    spec = ArrivalContinuationSpec(
        template=representative,
        artifact_key=ARRIVAL_CONTINUATION,
        templates_by_period={0: dated},
    )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            {
                **model._regimes,
                "root": replace(
                    root, solution=replace(root.solution, continuation_spec=spec)
                ),
            }
        ),
    )
    inventoried: list[jax.Array] = []
    original = backward_induction.concrete_device_bytes

    def fixed_owners(*, tree):
        inventoried.extend(
            leaf for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)
        )
        return original(tree=tree)

    def actual_inventory(*, templates, **_kwargs):
        actual = templates.continuation_templates_by_period["root"][0]
        assert actual.operands["extra"] is dated_leaf
        assert any(leaf is dated_leaf for leaf in inventoried)
        raise _ReachedDatedInventoryError

    monkeypatch.setattr(backward_induction, "concrete_device_bytes", fixed_owners)
    monkeypatch.setattr(
        backward_induction, "_resident_inventory_by_triple", actual_inventory
    )
    with pytest.raises(_ReachedDatedInventoryError):
        model.solve(params={}, log_level="off", runtime_checks=False)


@pytest.mark.parametrize(
    ("consumes", "expected"),
    [((), 37), (("a",), 17), (("alias_a",), 17), (("b",), 27), (("a", "b"), 7)],
)
def test_candidates_query_only_their_own_live_alias_groups(
    *, consumes: tuple[Hashable, ...], expected: int
) -> None:
    """An independent two-device allocation sum includes concurrent output."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            (1, "producer"): (),
            (0, "active"): ("a", "b"),
            (0, "peer"): (),
        },
        retained_artifacts=("a", "alias_a", "b"),
        aliases={"alias_a": "a"},
    )
    producer = ScheduledUnit(
        period=1,
        regime="producer",
        device_ids=(0, 1),
        produces=("a", "alias_a", "b"),
        consumes=(),
        output_bytes_per_device=30,
    )
    active = ScheduledUnit(
        period=0,
        regime="active",
        device_ids=(0, 1),
        produces=(),
        consumes=("a", "b"),
        output_bytes_per_device=0,
    )
    peer = ScheduledUnit(
        period=0,
        regime="peer",
        device_ids=(1,),
        produces=(),
        consumes=(),
        output_bytes_per_device=7,
    )
    inventory = footprint.plan_resident_inventory(
        waves_by_period={1: ((producer,),), 0: ((active, peer),)},
        fold_dispatches={},
        ledger=ledger,
        footprints={
            "a": ArtifactFootprint(bytes_per_device=20, device_ids=(0, 1)),
            "alias_a": ArtifactFootprint(bytes_per_device=20, device_ids=(0, 1)),
            "b": ArtifactFootprint(bytes_per_device=10, device_ids=(1,)),
        },
    )[(0, "active")]
    # Device 0 carries A=20. Device 1 carries A=20, B=10, peer output=7.
    # Candidate input buffers are excluded once; aliases never charge twice.
    assert inventory.resident_bytes(consumes=consumes) == expected
    assert inventory.resident_bytes(consumes=()) == 37
    assert inventory.resident_bytes() == 7


def test_concrete_fixed_owners_deduplicate_aliases_and_ignore_abstracts() -> None:
    """Two physical allocations count once each despite repeated/view names."""
    source = jax.device_put(np.arange(16, dtype=np.int32), jax.devices()[0])
    alias = source.addressable_shards[0].data
    distinct = jax.device_put(np.full(8, 7, dtype=np.int32), jax.devices()[0])
    assert source.unsafe_buffer_pointer() == alias.unsafe_buffer_pointer()
    abstract = jax.ShapeDtypeStruct((1024 * 1024,), jnp.float32)
    sizes = footprint.concrete_device_bytes(
        tree=(source, alias, source, distinct, abstract)
    )
    assert dict(sizes) == {jax.devices()[0].id: (16 + 8) * 4}
    np.testing.assert_array_equal(source, np.arange(16))


def test_fixed_and_reserved_storage_sum_per_device_before_the_maximum() -> None:
    """Disjoint per-device burdens do not become a false sum of maxima."""
    inventory = footprint.ResidentInventory(
        device_ids=(0, 1),
        live={},
        peer_bytes={0: 3, 1: 0},
        declared_inputs=(),
        fixed_bytes=MappingProxyType({0: 100, 1: 20}),
        internal_bytes=11,
        shared_copies=MappingProxyType(
            {"shared": ArtifactFootprint(bytes_per_device=80, device_ids=(1,))}
        ),
    )
    assert inventory.resident_bytes(consumes=()) == 114  # max(100+3+11,20+80+11)
    assert inventory.resident_bytes(consumes=(), temporary_bytes={1: 9}) == 120
    # A compiler-live shared copy is excluded only from its reserved destination.
    assert (
        inventory.resident_bytes(consumes=(), consumed_copies=frozenset({"shared"}))
        == 114
    )
    smaller_fixed = replace(inventory, fixed_bytes=MappingProxyType({0: 0, 1: 20}))
    assert smaller_fixed.resident_bytes(consumes=()) == 111
    assert (
        smaller_fixed.resident_bytes(consumes=(), consumed_copies=frozenset({"shared"}))
        == 31
    )
    assert inventory.resident_bytes(consumes=()) == 114


def _inventory_with_fixed_only_device() -> footprint.ResidentInventory:
    """Workspace on device 1, a fixed owner on device 0, nothing on device 2."""
    return footprint.ResidentInventory(
        device_ids=(1,),
        live={},
        peer_bytes={1: 3},
        declared_inputs=(),
        fixed_bytes=MappingProxyType({0: 500, 1: 20, 2: 0}),
        internal_bytes=11,
    )


def test_admission_covers_a_device_holding_only_fixed_owners() -> None:
    """A device with fixed-owner bytes but no workspace is admitted."""
    assert _inventory_with_fixed_only_device().admission_device_ids == (0, 1)


def test_fixed_only_device_is_charged_its_owners_without_peer_or_internal() -> None:
    """The fixed-only device carries 500 bytes; the workspace carries 20+3+11."""
    inventory = _inventory_with_fixed_only_device()
    assert inventory.resident_bytes(consumes=()) == 500
    smaller = replace(inventory, fixed_bytes=MappingProxyType({0: 30, 1: 20}))
    assert smaller.resident_bytes(consumes=()) == 34


def _internal_pair(*, first: PytreeValue, second: PytreeValue) -> PytreeValue:
    return first, second


def test_internal_reservation_uses_producer_identity_and_candidate_maximum() -> None:
    """Abstract identity does not decide the number of future allocations."""
    requirements = CoreExecutionRequirements(
        internal_inputs={
            "first": InternalInputRef(producer="producer", label="first"),
            "second": InternalInputRef(producer="producer", label="second"),
        }
    )
    program = ResolvedCoreProgram(
        name="consumer",
        function=_internal_pair,
        arguments=MappingProxyType({}),
        static_kwargs=MappingProxyType({}),
        requirements=requirements,
        output_roles=("internal", "internal"),
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
        tile_widths=MappingProxyType({}),
        specialization_key=(),
        input_transfer_plan=(),
    )
    narrow = (("acting", 0, "consumer"), (("width", 1),))
    wide = (("acting", 0, "consumer"), (("width", 2),))
    duplicate = (("acting", 0, "other_consumer"), ())
    small = jax.ShapeDtypeStruct((4,), jnp.int32)
    large = jax.ShapeDtypeStruct((8,), jnp.int32)
    # Identical representative object under two labels sizes two future outputs.
    # Another consumer of those same labels adds no third allocation.
    templates: dict[
        backward_induction._CoreCandidate, Mapping[str, jax.ShapeDtypeStruct]
    ] = {
        narrow: {"first": small, "second": small},
        wide: {"first": large, "second": small},
        duplicate: {"first": large, "second": small},
    }
    actual = backward_induction._internal_reservations_by_cell(
        programs={narrow: program, wide: program, duplicate: program},
        templates=templates,
    )
    assert actual == {("acting", 0): (8 + 4) * 4}
    assert footprint.concrete_device_bytes(tree=tuple(templates.values())) == {}
