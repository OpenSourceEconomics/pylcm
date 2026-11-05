"""Size household query frontiers symbolically, without a native query solve.

The local dimensions and node product are transcribed from lcm-zoo commit
`ad2658dae4df00aaf53934f9011adc4c60dcc959`,
`src/lcm_zoo/ecksteinCareerFamilyDecisions2019/model.py` and
`tasks/ecksteinCareerFamilyDecisions2019/{model,values}.py`:
https://github.com/OpenSourceEconomics/lcm-zoo/tree/ad2658dae4df00aaf53934f9011adc4c60dcc959

The other geometry, pool limit, headroom and chunk ceiling are hypothetical stress
inputs. Neither geometry certifies original-paper enumeration or native compiler fit.
The modeled owner set is source/target shock-free values, query value/action/mask
channels, simultaneous source/target scratch, and one integrated chunk output.
Compiler records are synthetic; unreported runtime storage remains outside this test.
"""

import dataclasses
import math
from collections.abc import Hashable, Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Literal

import jax
import numpy as np
import pytest

from _lcm.execution.core_program import TiledOutputAxis
from _lcm.execution.execution_plan import resolve_execution_config
from _lcm.execution.footprint import (
    ArtifactFootprint,
    ResidentInventory,
    ScheduledUnit,
    layout_footprint,
    plan_resident_inventory,
)
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.workspace_planning import WorkspacePlan, plan_workspace
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.typing import RegimeName
from tests.execution.test_compiler_allocation_reservation import memory_stats

_LOCAL_DIMENSIONS = MappingProxyType(
    {
        "education_f": 5,
        "education_m": 5,
        "experience_f": 5,
        "experience_m": 5,
        "children": 4,
        "job_f": 4,
        "job_m": 4,
        "skill_f": 3,
        "skill_m": 3,
        "lagged_pregnancy": 2,
        "health_f": 3,
        "health_m": 3,
        "leisure_f": 3,
        "leisure_m": 3,
        "parent_f": 2,
        "parent_m": 2,
    }
)
_NODES = math.prod((3, 3, 3, 2))
_CHUNK_CEILING = 4096
_HYPOTHETICAL_POOL_BYTES = 80 * 2**30
_CHUNK_TOTALS = MappingProxyType(
    {
        (233_280_000, 4): 3_867_213_824,
        (233_280_000, 8): 7_712_972_800,
        (700_000_000, 4): 11_334_733_824,
        (700_000_000, 8): 22_648_012_800,
    }
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _QueryBuffer:
    root: int
    """Full-query context; different mapped views cannot alias."""
    view: int
    """Raw views precede the participating couple view."""
    channel: Literal["value", "action", "mask"]
    """Separately owned value tuple, stable action identity or support mask."""


type _Artifact = str | _QueryBuffer


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Candidate:
    inventories: tuple[ResidentInventory, ...]
    """Every Full-root dispatch's modeled live-owner inventory."""
    temporary_bytes: int
    """Simultaneous source/target scratch, excluding external query owners."""
    output_bytes: int
    """One shock-integrated chunk result."""

    def memory_analysis(self) -> SimpleNamespace:
        return memory_stats(
            peak=0,
            temp_size_in_bytes=self.temporary_bytes,
            output_size_in_bytes=self.output_bytes,
        )

    def __call__(self) -> None:
        raise AssertionError("Symbolic sizing must never execute a candidate")

    def resident_bytes(self) -> int:
        # Conservatively charge all inputs, as if the compiler pruned their reads.
        return max(item.resident_bytes(consumes=()) for item in self.inventories)


@pytest.mark.parametrize("cells", [233_280_000, 700_000_000], ids=["local", "stress"])
@pytest.mark.parametrize("value_bytes", [4, 8], ids=["float32", "float64"])
def test_full_geometry_retention_refuses_but_chunk_recomputation_fits(
    *, cells: int, value_bytes: int
) -> None:
    """The same complete modeled owners refuse retention and admit recomputation."""
    with pytest.raises(ExecutionPlanningError, match="resident bytes"):
        _plan(cells=cells, value_bytes=value_bytes, recompute=False)
    plan = _plan(cells=cells, value_bytes=value_bytes, recompute=True)
    assert plan.reservation_bytes is not None
    assert (
        dict(plan.widths),
        plan.compiled.resident_bytes() + plan.reservation_bytes,
    ) == (
        {"cell": _CHUNK_CEILING},
        _CHUNK_TOTALS[(cells, value_bytes)],
    )


def test_declared_local_geometry_and_effective_budget_are_explicit() -> None:
    """Headroom reduces the hypothetical request before any width is admitted."""
    assert (math.prod(_LOCAL_DIMENSIONS.values()), _NODES, _budget()) == (
        233_280_000,
        54,
        68_719_476_736,
    )


def test_exhausted_modeled_owners_refuse_before_synthetic_compilation() -> None:
    """A width cannot recover storage already exhausted by fixed owners."""

    def forbidden_compiler(_widths: Mapping[str, int]) -> _Candidate:
        pytest.fail("Exhausted owners must refuse before compilation")

    with pytest.raises(ExecutionPlanningError, match="bytes resident"):
        plan_workspace(
            axes=(_axis(cells=700_000_000),),
            compile_candidate=forbidden_compiler,
            budget_bytes=_budget(),
            resident_bytes=_budget(),
        )


def test_no_legal_chunk_refuses_at_the_exact_modeled_boundary() -> None:
    """Even the narrowest tile must cover its query channels and scratch."""
    minimum = 22_400_060_550
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        _plan(
            cells=700_000_000,
            value_bytes=8,
            recompute=True,
            ceiling=1,
            budget=minimum - 1,
        )
    plan = _plan(
        cells=700_000_000, value_bytes=8, recompute=True, ceiling=1, budget=minimum
    )
    assert dict(plan.widths) == {"cell": 1}


def test_effective_budget_narrows_the_query_chunk() -> None:
    """Admission changes the tile while preserving the full geometry and nodes."""
    plan = _plan(
        cells=700_000_000,
        value_bytes=8,
        recompute=True,
        budget=22_524_006_400,
    )
    assert dict(plan.widths) == {"cell": 2048}


def test_full_geometry_query_owners_close_only_after_their_registered_reader() -> None:
    """Skipping the exact Full reader prevents release and successful completion."""
    _, ledger = _inventories(cells=233_280_000, query_cells=4096, value_bytes=8)
    first = _QueryBuffer(root=0, view=0, channel="value")
    ledger.commit_successful_dispatch(dispatch=(0, "owners"))
    ledger.commit_successful_dispatch(dispatch=(0, "queries_0"))
    assert not ledger.is_release_eligible(artifact=first)
    with pytest.raises(RuntimeError, match="uncommitted"):
        ledger.assert_solve_complete()
    released = ledger.commit_successful_dispatch(dispatch=(0, "full_0"))
    assert first in released
    assert not ledger.is_release_eligible(artifact="source")


def test_synthetic_candidate_cannot_execute_full_geometry() -> None:
    """The sizing executable is a guard rather than a numerical model."""
    plan = _plan(cells=700_000_000, value_bytes=8, recompute=True)
    with pytest.raises(AssertionError, match="never execute"):
        plan.compiled()


def _plan(
    *,
    cells: int,
    value_bytes: int,
    recompute: bool,
    ceiling: int = _CHUNK_CEILING,
    budget: int | None = None,
) -> WorkspacePlan[_Candidate]:
    def compile_candidate(widths: Mapping[str, int]) -> _Candidate:
        width = widths["cell"]
        inventories, _ = _inventories(
            cells=cells,
            query_cells=width if recompute else cells,
            value_bytes=value_bytes,
            retain_all=not recompute,
        )
        # Both sides use the coupled action/stakeholder maxima conservatively.
        scratch = 2 * (
            18 * 2 * value_bytes
            + 18 * value_bytes
            + 2 * 18
            + 2 * value_bytes
            + value_bytes
            + 4 * value_bytes
        )
        return _Candidate(
            inventories=inventories,
            temporary_bytes=width * _NODES * scratch,
            output_bytes=width * 2 * value_bytes,
        )

    return plan_workspace(
        axes=(_axis(cells=cells),),
        width_ceilings={"cell": ceiling},
        compile_candidate=compile_candidate,
        budget_bytes=_budget(
            request=_HYPOTHETICAL_POOL_BYTES if budget is None else budget
        ),
        resident_bytes=2 * cells * 2 * value_bytes,
        resident_bytes_for=lambda candidate: candidate.resident_bytes(),
    )


def _inventories(
    *, cells: int, query_cells: int, value_bytes: int, retain_all: bool = False
) -> tuple[
    tuple[ResidentInventory, ...],
    PlannedInputLiveness[tuple[int, RegimeName], _Artifact],
]:
    device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)

    def footprint(*, shape: tuple[int, ...], item_bytes: int) -> ArtifactFootprint:
        return layout_footprint(sharding=sharding, shape=shape, item_bytes=item_bytes)

    owners = ("source", "target")
    footprints: dict[Hashable, ArtifactFootprint] = {
        name: footprint(shape=(cells, 2), item_bytes=value_bytes) for name in owners
    }
    queries: dict[int, tuple[_QueryBuffer, ...]] = {}
    for root in range(3):
        members = []
        for view in range(5):
            for channel in ("value", "action", "mask"):
                key = _QueryBuffer(root=root, view=view, channel=channel)
                members.append(key)
                stakeholders = 2 if view == 4 and channel == "value" else 1
                footprints[key] = footprint(
                    shape=(query_cells, _NODES, stakeholders),
                    item_bytes=value_bytes
                    if channel == "value"
                    else (
                        np.dtype(np.int32).itemsize
                        if channel == "action"
                        else np.dtype(np.bool_).itemsize
                    ),
                )
        queries[root] = tuple(members)

    accesses: dict[tuple[int, RegimeName], tuple[_Artifact, ...]] = {}
    waves = []

    def unit(
        *,
        regime: RegimeName,
        produces: tuple[_Artifact, ...],
        consumes: tuple[_Artifact, ...] = (),
    ) -> None:
        accesses[(0, regime)] = consumes
        waves.append(
            (
                ScheduledUnit(
                    period=0,
                    regime=regime,
                    device_ids=(device.id,),
                    produces=produces,
                    consumes=consumes,
                    output_bytes_per_device=sum(
                        footprints[key].bytes_per_device for key in produces
                    ),
                ),
            )
        )

    unit(regime="owners", produces=owners)
    if retain_all:
        unit(
            regime="all_queries",
            produces=tuple(key for keys in queries.values() for key in keys),
        )
    for root, keys in queries.items():
        if not retain_all:
            unit(regime=f"queries_{root}", produces=keys)
        unit(regime=f"full_{root}", produces=(), consumes=owners + keys)
    ledger = PlannedInputLiveness(dispatch_accesses=accesses, pinned_artifacts=owners)
    inventory = plan_resident_inventory(
        waves_by_period={0: tuple(waves)},
        fold_dispatches={},
        ledger=ledger,
        footprints=footprints,
    )
    return tuple(inventory[(0, f"full_{root}")] for root in range(3)), ledger


def _axis(*, cells: int) -> TiledOutputAxis:
    return TiledOutputAxis(
        name="cell",
        state_names=("household",),
        extent=cells,
        width_keyword="cell_width",
    )


def _budget(*, request: int = _HYPOTHETICAL_POOL_BYTES) -> int:
    resolved = resolve_execution_config(
        config=ExecutionConfig(
            device_memory_bytes=request,
            device_memory_headroom_fraction=0.2,
        ),
        visible_device_ids=(jax.devices()[0].id,),
        state_names=frozenset(),
        device_pool_limit_bytes={jax.devices()[0].id: _HYPOTHETICAL_POOL_BYTES},
    )
    assert resolved.device_memory_bytes is not None
    return resolved.device_memory_bytes
