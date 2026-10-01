"""A restored NB-EGM result replays under a device-memory budget in a fresh model."""

from collections.abc import Callable
from functools import partialmethod
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import lcm.model as model_module
from _lcm.execution.execution_plan import visible_device_ids
from _lcm.persistence import solution as archive
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from lcm import ExecutionConfig, Model
from lcm._solver_api import authority as authority_module
from lcm._solver_api import entries
from lcm.exceptions import ExecutionPlanningError
from lcm.persistence import load_solution
from lcm.solver_api import SIMULATION_POLICY, LoadState, SolutionResult, ValueStore
from tests.simulation.test_foreign_result_allocation import _capture_owner
from tests.simulation.test_nnbegm_split_workflow_parity import (
    _INITIAL,
    _N_PERIODS,
    _PARAMS,
    _ROUTES,
    _SEED,
)
from tests.test_models import n_nbegm_toy as toy

_GENEROUS_BUDGET = 2**31
_POOL_LIMIT = 64_000_000_000
_BUDGETS = ["explicit", "device_default"]
# Labels `_copy_artifact_array_leaf` gives copies of authority templates.
_TEMPLATE_COPY_LABELS = ("Artifact template leaf", "Artifact cached leaf")
_REMEDIES = "To fit, lower `ExecutionConfig.device_memory_headroom_fraction`"


class _UnadmittedAllocationError(AssertionError):
    """A foreign payload reached a device allocator without budget admission."""


def _model(*, route: str, execution_config: ExecutionConfig) -> Model:
    return toy.build_model(
        variant="n_nbegm",
        n_periods=_N_PERIODS,
        outer_search=_ROUTES[route],
        execution_config=execution_config,
    )


def _budgeted_model(
    *, route: str, budget: str, monkeypatch: pytest.MonkeyPatch
) -> Model:
    """Return a model under an explicit budget or under a faked device default."""
    if budget == "device_default":
        # Report a preallocated pool on every visible device, as a GPU does.
        monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        monkeypatch.setattr(
            model_module,
            "visible_device_pool_limits",
            lambda: dict.fromkeys(visible_device_ids(), _POOL_LIMIT),
        )
        config = ExecutionConfig()
    else:
        config = ExecutionConfig(device_memory_bytes=_GENEROUS_BUDGET)
    model = _model(route=route, execution_config=config)
    assert model._execution.device_memory_bytes is not None
    return model


def _simulate(*, model: Model, solution: SolutionResult) -> pd.DataFrame:
    return model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=solution,
        log_level="off",
        seed=_SEED,
    ).to_dataframe(use_labels=False)


def _archive(*, route: str, directory: Path) -> Path:
    model = _model(
        route=route, execution_config=ExecutionConfig(device_memory_bytes=None)
    )
    solution = model.solve(params=_PARAMS, log_level="off")
    return solution.save(path=directory / "solution.lcm")


def _all_entries_unloaded(solution: SolutionResult) -> bool:
    values = cast("ValueStore", solution.values)
    return all(
        solution.replay_artifacts.load_state(ref) is LoadState.UNLOADED
        for ref in solution.replay_artifacts
    ) and all(
        values.load_state(period=period, regime=regime) is LoadState.UNLOADED
        for period in values
        for regime in values[period]
    )


@pytest.fixture(scope="module")
def finite_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Solve the finite outer-search model once, unbudgeted, and persist it."""
    return _archive(
        route="finite", directory=tmp_path_factory.mktemp("budgeted-foreign")
    )


@pytest.mark.parametrize("budget", _BUDGETS)
def test_restored_finite_nnbegm_result_replays_bitwise_under_a_budget(
    *, finite_archive: Path, budget: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh budgeted model replays a restored result exactly as without one."""
    expected = _simulate(
        model=_model(
            route="finite",
            execution_config=ExecutionConfig(device_memory_bytes=None),
        ),
        solution=load_solution(path=finite_archive),
    )
    model = _budgeted_model(route="finite", budget=budget, monkeypatch=monkeypatch)

    actual = _simulate(model=model, solution=load_solution(path=finite_archive))

    assert_frame_equal(actual, expected, check_exact=True)


def test_budgeted_restored_policies_reach_the_device_only_through_admission(
    *, finite_archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every archive upload and private copy of a restored result is admitted.

    Copies of the model's own authority templates are outside this check: they are
    validation declarations, not result payloads, and a budgeted solve makes the
    same unadmitted copies.
    """
    restored = load_solution(path=finite_archive)
    model = _budgeted_model(route="finite", budget="explicit", monkeypatch=monkeypatch)
    original_copy = authority_module._copy_artifact_array_leaf
    original_upload = archive._to_jax_without_narrowing

    def guarded_copy(*, leaf: object, label: str, **kwargs: Any) -> object:
        is_template = label.startswith(_TEMPLATE_COPY_LABELS)
        if kwargs.get("array_copier") is None and not is_template:
            raise _UnadmittedAllocationError(f"{label} was copied without admission")
        return original_copy(leaf=leaf, label=label, **kwargs)

    def guarded_upload(*, array: Any, label: str, **kwargs: Any) -> object:
        if kwargs.get("array_writer") is None:
            raise _UnadmittedAllocationError(f"{label} was uploaded without admission")
        return original_upload(array=array, label=label, **kwargs)

    for module in (authority_module, entries, archive):
        monkeypatch.setattr(module, "_copy_artifact_array_leaf", guarded_copy)
    monkeypatch.setattr(archive, "_to_jax_without_narrowing", guarded_upload)
    with pytest.raises(_UnadmittedAllocationError):
        archive._to_jax_without_narrowing(array=np.zeros(1), label="positive control")
    with pytest.raises(_UnadmittedAllocationError):
        authority_module._copy_artifact_array_leaf(leaf=None, label="positive control")

    _simulate(model=model, solution=restored)

    policy_refs = [
        ref for ref in restored.replay_artifacts if ref.key == SIMULATION_POLICY
    ]
    assert policy_refs
    assert all(
        restored.replay_artifacts.load_state(ref) is LoadState.LOADED
        for ref in policy_refs
    )


# keyword-only-exempt: library-callback=functools.partialmethod
def _starve_before_resolution(
    self: Model,
    *,
    original: Callable[..., object],
    owners: list[SimulationEntryAllocations],
    **arguments: Any,
) -> object:
    """Leave one byte of headroom above everything the entry already owns."""
    owner = owners[-1]
    live = owner.snapshot()
    owner.budget_bytes = 1 + max(
        sum(stop - start for start, stop in spans) for spans in live.spans.values()
    )
    return original(self, **arguments)


def test_a_too_small_budget_refuses_a_restored_result_before_any_copy(
    *, finite_archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget the restored payloads cannot fit refuses before decoding any of
    them, and names the remedies."""
    restored = load_solution(path=finite_archive)
    model = _budgeted_model(route="finite", budget="explicit", monkeypatch=monkeypatch)
    owners = _capture_owner(monkeypatch)
    copies: list[str] = []
    original_copy = SimulationEntryAllocations.copy_solution_leaf

    # keyword-only-exempt: library-callback=copy_solution_leaf
    def count_copy(self: SimulationEntryAllocations, *, leaf: Any, label: str) -> Any:
        copies.append(label)
        return original_copy(self, leaf=leaf, label=label)

    monkeypatch.setattr(SimulationEntryAllocations, "copy_solution_leaf", count_copy)
    monkeypatch.setattr(
        Model,
        "_resolve_solution_result",
        partialmethod(
            _starve_before_resolution,
            original=Model._resolve_solution_result,
            owners=owners,
        ),
    )

    with pytest.raises(ExecutionPlanningError) as refusal:
        _simulate(model=model, solution=restored)

    message = str(refusal.value)
    assert "Private copies of the supplied solution need at least" in message
    assert _REMEDIES in message
    assert copies == []
    assert _all_entries_unloaded(restored)


@pytest.fixture(scope="module")
def adaptive_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Solve the adaptive outer-search model once, unbudgeted, and persist it."""
    return _archive(
        route="adaptive", directory=tmp_path_factory.mktemp("budgeted-adaptive")
    )


# The adaptive solve alone approaches the per-test timeout on the macOS runner.
@pytest.mark.slow
@pytest.mark.parametrize("budget", _BUDGETS)
def test_restored_adaptive_result_is_refused_under_a_budget_before_any_copy(
    *, adaptive_archive: Path, budget: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adaptive replay runs through a host replay route no budget can profile, so a
    restored adaptive result is refused, with the remedies, before it is read."""
    restored = load_solution(path=adaptive_archive)
    model = _budgeted_model(route="adaptive", budget=budget, monkeypatch=monkeypatch)

    with pytest.raises(
        ExecutionPlanningError, match="host replay routes need complete stage profiles"
    ) as refusal:
        _simulate(model=model, solution=restored)

    assert _REMEDIES in str(refusal.value)
    assert _all_entries_unloaded(restored)
