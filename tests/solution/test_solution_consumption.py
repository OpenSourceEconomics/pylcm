"""How a model consumes a solution it built itself versus one from elsewhere.

A result the engine built in this process is consumed by reference: the model
reads the buffers it allocated. A result from anywhere else — restored from an
archive, built by a plugin, produced by another instance — is copied into
private buffers and validated exactly once, and later consumption reuses that
validated view.
"""

import gc
import weakref
from pathlib import Path

import jax.numpy as jnp
import pytest
from pandas.testing import assert_frame_equal

import lcm.model as model_module
import lcm.solver_api as solver_api_module
from _lcm.persistence import solution as solution_persistence
from _lcm.solution import fingerprint as fingerprint_module
from _lcm.solution import model_authority as model_authority_module
from lcm.persistence import load_solution
from tests.simulation.test_nnbegm_split_workflow_parity import (
    _INITIAL,
    _PARAMS,
    _build,
)
from tests.solution.test_solution_result import _small_grid_search_inputs
from tests.test_models.deterministic.regression import get_params


class _Counter:
    """Count calls to one wrapped function while delegating to it."""

    def __init__(self, target: object) -> None:
        self.calls = 0
        self._target = target

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        return self._target(*args, **kwargs)  # ty: ignore[call-non-callable]


def _count_leaf_copies(monkeypatch: pytest.MonkeyPatch) -> _Counter:
    counter = _Counter(solver_api_module._copy_artifact_array_leaf)
    monkeypatch.setattr(solver_api_module, "_copy_artifact_array_leaf", counter)
    return counter


def test_public_value_reads_still_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A value handed out at the public boundary is an independent buffer."""
    model, params, _initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")
    copies = _count_leaf_copies(monkeypatch)

    first = solution.values[0]["working_life"]
    second = solution.values[0]["working_life"]

    assert copies.calls == 2
    assert first is not second


def test_simulate_with_the_owning_model_copies_no_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine consumes its own result by reference."""
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")
    copies = _count_leaf_copies(monkeypatch)

    model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
        seed=0,
    )

    assert copies.calls == 0


def test_repeated_solves_copy_no_buffers_after_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Labelling a solve's outputs borrows them; only the first solve observes
    the model-owned artifact templates."""
    model, params, _initial_conditions = _small_grid_search_inputs()
    model.solve(params=params, log_level="off")
    copies = _count_leaf_copies(monkeypatch)

    model.solve(params=get_params(n_periods=2, discount_factor=0.9), log_level="off")

    assert copies.calls == 0


def test_declared_authority_is_built_once_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model-owned part of a solution's authority is shared by every solve
    whose grids agree, whatever the other parameters are."""
    model, params, _initial_conditions = _small_grid_search_inputs()
    builds = _Counter(model_authority_module.build_solution_authority)
    monkeypatch.setattr(model_module, "build_solution_authority", builds)
    structure_walks = _Counter(fingerprint_module.fingerprint_model_structure)
    monkeypatch.setattr(model_module, "fingerprint_model_structure", structure_walks)

    model.solve(params=params, log_level="off")
    model.solve(params=get_params(n_periods=2, discount_factor=0.9), log_level="off")

    assert builds.calls == 1
    assert structure_walks.calls == 0


def test_second_simulate_on_a_restored_solution_is_a_lookup(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A foreign result is validated and materialized once; the next consumer
    of the same object reads neither the archive nor a copy."""
    model, params, initial_conditions = _small_grid_search_inputs()
    path = tmp_path / "solution.lcm"
    model.solve(params=params, log_level="off").save(path=path)
    restored = load_solution(path=path)
    first = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=restored,
        log_level="off",
        seed=0,
    )
    reads = _Counter(solution_persistence._read_and_verify_leaves)
    monkeypatch.setattr(solution_persistence, "_read_and_verify_leaves", reads)
    copies = _count_leaf_copies(monkeypatch)

    second = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=restored,
        log_level="off",
        seed=0,
    )

    assert reads.calls == 0
    assert copies.calls == 0
    assert_frame_equal(first.to_dataframe(), second.to_dataframe())


def test_restored_solution_replays_like_the_owned_one(tmp_path: Path) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    path = tmp_path / "solution.lcm"
    owned = model.solve(params=params, log_level="off")
    owned.save(path=path)
    restored = load_solution(path=path)

    from_owned = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=owned,
        log_level="off",
        seed=0,
    )
    from_restored = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=restored,
        log_level="off",
        seed=0,
    )

    assert_frame_equal(from_owned.to_dataframe(), from_restored.to_dataframe())


def test_owned_solution_with_other_params_is_refused_before_forward() -> None:
    """Ownership does not bypass the parameter check."""
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")

    with pytest.raises(
        model_module.InvalidSimulationInputError,
        match="params_fingerprint does not match",
    ):
        model.simulate(
            params=get_params(n_periods=2, discount_factor=0.9),
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
        )


def test_dropping_an_adaptive_solution_releases_its_generated_authority() -> None:
    """Solve-generated replay authority lives and dies with its result, not
    with the model that produced it."""
    model = _build("adaptive")
    solution = model.solve(params=_PARAMS, log_level="off")
    generated = weakref.ref(solution._engine_view)
    assert generated() is not None
    assert not hasattr(model, "_solution_authorities")

    del solution
    gc.collect()

    assert generated() is None


def test_adaptive_solution_replays_after_the_model_forgets_it() -> None:
    """Replaying an adaptive solution needs nothing the model holds beside it."""
    model = _build("adaptive")
    solution = model.solve(params=_PARAMS, log_level="off")
    model.solve(params=_PARAMS, log_level="off")
    gc.collect()

    direct = model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=solution,
        log_level="off",
        seed=42,
    )
    automatic = model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        log_level="off",
        seed=42,
    )

    assert_frame_equal(direct.to_dataframe(), automatic.to_dataframe())


def test_a_copied_result_loses_ownership_and_is_validated() -> None:
    """`dataclasses.replace` yields a result the model no longer recognizes as
    its own; it takes the validated path and still replays identically."""
    from dataclasses import replace  # noqa: PLC0415

    model, params, initial_conditions = _small_grid_search_inputs()
    owned = model.solve(params=params, log_level="off")
    copied = replace(owned)
    assert copied._engine_view is None

    from_owned = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=owned,
        log_level="off",
        seed=0,
    )
    from_copy = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=copied,
        log_level="off",
        seed=0,
    )

    assert_frame_equal(from_owned.to_dataframe(), from_copy.to_dataframe())
    assert jnp.array_equal(
        owned.values[0]["working_life"], copied.values[0]["working_life"]
    )
