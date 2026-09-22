"""Same-runtime model identity for expert-owned callable graphs."""

from dataclasses import replace
from pathlib import Path

import cloudpickle
import jax.numpy as jnp
import pytest

import lcm.model as model_module
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.exceptions import (
    IncompatibleSolutionError,
    InvalidSimulationInputError,
    ModelInitializationError,
)
from lcm.persistence import save_solution
from lcm.solver_api import ResultRetention, SolutionSource
from lcm.typing import UserInitialConditions, UserParams
from tests.test_models.deterministic.regression import (
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)
from tests.test_models.deterministic.regression import (
    get_model as get_durable_model,
)


def _inputs(
    *, durable_identity: bool
) -> tuple[Model, UserParams, UserInitialConditions]:
    token = object()

    def wage(age: float) -> float:
        _ = token
        return 1 + 0.1 * age

    model = Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= 18,
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                functions={**working_life.functions, "wage": wage},
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=18, stop=19, step="Y"),
        regime_id_class=RegimeId,
        durable_identity=durable_identity,
    )
    params = get_params(n_periods=2)
    initial_conditions = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([18.0]),
        "regime_id": jnp.asarray([RegimeId.working_life], dtype=jnp.int32),
    }
    return model, params, initial_conditions


def test_ephemeral_model_solves_and_simulates_an_opaque_callable_graph() -> None:
    """An expert model can use a callable with unsupported captured state locally."""
    with pytest.raises(ModelInitializationError, match="no durable identity"):
        _inputs(durable_identity=True)

    model, params, initial_conditions = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off")
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
        seed=0,
    )

    assert solution.metadata.durable_identity is False
    assert result.n_subjects == 1
    assert solution.metadata.source is SolutionSource.IN_MEMORY
    assert len(result.to_dataframe()) == 2


def test_ephemeral_solution_rejects_persistence_and_foreign_models(
    tmp_path: Path,
) -> None:
    """An ephemeral result belongs to its producing model and cannot be archived."""
    model, params, initial_conditions = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off")
    other, _, _ = _inputs(durable_identity=False)

    with pytest.raises(IncompatibleSolutionError, match="ephemeral"):
        save_solution(solution=solution, path=tmp_path / "solution.lcm")
    with pytest.raises(IncompatibleSolutionError, match="ephemeral"):
        solution.save(path=tmp_path / "solution.lcm")
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
        seed=0,
    )
    with pytest.raises(IncompatibleSolutionError, match="ephemeral"):
        result.save(directory=tmp_path / "simulation")
    assert not (tmp_path / "solution.lcm").exists()
    with pytest.raises(InvalidSimulationInputError, match="ephemeral"):
        other.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
            seed=0,
        )


def test_ephemeral_result_rejects_parameter_changes_and_persisted_source() -> None:
    """The local mode still checks parameter values and result provenance."""
    model, params, initial_conditions = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off")
    changed = get_params(n_periods=2, disutility_of_work=0.7)

    with pytest.raises(InvalidSimulationInputError, match="params_fingerprint"):
        model.simulate(
            params=changed,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
            seed=0,
        )
    forged = replace(
        solution,
        metadata=replace(solution.metadata, source=SolutionSource.PERSISTED),
    )
    with pytest.raises(InvalidSimulationInputError, match="ephemeral"):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=forged,
            log_level="off",
            seed=0,
        )


def test_ephemeral_model_restoration_starts_a_new_runtime_identity() -> None:
    """A restored model makes fresh local results and rejects old ones."""
    model, params, initial_conditions = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off")
    restored = cloudpickle.loads(cloudpickle.dumps(model))

    with pytest.raises(InvalidSimulationInputError, match="ephemeral"):
        restored.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
            seed=0,
        )
    fresh = restored.solve(params=params, log_level="off")
    assert fresh.metadata.model_instance_id != solution.metadata.model_instance_id
    assert fresh.metadata.durable_identity is False


def test_ephemeral_model_cannot_run_after_its_process_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copied model cannot use its local identity in another process."""
    model, params, _ = _inputs(durable_identity=False)
    foreign_pid = model._identity_process_id + 1
    monkeypatch.setattr(model_module.os, "getpid", lambda: foreign_pid)

    with pytest.raises(InvalidSimulationInputError, match="another process"):
        model.solve(params=params, log_level="off")


def test_restored_legacy_model_defaults_to_durable_identity() -> None:
    """A model serialized before the option existed keeps durable replay."""
    model = get_durable_model(n_periods=2)
    del model.durable_identity

    restored = cloudpickle.loads(cloudpickle.dumps(model))

    assert restored.durable_identity is True


@pytest.mark.parametrize("retention", list(ResultRetention))
def test_ephemeral_solutions_remain_marked_at_every_retention(
    retention: ResultRetention,
) -> None:
    """Retaining replay artifacts does not grant a durable model identity."""
    model, params, _ = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off", retention=retention)
    assert solution.metadata.durable_identity is False
