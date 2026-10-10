"""Local solve and simulation for models without durable callable fingerprints."""

import os
from pathlib import Path

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pytest

import lcm.model as model_module
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.component_jobs import plan_component_jobs
from lcm.exceptions import (
    ExecutionPlanningError,
    IncompatibleSolutionError,
    InvalidSimulationInputError,
    ModelInitializationError,
    UnsupportedOperationError,
)
from lcm.persistence import PeriodCapture
from lcm.typing import UserInitialConditions, UserParams
from tests import conftest as test_conftest
from tests.test_models.deterministic.regression import (
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
    working_life_edges,
)


def _inputs(
    *, durable_identity: bool
) -> tuple[Model, UserParams, UserInitialConditions]:
    token = object()

    def wage(age: float) -> float:
        _ = token
        return 1 + 0.1 * age

    ages = AgeGrid(start=18, inclusive_stop=19, step="Y")
    model = Model(
        regimes={
            "working_life": working_life.replace(
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                functions={**working_life.functions, "wage": wage},
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
        durable_identity=durable_identity,
        initial_nodes={18: "working_life"},
        edges=working_life_edges(ages),
    )
    params = get_params(n_periods=2)
    initial_conditions = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([18.0]),
        "regime_id": jnp.asarray([RegimeId.working_life], dtype=jnp.int32),
    }
    return model, params, initial_conditions


@pytest.mark.parametrize("provide_solution", [False, True])
def test_ephemeral_model_solves_and_simulates_an_opaque_callable_graph(
    *, provide_solution: bool
) -> None:
    """Local execution returns the expected values and choices for opaque callables."""
    with pytest.raises(ModelInitializationError, match="no durable identity"):
        _inputs(durable_identity=True)

    model, params, initial_conditions = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off")
    np.testing.assert_allclose(
        solution.values[0]["working_life"],
        np.log([1.0, 2.0, 3.0]),
        rtol=0,
        atol=10 ** (-test_conftest.DECIMAL_PRECISION),
    )
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution if provide_solution else None,
        log_level="off",
        seed=0,
    )
    frame = result.to_dataframe(use_labels=False)
    assert frame["regime_name"].tolist() == ["working_life", "dead"]
    alive = frame.loc[frame["regime_name"] == "working_life"]
    np.testing.assert_array_equal(alive["wealth"], [2.0])
    np.testing.assert_array_equal(alive["consumption"], [2.0])
    np.testing.assert_array_equal(alive["labor_supply"], [LaborSupply.retire])


def test_ephemeral_solution_rejects_foreign_models_and_changed_parameters() -> None:
    """A local solution requires its producing model and solve parameters."""
    model, params, initial_conditions = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off")
    other, _, _ = _inputs(durable_identity=False)
    with pytest.raises(InvalidSimulationInputError, match="ephemeral"):
        other.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
            seed=0,
        )
    with pytest.raises(InvalidSimulationInputError, match="params_fingerprint"):
        model.simulate(
            params=get_params(n_periods=2, disutility_of_work=0.7),
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
            seed=0,
        )


def test_ephemeral_results_cannot_be_saved(tmp_path: Path) -> None:
    """Local solutions and simulation archives cannot be saved for later reuse."""
    model, params, initial_conditions = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off")
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
    assert list(tmp_path.iterdir()) == []


def test_ephemeral_model_requires_its_originating_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local model reports unsupported use after crossing a process boundary."""
    model, params, initial_conditions = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off")
    foreign_pid = os.getpid() + 1
    monkeypatch.setattr(model_module.os, "getpid", lambda: foreign_pid)
    with pytest.raises(UnsupportedOperationError, match="another process"):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
            seed=0,
        )


def test_restored_ephemeral_model_requires_a_fresh_solution() -> None:
    """Restoring a local model requires solving it again before simulation."""
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
    np.testing.assert_allclose(
        fresh.values[0]["working_life"],
        np.log([1.0, 2.0, 3.0]),
        rtol=0,
        atol=10 ** (-test_conftest.DECIMAL_PRECISION),
    )


def test_ephemeral_model_cannot_publish_durable_archives(tmp_path: Path) -> None:
    """Period captures and component campaigns require durable model identity."""
    model, params, _ = _inputs(durable_identity=False)
    capture = PeriodCapture(
        directory=tmp_path / "captures",
        periods=(("working_life", 0),),
        source_identity={"model": "local-model"},
    )
    with pytest.raises(UnsupportedOperationError, match="ephemeral"):
        model.solve(params=params, log_level="off", period_capture=capture)
    with pytest.raises(ExecutionPlanningError, match="ephemeral"):
        plan_component_jobs(
            model=model, params=params, directory=tmp_path / "jobs", n_jobs=1
        )
    assert list(tmp_path.iterdir()) == []
