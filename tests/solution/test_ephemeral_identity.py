"""Same-runtime model identity for expert-owned callable graphs."""

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
import lcm.model as model_module
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, PeriodCandidate
from lcm.exceptions import (
    IncompatibleSolutionError,
    InvalidSimulationInputError,
    ModelInitializationError,
    PyLCMError,
)
from lcm.persistence import save_solution
from lcm.solver_api import ResultRetention, SolutionSource
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
    np.testing.assert_allclose(
        solution.values[0]["working_life"],
        np.log([1.0, 2.0, 3.0]),
        rtol=0,
        atol=10 ** (-test_conftest.DECIMAL_PRECISION),
    )
    policy = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="working_life",
        states={"wealth": jnp.asarray([2.0])},
    )
    np.testing.assert_array_equal(policy.actions["consumption"], [2.0])
    np.testing.assert_array_equal(policy.actions["labor_supply"], [LaborSupply.retire])
    np.testing.assert_allclose(
        policy.value,
        [np.log(2.0)],
        rtol=0,
        atol=10 ** (-test_conftest.DECIMAL_PRECISION),
    )
    frame = result.to_dataframe(use_labels=False)
    assert frame["regime_name"].tolist() == ["working_life", "dead"]
    alive = frame.loc[frame["regime_name"] == "working_life"]
    np.testing.assert_array_equal(alive["wealth"], [2.0])
    np.testing.assert_array_equal(alive["consumption"], [2.0])
    np.testing.assert_array_equal(alive["labor_supply"], [LaborSupply.retire])


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


_RESTORED_MODEL_SCRIPT = """
import json
import os
import sys
from pathlib import Path

import cloudpickle
import jax
import lcm
import numpy as np
from lcm.exceptions import InvalidSimulationInputError
from tests.ci.probe_native import probe

native_probe = probe(root=Path.cwd())
assert native_probe.exit_code == 0, native_probe
model, params, initial, old_solution = cloudpickle.loads(Path(sys.argv[1]).read_bytes())
assert model._identity_process_id == os.getpid()
try:
    model.simulate(params=params, initial_conditions=initial,
                   solution=old_solution, log_level="off", seed=0)
except InvalidSimulationInputError as error:
    assert "ephemeral" in str(error)
else:
    raise AssertionError("A restored model must reject a foreign runtime's solution.")
fresh = model.solve(params=params, log_level="off")
assert fresh.metadata.model_instance_id != old_solution.metadata.model_instance_id
assert fresh.metadata.durable_identity is False
decimal_precision = int(sys.argv[2])
np.testing.assert_allclose(fresh.values[0]["working_life"], np.log([1., 2., 3.]),
                           rtol=0, atol=10 ** (-decimal_precision))
result = model.simulate(params=params, initial_conditions=initial,
                        solution=fresh, log_level="off", seed=0)
frame = result.to_dataframe(use_labels=False)
assert frame["regime_name"].tolist() == ["working_life", "dead"]
np.testing.assert_array_equal(frame.loc[frame["regime_name"] == "working_life",
                                        "consumption"], [2.])
print(json.dumps({"pid": os.getpid(), "lcm_file": lcm.__file__,
                  "x64": jax.config.read("jax_enable_x64"),
                  "native_probe": native_probe.status}, sort_keys=True))
"""


@pytest.mark.isolation(process="fresh")
def test_restored_ephemeral_model_solves_in_a_fresh_pixi_process(
    *, tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    """A fresh process restores local ownership and needs its own numerical solution."""
    model, params, initial_conditions = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off")
    payload = tmp_path / "runtime.pkl"
    payload.write_bytes(
        cloudpickle.dumps((model, params, initial_conditions, solution))
    )
    pixi = shutil.which("pixi")
    if pixi is None:
        raise RuntimeError("The fresh process requires the active Pixi executable.")
    completed = subprocess.run(  # noqa: S603
        [
            pixi,
            "run",
            "--as-is",
            "--manifest-path",
            os.environ["PIXI_PROJECT_MANIFEST"],
            "-e",
            os.environ["PIXI_ENVIRONMENT_NAME"],
            "python",
            "-c",
            _RESTORED_MODEL_SCRIPT,
            str(payload),
            str(test_conftest.DECIMAL_PRECISION),
        ],
        cwd=Path.cwd(),
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    receipt = json.loads(completed.stdout.splitlines()[-1])
    record_property("child_stdout", completed.stdout)
    record_property("child_stderr", completed.stderr)

    assert receipt["pid"] != os.getpid()
    assert receipt["lcm_file"] == lcm.__file__
    assert receipt["x64"] is jax.config.read("jax_enable_x64")
    assert receipt["native_probe"] == "ready"


@pytest.mark.parametrize(
    "operation",
    [
        "solve",
        "simulate",
        "lookup_policy",
        "lower_period_candidate",
        "validate_initial_conditions",
        "initial_conditions_feasibility",
    ],
)
def test_ephemeral_model_cannot_run_after_its_process_changes(
    *, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Public runtime operations report a common foreign-process identity error."""
    model, params, initial_conditions = _inputs(durable_identity=False)
    solution = model.solve(params=params, log_level="off")
    foreign_pid = model._identity_process_id + 1
    monkeypatch.setattr(model_module.os, "getpid", lambda: foreign_pid)
    invocations = {
        "solve": partial(model.solve, params=params, log_level="off"),
        "simulate": partial(
            model.simulate,
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
        ),
        "lookup_policy": partial(
            model.lookup_policy,
            params=params,
            solution=solution,
            period=0,
            regime_name="working_life",
            states={"wealth": jnp.asarray([2.0])},
        ),
        "lower_period_candidate": partial(
            model.lower_period_candidate,
            params=params,
            log_level="off",
            candidate=PeriodCandidate(
                regime="working_life", period=0, core="main", widths={"cell": 1}
            ),
        ),
        "validate_initial_conditions": partial(
            model.validate_initial_conditions,
            initial_conditions=initial_conditions,
            params=params,
        ),
        "initial_conditions_feasibility": partial(
            model.initial_conditions_feasibility,
            initial_conditions=initial_conditions,
            params=params,
        ),
    }

    with pytest.raises(PyLCMError, match="another process") as error:
        invocations[operation]()

    assert type(error.value).__name__ == "ModelIdentityError"


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
