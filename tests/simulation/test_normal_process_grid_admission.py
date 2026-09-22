"""Normal support preserves eager coordinates while admitting each producer."""

import inspect
import sys
from collections.abc import Callable
from dataclasses import replace
from functools import partial, partialmethod
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation import host_operations, process_grids
from _lcm.simulation.residency import measure_buffer_footprint, resident_bytes_by_device
from _lcm.utils.logging import LogLevel
from lcm import AgeGrid, ExecutionConfig, LinSpacedGrid, Model, NormalIIDProcess, Regime
from lcm.exceptions import ExecutionPlanningError
from lcm.persistence import load_solution
from lcm.typing import UserInitialConditions, UserParams, ValueND
from tests.execution.test_compiler_allocation_reservation import synthetic_memory
from tests.simulation.test_budget_lifecycle import _LifecycleRegimeId
from tests.simulation.test_process_grid_entry_admission import (
    _forbid_profiled_dispatch,
    _initial_age,
    _next_regime,
    _terminal_utility,
    _utility,
)


# keyword-only-exempt: library-callback=functools.partialmethod
def _inject_support_reservation(
    self: host_operations.ProfiledSimulationOperations,
    *,
    original: Callable[..., Any],
    profiled: list[jax.stages.Compiled],
    stages: list[str],
    refuse_at: int,
    **kwargs: Any,
) -> Any:
    result = original(self, **kwargs)
    if kwargs["function"].__module__ == process_grids.__name__:
        stages.append(kwargs["static_arguments"].get("stage", "uniform"))
        if len(stages) == refuse_at:
            profiled.append(result.executable)
            return replace(result, memory=synthetic_memory(2**30))
    return result


def _inputs(
    *, budget: int | None, fixed: tuple[str, ...] = (), n_points: int = 5
) -> tuple[Model, UserParams, UserInitialConditions]:
    parameters = {"mu": 0.1415, "sigma": 1.876, "n_std": 3.2}
    model = Model(
        regimes={
            "alive": Regime(
                transition=_next_regime,
                active=_initial_age,
                states={
                    "income": NormalIIDProcess(
                        n_points=n_points,
                        gauss_hermite=False,
                        mu=parameters["mu"] if "mu" in fixed else None,
                        sigma=parameters["sigma"] if "sigma" in fixed else None,
                        n_std=parameters["n_std"] if "n_std" in fixed else None,
                    )
                },
                actions={"saving": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _utility},
            ),
            "done": Regime(transition=None, functions={"utility": _terminal_utility}),
        },
        regime_id_class=_LifecycleRegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        execution_config=ExecutionConfig(device_memory_bytes=budget),
    )
    return (
        model,
        {
            "alive": {
                "income": {
                    name: value
                    for name, value in parameters.items()
                    if name not in fixed
                },
                "koopmans_aggregator": {"discount_factor": 0.9},
            },
            "done": {},
        },
        {
            "income": jnp.asarray([2.0]),
            "age": jnp.asarray([0.0]),
            "regime_id": jnp.asarray([_LifecycleRegimeId.alive]),
        },
    )


def _guard_normal_linspace(
    *args: Any, original: Callable[..., Any], attempts: list[int], **kwargs: Any
) -> Any:
    if (
        sys._getframe(1).f_code
        is inspect.unwrap(NormalIIDProcess.compute_gridpoints).__code__
    ):
        attempts.append(kwargs["num"])
        raise AssertionError("Normal support allocated before admission")
    return original(*args, **kwargs)


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("refuse_at", [1, 2, 3, 4, 5])
def test_normal_support_refuses_low_budget_before_grid_dispatch(
    *, monkeypatch: pytest.MonkeyPatch, supplied: bool, refuse_at: int
) -> None:
    """A support with a reported excessive reservation refuses before allocation."""
    producer, params, initial = _inputs(budget=None)
    solution = producer.solve(params=params, log_level="off") if supplied else None
    consumer, _, _ = _inputs(budget=2**20)
    attempts: list[int] = []
    profiled: list[jax.stages.Compiled] = []
    stages: list[str] = []
    monkeypatch.setattr(
        jnp,
        "linspace",
        partial(_guard_normal_linspace, original=jnp.linspace, attempts=attempts),
    )
    monkeypatch.setattr(
        host_operations.ProfiledSimulationOperations,
        "compile_candidate",
        partialmethod(
            _inject_support_reservation,
            original=host_operations.ProfiledSimulationOperations.compile_candidate,
            profiled=profiled,
            stages=stages,
            refuse_at=refuse_at,
        ),
    )
    monkeypatch.setattr(
        jax.stages.Compiled,
        "__call__",
        partialmethod(
            _forbid_profiled_dispatch,
            profiled=profiled,
            original=jax.stages.Compiled.__call__,
        ),
    )
    with pytest.raises(ExecutionPlanningError):
        consumer.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="off",
        )
    assert not attempts
    assert len(profiled) == 1
    assert stages == ["multiply", "subtract", "multiply", "add", "normal"][:refuse_at]
    with pytest.raises(AssertionError, match="over-budget process executable"):
        profiled[0]()


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("fixed", [(), ("mu",), ("n_std",), ("sigma", "n_std")])
@pytest.mark.parametrize("log_level", ["off", "debug"])
def test_normal_support_preserves_saved_identity_value_and_policy(
    *, tmp_path: Path, supplied: bool, fixed: tuple[str, ...], log_level: LogLevel
) -> None:
    """An eager archive and automatic solve yield saving one and value three."""
    producer, params, initial = _inputs(budget=None, fixed=fixed)
    solution = producer.solve(params=params, log_level="off")
    expected_values = np.asarray(solution.values[0]["alive"]).copy()
    restored = load_solution(path=solution.save(path=tmp_path / "normal"))
    consumer, _, _ = _inputs(budget=2**28, fixed=fixed)
    result = consumer.simulate(
        params=params,
        initial_conditions=initial,
        solution=restored if supplied else None,
        log_level=log_level,
    )
    np.testing.assert_array_equal(
        result.period_to_regime_to_V_arr[0]["alive"], expected_values
    )
    np.testing.assert_array_equal(
        result.to_dataframe(use_labels=False)
        .iloc[0][["income", "saving", "value"]]
        .to_numpy(dtype=float),
        [2.0, 1.0, 3.0],
    )


def _owner(*, parameters: object) -> process_grids.SimulationProcessGrids:
    return process_grids.SimulationProcessGrids(
        live_footprint=lambda: measure_buffer_footprint(tree=parameters),
        devices=(jax.devices()[0],),
        budget_bytes=2**28,
        operations=host_operations.ProfiledSimulationOperations(),
    )


@pytest.mark.parametrize("n_points", [2, 5, 17])
def test_normal_support_matches_eager_bytes_across_generated_parameters(
    n_points: int,
) -> None:
    """Separate scalar rounding reproduces eager support over deterministic scales."""
    rng = np.random.default_rng(7401)
    parameters = [
        {
            "mu": jnp.asarray(mu),
            "sigma": jnp.asarray(sigma),
            "n_std": jnp.asarray(n_std),
        }
        for mu, sigma, n_std in zip(
            rng.uniform(-10, 10, 6),
            np.exp(rng.uniform(-8, 4, 6)),
            rng.uniform(1, 4, 6),
            strict=True,
        )
    ]
    spec = NormalIIDProcess(n_points=n_points, gauss_hermite=False)
    owner = _owner(parameters=parameters)
    required = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    for operands in parameters:
        eager = np.asarray(spec.compute_gridpoints(**operands))
        actual = np.asarray(owner(spec=spec, parameters=operands, required=required))
        assert np.isfinite(eager).all()
        assert actual.dtype == eager.dtype
        assert actual.tobytes() == eager.tobytes()
    assert len(owner.grids) == 6


@pytest.mark.parametrize("fixed", [3, 3.0])
def test_normal_fixed_scale_preserves_weak_dtype_promotion(fixed: float) -> None:
    """A fixed weak float multiplies a strong float32 scale at its eager precision."""
    parameters = {
        "mu": jnp.asarray(0.0, dtype=jnp.float32),
        "sigma": jnp.asarray(0.1, dtype=jnp.float32),
    }
    spec = NormalIIDProcess(n_points=5, gauss_hermite=False, n_std=fixed)
    eager = np.asarray(spec.compute_gridpoints(**spec.params, **parameters))
    actual = np.asarray(
        _owner(parameters=parameters)(
            spec=spec,
            parameters=parameters,
            required=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        )
    )
    assert eager.dtype == np.dtype("float32")
    assert actual.dtype == eager.dtype
    assert actual.tobytes() == eager.tobytes()


def test_normal_equal_fixed_scalars_keep_distinct_binding_provenance() -> None:
    """Fixed scalar type and signed-zero bytes survive the fast binding lookup."""
    parameters = {"sigma": jnp.asarray(0.1)}
    owner = _owner(parameters=parameters)
    required = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    for mu, n_std in ((0.0, 3), (0.0, 3.0), (-0.0, 3.0)):
        spec = NormalIIDProcess(n_points=5, gauss_hermite=False, mu=mu, n_std=n_std)
        actual = np.asarray(owner(spec=spec, parameters=parameters, required=required))
        eager = np.asarray(spec.compute_gridpoints(**spec.params, **parameters))
        assert actual.tobytes() == eager.tobytes()
    assert len(owner.bindings) == 3
    assert len(owner.grids) == 3
    mixed = {
        "mu": jnp.asarray(0.0, dtype=jnp.float32),
        "sigma": jnp.asarray(0.1, dtype=jnp.float32),
    }
    owner = _owner(parameters=mixed)
    weak = jax.device_put(3.0)
    strong = jnp.asarray(3.0, dtype=weak.dtype)
    assert weak.weak_type
    assert not strong.weak_type
    assert np.asarray(weak).tobytes() == np.asarray(strong).tobytes()
    spec = NormalIIDProcess(n_points=5, gauss_hermite=False)
    dtypes = []
    for scale in (weak, strong):
        operands = mixed | {"n_std": scale}
        eager = np.asarray(spec.compute_gridpoints(**operands))
        actual = np.asarray(owner(spec=spec, parameters=operands, required=required))
        assert actual.dtype == eager.dtype
        assert actual.tobytes() == eager.tobytes()
        dtypes.append(actual.dtype)
    assert dtypes == [
        np.dtype("float32"),
        np.dtype("float64" if jax.config.x64_enabled else "float32"),
    ]
    assert len(owner.grids) == 2


# keyword-only-exempt: library-callback=functools.partialmethod
def _observe_stages(
    self: process_grids.SimulationProcessGrids,
    *,
    original: Callable[..., ValueND],
    produced: list[ValueND],
    **kwargs: Any,
) -> ValueND:
    for result in produced:
        missing = resident_bytes_by_device(
            live=measure_buffer_footprint(tree=result),
            arguments=self.snapshot(),
            devices=self.devices,
        )
        assert not any(missing.values())
    result = original(self, **kwargs)
    assert result.sharding == kwargs["required"]
    produced.append(result)
    return result


def test_normal_stages_retain_operands_and_prior_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every preceding scalar remains charged through the next stage's admission."""
    parameters = {
        "mu": jnp.asarray(2.0),
        "sigma": jnp.asarray(1.0),
        "n_std": jnp.asarray(2.0),
    }
    owner = _owner(parameters=parameters)
    produced: list[ValueND] = []
    monkeypatch.setattr(
        process_grids.SimulationProcessGrids,
        "_produce",
        partialmethod(
            _observe_stages,
            original=process_grids.SimulationProcessGrids._produce,
            produced=produced,
        ),
    )
    result = owner(
        spec=NormalIIDProcess(n_points=5, gauss_hermite=False),
        parameters=parameters,
        required=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )
    assert len(produced) == 5
    np.testing.assert_array_equal(
        [np.asarray(value).item() for value in produced[:4]], [2.0, 0.0, 2.0, 4.0]
    )
    np.testing.assert_array_equal(result, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert not owner.temporary_roots


def test_normal_refused_stage_releases_temporary_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused fourth stage releases all placed parameters and completed scalars."""
    parameters = {
        "mu": jnp.asarray(2.0),
        "sigma": jnp.asarray(1.0),
        "n_std": jnp.asarray(2.0),
    }
    owner = _owner(parameters=parameters)
    stages: list[str] = []
    profiled: list[jax.stages.Compiled] = []
    monkeypatch.setattr(
        host_operations.ProfiledSimulationOperations,
        "compile_candidate",
        partialmethod(
            _inject_support_reservation,
            original=host_operations.ProfiledSimulationOperations.compile_candidate,
            profiled=profiled,
            stages=stages,
            refuse_at=4,
        ),
    )
    with pytest.raises(ExecutionPlanningError):
        owner(
            spec=NormalIIDProcess(n_points=5, gauss_hermite=False),
            parameters=parameters,
            required=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        )
    assert stages == ["multiply", "subtract", "multiply", "add"]
    assert not owner.temporary_roots
    assert not owner.grids
    assert not owner.bindings


def test_normal_support_reuses_sealed_bindings_and_rejects_changed_operands() -> None:
    """A sealed owner reuses its exact support and rejects new parameter contents."""
    parameters = {
        "mu": jnp.asarray(2.0),
        "sigma": jnp.asarray(1.0),
        "n_std": jnp.asarray(2.0),
    }
    owner = _owner(parameters=parameters)
    spec = NormalIIDProcess(n_points=5, gauss_hermite=False)
    required = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    result = owner(spec=spec, parameters=parameters, required=required)
    owner.seal()
    assert owner(spec=spec, parameters=parameters, required=required) is result
    copied = {name: value.copy() for name, value in parameters.items()}
    assert owner(spec=spec, parameters=copied, required=required) is result
    with pytest.raises(ExecutionPlanningError, match="changed after entry"):
        owner(
            spec=spec,
            parameters=parameters | {"mu": jnp.asarray(3.0)},
            required=required,
        )
    np.testing.assert_array_equal(result, [0.0, 1.0, 2.0, 3.0, 4.0])
