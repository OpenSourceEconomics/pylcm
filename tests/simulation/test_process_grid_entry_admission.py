"""Runtime process support must be admitted before allocating its device grid."""

import gc
import inspect
import sys
import weakref
from collections.abc import Callable
from functools import partial, partialmethod
from pathlib import Path
from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.simulation import host_operations, process_grids
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.residency import measure_buffer_footprint, resident_bytes_by_device
from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    ExecutionConfig,
    LinSpacedGrid,
    LogNormalIIDProcess,
    Model,
    NormalIIDProcess,
    NormalMixtureIIDProcess,
    Regime,
    RouwenhorstAR1Process,
    TauchenAR1Process,
    TauchenNormalMixtureAR1Process,
    UniformIIDProcess,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.persistence import load_solution
from lcm.typing import FloatND, ScalarInt, UserInitialConditions, UserParams
from tests.simulation.test_budget_lifecycle import _LifecycleRegimeId


@pytest.fixture(autouse=True)
def _isolated_grid_profile_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep controlled compiler reports inside the test that publishes them."""
    monkeypatch.setattr(
        process_grids, "_UNIFORM_GRID_OPERATIONS", ProfiledSimulationOperations()
    )


def _utility(*, income: FloatND, saving: FloatND) -> FloatND:
    return income + saving


def _utility_with_companion(
    *, income: FloatND, saving: FloatND, companion: FloatND
) -> FloatND:
    return income + saving + 0 * companion


def _terminal_utility() -> float:
    return 0.0


def _next_regime() -> ScalarInt:
    return _LifecycleRegimeId.done


def _initial_age(age: float) -> bool:
    return age == 0


def _inputs(
    *,
    budget: int | None,
    fixed_start: float | None = None,
    companion: _ContinuousStochasticProcess | None = None,
    companion_params: dict[str, float] | None = None,
) -> tuple[Model, UserParams, UserInitialConditions]:
    model = Model(
        regimes={
            "alive": Regime(
                transition=_next_regime,
                active=_initial_age,
                states={"income": UniformIIDProcess(n_points=5, start=fixed_start)}
                | ({} if companion is None else {"companion": companion}),
                actions={"saving": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={
                    "utility": _utility
                    if companion is None
                    else _utility_with_companion
                },
            ),
            "done": Regime(transition=None, functions={"utility": _terminal_utility}),
        },
        regime_id_class=_LifecycleRegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        execution_config=ExecutionConfig(device_memory_bytes=budget),
    )
    params = {
        "alive": {
            "income": {"start": 1.0, "stop": 3.0},
            "koopmans_aggregator": {"discount_factor": 0.9},
        },
        "done": {},
    }
    if fixed_start is not None:
        del params["alive"]["income"]["start"]
    if companion is not None:
        assert companion_params is not None
        params["alive"]["companion"] = companion_params
    initial = {
        "income": jnp.asarray([2.0]),
        "age": jnp.asarray([0.0]),
        "regime_id": jnp.asarray([_LifecycleRegimeId.alive]),
    }
    if companion is not None:
        initial["companion"] = jnp.asarray([2.0])
    return model, params, initial


_MIXTURE_PARAMS = {
    "n_std": 3.2,
    "p1": 0.4321,
    "mu1": 0.13,
    "mu2": 0.92,
    "sigma1": 0.734,
    "sigma2": 1.652,
}
_COMPOSITE_CASES = [
    (
        NormalIIDProcess(n_points=5, gauss_hermite=False),
        {"mu": 0.1415, "sigma": 1.876, "n_std": 3.2},
    ),
    (
        LogNormalIIDProcess(n_points=5, gauss_hermite=False),
        {"mu": 0.1415, "sigma": 1.876, "n_std": 3.2},
    ),
    (NormalMixtureIIDProcess(n_points=5), _MIXTURE_PARAMS),
    (
        TauchenAR1Process(n_points=5, gauss_hermite=False),
        {"rho": 0.934, "mu": 0.125, "sigma": 0.568, "n_std": 3.2},
    ),
    (RouwenhorstAR1Process(n_points=5), {"rho": 0.934, "mu": 0.125, "sigma": 0.568}),
    (
        TauchenNormalMixtureAR1Process(n_points=5),
        {"rho": 0.934, "mu": 0.125, **_MIXTURE_PARAMS},
    ),
    (NormalIIDProcess(n_points=5, gauss_hermite=True), {"mu": 0.1415, "sigma": 1.876}),
    (
        LogNormalIIDProcess(n_points=5, gauss_hermite=True),
        {"mu": 0.1415, "sigma": 1.876},
    ),
    (
        TauchenAR1Process(n_points=5, gauss_hermite=True),
        {"rho": 0.934, "mu": 0.125, "sigma": 0.568},
    ),
]


@pytest.mark.parametrize(("companion", "companion_params"), _COMPOSITE_CASES)
def test_mixed_processes_preserve_saved_support_with_a_uniform_admission_profile(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    companion: _ContinuousStochasticProcess,
    companion_params: dict[str, float],
) -> None:
    """Composite support remains compatible while uniform support is admitted once."""
    producer, params, initial = _inputs(
        budget=None, companion=companion, companion_params=companion_params
    )
    path = producer.solve(params=params, log_level="off").save(path=tmp_path / "mixed")
    restored = load_solution(path=path)
    consumer, _, _ = _inputs(
        budget=2**28, companion=companion, companion_params=companion_params
    )
    observations: list[bool] = []
    monkeypatch.setattr(
        jnp,
        "linspace",
        partial(_guard_process_grid, original=jnp.linspace, observations=observations),
    )
    result = consumer.simulate(
        params=params, initial_conditions=initial, solution=restored, log_level="debug"
    )
    first = result.to_dataframe(use_labels=False).iloc[0]
    np.testing.assert_array_equal(
        first[["income", "saving", "value"]].to_numpy(dtype=float), [2.0, 1.0, 3.0]
    )
    assert observations == [True]


def _guard_process_grid(
    *args: Any, original: Callable[..., Any], observations: list[bool], **kwargs: Any
) -> Any:
    if (
        sys._getframe(1).f_code
        is inspect.unwrap(UniformIIDProcess.compute_gridpoints).__code__
    ):
        traced = isinstance(kwargs["start"], jax.core.Tracer)
        observations.append(traced)
        if not traced:
            raise AssertionError("Process grid allocated before compiler admission")
    return original(*args, **kwargs)


def _over_budget_peak(
    *, compiled: jax.stages.Compiled, profiled: list[jax.stages.Compiled], **kwargs: Any
) -> int:
    del kwargs
    profiled.append(compiled)
    return 2**30


# keyword-only-exempt: library-callback=functools.partialmethod
def _forbid_profiled_dispatch(
    self: jax.stages.Compiled,
    *args: Any,
    profiled: list[jax.stages.Compiled],
    original: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    if any(self is executable for executable in profiled):
        raise AssertionError("An over-budget process executable was dispatched")
    return original(self, *args, **kwargs)


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("log_level", ["off", "debug"])
def test_automatic_simulation_refuses_process_grid_before_allocation(
    *,
    monkeypatch: pytest.MonkeyPatch,
    supplied: bool,
    log_level: LogLevel,
) -> None:
    """A process producer exceeding the budget refuses before its device dispatch."""
    model, params, initial = _inputs(budget=2**20)
    solution = model.solve(params=params, log_level="off") if supplied else None
    observations: list[bool] = []
    profiled: list[jax.stages.Compiled] = []
    monkeypatch.setattr(
        jnp,
        "linspace",
        partial(_guard_process_grid, original=jnp.linspace, observations=observations),
    )
    monkeypatch.setattr(
        host_operations,
        "compiler_peak_bytes",
        partial(_over_budget_peak, profiled=profiled),
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
        model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level=log_level,
        )
    assert observations == [True]
    with pytest.raises(
        AssertionError, match="over-budget process executable was dispatched"
    ):
        profiled[0]()


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("budget", [None, 2**28])
@pytest.mark.parametrize("log_level", ["off", "debug"])
def test_process_support_preserves_simulated_value_and_action(
    *,
    monkeypatch: pytest.MonkeyPatch,
    supplied: bool,
    budget: int | None,
    log_level: LogLevel,
) -> None:
    """An income of two and the optimal saving of one produce utility three."""
    model, params, initial = _inputs(budget=budget)
    solution = model.solve(params=params, log_level="off") if supplied else None
    observations: list[bool] = []
    if budget is not None:
        monkeypatch.setattr(
            jnp,
            "linspace",
            partial(
                _guard_process_grid, original=jnp.linspace, observations=observations
            ),
        )
    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level=log_level,
    )
    rows = result.to_dataframe(use_labels=False)
    first = rows.iloc[0]
    np.testing.assert_array_equal(
        first[["income", "saving", "value"]].to_numpy(dtype=float), [2.0, 1.0, 3.0]
    )
    if budget is not None:
        assert observations == [True]


def test_saved_unbudgeted_solution_preserves_support_under_simulation_budget(
    tmp_path: Path,
) -> None:
    """Stored support retains its exact identity when a consumer adds a budget."""
    producer, params, initial = _inputs(budget=None)
    path = producer.solve(params=params, log_level="off").save(
        path=tmp_path / "solution"
    )
    restored = load_solution(path=path)
    consumer, _, _ = _inputs(budget=2**28)
    result = consumer.simulate(
        params=params, initial_conditions=initial, solution=restored, log_level="debug"
    )
    first = result.to_dataframe(use_labels=False).iloc[0]
    np.testing.assert_array_equal(
        first[["income", "saving", "value"]].to_numpy(dtype=float), [2.0, 1.0, 3.0]
    )


def _forbid_fixed_parameter_upload(spec: UniformIIDProcess) -> object:
    del spec
    raise AssertionError("Fixed process parameters allocated outside admission")


def test_fixed_uniform_parameters_enter_through_admitted_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed lower endpoint reaches its grid without an eager scalar upload."""
    model, params, initial = _inputs(budget=2**28, fixed_start=1.0)
    monkeypatch.setattr(
        UniformIIDProcess, "params", property(_forbid_fixed_parameter_upload)
    )
    result = model.simulate(params=params, initial_conditions=initial, log_level="off")
    first = result.to_dataframe(use_labels=False).iloc[0]
    np.testing.assert_array_equal(
        first[["income", "saving", "value"]].to_numpy(dtype=float), [2.0, 1.0, 3.0]
    )


# keyword-only-exempt: library-callback=functools.partialmethod
def _record_entry_owner(
    self: SimulationEntryAllocations,
    *,
    original: Callable[..., None],
    references: list[weakref.ReferenceType],
) -> None:
    original(self)
    references.append(weakref.ref(self))


def _refuse_automatic_solve(*args: Any, **kwargs: Any) -> Any:
    del args
    grids = kwargs["process_grid_resolver"].grids
    assert len(grids) == 1
    np.testing.assert_array_equal(next(iter(grids.values())), [1.0, 1.5, 2.0, 2.5, 3.0])
    raise ExecutionPlanningError("Controlled failure after process entry")


def _attempt_refused_simulation(
    *, model: Model, params: UserParams, initial: UserInitialConditions
) -> None:
    with pytest.raises(
        ExecutionPlanningError, match=r"^Controlled failure after process entry$"
    ):
        model.simulate(params=params, initial_conditions=initial, log_level="off")


def test_process_entry_failure_releases_owners_without_cyclic_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused call releases its complete entry owner through ordinary unwinding."""
    model, params, initial = _inputs(budget=2**28)
    references: list[weakref.ReferenceType] = []
    monkeypatch.setattr(
        SimulationEntryAllocations,
        "__post_init__",
        partialmethod(
            _record_entry_owner,
            original=SimulationEntryAllocations.__post_init__,
            references=references,
        ),
    )
    monkeypatch.setattr(Model, "_solve_from_flat_params", _refuse_automatic_solve)
    enabled = gc.isenabled()
    gc.disable()
    try:
        _attempt_refused_simulation(model=model, params=params, initial=initial)
        assert len(references) == 1
        assert references[0]() is None
    finally:
        if enabled:
            gc.enable()
        gc.collect()


# keyword-only-exempt: library-callback=functools.partialmethod
def _observe_cumulative_grid_ownership(
    self: process_grids.SimulationProcessGrids,
    *,
    original: Callable[..., FloatND],
    produced: list[FloatND],
    **kwargs: Any,
) -> FloatND:
    for grid in produced:
        missing = resident_bytes_by_device(
            live=measure_buffer_footprint(tree=grid),
            arguments=self.snapshot(),
            devices=self.devices,
        )
        assert not any(missing.values()), (
            "An earlier grid is absent before the next producer"
        )
    result = original(self, **kwargs)
    produced.append(result)
    return result


# keyword-only-exempt: library-callback=functools.partialmethod
def _observe_solve_grid_ownership(
    self: Model,
    *,
    original: Callable[..., Any],
    produced: list[FloatND],
    **kwargs: Any,
) -> Any:
    assert len(produced) == 2
    held = measure_buffer_footprint(tree=kwargs["retained_input_arrays"])
    for grid in produced:
        missing = resident_bytes_by_device(
            live=measure_buffer_footprint(tree=grid),
            arguments=held,
            devices=tuple(grid.sharding.device_set),
        )
        assert not any(missing.values()), (
            "Automatic solve omitted an admitted process grid"
        )
    return original(self, **kwargs)


def test_uniform_grids_remain_owned_during_later_production_and_automatic_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each completed support stays charged through the next producer and solve."""
    model, params, initial = _inputs(
        budget=2**28,
        companion=UniformIIDProcess(n_points=3),
        companion_params={"start": 1.0, "stop": 3.0},
    )
    produced: list[FloatND] = []
    monkeypatch.setattr(
        process_grids.SimulationProcessGrids,
        "_produce",
        partialmethod(
            _observe_cumulative_grid_ownership,
            original=process_grids.SimulationProcessGrids._produce,
            produced=produced,
        ),
    )
    monkeypatch.setattr(
        Model,
        "_solve_compiled",
        partialmethod(
            _observe_solve_grid_ownership,
            original=Model._solve_compiled,
            produced=produced,
        ),
    )
    result = model.simulate(params=params, initial_conditions=initial, log_level="off")
    np.testing.assert_array_equal(produced[0], [1.0, 1.5, 2.0, 2.5, 3.0])
    np.testing.assert_array_equal(produced[1], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        result.to_dataframe(use_labels=False)
        .iloc[0][["saving", "value"]]
        .to_numpy(dtype=float),
        [1.0, 3.0],
    )


def _read_parameter_once(
    *args: object, original: Callable[..., object], reads: dict[int, object]
) -> Any:
    (value,) = args
    if isinstance(value, jax.Array):
        assert id(value) not in reads, (
            "A repeated parameter binding caused another host read"
        )
        reads[id(value)] = value
    return original(value)


def test_uniform_binding_reuse_avoids_repeated_host_reads_and_warm_grid_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated consumers reuse exact bindings and later calls reuse only code."""
    model, params, initial = _inputs(budget=2**28)
    reads: dict[int, object] = {}
    observations: list[bool] = []
    monkeypatch.setattr(
        process_grids,
        "_parameter_bytes",
        partial(
            _read_parameter_once, original=process_grids._parameter_bytes, reads=reads
        ),
    )
    monkeypatch.setattr(
        jnp,
        "linspace",
        partial(_guard_process_grid, original=jnp.linspace, observations=observations),
    )
    for _ in range(2):
        result = model.simulate(
            params=params, initial_conditions=initial, log_level="off"
        )
        np.testing.assert_array_equal(
            result.to_dataframe(use_labels=False)
            .iloc[0][["saving", "value"]]
            .to_numpy(dtype=float),
            [1.0, 3.0],
        )
        reads.clear()
    assert observations == [True]
