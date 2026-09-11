"""Simulation preflight admits Cartesian action arrays before allocation."""

import dataclasses
import gc
import inspect
import sys
import weakref
from collections.abc import Callable, Mapping
from functools import partial, partialmethod
from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.workspace_planning import CompilerMemoryReservation
from _lcm.simulation import host_operations
from _lcm.simulation import initial_conditions as preflight
from _lcm.simulation.action_grids import PreflightActionGrids
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.residency import measure_buffer_footprint, resident_bytes_by_device
from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError, InvalidInitialConditionsError
from lcm.solver_api import SolutionResult
from lcm.typing import BoolND, FloatND, ScalarInt, UserInitialConditions, UserParams
from tests.execution.test_compiler_allocation_reservation import synthetic_memory
from tests.simulation.test_budget_lifecycle import _LifecycleRegimeId


@categorical(ordered=False)
class _Choice:
    low: ScalarInt
    high: ScalarInt


def _utility(*, wealth: FloatND, choice: ScalarInt, saving: FloatND) -> FloatND:
    return wealth + 3 * choice + saving


def _feasible(*, wealth: FloatND, choice: ScalarInt, saving: FloatND) -> BoolND:
    return choice + saving <= wealth


def _terminal_utility(wealth: FloatND) -> FloatND:
    return 0.0 * wealth


def _next_regime() -> ScalarInt:
    return _LifecycleRegimeId.done


def _initial_age(age: float) -> bool:
    return age == 0


@categorical(ordered=False)
class _ThreeRegimeId:
    alive: ScalarInt
    done: ScalarInt
    other: ScalarInt


def _inputs(
    *,
    budget: int | None,
    two_regimes: bool = False,
    devices: tuple[int, ...] | None = None,
) -> tuple[Model, UserParams, UserInitialConditions]:
    regimes = {
        "alive": Regime(
            transition=_next_regime,
            active=_initial_age,
            states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
            state_transitions={"wealth": fixed_transition("wealth")},
            actions={
                "choice": DiscreteGrid(_Choice),
                "saving": LinSpacedGrid(start=0, stop=2, n_points=3),
            },
            functions={"utility": _utility},
            constraints={"budget": _feasible},
        ),
        "done": Regime(
            transition=None,
            states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
            functions={"utility": _terminal_utility},
        ),
    }
    if two_regimes:
        regimes["other"] = dataclasses.replace(
            regimes["alive"],
            actions={
                "choice": DiscreteGrid(_Choice),
                "saving": LinSpacedGrid(start=0, stop=3, n_points=4),
            },
        )
    model = Model(
        regimes=regimes,
        regime_id_class=_ThreeRegimeId if two_regimes else _LifecycleRegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        execution_config=ExecutionConfig(device_memory_bytes=budget, devices=devices),
    )
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.9}}, "done": {}}
    if two_regimes:
        params["other"] = {"koopmans_aggregator": {"discount_factor": 0.9}}
    initial = {
        "wealth": jnp.asarray([2.0, 2.0] if two_regimes else [2.0]),
        "age": jnp.asarray([0.0, 0.0] if two_regimes else [0.0]),
        "regime_id": jnp.asarray([0, 2] if two_regimes else [0], dtype=jnp.int32),
    }
    return model, params, initial


@pytest.fixture(autouse=True)
def _isolated_preflight_profile_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preflight, "_PREFLIGHT_OPERATIONS", ProfiledSimulationOperations()
    )


def _guard_action_mesh(
    *args: Any, original: Callable[..., Any], observations: list[bool], **kwargs: Any
) -> Any:
    if (
        sys._getframe(1).f_code
        is inspect.unwrap(preflight._build_flat_action_grid).__code__
    ):
        traced = all(isinstance(value, jax.core.Tracer) for value in args)
        observations.append(traced)
        if not traced:
            raise AssertionError("Action grid allocated before compiler admission")
    return original(*args, **kwargs)


def _over_budget_peak(
    *, compiled: jax.stages.Compiled, profiled: list[jax.stages.Compiled], **kwargs: Any
) -> CompilerMemoryReservation:
    del kwargs
    profiled.append(compiled)
    return synthetic_memory(2**30)


# keyword-only-exempt: library-callback=functools.partialmethod
def _profile_action_grid(
    self: ProfiledSimulationOperations,
    *,
    original: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    profiled: list[jax.stages.Compiled],
    **kwargs: Any,
) -> Any:
    if inspect.unwrap(kwargs["function"]) is inspect.unwrap(
        preflight._build_flat_action_grid
    ):
        with monkeypatch.context() as context:
            context.setattr(
                host_operations,
                "compiler_memory_reservation",
                partial(_over_budget_peak, profiled=profiled),
            )
            return original(self, **kwargs)
    return original(self, **kwargs)


# keyword-only-exempt: library-callback=functools.partialmethod
def _forbid_profiled_dispatch(
    self: jax.stages.Compiled,
    *args: Any,
    profiled: list[jax.stages.Compiled],
    original: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    if any(self is executable for executable in profiled):
        raise AssertionError("Over-budget action-grid executable was dispatched")
    return original(self, *args, **kwargs)


@pytest.mark.parametrize("serial_retry", [False, True])
def test_simulate_refuses_preflight_action_grid_before_dispatch(
    *,
    monkeypatch: pytest.MonkeyPatch,
    serial_retry: bool,
) -> None:
    """A supplied solution does not bypass Cartesian action-grid admission."""
    model, params, initial = _inputs(budget=2**20)
    solution = model.solve(params=params, log_level="off")
    if serial_retry:
        monkeypatch.setattr(preflight, "_read_initial_cohorts", _require_serial_retry)
    observations: list[bool] = []
    profiled: list[jax.stages.Compiled] = []
    monkeypatch.setattr(
        jnp,
        "meshgrid",
        partial(_guard_action_mesh, original=jnp.meshgrid, observations=observations),
    )
    monkeypatch.setattr(
        ProfiledSimulationOperations,
        "compile_candidate",
        partialmethod(
            _profile_action_grid,
            original=ProfiledSimulationOperations.compile_candidate,
            monkeypatch=monkeypatch,
            profiled=profiled,
        ),
    )
    monkeypatch.setattr(
        jax.stages.Compiled,
        "__call__",
        partialmethod(
            _forbid_profiled_dispatch,
            original=jax.stages.Compiled.__call__,
            profiled=profiled,
        ),
    )
    with pytest.raises(ExecutionPlanningError):
        model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="debug",
        )
    assert observations == [True]
    assert len(profiled) == 1
    with pytest.raises(
        AssertionError, match="Over-budget action-grid executable was dispatched"
    ):
        profiled[0]()


def _require_serial_retry(**kwargs: Any) -> None:
    del kwargs
    raise preflight._SerialValidationRequired


# keyword-only-exempt: library-callback=functools.partialmethod
def _record_resolution(
    self: PreflightActionGrids,
    *,
    original: Callable[..., Any],
    products: list[Mapping[str, FloatND]],
    **kwargs: Any,
) -> Mapping[str, FloatND]:
    if self.memory is not None and products:
        owned = measure_buffer_footprint(tree=products)
        missing = resident_bytes_by_device(
            live=owned, arguments=self.memory.snapshot(), devices=tuple(owned.spans)
        )
        assert not any(missing.values()), (
            "An admitted Cartesian product disappeared from current residency"
        )
    result = original(self, **kwargs)
    products.append(result)
    return result


@pytest.mark.parametrize("budget", [None, 2**28])
@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_preflight_action_grid_preserves_order_dtypes_and_public_decisions(
    *,
    monkeypatch: pytest.MonkeyPatch,
    budget: int | None,
    supplied: bool,
    log_level: LogLevel,
) -> None:
    """Cartesian order keeps the optimal choice 1, saving 1 and value 6."""
    model, params, initial = _inputs(budget=budget)
    solution = model.solve(params=params, log_level="off") if supplied else None
    products: list[Mapping[str, FloatND]] = []
    monkeypatch.setattr(
        PreflightActionGrids,
        "resolve",
        partialmethod(
            _record_resolution, original=PreflightActionGrids.resolve, products=products
        ),
    )
    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level=log_level,
    )
    rows = result.to_dataframe(use_labels=False)
    np.testing.assert_array_equal(
        rows.loc[
            rows["regime_name"] == "alive", ["wealth", "choice", "saving", "value"]
        ].to_numpy(),
        [[2, 1, 1, 6]],
    )
    if log_level == "off":
        assert products == []
    else:
        assert len(products) == 1
        np.testing.assert_array_equal(products[0]["choice"], [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(products[0]["saving"], [0, 1, 2, 0, 1, 2])
        assert products[0]["choice"].dtype == np.dtype("int32")
        assert products[0]["saving"].dtype == np.dtype(
            "float64" if jax.config.jax_enable_x64 else "float32"
        )


def test_invalid_initial_conditions_reuse_admitted_products_during_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serial diagnostics retain charged product owners after summary cleanup."""
    model, params, initial = _inputs(budget=2**28)
    solution = model.solve(params=params, log_level="off")
    initial = {**initial, "wealth": jnp.asarray([-1.0])}
    products: list[Mapping[str, FloatND]] = []
    observations: list[bool] = []
    monkeypatch.setattr(
        PreflightActionGrids,
        "resolve",
        partialmethod(
            _record_resolution, original=PreflightActionGrids.resolve, products=products
        ),
    )
    monkeypatch.setattr(
        jnp,
        "meshgrid",
        partial(_guard_action_mesh, original=jnp.meshgrid, observations=observations),
    )
    with pytest.raises(
        InvalidInitialConditionsError, match="All actions are infeasible for 1 subject"
    ):
        model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="debug",
        )
    assert len(products) == 2
    assert products[0] is products[1]
    assert observations == [True]
    np.testing.assert_array_equal(products[0]["choice"], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(products[0]["saving"], [0, 1, 2, 0, 1, 2])


def test_repeated_simulation_reuses_only_action_grid_executable_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two simulations reuse one compiled producer and retain the same decisions."""
    model, params, initial = _inputs(budget=2**28)
    solution = model.solve(params=params, log_level="off")
    observations: list[bool] = []
    monkeypatch.setattr(
        jnp,
        "meshgrid",
        partial(_guard_action_mesh, original=jnp.meshgrid, observations=observations),
    )
    for _ in range(2):
        result = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="debug",
        )
        np.testing.assert_array_equal(
            result.to_dataframe(use_labels=False)
            .iloc[0][["choice", "saving", "value"]]
            .to_numpy(dtype=float),
            [1, 1, 6],
        )
    assert observations == [True]


def test_distinct_regime_products_are_cumulatively_owned_during_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each regime's next product is admitted beside all earlier live products."""
    model, params, initial = _inputs(budget=2**28, two_regimes=True)
    products: list[Mapping[str, FloatND]] = []
    monkeypatch.setattr(
        PreflightActionGrids,
        "resolve",
        partialmethod(
            _record_resolution, original=PreflightActionGrids.resolve, products=products
        ),
    )
    result = model.simulate(
        params=params, initial_conditions=initial, log_level="debug"
    )
    assert len(products) == 2
    np.testing.assert_array_equal(products[0]["saving"], [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(products[1]["saving"], [0, 1, 2, 3, 0, 1, 2, 3])
    rows = result.to_dataframe(use_labels=False)
    np.testing.assert_array_equal(
        rows.loc[
            rows["regime_name"].isin(["alive", "other"]), ["choice", "saving", "value"]
        ].to_numpy(),
        [[1, 1, 6], [1, 1, 6]],
    )


# keyword-only-exempt: library-callback=functools.partialmethod
def _observe_product_owners(
    self: PreflightActionGrids,
    *,
    original: Callable[..., Any],
    references: list[weakref.ReferenceType[object]],
    **kwargs: Any,
) -> Mapping[str, FloatND]:
    result = original(self, **kwargs)
    references.extend(weakref.ref(value) for value in result.values())
    return result


def _controlled_feasibility_failure(**kwargs: Any) -> None:
    np.testing.assert_array_equal(kwargs["flat_actions"]["choice"], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(kwargs["flat_actions"]["saving"], [0, 1, 2, 0, 1, 2])
    raise RuntimeError("Controlled failure after Cartesian admission")


def _attempt_failed_preflight(
    *,
    model: Model,
    params: UserParams,
    initial: UserInitialConditions,
    solution: SolutionResult,
) -> None:
    with pytest.raises(
        RuntimeError, match=r"^Controlled failure after Cartesian admission$"
    ):
        model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="debug",
        )


@pytest.mark.parametrize("fails", [False, True])
def test_preflight_releases_cartesian_owners_without_cyclic_gc(
    *,
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
) -> None:
    """Success and exceptions release temporary Cartesian arrays immediately."""
    model, params, initial = _inputs(budget=2**28)
    solution = model.solve(params=params, log_level="off")
    references: list[weakref.ReferenceType[object]] = []
    monkeypatch.setattr(
        PreflightActionGrids,
        "resolve",
        partialmethod(
            _observe_product_owners,
            original=PreflightActionGrids.resolve,
            references=references,
        ),
    )
    if fails:
        monkeypatch.setattr(
            preflight, "_batched_feasibility_check", _controlled_feasibility_failure
        )
    enabled = gc.isenabled()
    gc.disable()
    try:
        if fails:
            _attempt_failed_preflight(
                model=model, params=params, initial=initial, solution=solution
            )
        else:
            result = model.simulate(
                params=params,
                initial_conditions=initial,
                solution=solution,
                log_level="debug",
            )
            np.testing.assert_array_equal(
                result.to_dataframe(use_labels=False)
                .iloc[0][["choice", "saving", "value"]]
                .to_numpy(dtype=float),
                [1, 1, 6],
            )
        assert len(references) == (4 if fails else 2)
        assert all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()
