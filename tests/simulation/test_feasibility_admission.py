"""Public simulation feasibility producers respect the declared memory budget."""

import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from _lcm.simulation import initial_conditions as preflight
from _lcm.simulation.initial_conditions import _SerialValidationRequired
from _lcm.utils.logging import LogLevel
from lcm import AgeGrid, ExecutionConfig, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import ExecutionPlanningError, InvalidInitialConditionsError
from lcm.params import MappingLeaf, as_leaf
from lcm.persistence import load_solution
from lcm.solver_api import SolutionResult
from lcm.typing import BoolND, FloatND, ScalarInt, UserInitialConditions, UserParams


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    done: ScalarInt


def _utility(*, wealth: FloatND, saving: FloatND) -> FloatND:
    return wealth + saving


def _terminal_utility(*, wealth: FloatND) -> FloatND:
    return wealth * 0.0


def _next_regime() -> ScalarInt:
    return _RegimeId.done


def _initial_age(age: float) -> bool:
    return age == 0


def _costly_constraint(*, wealth: FloatND, saving: FloatND, cutoff: FloatND) -> BoolND:
    """Sorting requires real workspace while its reduced predicate is scalar."""
    sample = jnp.sin(jnp.arange(4096, dtype=wealth.dtype) + wealth + saving)
    return jnp.sort(sample)[2048] > cutoff


def _costly_action_constraint(*, saving: FloatND, cutoff: FloatND) -> BoolND:
    sample = jnp.sin(jnp.arange(4096, dtype=saving.dtype) + saving)
    return jnp.sort(sample)[2048] > cutoff


def _reject_every_action(*, wealth: FloatND) -> BoolND:
    return wealth < 0


def _parameter_constraint(*, cutoff: FloatND) -> BoolND:
    return cutoff < 0


def _mapping_constraint(
    *, wealth: FloatND, saving: FloatND, cutoff: MappingLeaf
) -> BoolND:
    limits = cast("FloatND", cutoff.data["limits"])
    return wealth + saving > limits.sum()


def _inputs(
    *,
    budget: int | None,
    cutoff: float = 2.0,
    n_actions: int = 2,
    n_subjects: int = 2,
    constant: bool = False,
    reject: bool = False,
    parameter_only: bool = False,
    structured: bool = False,
    diagnostic_checks: int = 0,
    devices: tuple[int, ...] | None = None,
) -> tuple[Model, UserParams, UserInitialConditions]:
    constraints: dict[str, Callable[..., BoolND]] = {
        "costly": _costly_action_constraint if constant else _costly_constraint
    }
    if parameter_only:
        constraints = {"costly": _parameter_constraint}
    if structured:
        constraints = {"costly": _mapping_constraint}
    if reject:
        constraints = {"reject": _reject_every_action, **constraints}
    constraints.update(
        {f"check_{index}": _reject_every_action for index in range(diagnostic_checks)}
    )
    model = Model(
        regimes={
            "alive": Regime(
                transition=_next_regime,
                active=_initial_age,
                states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
                state_transitions={"wealth": lcm.fixed_transition("wealth")},
                actions={"saving": LinSpacedGrid(start=0, stop=1, n_points=n_actions)},
                functions={"utility": _utility},
                constraints=constraints,
            ),
            "done": Regime(
                transition=None,
                states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
                functions={"utility": _terminal_utility},
            ),
        },
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        execution_config=ExecutionConfig(device_memory_bytes=budget, devices=devices),
    )
    params = {
        "alive": {
            "koopmans_aggregator": {"discount_factor": 0.9},
            "costly": {
                "cutoff": as_leaf({"limits": jnp.array([cutoff, cutoff])})
                if structured
                else cutoff
            },
        },
        "done": {},
    }
    initial = {
        "wealth": jnp.linspace(1.0, 2.0, n_subjects),
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
    }
    return model, params, initial


@dataclasses.dataclass
class _CompilerBoundary:
    profiled: list[tuple[jax.stages.Compiled, int]] = dataclasses.field(
        default_factory=list
    )
    dispatched: list[jax.stages.Compiled] = dataclasses.field(default_factory=list)

    def require_declined(self, name: str) -> jax.stages.Compiled:
        program, offset = self.profiled[-1]
        hlo = program.as_text()
        assert hlo is not None
        assert name in hlo
        assert all(program is not executed for executed in self.dispatched[offset:])
        return program


@pytest.fixture
def compiler_boundary(monkeypatch: pytest.MonkeyPatch) -> _CompilerBoundary:
    """Observe actual compiler and dispatch boundaries without changing their result."""
    observed = _CompilerBoundary()
    analyze_program = jax.stages.Compiled.memory_analysis
    dispatch_program = jax.stages.Compiled.__call__

    def analyze_and_record(self: Any, *args: Any, **kwargs: Any) -> Any:
        stats = analyze_program(self, *args, **kwargs)
        observed.profiled.append((self, len(observed.dispatched)))
        return stats

    def dispatch_and_record(self: Any, *args: Any, **kwargs: Any) -> Any:
        observed.dispatched.append(self)
        return dispatch_program(self, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Compiled, "memory_analysis", analyze_and_record)
    monkeypatch.setattr(jax.stages.Compiled, "__call__", dispatch_and_record)
    return observed


@pytest.fixture(scope="module")
def wide_solution(tmp_path_factory: pytest.TempPathFactory) -> SolutionResult:
    model, params, _ = _inputs(budget=2**24, n_actions=1024)
    directory = tmp_path_factory.mktemp("feasibility-solution") / "solution"
    model.solve(params=params, log_level="off").save(path=directory)
    return load_solution(path=directory)


@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("log_level", ["warning", "progress", "debug"])
@pytest.mark.requires(device="cpu")
def test_simulate_refuses_feasibility_workspace_before_diagnostics(
    *,
    serial: bool,
    supplied: bool,
    log_level: LogLevel,
    wide_solution: SolutionResult,
    compiler_boundary: _CompilerBoundary,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """The actual constraint workspace must fit before its predicate is evaluated."""
    assert (
        Path(lcm.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[2] / "src")
    )
    budget = (12 if jax.config.x64_enabled else 6) * 1024
    model, params, initial = _inputs(budget=budget, n_actions=1024)
    if serial:

        def require_serial(**kwargs: Any) -> None:
            del kwargs
            raise _SerialValidationRequired

        monkeypatch.setattr(preflight, "_read_initial_cohorts", require_serial)
    with pytest.raises(ExecutionPlanningError, match="compiler peak") as error:
        model.simulate(
            params=params,
            initial_conditions=initial,
            solution=wide_solution if supplied else None,
            log_level=log_level,
        )
    declined = compiler_boundary.require_declined("_batched_feasibility_check")
    stats = declined.memory_analysis()
    assert stats is not None
    record_property("compiler_peak", stats.peak_memory_in_bytes)
    record_property("refusal", str(error.value))


@pytest.mark.requires(device="cpu")
def test_simulate_refuses_cohort_constant_feasibility_workspace(
    compiler_boundary: _CompilerBoundary,
) -> None:
    """Action-only predicates require admission even when every subject shares them."""
    budget = (9 if jax.config.x64_enabled else 5) * 1024
    model, params, initial = _inputs(budget=budget, n_actions=1024, constant=True)
    with pytest.raises(ExecutionPlanningError, match="compiler peak"):
        model.simulate(params=params, initial_conditions=initial, log_level="debug")
    compiler_boundary.require_declined("_evaluate_constant_feasibility")


@pytest.mark.requires(device="cpu")
def test_simulate_refuses_individual_constraint_gather_with_joint_results_resident(
    compiler_boundary: _CompilerBoundary,
) -> None:
    """Natural invalid-result replay admits the gather for its diagnostic table."""
    budget = (40 if jax.config.x64_enabled else 24) * 1024
    model, params, initial = _inputs(budget=budget, n_subjects=1024, reject=True)
    with pytest.raises(
        ExecutionPlanningError, match=r"compiler peak|before allocation"
    ):
        model.simulate(params=params, initial_conditions=initial, log_level="debug")
    gathers = [
        program
        for program in compiler_boundary.dispatched
        if "_gather_feasibility_inputs" in (program.as_text() or "")
    ]
    # The cohort gather runs for summary and serial replay. The diagnostic subset
    # gather is refused during placement or compiler admission, before dispatch.
    assert len(gathers) == 2


@pytest.mark.requires(device="cpu")
def test_simulate_refuses_individual_predicate_after_diagnostic_gather(
    compiler_boundary: _CompilerBoundary,
) -> None:
    budget = (44 if jax.config.x64_enabled else 28) * 1024
    model, params, initial = _inputs(
        budget=budget, n_subjects=1024, reject=True, diagnostic_checks=8
    )
    with pytest.raises(ExecutionPlanningError, match="compiler peak"):
        model.simulate(params=params, initial_conditions=initial, log_level="debug")
    gathers = [
        program
        for program in compiler_boundary.dispatched
        if "_gather_feasibility_inputs" in (program.as_text() or "")
    ]
    assert len(gathers) == 3
    compiler_boundary.require_declined("_batched_feasibility_check")


def test_compiler_boundary_detects_execution(
    compiler_boundary: _CompilerBoundary,
) -> None:
    program = jax.jit(_terminal_utility).lower(wealth=jnp.asarray(1.0)).compile()
    program.memory_analysis()
    assert float(program(wealth=jnp.asarray(1.0))) == 0.0
    with pytest.raises(AssertionError):
        compiler_boundary.require_declined("_terminal_utility")


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_generous_budget_preserves_values_actions_and_caller_arrays(
    *, constant: bool, supplied: bool, log_level: LogLevel
) -> None:
    model, params, initial = _inputs(budget=2**24, cutoff=-2.0, constant=constant)
    snapshots = {name: np.array(value) for name, value in initial.items()}
    solution = model.solve(params=params, log_level="off") if supplied else None
    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level=log_level,
    )
    raw = result.raw_results["alive"][0]
    np.testing.assert_array_equal(raw.actions["saving"], [1.0, 1.0])
    np.testing.assert_allclose(raw.V_arr, [2.0, 3.0])
    for name, snapshot in snapshots.items():
        np.testing.assert_array_equal(initial[name], snapshot)


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("log_level", ["warning", "progress", "debug"])
def test_generous_budget_preserves_natural_invalid_diagnostics(
    *, constant: bool, log_level: LogLevel, caplog: pytest.LogCaptureFixture
) -> None:
    legacy, legacy_params, legacy_initial = _inputs(budget=None, constant=constant)
    with pytest.raises(InvalidInitialConditionsError) as oracle:
        legacy.simulate(
            params=legacy_params, initial_conditions=legacy_initial, log_level="debug"
        )
    model, params, initial = _inputs(budget=2**24, constant=constant)
    if log_level == "debug":
        with pytest.raises(InvalidInitialConditionsError) as actual:
            model.simulate(
                params=params, initial_conditions=initial, log_level=log_level
            )
        assert str(actual.value) == str(oracle.value)
    else:
        with caplog.at_level(logging.WARNING, logger="lcm"):
            model.simulate(
                params=params, initial_conditions=initial, log_level=log_level
            )
        messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert messages[0] == str(oracle.value)


@pytest.mark.parametrize(
    "error_type", [ExecutionPlanningError, MemoryError, jax.errors.JaxRuntimeError]
)
def test_feasibility_resource_errors_escape_without_serial_retry(
    *, error_type: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = error_type("feasibility producer unavailable")

    def fail(**kwargs: Any) -> None:
        del kwargs
        raise failure

    def forbidden(**kwargs: Any) -> None:
        del kwargs
        pytest.fail("A resource exception entered serial validation.")

    monkeypatch.setattr(preflight, "_run_profiled_feasibility", fail)
    monkeypatch.setattr(preflight, "validate_initial_conditions", forbidden)
    model, params, initial = _inputs(budget=2**24)
    with pytest.raises(error_type) as actual:
        model.simulate(params=params, initial_conditions=initial, log_level="debug")
    assert actual.value is failure


@pytest.mark.parametrize(
    ("parameter_only", "structured"), [(False, False), (True, False), (False, True)]
)
def test_feasibility_reads_current_parameters_on_each_call(
    *, parameter_only: bool, structured: bool
) -> None:
    model, params, initial = _inputs(
        budget=2**24, cutoff=-2.0, parameter_only=parameter_only, structured=structured
    )
    result = model.simulate(
        params=params, initial_conditions=initial, log_level="debug"
    )
    np.testing.assert_allclose(result.raw_results["alive"][0].V_arr, [2.0, 3.0])
    params = {
        "alive": {
            "koopmans_aggregator": {"discount_factor": 0.9},
            "costly": {
                "cutoff": as_leaf({"limits": jnp.array([2.0, 2.0])})
                if structured
                else 2.0
            },
        },
        "done": {},
    }
    with pytest.raises(
        InvalidInitialConditionsError, match="All actions are infeasible"
    ):
        model.simulate(params=params, initial_conditions=initial, log_level="debug")


@pytest.mark.requires(min_devices=2)
def test_preflight_accepts_a_cohort_shorter_than_the_subject_mesh() -> None:
    devices = tuple(device.id for device in jax.devices()[:2])
    model, params, initial = _inputs(
        budget=2**24, cutoff=-2.0, n_subjects=1, devices=devices
    )
    result = model.simulate(
        params=params, initial_conditions=initial, log_level="debug"
    )
    np.testing.assert_allclose(result.raw_results["alive"][0].V_arr, [2.0])
