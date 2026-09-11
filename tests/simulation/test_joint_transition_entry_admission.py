"""Joint transition preflight admits weight and support producers."""

import dataclasses
import os
import subprocess
import sys
import textwrap
import weakref
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import transition_checks
from _lcm.dtypes import canonical_float_dtype
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import (
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from lcm import (
    AgeGrid,
    ExecutionConfig,
    JointTransition,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidStateTransitionProbabilitiesError,
    RegimeInitializationError,
)
from lcm.typing import (
    FloatND,
    ScalarFloat,
    ScalarInt,
    UserInitialConditions,
    UserParams,
)

_FLOAT_DTYPE = canonical_float_dtype()


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt


def _utility() -> ScalarFloat:
    return jnp.asarray(0, dtype=_FLOAT_DTYPE)


def _target_utility(*, wealth: ScalarFloat, income: ScalarFloat) -> ScalarFloat:
    return wealth + income


def _active_source(age: float) -> bool:
    return age == 0


def _active_target(age: float) -> bool:
    return age == 1


def _certain_target() -> FloatND:
    return jnp.asarray(1, dtype=_FLOAT_DTYPE)


def _joint_probabilities() -> FloatND:
    return jnp.asarray([0.5, 0.5], dtype=_FLOAT_DTYPE)


def _invalid_joint_probabilities() -> FloatND:
    return jnp.asarray([0.25, 0.25], dtype=_FLOAT_DTYPE)


def _costly_joint_probabilities() -> FloatND:
    sample = jnp.sin(jnp.arange(4096, dtype=_FLOAT_DTYPE))
    probability = _FLOAT_DTYPE(0.5) + _FLOAT_DTYPE(0) * jnp.sort(sample)[2048]
    return jnp.stack((probability, probability))


def _joint_support() -> Mapping[str, FloatND]:
    return {
        "wealth": jnp.asarray([0, 1], dtype=_FLOAT_DTYPE),
        "income": jnp.asarray([0, 1], dtype=_FLOAT_DTYPE),
    }


def _costly_joint_support() -> Mapping[str, FloatND]:
    sample = jnp.cos(jnp.arange(4096, dtype=_FLOAT_DTYPE))
    shift = _FLOAT_DTYPE(0) * jnp.sort(sample)[2048]
    return {
        "wealth": jnp.asarray([0, 1], dtype=_FLOAT_DTYPE) + shift,
        "income": jnp.asarray([0, 1], dtype=_FLOAT_DTYPE) + shift,
    }


def _wrong_sized_joint_support() -> Mapping[str, FloatND]:
    return {
        "wealth": jnp.asarray([0, 0.5, 1], dtype=_FLOAT_DTYPE),
        "income": jnp.asarray([0, 0.5, 1], dtype=_FLOAT_DTYPE),
    }


def _next_wealth(match: Mapping[str, FloatND]) -> FloatND:
    return match["wealth"]


def _next_income(match: Mapping[str, FloatND]) -> FloatND:
    return match["income"]


def _inputs(
    *,
    probabilities: Callable[[], FloatND],
    support: Callable[[], Mapping[str, FloatND]],
    budget: int | None,
    devices: tuple[int, ...] | None = None,
) -> tuple[Model, UserParams, UserInitialConditions]:
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_certain_target)},
                active=_active_source,
                functions={"utility": _utility},
                joint_transitions={
                    "target": {
                        "match": JointTransition(
                            support_size=2,
                            support=support,
                            probabilities=probabilities,
                            outputs={
                                "wealth": _next_wealth,
                                "income": _next_income,
                            },
                        )
                    }
                },
            ),
            "target": Regime(
                transition=None,
                active=_active_target,
                states={
                    "wealth": LinSpacedGrid(start=0, stop=1, n_points=2),
                    "income": LinSpacedGrid(start=0, stop=1, n_points=2),
                },
                functions={"utility": _target_utility},
            ),
        },
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        execution_config=ExecutionConfig(device_memory_bytes=budget, devices=devices),
    )
    params: UserParams = {
        "source": {
            "target": {
                "next_regime": {},
                "match": {"support": {}, "probabilities": {}},
                "next_wealth": {},
                "next_income": {},
            },
            "koopmans_aggregator": {"discount_factor": 0.9},
        },
        "target": {"utility": {}},
    }
    initial: UserInitialConditions = {
        "age": jnp.zeros(1),
        "regime_id": jnp.asarray([_RegimeId.source]),
    }
    return model, params, initial


_SELECTED_DEVICE_SCRIPT = textwrap.dedent(
    """
    import jax

    from _lcm import transition_checks
    from lcm.exceptions import InvalidStateTransitionProbabilitiesError
    from tests.simulation.test_joint_transition_entry_admission import (
        _inputs,
        _invalid_joint_probabilities,
        _joint_support,
    )

    assert jax.device_count() == 4, jax.devices()
    selected_id = jax.devices()[2].id
    model, params, initial = _inputs(
        probabilities=_invalid_joint_probabilities,
        support=_joint_support,
        budget=2**28,
        devices=(selected_id,),
    )
    solution = model.solve(params=params, log_level="off")

    compiled = []
    weight_devices = []
    support_devices = []
    original_compile = transition_checks._TransitionLawCompiler.__call__
    original_weights = transition_checks._evaluate_joint_weights
    original_support = transition_checks._evaluate_joint_support

    def compile_and_record(self, widths):
        executable = original_compile(self, widths)
        compiled.append(executable)
        return executable

    def weights_and_record(**kwargs):
        evaluated = original_weights(**kwargs)
        if evaluated is not None:
            weight_devices.append(
                [tuple(device.id for device in leaf.devices())
                 for leaf in evaluated[0].values()]
            )
        return evaluated

    def support_and_record(**kwargs):
        support = original_support(**kwargs)
        if support is not None:
            support_devices.append(
                [tuple(device.id for device in leaf.devices())
                 for leaf in jax.tree.leaves(support)]
            )
        return support

    transition_checks._TransitionLawCompiler.__call__ = compile_and_record
    transition_checks._evaluate_joint_weights = weights_and_record
    transition_checks._evaluate_joint_support = support_and_record
    try:
        model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="debug",
        )
    except InvalidStateTransitionProbabilitiesError:
        pass
    else:
        raise AssertionError("The invalid joint weights skipped serial replay.")

    expected_weights = [[(selected_id,)]] * 3
    expected_supports = [
        [(selected_id,), (selected_id,)],
        [(selected_id,), (selected_id,)],
        [(selected_id,), (selected_id,)],
    ]
    output_devices = [
        tuple(sorted(device.id for device in sharding.device_set))
        for executable in compiled
        for sharding in jax.tree.leaves(executable.output_shardings)
    ]
    # The shared compiler also sees the valid regime producer in summary and serial.
    assert (weight_devices, support_devices, output_devices) == (
        expected_weights,
        expected_supports,
        [(selected_id,)] * 11,
    ), (weight_devices, support_devices, output_devices)
    print("JOINT-PRODUCER-PLACEMENT-OK")
    """
)


@dataclasses.dataclass
class _CompilerBoundary:
    profiled: list[tuple[jax.stages.Compiled, int]] = dataclasses.field(
        default_factory=list
    )
    """Costly producer profiles paired with dispatch count at profile time."""

    dispatched: list[jax.stages.Compiled] = dataclasses.field(default_factory=list)
    """Executables that crossed the device dispatch boundary."""

    def require_declined_sort_producer(self) -> None:
        """Require one sort producer to be profiled and never dispatched."""
        profiles = [
            item for item in self.profiled if "sort" in (item[0].as_text() or "")
        ]
        declined = profiles[0] if len(profiles) == 1 else None
        preserved = declined is not None and all(
            declined[0] is not item for item in self.dispatched[declined[1] :]
        )
        assert (len(profiles), preserved) == (1, True)


@pytest.fixture
def compiler_boundary(monkeypatch: pytest.MonkeyPatch) -> _CompilerBoundary:
    """Observe compiler accounting and dispatch without changing either result."""
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


def _controlled_post_validation_refusal(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise ExecutionPlanningError("controlled refusal after transition validation")


@pytest.mark.requires(device="cpu")
def test_joint_weight_workspace_refuses_before_completed_user_output(
    *, compiler_boundary: _CompilerBoundary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The complete weight mapping must fit before its first completed output."""
    model, params, initial = _inputs(
        probabilities=_costly_joint_probabilities,
        support=_joint_support,
        budget=16 * 1024,
    )
    completed: list[object] = []
    original_evaluate = transition_checks._evaluate_joint_weights

    def observe_completed_weights(**kwargs: Any) -> Any:
        evaluated = original_evaluate(**kwargs)
        if evaluated is not None:
            jax.block_until_ready(evaluated[0])
            completed.append(evaluated[0])
        return evaluated

    monkeypatch.setattr(
        transition_checks, "_evaluate_joint_weights", observe_completed_weights
    )
    monkeypatch.setattr(
        Model, "_solve_from_flat_params", _controlled_post_validation_refusal
    )

    with pytest.raises(ExecutionPlanningError) as error:
        model.simulate(
            params=params,
            initial_conditions=initial,
            log_level="warning",
        )

    assert (completed, "controlled refusal" in str(error.value)) == ([], False)
    compiler_boundary.require_declined_sort_producer()


@pytest.mark.requires(device="cpu")
def test_joint_support_workspace_refuses_before_completed_user_output(
    *, compiler_boundary: _CompilerBoundary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The complete support pytree must fit before its first completed output."""
    model, params, initial = _inputs(
        probabilities=_joint_probabilities,
        support=_costly_joint_support,
        budget=16 * 1024,
    )
    completed: list[object] = []
    original_evaluate = transition_checks._evaluate_joint_support

    def observe_completed_support(**kwargs: Any) -> Any:
        support = original_evaluate(**kwargs)
        if support is not None:
            jax.block_until_ready(support)
            completed.append(support)
        return support

    monkeypatch.setattr(
        transition_checks, "_evaluate_joint_support", observe_completed_support
    )
    monkeypatch.setattr(
        Model, "_solve_from_flat_params", _controlled_post_validation_refusal
    )

    with pytest.raises(ExecutionPlanningError) as error:
        model.simulate(
            params=params,
            initial_conditions=initial,
            log_level="warning",
        )

    assert (completed, "controlled refusal" in str(error.value)) == ([], False)
    compiler_boundary.require_declined_sort_producer()


@pytest.mark.requires(device="cpu")
def test_joint_producers_use_selected_device_during_serial_replay() -> None:
    """Summary and serial joint outputs use the selected nondefault device."""
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    }
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SELECTED_DEVICE_SCRIPT],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        timeout=600,
    )

    assert (result.returncode, "JOINT-PRODUCER-PLACEMENT-OK" in result.stdout) == (
        0,
        True,
    ), result.stderr[-4000:]


def test_admitted_joint_producers_preserve_seeded_simulation_and_inputs() -> None:
    """Budgeted preflight leaves joint draws and caller arrays unchanged."""
    baseline, params, initial = _inputs(
        probabilities=_joint_probabilities,
        support=_joint_support,
        budget=None,
    )
    budgeted, _, _ = _inputs(
        probabilities=_joint_probabilities,
        support=_joint_support,
        budget=2**28,
    )
    snapshots = {name: np.array(value) for name, value in initial.items()}

    baseline_result = baseline.simulate(
        params=params, initial_conditions=initial, log_level="debug", seed=17
    )
    budgeted_result = budgeted.simulate(
        params=params, initial_conditions=initial, log_level="debug", seed=17
    )

    assert budgeted_result.to_dataframe(use_labels=False).equals(
        baseline_result.to_dataframe(use_labels=False)
    )
    for name, snapshot in snapshots.items():
        np.testing.assert_array_equal(initial[name], snapshot)


def test_joint_mapping_owners_expose_all_array_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weight mappings stay visible beside both support leaves."""
    observed_leaf_counts: list[int] = []
    original_set_derived = SimulationMemory.set_derived

    # keyword-only-exempt: library-callback=SimulationMemory.set_derived
    def set_derived_and_record(self: SimulationMemory, tree: object) -> None:
        leaves = [leaf for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)]
        if leaves:
            observed_leaf_counts.append(len(leaves))
        original_set_derived(self, tree)

    monkeypatch.setattr(SimulationMemory, "set_derived", set_derived_and_record)
    model, params, initial = _inputs(
        probabilities=_joint_probabilities,
        support=_joint_support,
        budget=2**28,
    )
    solution = model.solve(params=params, log_level="off")

    model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="debug",
    )

    assert 3 in observed_leaf_counts


def _released_or_charged(
    *, references: list[weakref.ReferenceType[jax.Array]], memory: SimulationMemory
) -> bool:
    """Accept a prior output only when it is gone or in the next live inventory."""
    live = [value for reference in references if (value := reference()) is not None]
    if not live:
        return True
    missing = resident_bytes_by_device(
        live=measure_buffer_footprint(tree=live),
        arguments=memory.snapshot(),
        devices=memory.devices,
    )
    return not any(missing.values())


def test_joint_support_is_charged_through_probability_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A support local remains visible when its probability check is admitted."""
    references: list[weakref.ReferenceType[jax.Array]] = []
    observations: list[bool] = []
    original_support = transition_checks._evaluate_joint_support
    original_operation = transition_checks.run_simulation_operation

    def support_and_record(**kwargs: Any) -> Any:
        support = original_support(**kwargs)
        if support is not None:
            references[:] = [weakref.ref(leaf) for leaf in support.values()]
        return support

    def operation_and_check(**kwargs: Any) -> Any:
        if kwargs["function"] is transition_checks._joint_probability_flags:
            observations.append(
                _released_or_charged(
                    references=references,
                    memory=kwargs["memory"],
                )
            )
        return original_operation(**kwargs)

    monkeypatch.setattr(
        transition_checks, "_evaluate_joint_support", support_and_record
    )
    monkeypatch.setattr(
        transition_checks, "run_simulation_operation", operation_and_check
    )
    model, params, initial = _inputs(
        probabilities=_joint_probabilities,
        support=_joint_support,
        budget=2**28,
    )
    solution = model.solve(params=params, log_level="off")

    model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="debug",
    )

    assert (bool(observations), all(observations)) == (True, True)


def test_previous_joint_weights_are_released_before_next_weight_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed weight mapping is gone before the next weight producer."""
    references: list[weakref.ReferenceType[jax.Array]] = []
    observations: list[bool] = []
    original_weights = transition_checks._evaluate_joint_weights

    def weights_and_check(**kwargs: Any) -> Any:
        if references:
            observations.append(all(reference() is None for reference in references))
        evaluated = original_weights(**kwargs)
        if evaluated is not None:
            references[:] = [weakref.ref(leaf) for leaf in evaluated[0].values()]
        return evaluated

    monkeypatch.setattr(transition_checks, "_evaluate_joint_weights", weights_and_check)
    model, params, initial = _inputs(
        probabilities=_joint_probabilities,
        support=_joint_support,
        budget=2**28,
    )
    solution = model.solve(params=params, log_level="off")

    model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="debug",
    )

    assert (bool(observations), all(observations)) == (True, True)


def test_previous_joint_support_is_released_before_next_support_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed support has no full-array owner at the next support producer."""
    references: list[weakref.ReferenceType[jax.Array]] = []
    observations: list[bool] = []
    original_support = transition_checks._evaluate_joint_support

    def support_and_check(**kwargs: Any) -> Any:
        if references:
            observations.append(all(reference() is None for reference in references))
        support = original_support(**kwargs)
        if support is not None:
            references[:] = [weakref.ref(leaf) for leaf in support.values()]
        return support

    monkeypatch.setattr(transition_checks, "_evaluate_joint_support", support_and_check)
    model, params, initial = _inputs(
        probabilities=_joint_probabilities,
        support=_joint_support,
        budget=2**28,
    )
    solution = model.solve(params=params, log_level="off")

    model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="debug",
    )

    assert (bool(observations), all(observations)) == (True, True)


def test_joint_owner_contexts_restore_memory_after_probability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probability-check exception releases every scoped derived owner."""
    memories: list[SimulationMemory] = []
    original_set_derived = SimulationMemory.set_derived

    # keyword-only-exempt: library-callback=SimulationMemory.set_derived
    def set_derived_and_keep(self: SimulationMemory, tree: object) -> None:
        if all(self is not memory for memory in memories):
            memories.append(self)
        original_set_derived(self, tree)

    def raise_controlled_probability_error(**kwargs: Any) -> None:
        del kwargs
        raise RuntimeError("controlled joint probability failure")

    monkeypatch.setattr(SimulationMemory, "set_derived", set_derived_and_keep)
    monkeypatch.setattr(
        transition_checks,
        "_validate_joint_probabilities",
        raise_controlled_probability_error,
    )
    model, params, initial = _inputs(
        probabilities=_joint_probabilities,
        support=_joint_support,
        budget=2**28,
    )
    solution = model.solve(params=params, log_level="off")

    with pytest.raises(RuntimeError, match="controlled joint probability failure"):
        model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="debug",
        )

    assert (bool(memories), all(memory.derived == () for memory in memories)) == (
        True,
        True,
    )


def test_admitted_joint_weight_diagnostic_matches_unbudgeted_path() -> None:
    """Admission preserves the ordered invalid-weight diagnostic."""
    oracle, params, initial = _inputs(
        probabilities=_invalid_joint_probabilities,
        support=_joint_support,
        budget=None,
    )
    oracle_solution = oracle.solve(params=params, log_level="off")
    with pytest.raises(InvalidStateTransitionProbabilitiesError) as expected:
        oracle.simulate(
            params=params,
            initial_conditions=initial,
            solution=oracle_solution,
            log_level="debug",
        )

    budgeted, _, _ = _inputs(
        probabilities=_invalid_joint_probabilities,
        support=_joint_support,
        budget=2**28,
    )
    budgeted_solution = budgeted.solve(params=params, log_level="off")
    with pytest.raises(InvalidStateTransitionProbabilitiesError) as actual:
        budgeted.simulate(
            params=params,
            initial_conditions=initial,
            solution=budgeted_solution,
            log_level="debug",
        )

    assert str(actual.value) == str(expected.value)


def test_admitted_joint_support_diagnostic_matches_unbudgeted_path() -> None:
    """Admission preserves the ordered invalid-support diagnostic."""
    oracle, params, initial = _inputs(
        probabilities=_joint_probabilities,
        support=_wrong_sized_joint_support,
        budget=None,
    )
    oracle_solution = oracle.solve(params=params, log_level="off")
    with pytest.raises(RegimeInitializationError) as expected:
        oracle.simulate(
            params=params,
            initial_conditions=initial,
            solution=oracle_solution,
            log_level="debug",
        )

    budgeted, _, _ = _inputs(
        probabilities=_joint_probabilities,
        support=_wrong_sized_joint_support,
        budget=2**28,
    )
    budgeted_solution = budgeted.solve(params=params, log_level="off")
    with pytest.raises(RegimeInitializationError) as actual:
        budgeted.simulate(
            params=params,
            initial_conditions=initial,
            solution=budgeted_solution,
            log_level="debug",
        )

    assert str(actual.value) == str(expected.value)
