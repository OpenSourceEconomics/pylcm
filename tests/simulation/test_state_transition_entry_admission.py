"""State-transition validation admits its complete user-law producer."""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.dtypes import canonical_float_dtype
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidStateTransitionProbabilitiesError,
)
from lcm.persistence import load_solution
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
    alive: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class _Health:
    bad: ScalarInt
    good: ScalarInt


def _utility(*, health: ScalarInt) -> ScalarFloat:
    return health.astype(_FLOAT_DTYPE)


def _next_regime() -> ScalarInt:
    return _RegimeId.done


def _alive(age: float) -> bool:
    return age == 0


def _done(age: float) -> bool:
    return age == 1


def _costly_probability(*, health: ScalarInt) -> ScalarFloat:
    """Require visible sorting workspace without changing the probabilities."""
    sample = jnp.sin(jnp.arange(4096, dtype=_FLOAT_DTYPE) + health)
    return _FLOAT_DTYPE(0.25) + _FLOAT_DTYPE(0) * jnp.sort(sample)[2048]


def _valid_health_probabilities(*, health: ScalarInt) -> FloatND:
    probability = _costly_probability(health=health)
    return jnp.stack((probability, 1 - probability))


def _invalid_health_probabilities(*, health: ScalarInt) -> FloatND:
    probability = _costly_probability(health=health)
    return jnp.stack((probability, probability))


def _inputs(
    *, budget: int | None, valid: bool = True
) -> tuple[Model, UserParams, UserInitialConditions]:
    law: Callable[..., FloatND] = (
        _valid_health_probabilities if valid else _invalid_health_probabilities
    )
    grid = DiscreteGrid(category_class=_Health)
    model = Model(
        regimes={
            "alive": Regime(
                transition=_next_regime,
                active=_alive,
                states={"health": grid},
                state_transitions={"health": MarkovTransition(law)},
                functions={"utility": _utility},
            ),
            "done": Regime(
                transition=None,
                active=_done,
                states={"health": grid},
                functions={"utility": _utility},
            ),
        },
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        execution_config=ExecutionConfig(device_memory_bytes=budget),
    )
    return (
        model,
        {"alive": {"koopmans_aggregator": {"discount_factor": 0.9}}, "done": {}},
        {
            "health": jnp.asarray([_Health.bad, _Health.good]),
            "age": jnp.zeros(2),
            "regime_id": jnp.asarray([_RegimeId.alive, _RegimeId.alive]),
        },
    )


@dataclasses.dataclass
class _CompilerBoundary:
    profiled: list[tuple[jax.stages.Compiled, int]] = dataclasses.field(
        default_factory=list
    )
    dispatched: list[jax.stages.Compiled] = dataclasses.field(default_factory=list)

    def transition_profiles(self) -> list[tuple[jax.stages.Compiled, int]]:
        return [item for item in self.profiled if "sort" in (item[0].as_text() or "")]


@pytest.fixture
def compiler_boundary(monkeypatch: pytest.MonkeyPatch) -> _CompilerBoundary:
    """Observe explicit compiler accounting and dispatch in their actual order."""
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


@pytest.mark.requires(device="cpu")
def test_state_transition_workspace_refuses_before_user_law_dispatch(
    *,
    compiler_boundary: _CompilerBoundary,
    tmp_path: Path,
) -> None:
    """A small probability output cannot hide its sorting workspace."""
    producer, params, initial = _inputs(budget=None)
    path = producer.solve(params=params, log_level="off").save(
        path=tmp_path / "solution"
    )
    solution = load_solution(path=path)
    consumer, _, _ = _inputs(budget=16 * 1024)

    with pytest.raises(ExecutionPlanningError, match="compiler peak"):
        consumer.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="debug",
        )

    profiles = compiler_boundary.transition_profiles()
    assert len(profiles) == 1
    declined, offset = profiles[0]
    assert all(
        declined is not executed for executed in compiler_boundary.dispatched[offset:]
    )


@pytest.mark.requires(device="cpu")
def test_invalid_state_transition_replay_uses_admitted_producer(
    compiler_boundary: _CompilerBoundary,
) -> None:
    """The summary and ordered diagnostic replay each admit the user law."""
    oracle, params, initial = _inputs(budget=None, valid=False)
    with pytest.raises(InvalidStateTransitionProbabilitiesError) as expected:
        oracle.simulate(params=params, initial_conditions=initial, log_level="debug")

    model, _, _ = _inputs(budget=2**28, valid=False)
    with pytest.raises(InvalidStateTransitionProbabilitiesError) as actual:
        model.simulate(params=params, initial_conditions=initial, log_level="debug")

    assert str(actual.value) == str(expected.value)
    profiles = compiler_boundary.transition_profiles()
    assert len(profiles) == 2
    for compiled, offset in profiles:
        assert any(
            compiled is executed for executed in compiler_boundary.dispatched[offset:]
        )


def test_admitted_state_transition_preserves_seeded_simulation_and_inputs() -> None:
    """Budgeted validation leaves transition draws and caller arrays unchanged."""
    baseline, params, initial = _inputs(budget=None)
    budgeted, _, _ = _inputs(budget=2**28)
    snapshots = {name: np.array(value) for name, value in initial.items()}

    baseline_result = baseline.simulate(
        params=params, initial_conditions=initial, log_level="debug", seed=17
    )
    budgeted_result = budgeted.simulate(
        params=params, initial_conditions=initial, log_level="debug", seed=17
    )

    np.testing.assert_array_equal(
        budgeted_result.to_dataframe(use_labels=False).to_numpy(),
        baseline_result.to_dataframe(use_labels=False).to_numpy(),
    )
    for name, snapshot in snapshots.items():
        np.testing.assert_array_equal(initial[name], snapshot)
