"""Regime-transition validation admits its complete probability producer."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from _lcm import transition_checks
from _lcm.dtypes import canonical_float_dtype
from lcm import AgeGrid, ExecutionConfig, MarkovTransition, Model, Regime, categorical
from lcm.exceptions import ExecutionPlanningError
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


def _utility() -> ScalarFloat:
    return jnp.asarray(0, dtype=_FLOAT_DTYPE)


def _active_alive(age: float) -> bool:
    return age == 0


def _active_done(age: float) -> bool:
    return age == 1


def _invalid_costly_regime_probabilities() -> FloatND:
    """Return invalid mass after a visible sort workspace completes."""
    sample = jnp.sin(jnp.arange(4096, dtype=_FLOAT_DTYPE))
    probability = _FLOAT_DTYPE(0.25) + _FLOAT_DTYPE(0) * jnp.sort(sample)[2048]
    return jnp.stack((probability, probability))


def _valid_regime_probabilities() -> FloatND:
    return jnp.asarray([0, 1], dtype=_FLOAT_DTYPE)


def _inputs(
    *, budget: int, valid: bool = False
) -> tuple[Model, UserParams, UserInitialConditions]:
    probabilities = (
        _valid_regime_probabilities if valid else _invalid_costly_regime_probabilities
    )
    model = Model(
        regimes={
            "alive": Regime(
                transition=MarkovTransition(probabilities),
                active=_active_alive,
                functions={"utility": _utility},
            ),
            "done": Regime(
                transition=None,
                active=_active_done,
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
            "age": jnp.zeros(1),
            "regime_id": jnp.asarray([_RegimeId.alive]),
        },
    )


@dataclasses.dataclass
class _CompilerBoundary:
    profiled: list[tuple[jax.stages.Compiled, int]] = dataclasses.field(
        default_factory=list
    )
    """Costly producer profiles paired with dispatch count at profile time."""

    dispatched: list[jax.stages.Compiled] = dataclasses.field(default_factory=list)
    """Executables that crossed the device dispatch boundary."""

    def require_declined_regime_law(self) -> None:
        """Require one sort producer to be profiled and never dispatched."""
        profiles = [
            item for item in self.profiled if "sort" in (item[0].as_text() or "")
        ]
        assert len(profiles) == 1
        declined, offset = profiles[0]
        assert all(declined is not item for item in self.dispatched[offset:])


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
def test_admitted_regime_probability_pytree_completes() -> None:
    """A successful budgeted regime law completes every output leaf."""
    model, params, initial = _inputs(budget=2**28, valid=True)
    solution = model.solve(params=params, log_level="off")

    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="debug",
    )

    assert result.n_subjects == 1


@pytest.mark.requires(device="cpu")
def test_regime_law_workspace_refuses_before_completed_user_output(
    *,
    compiler_boundary: _CompilerBoundary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full Cartesian law must fit before its first completed output."""
    model, params, initial = _inputs(budget=16 * 1024)
    completed: list[object] = []
    original_check = transition_checks._validate_regime_transition_probs

    def observe_completed_law(**kwargs: Any) -> None:
        jax.block_until_ready(kwargs["regime_transition_probs"])
        completed.append(kwargs["regime_transition_probs"])
        original_check(**kwargs)

    monkeypatch.setattr(
        transition_checks, "_validate_regime_transition_probs", observe_completed_law
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
    compiler_boundary.require_declined_regime_law()
