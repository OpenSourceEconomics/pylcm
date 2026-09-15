"""Regime-transition validation admits its complete probability producer."""

import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import transition_checks
from _lcm.dtypes import canonical_float_dtype
from lcm import AgeGrid, ExecutionConfig, MarkovTransition, Model, Regime, categorical
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidRegimeTransitionProbabilitiesError,
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


def _alive_payoff() -> ScalarFloat:
    return jnp.asarray(2, dtype=_FLOAT_DTYPE)


def _done_payoff() -> ScalarFloat:
    return jnp.asarray(6, dtype=_FLOAT_DTYPE)


def _parameterized_regime_probabilities(done_probability: float) -> FloatND:
    return jnp.stack((jnp.asarray(0, dtype=_FLOAT_DTYPE), done_probability))


def _numerical_inputs(
    *, budget: int | None
) -> tuple[Model, UserParams, UserInitialConditions]:
    """A two-period oracle: V_alive=2+0.5*6=5 and V_done=6."""
    model = Model(
        regimes={
            "alive": Regime(
                transition=MarkovTransition(_parameterized_regime_probabilities),
                active=_active_alive,
                functions={"utility": _alive_payoff},
            ),
            "done": Regime(
                transition=None,
                active=_active_done,
                functions={"utility": _done_payoff},
            ),
        },
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        execution_config=ExecutionConfig(device_memory_bytes=budget),
    )
    return (
        model,
        {
            "alive": {
                "koopmans_aggregator": {"discount_factor": 0.5},
                "next_regime": {"done_probability": 1.0},
            },
            "done": {},
        },
        {
            "age": jnp.zeros(3),
            "regime_id": jnp.full(3, _RegimeId.alive),
        },
    )


def _assert_same_raw_results(*, actual: Any, expected: Any) -> None:
    """Compare the complete public record, including routes and masks."""
    assert jax.tree.structure(actual.raw_results) == jax.tree.structure(
        expected.raw_results
    )
    for got, want in zip(
        jax.tree.leaves(actual.raw_results),
        jax.tree.leaves(expected.raw_results),
        strict=True,
    ):
        np.testing.assert_array_equal(got, want)


@pytest.mark.requires(device="cpu")
def test_admitted_regime_probability_pytree_completes() -> None:
    """Admission preserves analytical values, destinations and caller arrays."""
    model, params, initial = _numerical_inputs(budget=2**28)
    oracle, _, _ = _numerical_inputs(budget=None)
    snapshots = {name: np.array(value) for name, value in initial.items()}
    solution = model.solve(params=params, log_level="off")
    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=17,
        log_level="debug",
    )
    expected = oracle.simulate(
        params=params, initial_conditions=initial, seed=17, log_level="debug"
    )
    assert result.n_subjects == expected.n_subjects == 3
    assert set(result.raw_results) == {"alive", "done"}
    assert set(result.raw_results["alive"]) == {0}
    assert set(result.raw_results["done"]) == {1}
    for regime, period, value in (("alive", 0, 5), ("done", 1, 6)):
        data = result.raw_results[regime][period]
        np.testing.assert_array_equal(data.V_arr, np.full(3, value))
        np.testing.assert_array_equal(data.in_regime, np.ones(3, dtype=bool))
    _assert_same_raw_results(actual=result, expected=expected)
    for name, value in initial.items():
        assert isinstance(value, jax.Array)
        assert not value.is_deleted()
        np.testing.assert_array_equal(value, snapshots[name])


@pytest.mark.requires(device="cpu")
def test_invalid_regime_diagnostic_matches_unbudgeted_and_recovers() -> None:
    """A rejected probability law cannot poison the same model's valid reuse."""
    models = [_numerical_inputs(budget=budget) for budget in (None, 2**28)]
    errors = []
    recovered_results = []
    for model, params, initial in models:
        snapshots = {name: np.array(value) for name, value in initial.items()}
        first = model.simulate(
            params=params, initial_conditions=initial, seed=17, log_level="debug"
        )
        invalid = copy.deepcopy(params)
        assert isinstance(invalid, dict)
        alive_params = invalid["alive"]
        assert isinstance(alive_params, dict)
        law_params = alive_params["next_regime"]
        assert isinstance(law_params, dict)
        law_params["done_probability"] = 0.5
        with pytest.raises(InvalidRegimeTransitionProbabilitiesError) as error:
            model.simulate(
                params=invalid,
                initial_conditions=initial,
                seed=17,
                log_level="debug",
            )
        errors.append(str(error.value))
        recovered = model.simulate(
            params=params, initial_conditions=initial, seed=17, log_level="debug"
        )
        _assert_same_raw_results(actual=recovered, expected=first)
        recovered_results.append(recovered)
        for name, value in initial.items():
            assert isinstance(value, jax.Array)
            assert not value.is_deleted()
            np.testing.assert_array_equal(value, snapshots[name])
    assert errors[0] == errors[1]
    assert errors[0]
    _assert_same_raw_results(actual=recovered_results[1], expected=recovered_results[0])


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
