"""Simulation population belongs to each call's initial conditions."""

import inspect
from typing import Any

import cloudpickle
import jax.numpy as jnp
import jax.stages
import pandas as pd
import pytest
from jax import Array

from lcm import AgeGrid, Model
from lcm.exceptions import InvalidInitialConditionsError, InvalidSimulationInputError
from tests.simulation.test_process_grid_entry_admission import _inputs
from tests.test_models.deterministic.regression import (
    RegimeId,
    dead,
    get_params,
    working_life,
)


def test_model_constructor_has_no_population_parameter() -> None:
    """A model's structural constructor does not declare a simulation population."""
    assert "n_subjects" not in inspect.signature(Model).parameters


def _model() -> Model:
    """Build a small three-period model with one acting regime."""
    return Model(
        regimes={
            "working_life": working_life.replace(active=lambda age: age <= 19),
            "dead": dead,
        },
        ages=AgeGrid(start=18, stop=20, step="Y"),
        regime_id_class=RegimeId,
    )


def _initial(*, count: int) -> dict[str, Array]:
    return {
        "wealth": jnp.linspace(20.0, 320.0, num=count),
        "age": jnp.full(count, 18.0),
        "regime_id": jnp.full(count, RegimeId.working_life, dtype=jnp.int32),
    }


@pytest.mark.parametrize("population", [None, 0, 4])
def test_model_constructor_rejects_population_keyword(population: int | None) -> None:
    """Unsupported constructor keywords raise instead of being silently ignored."""
    with pytest.raises(TypeError, match="n_subjects"):
        Model(
            regimes={"working_life": working_life, "dead": dead},
            ages=AgeGrid(start=18, stop=20, step="Y"),
            regime_id_class=RegimeId,
            n_subjects=population,  # ty: ignore[unknown-argument]
        )


@pytest.mark.parametrize("counts", [(1, 4, 2, 4), (4, 2, 4, 1)])
def test_one_model_accepts_different_call_time_populations(
    *, counts: tuple[int, ...]
) -> None:
    """Both shape orders preserve complete trajectories against fresh models."""
    model = _model()
    params = get_params(n_periods=3)
    solution = model.solve(params=params, log_level="debug")
    for count in counts:
        initial = _initial(count=count)
        result = model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            log_level="debug",
            seed=0,
        )
        frame = result.to_dataframe(use_labels=False)
        expected = (
            _model()
            .simulate(
                params=params, initial_conditions=initial, log_level="debug", seed=0
            )
            .to_dataframe(use_labels=False)
        )
        assert result.n_subjects == count
        assert frame.loc[frame["period"] == 0, "wealth"].tolist() == (
            initial["wealth"].tolist()
        )
        pd.testing.assert_frame_equal(frame, expected, check_exact=True)


def test_repeated_simulation_reuses_compilation_and_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm shape consumes changed rows without recompiling or reusing old data."""
    model = _model()
    params = get_params(n_periods=3)
    solution = model.solve(params=params, log_level="debug")
    initial = _initial(count=4)
    first = model.simulate(
        params=params, solution=solution, initial_conditions=initial, log_level="debug"
    ).to_dataframe()
    changed = {**initial, "wealth": jnp.asarray([120.0, 40.0, 280.0, 80.0])}
    expected = (
        _model()
        .simulate(params=params, initial_conditions=changed, log_level="debug")
        .to_dataframe()
    )
    assert not first.equals(expected)
    compilations: list[None] = []
    original = jax.stages.Lowered.compile

    def observe(self: jax.stages.Lowered, *args: Any, **kwargs: Any) -> Any:
        compilations.append(None)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Lowered, "compile", observe)
    second = model.simulate(
        params=params, solution=solution, initial_conditions=changed, log_level="debug"
    ).to_dataframe()
    pd.testing.assert_frame_equal(expected, second, check_exact=True)
    repeated = model.simulate(
        params=params, solution=solution, initial_conditions=initial, log_level="debug"
    ).to_dataframe()
    pd.testing.assert_frame_equal(first, repeated, check_exact=True)
    assert compilations == []


def test_warm_simulation_revalidates_changed_process_support() -> None:
    """A cached forward executor cannot make a stale solution support admissible."""
    model, params, initial = _inputs(budget=2**28)
    solution = model.solve(params=params, log_level="debug")
    first = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        log_level="debug",
        seed=0,
    ).to_dataframe()
    alive = params["alive"]
    assert isinstance(alive, dict)
    changed = {
        **params,
        "alive": {**alive, "income": {"start": 1.0, "stop": 4.0}},
    }
    with pytest.raises(InvalidSimulationInputError, match="params_fingerprint"):
        model.simulate(
            params=changed,
            solution=solution,
            initial_conditions=initial,
            log_level="debug",
        )

    recovered = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        log_level="debug",
        seed=0,
    ).to_dataframe()
    pd.testing.assert_frame_equal(first, recovered, check_exact=True)


def test_warm_simulation_rejects_invalid_rows_before_dispatch_and_recovers(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached shapes cannot bypass validation or poison the next valid call."""
    model = _model()
    params = get_params(n_periods=3)
    solution = model.solve(params=params, log_level="debug")
    initial = _initial(count=4)
    first = model.simulate(
        params=params, solution=solution, initial_conditions=initial, log_level="debug"
    ).to_dataframe()
    invalid = {**initial, "regime_id": jnp.full(4, 999, dtype=jnp.int32)}
    assert invalid["regime_id"].tolist() == [999] * 4

    def refuse_runtime_selection(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("Invalid population reached forward runtime selection")

    with monkeypatch.context() as patch:
        patch.setattr(model, "_runtime_regimes_for_shape", refuse_runtime_selection)
        with pytest.raises(InvalidInitialConditionsError, match="Invalid regime IDs"):
            model.simulate(
                params=params,
                solution=solution,
                initial_conditions=invalid,
                log_level="debug",
            )
    recovered = model.simulate(
        params=params, solution=solution, initial_conditions=initial, log_level="debug"
    ).to_dataframe()
    pd.testing.assert_frame_equal(first, recovered, check_exact=True)


def test_runtime_model_and_result_round_trip_through_pickle() -> None:
    """Serialized models rebuild runtime owners and reproduce the complete result."""
    model = _model()
    params = get_params(n_periods=3)
    first = model.simulate(
        params=params, initial_conditions=_initial(count=4), log_level="debug"
    )
    restored_model = cloudpickle.loads(cloudpickle.dumps(model))
    restored_result = cloudpickle.loads(cloudpickle.dumps(first))
    second = restored_model.simulate(
        params=params, initial_conditions=_initial(count=4), log_level="debug"
    )
    pd.testing.assert_frame_equal(first.to_dataframe(), restored_result.to_dataframe())
    pd.testing.assert_frame_equal(first.to_dataframe(), second.to_dataframe())
