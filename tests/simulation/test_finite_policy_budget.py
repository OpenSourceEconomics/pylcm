"""Finite preparation, its bank and canonical ranking enter real chunk admission."""

from typing import Any

import jax
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from _lcm.simulation.runtime import SimulationRuntime
from lcm import ExecutionConfig, Model
from tests.test_models import n_nbegm_discrete_toy as discrete_toy
from tests.test_models import n_nbegm_toy as smooth_toy


def _inputs(
    *, discrete: bool, budget: int | None, width: int = 1
) -> tuple[Model, dict, dict]:
    """Keep every result owned by its configured public model instance."""
    factory = discrete_toy if discrete else smooth_toy
    base = factory.build_model(variant="n_nbegm", n_periods=2)
    model = Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=smooth_toy.RegimeId,
        fixed_params=base.fixed_params,
        execution_config=ExecutionConfig(
            device_memory_bytes=budget, axis_widths={"subject": width}
        ),
    )
    params = {"discount_factor": 0.95}
    if discrete:
        params["premium"] = 0.25
    initial = {
        "wealth": np.array([4.3, 11.7]),
        "illiquid": np.array([1.37, 6.6]),
        "age": np.array([20.0, 20.0]),
        "regime_id": np.zeros(2, dtype=np.int32),
    }
    return model, params, initial


@pytest.mark.parametrize("discrete", [False, True])
def test_budgeted_finite_policy_profiles_actual_prepare_and_rank(
    *, discrete: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A public admitted replay profiles both stages before their first execution."""
    model, params, initial = _inputs(discrete=discrete, budget=2**32)
    solution = model.solve(params=params, log_level="off")
    prepared: list[str] = []
    dispatched: list[str] = []
    prepare = SimulationRuntime.prepare_abstract
    dispatch = SimulationRuntime.dispatch

    def observe_prepare(self: SimulationRuntime, **call: Any) -> object:
        assert all(
            isinstance(leaf, jax.ShapeDtypeStruct)
            for leaf in jax.tree.leaves(call["arguments"])
        )
        prepared.append(call["program"].name)
        return prepare(self, **call)

    def observe_dispatch(self: SimulationRuntime, **call: Any) -> object:
        name = call["program"].name
        assert name in prepared, "A finite stage ran before its abstract admission"
        dispatched.append(name)
        known = frozenset(self.cache)
        result = dispatch(self, **call)
        assert frozenset(self.cache) == known, "Dispatch compiled an unprofiled stage"
        return result

    monkeypatch.setattr(SimulationRuntime, "prepare_abstract", observe_prepare)
    monkeypatch.setattr(SimulationRuntime, "dispatch", observe_dispatch)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        log_level="off",
        seed=17,
    )
    assert prepared.count("simulate_policy_prepare") >= 1
    assert prepared.count("simulate_policy_rank") >= 1
    assert dispatched.count("simulate_policy_prepare") == 2
    assert dispatched.count("simulate_policy_rank") == 2
    rows = result.to_dataframe().query("regime_name == 'alive' and period == 0")
    assert len(rows) == 2
    assert np.isfinite(rows["value"]).all()


@pytest.mark.parametrize("discrete", [False, True])
def test_admission_preserves_every_public_finite_replay_column(
    *, discrete: bool
) -> None:
    """Budget policy changes ownership and scheduling without changing the frame."""
    frames = []
    for budget in (None, 2**32):
        model, params, initial = _inputs(discrete=discrete, budget=budget)
        solution = model.solve(params=params, log_level="off")
        result = model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            log_level="off",
            seed=17,
        )
        frames.append(result.to_dataframe(use_labels=False))
    assert_frame_equal(frames[0], frames[1], check_exact=True)
