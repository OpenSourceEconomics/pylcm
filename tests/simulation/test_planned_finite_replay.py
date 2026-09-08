"""Finite replay executes declared programs without a Cartesian decision."""

import importlib
from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from _lcm.simulation.runtime import SimulationRuntime
from lcm import ExecutionConfig, Model
from tests.test_models import n_nbegm_discrete_toy as discrete_toy
from tests.test_models import n_nbegm_toy as toy

_replay = importlib.import_module("_lcm.simulation.simulate")


def _legacy_finite_replay(**arguments):
    """Use the unchanged pre-program coordinator as the comparison route."""
    n_subjects = arguments.pop("n_subjects")
    actions, values = _replay._replay_nnbegm_candidates(
        optimal_actions=MappingProxyType({}),
        action_names=arguments["regime"].simulation.action_names,
        **arguments,
    )
    return actions, values, jnp.zeros(n_subjects, dtype=bool)


def test_finite_replay_dispatches_reconstruction_and_ranking_programs(monkeypatch):
    """Observe actual public forward dispatch, not just a solver's metadata."""
    observed = []
    original = SimulationRuntime.dispatch

    # keyword-only-exempt: library-callback=SimulationRuntime.dispatch
    def record(self, *, program, arguments, period, n_subjects):
        observed.append((program.name, period))
        return original(
            self,
            program=program,
            arguments=arguments,
            period=period,
            n_subjects=n_subjects,
        )

    monkeypatch.setattr(SimulationRuntime, "dispatch", record)
    model = toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        execution_config=ExecutionConfig(axis_widths={"subject": 1}),
    )
    result = model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions={
            "wealth": np.array([1.0467, 4.3]),
            "illiquid": np.array([2.04, 1.37]),
            "age": np.array([20.0, 20.0]),
            "regime_id": np.zeros(2, dtype=np.int32),
        },
        log_level="off",
        seed=17,
    )
    rows = result.to_dataframe().query("regime_name == 'alive' and period == 0")
    assert len(rows) == 2
    assert np.isfinite(rows["value"]).all()
    assert ("simulate_policy_prepare", 0) in observed
    assert ("simulate_policy_rank", 0) in observed
    assert ("simulate_decision", 0) not in observed


@pytest.mark.parametrize(
    ("discrete", "prewarm"), [(False, False), (True, False), (False, True)]
)
def test_planned_replay_preserves_public_legacy_frames(
    *, monkeypatch, discrete, prewarm
):
    """Real fixed-parameter, discrete, and prewarmed routes keep legacy choices."""
    factory = discrete_toy if discrete else toy
    base = factory.build_model(variant="n_nbegm", n_periods=2)
    model = Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=toy.RegimeId,
        fixed_params=base.fixed_params,
        execution_config=ExecutionConfig(axis_widths={"subject": 1}),
        n_subjects=2 if prewarm else None,
    )
    params = {"discount_factor": 0.95}
    if discrete:
        params["premium"] = 0.25
    solution = model.solve(params=params, log_level="off")
    initial = {
        "wealth": np.array([4.3, 11.7]),
        "illiquid": np.array([1.37, 6.6]),
        "age": np.array([20.0, 20.0]),
        "regime_id": np.zeros(2, dtype=np.int32),
    }
    simulate = partial(
        model.simulate,
        params=params,
        solution=solution,
        initial_conditions=initial,
        log_level="off",
        seed=17,
    )
    planned = simulate().to_dataframe(use_labels=False)
    monkeypatch.setattr(_replay, "_execute_finite_replay", _legacy_finite_replay)
    legacy = simulate().to_dataframe(use_labels=False)
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    tolerance = float(8 * np.finfo(dtype).eps)
    assert_frame_equal(planned, legacy, rtol=tolerance, atol=tolerance)
