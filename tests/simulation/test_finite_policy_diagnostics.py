"""Independent mask counts and diagnostic ordering in actual finite dispatch."""

from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.policy_diagnostics import dropped_candidate_counts
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.utils.logging import LogLevel
from lcm.exceptions import UnrepresentableOuterCandidateError
from tests.simulation.test_finite_policy_budget import _inputs


@pytest.mark.parametrize(
    ("live", "represented", "expected"),
    [
        ([[False, False]], [[False, True]], [0, 0]),
        ([[True, True]], [[True, True]], [0, 2]),
        ([[True, True]], [[False, False]], [2, 2]),
        (
            [[True, False, True], [False, True, True]],
            [[False, False, True], [True, False, True]],
            [2, 4],
        ),
    ],
)
def test_candidate_count_uses_only_live_unrepresented_cells(
    *,
    live: list,
    represented: list,
    expected: list,
) -> None:
    """Dead slots never enter either count, even with arbitrary represented flags."""
    actual = jax.jit(dropped_candidate_counts)(
        live=jnp.asarray(live),
        represented=jnp.asarray(represented),
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_real_finite_diagnostic_preserves_log_gate_and_precedes_ranking(
    *,
    log_level: LogLevel,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A seeded unsupported bank keeps its numeric mask and original host policy."""
    model, params, initial = _inputs(discrete=False, budget=2**32)
    solution = model.solve(params=params, log_level="off")
    dispatch = SimulationRuntime.dispatch
    run = SimulationMemory.run
    events: list[str] = []
    expected_live: list[int] = []

    def seed_bank(self: SimulationRuntime, **call: Any) -> object:
        name = call["program"].name
        if name == "simulate_policy_rank":
            events.append("rank")
        result = dispatch(self, **call)
        if name != "simulate_policy_prepare":
            return result
        inner, outer, live, represented = cast("tuple[jax.Array, ...]", result)
        expected_live.append(int(np.count_nonzero(np.asarray(live))))
        assert expected_live[-1] > 0
        events.append("prepare")
        return inner, outer, live, jnp.zeros_like(represented)

    def observe_count(self: SimulationMemory, **call: Any) -> object:
        result = run(self, **call)
        if call["function"] is dropped_candidate_counts:
            events.append("count")
            np.testing.assert_array_equal(result, [expected_live[-1]] * 2)
        return result

    monkeypatch.setattr(SimulationRuntime, "dispatch", seed_bank)
    monkeypatch.setattr(SimulationMemory, "run", observe_count)
    simulate = partial(
        model.simulate,
        params=params,
        solution=solution,
        initial_conditions=initial,
        log_level=log_level,
        seed=17,
    )
    if log_level == "debug":
        with pytest.raises(
            UnrepresentableOuterCandidateError, match="live outer candidates"
        ):
            simulate()
        assert events == ["prepare", "count"]
    else:
        result = simulate()
        expected_events = (
            ["prepare", "rank"] if log_level == "off" else ["prepare", "count", "rank"]
        )
        assert events == expected_events * 2
        rows = result.to_dataframe().query("regime_name == 'alive' and period == 0")
        assert np.isneginf(rows["value"]).all()
        assert np.isnan(rows["consumption"]).all()
        assert np.isnan(rows["illiquid_investment"]).all()
        messages = [
            record.message
            for record in caplog.records
            if "live outer candidates could not be reconstructed" in record.message
        ]
        if log_level == "off":
            assert messages == []
        else:
            assert len(messages) == 2
            for message, count in zip(messages, expected_live, strict=True):
                assert f"{count} of {count} live outer candidates" in message
