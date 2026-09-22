"""Tests for the shared-measurement simulation-dispatch benchmark."""

import pytest

from benchmarks.asv import bench_simulation_dispatch as dispatch


class _FakeResult:
    def __init__(self) -> None:
        self.raw_results = object()


class _FakeModel:
    """A witness stand-in with a fixed (period, regime) iteration count."""

    def __init__(self, *, n_iterations: int) -> None:
        self.n_iterations = n_iterations
        self.simulate_calls = 0

    def solve(self, **_: object) -> object:
        return object()

    def simulate(self, **_: object) -> _FakeResult:
        self.simulate_calls += 1
        return _FakeResult()


def test_setup_cache_measures_each_combination_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both trackers must read one shared measured call per witness/log_level."""
    models = {name: _FakeModel(n_iterations=7) for name in dispatch.WITNESS_NAMES}

    def _fake_witness_builder(name: str):
        def _build():
            return models[name], {"p": 1}, {"c": 2}

        return _build

    monkeypatch.setattr(
        dispatch,
        "count_period_regime_iterations",
        lambda model: model.n_iterations,
    )
    monkeypatch.setattr(
        "benchmarks.asv._simulation_witnesses.WITNESSES",
        {name: _fake_witness_builder(name) for name in dispatch.WITNESS_NAMES},
    )
    monkeypatch.setattr("jax.block_until_ready", lambda x: x)

    instance = dispatch.SimulationDispatch()
    cache = instance.setup_cache()

    assert set(cache) == {
        (witness, log_level)
        for witness in dispatch.WITNESS_NAMES
        for log_level in dispatch.LOG_LEVELS
    }
    for model in models.values():
        # One warm call plus one measured call per log level.
        assert model.simulate_calls == 2 * len(dispatch.LOG_LEVELS)

    for (witness, log_level), measurement in cache.items():
        instance.setup(cache, witness, log_level)
        host_ms = instance.track_host_ms_per_period_regime(cache, witness, log_level)
        compiles = instance.track_second_call_compiles(cache, witness, log_level)
        assert host_ms == pytest.approx(1e3 * measurement["elapsed"] / 7)
        assert compiles == measurement["compile_requests"]


def test_version_identifies_the_shared_measurement_protocol() -> None:
    """A version bump marks the switch from per-tracker to shared measurement."""
    assert dispatch.SimulationDispatch.version == "2"
