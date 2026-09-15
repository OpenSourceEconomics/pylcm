"""Second-call compile requests and host time per (period, regime) of simulation.

Two CPU witnesses cover the two forward-simulation shapes pylcm has: a
multi-regime model whose states carry a discretised shock process, and a
collective model whose regime transition runs through gated edges. Each
benchmark warms the witness with one call at the measured log level, so what
the second call reports is the steady-state compile-request and host cost.

Model, params, and JAX imports are deferred into the method bodies: ASV's
forkserver imports every `bench_*.py` module to discover benchmarks before it
forks workers, and a JAX import at module scope puts the multithreaded XLA
backend into the forkserver, which every fork then inherits.

The two metrics answer two different questions, which is why both are tracked:
`track_second_call_compiles` says whether a warm call still asks JAX to compile
anything (it should ask for nothing), and `track_host_ms_per_period_regime`
says what a warm call costs per (period, regime) once nothing compiles, which
is how the cost of runtime validation becomes visible. An asv metric is
identified by module, class and method name, so those names are the identity
under which a measurement is stored and compared across commits: renaming one
starts a new series and silently orphans the old, which is why they stay as
they are even though what the first counts is a compile *request*.

`setup_cache` measures every witness/log_level combination exactly once (one
warm call, then one counted+timed call) and both trackers read the shared
result. Previously each tracker ran its own independent warm-plus-measured
cycle per combination, doubling the simulate calls for no additional
information: the elapsed time and the compile-request count come from the
same call.
"""

import time

# Names of the CPU witnesses, as `_simulation_witnesses.WITNESSES` keys.
WITNESS_NAMES = ("dissolution", "multi_regime")

# Log levels measured: the validation-free path and the one a long run uses.
LOG_LEVELS = ("off", "progress")


def count_period_regime_iterations(model: object) -> int:
    """Return the number of (period, regime) iterations one chunk runs."""
    return sum(
        1
        for regime in model._regimes.values()  # noqa: SLF001
        for period in range(model.n_periods)
        if period in regime.active_periods
    )


def _measure_combination(*, witness: str, log_level: str) -> dict[str, float]:
    """Build one witness, warm it, and take one measured call at `log_level`."""
    import jax

    from benchmarks.asv._compile_counters import count_compile_requests
    from benchmarks.asv._simulation_witnesses import WITNESSES

    model, model_params, initial_conditions = WITNESSES[witness]()
    solution = model.solve(params=model_params, log_level="off")

    def _simulate() -> object:
        return model.simulate(
            params=model_params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level=log_level,
            seed=0,
        )

    _simulate()  # warm-up call; not measured

    with count_compile_requests() as counts:
        start = time.perf_counter()
        result = _simulate()
        jax.block_until_ready(result.raw_results)
        elapsed = time.perf_counter() - start

    return {
        "elapsed": elapsed,
        "compile_requests": counts.compile_requests,
        "period_regime_iterations": count_period_regime_iterations(model),
    }


class SimulationDispatch:
    """One warm call, then one measured call, per witness and log level."""

    # Bumped from "1": measurement moved from a per-tracker warm+measured
    # cycle into one shared `setup_cache` producer per combination.
    version = "2"
    timeout = 600
    params = (list(WITNESS_NAMES), list(LOG_LEVELS))
    param_names = ["witness", "log_level"]

    def setup_cache(self) -> dict[tuple[str, str], dict[str, float]]:
        """Warm and measure each witness/log_level combination exactly once."""
        return {
            (witness, log_level): _measure_combination(
                witness=witness, log_level=log_level
            )
            for witness in WITNESS_NAMES
            for log_level in LOG_LEVELS
        }

    # keyword-only-exempt: library-callback=asv
    def setup(
        self,
        cache: dict[tuple[str, str], dict[str, float]],
        witness: str,
        log_level: str,
    ) -> None:
        self._measurement = cache[(witness, log_level)]

    # keyword-only-exempt: library-callback=asv
    def track_host_ms_per_period_regime(
        self,
        cache: dict[tuple[str, str], dict[str, float]],
        witness: str,
        log_level: str,
    ) -> float:
        """Return host milliseconds per (period, regime) of the measured call."""
        measurement = self._measurement
        return 1e3 * measurement["elapsed"] / measurement["period_regime_iterations"]

    # keyword-only-exempt: library-callback=asv
    def track_second_call_compiles(
        self,
        cache: dict[tuple[str, str], dict[str, float]],
        witness: str,
        log_level: str,
    ) -> int:
        """Return the backend compilations the measured call issued."""
        return self._measurement["compile_requests"]
