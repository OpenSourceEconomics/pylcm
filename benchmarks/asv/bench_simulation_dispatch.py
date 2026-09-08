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

An asv metric is identified by module, class and method name, and the design
spec's acceptance clause names `track_second_call_compiles` and
`track_host_ms_per_period_regime` directly. Those three names therefore stay as
they are, even though what the first counts is a compile *request*: the counter
itself says so, and renaming the metric would silently break every reference to
it outside this file.
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


class SimulationDispatch:
    """One warm call, then a measured call, per witness and log level."""

    version = "1"
    timeout = 600
    params = (list(WITNESS_NAMES), list(LOG_LEVELS))
    param_names = ["witness", "log_level"]

    # keyword-only-exempt: library-callback=asv
    def setup(self, witness: str, log_level: str) -> None:
        """Build the witness, solve it, and warm the measured log level."""
        from ._simulation_witnesses import WITNESSES

        model, model_params, initial_conditions = WITNESSES[witness]()
        self.model = model
        self.model_params = model_params
        self.initial_conditions = initial_conditions
        self.solution = model.solve(params=model_params, log_level="off")
        self._simulate(log_level=log_level)

    # keyword-only-exempt: library-callback=asv
    def track_host_ms_per_period_regime(self, witness: str, log_level: str) -> float:
        """Return host milliseconds per (period, regime) of a warm simulate call."""
        elapsed, _ = self._measure(log_level=log_level)
        return 1e3 * elapsed / count_period_regime_iterations(self.model)

    # keyword-only-exempt: library-callback=asv
    def track_second_call_compiles(self, witness: str, log_level: str) -> int:
        """Return the backend compilations a warm simulate call issues."""
        return self._measure(log_level=log_level)[1]

    def _simulate(self, *, log_level: str) -> object:
        return self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            solution=self.solution,
            log_level=log_level,
            seed=0,
        )

    def _measure(self, *, log_level: str) -> tuple[float, int]:
        import jax

        from benchmarks.asv._compile_counters import count_compile_requests

        with count_compile_requests() as counts:
            start = time.perf_counter()
            result = self._simulate(log_level=log_level)
            jax.block_until_ready(result.raw_results)
            elapsed = time.perf_counter() - start
        return elapsed, counts.compile_requests
