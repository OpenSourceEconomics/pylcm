"""DS-2024 housing RFC-vs-NEGM method comparison (timing + accuracy).

Reproduces the Dobrescu-Shanker (2024) housing-model method comparison — the
paper reports RFC at 19.03 s/iter against NEGM at 28.12 s/iter (RFC faster). Here
pylcm solves its *own* RFC-vs-NEGM comparison on the same housing model and
reports absolute per-method timings (compilation split from runtime, since pylcm
JIT-compiles the solve) plus a value-function accuracy check against the
grid-search (VFI) oracle:

- the **NEGM** column is the nested-EGM solver (`ds2024_housing.py`): outer
  housing search, inner liquid DC-EGM;
- the **RFC** / **FUES** columns are the discrete-housing DC-EGM
  (`ds2024_housing_fues.py`) with the 1-D rooftop-cut / fast-upper-envelope-scan
  backend nested over the housing grid — the source's per-housing-column
  refinement.

Both solve the same DS-2024 housing economics (faithful at `delta = 0`). The
comparison is timing (does RFC beat NEGM?) and accuracy (each method against the
VFI oracle), not a reproduction of the paper's absolute seconds.
"""

import time
from dataclasses import dataclass

import jax
import numpy as np

from tests.test_models import ds2024_housing, ds2024_housing_fues


@dataclass(frozen=True)
class HousingMethodBenchmark:
    """One method's compile/runtime split and accuracy at one grid resolution."""

    method: str
    """Solution-method name (`"NEGM"`, `"RFC"`, `"FUES"`)."""
    n_grid: int
    """Liquid / housing grid points."""
    compile_seconds: float
    """Wall-clock of the first (compiling) solve minus the warm runtime."""
    runtime_seconds: float
    """Wall-clock of a warm (compiled) solve to device-ready."""
    accuracy_mean_abs: float
    """Mean absolute interior value-function difference from the VFI oracle."""


def _time_solve(solve, n_warm: int = 2) -> tuple[float, float, dict]:
    """Return `(compile_seconds, runtime_seconds, solution)` for a solve closure.

    The first call compiles and runs; the warm calls run the compiled program.
    Compile time is the cold call minus the fastest warm runtime.
    """
    start = time.perf_counter()
    solution = solve()
    jax.block_until_ready([v for regime in solution.values() for v in regime.values()])
    cold = time.perf_counter() - start

    warm_times = []
    for _ in range(n_warm):
        start = time.perf_counter()
        warm = solve()
        jax.block_until_ready([v for regime in warm.values() for v in regime.values()])
        warm_times.append(time.perf_counter() - start)
    runtime = min(warm_times)
    return max(cold - runtime, 0.0), runtime, solution


def _interior_mean_abs(solution: dict, oracle: dict, n_grid: int) -> float:
    """Mean absolute alive-regime value difference from the oracle on the interior."""
    interior = (Ellipsis, slice(3, n_grid - 2))
    differences = []
    for period in sorted(solution):
        if "alive" not in solution[period]:
            continue
        solved = np.asarray(solution[period]["alive"])[interior]
        reference = np.asarray(oracle[period]["alive"])[interior]
        finite = np.isfinite(solved) & np.isfinite(reference)
        differences.append(np.abs(solved - reference)[finite])
    return float(np.concatenate(differences).mean())


def benchmark_ds2024_housing(
    *,
    n_grid: int = 12,
    n_housing: int = 6,
    n_consumption: int = 300,
    n_periods: int = 5,
) -> list[HousingMethodBenchmark]:
    """Benchmark NEGM, RFC, and FUES on the DS-2024 housing model.

    Args:
        n_grid: Liquid / housing grid points.
        n_housing: Discrete housing levels for the DC-EGM column.
        n_consumption: Consumption-grid points for the grid-search oracle.
        n_periods: Lifecycle periods.

    Returns:
        List of `HousingMethodBenchmark` rows, one per method.
    """
    brute_fues = ds2024_housing_fues.build_model(
        variant="brute",
        n_grid=n_grid,
        n_housing=n_housing,
        n_consumption=n_consumption,
        n_periods=n_periods,
        upper_envelope="rfc",
    ).solve(
        params=ds2024_housing_fues.build_params(variant="brute", delta=0.0),
        log_level="off",
    )

    rows: list[HousingMethodBenchmark] = []

    def negm_solve() -> dict:
        model = ds2024_housing.build_model(
            variant="negm", n_grid=n_grid, n_periods=n_periods
        )
        return model.solve(
            params=ds2024_housing.build_params(variant="negm", delta=0.0),
            log_level="off",
        )

    compile_s, runtime_s, negm_solution = _time_solve(negm_solve)
    # The NEGM model carries housing as a continuous state, so its oracle is its own
    # grid-search twin (same continuous-housing economics).
    negm_oracle = ds2024_housing.build_model(
        variant="brute", n_grid=n_grid, n_periods=n_periods
    ).solve(
        params=ds2024_housing.build_params(variant="brute", delta=0.0), log_level="off"
    )
    rows.append(
        HousingMethodBenchmark(
            method="NEGM",
            n_grid=n_grid,
            compile_seconds=compile_s,
            runtime_seconds=runtime_s,
            accuracy_mean_abs=_interior_mean_abs(negm_solution, negm_oracle, n_grid),
        )
    )

    for method, backend in (("RFC", "rfc"), ("FUES", "fues")):

        def dcegm_solve(backend: str = backend) -> dict:
            model = ds2024_housing_fues.build_model(
                variant="dcegm",
                n_grid=n_grid,
                n_housing=n_housing,
                n_periods=n_periods,
                upper_envelope=backend,
            )
            return model.solve(
                params=ds2024_housing_fues.build_params(variant="dcegm", delta=0.0),
                log_level="off",
            )

        compile_s, runtime_s, solution = _time_solve(dcegm_solve)
        rows.append(
            HousingMethodBenchmark(
                method=method,
                n_grid=n_grid,
                compile_seconds=compile_s,
                runtime_seconds=runtime_s,
                accuracy_mean_abs=_interior_mean_abs(solution, brute_fues, n_grid),
            )
        )

    return rows


def format_comparison_table(rows: list[HousingMethodBenchmark]) -> str:
    """Render the housing method-comparison rows as a fixed-width table."""
    header = (
        f"{'method':<8} {'n_grid':>6} {'compile_s':>10} "
        f"{'runtime_s':>10} {'accuracy':>10}"
    )
    separator = "-" * len(header)
    body = [
        f"{row.method:<8} {row.n_grid:>6} {row.compile_seconds:>10.3f} "
        f"{row.runtime_seconds:>10.4f} {row.accuracy_mean_abs:>10.4f}"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


if __name__ == "__main__":
    benchmark_rows = benchmark_ds2024_housing()
    print(format_comparison_table(benchmark_rows))
