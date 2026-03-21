"""Performance benchmarks for PyLCM using ASV (airspeed velocity).

ASV discovers benchmark classes in `bench_*.py` files automatically.
Each class follows a standard protocol:

- `setup` / `teardown` — called before/after each benchmark method
  (like pytest fixtures). Use `setup` to build the model and run it
  once for JAX warmup (see below).
- `time_*` — wall-clock timing (ASV runs the method repeatedly and
  reports statistics).
- `peakmem_*` — peak memory usage during execution.
- `track_*` — return an arbitrary scalar metric (e.g.
  `track_compile_time` returns the JIT compilation time captured in
  `setup`).

Parametrised benchmarks declare `params` (list of value lists) and
`param_names` on the class; ASV passes every combination to
`setup` and the benchmark methods.

## JAX warmup pattern

JAX compiles (traces + XLA-compiles) a function the first time it is
called with a new input shape. To separate compilation overhead from
steady-state runtime:

1. `setup` calls the operation once and stores the elapsed time in
   `self._compile_time`.  After this call all relevant JAX traces are cached.
2. `time_*` / `peakmem_*` then measure post-compilation performance.
3. `track_compile_time` reports the first-call time so we can track compilation cost
   separately.

## Running benchmarks

```console
pixi run asv-run              # run benchmarks (requires clean worktree)
pixi run asv-publish          # generate static HTML report
pixi run asv-preview          # live-preview the report in a browser
pixi run asv-run-and-publish  # both in sequence
```

See `docs/development/benchmarking.md` for full details.
"""
