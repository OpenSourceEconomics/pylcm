# Regime checker regressions

Run the checker tests from the repository root with the CPU test environment.

```sh
pixi run -e tests-cpu pytest tests/regime_finder -v --junitxml=reports/checker-tests.xml
```

The suite checks native declaration capture, state and action dependencies, source and
certificate mutations, categorical projections, joint transition laws, heuristic
candidate checks, exact costs on small independent families, report generation and
filesystem safety. The CI workload manifest selects it on Linux CPU and collects
coverage, including the checker subprocesses.

The Python files stored with the `.py.txt` suffix are parser input, not imported test
modules. The source adapter tests materialize their unchanged bytes in a temporary
directory so sibling-module lookup works without treating historical model sources as
current installed APIs.

The native conditional-shock test preserves a state used only by a registered shock
probability and checks that omitting or miswiring the law fails independent
verification. The resource probe runs in a fresh interpreter so the test runner's
retained native allocations do not affect its memory measurement.
