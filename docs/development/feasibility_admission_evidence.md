# Feasibility producer admission

Budgeted `Model.simulate` compiles and admits the complete feasibility predicate before
dispatch. The same call-local path covers summary validation, ordered serial replay,
cohort-constant predicates, and individual constraints. Diagnostic state gathers use the
existing profiled operation owner. Runtime parameters and arrays are explicit dynamic
operands; user-DAG executables stay local to the current call. Preflight cohorts are
replicated because their lengths may be smaller than the subject mesh. Earlier action
products remain owned by `PreflightActionGrids`.

This change is relative to `32549408bbbfb39b906ad0df99a5128ada6df1b9`.
`tests/simulation/test_feasibility_admission.py` exercises the public simulation
boundary and observes actual JAX memory profiles and dispatch boundaries. Its principal
low-budget example has two subjects, 1,024 actions, and a constraint that sorts 4,096
sine samples before comparing the median with a parameter. At 12 KiB in float64 the
baseline executes feasibility and raises `InvalidInitialConditionsError`; the admitted
implementation raises `ExecutionPlanningError` before executing the declined feasibility
program. The original red and first green runs contained one test each, respectively
zero and one passing tests.

The CPU checks pass all 45 new cases in both float64 and float32. These include
automatic and supplied solutions, enabled log levels, forced serial validation, natural
per-constraint replay, parameter freshness with a structured leaf, and a cohort shorter
than its subject mesh. Another 84 existing simulation checks pass in float64. The
low-budget thresholds are declared CPU-specific; their numerical values come from the
observed compiler profiles. The 11 feasibility mutation controls form a separate
population alongside the existing 406, 50, 37, and 10 controls.

## Compiler-memory reporting discrepancy and central repair

The original example had **two actions and a 16 KiB budget**. Its intended
pre-allocation resource refusal failed under the peak-only compiler contract of this
feasibility change. The subsequent central reservation repair is documented in
[compiler allocation admission](compiler_allocation_admission.md). On CPU, Python
3.14.7, JAX/JAXLIB 0.11.1, float64, the full combined feasibility executable reported:

| Compiler field           |   Bytes |
| ------------------------ | ------: |
| `argument_size_in_bytes` |      56 |
| `output_size_in_bytes`   |       2 |
| `temp_size_in_bytes`     | 131,072 |
| `peak_memory_in_bytes`   |      90 |

These fields were read from `Compiled.memory_analysis()` after compiling the actual
combined predicate. That peak-only `compiler_peak_bytes` admission contract consumed
`peak_memory_in_bytes`. The example therefore passed admission, executed, and reached
the infeasibility diagnostic. Summing arguments, outputs, and temporary storage would
change that contract. No such sum is labelled as a measured compiler peak in this
repair. A larger, 256-action sorting profile likewise reported 16,777,728 temporary
bytes and a 4,155-byte peak, so the discrepancy is reproducible beyond one shape.

To reproduce the original public example from the repository in the tests-cpu Pixi
environment, enable JAX float64 and import `_inputs` from
`tests.simulation.test_feasibility_admission`. Construct
`model, params, initial = _inputs(budget=16_384, n_actions=2)`, then call
`model.simulate(params=params, initial_conditions=initial, log_level="debug")`. To
inspect its statistics, observe `jax.stages.Lowered.compile` and read
`memory_analysis()` from the program whose HLO contains `_batched_feasibility_check`,
before `jax.stages.Compiled.__call__`.

The central follow-up captures the exact executable and refuses it before dispatch using
represented allocation accounting. It preserves the raw 90-byte peak separately from its
computed reservation. Stochastic state-transition validation now has a separate admitted
producer and serial replay contract. The broader producer inventory still retains
regime-transition and joint-transition law evaluation, composite stochastic process
grids, and remaining structural invalid-input operations as open work. This feasibility
change provides no general guarantee for those allocations or for memory omitted from a
backend's reported peak.
