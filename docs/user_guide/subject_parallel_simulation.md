# Distribute simulated subjects independently of solve states

`ExecutionConfig(simulation_sharding="subjects")` is an explicit forward-execution
opt-in. It selects all configured devices for simulated subjects even when
`sharded_states=()`. It does not change the solve's state axes, economic grids,
transition laws, action ordering, seeds, or solution retention.

The default `"legacy"` preserves solve-derived subject placement and the existing
compiler-selected partitioning of the global subject tile loop.

## Whole cohorts and local tiles are different

For 226,848 subjects and eight selected GPUs, a full-cohort simulation has exactly
28,356 rows on each GPU. With an inner subject width of 2,048, each device evaluates 13
complete local tiles and a final tile of 1,732 rows. Each device executes its own loop
concurrently; Python does not run eight separate simulations.

This does not imply that all arrays shrink by eight. A device still needs its shared
parameters and required continuation/policy reads. Published solution owners,
transferred replicas, transfer workspace, compiler storage, current states and retained
outputs all remain subject to the existing admission checks.

```python
from lcm import ExecutionConfig

execution_config = ExecutionConfig(
    devices=tuple(range(8)),
    sharded_states=("assets",),  # Keep the intended SOLVE policy; not required here.
    simulation_sharding="subjects",
    simulation_chunk_policy="independent",
    axis_widths={"subject": 2048},
    device_memory_bytes=per_device_budget_bytes,  # Measured allocator envelope.
)
```

Pass this config when constructing the model. `per_device_budget_bytes` must be a
positive integer resolved for the selected hardware and existing owners; the population
count is not a memory budget. Other valid solve-state declarations, including
`sharded_states=()`, can use the same forward mode.

`simulation_chunk_policy="independent"` lets the existing planner explore larger
**global** outer cohorts. It does not force full-population admission. Inspect the
selected global extent and divide by the number of subject devices to obtain the local
row count. A fixed inner width plus the legacy outer policy still selects small outer
cohorts; selecting eight GPUs alone does not change that coupling.

In `"subjects"` mode the configured inner subject width is an upper bound per device,
clamped to the local population. In `"legacy"` mode its meaning is unchanged.
Action-product widths remain unchanged. Width/profile selection and live admission
compile and inspect the new mapped program, not an old memory estimate divided by the
number of devices.

## What is replicated and what is partitioned

Named per-subject argument trees, including state arrays and already-generated
per-person keys, are partitioned along their leading axis. Other argument trees are
replicated, even when their length happens to equal the population. The existing period
owner obtains the required value/policy replicas before execution; the wrapper does not
create a second value-transfer or ownership mechanism.

The complete existing tile loop executes inside `jax.shard_map`. Outputs retain their
leading subject axis and are concatenated logically by that map into the same global row
order. No full-population host concatenate is added. A single admitted outer cohort
keeps the existing device-result path; multiple cohorts retain the existing
offload/assembly behavior. Requesting a DataFrame or writing outputs can still require
host transfers.

Padding and random-number construction stay outside this wrapper. A GPU is not assigned
a new seed, and keys are not restarted using local row numbers. Padding continues to
duplicate the existing last subject and is trimmed at the original public result
boundary. An unaligned trimmed result is not guaranteed to retain equal physical output
shards; divisible populations avoid that final resharding.

## Bounded support and validation

Multi-device `"subjects"` mode requires JIT-compiled functions that explicitly certify
independent subject inputs and leading-axis outputs. Built-in `_SubjectTiled` programs
publish that capability. An undeclared function or a host-driven/eager program is
refused rather than silently given a new meaning. Shared-only `_SubjectTiled` functions
retain their existing execution path.

The mode does not introduce parallel regimes, cross-period scheduling, an adaptive
residency policy, a faster outer-cohort planner, or multi-host support. It is a
single-process subject partition. Physical GPU resource and numerical acceptance must be
established on the target software stack before production use.
