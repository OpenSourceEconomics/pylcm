# Distribute simulated subjects independently of solve states

`ExecutionConfig(simulation_sharding="subjects")` is an explicit forward-execution
opt-in. It selects all configured devices for simulated subjects even when
`sharded_states=()`. It does not change the solve's state axes, economic grids,
transition laws, action ordering, seeds, or solution retention.

The default `"legacy"` preserves solve-derived subject placement and the existing
compiler-selected partitioning of the global subject tile loop.

Use it when simulation, not the solve, dominates a call. Measured on the ACA retirement
model, 8×A40, fp32, 226,848 subjects, one pair against `"legacy"`: warm simulation went
from 801 s to 191 s and the warm call from 1,686 s to 1,064 s; the solve time and device
peak were unchanged.

## Same panel as the unsharded run

With a fixed seed, parameters and initial conditions, `"subjects"` produces a panel that
is bitwise equal to the `"legacy"` run's. Two mechanisms make that hold:

- **Sealed shock draws.** Each shock draw rounds `sigma * standard_normal` exactly as
  written, behind an optimization barrier, so the compiler cannot fold constants into
  the draw differently in the sharded and unsharded programs.
- **Padded last tile.** When the subjects do not fill the last tile of width `k`, the
  tile is padded with copies of the last subject and the copies are sliced off
  afterwards. Every subject therefore runs in the same compiled program, instead of the
  remainder running in a second program that can round differently.

The guarantee covers the simulation only. The solve is unchanged by this mode; a solve
that shards states has its own fp32 caveats (see
[Performance and memory tuning](tuning.md#which-states-can-be-sharded)).

## Whole cohorts and local tiles are different

For 226,848 subjects and eight selected GPUs, a full-cohort simulation has exactly
28,356 rows on each GPU. With an inner subject width of 2,048, each device evaluates 13
complete local tiles and a final tile of 1,732 rows, padded to 2,048. Each device
executes its own loop concurrently; Python does not run eight separate simulations.

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
    axis_widths={"subject": 2048},
    device_memory_bytes=per_device_budget_bytes,  # Measured allocator envelope.
)
```

Pass this config when constructing the model. `per_device_budget_bytes` must be a
positive integer resolved for the selected hardware and existing owners; the population
count is not a memory budget. Other valid solve-state declarations, including
`sharded_states=()`, can use the same forward mode.

A positive `device_memory_bytes` budget lets the top-first planner explore larger
**global** outer cohorts. It does not force full-population admission. Inspect the
selected global extent and divide by the number of subject devices to obtain the local
row count.

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
a new seed, and keys are not restarted using local row numbers. Padding duplicates the
existing last subject and is trimmed at the original public result boundary. An
unaligned trimmed result is not guaranteed to retain equal physical output shards;
divisible populations avoid that final resharding.

## Bounded support and validation

Multi-device `"subjects"` mode requires JIT-compiled functions that explicitly certify
independent subject inputs and leading-axis outputs. Built-in `_SubjectTiled` programs
publish that capability. An undeclared function or a host-driven/eager program is
refused rather than silently given a new meaning. Shared-only `_SubjectTiled` functions
retain their existing execution path.

A regime declaring `gated_edges` is supported on more than one device, because the two
programs a gated edge needs split cleanly along the same line as everything else here.
The gate fold reads the next period's value and dissolution arrays over each target's
own regime-level grid and writes the substituted continuation over that same grid, so it
carries no subject axis and is replicated. The gate route recomputes the gate at each
subject's own realized state and writes that row's destination under an elementwise
mask, so its per-subject operands --- the candidate next states, the new regime ids, the
in-regime membership, and the current and new stakeholder roles --- are partitioned like
any other per-subject tree, while the folded continuation grids and the params stay
shared. Nothing in the route reduces, sorts, or scatters across rows.

One solve-side restriction is unaffected by the forward mode: a gate predicate that
reads a state named in `sharded_states` is refused when the continuation co-maps that
state as fixed distributed state, because a co-mapped axis is sliced off before the
continuation is read and leaves no landing coordinate to evaluate the gate at. Read the
state through a gate reference, or drop it from `sharded_states`.

The mode does not introduce parallel regimes, cross-period scheduling, an adaptive
residency policy, a faster outer-cohort planner, or multi-host support. It is a
single-process subject partition. Physical GPU resource and numerical acceptance must be
established on the target software stack before production use.
