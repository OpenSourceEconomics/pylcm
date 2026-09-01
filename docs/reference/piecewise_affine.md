---
title: Piecewise-affine schedules
---

# Piecewise-affine schedules

`piecewise_affine` declares that one output is affine in a named variable between known
breakpoints. It is the preferred representation for tax brackets, transfer phase-outs,
floors, and schedules with several kinks or jumps.

```python
import lcm


@lcm.piecewise_affine(
    output="resources",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint(
            threshold="first_threshold",
            kind="continuous_kink",
        ),
        lcm.affine_breakpoint(
            threshold="asset_test",
            kind="jump",
        ),
    ),
)
def resources(*, liquid, first_threshold, asset_test, transfer):
    # The function remains the executable economic schedule.
    ...
```

(api-affine-breakpoint)=

## `affine_breakpoint`

`affine_breakpoint(threshold, kind="continuous_kink", indexed_by=None, static_index=None, threshold_subkey=None)`
identifies one boundary.

- `threshold` names the flat parameter or parameter table.
- `kind` is `"continuous_kink"`, `"jump"`, or `"hard_constraint"`.
- `indexed_by` names a categorical state when the threshold varies across categories.
- `static_index` selects a fixed table position.
- `threshold_subkey` selects a nested parameter leaf.

Use only the indexing fields required by the parameter shape. Invalid or contradictory
combinations are rejected when the declaration is built.

## Solver contract

The decorated function remains ordinary Python and is valid under `GridSearch`. `NBEGM`
uses the metadata to partition the liquid axis into smooth affine runs, create one-sided
boundary candidates, and choose the appropriate envelope route.

The declaration asserts affinity **between** breakpoints. Solver validation probes the
assembled economic DAG and rejects hidden non-affinity or current-state dependence that
would invalidate the per-run inversion. `NBEGM(probe_failure="assume_declared")` is an
explicit author assertion for a probe that cannot execute; it should be paired with
independent grid-search comparisons and is not the default.

Use [case pieces](case_pieces.md) for one supported binary formula split. Feasibility
comparisons belong to [structured Conditions](conditions.md).
