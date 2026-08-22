---
title: Outer search and branch aggregation
---

# Outer search and branch aggregation

`NNBEGM` separates two decisions: how adjuster candidates are generated, and how the
keeper and adjuster branches combine.

## Outer search

### `FiniteOuterGrid`

```python
FiniteOuterGrid(grid=..., batch_size=0)
```

Solves one exact inner problem per grid node and returns a grid-snapped outer action.
The result is exact relative to that finite candidate set. Positive `batch_size` streams
nodes before folding them into the running maximum.

### `AdaptiveOuterMesh`

Starts from `initial_grid`, evaluates exact inner solves on a shared mesh, validates
interpolation at proposed points, and refines bracket-local optima.

Important fields:

- `max_nodes` and `max_refinement_rounds` are hard resource limits;
- `batch_size` streams mesh nodes;
- `value_atol` and `value_rtol` govern exact-versus-interpolated validation;
- `local_refiner` and `golden_iterations` control local refinement;
- `outer_lipschitz_bound` upgrades mesh-relative validation to a global branch-and-bound
  certificate under the supplied Lipschitz constant;
- `fail_closed=True` raises when refinement remains unresolved; `False` returns a
  flagged best effort for explicit diagnostic use.

Without a valid Lipschitz bound, midpoint/mesh validation cannot exclude an arbitrarily
narrow peak between sampled points.

### `LegacyGoldenSection`

Reproduces a historical global golden-section algorithm with explicitly labeled endpoint
and tie rules. It assumes global unimodality and is not the canonical general search.

`OuterSearch` is the abstract configuration marker.

## Branch aggregation

### `DeterministicOuterMaximum`

```python
DeterministicOuterMaximum()
```

Takes `max(V_keeper, V_adjuster)` with the keeper winning exact ties.

### `UniformObservedFixedCost`

Analytically integrates a shock `chi ~ U(lower, upper)` entering only the adjuster's
fixed adjustment cost through a non-negative `scale_function`. The shock must be
observed before branch choice, must not change conditional actions, and must not enter
state transitions except through that branch choice. It is supported with
`AdaptiveOuterMesh` and needs no solve-state grid.

`BranchAggregateResult` contains expected value, adjustment and no-adjustment
probabilities, and the cutoff draw. `OuterBranchAggregator` is the abstract
configuration marker.

Method background: [Nested endogenous-grid methods](../methods/nested_egm.md).
