---
title: Migrating regime graph declarations
---

# Migrating regime graph declarations

The model owns every regime transition through a required `edges` argument, both its
structure and its law. Initial age–regime pairs are required explicitly through
`initial_nodes`.

## Move transitions onto Model

Delete `regime_transitions=` from every `Regime`. Where each source age has a single
destination, the edges alone are the law:

```python
# Current declaration fragment: working stays until 62, then retires.
working = Regime(
    functions={"utility": utility},
    states={"assets": assets_grid},
    state_transitions={"assets": next_assets},
)
edges = {
    "working": {
        "working": AgeRange(start=60, exclusive_stop=62),
        "retired": 62,
    },
}
model = Model(
    regimes={"working": working, "retired": retired},
    regime_id_class=RegimeId,
    ages=AgeGrid(start=60, inclusive_stop=63, step="Y"),
    edges=edges,
    initial_nodes=((60, "working"),),
)
```

This replaces laws such as `"dead"`, `ByAge.until(law="working", then="retired")` or a
selector that only ever returns the one available destination. A regime that declared
`regime_transitions=None` simply has no outgoing edges.

Code that read a regime's law back reads it from the model graph:
`model.graph.laws[name].terminal` replaces `regime.terminal`, and
`model.graph.laws[name].gated_edges` replaces `regime.gated_edges`.

Where a source age has several destinations, move the former `regime_transitions` value
unchanged into a `Transition` that replaces the source's destination mapping:

```python
# Fragment: a choice between continuing to work and retiring.
edges = {
    "working": Transition(
        targets={"working": AgeRange(start=60, exclusive_stop=63), "retired": (61, 62)},
        law=DeterministicTransition(func=destination),
    ),
}
```

A `Transition` on a source with at most one destination per age is rejected; drop its
law instead. A `ByAge` law may leave single-destination ages unselected.

Public `DeterministicTransition` and `StochasticTransition` refuse `targets` with an
error that points to `Model(edges=...)`, the only place regime transitions are declared.
Their targetless decorator factories are `@deterministic_transition()` and
`@stochastic_transition()`. Plain functions remain deterministic for both state and
regime laws. Full-vector stochastic laws retain global regime-code ordering;
probabilities outside graph support must be zero.

Per-target probability mappings still supply scalar probability laws. Move their
structural destination and age restrictions into `edges`, and keep target-specific state
handoffs on the source regime. `ByAge` selects complete laws, not topology or solved
coverage. Replace age activity predicates with graph selectors and explicit starts.

## Name age bounds explicitly

`AgeGrid(start=..., inclusive_stop=..., step=...)` includes its final age.
`AgeRange(start=..., exclusive_stop=...)` excludes its stop. The old `stop` keywords are
removed. `ByAge.until(stop_age_exclusive=...)` continues to use its existing explicit
bound name. Every edge refers to source ages and lands at the next grid coordinate,
including on irregular or multi-year grids.

## Declare initial nodes

Replace `initial_regimes` with `initial_nodes`. Prefer explicit pairs:

```python
initial_nodes = ((60, "working"), (62, "retired"))
```

Selector mappings such as `{60: "working", 62: "retired"}` remain accepted convenience.
There is no default. These declarations govern admissible starts; subject weights and
initial states still belong to `Population` and `InitialConditions`.

## Migrate phase differences and inspection

A shared edge mapping applies to both phases. Use `Phased(solve=..., simulate=...)` on
`edges` for perceived versus realized connectivity; a former
`regime_transitions=Phased(solve=..., simulate=...)` moves into each phase's
`Transition`, or stays whole as the law of one `Transition` on shared edges. State
handoffs stay phase-specific on the source regime. Physically visited nodes require
solve values plus their perceived dependencies; extra valued nodes do not imply realized
visits.

Inspect `model.graph.edges.solve` and `.simulate` for declared exact source ages,
`.solution` and `.simulation` for effective period-indexed graphs, `.nodes` for valued
pairs and `.visited_nodes` for realized pairs. `.pruned_edges["solve"]` and
`["simulate"]` map `(source_age, source, target)` to `"fixed_zero_probability"`.
Construction-fixed scalar zero proofs can prune effective edges; runtime probability
zeros cannot. Declared edges remain inspectable and probability mass is never
renormalized.
