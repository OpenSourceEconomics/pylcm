---
title: Transitions and phase specialization
---

# Transitions and phase specialization

## State transitions

A non-terminal regime declares one transition for every ordinary state it carries:

- an ordinary callable is deterministic;
- `MarkovTransition(func)` wraps a probability-vector function;
- `fixed_transition("state_name")` declares the identity law;
- a per-target mapping gives different laws for different reachable target regimes.

Stochastic process states already own their transitions and must not appear in
`state_transitions`. A terminal regime has no state transitions.

Per-target state-transition mappings must cover exactly the reachable targets that carry
the state. Reachability comes from the regime's `transition` declaration; extra or
missing targets are errors.

## Regime transitions

`Regime.transition` accepts:

- `None` for a terminal regime;
- a deterministic callable returning a regime code;
- `MarkovTransition(func)` returning probabilities over all regimes;
- a mapping `{target_name: MarkovTransition(probability_function)}`.

The key set of the mapping declares structural reachability. Use it when some regimes
cannot follow the source; do not encode structural impossibility only as a zero
probability in an all-regime vector.

## Joint transitions

`JointTransition(support_size, support, probabilities, outputs)` declares several next
states driven by one shared draw. `support` describes the literal joint nodes,
`probabilities` returns a vector of length `support_size`, and each `outputs` entry
projects a sampled joint node into one target state.

Joint laws occupy the separate `Regime.joint_transitions` slot. Its public shape is a
mapping from target regime, to local joint-node name, to the `JointTransition`:

```python
source = Regime(
    transition={"target_regime": MarkovTransition(target_probability)},
    joint_transitions={
        "target_regime": {
            "joint_draw": JointTransition(
                support_size=2,
                support={
                    "wealth": wealth_nodes,
                    "health": health_nodes,
                },
                probabilities=joint_probabilities,
                outputs={
                    "wealth": next_wealth,
                    "health": next_health,
                },
            )
        }
    },
    functions={"utility": utility},
)
```

The outer key names the reachable target regime. The inner key names the sampled joint
node that output functions may read. Every output state is owned by that joint law and
must not also have an independent entry in `state_transitions`.

Use `Phased` around the entire `JointTransition` for perceived and realized variants;
both variants keep the same output names, support size, and literal support schema.

## Age specialization

`AgeSpecializedFunction(build, signature)` and `AgeSpecializedGrid(build, signature)`
produce age-specific declarations during model construction. `build(age)` returns the
function or continuous grid for that age. `signature(age)` returns a stable hashable
key; equal keys must mean identical resolved behavior because those periods may share
one compiled program.

Exact function placement:

- accepted in `functions` and `constraints` of non-terminal regimes;
- not accepted as the regime transition, inside `MarkovTransition`, or directly as a
  state-transition value;
- a state law may be a plain function that reads an age-specialized helper;
- terminal regimes do not accept age-specialized functions;
- additional DataFrame targets may not depend on them because published target functions
  use a representative age.

`AgeSpecializedGrid` is accepted only as a top-level continuous-state grid. It is not an
action, discrete/process grid, runtime-points grid, or a member of a carried
`Phased(solve=callable, simulate=Grid)` state. Grid class, node count, shape, and dtype
remain constant across ages.

Factories run while the model is built. Solve, simulation, compilation, and diagnostics
select the already-resolved period objects and never call `build(age)`.

## Solve and simulation phases

`Phased(solve=..., simulate=...)` is the outermost wrapper for declarations that may
differ by phase:

- `functions` and `state_transitions` accept phase-specific variants;
- `transition` accepts variants with the same transition form and target keys;
- `states` accepts the special carried-state form
  `Phased(solve=callable, simulate=Grid)`.

Constraints, actions, `active`, and derived categoricals are phase-invariant and reject
`Phased`. Nested phase wrappers and wrappers inside per-target mappings are invalid.

A carried state is derived during backward induction, so it adds no solve-grid axis, but
is seeded and evolved as a genuine simulation state. Its law of motion still belongs in
`state_transitions`.

Workflow: [Transitions](../user_guide/transitions.ipynb) and
[Age-specialized functions and grids](../user_guide/age_specialized.md). Rationale:
[Phase-dependent model structure](../explanations/phase_grammar.ipynb).
