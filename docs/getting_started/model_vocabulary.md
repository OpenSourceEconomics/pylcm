---
title: Model vocabulary
---

# Model vocabulary

A pylcm model is a collection of **regimes** observed over a finite **age grid**. Each
regime describes one decision problem, and the model's edges connect them:

| Economic object                          | pylcm declaration   | Example                               |
| ---------------------------------------- | ------------------- | ------------------------------------- |
| Predetermined information                | `states`            | wealth, health, marital status        |
| Choices made now                         | `actions`           | consumption, work, next durable stock |
| Flow payoffs and intermediate quantities | `functions`         | utility, resources, taxes             |
| Feasible choices                         | `constraints`       | borrowing or time constraints         |
| Laws of motion                           | `state_transitions` | next wealth, health probabilities     |
| Movement between decision problems       | `Model(edges=...)`  | work → retirement → death             |
| Numerical method                         | `solver`            | grid search, EGM-family solvers       |

Functions form a dependency graph through their argument names. If `utility` takes
`consumption` and `leisure`, pylcm supplies those names from actions, states, other
functions, or parameters. Parameters are the names that remain after the graph is
assembled.

## Regimes are economic states, not code branches

A regime is useful when the available states, actions, equations, or transition laws
change qualitatively: employment versus retirement, single versus married, alive versus
dead. The regime's name is the key in `Model(regimes=...)` and in `Model(edges=...)`,
which declares where the regime can go and, where it has a choice of destinations, the
law that picks one.

## Grids and transition laws answer different questions

A grid states which values a variable may take. A transition law states how its next
value is produced. Deterministic laws are ordinary callables; stochastic laws use
`StochasticTransition`, a stochastic process, or `JointTransition` when several outcomes
must share one draw.

## Solvers constrain how the economics is declared

`GridSearch` evaluates complete state-action candidates and therefore accepts the
broadest model vocabulary. EGM-family solvers gain speed by using particular economic
roles: liquid resources, consumption, savings, and sometimes an outer durable or
illiquid margin. Those roles live on specialized regime classes rather than in the
solver configuration.

Read [Choose your starting declaration](next_steps.md) before writing a larger model.
The [User Guide](../user_guide/index.md) then develops each object, while the
[Reference](../reference/index.md) states the exact contracts.

`Model(edges={source: {target: source_ages}}, initial_nodes=InitialNodes(by_age={age: regime}))`
declares connectivity and admissible starts. A source with one destination at every age
needs nothing more: the graph is its law. A source with several destinations at some age
is declared as `Transition(targets={target: source_ages, ...}, law=...)`, and a regime
with no outgoing edges is terminal.
