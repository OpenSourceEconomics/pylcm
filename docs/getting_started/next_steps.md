---
title: Choose your starting declaration
---

# Choose your starting declaration

Do not begin every model by copying the tiny model. Decide first how its continuous
choices will be solved:

| Expected route                               | Start with                                       | Declare now                             |
| -------------------------------------------- | ------------------------------------------------ | --------------------------------------- |
| General brute-force solution                 | `Regime` + `GridSearch()`                        | States, actions, functions, constraints |
| One smooth liquid Euler margin               | `ConsumptionSavingsRegime` + `EGM(...)`          | `LiquidMargin` roles and savings grid   |
| Liquid margin with discrete choice           | `ConsumptionSavingsRegime` + `DCEGM(...)`        | Liquid roles and envelope configuration |
| Nested liquid and durable/illiquid margins   | `NestedConsumptionSavingsRegime` + `NEGM(...)`   | Liquid and outer margin roles           |
| Declared institutional kinks or cliffs       | `ConsumptionSavingsRegime` + `NBEGM(...)`        | Margin roles and boundary structure     |
| Nested outer margin plus liquid kinks/cliffs | `NestedConsumptionSavingsRegime` + `NNBEGM(...)` | Both margins, boundaries, outer search  |

This is an authoring decision. A specialized solver cannot infer that an arbitrary DAG
node is “resources” or that a hidden `where` encodes a policy cliff. pylcm asks you to
name those roles and boundaries so it can validate the solver's assumptions before it
runs.

Use [Authoring for EGM-family solvers](../user_guide/authoring_specialized_solvers.md)
for complete executable declarations, and
[Choosing a solver](../user_guide/choosing_a_solver.md) for the full correctness and
performance decision. Then continue with
[Defining models](../user_guide/defining_models.md) or select a model with related
features from the [Examples](../examples/index.md).
