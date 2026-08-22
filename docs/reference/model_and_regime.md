---
title: Model and Regime
---

# Model and Regime

## `Model`

`Model(...)` assembles regimes over one lifecycle:

```python
import lcm

model = lcm.Model(
    ages=lcm.AgeGrid(start=25, stop=80, step="Y"),
    regimes={"working": working, "retired": retired},
    regime_id_class=RegimeId,
    enable_jit=True,
)
```

Required arguments are `ages`, `regimes`, and a class created with
`@categorical(ordered=False)` whose fields match the regime names. A model must contain
at least one non-terminal and one terminal regime.

The mapping-valued slots `functions`, `constraints`, `states`, `state_transitions`,
`actions`, and `derived_categoricals` broadcast declarations to regimes. A name is
defined at model or regime level, never both. A regime-level `None` masks a broadcast
entry. Broadcast variables unused by every root computation in both phases are pruned;
inspect `model.pruned_variables`.

`koopmans_aggregator` and `certainty_equivalent` are single broadcast values. Declare
each at model level or in every non-terminal regime, not a mixture. `fixed_params` binds
parameters when the model is built. `n_subjects` optionally prepares simulation programs
for one population shape; parameter shapes and dtypes must remain stable for reuse.

Public inspection attributes include:

- `ages`, `n_periods`, and `regime_names_to_ids`;
- `user_regimes`, the finalized declarations in user vocabulary;
- `pruned_variables`;
- `get_params_template()`, which returns a mutable nested template.

`model._regimes` is private canonical engine state.

## `Regime`

A general regime declares:

| Field                  | Contract                                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| `transition`           | Regime transition callable, stochastic declaration, per-target mapping, or `None` for terminal |
| `active`               | Age predicate; omitted means always active                                                     |
| `states` / `actions`   | Name-to-grid mappings                                                                          |
| `functions`            | Named DAG functions; a finalized regime needs utility                                          |
| `constraints`          | Ordinary predicates or structured `Condition` objects                                          |
| `state_transitions`    | One law per non-process state in a non-terminal regime                                         |
| `derived_categoricals` | Discrete grids for categorical DAG outputs                                                     |
| `solver`               | `lcm.solvers.GridSearch()` by default                                                          |
| `taste_shocks`         | Optional EV1 taste-shock configuration                                                         |
| `koopmans_aggregator`  | Optional regime-level continuation aggregator                                                  |
| `certainty_equivalent` | Optional regime-level lottery reduction                                                        |
| `description`          | Human-readable description                                                                     |

Use `Regime.replace(...)` to derive a modified immutable declaration.

Terminality is defined by `transition is None`. Terminal regimes declare no
`state_transitions`, Koopmans aggregator, or certainty equivalent because they have no
continuation.

`ConsumptionSavingsRegime` and `NestedConsumptionSavingsRegime` add the economic roles
required by EGM-family solvers. See
[Consumption-saving regimes and margins](consumption_savings.md). Collective fields on
`Regime` are documented separately in [Collective regimes](collective_regimes.md).

## Derived categoricals

Use `derived_categoricals={"name": DiscreteGrid(Category)}` when a parameter is indexed
by a categorical function output rather than by a state or action. The function must
return an integer code, not a Boolean, because it is used as an array index under JIT.

## Parameter ownership

A free function argument becomes a model parameter unless another state, action, DAG
function, context value, or fixed parameter supplies it. Values may be given at model,
regime, or function level, but each parameter value has one unambiguous source. Start
from `model.get_params_template()` rather than constructing a nested parameter mapping
from memory.

Workflow: [Defining models](../user_guide/defining_models.md) and
[Parameters](../user_guide/parameters.md).
