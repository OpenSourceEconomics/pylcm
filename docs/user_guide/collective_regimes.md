---
title: Households and value-dependent choice
---

# Households and value-dependent choice

A single decision maker maximizes her own value. A household does not: it takes one
action for two people, it is only viable while both of them want it, and whether it
forms or ends depends on what each member would be worth outside it. Three declarations
cover that, and each goes in a slot a regime already has.

| Declaration                | Goes in                | Says                                                                 |
| -------------------------- | ---------------------- | -------------------------------------------------------------------- |
| `CollectiveUtility`        | `functions["utility"]` | who the stakeholders are, and how their action values are traded off |
| `ValueDependentConstraint` | `constraints`          | where the cell is feasible, reading values as well as states         |
| `ValueDependentTransition` | `transition`           | which target, and which branch within it                             |

`ProjectedRegimeValue` is what the last two share: a reading of another regime's value
in the *same* period.

The complete worked model is `lcm_examples.collective_household`. This page walks
through the pieces.

## A household with two stakeholders

The `utilities` keys **are** the regime's stakeholders, in the order you write them.
That order fixes the trailing axis of the regime's value function and of every published
array, so `value_f` comes before `value_m` in the simulated frame.

```python
from lcm import CollectiveUtility, Regime

couple = Regime(
    transition=...,
    states={"wealth": couple_wealth},
    state_transitions={"wealth": next_couple_wealth},
    actions={"consumption": consumption},
    functions={
        "utility": CollectiveUtility(
            utilities={"f": her_utility, "m": his_utility},
        )
    },
)
```

The household maximizes one objective over the feasible actions and reads off each
stakeholder's own action value `Q_s` at that common choice. Omitting `objective` weights
the stakeholders equally. To weigh them differently, declare a `ParetoObjective`:

```python
from lcm import ParetoObjective

CollectiveUtility(
    utilities={"f": her_utility, "m": his_utility},
    objective=ParetoObjective(weights={"f": bargaining_weight, "m": 1.0}),
)
```

A weight is a constant or a function of the regime's **states** and of `period` / `age`
— so a state-dependent bargaining position works. Every other argument a weight names
becomes a free parameter under the regime's `pareto_objective` key in
`get_params_template()`, estimated like anything else. That cuts both ways: an argument
spelled like one of the regime's other functions does not receive that function's
output, it becomes a parameter you have to supply. A weight may not read an *action*: a
weight that varied with the choice would state a different objective per candidate,
whose maximizer is a Pareto optimum of no fixed weighting.

Declaring the weights rather than writing the sum as an ordinary function is what lets
the engine own what a Pareto weight means — one per stakeholder, finite and
non-negative, with a strictly positive total, normalized cell by cell, and multiplied in
zero-safely. That last one matters: a stakeholder carrying zero weight may still hold an
admissible `-inf` of her own, and `0.0 * -inf` is `NaN`, which through the argmax would
let the wrong partner decide the household's choice.

Normalization is `"pointwise"` by default, which rescales the declared weights to sum to
one wherever the objective is evaluated, so a state-dependent weight cannot change the
scale of the household's objective from cell to cell. `normalization="none"` uses the
weights as written, for a scalarization whose level is itself meant to vary.

## Participation: a constraint that reads values

A household is feasible only where each partner is at least as well off inside it as
alone. "Alone" is that partner's own regime, in the *same* period — not next period's
continuation. `ProjectedRegimeValue` reads it, and `ValueDependentConstraint` compares
against it:

```python
from lcm import ProjectedRegimeValue, ValueDependentConstraint


def participation_f(Q_f, V_alone_f, slack):
    return Q_f >= V_alone_f - slack


constraints = {
    "affordable": consumption_within_wealth,
    "participation_f": ValueDependentConstraint(
        predicate=participation_f,
        references={
            "V_alone_f": ProjectedRegimeValue(
                regime="single_f",
                projection={"wealth": half_of_couple_wealth},
            )
        },
    ),
}
```

The predicate may read `Q_<s>` for each stakeholder, each key of its **own**
`references`, and ordinary states, actions, functions and parameters. `projection` owes
one entry per state of the *referenced* regime; the value is interpolated at the
resulting coordinates. A constraint-local projection may not introduce free parameters —
that is checked at model construction, naming the reference, the projection and the
offending argument.

The regime's mask is the AND of the ordinary constraints and the value-dependent ones. A
cell whose mask is empty publishes the regime's **dissolution flag** and the `-inf`
sentinel: there is no viable household there.

`solve(return_dissolution_flags=True)` hands those flags back, and `simulate` takes them
as `period_to_regime_to_dissolution_flags=`. A gate that reads `D_target` needs them;
let `simulate` solve for you and they are threaded automatically.

## Marrying and dissolving: a transition with a gate

A raw transition between regimes with different stakeholder structure stays rejected —
there is no rule that would say who the arriving row is. A `ValueDependentTransition`
supplies one:

```python
from lcm import ProjectedRegimeValue, StakeholderRoute, ValueDependentTransition
from lcm.transition import MarkovTransition

single_f = Regime(
    transition={
        "couple": ValueDependentTransition(
            probability=MarkovTransition(meets_a_partner),
            gate=mutual_consent,
            routes={
                "her": StakeholderRoute(
                    target_stakeholder="f",
                    fallback=ProjectedRegimeValue(
                        regime="single_f",
                        projection={"wealth": half_of_couple_wealth},
                    ),
                )
            },
            gate_references={
                "V_alone_f": ProjectedRegimeValue(regime="single_f", projection=...),
                "V_alone_m": ProjectedRegimeValue(regime="single_m", projection=...),
            },
        ),
        "single_f": MarkovTransition(meets_nobody),
    },
    states={"wealth": single_wealth},
    state_transitions={"wealth": next_single_wealth},
    actions={"consumption": consumption},
    functions={"utility": utility_single},
)
```

`probability` and `gate` are two distinct operations. The first decides whether this
target edge is attempted at all — it is exactly what a plain `transition` entry accepts.
The second decides, having arrived at the target's coordinates, whether the row keeps
that target or takes its route's fallback.

The gate is a **Boolean** predicate on the *target* regime's grid. It may read the
target's value — `V_target` for a singleton target, `V_target_<s>` per stakeholder for a
collective one — the target's dissolution flag `D_target`, each key of
`gate_references`, ordinary target states and params, and the target fold's `period` /
`age`.

Two of those carry a timing that is worth keeping straight, because they are checked in
different places. `D_target` exists only for a collective target, and a gate that names
it on a singleton target is refused **while the model is built**. The Boolean
requirement is not a build-time check: the dtype is examined where the gate is actually
evaluated, so a gate returning a probability is refused on the first `solve()` rather
than at construction. The branch is selected with a strict `where`, in which every
nonzero value is true, so a `0.25` would otherwise open the edge for every row.

### The key is always the gate-open target

This is the rule that decides how a dissolution edge is written. A couple that stays
together continues in the couple regime, so **that** is the key, under
`gate = ~D_target`:

```python
couple = Regime(
    transition={
        "couple": ValueDependentTransition(
            probability=MarkovTransition(stays_married),
            gate=no_dissolution,  # ~D_target
            routes={
                "f": StakeholderRoute(
                    target_stakeholder="f",
                    fallback=ProjectedRegimeValue(regime="single_f", projection=...),
                ),
                "m": StakeholderRoute(
                    target_stakeholder="m",
                    fallback=ProjectedRegimeValue(regime="single_m", projection=...),
                ),
            },
        ),
    },
    states={"wealth": couple_wealth},
    state_transitions={"wealth": next_couple_wealth},
    actions={"consumption": consumption},
    functions={"utility": household_utility},
)
```

Keying it by `single_f` instead would send *both* partners to `single_f` whenever the
couple stays together, because the key is where the open branch goes.

Each route owns four destinations, and simulation carries all four: the open branch's
regime (the key) and role (`target_stakeholder`), and the closed branch's regime and
role (`fallback.regime` and `fallback.stakeholder`). A row landing in a singleton regime
carries no role.

### Entering on the target's own terms

The candidate target state is computed by the *source's* law of motion, so a law that is
right for staying single can be wrong for marrying. Two people pool their wealth, and
the couple's grid is on that scale — declare the entry law per target:

```python
state_transitions = {
    "wealth": {
        "couple": pooled_wealth,  # 2 * (1 + r) * (wealth - consumption)
        "single_f": next_single_wealth,  # (1 + r) * (wealth - consumption)
        "single_f_terminal": next_single_wealth,
    }
}
```

Without it every bride lands at the bottom of the couple's grid, where the household is
infeasible — and the consent gate then shuts for a reason that is an artifact of the law
rather than an economic decision.

### Landing between grid nodes

`off_grid` says what the edge promises when the realized target coordinate falls between
the target's nodes:

- `"pointwise"` (the default) reads every operand at the landing point and applies the
  gate there, in both phases. The operands are interpolated, so the value carries the
  ordinary interpolation error of any continuation — but it is a value one branch really
  delivers, and the branch the solve priced is the branch simulation routes down.
- `"reject"` demands that no such point exists: the model refuses to build unless the
  target regime's grid is reached exactly, i.e. it carries no continuous state. Declare
  it where a straddled gate would be an economic error rather than an approximation.

## Simulating: every row carries its own role

Stakeholder identity is per subject, not per call. A row's role picks its leg on a gated
edge; the branch it takes then sets the role it carries onward.

- A subject starting in a **singleton** regime has no role.
- A subject starting in a **collective** regime declares one, through
  `initial_conditions["own_stakeholder"]`, whenever a route it can still reach turns on
  the role — that is, whenever some collective regime in the starting regime's forward
  closure declares a value-dependent transition with more than one route. A row keeps
  its role across an ordinary regime transition, so a two-leg transition the cohort runs
  into later counts; one in a regime the cohort can never arrive at does not, and no
  seed is required for it.
- Entering a collective regime through a value-dependent transition sets the role from
  the route's `target_stakeholder`; leaving through the closed branch sets it from the
  fallback's `stakeholder`, or clears it for a singleton destination.

```python
initial_conditions = {
    "wealth": jnp.asarray([...]),
    "age": jnp.zeros(n_subjects),
    "regime_id": jnp.full(n_subjects, model.regime_names_to_ids["couple"], jnp.int32),
    "own_stakeholder": jnp.asarray([model.stakeholder_names_to_ids["f"], ...]),
}
```

The role is published as an `own_stakeholder` column beside the columns that identify a
row, carrying the stakeholder's label and missing wherever the row occupies no role. One
cohort can therefore hold both partners, and a model whose collective regimes use
disjoint role vocabularies — `carer` and `ward` in one, `f` and `m` in another — is
expressible.

## Parameters

Everything lands where the declaration is:

```python
template["couple"]["utility_f"]["crra"]  # a stakeholder's own utility
template["couple"]["pareto_objective"]["weight_f"]  # the objective's free arguments
template["couple"]["participation_f"]["slack"]  # a value constraint's predicate
template["single_f"]["couple"]["gate"]["bonus"]  # one edge's gate
```

Edge parameters nest under the **target** name and are discovered from that edge's own
target states and injected operands — never from a union over unrelated regimes.

## What this costs

See [the resource contract](../development/collective_resource_contract.md) for the
workloads these declarations are benchmarked on and which axis each cost is allowed to
grow along.
