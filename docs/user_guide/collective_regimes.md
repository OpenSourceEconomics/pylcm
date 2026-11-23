---
title: Households and value-dependent choice
---

# Households and value-dependent choice

A single decision maker maximizes her own value. A household does not: it takes one
action for two people, it is only viable while both of them want it, and whether it
forms or ends depends on what each member would be worth outside it. Two declarations
cover the household itself, and each goes in a slot an ordinary model already has.

| Declaration         | Goes in                 | Says                                                             |
| ------------------- | ----------------------- | ---------------------------------------------------------------- |
| `CollectiveUtility` | `functions["utility"]`  | who the members are, and how their action values are traded off  |
| `Gate`              | `Transition(gates=...)` | which branch within a target, from the landing state             |
| `Exit`              | `Regime(exit=...)`      | formation (with `Consent`) and dissolution, from compared values |

`ProjectedRegimeValue` reads another regime's value in the *same* period, for a route's
fallback. A move decided by comparing values is not a `Gate`: it is an `Exit` of the
regime that decides. A household forms through a single's `Exit` with a `Consent` rule,
and dissolves through the household's own `Exit`.

The complete worked model is `lcm_examples.collective_household`. This page walks
through the pieces; the exact declaration contracts are in
[Collective regimes](../reference/collective_regimes.md), and a staged build-up of one
is in the [collective example](../examples/collective_regimes.md).

## A household with two members

The `utilities` keys **are** the regime's members, in the order you write them. That
order fixes the trailing axis of the regime's value function and of every published
array, so `value_f` comes before `value_m` in the simulated frame.

```python
from lcm import CollectiveUtility, Regime

couple = Regime(
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

The `CollectiveUtility` remains the raw `functions["utility"]` declaration. Pylcm
derives the member metadata from it and gives the numerical engine a
`decomposed_functions` view with `utility_f` and `utility_m`; it does not replace what
the model author wrote. Each value in `utilities` can be a callable, a
`Phased(solve=..., simulate=...)` pair, or `None`. `None` delegates that member's body
to a `utility_<member>` function supplied at regime or model level; successful model
finalization guarantees that every delegated body arrived.

The household maximizes one objective over the feasible actions and reads off each
member's own action value `Q_s` at that common choice. Omitting `objective` weights the
members equally. To weigh them differently, declare a `ParetoObjective`:

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
the engine own what a Pareto weight means — one per member, finite and non-negative,
with a strictly positive total, normalized cell by cell, and multiplied in zero-safely.
That last one matters: a member carrying zero weight may still hold an admissible `-inf`
of her own, and `0.0 * -inf` is `NaN`, which through the argmax would let the wrong
partner decide the household's choice.

Normalization is `"pointwise"` by default, which rescales the declared weights to sum to
one wherever the objective is evaluated, so a state-dependent weight cannot change the
scale of the household's objective from cell to cell. `normalization="none"` uses the
weights as written, for a scalarization whose level is itself meant to vary.

## Dissolving in the current period: where the members compare

A regime's `Exit` with `rule=Unilateral()` or `rule=MutualConsent()` decides before the
period's action whether the household stays. `compares` says which stay values each
member compares with its outside value (the
[reference](../reference/model_and_regime.md#api-current-exits) has the details):

- `compares="each_action"`, the default without an adjustment: every stay action is
  tested, and the household takes its best action that passes the rule. It separates
  only if none passes.
- `compares="optimum"`: only the household's unconstrained optimum is tested.
  `Unilateral` separates when any member refuses there, `MutualConsent` when every
  member does. A household that stays acts at its optimum.

Under mutual consent the members can also shift the division on separation before they
decide. `Settlement` names it as an argument the exit's projections and utility
adjustments read:

```python
from lcm import Exit, ExitTarget, LinSpacedGrid, MutualConsent, Settlement


def wife_assets(*, wealth, share):
    return share * wealth


def husband_assets(*, wealth, share):
    return (1.0 - share) * wealth


def equal_split():
    return 0.5


dissolution = Exit(
    rule=MutualConsent(
        settle=Settlement(
            name="share",
            candidates=LinSpacedGrid(start=0.0, stop=1.0, n_points=21),
            default=equal_split,
        )
    ),
    targets={
        "f": ExitTarget(regime="single_f", projection={"wealth": wife_assets}),
        "m": ExitTarget(regime="single_m", projection={"wealth": husband_assets}),
    },
)
```

The settlement rule, with outside values evaluated at a settlement:

1. Each member compares its value at the optimum with its outside value at the `default`
   settlement.
1. If nobody wants to leave, the household stays. If both want to leave, it separates at
   the default.
1. If exactly one wants to leave and the other is exactly indifferent at the default,
   the household separates at the default, provided both destinations have a feasible
   action there: no shift is needed to compensate the indifferent member. Under
   `tie="leave"` an indifferent member wants to leave, so this case does not arise.
1. Otherwise, if exactly one wants to leave, the settlement moves toward the other
   member: among the candidates on that member's side of the default (increasing values
   favour the first member), take the nearest at which that member accepts separation.
   The household separates there if the member who wants to leave still gains, and stays
   otherwise. Without such a candidate it stays.

A shifted settlement is searched on the candidates only, never between them. A
settlement needs a regime with two members. The default reads what `available` reads, at
the decision; it is clipped to the candidates' range. A household that separates lands
where the projections put it at the selected settlement and receives the adjustments
evaluated there.

## Renegotiating a bargaining state before dissolving

A settlement shifts what the members get if they separate. Under limited commitment the
members shift what they get if they **stay**: a member who would rather leave is offered
a better position inside the household. Declare that position as an ordered discrete
state of the couple, for example a Pareto weight on a grid, and name it in
`Unilateral(renegotiate=Renegotiate(state=...))` (the
[reference](../reference/model_and_regime.md#api-current-exits) has the contract):

```python
from lcm import (
    DiscreteGrid,
    Exit,
    ExitTarget,
    ParetoObjective,
    Renegotiate,
    Unilateral,
    fixed_transition,
)


def her_weight(*, weight):
    return weight


def his_weight(*, weight):
    return 1.0 - weight


couple = Regime(
    states={
        "wealth": couple_wealth,
        "weight": DiscreteGrid(values=jnp.linspace(0.1, 0.9, 9)),
    },
    state_transitions={
        "wealth": next_couple_wealth,
        "weight": fixed_transition("weight"),
    },
    actions={"consumption": consumption},
    functions={
        "utility": CollectiveUtility(
            utilities={"f": her_utility, "m": his_utility},
            objective=ParetoObjective(weights={"f": her_weight, "m": his_weight}),
        )
    },
    exit=Exit(
        rule=Unilateral(renegotiate=Renegotiate(state="weight")),
        targets={
            "f": ExitTarget(
                regime="single_f", projection={"wealth": half_of_couple_wealth}
            ),
            "m": ExitTarget(
                regime="single_m", projection={"wealth": half_of_couple_wealth}
            ),
        },
    ),
)
```

The state's grid order is the direction of the bargaining: moving to a higher node
favours the first member in `utilities` (`f` here), moving to a lower node the second.
Here the higher node is the larger weight on her utility, so the grid and the objective
agree; the engine reads only the order of the nodes, not what the objective does with
them.

At every node of the state the household's optimum is computed ignoring the exit, with
all other states held at their current values. Each member's value at that optimum is
compared with the member's outside value. A member accepts a node when its value there
is above its outside value, or equal to it under the default `tie="stay"`; `tie="leave"`
makes an indifferent member refuse. A node at which the household has no feasible action
is accepted by nobody. Starting from the household's current node:

1. If both members accept it, the household stays there and acts at its optimum.
1. If both refuse it, the household separates.
1. If exactly one refuses, the state moves one node at a time toward that member. The
   household stays at the first node both members accept, and acts at that node's
   optimum.
1. The walk ends in separation instead if it reaches the end of the grid first, or a
   node the other member refuses. A node further along that both would accept does not
   reopen the path.

The outside values are those at the household's current states, its current node
included. They are read once and do not move along the walk, so a projection that read
the bargaining state would see the current node, not the node being tried. A household
that separates goes where the exit's targets put it, as without renegotiation; it needs
a feasible action in every destination, and without one the cell has no value. Where the
exit's `available` does not hold, nothing is renegotiated and the household stays at its
current node.

The node the household stays at is its state for the rest of the period: it supplies the
action, each member's value, and the state the laws of motion read. In a regime that
continues, declare the bargaining state's law as `fixed_transition` so that the node
also carries into next period; any other law, or a joint transition that updates the
state, is refused when the model is built. A simulated row therefore shows the
renegotiated node, both in the period it moved and in the next one.

What renegotiation requires:

- a regime with exactly two members;
- a state declared on an ordered `DiscreteGrid`: a grid declared by its `values`, or by
  a `@categorical(ordered=True)` class. An unordered or continuous state, or a name that
  is not a state, is refused;
- the rule `Unilateral`, comparing at the optimum: `compares` is `"optimum"` by default
  with `renegotiate`, and `compares="each_action"` is refused.

The search runs on the grid's nodes. Renegotiation does not solve for a continuous
weight at which a member is exactly indifferent; refine the grid where that point
matters. `model.lookup_policy` refuses a period in which the regime can renegotiate,
because it cannot report a changed state or a separation; read the decision from
`model.simulate`.

## Regimes that differ only in their exit

Couples that face the same budget, preferences and laws of motion can still differ in
how they may end the match, one dissolving unilaterally and another only by mutual
consent, or in where they go next period. Each is its own regime, because initial
conditions, edges and results name regimes, and each declares its slots itself. Ordinary
Python builds them from one definition:

```python
from lcm import Exit, ExitTarget, MutualConsent, Regime, Unilateral


def couple(rule):
    return Regime(
        states={"wealth": couple_wealth},
        state_transitions={"wealth": next_couple_wealth},
        actions={"consumption": consumption},
        functions={
            "utility": CollectiveUtility(utilities={"f": her_utility, "m": his_utility})
        },
        exit=Exit(
            rule=rule,
            targets={
                member: ExitTarget(
                    regime=f"single_{member}",
                    projection={"wealth": half_of_couple_wealth},
                )
                for member in ("f", "m")
            },
        ),
    )


regimes = {
    "couple_unilateral": couple(Unilateral()),
    "couple_mutual": couple(MutualConsent()),
    # ... the singles
}
```

Each regime's value follows from its own exit and edges. Parameters stay per regime: the
couple's functions appear under each regime's name in `get_params_template()`, here once
under `couple_unilateral` and once under `couple_mutual`, and are set separately.

## Entering a household in a role

Where nothing is decided on landing — marriage arrives with a probability and both
partners take it — a singleton source names the role its row continues in with `roles`:

```python
edges = {
    "single_f": Transition(
        law={
            "single_f": StochasticTransition(func=meets_nobody),
            "couple": StochasticTransition(func=meets_a_partner),
        },
        roles={"couple": "f"},
    ),
    # ... the other sources' edges
}
```

The single woman's continuation toward `couple` is the couple's value for `f`, and a
simulated row entering `couple` carries the role `f`. The edge is otherwise a plain one:
a row landing where the household has no feasible action raises `NoFeasibleActionError`.
To send such a row somewhere else depending on the landing state, declare a gate; to
make the move depend on values, declare it as an exit of the regime that decides.

## Marrying: a consented exit

Without a role, a raw transition between regimes with different member structure stays
rejected — there is no rule that would say who the arriving row is. A marriage that both
partners must want is decided in the current period, by the single who meets a partner:
it is an `Exit` of her regime toward the household, with a `Consent` rule.
`lcm_examples.collective_household` declares it on each single's regime:

```python
single_f = Regime(
    states={"wealth": wealth},
    state_transitions={"wealth": next_single_wealth_f},
    actions={"consumption_share": consumption_share},
    functions={"consumption": consumption, "utility": single_utility},
    shocks=("meeting_f",),
    exit=Exit(
        targets={
            "self": ExitTarget(
                regime="couple",
                role="f",
                projection={"wealth": pooled_from_f},
            )
        },
        rule=Consent(
            counterparty=Counterparty(
                regime="single_m",
                role="m",
                projection={"wealth": half_pooled_from_f},
            )
        ),
        available=meets_f,
        observes=("meet_f",),
    ),
)
```

- The target `"self"` moves the single into `couple` as its member `f`, at the pooled
  wealth `projection` computes from her state.
- The `Counterparty` is the partner she meets: his own regime `single_m`, read at the
  wealth `projection` assigns him, in the role `m` he would take in the couple.
- `Consent` moves her only where both prefer the couple: her value as `f` in the formed
  couple against her value of staying single, and his value as `m` against his value of
  staying in `single_m` at the projected wealth. Both comparisons are strict.
- `available` says when the move can be made at all — here when the meeting shock
  `meet_f` says she met someone — and `observes` names the shocks the choice sees.

The formed couple applies its own rule for staying, so a couple that would dissolve at
once is priced as one. Dissolution is the household's own `Exit`, with `Unilateral` or
`MutualConsent`; see
[Dissolving in the current period](#dissolving-in-the-current-period-where-the-members-compare).
An exit, and a consent counterparty, may name a regime that subjects are in, even one
with an exit of its own: an exit that needs no consent reads its destination's problem,
and a counterparty its regime's problem, without that regime's exit. No regime is needed
only to be read.

### A gate routes on states

A `Gate` in `Transition(gates=...)` decides between a target and its routes' fallbacks
from the landing state alone: its `predicate` is a **Boolean** function on the *target*
regime's grid that reads ordinary target states and params and the target fold's
`period` / `age`.

One `Gate` per target holds at every age the target is reached and in both phases; it is
never wrapped in `ByAge` or `Phased`. When the source's law is `Phased`, only the law
differs between the phases, which permits a perceived probability in solve and a
realized probability in simulation without changing what the gate decides. When the
edges are `Phased` with a `Transition` in each phase, a target reached in both carries
the equal `Gate` in both or none. A route's `fallback` is the one part of a gate that
may be `Phased`.

A gate never reads a value. A predicate naming `V_target`, `V_target_<s>` or `D_target`
raises `GateReadsValueError` when the `Gate` is declared, and a `Gate` takes no
`references`. Write a comparison of values as an `Exit` of the regime that decides.

A gate must return a Boolean, such as a comparison or `True`. The requirement is not a
build-time check: the dtype is examined where the gate is actually evaluated, so a gate
returning an integer or a floating probability is refused on the first `solve()` rather
than at construction. The branch is selected with a strict `where`, in which every
nonzero value is true, so a `0.25` would otherwise open the edge for every row.

### The key is always the open branch

This is the rule that decides how a dissolution edge is written. A couple that stays
together continues in the couple regime, so **that** is the key, with a predicate that
holds where the household continues — here, where its wealth stays above a floor:

```python
def keeps_household(*, wealth: ContinuousState, wealth_floor: float) -> BoolND:
    return wealth >= wealth_floor


edges = {
    "couple": Transition(
        law={"couple": StochasticTransition(func=stays_married)},
        gates={
            "couple": Gate(
                predicate=keeps_household,
                routes={
                    "f": MemberRoute(
                        target_member="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f", projection=...
                        ),
                    ),
                    "m": MemberRoute(
                        target_member="m",
                        fallback=ProjectedRegimeValue(
                            regime="single_m", projection=...
                        ),
                    ),
                },
            ),
        },
    ),
    # ... the other sources' edges
}
```

Keying it by `single_f` instead would send *both* partners to `single_f` whenever the
couple stays together, because the key is where the open branch goes.

Each route owns four destinations, and simulation carries all four: the open branch's
regime (the key) and role (`target_member`), and the closed branch's regime and role
(`fallback.regime` and `fallback.member`). A row landing in a singleton regime carries
no role.

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

Without it every row entering the couple on this edge lands at the bottom of the
couple's grid, where the household is infeasible. An exit needs no such law: its
target's `projection` computes the landing state.

### Landing between grid nodes

A gate's `off_grid` says what the edge promises when the realized target coordinate
falls between the target's nodes:

- `"pointwise"` (the default) applies the predicate at the landing point, in both
  phases, and interpolates each branch's value there. That value carries the ordinary
  interpolation error of any continuation — but it is a value one branch really
  delivers, and the branch the solve priced is the branch simulation routes down.
- `"reject"` demands that no such point exists: the model refuses to build unless the
  target regime's grid is reached exactly, i.e. it carries no continuous state. Declare
  it where a straddled gate would be an economic error rather than an approximation.

## Simulating: every row carries its own role

Member identity is per subject, not per call. A row's role picks its route on a
value-dependent transition; the branch it takes then sets the role it carries onward.

- A subject starting in a **singleton** regime has no role.
- A subject starting in a **collective** regime declares one, through
  `initial_conditions["own_member"]`, whenever a route it can still reach turns on the
  role — that is, whenever some collective regime in the starting regime's forward
  closure declares a value-dependent transition with more than one route. A row keeps
  its role across an ordinary regime transition, so a two-leg transition the cohort runs
  into later counts; one in a regime the cohort can never arrive at does not, and no
  seed is required for it.
- Entering a collective regime through `roles` sets the declared role, and through a
  value-dependent transition the route's `target_member`; leaving through the closed
  branch sets it from the fallback's `member`, or clears it for a singleton destination.

```python
initial_conditions = {
    "wealth": jnp.asarray([...]),
    "age": jnp.zeros(n_subjects),
    "regime_id": jnp.full(n_subjects, model.regime_names_to_ids["couple"], jnp.int32),
    "own_member": jnp.asarray([model.member_names_to_ids["f"], ...]),
}
```

The role is published as an `own_member` column beside the columns that identify a row,
carrying the member's label and missing wherever the row occupies no role. One cohort
can therefore hold both partners, and a model whose collective regimes use disjoint role
vocabularies — `carer` and `ward` in one, `f` and `m` in another — is expressible.

## Parameters

Everything lands where the declaration is:

```python
template["couple"]["utility_f"]["crra"]  # a member's own utility
template["couple"]["pareto_objective"]["weight_f"]  # the objective's free arguments
template["edges"]["single_f"]["couple"]["predicate"]["bonus"]  # one gate's predicate
```

Edge parameters belong to the edges, not to the source regime: each sits at its
declaration path under `params["edges"][source][target]` and is discovered from that
edge's own target states — never from a union over unrelated regimes. See
[Edge parameter paths](../reference/transitions.md#api-edge-parameters).

## What this costs

See [the resource contract](../development/collective_resource_contract.md) for the
workloads these declarations are benchmarked on and which axis each cost is allowed to
grow along.

A simulation decides an exit for a block of subjects at once. Every projection and
adjustment of the exit, including each settlement candidate's, is evaluated for every
subject of the block and discarded where the subject does not take that branch. An exit
whose projections are expensive and rarely taken therefore pays for them on every
subject.
