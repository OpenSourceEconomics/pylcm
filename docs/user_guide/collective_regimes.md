---
title: Several members and the decision on the composition
---

# Several members and the decision on the composition

An agent in a regime with one member maximizes its own value. A regime with several
members works differently. Its members take one action together. They stay together only
while each of them wants to. Whether a couple forms or separates depends on the value of
each member's outside option. Three declarations cover this, and each goes in a slot an
ordinary model already has.

| Declaration         | Goes in                          | Says                                                                       |
| ------------------- | -------------------------------- | -------------------------------------------------------------------------- |
| `CollectiveUtility` | `functions["utility"]`           | who the members are, and how their action values are traded off            |
| `Gate`              | `Transition(gates=...)`          | which branch within a target, from the landing state                       |
| `CompositionAction` | `Regime(composition_action=...)` | the decision on the composition: formation (with `Consent`) and separation |

`ProjectedRegimeValue` reads another regime's value in the period a gated row lands in,
for a route's fallback. A decision that compares values is not a `Gate`. It is the
`CompositionAction` of the regime that decides. A couple forms through the
`CompositionAction` of a single's regime, with a `Consent` rule. A couple separates
through its own `CompositionAction`.

The complete worked model is `lcm_examples.collective_household`. This page walks
through the pieces. The exact declaration contracts are in
[Collective regimes](../reference/collective_regimes.md), and a staged build-up of one
model is in the [collective example](../examples/collective_regimes.md).

## A regime with two members

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

The members maximize one objective over the feasible actions. Each member's own action
value $Q_\kappa$ is read at that common choice. Without an `objective`, the members
carry equal Pareto weights. To weigh them differently, declare a `ParetoObjective`:

```python
from lcm import ParetoObjective

CollectiveUtility(
    utilities={"f": her_utility, "m": his_utility},
    objective=ParetoObjective(weights={"f": her_pareto_weight, "m": 1.0}),
)
```

A Pareto weight is a constant or a function of the regime's **states** and of `period` /
`age`, so a Pareto weight that depends on a state works. Every other argument a Pareto
weight names becomes a free parameter under the regime's `pareto_objective` key in
`get_params_template()`, estimated like anything else. That cuts both ways: an argument
spelled like one of the regime's other functions does not receive that function's
output, it becomes a parameter you have to supply. A Pareto weight may not read an
*action*. A Pareto weight that varied with the choice would state a different objective
per candidate, whose maximizer is a Pareto optimum of no fixed weighting.

Declaring the Pareto weights rather than writing the sum as an ordinary function lets
the engine own what a Pareto weight means: one per member, finite and non-negative, with
a strictly positive total, normalized cell by cell, and multiplied in zero-safely. That
last one matters. A member with a zero Pareto weight may still hold an admissible `-inf`
of its own, and `0.0 * -inf` is `NaN`. Through the argmax, that `NaN` would let the
wrong member decide the common choice.

Normalization is `"pointwise"` by default. It rescales the declared Pareto weights to
sum to one wherever the objective is evaluated, so a Pareto weight that depends on a
state cannot change the scale of the members' objective from cell to cell.
`normalization="none"` uses the Pareto weights as written, for a scalarization whose
level is itself meant to vary.

### Taste shocks when one member decides

A regime with several members may declare `taste_shocks=ExtremeValueTasteShocks()` when
exactly one member has a positive Pareto weight. That member decides, as an agent in a
regime with one member and the same taste shocks would. Its value is its own smoothed
expected maximum over the discrete actions, after the maximum over the continuous ones.
Every other member's value is the mean of its own action value under the deciding
member's choice probabilities. That value is read at the deciding member's best
continuous action for each discrete one. Simulation draws the deciding member's choice
exactly as for a regime with one member. `model.lookup_policy` returns, per discrete
action, the deciding member's conditional value `Qc`, its choice probability and the
continuous actions all members share, and one value per member, in member order.
Constant Pareto weights must give exactly one member a positive Pareto weight. A Pareto
weight declared as a function is checked at every solve, at every cell of the regime's
grid, so a state that holds the Pareto weight may hand the decision from one member to
the other.

A regime with taste shocks, with one member or several, cannot have its own composition
action. Drop `taste_shocks` on it, or declare the composition action on a regime without
taste shocks. Another regime's composition action may read it, as a destination or as
the counterparty of a `Consent`. There, the deciding member's value is its smoothed
expected maximum, and every other member's value is its mean under the deciding member's
choice probabilities. Where every feasible alternative is worth minus infinity to the
deciding member, every member's value is minus infinity, and a simulation that reaches
that cell raises `NoFeasibleActionError`. A regime with several members and taste shocks
may observe current-period `shocks`.

## Separating in the current period: where the members compare

A regime's `CompositionAction` holds the decision to stay or to separate. The members
make it before the period's action. The `rule` says at which actions each member
compares its value of staying with the value of its outside option (the
[reference](../reference/model_and_regime.md#api-current-exits) has the details):

- `PCOnActions()`, unilateral divorce with participation constraints on actions: every
  stay action is tested. The members take their best action that every member accepts.
  They separate only if there is no such action.
- `PCOnWeight()`, unilateral divorce with participation constraints on the Pareto
  weight: only the members' unconstrained optimum is tested. Without a `state`, the
  members separate when any member refuses there. Members who stay act at their optimum.
  With a `state`, the Pareto weight first moves toward the refusing member; see
  [Moving the Pareto weight before separating](#moving-the-pareto-weight-before-separating).
- `MutualConsent(transfer=...)`, mutual consent divorce: the members compare at the
  optimum. They separate only if both agree, at the default division or at a transfer
  that one member offers.

Each rule takes `tie`. A member who is exactly indifferent between staying and leaving
stays under `tie="stay"`, the default, and leaves under `tie="leave"`.

The value of a member's outside option is its destination's value without the
composition action at the state the destination's `projection` gives. A member who
leaves lands there. That state follows the destination grid's `off_grid` rule, as a
realised next state does. A `"clip_state"` grid caps it to its outer nodes, both for the
value of the outside option and for the landing. A `"raise"` grid refuses a read outside
it, even where the members then stay. A member whose destination has no feasible action
there has an outside option worth minus infinity. It accepts every feasible stay action,
whatever the rule's `tie` says, also where its own value of staying is minus infinity.

Under mutual consent divorce, the members compare at the optimum and can shift the
division before they decide. `Transfer` names this division as an argument that the
projections and utility adjustments of the destinations read:

```python
from lcm import (
    CompositionAction,
    Destination,
    LinSpacedGrid,
    MutualConsent,
    Transfer,
)


def wife_assets(*, wealth, share):
    return share * wealth


def husband_assets(*, wealth, share):
    return (1.0 - share) * wealth


def equal_split():
    return 0.5


separation = CompositionAction(
    rule=MutualConsent(
        transfer=Transfer(
            name="share",
            candidates=LinSpacedGrid(start=0.0, stop=1.0, n_points=21),
            default=equal_split,
        )
    ),
    destinations={
        "f": Destination(regime="single_f", projection={"wealth": wife_assets}),
        "m": Destination(regime="single_m", projection={"wealth": husband_assets}),
    },
)
```

The rule for the transfer evaluates the values of the outside options at a transfer:

1. Each member compares its value of staying at the optimum with the value of its
   outside option at the `default` transfer. In every comparison of the rule, an
   indifferent member does what `tie` says: it prefers staying under `"stay"` and
   leaving under `"leave"`. A member whose destination has no feasible action prefers
   staying wherever the members have a feasible action.
1. If no member wants to leave, the members stay. If both want to leave, they separate
   at the default, without a search of the candidates. Where a destination has no
   feasible action there, the node has no value.
1. If exactly one member wants to leave, the other member is compensated. Among the
   candidates beyond the default on the other member's side (increasing values favour
   the first member), take the nearest at which the other member prefers the value of
   its outside option to staying and every destination has a feasible action. The
   members separate there if the member who wants to leave prefers the value of its
   outside option there to staying. Otherwise they stay; no farther candidate is tried.
   Without such a candidate, they stay.

A shifted transfer is searched on the candidates only, never between them. A transfer
needs a regime with two members. The default is one number at the members' node before
the action. It may read states, parameters, the period, functions of the regime and
shocks the regime itself draws. It is clipped to the candidates' range and need not be a
candidate. Members who separate land where the projections put them at the selected
transfer, and receive the utility adjustments evaluated there.

## Moving the Pareto weight before separating

A transfer shifts what the members get if they separate. Under limited commitment, the
members shift what they get if they **stay**: a member who would rather leave receives a
higher Pareto weight inside the couple. Declare the Pareto weight as an ordered discrete
state of the couple and name it in `PCOnWeight(state=...)` (the
[reference](../reference/model_and_regime.md#api-current-exits) has the contract):

```python
from lcm import (
    CompositionAction,
    Destination,
    DiscreteGrid,
    ParetoObjective,
    PCOnWeight,
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
    composition_action=CompositionAction(
        rule=PCOnWeight(state="weight"),
        destinations={
            "f": Destination(
                regime="single_f", projection={"wealth": half_of_couple_wealth}
            ),
            "m": Destination(
                regime="single_m", projection={"wealth": half_of_couple_wealth}
            ),
        },
    ),
)
```

The grid order of the state sets the direction. A higher node favours the first member
in `utilities` (`f` here), a lower node the second. Here the higher node is the larger
Pareto weight on her utility, so the grid and the objective agree. The engine reads only
the order of the nodes, not what the objective does with them.

At every node of the state, the members' optimum is computed without the composition
action, with all other states held at their current values. Each member's value at that
optimum is compared with the value of the member's outside option. A member accepts a
node when its value there is above the value of its outside option, or equal to it under
the default `tie="stay"`. Under `tie="leave"`, an indifferent member refuses. A member
whose destination has no feasible action accepts every node at which the members have a
feasible action. No member accepts a node at which the members have no feasible action.
Starting from the members' current node:

1. If both members accept it, the members stay there and act at its optimum.
1. If both refuse it, the members separate.
1. If exactly one refuses, the Pareto weight moves one node at a time toward that
   member. The members stay at the first node both members accept, and act at that
   node's optimum.
1. The walk ends in separation instead if it reaches the end of the grid first, or a
   node the other member refuses. A node further along that both would accept does not
   reopen the path.

The values of the outside options are those at the members' current states, the current
node included. They are read once and do not change along the walk. A projection that
read the Pareto weight state would therefore see the current node, not the node being
tried. Members who separate go where the destinations put them, as under the other
rules. They need a feasible action in every destination; without one, the cell has no
value. Where the composition action's `available` does not hold, the Pareto weight does
not move, and the members stay at their current node.

The node the members stay at is their state for the rest of the period. It supplies the
action, each member's value, and the state the laws of motion read. In a regime that
continues, declare the law of the Pareto weight state as `fixed_transition`, so that the
node also carries into the next period. The model refuses any other law, or a joint
transition that updates the state, when it is built. A simulated row therefore shows the
node the members stay at, both in the period the Pareto weight moved and in the next
one.

What `PCOnWeight(state=...)` requires:

- a regime with exactly two members;
- a state declared on an ordered `DiscreteGrid`: a grid declared by its `values`, or by
  a `@categorical(ordered=True)` class. An unordered or continuous state, or a name that
  is not a state, is refused.

The search runs on the grid's nodes. The rule does not solve for a continuous Pareto
weight at which a member is exactly indifferent; refine the grid where that point
matters. `model.lookup_policy` refuses a period in which the Pareto weight of the regime
can move, because it cannot report a changed state or a separation. Read the decision
from `model.simulate`.

## Regimes that differ only in their composition action

Couples that face the same budget, preferences and laws of motion can still differ in
how they may separate. One may keep a member who is indifferent, and another may let
that member leave. They can also differ in where they go next period. Each is its own
regime, because initial conditions, edges and results name regimes, and each declares
its slots itself. Ordinary Python builds them from one definition:

```python
from lcm import CompositionAction, Destination, PCOnActions, Regime


def couple(rule):
    return Regime(
        states={"wealth": couple_wealth},
        state_transitions={"wealth": next_couple_wealth},
        actions={"consumption": consumption},
        functions={
            "utility": CollectiveUtility(utilities={"f": her_utility, "m": his_utility})
        },
        composition_action=CompositionAction(
            rule=rule,
            destinations={
                member: Destination(
                    regime=f"single_{member}",
                    projection={"wealth": half_of_couple_wealth},
                )
                for member in ("f", "m")
            },
        ),
    )


regimes = {
    "couple_stays": couple(PCOnActions()),
    "couple_leaves": couple(PCOnActions(tie="leave")),
    # ... the singles
}
```

Each regime's value follows from its own composition action and edges. Parameters stay
per regime. The couple's functions appear under each regime's name in
`get_params_template()`, here once under `couple_stays` and once under `couple_leaves`,
and are set separately.

## Entering a regime with several members in a role

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
a row landing where the members have no feasible action raises `NoFeasibleActionError`.
To send such a row somewhere else depending on the landing state, declare a gate. To
make the change of regime depend on values, declare it as the composition action of the
regime that decides.

## Forming a couple: formation by consent

Without a role, a raw transition between regimes with different members is rejected: no
rule says who the arriving row is. A marriage that both partners must want is decided in
the current period, by the single who meets a partner. It is a `CompositionAction` of
her regime toward the couple, with a `Consent` rule. `lcm_examples.collective_household`
declares it on each single's regime:

```python
single_f = Regime(
    states={"wealth": wealth},
    state_transitions={"wealth": next_single_wealth_f},
    actions={"consumption_share": consumption_share},
    functions={"consumption": consumption, "utility": single_utility},
    shocks=("meeting_f",),
    composition_action=CompositionAction(
        destinations={
            "self": Destination(
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

- The destination `"self"` takes the single into `couple` as its member `f`, at the
  pooled wealth that `projection` computes from her state.
- The `Counterparty` is the partner she meets: his own regime `single_m`, read at the
  wealth `projection` assigns him, in the role `m` he would take in the couple.
- Under `Consent`, she enters the couple only where both prefer it. Her value as `f` in
  the formed couple is compared with her value of staying single. His value as `m` is
  compared with his value of staying in `single_m` at the projected wealth. Both
  comparisons are strict.
- `available` says when the decision to form a couple is open at all — here when the
  meeting shock `meet_f` says she met someone. `observes` names the shocks the decision
  sees.

The formed couple applies its own rule for staying, so a couple that would separate at
once is priced as one. The decision to stay or to separate is the couple's own
`CompositionAction`, with `PCOnActions`, `PCOnWeight` or `MutualConsent`; see
[Separating in the current period](#separating-in-the-current-period-where-the-members-compare).
A destination, and the regime of a counterparty, may be a regime that agents are in,
even one with a composition action of its own. Outside a formation by consent, the
decision reads a destination's value without the composition action. It reads a
counterparty's regime in the same way. No regime is needed only to be read.

### A gate routes on states

A `Gate` in `Transition(gates=...)` decides between a target and its routes' fallbacks
from the landing state alone: its `predicate` is a **Boolean** function on the *target*
regime's grid that reads ordinary target states and params and the target fold's
`period` / `age`.

One `Gate` per target holds at every age the target is reached and in both phases; it is
never wrapped in `ByAge` or `Phased`. When the source's law is `Phased`, only the law
differs between the phases. This permits a perceived probability in solve and an actual
probability in simulation without changing what the gate decides. When the edges are
`Phased` with a `Transition` in each phase, a target reached in both carries the equal
`Gate` in both or none. A route's `fallback` is the one part of a gate that may be
`Phased`.

A gate never reads a value. A predicate naming `V_target`, `V_target_<s>` or `D_target`
raises `GateReadsValueError` when the `Gate` is declared, and a `Gate` takes no
`references`. Write a comparison of values as a `CompositionAction` of the regime that
decides.

A gate must return a Boolean, such as a comparison or `True`. The requirement is not a
build-time check: the dtype is examined where the gate is actually evaluated, so a gate
returning an integer or a floating probability is refused on the first `solve()` rather
than at construction. The branch is selected with a strict `where`, in which every
nonzero value is true, so a `0.25` would otherwise open the edge for every row.

### The key is always the open branch

This rule decides how a separation edge is written. A couple that stays together
continues in the couple regime, so **that** is the key, with a predicate that holds
where the couple continues — here, where its wealth stays above a floor:

```python
def keeps_couple(*, wealth: ContinuousState, wealth_floor: float) -> BoolND:
    return wealth >= wealth_floor


edges = {
    "couple": Transition(
        law={"couple": StochasticTransition(func=stays_married)},
        gates={
            "couple": Gate(
                predicate=keeps_couple,
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

Without it, every row entering the couple on this edge lands at the bottom of the
couple's grid, where the couple has no feasible action. A composition action needs no
such law: its destination's `projection` computes the landing state.

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

Member identity is per agent, not per call. A row's role picks its route on a gated
transition; the branch it takes then sets the role it carries onward.

- An agent starting in a **singleton** regime has no role.
- An agent starting in a **collective** regime declares one, through
  `initial_conditions["own_member"]`, whenever a route it can still reach turns on the
  role. That is the case whenever some regime in the starting regime's forward closure,
  the starting regime included, declares a gated transition with more than one route or
  a `CompositionAction` with destinations for more than one member, such as a separation
  that sends `f` to `single_f` and `m` to `single_m`. A row keeps its role across an
  ordinary regime transition, so such a transition or composition action that the cohort
  runs into later counts. One in a regime the cohort can never arrive at does not, and
  no seed is required for it.
- Entering a collective regime through `roles` sets the declared role, through a gated
  transition the route's `target_member`, and through a composition action the
  destination's `role`. Leaving through the closed branch sets it from the fallback's
  `member`, or clears it for a singleton destination.

```python
initial_conditions = {
    "wealth": jnp.asarray([...]),
    "age": jnp.zeros(n_agents),
    "regime_id": jnp.full(n_agents, model.regime_names_to_ids["couple"], jnp.int32),
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

A simulation evaluates a composition action for a block of agents at once. Every
projection and utility adjustment of the composition action, including the one for each
candidate transfer, is evaluated for every agent of the block. It is discarded where the
agent does not take that branch. A composition action whose projections are expensive
and rarely taken therefore pays for them on every agent.
