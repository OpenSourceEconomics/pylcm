---
title: Households and value-dependent choice
---

# Households and value-dependent choice

A single decision maker maximizes her own value. A household does not: it takes one
action for two people, it is only viable while both of them want it, and whether it
forms or ends depends on what each member would be worth outside it. Three declarations
cover that, and each goes in a slot an ordinary model already has.

| Declaration                | Goes in                 | Says                                                                 |
| -------------------------- | ----------------------- | -------------------------------------------------------------------- |
| `CollectiveUtility`        | `functions["utility"]`  | who the stakeholders are, and how their action values are traded off |
| `ValueDependentConstraint` | `constraints`           | where the cell is feasible, reading values as well as states         |
| `Gate`                     | `Transition(gates=...)` | which branch within a target, from the landing state                 |

`ProjectedRegimeValue` reads another regime's value in the *same* period, for a value
constraint's comparison and for a route's fallback. A move decided by comparing values
is not a `Gate`: it is a participation `Exit` of the source regime, or a
`ParticipationCheck` evaluated on landing.

The complete worked model is `lcm_examples.collective_household`. This page walks
through the pieces; the exact declaration contracts are in
[Collective regimes](../reference/collective_regimes.md), and a staged build-up of one
is in the [collective example](../examples/collective_regimes.md).

## A household with two stakeholders

The `utilities` keys **are** the regime's stakeholders, in the order you write them.
That order fixes the trailing axis of the regime's value function and of every published
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
derives the stakeholder metadata from it and gives the numerical engine a
`decomposed_functions` view with `utility_f` and `utility_m`; it does not replace what
the model author wrote. Each value in `utilities` can be a callable, a
`Phased(solve=..., simulate=...)` pair, or `None`. `None` delegates that stakeholder's
body to a `utility_<stakeholder>` function supplied at regime or model level; successful
model finalization guarantees that every delegated body arrived.

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


def participation_f(*, Q_f, V_alone_f, slack):
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
that is checked when the `Regime` is constructed, naming the reference, the projection
and the offending argument.

The regime's mask is the AND of the ordinary constraints and the value-dependent ones. A
cell whose mask is empty carries the `-inf` sentinel: there is no viable household
there. Where its members go instead is declared on the regime as a participation `Exit`,
or on the incoming edge as a `ParticipationCheck` reading the household's policy
support.

A value constraint cannot excuse a discrete law that leaves its grid. Such a law sends a
state to a code outside `0..n-1`, and an off-grid code has no value to compare with. The
transition check therefore counts every cell the ordinary constraints allow, in the
solve and the simulation, and raises `InvalidStateTransitionError` where a law leaves
its grid there. When a restriction does not read a value, declare it as an ordinary
constraint beside the value-dependent one:

```python
def rest_only(*, work):
    return work == 0


def participation_f(*, Q_f):
    return Q_f >= 0.0


constraints = {
    "rest_only": rest_only,  # value-free: an ordinary constraint
    "participation_f": ValueDependentConstraint(predicate=participation_f),
}
```

Written as one predicate, `(work == 0) & (Q_f >= 0.0)`, the same restriction is refused
when only `work == 1` sends a discrete state off its grid.

## Marrying and dissolving: a participation check on landing

A raw transition between regimes with different stakeholder structure stays rejected —
there is no rule that would say who the arriving row is. A `ParticipationCheck` supplies
one. It lives on the source's edges in `Model(edges=...)`, in the `Transition`'s
`participation`, keyed by the destination it decides on. Its `reads` name every value
the predicate compares, each a `ParticipationRead` of a regime at projected coordinates:

```python
from lcm import (
    ParticipationCheck,
    ParticipationRead,
    ProjectedRegimeValue,
    StakeholderRoute,
    StochasticTransition,
    Transition,
)

single_f = Regime(
    states={"wealth": single_wealth},
    state_transitions={"wealth": next_single_wealth},
    actions={"consumption": consumption},
    functions={"utility": utility_single},
)

edges = {
    "single_f": Transition(
        law={
            "couple": StochasticTransition(func=meets_a_partner),
            "single_f": StochasticTransition(func=meets_nobody),
        },
        participation={
            "couple": ParticipationCheck(
                predicate=mutual_consent,
                routes={
                    "her": StakeholderRoute(
                        target_stakeholder="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f",
                            projection={"wealth": half_of_couple_wealth},
                        ),
                    )
                },
                reads={
                    "V_couple_f": ParticipationRead(
                        kind="IntegratedFull", regime="couple", stakeholder="f"
                    ),
                    "V_couple_m": ParticipationRead(
                        kind="IntegratedFull", regime="couple", stakeholder="m"
                    ),
                    "V_alone_f": ParticipationRead(
                        kind="IntegratedFull", regime="single_f", projection=...
                    ),
                    "V_alone_m": ParticipationRead(
                        kind="IntegratedFull", regime="single_m", projection=...
                    ),
                },
            ),
        },
    ),
    # ... the other sources' edges
}
```

The law names its targets, so the `Transition` derives its destinations from it: each
key at every non-final age, and each route's fallback regime wherever its decided target
is reached. Supplying `targets` as well is allowed only if it says exactly the same.

The law and the decision are two distinct operations. The law's `"couple"` cell decides
whether this target edge is attempted at all, like any other per-target cell. The
decision, having arrived at the target's coordinates, says whether the row keeps that
target or takes its route's fallback. A move decided in the current period instead — the
household dissolving because a partner walks out now — is a participation `Exit` in the
source's `Regime(exits=...)`; see the [migration guide](migrating_dated_regimes.md).

### A gate routes on states

A `Gate` in `Transition(gates=...)` takes the same `routes` as a participation check,
but its `predicate` decides from the landing state alone: a **Boolean** function on the
*target* regime's grid that reads ordinary target states and params and the target
fold's `period` / `age`.

One `Gate` per target holds at every age the target is reached and in both phases; it is
never wrapped in `ByAge` or `Phased`. When the source's law is `Phased`, only the law
differs between the phases, which permits a perceived probability in solve and a
realized probability in simulation without changing what consent means. When the edges
are `Phased` with a `Transition` in each phase, a target reached in both carries the
equal `Gate` in both or none. A route's `fallback` is the one part of a gate that may be
`Phased`.

A gate never reads a value. A predicate naming `V_target`, `V_target_<s>` or `D_target`
raises `GateReadsValueError` when the `Gate` is declared, and a `Gate` takes no
`references`. Write a comparison of values as a `ParticipationCheck` with named reads,
or as an `Exit` of the source regime.

The Boolean requirement is not a build-time check: the dtype is examined where the gate
is actually evaluated, so a gate returning a probability is refused on the first
`solve()` rather than at construction. The branch is selected with a strict `where`, in
which every nonzero value is true, so a `0.25` would otherwise open the edge for every
row.

### The key is always the open branch

This is the rule that decides how a dissolution edge is written. A couple that stays
together continues in the couple regime, so **that** is the key, with a predicate that
holds where the household still has a feasible action:

```python
edges = {
    "couple": Transition(
        law={"couple": StochasticTransition(func=stays_married)},
        participation={
            "couple": ParticipationCheck(
                predicate=has_policy_support,  # target_support
                reads={
                    "target_support": ParticipationRead(
                        kind="Participating",
                        regime="couple",
                        stakeholder="f",
                        field="support",
                        projection=...,
                    )
                },
                routes={
                    "f": StakeholderRoute(
                        target_stakeholder="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f", projection=...
                        ),
                    ),
                    "m": StakeholderRoute(
                        target_stakeholder="m",
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

The same rule holds for a `Gate`. Each route owns four destinations, and simulation
carries all four: the open branch's regime (the key) and role (`target_stakeholder`),
and the closed branch's regime and role (`fallback.regime` and `fallback.stakeholder`).
A row landing in a singleton regime carries no role.

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
infeasible — and the consent decision then shuts for a reason that is an artifact of the
law rather than an economic decision.

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

(participation-check-depth)=

## Participation checks over several periods

An [`ParticipationCheck`](../reference/model_and_regime.md#api-participation-checks) is
a participation check at the date a scheduled transition lands. The transition produces
the landing state, the check compares values there, and the row then either takes the
target or its route's fallback. `lcm_examples.collective_household` declares two:

- a **continuing couple** is re-tested every period: it stays together where its
  participating policy still has support at the new wealth, and dissolves into two
  singles at half the household's wealth otherwise;
- a **new match** is checked once, on landing: the couple forms only if both partners
  are better off in the household than single at the landing.

A household choosing today has to price these checks: its savings depend on what each
member will be worth after tomorrow's check. Tomorrow's value depends on the household's
own choice tomorrow, which prices the check the day after, and so on to the horizon.
Pylcm evaluates the first of these checks exactly and every later one from a stored
table of what the check compares. This section says what that means for your results.

### What is exact

- **The next check, at every realized landing.** A regime choosing at date $d$ prices
  the check at $d+1$ by rebuilding the target's action problem exactly at each landing
  state, comparing there, and taking the branch the comparison selects. Simulation does
  the same at each subject's realized landing, so whether a row marries, stays or
  dissolves is decided by a comparison at that landing, never read off an interpolated
  table of verdicts.
- **Landings on grid nodes.** Where every landing falls on a node of the target's grid,
  the solution is identical, bit for bit, to rebuilding every later check at its landing
  as well. This holds on a discrete-wealth twin of the household example in both
  precisions.

The rebuilt action problem still reads its own continuation from the stored value
function by interpolation, like every `GridSearch` problem. "Exact" means the check adds
no approximation of its own.

### What is interpolated

The action problem rebuilt at $d+1$ contains the next check, at $d+2$. Pylcm does not
rebuild that one in turn. It reads a **stored surface** instead, tabulated at every node
of the target's grid at $d+2$. The surface does not store the verdict or the value after
it. It stores what the check consumes, separately:

- each member's value if the row takes the target (the **accept branch**) and if it
  takes its route's fallback (the **reject branch**);
- every operand the check compares: each declared read's value, or its policy support
  when the read asks for `field="support"`.

At a landing between nodes each of these is interpolated on its own, exactly as a value
function is, and the check is run again on the interpolated operands at that landing.
Each member then takes the branch the check selects there. On a node every stored entry
is that node's own, so the read is the check the rebuild takes at that node.

Policy support is a yes-or-no answer, so it is stored as an indicator, one or zero. Read
between nodes, it counts as support where the interpolated indicator is at least one
half: exact on nodes, and in a cell whose two ends disagree the switch falls halfway
between them. A value read off an answer with support is the support-weighted mean of
the supported nodes around the landing, so a node without support, whose value is
$-\infty$, neither enters it nor makes it infinite.

A check that observes realized physical information (an `observes=` on the decision)
cannot be run again between nodes, because the realization it reads exists only at a
node. Its surface stores the value each member holds after the check instead, integrated
over that information at each node, and is interpolated like a value function; where
that check switches, a member's value can jump inside the cell, with the error up to the
jump that the next section describes for a check on support.

A node whose check fails its integrity test stores `NaN` in every entry, so any value
read from it is `NaN` as well rather than a silently wrong number.

This affects a period only when

- at least two participation dates lie ahead of it, so the checks follow each other, and
- landings fall between grid nodes, which needs a continuous target state.

In a four-period household model only period 0 is affected; periods 1 to 3 are
unchanged. In a longer model every period up to the last few is.

Grid nodes of today's value function are not exempt. A node's value prices landings
tomorrow, and those generally fall between nodes of the next date's surface.

Saved solutions carry the surfaces beside the value functions, so a loaded solution
simulates exactly like the one that wrote it.

### Why: the cost of rebuilding every check

Rebuilding every check at its landing nests one action problem inside the next, down to
the horizon. Each further participation date multiplies the cost of a period by roughly
the target's number of actions times its reads and lottery nodes. In the household
example with 10 wealth nodes and 5 consumption shares, going from three to four periods
raised the couple's period-0 query time from 57 ms to 1048 ms on one CPU machine, and
the four-period solve took 84.9 s with 10.9 GB peak memory.

With stored surfaces each period prices exactly one check, whatever the horizon, so the
solve grows linearly in the number of periods. The same four-period solve takes 37.0 s
and 3.8 GB peak memory, compiling from an empty cache. A surface holds a few entries per
node instead of one value per member (eight instead of two for the household's
continuing couple), which is small next to the target's own value function and its
action problem.

There is no option to rebuild every check instead.

### How large the error is

Between two grid nodes $x_i < x_{i+1}$, a distance $h$ apart, each branch and each
compared value is read off its own straight chord. Where the check does not switch
inside the cell, every member's value is a smooth branch read off its chord, and its
error is second order in $h$. What happens where the check switches, at $x^*$, depends
on what the check compares:

- **Values.** A new match forms where each partner's household value beats their single
  value; a check may compare any declared reads. The interpolated values cross within a
  distance of order $h^2$ of where the true ones do, so the check switches there, and
  every member is priced on the side the check selects everywhere else. For a member who
  triggers the switch her value has only a kink at $x^*$, and her error stays second
  order. A member who is **overruled** — he would rather stay, she leaves — loses the
  gap $J$ between his two branches at $x^*$: inside that band of order $h^2$ his error
  can be as large as $J$, and outside it it is second order.
- **Policy support.** The continuing couple stays where its feasible set is not empty.
  Nodes know only whether they are supported, so the switch is placed halfway between a
  supported and an unsupported node. Between that midpoint and the true $x^*$, a stretch
  of at most $h/2$, the check takes the wrong branch and every member's error can be as
  large as the jump in their value; refining shrinks that stretch, not the error in it.

```text
 values decide: the switch is found again        support decides: the switch is put
 at the landing, from the two chords             halfway, wherever x* is in the cell

  ^                 accept (his)                  ^                 accept
  |         o-------*------------o                |         o--------------o
  |                 :                             |                 :   :
  |                 : J                           |                 : J : wrong side
  |                 :                             |                 :   :
  |  o--------------*-----o  reject               |  o--------------:---:--o  reject
  +--+--------------+-----+---> y                 +--+--------------+---+--+---> y
    x_i            x*   x_{i+1}                     x_i            x*  mid x_{i+1}
```

With $J$ the gap between a member's branches at $x^*$, the error at a landing $y$ is at
most

```{math}
\underbrace{\tfrac{h^2}{8}\max|B''|}_{\text{branch } B \text{ the check selects}}
\qquad \text{except} \qquad
\underbrace{|J|}_{\substack{|y - x^*| = O(h^2) \text{ (values decide)} \\ |y - x^*| \le h/2 \text{ (support decides)}}}.
```

Three measurements show the pattern. All compare against rebuilding every check at its
landing; "value after the check" is the surface the previous release stored, the value
each member holds after the check, interpolated across the switch.

**A two-member reference model** (pure Python; nine consumption shares; she leaves where
her single value beats the household, he always prefers to stay). The largest error of a
read at the last check, over landings in the cell holding the switch but at least $h/4$
from it:

| node spacing $h$ | value after the check | branches, decided at the landing |
| ---------------: | --------------------: | -------------------------------: |
|             4.75 |                  1.37 |                            0.069 |
|             2.38 |                  0.84 |                            0.015 |
|             1.19 |                  1.64 |                           0.0036 |
|             0.59 |                  1.25 |                          0.00084 |
|             0.30 |                  0.90 |                          0.00026 |

His jump is 2.7. Read after the check, the error near the switch does not shrink; read
as branches, it falls by about four for each halving of $h$. With support deciding the
same model's check instead, neither read converges near the switch (both stay between
0.36 and 2.0, for a jump of 1.9), as the bound above says.

**A pylcm witness** (`tests/solution/test_participation_pricing_depth.py`; the same
structure: she leaves rich households at the last check, he is overruled, and every
period-0 node lands at the middle of a cell there). The largest error of the couple's
period-0 values, both members, all nodes:

| node spacing $h$ | value after the check | branches, decided at the landing |
| ---------------: | --------------------: | -------------------------------: |
|                4 |                  2.48 |                            0.048 |
|                2 |                  2.46 |                            0.017 |
|                1 |                  2.45 |                           0.0050 |
|              0.5 |                  2.45 |                           0.0014 |

**The household example** (four periods, three consumption shares, 64-bit), the couple's
period-0 values for both members, whose values range from about 1.9 to 7.9:

| wealth nodes | node spacing | mean error | max error |
| -----------: | -----------: | ---------: | --------: |
|            4 |         12.7 |      0.102 |     0.284 |
|            6 |          7.6 |      0.116 |     0.241 |
|           10 |          4.2 |      0.075 |     0.693 |
|           16 |          2.5 |      0.034 |     0.548 |

The only check this model reads from a surface is the continuing couple's, and support
decides it, so storing branches changes only the nodes whose landings fall in a cell
where support switches: no node at 4 wealth nodes, two at 6 and 16, three at 10 (where
the second-largest error falls from 0.56 to 0.21). At 10 and 16 nodes the largest error
is not among them; it is the same whether the surface stores branches or the value after
the check. It comes from a decision taken on the surfaces' small smooth error near a
threshold — the household's own choice, or the exact check at the next landing — which
then flips and moves a member's value by a jump. Such a flip is an error in the stretch
of wealth around a threshold, not at a fixed node: it moves as the grid is refined, and
the mean, which falls to 0.034 at 16 nodes, is the better summary of how far the
solution is from rebuilding every check.

This is the same kind of error a `GridSearch` model already makes wherever it
interpolates a value function: a value function is a maximum over discrete choices, with
a kink wherever the best choice switches, and any choice made on interpolated values can
flip near a tie.

### When to care and what to do

No verdict is interpolated, but the values a check compares price the checks after it,
and so do earlier choices such as saving ahead of a later check. Both carry the
surfaces' error. It matters most when

- a participation threshold lies in a region of the state space that drives your
  results, and
- the check there is on policy support, or a choice is close to a tie there, so a
  member's value can jump within the cells around it.

To check and reduce it:

- **Compare two resolutions.** Solve with a finer grid on the target's continuous state.
  Differences concentrated at the few nodes around a threshold come from this
  approximation.
- **Put nodes where participation switches.** An `IrregSpacedGrid(points=...)` or a
  `PiecewiseLinSpacedGrid` with a breakpoint near the threshold shrinks the cells around
  it without paying for a finer grid everywhere (see [Grids](grids.md)). A denser grid
  there narrows the stretch of wealth where a check on support can land on the wrong
  side.
- **Compare values rather than support where the economics allow it.** A check on
  declared value reads is located to second order between nodes; a check on support only
  to half a cell.

## Simulating: every row carries its own role

Stakeholder identity is per subject, not per call. A row's role picks its route on a
value-dependent transition; the branch it takes then sets the role it carries onward.

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
