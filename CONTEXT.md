# PyLCM

Specification, solution and simulation of finite-horizon discrete-continuous dynamic
choice models. This glossary fixes the vocabulary shared by the public API, the
documentation and the engine; it is not a specification.

## Language

### Population

**Agent**:
One person, followed through life. The population handed to a simulation or a
feasibility check is a set of agents. An agent starts in exactly one regime at one model
time coordinate.
_Avoid_: subject, decision maker, individual, row

**Composition**:
Who is in the unit: single, couple, parent with child. It is a state of the agent. It
changes by a probability on an edge or by the decision on the composition.
_Avoid_: household (for the unit), status

**Initial conditions**:
The agent-indexed collection of starting regime and starting states, supplied
either as a mapping of arrays or as a DataFrame with one row per agent.
_Avoid_: initial observations, initial population, starting sample

**Initial states**:
The state part of the initial conditions, i.e. everything except the regime and the
member role.
_Avoid_: initial values, start states

### Model structure

**Regime**:
What an agent is in at a step: an action problem (states, actions, constraints,
functions and laws of motion) together with the rule for leaving it. It is the model's
only noun for this; the action problem is part of the regime, not an object of its own.
_Avoid_: problem (as a separate object), visible regime, status, view

**Member**:
An agent's position in a regime. Every regime has at least one member, and its value is
tracked per member. A regime declared with a plain utility function has the single
member "self". A regime with several members names them in its collective utility.
_Avoid_: stakeholder, subject

**Collective regime**:
A regime that names its members in a collective utility and takes one action for all of
them. A collective regime may have one member.
_Avoid_: household, couple (unless it has two members)

**Counterparty**:
The other agent at a formation: its regime, its member and its landing states.

**Pareto weight**:
The weight on one member's action value when a collective regime takes one action for
all members. Under `PCOnWeight` with a `state`, a discrete state holds the Pareto weight,
and the rule can move it within the period.
_Avoid_: bargaining state, weight (bare)

**Deciding member**:
The one member of a collective regime with a positive Pareto weight when the regime has
taste shocks. Its choice is the regime's choice.
_Avoid_: mover, leader, decision maker

**Composition action**:
The object a regime declares for the decision on the composition (`CompositionAction`).
It names one destination per member and the rule that decides. The decision happens at
the current step, before the action.
_Avoid_: exit, participation check, current-period transition

**Decision on the composition**:
What the members decide through the composition action: the decision to stay or to
separate, or the decision to form a couple. A member who separates leaves the regime. An
agent decides on its composition at most once per period.
_Avoid_: move (as a noun), exit, settlement

**Rule**:
How a regime turns its members' action values and the values of their outside options
into its value at a step: which action is taken, and whether the composition changes. It reads
the regime's own action values and the values of the outside options, and nothing else.
The rules carry the literature's names:

- unilateral divorce with participation constraints on actions (`PCOnActions`): each
  member's value of staying is compared with the value of its outside option at every
  stay action, and the members keep the best action that all of them accept;
- unilateral divorce with participation constraints on the Pareto weight (`PCOnWeight`):
  the comparison is at the unconstrained optimum; with a `state`, the Pareto weight
  first moves toward the refusing member to the first node where both members accept;
- mutual consent divorce (`MutualConsent`): the comparison is at the unconstrained
  optimum, and the members can shift the transfer;
- formation by consent (`Consent`): both agents must prefer the formed couple to staying
  apart.

Write "separation" instead of "divorce" where the unit is not a marriage.
_Avoid_: participation check, constraint on values

**Transfer**:
The divorce transfer under mutual consent divorce. Its default is the division by law,
and the members can shift it to one of its candidates before they separate.
_Avoid_: settlement

**Value without the composition action**:
The value of a regime at a step when it does not decide on its composition. It reads the
next step's values and nothing of the current step.
_Avoid_: problem value, hidden value

**Value**:
What a member of a regime gets at a step after the regime's rule, including the decision
on the composition.

**Value of staying**:
A regime's value where its rule keeps the members in it; it is not defined where the
members separate. Under `PCOnWeight` with a `state`, it is the value without the
composition action at the Pareto weight that the rule selects.
_Avoid_: staying value, participation value

**Landing state**:
The state at which a member arrives in its destination. The destination's projection
gives it from the current states.
_Avoid_: projected state

**Value of the outside option**:
What a member of a regime gets by leaving it at the current step: the destination's
value without the composition action at the member's landing state under the
destination grid's `off_grid`
rule, given the current shocks the decision observes, and with no further decision on
the composition in that period. Under formation by consent, the destination is the
formed couple, and the value is the couple's value of staying.
_Avoid_: outside value (in prose; code identifiers keep it), benchmark, threat point

**Gate**:
A branch on the transition into one target regime that keeps the target or takes a
fallback, decided from the landing state alone; it never reads a value.
_Avoid_: value-dependent transition

**Action-free regime**:
A regime that declares no actions. Its constraints, if any, depend only on states
and parameters and still bind.
_Avoid_: absorbing regime (an absorbing regime may have actions), terminal regime

**Edge**:
A transition from a source regime to a target regime, declared with the model's
edges.
_Avoid_: arc, link, regime transition (that is the probability of taking an edge)

**Period target**:
A regime that a source regime may enter at the next period along an edge that
period's graph retains. It is a one-step property of a source and a period.
_Avoid_: reachable regime, reachable target

**Reachable regime**:
A regime some agent can be in at a given period, given the regimes the agents
start in; the forward closure of period targets from the starting regimes.
_Avoid_: period target, active regime (an active regime need not be reachable)

**Regime-transition law**:
The law that decides which of a source regime's outgoing edges an agent takes,
together with any gates and route fallbacks declared with it. It
belongs to the edges, and so do its parameters; the source regime owns neither.
_Avoid_: next-regime function, edge law

**Per-target state law**:
A state law of a source regime that differs by target regime. It belongs to the
source regime's problem, not to the edge, even though it is evaluated only toward its
target.
_Avoid_: edge law, handoff law

**Carried state**:
A state a regime keeps in its state space, and so in its value function. A
regime carries a state it reads; reading the state's next-period draw counts.
It also carries a state it hands, with any law of motion including
`fixed_transition`, to a reachable target that carries it.
_Avoid_: retained variable, kept state (in user-facing text)

**Transition-local draw**:
A next-period value of a random state (a process or a Markov state) that is drawn
on an edge and consumed by the source's state laws toward that target, but not kept
by the target, because the target does not carry the state. It is drawn from the
source's law at the source's current value.
_Avoid_: dropped shock, ephemeral state

**Age**:
The agent's biological age, when the model uses an age grid as its time coordinate.

**Period**:
An interval of economic time, such as a year or a quarter; in an age model, one to one
with age.
_Avoid_: date, t (in user-facing text)

**Stage**:
A decision inside a period, declared with `Model(stages=...)`; every period has the
same stages, in order, and every regime belongs to one of them. A period has one stage
unless the model declares several. Stages of one period are linked without discounting.
_Avoid_: date, sub-period

**Step**:
A pair of a period and a stage: what the recursion and the solver run over.
`model.n_periods`, the solution and the raw simulation results count steps; `period`
in user functions and in the simulated frame is the period.
_Avoid_: date, time step

**Next occurrence**:
The first step after the current one that belongs to a regime's stage; where an edge
into the regime lands and whose value it reads. An agent has no row at a step its
regime sits out.
_Avoid_: pass-through, waiting regime

### Phases

A `Phased` declaration has three parts.

**Perceived model**:
The `solve` variant: the laws the agent believes. They price the continuation, and the
value function is solved under them.

**Current self**:
The agent as it chooses now. It uses the `simulate` variant of the current-period
primitives: period utility, feasibility and the Koopmans aggregator.

**Actual laws**:
The `simulate` variant of the laws of motion. They govern what happens as the simulation
goes forward.
_Avoid_: realized laws, truth

### Parameters

**Fixed parameter**:
A parameter bound when the model is built and not supplied at call time.

**Runtime parameter**:
A parameter supplied with every `solve`, `simulate` or feasibility call, in a slot
that has not been fixed at model construction.
_Avoid_: free parameter, estimated parameter

**Temporal parameter**:
A parameter whose value is labelled by model time and selected at the period of the
function that consumes it.

### Feasibility

**Feasible (agent)**:
An agent for which at least one combination of the regime's declared action-grid
points satisfies every constraint jointly; in an action-free regime, an agent whose
state-only constraints all hold.
_Avoid_: valid, admissible (reserved for individual actions), allowed

**Infeasible (agent)**:
An agent that is not feasible. Infeasibility is a property of the supplied initial
conditions, never of the model.

**Feasibility mask**:
The one-dimensional boolean array, in agent order, that is `True` exactly for the
feasible agents.
_Avoid_: acceptance mask, validity flags

**Invalid initial conditions**:
Initial conditions that are structurally malformed (unknown regime, missing or
extra state, wrong length, off-grid age, inactive regime at that age, unknown
categorical label) or that contain an infeasible agent.

**Unsupported check**:
A feasibility question the engine cannot answer for the given model shape, e.g. a
constraint that depends on an age-specialized function while agents start at
different ages. An unsupported check is a property of the model, is reported by
raising, and never counts as feasible.
_Avoid_: skipped check, unavailable check
