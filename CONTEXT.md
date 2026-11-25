# PyLCM

Specification, solution and simulation of finite-horizon discrete-continuous dynamic
choice models. This glossary fixes the vocabulary shared by the public API, the
documentation and the engine; it is not a specification.

## Language

### Population

**Subject**:
One unit of the population handed to a simulation or a feasibility check; a
subject starts in exactly one regime at one model time coordinate.
_Avoid_: individual, agent, row, person

**Initial conditions**:
The subject-indexed collection of starting regime and starting states, supplied
either as a mapping of arrays or as a DataFrame with one row per subject.
_Avoid_: initial observations, initial population, starting sample

**Initial states**:
The state part of the initial conditions, i.e. everything except the regime and the
member role.
_Avoid_: initial values, start states

### Model structure

**Regime**:
What a subject is in at a step: an action problem (states, actions, constraints,
functions and laws of motion) together with the rule for leaving it. It is the model's
only noun for this; the action problem is part of the regime, not an object of its own.
_Avoid_: problem (as a separate object), visible regime, status, view

**Member**:
A party of a regime whose value is tracked. Every regime has at least one member: a
regime declared with a plain utility function has the single member "self", and a
household names its members in its collective utility.
_Avoid_: stakeholder

**Household**:
A regime that names its members in a collective utility and takes one action for all of
them. A household may have one member.
_Avoid_: couple (unless it has two members)

**Decider**:
The one member of a household with a positive Pareto weight when the household has taste
shocks; its choice is the household's.
_Avoid_: mover, leader

**Rule**:
How a regime turns its members' action values and their outside values into its value at
a step: which action is taken, and whether the subject stays or moves. It reads the
regime's own action values and the outside values, and nothing else.
_Avoid_: participation check, constraint on values

**Problem value**:
The value of a regime's action problem at a step, without its rule for leaving; it reads
the next step's values and nothing of the current step.
_Avoid_: hidden value

**Value**:
What a member of a regime gets at a step after the regime's rule, including any move.

**Staying value**:
A regime's value where its rule keeps the subject in it; it is not defined where the rule
moves the subject out. Under renegotiation it is the problem value at the moved
bargaining state.
_Avoid_: participation value

**Outside value**:
What a member of a regime gets by leaving it at the current step: the destination's
problem value at the member's projected state under the destination grid's `off_grid`
rule, given the current shocks the decision observes, and with no further move in that
period. A move that needs the consent of the destination's members reads the
destination's staying value instead.
_Avoid_: benchmark, threat point

**Exit**:
A regime's move at the current step, before the action, decided by its rule from
compared values; it names the destination regime of each member.
_Avoid_: participation check, current-period transition

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
A regime some subject can be in at a given period, given the regimes the subjects
start in; the forward closure of period targets from the starting regimes.
_Avoid_: period target, active regime (an active regime need not be reachable)

**Regime-transition law**:
The law that decides which of a source regime's outgoing edges a subject takes,
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
The subject's biological age, when the model uses an age grid as its time coordinate.

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
into the regime lands.

**Pass-through**:
The generated regime a subject is in at a step of a stage its regime sits out on the way
to its next occurrence: the regime's states and members, one action, no shocks, zero
utility and the identity transition, so its value is the next step's passed through
unchanged. The user never declares one, and the simulated frame has no rows for it.
_Avoid_: waiting regime, dummy regime

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

**Feasible (subject)**:
A subject for which at least one combination of the regime's declared action-grid
points satisfies every constraint jointly; in an action-free regime, a subject whose
state-only constraints all hold.
_Avoid_: valid, admissible (reserved for individual actions), allowed

**Infeasible (subject)**:
A subject that is not feasible. Infeasibility is a property of the supplied initial
conditions, never of the model.

**Feasibility mask**:
The one-dimensional boolean array, in subject order, that is `True` exactly for the
feasible subjects.
_Avoid_: acceptance mask, validity flags

**Invalid initial conditions**:
Initial conditions that are structurally malformed (unknown regime, missing or
extra state, wrong length, off-grid age, inactive regime at that age, unknown
categorical label) or that contain an infeasible subject.

**Unsupported check**:
A feasibility question the engine cannot answer for the given model shape, e.g. a
constraint that depends on an age-specialized function while subjects start at
different ages. An unsupported check is a property of the model, is reported by
raising, and never counts as feasible.
_Avoid_: skipped check, unavailable check
