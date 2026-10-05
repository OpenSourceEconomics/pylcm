# PyLCM

Specification, solution and simulation of finite-horizon discrete-continuous dynamic
choice models. This glossary fixes the vocabulary shared by the public API, the
documentation and the engine; it is not a specification.

## Language

### Population

**Subject**:
One unit of the population handed to a simulation or a feasibility check; a
subject starts in exactly one regime at one age.
_Avoid_: individual, agent, row, person

**Initial conditions**:
The subject-indexed collection of starting regime and starting states, supplied
either as a mapping of arrays or as a DataFrame with one row per subject.
_Avoid_: initial observations, initial population, starting sample

**Initial states**:
The state part of the initial conditions, i.e. everything except the regime and the
stakeholder role.
_Avoid_: initial values, start states

### Model structure

**Regime**:
A self-contained block of the model with its own states, actions, constraints and
functions; a subject is in exactly one regime per period.

**Action-free regime**:
A regime that declares no actions. Its constraints, if any, depend only on states
and parameters and still bind.
_Avoid_: absorbing regime (an absorbing regime may have actions), terminal regime

**Edge**:
A transition from a source regime to a target regime. The source's state laws
toward that target are the edge's laws.
_Avoid_: arc, link, regime transition (that is the probability of taking an edge)

**Carried state**:
A state a regime keeps in its state space, and so in its value function. A
regime carries a state it reads; reading the state's next-period draw counts.
_Avoid_: retained variable, kept state (in user-facing text)

**Transition-local draw**:
A next-period value of a random state (a process or a Markov state) that is drawn
on an edge and consumed by that edge's laws, but not kept by the target, because
the target does not carry the state. It is drawn from the source's law at the
source's current value.
_Avoid_: dropped shock, ephemeral state

**Age**:
The user-facing time index of a subject, on the model's age grid.

**Period**:
The engine's zero-based index into the age grid. Users supply ages; the engine
converts.
_Avoid_: t, time step (in user-facing text)

### Parameters

**Fixed parameter**:
A parameter bound when the model is built and not supplied at call time.

**Runtime parameter**:
A parameter supplied with every `solve`, `simulate` or feasibility call; when a name
is both fixed and runtime, the runtime value binds.
_Avoid_: free parameter, estimated parameter

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
