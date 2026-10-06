# 1. Laws may read draws the target does not carry

## Context

A state law on an edge may read the next-period value `next_<state>` of a random state —
a process or a Markov state — that the source carries. Costs realized after the period's
choices and paid out of next period's wealth are the common case. When the target also
carries the state, the draw becomes the target's state. When it does not, for instance a
terminal regime that values only wealth, the value has nowhere to be stored.

## Decision

Such a draw is a transition-local draw: it is taken from the source's law at the
source's current value, read by the edge's laws, and discarded. Processes and Markov
states behave alike. A Markov state's draw toward a target that does not carry it needs
a law declared once for every target, since a per-target law names no law for that edge.

Reading a random state's draw is a read of the state, so the reading regime carries it,
together with every state the draw's law reads. A law that reads `next_<state>` of any
other state the target does not carry is refused when the model is built.

## Consequences

- A target's state space holds only the states it reads; no axis is added to carry a
  value nothing in the target uses.
- Solve and simulate agree: both enumerate or sample the draw on the edge from the same
  source law.
- A per-target Markov law cannot feed a target it does not name.

## Alternatives considered

- **Force the target to carry the state.** Adding the state to the target with no role
  in its value gives the same continuation values, but multiplies the target's state
  space by the state's support size and makes users declare a state they do not model.
