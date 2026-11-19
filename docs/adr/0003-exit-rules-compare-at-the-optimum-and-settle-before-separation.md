# 3. Exit rules may compare at the optimum and settle before separation

Status: accepted on 2026-11-19. Implemented for two-member regimes with grid search.

## Context

The built-in dissolution rules `Unilateral` and `MutualConsent` compare each member's
value at every stay action with that member's outside value, and the household keeps its
best action that passes. ADR 2 rejected masking the maximized value instead, because a
couple would then never choose the best action both accept.

Many models of collective households state their dissolution rules differently.

- The test is made at the household's unconstrained optimum: a member compares the value
  it gets from what the household would do anyway with what it gets on leaving.
- Under mutual consent the members can shift the division on separation before they
  decide. When one member wants to leave and the other does not, the division moves in
  favour of the member who wants to stay, by the least amount that makes that member
  accept separation; the pair separates there if the other still gains, and stays
  otherwise.
- Under unilateral dissolution with commitment, the match ends whenever either member
  refuses at the optimum.

`Unilateral(renegotiate=...)` already decides at the optimum, but only together with a
walk over a state of the source.

## Decision

- `Unilateral` and `MutualConsent` gain `compares: "each_action" | "optimum" | None`.
  `None` means the optimum when the rule declares an adjustment and every stay action
  otherwise, so existing declarations keep their meaning and their fingerprint. An
  adjustment together with `"each_action"` is refused when the rule is constructed.
- At the optimum a member is willing when its value at the unconstrained optimum is
  above its outside value, or equal to it under `tie="stay"`. `Unilateral` stays when
  every member is willing, `MutualConsent` when at least one is. A pair that stays acts
  at the optimum. With no feasible action nobody is willing. A separation needs a
  feasible action in every destination where the exit puts its member; without one the
  cell has no value, as under the per-action rule.
- `MutualConsent(settle=Settlement(name=..., candidates=..., default=...))` adds a
  settlement read by exit-target projections and utility adjustments under `name`.
  1. Each member compares its stay value with its outside value at the default.
  1. Nobody refuses: stay. Both refuse: separate at the default.
  1. Exactly one refuses: among the candidates strictly on the other member's side of
     the default, take the one nearest the default at which that member weakly accepts
     separation. Separate there if the member who wants to leave still gains under the
     rule's `tie`; otherwise, or without such a candidate, stay.
- Two members only. `settle` exists only on `MutualConsent` and `renegotiate` only on
  `Unilateral`.
- The default reads what `available` reads and is clipped to the candidates' range.
- A separating household lands where the projections evaluated at the selected
  settlement put it and receives the adjustments evaluated there. The simulated output
  shows the settlement through those landing states and values; it has no column of its
  own.

## Consequences

- A rule at the optimum publishes the stay values of the unconstrained optimum or the
  outside values; it can leave a member below its outside value only under
  `MutualConsent`, as the per-action rule can.
- Each candidate whose settlement changes a member's projection is one more outside read
  of that member's destination. A member whose projection does not read the settlement
  is read once, and only its adjustment varies. A candidate read whose projections land
  on the destination's grid reads the stored table; the default, a function of the cell,
  is evaluated at its exact point unless its projections also provably land on the grid.
- The engine does not check that the stayer's outside value rises in the stayer's
  direction. The rule is stated for the nearest accepting candidate, which is the least
  shift whenever it does.
- A default that is random and not observed at the decision cannot be declared.

## Alternatives considered

- **Keep per-action masking as the only rule.** It remains the default. Comparing at the
  optimum is a different economic assumption, not an approximation of it, and models
  that state it cannot be written with a mask.
- **Renegotiate a state of the source instead of settling on the exit side.** A
  settlement that changes only what members get on separation would become a grid state
  of the couple, with a fixed transition and a walk, although nothing in the couple's
  problem reads it, and every outside value would be evaluated at every node of that
  state. Settling on the exit side reads the outside values at the candidates only, and
  leaves the couple's state space unchanged.
- **Find the settlement by continuous root finding.** The least settlement would make
  the stayer exactly indifferent. A root finder needs the outside values as a
  differentiable function of the settlement and a bracketing rule at every cell, and
  fails where the destination's value has kinks. A candidate grid follows the convention
  of `Renegotiate`, gives the stayer a non-negative surplus that a finer grid shrinks,
  and keeps every decision a comparison of computed values.
