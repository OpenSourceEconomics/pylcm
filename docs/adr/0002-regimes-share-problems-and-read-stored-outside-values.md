# 2. Regimes share problems and read stored outside values

Status: accepted on 2026-11-17, not implemented. The engine step comes first, the
language step second.

## Context

A regime with an exit compares what each member gets by staying with what the member
gets by leaving. The engine obtains the value of leaving by rebuilding the destination's
action problem at the projected state, inside the leaving regime's own program. Three
things follow.

- The reads nest. The destination keeps its own exits, whose destinations are rebuilt in
  turn, and a repeated regime in that chain is refused as a cycle.
- Nothing is shared. Regimes with the same states, actions, functions and laws are
  solved separately, and a regime that can leave for another one solves that regime's
  problem again in its own program.
- Models declare regimes nobody is ever in. A collective model with marriage and divorce
  needs a single's regime without the marriage exit, only so that a couple's exit has an
  acyclic regime to read.

## Decision

A period has two layers.

- A **problem** is an action problem: states, actions, constraints, functions and laws
  of motion. It reads next period only, never the current one.
- A **regime** is what a subject is in. It is a problem together with the rule for
  leaving it in the current period. Initial conditions, edges and results name regimes.
  Only something a subject can enter a period in is a regime; the outcome of a move is
  the problem the subject then acts in.

A regime's value comes from a masked reduction over its problem's action values, one
value per member.

- The mask keeps an action when each member prefers it to that member's **outside
  value**, as the regime's rule defines "prefers" and "each".
- An outside value is the value of the destination's problem at the member's projected
  state, given the current shocks the decision observes, and without any exit of the
  destination. A subject moves at most once per period, so the recursion ends after one
  step.
- A move into a problem on another state space, such as a single who marries, evaluates
  the destination's action values at the mover's own cells in one step, applies the
  destination regime's rule for staying, and then the mover's rule for leaving. Exits
  therefore keep naming regimes as their targets.
- The rule owns who decides, where each member lands, and payments made on the move. A
  Pareto weight that renegotiation adjusts stays a grid state of the problem.
- A state that only a rule reads belongs to the problem, which ignores it in its
  functions.

Outside values are stored, not rebuilt.

- A problem that serves as an outside value publishes its values with one axis per
  current shock that some rule observes. The other shocks are integrated out in the
  problem. These arrays live for the solve of their period; simulation evaluates the
  outside problem at each subject's realized state and shocks.
- A read that lands on a grid point and a shock node takes the stored value. Any other
  read evaluates the outside problem at that exact point. Interpolating a stored value
  in a comparison between staying and leaving needs a declaration by the user.
- Two regimes share a shock axis when they refer to the same shared shock declaration.
- A problem that only serves as an outside value may use any solver. If it cannot be
  evaluated at a point, a read that may miss the grid is refused when the model is built
  unless interpolation is declared. A problem whose action values a regime reduces uses
  grid search.

The model language gains one noun and one field: `Problem`, and `Regime(problem=...)`. A
regime that declares states, actions and functions itself owns a problem no other regime
shares, so existing models keep their meaning. `Exit`, `ExitTarget`, the exit rules,
`ParticipationCheck` and `ValueDependentConstraint` stay. A participation read
distinguishes a problem's value from a regime's value; the kind that keeps a
destination's exits without moving goes away.

### Order of work

1. The engine reads stored outside values under the unchanged model language. The
   current engine is the reference: the same model must give the same values to a few
   units in the last place, on aligned and on unaligned reads.
1. `Problem` enters the model language, and models drop the regimes they declared only
   to be read.

### Constraints that keep multi-device solves open

Solving a collective model with exits across several devices is a later goal. Until
then:

- Every new array is a planned value with a layout, sharded states first, and moves
  between devices through the existing value-transfer classification.
- An outside read is device-local wherever a next-period read of the same array would
  be, and a replica otherwise. Nothing is gathered on the host.
- Reducing a regime inside its problem's program is an optimization the planner may
  decline.
- A test combines an exit with a discrete sharded state from the first step on.

## Consequences

- An outside value two moves away is a problem's value. A destination's own exits no
  longer constrain it. The two agree whenever the destination's only exit needs consent,
  which the current engine also drops at that depth.
- Cycles among same-period reads cannot be declared, and no regime exists only to break
  one.
- A problem solved with an endogenous-grid method can be the outside value of a
  grid-search problem.
- An exit whose destination has a device-local sharded state becomes solvable, since the
  destination's problem is read and not rebuilt.
- The kernel that evaluates one problem's action values at another regime's cells stays.
  A decision taken after observing the destination's shocks needs those values at each
  of the mover's cells and shock nodes, and no stored array provides them at a tolerable
  size.
- A model can make every outside read aligned by putting the states its projections
  produce on the grid.

### Expected effect, measured on one model

The numbers come from a profile of the Eckstein-Keane-Lifshitz replication taken in
November 2026 and bound what this decision can deliver there.

- Deciding on divorce takes 19 to 30 percent of the married regimes' kernel time, and
  those regimes take about 94 percent of the total. Replacing the rebuilt outside
  problem by a read removes at most that share: at most about 1.4 times faster.
- Stored values do not shrink. That model already avoids storing the regimes it declares
  only to be read.
- The single's outside-value arrays with their shock axes take about 112 MB and 6 MB per
  period in the largest block, next to a couple grid of about 131 million cells.
- The effect on compile time has not been measured.

## Alternatives considered

- **Mask the maximized value by state.** A regime would take its problem's value where
  every member agrees and leave elsewhere. A couple would then either reach its
  unconstrained optimum or separate, and never choose the best action both accept.
- **Store the destination's action values for moves across state spaces.** For a
  decision that observes the destination's shocks this takes the destination's grid
  times every observed shock node, terabytes in the model above.
- **Decide before the destination's shocks are observed.** The decision could then read
  a stored, integrated array. This changes the date of the decision. A model can still
  choose it by declaring that the rule observes nothing.
- **Interpolate reads that miss the grid by default.** A discrete decision would rest on
  interpolated values near kinks without the user having asked for it.
- **Expand `Problem` into the regimes models declare by hand, leaving the engine as it
  is.** This gives the smaller model at no risk to results, but keeps the nested reads,
  the repeated solves and the cycle refusals.
- **Name the occupied layer anew and keep "regime" for the action problem.** Initial
  conditions, edges and results would change their meaning in every existing model.
- **Replace `ParticipationCheck` and `ValueDependentConstraint` by exits.** They decide
  at the landing in the next period, accept any predicate, and attach a fallback to the
  incoming edge. Exits do none of these.
- **Allow several moves per period.** The general object is a graph of stages within a
  period. No model at hand needs more than one move, and a graph of depth one keeps
  every outside value a plain problem value.
