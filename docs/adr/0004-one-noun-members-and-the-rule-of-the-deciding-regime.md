# 4. One noun, members, and the rule of the regime that decides

Status: accepted on 2026-11-23. Supersedes
[ADR 2](0002-regimes-share-problems-and-read-stored-outside-values.md) and
[ADR 3](0003-exit-rules-compare-at-the-optimum-and-settle-before-separation.md); the
last section says which parts of them remain in force.

## Context

ADR 2 added a second noun, `Problem`, which several regimes could hold. It kept two
mechanisms that compare values outside an exit: `ParticipationCheck`, which decides on
landing in the next period with any predicate and attaches a fallback to the incoming
edge, and `ValueDependentConstraint`, which restricts a household's actions by values of
the same period. A household was a regime with stakeholders; a regime declared with a
plain utility function had none.

The stated aims of the rewrite are fewer evaluations and an obvious mapping of model
code to model equations; a shorter language is not an aim of its own.

## Decision

### Words

- A **period** is economic time, one to one with age.
- A **stage** is a decision inside a period.
- A **step** is a pair (period, stage). The recursion and the solver run over steps.

The documentation uses these words; the model language has no stage object. Its `period`
counts steps, which are periods wherever each period has one stage.

### One noun: regime

A subject is in a regime. There is no `Problem` object, no `Regime(problem=...)` and no
`Model.problems`. Regimes that share their states, actions, functions and laws but
differ in their exit or edges are separate regimes, built from one definition with
ordinary Python, each with its own parameters.

### Two values and a flag per regime and step

- The **problem value** $S_r$: the value of the regime's problem (its states, actions,
  constraints, functions and laws of motion) without its rule for leaving.
- The **value** $V_r$: the value after the rule, with the move.
- A **stays** flag: where the rule keeps the subject in the regime.
- The **staying value** $P_r$ is derived: $P_r = V_r$ where the regime stays. Under
  renegotiation it is the problem value at the moved bargaining state. It is not defined
  where the regime would dissolve.

For member $m$ of regime $r$, with successor $r^+$ in the next step, outside target $q$,
projection $g_r^m$ and observed shocks $e_{\text{obs}}$:

```{math}
Q_r^m(x, a, e) = W_r\big(u_r^m(x, a, e),\ \mathrm{CE}_r[V_{r^+}^m(x') \mid x, a, e]\big),
\qquad
V_r(x) = \mathrm{E}_e\big[L_r[Q_r(x, \cdot, e);\ O_r(x, e_{\text{obs}})]\big],
```

```{math}
O_r^m = \mathrm{E}_{\text{hidden}}\big[S_q^m(g_r^m(x), e_{\text{obs}}, e_{\text{hid}})\big]
+ \text{adjustment}.
```

$W_r$ and $\mathrm{CE}_r$ are the regime's, the same for every member. A household uses
the linear expectation.

In the engine $S$, $V$ and $P$ are the `Raw`, `Full` and `Participating` policy queries.

### The rule

- $L_r$ is the regime's rule. The shipped rules are the maximum, the expected maximum
  under taste shocks, one member decides under taste shocks, and the exit rules:
  `Unilateral` and `MutualConsent` (at each stay action or at the household's optimum,
  `MutualConsent` with a settlement, `Unilateral` with renegotiation), and `Consent`.
- A rule reads the regime's own action values at the states it names (renegotiation
  reads the other nodes of the bargaining state) and the outside values, and nothing
  else.
- It reads another regime of the same step as that regime's problem value, or, under
  `Consent`, as its staying value; never as its value $V$.
- A subject moves at most once per period. An outside value is the destination's problem
  value at the member's projected state under the destination grid's `off_grid` rule,
  given the current shocks the decision observes, with hidden shocks integrated and the
  exit's utility adjustment added.

### One mechanism for value comparisons

A move decided by comparing values is the rule of the regime that decides, applied in
the step it decides in:

- formation is `Exit(rule=Consent(counterparty=...))` on the source, which compares the
  staying value of the formed household with each party's problem value;
- dissolution is the household's own `Exit`, with `Unilateral` or `MutualConsent`;
- a move that depends only on states, or on nothing, is a `Gate` on an ordinary
  transition; its routes' fallbacks read another regime's value in the landing period
  through `ProjectedRegimeValue`.

`ParticipationCheck`, `ParticipationRead`, `Transition(participation=...)` and
`ValueDependentConstraint` are removed, the last without a replacement. The model that
motivated it (each spouse's married value at least the divorced value, the weight moving
where this fails, divorce where no weight satisfies it) is renegotiation with a leave
branch, a rule of the household's exit. What only the removed constructs could express —
a predicate with slack or a band, a predicate on some members only, a comparison with a
constant — is used by no example model of pylcm.

### Two layers inside one step

A rule that is read through a staying value reads problem values only. A problem value
then depends on no other regime of the step, and a staying value on problem values only.
A longer chain is refused when the model's queries are ordered. No shipped model has
one.

### Members

- Every regime has members. `Regime.members` is a tuple: `("self",)` for a regime
  declared with a plain utility function, and the `utilities` keys of a
  `CollectiveUtility` for a household. A household may not name a member `"self"`.
- Public names say member: `MemberRoute`, `target_member`,
  `ProjectedRegimeValue.member`, `own_member`, `Model.member_names_to_ids` and the value
  axis `"member"`.
- Two engine paths remain, one for a regime declared with a plain utility function and
  one for a household; `Regime.member_axis` marks the household. A household declared
  with one named member stays on the household path, a ready object for the measurements
  that precede one engine path. Values, fingerprints and saved formats of models without
  a household are those of the engine before this decision.
- A household admits taste shocks when exactly one member has a positive Pareto weight.
  That member decides: its value is its own expected maximum; every other member's value
  is the mean of its own action values under the decider's choice probabilities, at the
  decider's best continuous action. With an exit, the decider's logit runs over the
  stays and the exit, and every other member gets its expected value under the decider's
  probabilities.

## Not shipped

None of these is implemented:

- **One engine path**, on which a regime with one member has a member axis of length
  one. It follows three measurements on a one-member model run on the household path:
  the bit differences against the single path, the working memory of the streamed solve,
  and the number of places where the member axis must be removed for saved results and
  cache keys to keep their shape.
- **Publish once and gather**: each regime publishes $S$ and $V$ once per step, with one
  axis per observed shock, and readers gather from it. It needs no change of the model
  language and is accepted on its own. Until then each reader tabulates, once per
  period, the destinations it reads at grid points, and evaluates every other read at
  its exact point.
- **$W$ and $\mathrm{CE}$ per member**, so that members of one household differ in
  patience or risk attitude. It comes with the one engine path; a household still
  refuses a nonlinear certainty equivalent.
- **The endogenous-grid method for households.** With one engine path its condition is
  that the regime has one member. The weaker condition "exactly one member decides" is
  not sufficient: the Euler inversion needs the decider's continuation to be the
  decider's own optimised value, and in a game with alternating moves the continuation
  is read at the other member's policy.
- **A user-written rule.** pylcm ships the rules listed above.
- **A public stage object**, so that `period` in model code means economic time and a
  model with stages no longer writes their mapping to ages and their discounting by
  hand. It comes with the one engine path.
- **Value comparisons no shipped rule expresses**: a value compared with a threshold, a
  parameter or a third regime's value; a comparison between members of one household; a
  read of next period's integrated value.

## Consequences

- `lcm_examples.collective_household` forms couples with `Consent` and dissolves them
  with the couple's `Unilateral` exit. Its values differ from those of the
  participation-check form. Three causes were tested on a rule-form copy at `2ac1c420a`:
  the meeting of period $t$ enters the single's value of $t$ rather than the
  continuation of $t - 1$; `Consent` compares at the single's own wealth, and a refused
  single keeps it; the participation-check form evaluates the next period's problem
  exactly at each landing, while the rule form interpolates that period's stored value
  between grid nodes.
- The `bench_collective_household` benchmarks measure the rule form; their history
  breaks there.
- A saved solution that holds participation surfaces does not load.
- Moves that read only states keep their gates and fallbacks; a fallback attached to an
  incoming edge exists only for such moves.

### Effect measured on one model

On `lcm_examples.collective_household` with 20 wealth and 20 consumption nodes, five
periods, on CPU, counting the action values the solve executes, without padded lanes:

- the participation-check form at `2ac1c420a`: 1,382,400;
- a rule-form copy of the example at the same commit: 38,400;
- the regime finder's count of the work, with the couple counted once: 9,600.

The shipped rule-form example at `34315168e` executes 40,320 action values including
padded lanes, as the rule-form copy did. The remaining factor of four against the
finder's count is the consent and outside-value reads, which publishing once and
gathering addresses.

## Alternatives considered

- **Formation as a rule of the landing regime.** The economics is a rule of the regime
  that decides, the single who forms a household, which reads the staying value of the
  landing regime.
- **Publish once and gather in the same change.** It is an engine change that needs no
  change of the model language, and its acceptance is separate.
- **A household with one named member on the single path.** Keeping it on the household
  path gives the measurements of the one engine path a ready test object.

## Parts of ADR 2 and ADR 3 that remain in force

- ADR 2, "Constraints that keep multi-device solves open": in force.
- ADR 2, "Consequences": in force, with "problem" read as a regime's problem rather than
  a `Problem` object, except the last sentence of the bullet on sharded states (a
  participation check), which no longer applies.
- ADR 2, "Expected effect, measured on one model": the record of that measurement.
- ADR 2, the rest: superseded by this record. Where ADR 2 says outside values are
  published once per problem, read that each reader tabulates its aligned reads once per
  period, and that a read the build cannot prove aligned is evaluated at its exact point
  without interpolation.
- ADR 3: in force in full. Its rules are instances of the exit rules above.
