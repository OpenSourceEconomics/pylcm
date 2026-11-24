---
title: Layout search
---

# Layout search

The regime analysis searches one family of layouts: the family of the paper's Supplement
(lcm-paper, `paper/sections/supplement-choice.tex`). Line numbers below refer to that
file at lcm-paper `d28955bd`. The search lives in
`src/_lcm_regime_finder/layout_search.py`, and an independent checker in
`layout_check.py`. The public interface is documented in
[Check a regime graph](../user_guide/regime_checker.md). The per-fibre facts are
described in [Dependency facts](regime_facts.md).

## The family

- **Dense regimes** (:126-146). Nonterminal regimes of one kind pool according to their
  taste shocks. A regime's kind is its stage, solver, aggregator, certainty equivalent,
  Pareto objective, members and exit (rule, tie, comparison, renegotiation, settlement,
  targets with their roles, observed shocks and availability), so a collective regime
  never pools with a single-agent one, and regimes with different exits never pool:

  - Regimes without taste shocks pool together.
  - A regime with taste shocks stays alone.
  - The regimes without taste shocks join a single shocked regime only when none of them
    has a discrete choice.

  Every terminal regime is a dense regime of its own.

- **Statuses, fibres and lanes** (:181-229).

  - A status is a categorical state. The state an exit's `Renegotiate` moves is not a
    status of that regime: the value at one of its codes reads the stay values at the
    others within the period, so it is an ordinary state there, live in every period,
    never split into lanes and never carried as a status.
  - A fibre is a regime and a period, with one value for each of the regime's statuses.
  - A lane is a regime with one value for each status, across periods.

  The kept fibres are the fibres reachable from the initial fibres along links. A link
  fixes only the statuses it carries unchanged; every other status of the target takes
  all its values. A relation whose target is its own cell is not a link.

- **Blocks and prices** (:516-566, :843-859). A block keeps the union of its fibres'
  live coordinates. It charges each coordinate as follows:

  - a status by the values it takes on the block's fibres;
  - an ordinary state by its largest grid;
  - an action by its largest grid.

  A block pays `|T_B| · |X_B| · |S_B| · |A_B|` evaluations and `|T_B| · |X_B| · |S_B|`
  stored values, where `T_B` is the set of the block's own periods and `S_B` the members
  of its regimes (one for a single-agent regime): a collective regime stores one value
  and computes one action value per member.

- **Layouts** (:594-617). The lanes of a dense regime are partitioned into groups, which
  stay fixed for all periods. Each group's active periods are cut into intervals. Each
  interval is one block.

Two rules settle cases that the Supplement leaves to the implementation:

- The regime label is live at every fibre of a pooled dense regime. A block therefore
  pays for the number of regimes it holds.
- A member without a status holds it at the first label of its grid. If the members of a
  dense regime start that grid with different labels, a note records it. If no member of
  a block has an action, every value of that action except the first is excluded, and
  the block records these values as `excluded_actions`. This does not change prices,
  because a block's actions come from its members' own cells.

`search_layouts` lists both rules under `deviations`.

## The search

For each dense regime, two recurrences give the least number of evaluations with exactly
`n` blocks (:923-959):

- The interval recurrence `f` cuts the active periods of a group.
- The subset recurrence `h` enumerates groups of lanes, anchored on the lowest lane. Its
  cost grows quickly with the number of lanes.

A convolution over the dense regimes gives the exact table `w(n)`. Its at-most envelope
gives the curve `ev*(k)`. Ties are broken by evaluations, then by the number of blocks,
then by storage. Every point carries its witness layout.

| `limits` key        | Default   | Meaning                                                                      |
| ------------------- | --------- | ---------------------------------------------------------------------------- |
| `max_blocks`        | `n0 + 12` | Largest block budget `k`. `n0` is the number of dense regimes.               |
| `lanes_per_regime`  | 12        | Above this, statuses are contracted until every dense regime fits.           |
| `contract_unproved` | `True`    | Contract statuses that pass (I1) only over each member's own grid.           |
| `seconds`           | none      | Time limit for facts and search; on expiry the search raises `TimeoutError`. |

## Contraction and exactness

Lanes that differ only in an inert status can be merged (lem:inert, :1737-1889). A
status is certified inert when two conditions hold:

- (I1): every part of each kept contracted fibre is kept, where the parts range over the
  dense regime's grid of the status.
- (I2): every part has the contracted fibre's live set.

A status fails (I1) if its grid differs across the members of a dense regime, or if some
member lacks it.

By default, such a status is still contracted, but as a restriction. Every point is then
exact for the contracted family and an upper bound for the full family. The reason: a
contracted lane is a union of full lanes, and a block's price depends only on its
fibres, so a layout of contracted lanes is a layout of the full family at the same
price. The same holds when a status is contracted because a dense regime had more than
`lanes_per_regime` lanes (rem:truncated, :2005-2021).

In either case the result reads `RESTRICTED`, and `restriction_reasons` gives the reason
for each restricted status. A `RESTRICTED` result differs from a complete one in two
ways:

- The first point, with one block per dense regime, is marked `exact` (lem:first,
  :1482-1503). So is any point whose evaluations equal `lower_bound`, with
  `exact_reason` saying so.
- `floor` is `lower_bound` when a point attains it, and null otherwise: the floor of
  lem:lowest (iv) over contracted lanes is not a bound for the full family.

`lower_bound` is lem:lowest (vi) on the full kept fibres, whatever was contracted. In
each cell, two fibres join when they agree in every status live at either, and the bound
sums the largest one-fibre price of each component. No layout of the full family costs
less, so a found layout that attains it is the full family's lowest.

With `contract_unproved=False`, these statuses stay lanes instead. The result can then
be `COMPLETE`, at the price of a larger subset recurrence.

## The record

`search_layouts` returns a `supplement-layout-curve/v1` record with these fields:

- `status`: `COMPLETE` or `RESTRICTED`.
- `dense_regimes`, `inert_statuses`, `restricted_statuses`, `restriction_reasons` and
  `lanes`.
- `dense_count`: the count `ev^0` of the dense model (:851-859). Each dense regime
  charges its label, the union grid of each state and the largest grid of each action,
  in every active period.
- `original_count`: the count of the model as written.
- `exact`: the table `w(n)`. `curve`: the curve `ev*(k)`. Each point has its blocks,
  evaluations, storage and witness layout, and each curve point also has an `exact`
  flag.
- `floor` and `flat_from`.
- `lower_bound` (lem:lowest (vi)) and `finest_count`, the count of one block per kept
  fibre (lem:lowest (iv)).
- `deviations` and `notes`.
- `priced`: the prices of blocks requested through `price`.

A witness layout lists groups of lanes, and each group lists its cuts and its blocks.
Each block lists its periods, its states with their grids, its actions and excluded
actions, its evaluations and storage, and its fibres.

## The checker

`check_layouts(declarations, facts, result)` shares only the per-fibre facts with the
search. With its own code, it re-derives:

- the dense regimes and the kept fibres;
- (I1) and (I2) for each claimed inert status;
- the lanes;
- `dense_count`, `original_count`, `lower_bound` and `finest_count`.

It then prices every witness block literally and checks that:

- the groups partition the lanes;
- the cuts partition each group's active periods;
- the totals match;
- the curve is the at-most envelope of the exact table, no point lies below
  `lower_bound`, and each point's `exact` flag follows the rules above;
- the result's `lanes` equal the lanes it re-derived;
- each block's periods, states with their grids, actions, excluded actions and fibres
  equal those it derives from the block's kept fibres, in any order of the fibres.

The tests also compare the search with a literal enumeration of the family on small
models.
