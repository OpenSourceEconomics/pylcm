---
title: Dependency facts for regime analysis
---

# Dependency facts for regime analysis

The regime analysis asks one question of a model: in each regime-period (a *cell*),
which states and actions does each callback read? Everything else follows from the
answers. Liveness, constant states, forced actions, characteristic maps, status splits
and the checker's re-derivations all read them through one contract, `Facts`, in
`src/_lcm_regime_finder/facts.py`. These are internal interfaces. The public interface
is documented in [Check a regime graph](../user_guide/regime_checker.md).

## The contract

A `Facts` backend exposes the normalized `cells`, `callbacks`, `domains` and
`relations`, each cell's `constants` and `forced` actions, and three queries:

- `callback_reads(callback, cell, phase, *, stops, relation, statuses)` returns the
  coordinates a callback reads in that cell and phase. `stops` cuts characteristic
  providers. `relation` adds a transition's context. `statuses` fixes some status codes,
  and `None` means the whole cell.
- `identity_update(update, axis, target_axis)` tells whether an update carries a state
  unchanged.
- `characteristic_image(spec)` returns the values a characteristic map takes, or `None`.

Shared functions in the same module derive everything else from these queries, for every
backend alike:

- `current_reads`, `update_reads` and `live_relations` read a cell's consumers and its
  transitions. Relations with zero support are skipped everywhere.
- `live_states` computes the least set of states each cell keeps: its own consumers'
  reads, plus the update reads for every live state of a successor.
- `characteristic_cells` uses these to decide where a characteristic map is eligible.
- `status_axes`, `status_points` and `lane_needs` read the facts at fixed status codes.

## Backends

`facts_backend` is a required argument everywhere the analysis runs, and it has no
default.

| Backend    | Reads                                                                                                                        | Cost                                        |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `metadata` | Scalar expression language on the declared metadata. It ignores fixed statuses and answers for the whole cell.               | Cheap, conservative                         |
| `dense`    | `SourceProof` over the captured original source, with dates, constants, forced actions and fixed statuses bound.             | Exact; one interpretation per query, cached |
| `implicit` | `dense` at every point, stored as a decision diagram that splits a status only where an evaluation with it unfixed reads it. | Exact; shares leaves between equal points   |

Some callbacks get fixed answers:

- A `LinearExpectation` reducer and a closed native constant read nothing.
- A native callback that cannot be interpreted reads every coordinate.

The backends are ordered. At every query, `metadata` reads at least what `dense` reads
and forces at most what `dense` forces. `dense` and `implicit` agree at every status
point, and fixing a status only removes reads.
`tests/regime_finder/test_facts_backends.py` checks the ordering on the bundled models.
It also contains a fixture on which the backends separate: `dense` drops a state that
`metadata` must keep.

An exact backend that exceeds its time budget raises `FactsBudgetError`. The error names
the unfinished regime-periods and suggests `facts_backend="metadata"`. It never returns
an incomplete answer. Setting `limits["facts_workers"]` above 1 computes each cell's
dense facts in a process pool that shares one deadline. Serial is the default: on a
176-cell model the serial facts took 0.5 s, while starting a pool cost several seconds.

## Joint reads

A read set must suffice when every coordinate outside it is free *together*. Testing one
coordinate at a time while fixing the others is not enough. With three binary statuses,
a majority payoff is unchanged by any single flip at `(0, 0, 0)`, yet differs at
`(1, 1, 1)`. The exact backends therefore interpret a callback with only the requested
statuses bound, and every other state, action, shock and parameter free.

## Status splitting

A status state is a categorical state with several codes. `lane_needs` returns the
states a cell needs at the points where some statuses are fixed: what its consumers read
there, plus what its updates read for every state a successor needs. A status that an
update carries unchanged stays fixed in the successor, so only the successor's points
with that code count. For any other update, the successor's whole-cell live states
count.

A lane fixes some status states and leaves the others free. It keeps the states its
points need, including any free status it reads. The lanes are the leaves of a tree that
splits by one status at a time, so together they partition the cell's joint status
rectangle. Splitting by employment and then, among the employed only, by health is one
such tree. A memoized recursion finds the tree with the fewest rows: at each set of
fixed statuses it keeps the lane unless splitting by one more status saves rows. The
rectangle may hold at most `limits["status_lanes"]` points, 128 by default.

A split is one interface option whose single state axis is a derived `lanes` domain.
That domain is sized by the summed lane rows, so cost, merging and both searches use it
unchanged. Cells split the same way share the domain and can share a block. All splits
enter one added projection policy. The `metadata` backend has no per-status facts and
reports `NOT_OFFERED`.

The checker re-derives each split from the analysis's facts. It confirms that:

- every point of the status rectangle lies in exactly one lane;
- every lane keeps each state its points need;
- the domain's rows equal the lanes' grids.

Transitions need no separate check: a lane's needs already include what every state a
successor needs reads in the update.

## What the checker reads

`semantic_check.check_semantics` takes the facts the analysis used. Before re-checking a
stored report, it rebuilds them from callback records re-normalized from the original
capture, so a report cannot change the evidence it is checked against. It still
re-derives two things independently. A consumer that `SourceProof` cannot read keeps the
result unresolved. Each declared function interface is checked against the source,
argument by argument.
