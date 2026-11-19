---
title: Check a regime graph
---

# Check a regime graph

The regime checker asks how the age- or period-indexed regimes of a constructed `Model`
could be laid out more cheaply. It reads the model's declarations and original grid
sizes and drops states that no calculation needs. It then searches for ways to pool
regimes or split them by a categorical status. Model construction is sufficient; the
checker does not solve or simulate the model.

## Check your model

```python
from lcm.graphs import analyze_regime_structure

result = analyze_regime_structure(model, facts_backend="dense")
print(result.original.cost)
for candidate in result.candidates:
    print(candidate.max_blocks, candidate.cost)
print(result.recommendation.identifier)
result.write_report("regime-check-results")
```

Open `regime-check-results/report.html` to see the curve, each layout's blocks and
graphs, and the checks. `write_report` takes a string or a `Path` and requires a new
directory. Review saved reports before you share them, because they can contain source
and model inputs.

`facts_backend` is required. It decides where the analysis learns what each
regime-period depends on:

- `"metadata"` reads the declared scalar expressions. It is fast and conservative: a
  function outside that language keeps every input it declares.
- `"dense"` interprets each function's captured source, with the period's dates,
  constant states, forced choices and fixed statuses bound. It finds every coordinate
  the model needs and no more.
- `"implicit"` gives the same answers as `"dense"`, and stores the facts that differ
  across a status as decision diagrams.

When an exact backend runs out of time, it raises `FactsBudgetError` and names the
regime-periods it could not finish. `"metadata"` then gives a conservative result.

`find_regimes(model, facts_backend=...)` returns the same analysis as a dictionary. It
writes the report when you pass `output`.

### Run the layout search alone

```python
from lcm.graphs import layout_curve

curve = layout_curve(model, facts_backend="dense", limits={"max_blocks": 6})
```

`layout_curve` runs only the layout search and returns its record, described in
[Layout search](../development/regime_layout_costs.md). It runs no source check, no
layout check and no recommendation. Use it when you need the curve itself, for example
to price given blocks through `price` or to time the search; use
`analyze_regime_structure` when you need a checked curve and a recommendation.

## Read the result

`result.candidates` is the layout curve. For each block budget `k`, its candidate is a
layout with the fewest state-action evaluations among those with at most `k` regimes. A
layout pools whole regimes, and splits a regime into status lanes when different values
of a categorical state need different coordinates. Each block in `candidate.blocks`
lists:

- the original regimes, periods and dates it covers;
- the states and actions it keeps;
- its `lanes`, as `lcm.graphs.StatusLane(regime, statuses)` records;
- its own cost.

| Cost                                   | Meaning                                                       |
| -------------------------------------- | ------------------------------------------------------------- |
| Work                                   | State-action evaluations: periods × state grid × action grid. |
| Storage                                | Stored values: periods × state grid.                          |
| Regime count (`clarity` in the report) | The number of regimes (blocks).                               |

These figures count grid entries. They do not measure runtime, peak memory or
readability.

`result.status` is `COMPLETE` when three conditions hold:

- the source check passed;
- every point of the curve is exact in the searched family;
- the independent layout check passed.

The status is `UNRESOLVED` when a source premise remains open, or when the curve is
`RESTRICTED`. A restricted curve merged some status lanes it could not prove mergeable,
so its points above the first are upper bounds. In the report data,
`restriction_reasons` says which statuses were merged and why. The status is `INVALID`
when a check failed.

The family, the counts and the exactness rules are those of the paper's Supplement. See
[Layout search](../development/regime_layout_costs.md).

### Does a layout preserve the decision problem?

The checker follows the model's declared functions, constraints and conditional laws. A
state is kept wherever a current calculation reads it, or a calculation in a successor
period needs it. A state that is absent from current utility can still affect an offer
probability or a future state, and it is kept for that reason. Every layout of the
family solves the same decision problem on the coordinates it keeps.

These checks are relative to the captured declarations and to the model's pure-function
contracts. The checker does not prove that arbitrary Python functions carry their
intended economic meaning, or that the recursion has finite values and attained maxima.
Those remain premises of the model. Several decisions can belong to one economic
interval; see [economic time and computational periods](period_time.md).

### Collective models

In a model with collective regimes, the checker holds to these rules:

- A collective regime pools only with regimes that have the same stakeholders, so never
  with a single-agent regime.
- Regimes pool only when their exits are equal: the same rule, tie, comparison,
  renegotiation and settlement, the same targets with the same roles, the same observed
  shocks and the same availability.
- Work and storage count one action value and one stored value per stakeholder, as the
  solve computes them.
- The state an exit's `Renegotiate` moves stays in its regime whole: it is kept in every
  period and never split into status lanes, because renegotiation compares the stay
  values at its codes within the period.

A regime that holds a `Problem` is checked as the regime that declares the problem's
slots itself, with the same result: the model takes each holder's slots at construction.

### Which layout is recommended?

`result.recommendation` is the curve point that minimizes a linear cost model. If no
point improves on the original model by the required margin, it is the original. Pass a
`LinearRegimeCostModel` to change the weights:

```python
from lcm.graphs import LinearRegimeCostModel, analyze_regime_structure

cost = LinearRegimeCostModel(
    work_weight=7.0,
    storage_weight=3.0,
    complexity_weight=2.0,
    reference_scales=(1, 1, 1),
    minimum_improvement=0.05,
)
result = analyze_regime_structure(model, facts_backend="dense", cost_model=cost)
```

The score is the weighted sum of work, storage and regime count, each divided by its
reference scale. When `reference_scales` is omitted, the scales are the original model's
costs. The coefficients above are illustrative priorities, not fitted runtime
predictions. The curve itself does not depend on the cost model; only the choice among
its points does.

`RegimeAnalysisResult` and its candidate, block, lane, check and cost records are frozen
dataclasses. `result.data` exposes the detailed records, recursively immutable, and
`result.to_dict()` returns a detached, serializable copy.

## Replay the bundled snapshot

From the root of a PyLCM checkout, run:

```sh
pixi run -e default python -m _lcm_regime_finder.pipeline \
    tools/regime-finder/examples/declarations/tiny-huge-grid.json \
    --facts-backend metadata --output regime-snapshot-results
```

Open `regime-snapshot-results/next-actions.html` for the curve and the recommendation.
The command-line analysis accepts the same limits as the search: `--max-blocks`,
`--lanes-per-regime` and `--seconds`.
