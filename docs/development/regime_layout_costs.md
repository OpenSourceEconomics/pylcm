# Layout cost reporting

The cost search uses the offered coherent state projections, original regime-label
groupings, and contiguous intervals of the original period buckets. This family is
relative to the supplied representation. It does not start from a dense model or split
categorical states into fibres. Reports identify this scope as
`existing family, no dense start`.

## Admission and independent checking

`src/_lcm_regime_finder/block_admission_v1.json` is the versioned declarative admission
specification. `block_admission.py` interprets it for the producer and its diagnostics.
`block_admission_check.py` independently interprets it for the checkers; neither
interpreter calls the other.

New exact results use `regime-frontier/v6`. Historical frontier versions retain their
historical admission rules and default prices when checked. A raw current curve rejects
those legacy certificates instead of treating a historical family as coverage of the
current one. Source fingerprints and worker stability checks include the normative JSON
as well as the implementation modules.

Cells in a block need compatible execution-interface fields and identical domain
identities for every shared axis name. Their original role, actor, ownership, terminal
classification and report metadata are preserved separately per source cell, with a
checked source-context binding in each block witness. Admission does not require the
same actor or one choice per period. Missing roles are preserved as missing, rather than
treated as an equivalence certificate. An allocation block is not an identification of
the original economic decision nodes.

These structural checks do not establish economic equivalence. Callback laws,
information, terminal payoffs, ownership, queries and projection reconstruction still
need their independent source-semantic checks before a proposed representation can be
adopted.

## Raw costs and curves

The defaults are `{"block": 1, "characteristic": 0, "dispatch": 0}`. Existing explicit
price overrides remain supported by the search. A regime-count curve requires these
default prices; it rejects other prices rather than relabelling a weighted complexity
score as a regime count.

The installed checker command produces a raw curve from an existing `analysis.json`
export (or a report containing that analysis):

```console
pixi run python -m _lcm_regime_finder evaluation-curve analysis.json --output curve
```

It writes `curve/report.json`, sets the unpriced-link objective before searching, and
independently checks retained witnesses. The JSON status is `COMPLETE` or `PARTIAL`; a
successful command does not promote a partial curve to full coverage. The command uses
the existing worker resource and input-protection boundary. It does not construct a
model or run numerical solution code.

The internal reporting functions can also be used on an existing analysis export:

```python
from _lcm_regime_finder.curve import build_curve_report
from _lcm_regime_finder.frontier import search_frontier

allocation = analysis.get("allocation_scenario", analysis["declarations"]["allocation"])
analysis = dict(analysis, allocation_scenario=dict(allocation, boundary_unit=0))
result = search_frontier(analysis, reporting_mode="curve")
report = build_curve_report(analysis, result)
```

`analysis` is the `analysis` object in a checker report, including its normalized
declarations, interfaces and offered projection policies. If an allocation scenario is
supplied, set `analysis["allocation_scenario"]` to that exact scenario before searching.
Raw mode does not call the recommendation function. The reporting function independently
checks the result again; a serialized `PASS` field cannot authorize a completeness
claim.

The `regime-evaluation-curve/v1` JSON contains full frontier witnesses and a curve of
minimum evaluations with **at most** each regime budget. Curve rows reference the
corresponding frontier witness by `graph_id`. Ties use fewer regimes, then less storage,
then the canonical witness. The report contains no normalized score, improvement
threshold or recommended layout.

Each point reports grid evaluations, directed link counts and unpriced link row units
separately. Links are not priced in the curve objective; both search and reporting
require `boundary_unit=0`. A frontier searched with charged links cannot be repriced
afterwards because that search can have discarded relevant layouts. The generic search
still supports explicit nonnegative link prices. Storage counts value entries.
Largest-array sizes count state-action entries in one source regime-period array;
largest stored-value array size is also supplied. Counts of distinct periods and of
regime-period arrays at the largest size distinguish simultaneous regime arrays from
dates. These are entry counts, not measured bytes or peak live memory.

Per-block diagnostics distinguish source-cell allocation from a hypothetical emitted
regime with a source-identity state. The latter rectangle has one slice for each
original source regime at every active period bucket of the block. Its evaluation count
can exceed the current source-cell count when those sources have different active dates.
Both counts and their difference are reported; the search objective is unchanged.

`emittable_as_one_regime` is conservative. A positive value requires a bound original
constructed-Model capture and a complete, unchanged, single-regime identity witness. A
negative value means that such a sufficient witness has not been established, rather
than that emission is impossible. The report lists the criteria and reasons, including
terminal consistency within each period and the absence of an emitted source-selector
dispatcher for pooled blocks. No new regime emitter or native emission certificate is
provided by these diagnostics.

`our_layout` denotes the unchanged input representation. It is admitted only if an
offered coherent identity policy and its original label blocks pass independent
membership checking. The metadata producer can remove unused coordinates or forced
actions before offering any policies, so identity need not be offered. Such a case is
`not_admitted`, with a reason; a projected reference is not silently substituted for the
unchanged layout.

Pareto membership and evaluation-curve attainment are separate fields. A layout with
lower storage can be nondominated while needing more evaluations at its regime budget. A
`dominated` result includes an actual admitted dominating witness. Incomplete and
heuristic results describe checked retained candidates, with no complete-family optimum
claim. Even a complete cost result establishes neither source-semantic preservation nor
optimality over arbitrary equivalent models.
