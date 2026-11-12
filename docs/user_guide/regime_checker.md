---
title: Check a regime graph
---

# Check a regime graph

The regime checker looks for ways to simplify the age- or period-indexed regime graph of
a constructed `Model`. It uses the model's declarations and original grid sizes to
propose changes such as removing unused state variables or combining compatible regimes.
Model construction is sufficient. The checker does not solve or simulate the model.

## Check your model

After installing PyLCM, construct your `Model` using your usual setup. Call
`analyze_regime_structure` in the same Python environment.

```python
from lcm.graphs import analyze_regime_structure

result = analyze_regime_structure(model)
print(result.original.cost)
for candidate in result.recommendations:
    print(candidate.cost, candidate.blocks, candidate.checks)
result.write_report("regime-check-results")
```

Open `regime-check-results/report.html` for the graph, alternatives and supporting
checks. Analysis and writing are separate operations. `write_report` accepts a string or
`Path` and requires a new directory. Review saved reports before sharing them because
they can contain source and model inputs.

`RegimeAnalysisResult` and its candidate, block, check and cost records are frozen
dataclasses. Recommendations reference candidate objects directly. `result.checks`
contains global layers; each candidate has its own scoped checks and reasons.
`result.data` exposes recursively immutable detailed records, and `result.to_dict()`
returns a detached serializable copy. The existing `find_regimes(model, output=...)`
helper remains available for callers needing the dictionary interface.

## Read the result

Distinguish preservation of the decision problem, estimated computational improvement,
and optimality within a specified family of representations. Passing one of these checks
does not establish the others.

### Does the proposal preserve the decision problem?

The checker follows the model's declared functions, constraints and conditional laws.
For a proposed reduction, it checks whether the carried information and reconstructed
choices suffice for the affected calculations and requested outputs. Dependence through
successor states, observations before choice and other local decision problems matters.
A state absent from current utility can still affect an offer probability or a future
state. Any proposed compression must preserve that dependence. Preserving shock supports
or separate marginal distributions alone does not preserve their full joint law.

These checks are relative to the captured declarations and the model's pure-function
contracts. An unchanged opaque function can be preserved by retaining its identity,
bindings and required arguments. The checker does not prove that arbitrary Python
functions have the intended economic meaning, or that the recursion has finite values
and attained maxima. Those remain premises of the model.

Read each proposal's independent disposition:

- **Representation / adopt:** the proposed representation specification passes its
  source-bound checks. Application code edited to implement it still needs validation.
- **Retain original / adopt:** keeping the supplied representation is safe by identity.
  This does not establish that it is optimal.
- **Investigate:** a required dependency, transition or other obligation remains
  unresolved. The proposal is not certified for implementation.
- **Reject:** the checks found a contradiction.

Failure to certify a reduction does not prove that no further reduction exists. The
checks preserve the separate solve and simulate declarations and their information and
decision order; they do not require subjective and realized laws to agree.

Several decisions can belong to the same economic interval. A commitment made before an
observation cannot use the information revealed by that observation, while later choices
can. Changing the representation must preserve that order and the model's declared
payoffs and discounting. An additional decision stage does not itself imply an
additional economic discount period. See
[economic time and computational periods](period_time.md).

### Is it an attractive representation?

The search compares structural costs using the original grid sizes and the stated
allocation contract:

| Cost                                            | Meaning                                                                                                                                            |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Work                                            | Declared state-action evaluations plus an assigned cost for interfaces between local problems.                                                     |
| Storage                                         | Allocated value entries. Under separate arrays per representation block, a block's common state domain determines its allocation across its dates. |
| Structural complexity (`clarity` in the report) | A weighted count of blocks, state characteristics and dispatch rules.                                                                              |

These are estimates of computational work, storage and specification complexity. They do
not measure hardware runtime, peak live or device memory, or human readability.

The heuristic recommendation uses a positive weighted sum of costs divided by fixed
reference scales. The report records those weights, scales and the minimum relative
improvement required over the unchanged original. It selects an eligible, independently
checked candidate only when that improvement is established; otherwise it retains the
original. Changing the weights and rerunning can change which candidates are examined.

The default recommendation also defers an extra regime used only to omit newly inactive
coordinates at the final decision date. Such candidates remain visible with their
checks. This is a recommendation preference, not a claim that the transformation is
invalid or absent from the admissible family. The snapshot runner below exposes
`--allow-final-only-split` to override it.

### Is it the best representation?

`analyze_regime_structure(model)` uses a heuristic search. Its recommendation is
selected from the eligible checked candidates it found. Better unexamined
representations can exist, and the global optimality gap is unknown. A checked
improvement is useful without being a certified optimum.

The bundled analysis machinery also includes exact searches and independent coverage
checkers for specified finite families of state projections, regime groupings and date
intervals. The frontier describes tradeoffs rather than a single best graph: improving
one cost can require worsening another. A complete cost-frontier certificate establishes
that every attainable cost vector in that family is weakly dominated by a reported
frontier point with a feasible representation. It supports optimal positive-weight cost
choices and questions about attainable cost budgets within that family and allocation
contract. It does not prove optimality over arbitrary model representations or measured
implementations.

Cost-family coverage and source preservation are separate checks. A complete cost
certificate alone does not establish economic equivalence. Conversely, a proposal can
pass its preservation checks without any proof that search covered the full family. An
incomplete search or verification result cannot support the stronger optimality claim.
Worker completion or a successful command exit is not a completeness certificate.

The public model-analysis functions do not expose an exact-search option. The exact
machinery does not strengthen the guarantee of their default heuristic result. The
snapshot runner exposes exact search explicitly, as described below.

### What must a complete transformation family include?

An optimum over all representations permitted by a stated rule set requires complete
generation of that family as well as complete cost coverage. A constructive exact method
can enumerate the finite local choices for coordinate removal, declared categorical
characteristics and reconstructible forced actions, test their global coherence, and
then enumerate compatible regime groupings and contiguous date intervals. Conditional
observation and joint transition laws remain part of the admission tests. Completeness
is relative to those explicit rules and correct primitive dependency and factorization
declarations; it does not require discovering arbitrary identities between callbacks.

A cost certificate over supplied projection policies alone does not establish that every
rule-permitted policy was supplied. Check generation coverage separately from the
cost-frontier and preservation checks. The resulting theoretical minimum concerns the
declared structural objective and allocation contract, even if exact enumeration is
exponential. It is not a minimum over all economically equivalent descriptions.

This mathematical construction also differs from certifying an implementation's model
capture and family generator. A supplied-family proof or passing finite examples cannot
establish those implementation properties. The public heuristic retains its stated
preservation and improvement checks with an unknown global optimality gap.

## Compare alternative linear cost models

Pass a `LinearRegimeCostModel` to `analyze_regime_structure` to select a supported
objective before candidate generation and pruning. This reruns the search. Reranking the
candidates from an earlier search is not equivalent, because the objective also affects
which candidates are explored and retained.

```python
from lcm.graphs import LinearRegimeCostModel, analyze_regime_structure

cost = LinearRegimeCostModel(
    work_weight=7.0,
    storage_weight=3.0,
    complexity_weight=2.0,
    reference_scales=(1, 1, 1),
    minimum_improvement=0.05,
)
result = analyze_regime_structure(model, cost_model=cost)
```

The coefficients above are illustrative priorities, not fitted runtime predictions.
Weights must be finite and strictly positive. Reference scales must be positive
integers; when omitted, the search uses the unchanged original costs. The minimum
relative improvement is in `[0, 1)`. `allow_final_only_split=True` permits the endpoint
preference override. The supplied objective is recorded with the result.

The supported class is a positive linear combination of declared work, allocated value
entries and structural complexity. Arbitrary nonlinear functions and custom callbacks
are not supported. This restriction preserves the additive cost and dominance conditions
used by the search and independent checks.

The structural score is not an empirically validated predictor of runtime or peak
memory. To evaluate alternatives, measure equivalent implementations separately,
distinguishing construction, compilation, warm execution and memory. Fit coefficients
outside the checker and compare predicted rankings on held-out models or hardware. Fixed
unit reference scales can express fitted coefficients directly. Report the measured
evidence alongside the resulting cost model; running this analysis does not perform that
validation.

## Replay the bundled snapshot

From the root of a PyLCM checkout on Linux, run the following with Pixi.

```sh
pixi run -e default python -m _lcm_regime_finder.pipeline \
    tools/regime-finder/examples/declarations/tiny-huge-grid.json \
    --output regime-snapshot-results
```

Open `regime-snapshot-results/next-actions.html` for the recommendation.

This command also defaults to heuristic search. For an explicit exact-family attempt,
use a new output directory and add `--search-mode exact`. Such searches can be expensive
on large graph families. Inspect the independent coverage and source-check results
before making an optimality claim. The input remains the same declared model and grids;
switching search modes does not justify substituting a smaller model.
