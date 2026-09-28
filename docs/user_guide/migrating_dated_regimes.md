(migrating-dated-regimes)=

# Migrate for simpler economics, not another framework

```{important}
This guide accompanies the proposed v3 implementation. The consumer patches are
not runnable against unchanged PR #474. Terminality, ordinary solution access and
solver protocols are preserved; entry-based pruning is not part of this migration.
```

Save original parameter templates, dated value/policy keys, data/parameters, seeds and
small-grid numerical outputs before editing. Inspect the shared population drawer and
every external value reader as well as the Model constructor.

## 1. Separate structural timing from probability zeros

An old survival calculation may contain:

```python
# H is the terminal destination age; old annual-grid body.
return jnp.where(age >= H - 1, 0.0, ordinary_survival)
```

Adding support `("alive", "dead")` without changing its age domain leaves a structural
live self-edge at the horizon. Declare the complete exit law:

```python
transition = ByAge.until(
    H,
    law=MarkovTransition(mortality, targets=("alive", "dead")),
    then="dead",
    start=START,
)
```

Use the existing grid's predecessor, not a year subtraction. Remove obsolete switch
parameters and filler code only after checking utility, taxes, state laws and tests for
other uses. A survival-data zero may remain as data but no longer defines topology.
Preserve state/action-dependent and freely estimated zeros as numerical behavior;
declared support does not shrink with their current value. Structural timing changes
rebuild the model.

## 2. Recover complete laws before deleting target-activity filtering

Some old target mappings give the same remain probability to two stage destinations and
rely on activity to remove one. Keeping both can double-count mass. Split the complete
lotteries rather than renormalizing the erroneous union:

```python
transition = ByAge(
    {
        early_sources: {
            "old_stage": MarkovTransition(remain),
            "exit": MarkovTransition(leave),
        },
        boundary_source: {
            "new_stage": MarkovTransition(remain),
            "exit": MarkovTransition(leave),
        },
    }
)
```

Check ordinary and boundary rows under normal and off logging. Shared function identity
is only a review clue: two half-probabilities can be correct. No new callable-identity
warning or probability-literal syntax is needed to catch actual bad mass. The engine
must reject real invalid regime-selection rows, including ones appearing only at later
ages/actions.

## 3. Preserve terminality and account for terminal availability

Keep terminal regimes as they are:

```python
Regime(regime_transitions=None, functions={"utility": terminal_payoff})
```

Remove their activity predicate, not the regime-level terminal rule. Do not wrap `None`
in an age/phase declaration. Their values are now available at every clock age. Keep the
final living decision and transition into the terminal payoff; changing that source to
`None` removes its continuation.

Compare old and new output keys explicitly. Extra early terminal values are not extra
death transitions, and all old living/counterfactual values must remain. Check added
terminal payoffs and their primitive inputs rather than declaring the key set unchanged.
If a terminal payoff is undefined at newly covered ages, record the incompatibility and
resolve its economic specification; do not silently invent an extension or sneak a
terminal-domain mask back into this API.

**Borella:** removing its two terminal-only schedules adds the zero-payoff regimes at
25–66. Its 365 previous nodes remain; there are now 449 (297 living, 152 terminal). The
supplied smoke patch asserts exactly the new age-25 keys, terminal values of zero at
every age and the 449 total, while preserving the paper-derived negative- infinity mask
and living grid-shape checks. The full smoke file is intentionally not byte-identical.
Existing live transition support has no newly added early-death edges. **ACA:** the
terminal problem already covered every age, so coverage stays 182.

## 4. Keep ordinary numerical interfaces

```python
transition = MarkovTransition(old_vector_law, targets=("alive", "dead"))
transition = Choose(old_selector, targets=("working", "retired"))
```

Preserve global regime IDs and vector packing. A deterministic selector still returns a
global code and adds no draw. Retain current per-target wrappers where their parameter
paths are public. Do not turn a vector into scalar cells or a selector into indicator
probabilities just to fit the syntax.

A string exit intentionally declares determinism. Check downstream RNG-site stability
before accepting a conversion from an old stochastic exit; equal probability vectors
alone do not certify equal seeded panels. A singleton Markov mapping remains stochastic
and must not be silently optimized into a string law.

State schemas, state/joint producers, collective routes and carried-state phase
semantics stay unchanged. The graph consumes those declarations rather than requiring a
new all-purpose Edge object.

## 5. Replace duplicate timing authority, not all internal names

A work regime ending before 62 can use:

```python
transition = ByAge.until(62, law=work_law, then="early", start=25)
```

For fractional/irregular grids use exact ages or `AgeRange`. Python `range` still
selects integers. Interval selectors do not create new clock points. A shared
model-specific stage table is useful when it generates both numerical routing and
structural support. It must not filter invalid destinations independently.

Delete `Regime.active` declarations and reads. A model that uses any dated form rejects
`active`; models that use none of them keep evaluating it until every consumer has
migrated. Keep existing period-indexed storage and `Model.reachability`. A derived
internal field named `active_periods` is not a second authority merely because of its
name; mechanical renaming is not a benefit. Keep ordinary period-keyed solution reads
and existing exceptions. There is no mandatory new graph, history or result API.

## 6. Migrate the shared drawer and retain external values

The drawer replaces calls to `.active(age)` with joint membership in
`model.initial_nodes`. Validate explicit correlated recipe support rather than
multiplying age/regime marginals, resampling invalid declared pairs, or silently
renormalizing its probability distribution. Model-dependent draw grids may still be
computed after Model build.

Most callers need no `initial_regimes`: default permissions are all coverage and the
recipe specifies the actual population once. A narrower protocol may declare:

```python
initial_regimes = {25: ("single", "couple")}
```

It changes admission only. `{}` is just empty permissions; `solve` behaves normally. A
bare name selects its covered ages, not a blind cross-product with the clock. No
recipe-export framework or new public validation method is required.

Inventory counterfactual comparisons, second-stage policy reads and time-zero outside
options. Every declared value problem remains solved regardless of entry. An absent
lookup must fail, not fabricate zero or re-solve lazily. Keep entry metadata out of
economic solution identity without bypassing existing provenance or
code/grid/parameter/artifact checks. General cross-model/partial-solution reuse is a
different feature.

## 7. Lower all cases without broadening phases or solver interfaces

Use `Phased` inside age cases and preserve its current form/support/gate rules. Union
parameter discovery, variable liveness and handoff requirements over all selected cases
and phases. Do not analyze only the first covered age. No new case level is added to
parameters: repeated compatible paths share a leaf; incompatible schemas/namespace roles
raise. Different values need distinct names or the existing period-indexed array
mechanism.

Gate folds belong to the source's selected transition. A gate absent at a boundary must
not demand reference values there; an incoming gate is not a local dependency of its
attempted target. Retain ordinary local same-period cycle checks.

Normalize schedules before ordinary solver validation. An equivalent resolved law should
follow its existing supported route regardless of wrapper spelling. Genuine unsupported
variation raises using existing errors; do not add a schedule opt-in flag or change
actions to pass. Validate GridSearch first, other adapters separately.

## 8. Three passes to obtain the benefits

| Pass                          | Purpose                                                                                     | Evidence before moving on                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Preserve meaning              | Complete support, valid probabilities, retained living/caller values                        | Baseline/new common-domain values, exact masks, paths, RNG and handoffs   |
| Remove duplicate declarations | Delete stage switches/fillers, old predicates and redundant entry tables; share law objects | One timing authority, all-case discovery, migrated drawer/probes/docs     |
| Measure optimization          | Reuse equivalent programs; specialize only where total cost benefits                        | Matched build/compile, warm solve/simulate/validation and memory receipts |

Do not generate a separate numerical closure for each age simply because schedules allow
it. ACA reuses one source numerical function and its probability cells across cases;
different destination schemas may still split continuation programs. Added terminal
outputs may also cost storage/work. Measure actual backend compiles rather than infer
them from Python identity or assume the expansion is free.

## Completion

A consumer migration is not complete until real package constructors, GridSearch
solves/simulations, external reads, default/off-log invalid-row controls, parameter/
RNG/replay checks and its shared-drawer/probe/test collateral are covered. The provided
Borella archives omit collateral identified by the census; locate it in the full local
checkout. Do not invent hunks or count missing-source cases as passes.

Apply the prepared documentation pages and fix contradictory existing notebooks,
reference pages and examples; then execute relevant fixtures and build the book. Run the
broader paper campaign in a source-linked ledger, without making a new solver or an
unavailable paper dataset a prerequisite for the solver-independent core. Do not label
that core acceptance as whole-zoo numerical acceptance.
