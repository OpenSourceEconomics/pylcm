(migrating-dated-regimes)=

# Migrating to age-indexed regimes and explicit roots

This guide moves a consumer model to required `initial_regimes`, keyword-only
declarations and demand-derived solved problems. Before editing, save parameter
templates, age-indexed value/policy keys, data, seeds and small-grid numerical outputs,
and inventory the shared population drawer and every external value reader as well as
the `Model` constructor.

## 1. Replace optional, bare and empty roots

`initial_regimes` is required and has no default. Replace:

- an omitted argument or `None` by the mapping of intended starting problems;
- a bare name or sequence (`"single"`) by an explicit age rule, e.g.
  `{AGE_START: "single"}`;
- `{}`, which is rejected, by the roots the caller actually needs.

A fixed paper-model factory may own its concrete, documented roots. A configurable
factory requires the caller to pass them rather than re-creating a default one level
higher.

## 2. Translate solve-only and external-query callers

A caller that only solves still declares the problems whose values it reads: those are
its starting problems, and it simply never calls `simulate`. A second-stage policy read,
a counterfactual comparison or a time-zero outside option at a pair that the baseline
roots do not demand belongs to a separately rooted model that declares that pair. Do not
widen the baseline factory's roots for it.

## 3. Do not blindly use the first age or first regime

Rooting at the first age of the first regime is wrong for a mixed-age application and
for a model whose population starts in several regimes. It silently drops problems the
application starts in, and wrongly admits histories in a regime the data never starts
in. Write the intended starting contract from the model's own timing constants and
regime groups.

## 4. Do not blindly preserve every old output key

Values that were solved because a schedule covered an age are not a requirement. Keep a
key only if a real reader needs it; otherwise let demand drop it. For each reader, name
the key it reads, check it is in the derived set, and add a root only when the reader
genuinely starts there. Roots never stand in for prerequisites the engine infers:
continuations, gate references and outside options are followed automatically.

## 5. Structural horizon zeros

An old survival calculation may contain:

```python
# fragment: H is the terminal destination age
return jnp.where(age >= H - 1, 0.0, ordinary_survival)
```

Declaring support `("alive", "dead")` without changing its age domain leaves a
structural living self-edge at the horizon. Declare the complete exit law:

```python
# fragment
regime_transitions = ByAge.until(
    start_age_inclusive=START,
    stop_age_exclusive=H,
    law=MarkovTransition(func=mortality, targets=("alive", "dead")),
    then="dead",
)
```

The last source below `H` uses `then`, not `law`; its exit lands at `H`. Use the grid,
not a year subtraction. A survival-data zero may remain as data but does not define
topology; state/action-dependent and estimated zeros stay numerical behavior.

## 6. Doubled mass from over-wide maps

Some target mappings give the same remain probability to two stage destinations and rely
on a filter to remove one. Keeping both double-counts mass. Split the complete lotteries
instead of renormalizing the union:

```python
# fragment
regime_transitions = ByAge(
    cases={
        early_sources: {
            "old_stage": MarkovTransition(func=remain),
            "exit": MarkovTransition(func=leave),
        },
        boundary_source: {
            "new_stage": MarkovTransition(func=remain),
            "exit": MarkovTransition(func=leave),
        },
    },
)
```

Check ordinary and boundary rows with logging on and off. Shared function identity is
only a review clue: two half-probabilities can be correct.

## 7. Source-owned gates

A gate on a `ValueDependentTransition` is a requirement of the **source edge**,
evaluated at the landing age; its references are solved because that edge demands them,
not because the attempted target depends on them. A case that removes the gate at an age
removes its reference demand there. Reciprocal marriage/divorce edges do not form a
same-period cycle.

## 8. Mixed-age recipes versus merely reachable states

A root is a pair where the population recipe may genuinely start a history. A pair that
histories only reach through transitions is not a root, even though it is solved. The
recipe owns the distribution and age/regime correlation; the drawer checks joint
membership in `model.initial_nodes` and never resamples or renormalizes invalid pairs.

The paper Borella model starts at one age in three regimes:

```python
# fragment
initial_regimes = {
    AGE_START: (
        "single_m_work",
        "single_f_work",
        "couple_work",
    ),
}
```

The claimed-couple and widow problems are reached from these roots. A separate query
that genuinely starts at the age-62 claimed-couple or age-66 widow problems adds those
pairs in its own model; that is not the paper factory's starting universe.

ACA admits mixed-age starts at ages 51–60 in five living regimes, and later ages are
reached only through transitions:

```python
# fragment
initial_regimes = {
    AgeRange(start=51, stop=61): (
        "retiree_nomc_inelig_canwork",
        "tied_nomc_inelig_canwork",
        "nongroup_nomc_inelig_canwork",
        "retiree_dimc_inelig_canwork",
        "nongroup_dimc_inelig_canwork",
    ),
}
```

`dead` is never an ACA root. Its values are solved as continuations, and an empirical
start in `dead` — at any age, including ages where that value is solved — or at an age
of 61 or above is rejected. Terminal roots are shown in a separate synthetic model in
[the interface guide](dated_regime_graph.md), not by admitting death in an application.

## 9. Parameter paths and random sites

Roots and schedules add no parameter level: paths, per-target cells, regime IDs and
vector packing keep their meaning. A string exit declares determinism; check downstream
RNG-site stability before converting a stochastic exit, since equal probability vectors
do not certify equal seeded panels. A singleton Markov mapping stays stochastic.

Simulation draws are keyed by site: a subject's draws at a `(period, regime)` pair
derive from the seed folded with that period and the regime's registered code, so they
do not move when the declared starts make other pairs visitable. This key derivation is
new with age-indexed declarations, so a panel simulated with a given seed differs once
from the one the same seed produced before; re-record seeded panel fixtures once, and
compare them afterwards across root changes.

## 10. Approved consumer key changes

Compare old and new output keys explicitly and record each difference as an approved
consumer change: problems that demand no longer reaches (for example early terminal
values nothing reads) disappear, and every value a real reader needs remains. An absent
lookup raises; it is never filled with zero or re-solved lazily.

## 11. Semantic parity before optimization

| Pass                          | Purpose                                                                     | Evidence before moving on                                                 |
| ----------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Preserve meaning              | Explicit roots, complete support, valid probabilities, retained reader keys | Common-domain values, exact masks, paths, RNG and handoffs                |
| Remove duplicate declarations | One timing authority, keyword-only declarations, migrated drawer            | All-case discovery, migrated probes, tests and docs                       |
| Measure optimization          | Reuse equivalent programs; specialize only where total cost benefits        | Matched build/compile, warm solve/simulate/validation and memory receipts |

Do not generate a separate numerical closure for each age simply because schedules allow
it, and measure backend compiles rather than inferring them from Python identity.
