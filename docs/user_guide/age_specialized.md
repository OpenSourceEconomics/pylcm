---
title: Age-specialized functions
---

# Age-specialized functions (`AgeSpecialized`)

Some model functions do not just *read* the agent's age — their whole definition changes
with it. The canonical case is a tax-transfer system pinned to a policy date: as a
cohort ages through calendar time, the law that applies to it changes, so the function
computing net income at age 58 is a *different closure* from the one at age 63, not the
same closure with a different `age` argument.

`AgeSpecialized` marks such a function. At model build, pylcm resolves the marker **per
period**: each period's age gets its own concrete function, compiled into that period's
programs. There is no runtime dispatch on age or calendar year.

```python
from lcm import AgeSpecialized


def make_net_income(age: float):
    """Build the net-income closure for the policy year this age falls in."""
    policy_env = load_policy_environment(year_for(age))

    def net_income(gross_income):
        return policy_env.apply(gross_income)

    return net_income


def policy_key(age: float):
    """Identify the closure: ages in the same policy year share a program."""
    return year_for(age)


functions = {
    "net_income": AgeSpecialized(build=make_net_income, signature=policy_key),
    ...,
}
```

## The two contracts

`AgeSpecialized(build, signature)` places two obligations on the user; pylcm cannot
verify either automatically.

1. **Invariant call signature.** Every concrete function `build(age)` returns must
   expose the *same* argument list at every age. Only the constants the closure binds
   may differ. pylcm's static passes (the parameter template, used-variable validation)
   read the function at one representative age and rely on this.

1. **`signature` is a correctness precondition, not a performance hint.** Periods whose
   `signature(age)` values are equal share a single compiled program. An equal signature
   must therefore imply *identical closure behavior* — same policy date, same price
   level, same overrides, same everything the closure binds. If two ages collide to one
   signature but `build` returns different closures, the solve is silently wrong. When
   in doubt, put more into the signature (a tuple of every varying ingredient), never
   less.

Deduplication is what keeps this affordable: an age-invariant `signature` collapses to
one program for the whole horizon (identical to not using the wrapper), and a
policy-date signature compiles one program per distinct policy year, not per period.

## Where it is allowed

`AgeSpecialized` is accepted in `functions` and `constraints` of **non-terminal**
regimes. Everything else is rejected at `Regime` construction, loudly:

- **the regime `transition`**, bare or wrapped in `MarkovTransition` — periodizing the
  regime-transition probability path is not supported;
- **a regime transition that _reads_ an `AgeSpecialized` function**, directly or through
  plain helper functions — regime-transition probabilities are built once, so a
  policy-specialized value flowing into them would reuse one age's closure everywhere;
- **`MarkovTransition(AgeSpecialized(...))`** — policy-specialized stochastic
  transitions are not supported;
- **a direct `state_transitions` value** — see the pattern below instead;
- **anything in a terminal regime** — the terminal value program is built once and
  shared across all periods.

## Policy-dependent laws of motion

A state transition whose content depends on the policy year is written as a **plain
transition reading an age-specialized helper function**:

```python
functions = {
    "new_pension_points": AgeSpecialized(build=make_points, signature=policy_key),
    ...,
}

state_transitions = {
    # A plain function — the policy content lives in the helper it reads.
    "pension_points": lambda pension_points, new_pension_points: (
        pension_points + new_pension_points
    ),
}
```

The simulate-phase next-state programs are built per period, so the helper resolves to
the *current period's* closure inside every transition — the law of motion tracks the
policy year without the transition itself carrying a marker.

## What is resolved when

- **Per period (exact):** the solve `Q_and_F` programs, the simulate decision programs,
  and the simulate next-state programs. These are what determine model behavior, and
  they always use each period's own closure.
- **At one representative age (the regime's first active age):** the parameter template,
  used-variable validation, and the *published* function sets
  (`regime.solution.functions`, `regime.simulation.functions`). These consumers only
  need the (age-invariant) call signature — except one:

`to_dataframe(additional_targets=...)` computes targets from the published simulation
functions. A target that depends on an age-specialized function would therefore be
evaluated under the representative age's policy, not each row's period — so pylcm
**rejects** such targets with `InvalidAdditionalTargetsError`. Quantities you need per
period from a specialized function should be carried inside the model (for example as a
state fed by a plain transition reading the helper).
