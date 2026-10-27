---
title: Categories with characteristics
---

# Categories with characteristics

In some models the categories of a discrete state or action are combinations of
characteristics. In {cite}`lowpistaferri2015`, a labour-market status is "employed",
"applying for disability insurance", "on disability insurance for one quarter", and so
on, and each status fixes whether the person holds a job, whether they may apply, and
how long they have received benefits. Other models have the same structure: family
composition (number of children and age of the youngest, where "no children" has no
age), job offers (hours and insurance, where "no offer" has neither), or a migration
choice (destination and border-crossing mode). The
[LCM Zoo](https://github.com/OpenSourceEconomics/lcm-zoo) contains several such models.

Only some combinations occur, so the model tracks one categorical variable with one
category per valid combination. Model functions, however, reason in characteristics.
This page shows how to connect the two with a single table that pylcm checks.

## Declare each category's characteristics once

Declare the categories as usual, and next to them a table that lists every category's
characteristics, keyed by category name. `characteristics` checks the table against the
class and returns one array per characteristic, indexed by code.

```python
from lcm import categorical, characteristics
from lcm.typing import ScalarInt


@categorical(ordered=False)
class LabourStatus:
    employed: ScalarInt
    unemployed: ScalarInt
    applicant: ScalarInt
    di_1: ScalarInt
    di_2: ScalarInt


LABOUR_STATUS_CASES = {
    "employed": {"has_job": True, "is_applicant": False, "di_years": 0},
    "unemployed": {"has_job": False, "is_applicant": False, "di_years": 0},
    "applicant": {"has_job": False, "is_applicant": True, "di_years": 0},
    "di_1": {"has_job": False, "is_applicant": False, "di_years": 1},
    "di_2": {"has_job": False, "is_applicant": False, "di_years": 2},
}
LABOUR_STATUS = characteristics(
    category_class=LabourStatus, by_category=LABOUR_STATUS_CASES
)
```

The check raises when the table and the class disagree:

- the table lists the categories of the class, in declaration order;
- every category sets the same characteristics, in any order;
- each column contains only Python `bool`, only Python `int`, or only finite Python
  `float` values; write `0.0` rather than `0` in a float column;
- float characteristics lie within the float32 range, so they stay finite whether x64 is
  enabled or not;
- integer characteristics fit int32 and differ from the int32 minimum, which marks a
  missing category;
- no two categories share all characteristics.

The last rule makes each category equal to its combination of characteristics. Two
categories that the model treats differently differ in some characteristic; name it.
Above, `is_applicant` separates applicants from the unemployed.

## Read characteristics by name in model functions

Publish each characteristic a model function needs as a one-line function of the state
or action, named `<variable>_<characteristic>`, and register it in the regime's
`functions`:

```python
import jax.numpy as jnp

from lcm.typing import BoolND, DiscreteAction, DiscreteState, FloatND, IntND


def labour_status_has_job(labour_status: DiscreteState) -> BoolND:
    return jnp.asarray(LABOUR_STATUS["has_job"])[labour_status]


def labour_status_di_years(labour_status: DiscreteState) -> IntND:
    return jnp.asarray(LABOUR_STATUS["di_years"])[labour_status]


def earnings(*, wage: FloatND, labour_status_has_job: BoolND) -> FloatND:
    return wage * labour_status_has_job
```

- The prefix keeps characteristics apart from economic quantities. A function that
  combines a characteristic with a choice, such as benefits received only when not
  working, keeps its own economic name and takes the characteristic as an argument.
- The table holds NumPy arrays, and `jnp.asarray` inside the function moves the column
  to the device when the function runs.

## Attach numerical properties to a category

A discrete category can carry several numerical properties. For example, each job type
can specify both hours and a wage multiplier:

```python
@categorical(ordered=False)
class Job:
    none: ScalarInt
    part_time: ScalarInt
    full_time: ScalarInt


JOBS = characteristics(
    category_class=Job,
    by_category={
        "none": {"hours": 0, "wage_multiplier": 0.0},
        "part_time": {"hours": 20, "wage_multiplier": 0.8},
        "full_time": {"hours": 40, "wage_multiplier": 1.15},
    },
)


def job_wage_multiplier(job: DiscreteState) -> FloatND:
    return jnp.asarray(JOBS["wage_multiplier"])[job]
```

These values are illustrative. The job remains a discrete category with an integer code.
Its wage multiplier is a property read by code, not a continuous action. A
[numeric discrete grid](grids.md) instead supplies the numerical value of the variable
itself; a continuous grid supports approximation over a continuous variable.

The checked table stores bool columns as NumPy bool, integer columns as int32, and float
columns as float64. Each column is read-only. Conversion with `jnp.asarray` uses the
active JAX precision policy. Keep floats as properties: `category_with` accepts only
integer and Boolean identifying columns.

## Map characteristics back to a category

A deterministic transition can describe the next category by its characteristics.
`category_with` returns the code of the category with the given characteristics:

```python
from lcm import category_with


@categorical(ordered=False)
class Work:
    no: ScalarInt
    yes: ScalarInt


def next_labour_status(*, labour_status_di_years: IntND, work: DiscreteAction) -> IntND:
    on_di = category_with(
        LABOUR_STATUS,
        has_job=False,
        is_applicant=False,
        di_years=jnp.minimum(labour_status_di_years + 1, 2),
    )
    return jnp.where(work == Work.yes, LabourStatus.employed, on_di)
```

- The characteristics you pass must tell all categories apart. A subset is enough when
  no two categories share it; otherwise `category_with` raises before reading any value.
- Queried columns must have bool or signed-integer dtype. Float properties may be
  present in the table, but cannot be queried, even with an integer query value.
- Values are Python ints or bools, or JAX arrays of bool or signed-integer dtype, also
  inside `jit` and `vmap`, and broadcast against each other. Unsigned and float arrays
  raise, because JAX can compare them with int32 characteristics in a way that matches
  the wrong category.
- A combination that no category has returns the missing-category code, the int32
  minimum. So does any Python int outside int32.

Codes and class constants are the same kind of value, so branches can mix them freely.

## Stochastic transitions

A stochastic transition returns one probability per category, in declaration order.
Build the probabilities by name and stack them in the order of the checked table, so
that adding a category cannot shift them silently:

```python
def next_labour_status_probs(
    *, labour_status_di_years: IntND, work: DiscreteAction
) -> FloatND:
    probs = {
        "employed": ...,
        "unemployed": ...,
        "applicant": ...,
        "di_1": ...,
        "di_2": ...,
    }
    return jnp.stack([probs[name] for name in LABOUR_STATUS_CASES], axis=-1)
```

## Output

Published characteristics are model functions, so simulation output includes them on
request:

```python
result.to_dataframe(
    additional_targets=["labour_status_has_job", "labour_status_di_years"]
)
```
