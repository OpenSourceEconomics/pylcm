# EP-01: Internal State-Action-Space Representation

## Abstract

This LEP outlines the use of an internal index representation for states and actions to
address two challenges. First, it enables the implementation of discrete actions with a
time-dependent set of valid actions, thereby reducing the state–action space in certain
periods. Second, it supports continuous stochastic states and allows continuous states
and actions to influence stochastic transitions.

## Backwards Compatibility

All changes are fully backward compatible.

## Motivation

To Be Written.

## Index Representation

To Be Written.

## Period-dependent Discrete Actions

Consider a model where agents have to work until they are 60, then they can decide
whether they want to work or retire until they are 70, and then they must retire. This
is a working decision (action) with categories `WORK` and `RETIRE`, where the valid
action categories are constrained by the agents' age.

To model this in the current implementation of PyLCM, we have to build a state-action
space including both categories in all periods, and evaluate all required functions on
the full state-action space, only to mask all action combinations that are not valid for
a given age. In this example, this means that for the age <=60, and >70 we have to
evaluate on a state-action space that is *twice* as large as it has to be.

Ideally, we would like to have a state-action space that only includes the valid actions
for each period, where the validity of an action-category depends on the period. In the
following, we will map out how to implement this for discrete actions.

### Benefits

The main benefits would be reduced computation time, as we do not have to evaluate
functions on state-action combinations that include invalid actions.

### Problem: Index interpretation

The main problem is that discrete actions must be represented using index values, and
these index values are used in the user code. Following the above example, the valid
working action grids would be

- Up to 60: `[WORK]`
- 61 - 70: `[WORK, RETIRE]`
- 71 and above: `[RETIRE]`

but since discrete actions are modelled through indices, this amounts to

- Up to 60: `[0]`
- 61 - 70: `[0, 1]`
- 71 and above: `[0]`

And therefore, depending on what the user expects index 0 to represent, either in the
first time block or the last time block, a mistake occurs.

### Solution: Index interpretation

To solve the above problem, we require two things:

1. A ground truth of what all possible action-categories are, which defines the
   ordering.

   In the above example, this would be `[WORK, RETIRE]`, implying `WORK` &#10132; 0 and
   `RETIRE` &#10132; 1.

2. Information about which action-category is valid in which period.

   In the above example, in the case >70, we would need to know that only `RETIRE` is
   valid, giving us enough information to know that index 0 of the ">70"-case action
   grid, actually corresponds to the global action-category index 1.

### Interface Design

There are several ways to design a user-interface that allows us to collect enough
information to solve the index interpretation problem.

#### Via Constraints

To define the ground truth we stick to our standard way for defining discrete actions:

```python
from dataclasses import dataclass
from lcm import DiscreteGrid, Model

@dataclass
class WorkingAction:
    work: int = 0
    retire: int = 1

actions={
   "working": DiscreteGrid(WorkingAction),
}
```

And to collect information about which action-category is valid in which period, we
can use constraints:

```python
from lcm.typing import BoolND
import jax.numpy as jnp

def working_constraints(working, _period) -> BoolND:
   if _period <= 60:
      return working == WorkingAction.work
   elif _period <= 70:
      return True
   else:
      return working == WorkingAction.retire
```

We can capture such constraints easily by checking whether a constraint function only
depends on a discrete action and the period. Once captured, we have to use the information
in PyLCM's model processing step and build corresponding period-dependent action grids,
as well as period-dependent model functions.

Since the meaning of the index changes (see Section "Problem: Index interpretation")
we must make sure that the index interpretation remains as expected by the user. To
fulfill this, we can hijack the generated model function `working` and define it
on a per-period basis:

```python
def get_working_model_function(_period):
   if _period <= 60:
      categories = [0]
   elif _period <= 70:
      categories = [0, 1]
   else:
      categories = [1]

   def working(__working__):
      return categories[__working__]

   return working
```

#### Via Grids

Another approach is to define the valid action-categories for each period. Since these
will most likely be constant over multiple time periods, one could work with slices that
define the valid action-categories.

```python
from dataclasses import dataclass
from lcm import DiscreteGrid, Model

@dataclass
class WorkingAction:
    work: int = 0
    retire: int = 1

actions={
	"working": DiscreteGrid(
		WorkingAction,
		valid={
			":60": [WorkingAction.work],
			"61:70": WorkingAction,
			"70:": [WorkingAction.retire],
		}
	),
}
```

We would then have enough information to build the corresponding period-dependent action
grids, as well as period-dependent model functions (see previous section).

#### Continuous Actions

Using the "Constraint"-approach, we can extend this to continuous actions as well. If
naively implemented, this will, however, clash with the proposed continuous stochastic
variables (see Section "Continuous Stochastic Variables"), since there is no *single
ground truth* of indices in the continuous case, rendering it impossible to index into
the Markov transition kernel. For this to work, the Markov transition kernel must treat
the period dimension differently as well.

**Proposal: Time-dependent Markov Transitions**

Currently the Markov transition kernel is defined as n-dimensional array, where time can
be one of the dimensions. To allow for differently shaped grids per time-period, the
complete Markov transition could be defined as a dictionary mapping the time-period to
the respective transition kernel. If it is an array, it will be assumed time-constant.


## Continuous Stochastic Variables

To Be Written.
