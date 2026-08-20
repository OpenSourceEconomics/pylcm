"""Declaring a condition once, for every solver to read.

A condition states a fact about the model — `savings >= 0`, `assets < limit` —
without saying what should happen at that boundary. What happens is decided by
whoever consumes it: a constraint makes the region infeasible, and a case
selects which formula applies on each side. The same declaration therefore
serves both, and cannot mean two different things in the two places.

Build one from `ref`:

```python
savings = lcm.ref("savings")
borrowing = savings >= 0.0
```

Combine with `&`, `|`, `~`, and `implies`. The comparison operator settles who
owns the boundary point: `<` leaves it outside the region, `<=` brings it in.

An arbitrary predicate is still legal — `Condition.from_callable` carries it
unchanged — but it exposes no structure, so a solver that must reason about a
condition refuses it instead of accepting it and ignoring what it says.
"""

from _lcm.constraints.ir import Condition, Implies, Ref

__all__ = ["Condition", "implies", "ref"]


def ref(name: str) -> Ref:
    """Refer to a named value by name, for use in a condition.

    Args:
        name: A state, action, DAG output, parameter, or declared margin role.

    Returns:
        A reference whose comparison operators build conditions.

    """
    return Ref(name)


def implies(*, premise: Condition, consequent: Condition) -> Condition:
    """Require the consequent wherever the premise holds.

    True wherever the premise fails, so it constrains only the region the
    premise selects.

    Args:
        premise: The condition selecting where the requirement applies.
        consequent: What must hold there.

    Returns:
        The implication as a single condition.

    """
    return Condition(
        expression=Implies(premise=premise.expression, consequent=consequent.expression)
    )
