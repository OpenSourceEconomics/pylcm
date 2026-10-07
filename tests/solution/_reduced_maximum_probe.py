"""Find action identities matched by equality against a separately reduced maximum.

A reduction that publishes a maximum and the identity attaining it must take both
from one reduction. Matching the values against a separately reduced maximum
loses the winner whenever the compiler evaluates the values once per reduction
and the two evaluations round differently: no value then equals the maximum.
"""

from jax.extend.core import Jaxpr, Var


def count_equalities_with_a_reduced_maximum(jaxpr: Jaxpr) -> int:
    """Count `eq` equations in `jaxpr` that read a value derived from `reduce_max`.

    Nested jaxprs (e.g. inlined `jit` calls) are followed.
    """
    count, _ = _count_in(jaxpr=jaxpr, derived_inputs=(False,) * len(jaxpr.invars))
    return count


def _count_in(
    *, jaxpr: Jaxpr, derived_inputs: tuple[bool, ...]
) -> tuple[int, tuple[bool, ...]]:
    """Return the count and, per output of `jaxpr`, whether it is a derived value."""
    derived = {
        var
        for var, is_derived in zip(jaxpr.invars, derived_inputs, strict=True)
        if is_derived
    }
    count = 0
    for eqn in jaxpr.eqns:
        reads = tuple(isinstance(var, Var) and var in derived for var in eqn.invars)
        if eqn.primitive.name == "eq" and any(reads):
            count += 1
        nested = eqn.params.get("jaxpr")
        if nested is None:
            outputs = (any(reads) or eqn.primitive.name == "reduce_max",) * len(
                eqn.outvars
            )
        else:
            nested_count, outputs = _count_in(
                jaxpr=getattr(nested, "jaxpr", nested), derived_inputs=reads
            )
            count += nested_count
        derived.update(
            var
            for var, is_derived in zip(eqn.outvars, outputs, strict=True)
            if is_derived
        )
    return count, tuple(
        isinstance(var, Var) and var in derived for var in jaxpr.outvars
    )
