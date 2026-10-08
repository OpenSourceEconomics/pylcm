"""Every shipped hard maximum differentiates like `jnp.max`.

The maximum of a block is a function of its values. Where one feasible value
is strictly largest the derivative is that value's tangent; on a tie it is the
average of the tied values' tangents. The identity reported alongside the
maximum is an integer and carries no derivative. The expected derivatives come
from an exact `Fraction` evaluation of that rule, independent of JAX.
"""

import itertools
from collections.abc import Callable
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from _lcm.execution.reductions import HARD_MAX_WITH_CARRY_REDUCTION
from _lcm.regime_building.argmax import argmax_and_max
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from _lcm.solution.collective_action_reduction import COLLECTIVE_HARD_MAX_REDUCTION
from lcm.typing import BoolND, FloatND, IntND


def _argmax(
    *, values: FloatND, feasible: BoolND, action_ids: IntND
) -> tuple[FloatND, IntND]:
    del action_ids  # `argmax_and_max` reports positions
    identity, maximum = argmax_and_max(
        a=values, axis=values.ndim - 1, initial=-jnp.inf, where=feasible
    )
    return maximum, identity


def _hard_max(
    *, values: FloatND, feasible: BoolND, action_ids: IntND
) -> tuple[FloatND, IntND]:
    result = HARD_MAX_REDUCTION.finalize(
        accumulator=HARD_MAX_REDUCTION.add(
            accumulator=HARD_MAX_REDUCTION.initialize(
                value_template=jnp.zeros(values.shape[:-1], dtype=values.dtype)
            ),
            values=values,
            feasible=feasible,
            action_ids=action_ids,
        )
    )
    return result.best_value, result.best_global_action_id


def _collective(
    *, values: FloatND, feasible: BoolND, action_ids: IntND
) -> tuple[FloatND, IntND]:
    result = COLLECTIVE_HARD_MAX_REDUCTION.finalize(
        accumulator=COLLECTIVE_HARD_MAX_REDUCTION.add(
            accumulator=COLLECTIVE_HARD_MAX_REDUCTION.initialize(
                stakeholder_template=jnp.zeros(
                    (*values.shape[:-1], 2), dtype=values.dtype
                )
            ),
            objectives=values,
            stakeholder_values=jnp.stack((values, values + 2), axis=-1),
            feasible=feasible,
            action_ids=action_ids,
        )
    )
    return result.best_objective, result.best_global_action_id


def _hard_max_with_carry(
    *, values: FloatND, feasible: BoolND, action_ids: IntND
) -> tuple[FloatND, IntND]:
    result = HARD_MAX_WITH_CARRY_REDUCTION.finalize(
        accumulator=HARD_MAX_WITH_CARRY_REDUCTION.add(
            accumulator=HARD_MAX_WITH_CARRY_REDUCTION.initialize(
                value_template=jnp.zeros(values.shape[:-1], dtype=values.dtype)
            ),
            values=values,
            feasible=feasible,
            action_ids=action_ids,
        )
    )
    return result.best_value, result.best_candidate_id


_REDUCERS: dict[str, Callable[..., tuple[FloatND, IntND]]] = {
    "argmax_and_max": _argmax,
    "hard_max": _hard_max,
    "collective_hard_max": _collective,
    "hard_max_with_carry": _hard_max_with_carry,
}


def _exact_maximum(
    *,
    values: tuple[float, ...],
    feasible: tuple[bool, ...],
    tangent: tuple[float, ...],
    action_ids: tuple[int, ...],
) -> tuple[float, int, Fraction, tuple[Fraction, ...]]:
    """Return the maximum, the smallest winning id, the JVP and the gradient."""
    exact = tuple(Fraction(value) for value in values)
    best = max(value for value, live in zip(exact, feasible, strict=True) if live)
    winners = tuple(
        index
        for index, (value, live) in enumerate(zip(exact, feasible, strict=True))
        if live and value == best
    )
    gradient = tuple(
        Fraction(1, len(winners)) if index in winners else Fraction(0)
        for index in range(len(values))
    )
    jvp = sum(
        (weight * Fraction(t) for weight, t in zip(gradient, tangent, strict=True)),
        start=Fraction(0),
    )
    return float(best), min(action_ids[index] for index in winners), jvp, gradient


_CASES = {
    "unique_winner": ([1.0, 3.0], [True, True], [0.0, 1.0]),
    "masked_higher_candidate": ([10.0, 3.0, 7.0], [False, True, True], [5.0, 2.0, 4.0]),
    "two_way_tie": ([3.0, 3.0], [True, True], [1.0, 3.0]),
    "zero_tie": ([0.0, 0.0], [True, True], [1.0, 1.0]),
    "signed_zero_tie": ([-0.0, 0.0], [True, True], [1.0, 1.0]),
}


@pytest.mark.parametrize("jitted", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("reducer", list(_REDUCERS))
def test_hard_max_value_jvp_and_gradient_follow_the_max_rule(
    *, reducer: str, case: str, jitted: bool
) -> None:
    """Primal, identity, JVP and gradient match the exact max rule."""
    raw_values, raw_feasible, raw_tangent = _CASES[case]
    dtype = jnp.zeros(()).dtype
    values = jnp.asarray(raw_values, dtype=dtype)
    feasible = jnp.asarray(raw_feasible)
    tangent = jnp.asarray(raw_tangent, dtype=dtype)
    ids = jnp.arange(values.shape[-1], dtype=jnp.int32)
    best, best_id, jvp, gradient = _exact_maximum(
        values=tuple(raw_values),
        feasible=tuple(raw_feasible),
        tangent=tuple(raw_tangent),
        action_ids=tuple(range(len(raw_values))),
    )

    def pair(x: FloatND) -> tuple[FloatND, IntND]:
        return _REDUCERS[reducer](values=x, feasible=feasible, action_ids=ids)

    def value(x: FloatND) -> FloatND:
        return pair(x)[0]

    transform = jax.jit if jitted else (lambda func: func)
    observed_value, observed_id = transform(pair)(values)
    _, observed_jvp = jax.jvp(transform(value), (values,), (tangent,))
    observed_gradient = transform(jax.grad(value))(values)

    assert_array_equal(
        [
            float(observed_value),
            int(observed_id),
            float(observed_jvp),
            *np.asarray(observed_gradient).tolist(),
        ],
        [best, best_id, float(jvp), *(float(weight) for weight in gradient)],
    )


def _mutation_cases() -> list[
    tuple[tuple[float, ...], tuple[bool, ...], tuple[int, ...], tuple[float, ...]]
]:
    """Return rows over patterns, exact rescalings, permutations and masks.

    Patterns cover a unique winner, two- and three-way ties and values one
    representable step apart around one; every transformed value is exact in
    the active precision.
    """
    dtype = np.dtype(jnp.zeros(()).dtype)
    one = dtype.type(1.0)
    patterns = (
        (-2.0, 1.0, 3.0),
        (3.0, 3.0, 1.0),
        (1.0, 1.0, 1.0),
        (
            float(np.nextafter(one, dtype.type(0.0))),
            1.0,
            float(np.nextafter(one, dtype.type(2.0))),
        ),
    )
    rows = []
    for pattern in patterns:
        for scale, shift in itertools.product((0.5, 2.0), (0.0, -1.0)):
            values = tuple(scale * (value + shift) for value in pattern)
            assert all(float(dtype.type(value)) == value for value in values)
            tangent = (scale * 1.0, scale * 2.0, scale * 4.0)
            for order in itertools.permutations(range(3)):
                for mask in itertools.product((False, True), repeat=3):
                    if not any(mask):
                        continue
                    rows.append(
                        (
                            tuple(values[index] for index in order),
                            tuple(mask[index] for index in order),
                            order,
                            tuple(tangent[index] for index in order),
                        )
                    )
    return rows


@pytest.mark.parametrize("reducer", list(_REDUCERS))
def test_hard_max_derivatives_hold_across_masks_orders_and_ties(
    *, reducer: str
) -> None:
    """Under jit and vmap, each row's JVP and gradient match the exact max rule."""
    rows = _mutation_cases()
    assert len(rows) == 672
    dtype = jnp.zeros(()).dtype
    values = jnp.asarray([row[0] for row in rows], dtype=dtype)
    feasible = jnp.asarray([row[1] for row in rows])
    ids = jnp.asarray([row[2] for row in rows], dtype=jnp.int32)
    tangents = jnp.asarray([row[3] for row in rows], dtype=dtype)
    expected = [
        _exact_maximum(
            values=row[0],
            feasible=row[1],
            tangent=row[3],
            action_ids=tuple(range(3)) if reducer == "argmax_and_max" else row[2],
        )
        for row in rows
    ]

    # keyword-only-exempt: library-callback=jax.vmap
    def one(
        row: FloatND, live: BoolND, row_ids: IntND, tangent: FloatND
    ) -> tuple[FloatND, IntND, FloatND, FloatND]:
        def value(x: FloatND) -> FloatND:
            return _REDUCERS[reducer](values=x, feasible=live, action_ids=row_ids)[0]

        best, best_id = _REDUCERS[reducer](
            values=row, feasible=live, action_ids=row_ids
        )
        _, jvp = jax.jvp(value, (row,), (tangent,))
        return best, best_id, jvp, jax.grad(value)(row)

    best, best_id, jvp, gradient = jax.jit(jax.vmap(one))(
        values, feasible, ids, tangents
    )

    # A three-way tie divides by three, which can round to either neighbour of
    # the exact quotient; every other JVP and every gradient weight is exact.
    exact_jvp = np.asarray([float(row[2]) for row in expected], dtype=dtype)
    n_winners = np.asarray([sum(w != 0 for w in row[3]) for row in expected])
    assert_array_equal(
        np.asarray(best), np.asarray([row[0] for row in expected], dtype=dtype)
    )
    assert_array_equal(np.asarray(best_id), [row[1] for row in expected])
    assert_array_equal(
        np.asarray(gradient),
        np.asarray([[float(w) for w in row[3]] for row in expected], dtype=dtype),
    )
    assert_array_equal(np.asarray(jvp)[n_winners < 3], exact_jvp[n_winners < 3])
    np.testing.assert_array_max_ulp(
        np.asarray(jvp)[n_winners == 3], exact_jvp[n_winners == 3], maxulp=1
    )


def test_hard_max_tangent_compares_the_values_it_reduced() -> None:
    """The tangent's tie test reads the same materialized values as the max.

    Were the values' producer evaluated once for the max and once for the
    equality test, the two copies could round apart and no element would equal
    the max, losing the derivative.
    """
    values = jnp.asarray([1.0, 3.0], dtype=jnp.zeros(()).dtype)

    def value(x: FloatND) -> FloatND:
        return _hard_max(
            values=jnp.exp(x),
            feasible=jnp.ones(x.shape, dtype=bool),
            action_ids=jnp.arange(x.shape[-1], dtype=jnp.int32),
        )[0]

    jaxpr = jax.make_jaxpr(lambda x, t: jax.jvp(value, (x,), (t,)))(values, values)

    assert "optimization_barrier" in str(jaxpr)
