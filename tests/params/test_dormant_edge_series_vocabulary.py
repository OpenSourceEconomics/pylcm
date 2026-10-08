"""A Series an edge law indexes by a declared state keeps its labels.

At two periods the law out of `alive` is dormant and demand prunes the state
`x` it reads; at three periods the law runs and `x` stays. The `cutoffs` slot
exists at both horizons, so a labelled Series for it converts by `x`'s declared
categories either way, bound free or fixed.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    DeterministicTransition,
    DiscreteGrid,
    Model,
    Regime,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.typing import DiscreteState, Float1D, FloatND, ScalarInt


@categorical(ordered=False)
class LifeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class TypeId:
    low: ScalarInt
    high: ScalarInt


def utility() -> FloatND:
    return jnp.asarray(1.0)


def choose(*, x: DiscreteState, cutoffs: Float1D) -> ScalarInt:
    # One bare-name index, as required by the documented Series converter.
    return jnp.where(cutoffs[x] > 0.5, LifeId.alive, LifeId.dead)


def model(*, n_periods, fixed_params):
    """Leave `alive` by `choose` until the last source age, then for `dead`."""
    last_age = n_periods - 1
    last_source_age = last_age - 1
    targets = {"dead": AgeRange(start=0)}
    if last_source_age > 0:
        targets["alive"] = AgeRange(start=0, exclusive_stop=last_source_age)
    return Model(
        ages=AgeGrid(start=0, inclusive_stop=last_age, step="Y"),
        regimes={
            "alive": Regime(functions={"utility": utility}),
            "dead": Regime(functions={"utility": utility}),
        },
        regime_id_class=LifeId,
        initial_nodes={0: "alive"},
        states={"x": DiscreteGrid(category_class=TypeId)},
        state_transitions={"x": fixed_transition("x")},
        fixed_params=fixed_params,
        edges={
            "alive": Transition(
                targets=targets,
                law=ByAge.until(
                    stop_age_exclusive=last_age,
                    law=DeterministicTransition(func=choose),
                    then="dead",
                ),
            ),
        },
    )


def fingerprints(values):
    """Each solved array's shape, dtype and bytes, by period and regime."""
    return {
        (period, regime): (
            (array := np.asarray(value)).shape,
            array.dtype.str,
            array.tobytes(order="C"),
        )
        for period, by_regime in values.items()
        for regime, value in by_regime.items()
    }


@pytest.mark.parametrize("n_periods", [2, 3])
@pytest.mark.parametrize("binding", ["free", "fixed"])
@pytest.mark.parametrize("reverse_rows", [False, True])
def test_dormant_declared_state_still_supplies_its_series_axis(
    *, n_periods, binding, reverse_rows
):
    """A labelled Series solves to the same bytes as its canonical array."""
    series = pd.Series([0.0, 1.0], index=pd.Index(["low", "high"], name="x"))
    if reverse_rows:
        series = series.iloc[::-1]
    canonical_array = np.asarray([0.0, 1.0])
    results = {}
    for spelling, leaf in (("array", canonical_array), ("series", series)):
        edges = {"edges": {"alive": {"cutoffs": leaf}}}
        instance = model(
            n_periods=n_periods,
            fixed_params=edges if binding == "fixed" else {},
        )
        assert ("x" in instance.pruned_variables["alive"]) == (n_periods == 2)
        params = {"discount_factor": 0.95, **(edges if binding == "free" else {})}
        results[spelling] = instance.solve(params=params, log_level="debug").values
    assert fingerprints(results["series"]) == fingerprints(results["array"])


def solve_with_leaf(*, n_periods, binding, leaf):
    """Solve with `leaf` as the `cutoffs` slot, bound free or fixed."""
    edges = {"edges": {"alive": {"cutoffs": leaf}}}
    instance = model(
        n_periods=n_periods,
        fixed_params=edges if binding == "fixed" else {},
    )
    params = {"discount_factor": 0.95, **(edges if binding == "free" else {})}
    return instance.solve(params=params, log_level="debug").values


@pytest.mark.parametrize("n_periods", [2, 3])
@pytest.mark.parametrize("binding", ["free", "fixed"])
def test_series_value_mutation_is_observable_exactly_when_the_law_is_active(
    *, n_periods, binding
):
    """Swapping the Series' values changes the solution exactly when the law runs."""
    outputs = []
    for values in ([0.0, 1.0], [1.0, 0.0]):
        series = pd.Series(values, index=pd.Index(["low", "high"], name="x"))
        outputs.append(
            fingerprints(
                solve_with_leaf(n_periods=n_periods, binding=binding, leaf=series)
            )
        )
    assert (outputs[0] == outputs[1]) == (n_periods == 2)


@pytest.mark.parametrize("n_periods", [2, 3])
@pytest.mark.parametrize("binding", ["free", "fixed"])
def test_invalid_declared_category_still_raises_at_both_horizons(*, n_periods, binding):
    """A label `x` does not declare is refused, dormant law or not."""
    series = pd.Series([0.0, 1.0], index=pd.Index(["low", "missing"], name="x"))
    with pytest.raises(ValueError, match=r"Invalid labels for level 'x'.*missing"):
        solve_with_leaf(n_periods=n_periods, binding=binding, leaf=series)
