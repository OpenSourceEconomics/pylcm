"""An edge parameter's numeric role comes from the laws the model demands.

A `ByAge` case that no demanded age selects keeps its parameter slots, but its
annotations do not decide how the demanded case's inputs are converted.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, InitialNodes, Model, Regime, categorical
from lcm.exceptions import ModelInitializationError
from lcm.transition import (
    ByAge,
    DeterministicTransition,
    StochasticTransition,
    Transition,
)
from lcm.typing import FloatND, ScalarFloat, ScalarInt, UserParams


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    sink: ScalarInt


@categorical(ordered=False)
class ExitRegimeId:
    source: ScalarInt
    sink: ScalarInt
    dead: ScalarInt


def _utility() -> FloatND:
    return jnp.asarray(0.0)


def _live_probability(*, hazard: ScalarFloat) -> FloatND:
    # Trace-time metadata: the callback must receive a float whatever was passed.
    assert hazard.dtype.kind == "f"
    return hazard


def _dormant_probability(*, hazard: ScalarInt) -> FloatND:  # noqa: ARG001
    msg = "A law that no demanded age selects was evaluated."
    raise AssertionError(msg)


def _model(*, include_dormant: bool) -> Model:
    cases = {0: {"sink": StochasticTransition(func=_live_probability)}}
    if include_dormant:
        cases[1] = {"sink": StochasticTransition(func=_dormant_probability)}
    return Model(
        regimes={
            name: Regime(functions={"utility": _utility}) for name in ("source", "sink")
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes=InitialNodes(by_age={0: "source"}),
        edges={"source": Transition(law=ByAge(cases=cases))},
        durable_identity=False,
    )


def _params(hazard: float | np.ndarray) -> UserParams:
    return {
        "discount_factor": 1.0,
        "edges": {"source": {"sink": {"hazard": hazard}}},
    }


@pytest.mark.parametrize("include_dormant", [False, True])
@pytest.mark.parametrize("hazard", [1, 1.0, np.asarray(1, dtype=np.int32)])
def test_demanded_float_law_receives_a_float(
    *, include_dormant: bool, hazard: float | np.ndarray
) -> None:
    _model(include_dormant=include_dormant).solve(
        params=_params(hazard), log_level="off"
    )


def _first_cell(*, hazard: ScalarFloat) -> FloatND:
    assert hazard.dtype.kind == "f"
    return hazard / 2.0


def _second_cell(*, hazard: ScalarFloat) -> FloatND:
    assert hazard.dtype.kind == "f"
    return hazard


def _int_cell(*, hazard: ScalarInt) -> FloatND:
    return jnp.asarray(hazard, dtype=float)


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _first_vector(*, hazard: ScalarFloat) -> FloatND:
    assert hazard.dtype.kind == "f"
    return jnp.stack([0.0 * hazard, 1.0 - hazard / 2.0, hazard / 2.0])


def _second_vector(*, hazard: ScalarFloat) -> FloatND:
    assert hazard.dtype.kind == "f"
    return jnp.stack([0.0 * hazard, 1.0 - hazard, hazard])


def _int_vector(*, hazard: ScalarInt) -> FloatND:
    return jnp.stack([0.0 * hazard, 1.0 - hazard, 1.0 * hazard])


def _first_selector(*, hazard: ScalarFloat) -> ScalarInt:
    assert hazard.dtype.kind == "f"
    return jnp.where(hazard > 2.0, ExitRegimeId.dead, ExitRegimeId.sink)


def _second_selector(*, hazard: ScalarFloat) -> ScalarInt:
    assert hazard.dtype.kind == "f"
    return jnp.where(hazard > 3.0, ExitRegimeId.dead, ExitRegimeId.sink)


def _int_selector(*, hazard: ScalarInt) -> ScalarInt:
    return jnp.where(hazard > 2, ExitRegimeId.dead, ExitRegimeId.sink)


_LAW_FORMS = {
    # Per-target cells that differ by age lower to period-masked cells and
    # their sum.
    "cells": (
        {
            "source": StochasticTransition(func=_half),
            "sink": StochasticTransition(func=_first_cell),
        },
        {"sink": StochasticTransition(func=_second_cell)},
        {"sink": StochasticTransition(func=_int_cell)},
    ),
    # Laws over all targets lower to one period-dispatched law.
    "vector": tuple(
        StochasticTransition(func=func)
        for func in (_first_vector, _second_vector, _int_vector)
    ),
    "selector": tuple(
        DeterministicTransition(func=func)
        for func in (_first_selector, _second_selector, _int_selector)
    ),
}
_SLOT_UNDER_SINK = {"cells": True, "vector": False, "selector": False}


def _two_law_model(
    *,
    form: str,
    enable_jit: bool,
    second_reads_integer: bool = False,
    fixed_params: UserParams | None = None,
) -> Model:
    """Ages 0 and 1 are demanded and read `hazard` as a float; age 2 never is."""
    first, second, integer = _LAW_FORMS[form]
    cases = {0: first, 1: integer if second_reads_integer else second, 2: integer}
    # A law over all targets cannot return to `source` before the last age, so
    # those forms enter `source` at both demanded ages and leave it for good.
    names_targets = _SLOT_UNDER_SINK[form]
    regime_id_class = RegimeId if names_targets else ExitRegimeId
    return Model(
        regimes={
            name: Regime(functions={"utility": _utility})
            for name in regime_id_class.__annotations__
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=regime_id_class,
        initial_nodes=InitialNodes(
            by_age={0: "source"} if names_targets else {0: "source", 1: "source"}
        ),
        edges={
            "source": Transition(
                targets=None if names_targets else {"sink": (0, 1), "dead": (0, 1)},
                law=ByAge(cases=cases),
            )
        },
        fixed_params={} if fixed_params is None else fixed_params,
        enable_jit=enable_jit,
        durable_identity=False,
    )


def _two_law_params(*, form: str, hazard: float | np.ndarray) -> UserParams:
    slot = {"hazard": hazard}
    return {
        "discount_factor": 1.0,
        "edges": {"source": {"sink": slot} if _SLOT_UNDER_SINK[form] else slot},
    }


@pytest.mark.parametrize("form", list(_LAW_FORMS))
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("hazard", [1, 1.0, np.asarray(1, dtype=np.int32)])
def test_two_demanded_float_laws_receive_a_float_after_lowering(
    *, form: str, fixed: bool, enable_jit: bool, hazard: float | np.ndarray
) -> None:
    """Two demanded laws on one slot are lowered into one; the role survives."""
    params = _two_law_params(form=form, hazard=hazard)
    model = _two_law_model(
        form=form, enable_jit=enable_jit, fixed_params=params if fixed else None
    )
    values = model.solve(params={} if fixed else params, log_level="off").values
    # Utility is zero everywhere and every probability row sums to one.
    for by_regime in values.values():
        for value in by_regime.values():
            np.testing.assert_array_equal(np.asarray(value), 0.0)


@pytest.mark.parametrize("form", list(_LAW_FORMS))
def test_two_demanded_laws_reading_one_slot_differently_are_refused(
    *, form: str
) -> None:
    with pytest.raises(ModelInitializationError, match="annotated"):
        _two_law_model(form=form, enable_jit=False, second_reads_integer=True)
