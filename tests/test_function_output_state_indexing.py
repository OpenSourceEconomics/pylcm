"""Tests for the regime-function-output / state-indexed-input name clash.

A regime function whose output is then re-indexed by a discrete state
inside another function is a silent footgun: pylcm broadcasts function
outputs to per-cell scalars before consumption, so the indexing produces
NaN at runtime instead of the intended scalar.

The validation layer must raise on construction with a clear message
pointing the user at the safe pattern (see `discount_factor` in
`aca_model.agent.preferences`: take the state as input, return a scalar).
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousAction, DiscreteState, FloatND


@categorical(ordered=False)
class PrefType:
    type_0: int
    type_1: int


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


def _per_type_scale(some_param: FloatND) -> FloatND:
    """Returns one scale per pref_type — depends only on a per-type Series param."""
    return jnp.abs(1.0 / (1.0 - some_param))


def _utility_with_state_indexed_function_output(
    consumption: ContinuousAction,
    pref_type: DiscreteState,
    per_type_scale: FloatND,
) -> FloatND:
    # The clash: per_type_scale is registered as a regime function output
    # (returns shape (n_pref_types,)), but here it is consumed and indexed
    # by the `pref_type` discrete state.
    return per_type_scale[pref_type] * jnp.log(consumption + 1.0)


def _next_regime(period: int) -> FloatND:
    return jnp.where(period >= 1, RegimeId.dead, RegimeId.alive)


def _make_clashing_model() -> Model:
    alive = Regime(
        functions={
            "utility": _utility_with_state_indexed_function_output,
            "per_type_scale": _per_type_scale,
        },
        states={"pref_type": DiscreteGrid(PrefType)},
        state_transitions={"pref_type": None},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=5)},
        transition=_next_regime,
        active=lambda age: age < 2,
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 2,
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


def test_function_output_indexed_by_state_raises():
    """A regime function output indexed by a discrete state inside another regime
    function must raise on construction (silent NaN bug otherwise)."""
    with pytest.raises(
        RegimeInitializationError,
        match=r"per_type_scale.*pref_type",
    ):
        _make_clashing_model()


def _utility_safe(
    consumption: ContinuousAction,
    pref_type: DiscreteState,  # noqa: ARG001
    per_type_scale: FloatND,
) -> FloatND:
    # Safe variant: per_type_scale is consumed as a scalar (no [pref_type] indexing).
    return per_type_scale * jnp.log(consumption + 1.0)


def _per_type_scale_safe(pref_type: DiscreteState, some_param: FloatND) -> FloatND:
    """Safe variant: takes pref_type, returns scalar (mirrors discount_factor)."""
    return jnp.abs(1.0 / (1.0 - some_param[pref_type]))


def test_safe_pattern_does_not_raise():
    """The safe pattern (function takes the state, returns a scalar) builds fine."""
    alive = Regime(
        functions={
            "utility": _utility_safe,
            "per_type_scale": _per_type_scale_safe,
        },
        states={"pref_type": DiscreteGrid(PrefType)},
        state_transitions={"pref_type": None},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=5)},
        transition=_next_regime,
        active=lambda age: age < 2,
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 2,
    )
    Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


def test_constraint_indexing_function_output_by_state_raises():
    """The check applies to regime constraints too, not only `functions`."""

    def _constraint_indexing_function_output(
        consumption: ContinuousAction,
        pref_type: DiscreteState,
        per_type_scale: FloatND,
    ) -> FloatND:
        return consumption <= per_type_scale[pref_type]

    with pytest.raises(
        RegimeInitializationError,
        match=r"per_type_scale.*pref_type",
    ):
        Regime(
            functions={
                "utility": lambda consumption, pref_type, per_type_scale: jnp.log(  # noqa: ARG005
                    consumption + 1.0
                ),
                "per_type_scale": _per_type_scale,
            },
            states={"pref_type": DiscreteGrid(PrefType)},
            state_transitions={"pref_type": None},
            actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=5)},
            constraints={"feasibility": _constraint_indexing_function_output},
            transition=_next_regime,
            active=lambda age: age < 2,
        )
