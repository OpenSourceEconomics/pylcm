"""Tests for the regime-function-output / discrete-grid-indexed-input name clash.

The unsafe pattern is: a regime function `f` takes a discrete grid `g` (state,
action, or derived categorical) as an input — so `f`'s output is a per-cell
scalar — and a consumer then indexes `f[g]`. The consumer is indexing a 0-d
array by a scalar integer, which raises `IndexError` at trace time. The
validator catches the pattern at construction so the user gets a clear message
instead of a cryptic JAX trace error during solve.

The fix is to drop the redundant `[g]` in the consumer (or refactor `f` not
to take `g`).
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousAction, DiscreteAction, DiscreteState, FloatND


@categorical(ordered=False)
class PrefType:
    type_0: int
    type_1: int


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


def _per_type_scale_takes_pref_type(
    pref_type: DiscreteState, some_param: FloatND
) -> FloatND:
    """Takes pref_type — output is per-cell scalar; consumer must not re-index."""
    return jnp.abs(1.0 / (1.0 - some_param[pref_type]))


def _utility_redundantly_indexes(
    consumption: ContinuousAction,
    pref_type: DiscreteState,
    per_type_scale: FloatND,
) -> FloatND:
    # The clash: per_type_scale's producer takes pref_type, so its output is a
    # per-cell scalar. Indexing that scalar by pref_type again raises IndexError
    # at trace time.
    return per_type_scale[pref_type] * jnp.log(consumption + 1.0)


def _next_regime(period: int) -> FloatND:
    return jnp.where(period >= 1, RegimeId.dead, RegimeId.alive)


def _make_clashing_model() -> Model:
    alive = Regime(
        functions={
            "utility": _utility_redundantly_indexes,
            "per_type_scale": _per_type_scale_takes_pref_type,
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
    """A regime function output redundantly indexed by a discrete state inside
    another regime function must raise on construction."""
    with pytest.raises(
        RegimeInitializationError,
        match=r"per_type_scale.*pref_type",
    ):
        _make_clashing_model()


def _utility_consumes_scalar(
    consumption: ContinuousAction,
    pref_type: DiscreteState,  # noqa: ARG001
    per_type_scale: FloatND,
) -> FloatND:
    # Safe variant: per_type_scale is consumed as a scalar (no [pref_type] indexing).
    return per_type_scale * jnp.log(consumption + 1.0)


def test_safe_pattern_does_not_raise():
    """The safe pattern (function takes the state, returns a scalar; consumer
    uses it directly) builds fine."""
    alive = Regime(
        functions={
            "utility": _utility_consumes_scalar,
            "per_type_scale": _per_type_scale_takes_pref_type,
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


def _per_type_scale_array_output(some_param: FloatND) -> FloatND:
    """Does NOT take pref_type — output is `(n_pref_types,)`-shaped.

    A consumer indexing this output by `pref_type` is correct: the indexing
    selects the per-type entry. The validator must NOT flag this case.
    """
    return jnp.abs(1.0 / (1.0 - some_param))


def test_array_valued_producer_indexed_by_state_does_not_raise():
    """When the producing function does NOT take the discrete grid as input,
    its output stays array-shaped and `func_output[grid]` is correct code."""
    alive = Regime(
        functions={
            "utility": _utility_redundantly_indexes,
            "per_type_scale": _per_type_scale_array_output,
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


def test_function_output_indexed_by_derived_categorical_raises():
    """The check applies to derived categoricals (function outputs treated as
    categoricals via `derived_categoricals`), not only states."""

    @categorical(ordered=False)
    class IsMarried:
        single: int
        married: int

    def _is_married(spousal_income: DiscreteState) -> DiscreteState:
        return jnp.int32(spousal_income > 0)

    def _per_marital_scale(is_married: DiscreteState, some_param: FloatND) -> FloatND:
        return jnp.abs(1.0 / (1.0 - some_param[is_married]))

    def _utility_clash(
        consumption: ContinuousAction,
        is_married: DiscreteState,
        per_marital_scale: FloatND,
    ) -> FloatND:
        return per_marital_scale[is_married] * jnp.log(consumption + 1.0)

    @categorical(ordered=True)
    class SpousalIncome:
        single: int
        married_no_inc: int
        married_has_inc: int

    with pytest.raises(
        RegimeInitializationError,
        match=r"per_marital_scale.*is_married",
    ):
        Regime(
            functions={
                "utility": _utility_clash,
                "per_marital_scale": _per_marital_scale,
                "is_married": _is_married,
            },
            states={"spousal_income": DiscreteGrid(SpousalIncome)},
            state_transitions={"spousal_income": None},
            actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=5)},
            derived_categoricals={"is_married": DiscreteGrid(IsMarried)},
            transition=_next_regime,
            active=lambda age: age < 2,
        )


def test_function_output_indexed_by_discrete_action_raises():
    """The check applies to discrete actions, not only states/derived categoricals."""

    @categorical(ordered=False)
    class WorkChoice:
        no_work: int
        work: int

    def _per_choice_scale(labor_supply: DiscreteAction, some_param: FloatND) -> FloatND:
        return jnp.abs(1.0 / (1.0 - some_param[labor_supply]))

    def _utility_clash(
        consumption: ContinuousAction,
        labor_supply: DiscreteAction,
        per_choice_scale: FloatND,
    ) -> FloatND:
        return per_choice_scale[labor_supply] * jnp.log(consumption + 1.0)

    with pytest.raises(
        RegimeInitializationError,
        match=r"per_choice_scale.*labor_supply",
    ):
        Regime(
            functions={
                "utility": _utility_clash,
                "per_choice_scale": _per_choice_scale,
            },
            states={"pref_type": DiscreteGrid(PrefType)},
            state_transitions={"pref_type": None},
            actions={
                "consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=5),
                "labor_supply": DiscreteGrid(WorkChoice),
            },
            transition=_next_regime,
            active=lambda age: age < 2,
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
                "per_type_scale": _per_type_scale_takes_pref_type,
            },
            states={"pref_type": DiscreteGrid(PrefType)},
            state_transitions={"pref_type": None},
            actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=5)},
            constraints={"feasibility": _constraint_indexing_function_output},
            transition=_next_regime,
            active=lambda age: age < 2,
        )
