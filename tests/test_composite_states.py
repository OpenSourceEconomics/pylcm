"""A composite state solves and simulates the model on reachable member tuples only.

Declaring deterministic discrete member states as one composite state replaces their
axes by a single generated state whose codes index the tuples reachable from the
declared start codes. User functions keep reading the members. Two variants exist:

- `"closed"`: the code set is closed under every period's laws, so no tuple in it
  steps outside it under an allowed action;
- `"masked"`: the code set holds the forward-reachable tuples only, and a generated
  constraint makes every tuple infeasible at the periods it cannot be reached in.

Both reproduce the dense model's values at every reachable cell and its simulated
panel exactly.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    CompositeState,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    categorical,
    fixed_transition,
)
from lcm.composite_state import CompositeVariant
from lcm.exceptions import InvalidInitialConditionsError, ModelInitializationError
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    RegimeName,
    ScalarInt,
    StateName,
)
from tests.conftest import assert_agrees_to_ulp

_CAP = 2
_MEMBERS = ("count_a", "count_b", "count_c")
_N_PERIODS = 6
_PARAMS = {"discount_factor": 0.95}


@categorical(ordered=True)
class _Count:
    zero: ScalarInt
    one: ScalarInt
    two: ScalarInt


@categorical(ordered=False)
class _Choice:
    a: ScalarInt
    b: ScalarInt
    c: ScalarInt
    rest: ScalarInt


@categorical(ordered=False)
class _Offer:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _next_count_a(*, count_a: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return jnp.where(choice == _Choice.a, jnp.minimum(count_a + 1, _CAP), count_a)


def _next_count_b(
    *, count_b: DiscreteState, choice: DiscreteAction, cap: ScalarInt
) -> DiscreteState:
    return jnp.where(choice == _Choice.b, jnp.minimum(count_b + 1, cap), count_b)


def _next_count_c(*, count_c: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return jnp.where(choice == _Choice.c, jnp.minimum(count_c + 1, _CAP), count_c)


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption + 1.0


def _offer_probs(count_a: DiscreteState) -> FloatND:
    p = 0.2 + 0.3 * count_a
    return jnp.stack([1 - p, p])


def _c_needs_a(*, choice: DiscreteAction, count_a: DiscreteState) -> BoolND:
    return (choice != _Choice.c) | (count_a > 0)


def _b_needs_offer(*, choice: DiscreteAction, offer: DiscreteState) -> BoolND:
    return (choice != _Choice.b) | (offer == _Offer.yes)


def _budget(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _utility(
    *,
    choice: DiscreteAction,
    count_a: DiscreteState,
    count_b: DiscreteState,
    count_c: DiscreteState,
    consumption: ContinuousAction,
) -> FloatND:
    skill = 0.3 * count_a + 0.2 * count_b + 0.5 * count_c * count_a
    return jnp.log(consumption) + skill - 0.05 * (choice != _Choice.rest)


def _retired_utility(
    *, count_a: DiscreteState, count_b: DiscreteState, count_c: DiscreteState
) -> FloatND:
    return 0.4 * count_a + 0.3 * count_b + 0.2 * count_c


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _model(
    *, composite_states: Mapping[StateName, CompositeState] | None = None
) -> Model:
    """Counters move in `alive`, freeze in `retired`, and `dead` is terminal."""
    alive = Regime(
        states={name: DiscreteGrid(_Count) for name in _MEMBERS}
        | {
            "offer": DiscreteGrid(_Offer),
            "wealth": LinSpacedGrid(start=1, stop=4, n_points=4),
        },
        state_transitions={
            "count_a": _next_count_a,
            "count_b": _next_count_b,
            "count_c": _next_count_c,
            "offer": StochasticTransition(func=_offer_probs),
            "wealth": _next_wealth,
        },
        actions={
            "choice": DiscreteGrid(_Choice),
            "consumption": LinSpacedGrid(start=0.5, stop=3, n_points=4),
        },
        functions={"utility": _utility},
        constraints={
            "c_needs_a": _c_needs_a,
            "b_needs_offer": _b_needs_offer,
            "budget": _budget,
        },
    )
    retired = Regime(
        states={name: DiscreteGrid(_Count) for name in _MEMBERS},
        state_transitions={name: fixed_transition(name) for name in _MEMBERS},
        functions={"utility": _retired_utility},
    )
    return Model(
        regimes={
            "alive": alive,
            "retired": retired,
            "dead": Regime(functions={"utility": _zero_utility}),
        },
        edges={
            "alive": {"alive": range(_N_PERIODS - 3), "retired": _N_PERIODS - 3},
            "retired": {"dead": _N_PERIODS - 2},
        },
        ages=AgeGrid(start=0, inclusive_stop=_N_PERIODS - 1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        fixed_params={"alive": {"next_count_b": {"cap": _CAP}}},
        execution_config=ExecutionConfig(device_memory_bytes=None),
        composite_states=composite_states or {},
    )


def _composite(variant: CompositeVariant) -> Model:
    return _model(
        composite_states={
            "history": CompositeState(
                members=_MEMBERS,
                start_codes=dict.fromkeys(_MEMBERS, (0,)),
                variant=variant,
            )
        }
    )


def _member_first(
    *,
    model: Model,
    regime: RegimeName,
    values: np.ndarray,
    names: tuple[StateName, ...],
) -> np.ndarray:
    """Move the named state axes of a value array to the front."""
    order = list(model._regimes[regime].solution.state_names)
    return np.moveaxis(
        values, [order.index(name) for name in names], list(range(len(names)))
    )


@pytest.mark.parametrize("variant", ["closed", "masked"])
def test_values_agree_with_dense_model_at_every_reachable_cell(
    variant: CompositeVariant,
) -> None:
    """Composite V equals dense V wherever a member tuple is reached."""
    dense, composite = _model(), _composite(variant)
    lowering = composite._composite_states["history"]
    dense_values = dense.solve(params=_PARAMS, log_level="off").values
    composite_values = composite.solve(params=_PARAMS, log_level="off").values
    gathered_dense, gathered_composite = [], []
    for (period, regime), tuples in lowering.reachable.by_node.items():
        full = _member_first(
            model=dense,
            regime=regime,
            values=np.asarray(dense_values[period][regime]),
            names=_MEMBERS,
        )
        reduced = _member_first(
            model=composite,
            regime=regime,
            values=np.asarray(composite_values[period][regime]),
            names=("history",),
        )
        for members in sorted(tuples):
            gathered_dense.append(np.ravel(full[members]))
            gathered_composite.append(np.ravel(reduced[lowering.encode(members)]))
    assert_agrees_to_ulp(
        got=np.concatenate(gathered_composite),
        expected=np.concatenate(gathered_dense),
        n_ulp=4,
    )


def _initial_conditions() -> dict[str, jnp.ndarray]:
    n = 40
    rng = np.random.default_rng(seed=3)
    return {
        "regime_id": jnp.full(n, _RegimeId.alive, dtype=jnp.int32),
        "age": jnp.zeros(n),
        "wealth": jnp.asarray(rng.uniform(1, 4, n)),
        "offer": jnp.asarray(rng.integers(0, 2, n), dtype=jnp.int32),
    } | {name: jnp.zeros(n, dtype=jnp.int32) for name in _MEMBERS}


def _panel(*, model: Model, composite: bool) -> pd.DataFrame:
    frame = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        log_level="off",
        seed=11,
    ).to_dataframe(
        use_labels=False, additional_targets=list(_MEMBERS) if composite else None
    )
    columns = ["subject_id", "period", "regime_name", "choice", "consumption"]
    out = frame[[*columns, "wealth", "offer", *_MEMBERS]].copy()
    out[list(_MEMBERS)] = out[list(_MEMBERS)].astype(float)
    return out.reset_index(drop=True)


@pytest.mark.parametrize("variant", ["closed", "masked"])
def test_simulated_panel_equals_dense_panel(variant: CompositeVariant) -> None:
    """Choices, states and decoded members match the dense model row for row."""
    pd.testing.assert_frame_equal(
        _panel(model=_composite(variant), composite=True),
        _panel(model=_model(), composite=False),
        check_dtype=False,
    )


def test_masked_code_set_is_the_forward_union_and_closed_set_is_larger() -> None:
    """The masked variant stores fewer codes than the closed one on this model."""
    sizes = tuple(
        len(_composite(variant)._composite_states["history"].table)
        for variant in ("masked", "closed")
    )
    forward = len(
        set().union(
            *_composite("masked")
            ._composite_states["history"]
            .reachable.by_node.values()
        )
    )
    assert (sizes[0], sizes[0] < sizes[1]) == (forward, True)


@pytest.mark.parametrize("variant", ["closed", "masked"])
def test_value_array_has_one_entry_per_code_and_other_state_cell(
    variant: CompositeVariant,
) -> None:
    """The composite axis replaces the member product in the stored value array."""
    composite = _composite(variant)
    values = composite.solve(params=_PARAMS, log_level="off").values
    n_codes = len(composite._composite_states["history"].table)
    assert np.asarray(values[0]["alive"]).shape == (n_codes, 2, 4)


def test_masked_variant_is_infeasible_where_a_code_is_unreachable() -> None:
    """At period 0 only the start tuple is reachable, so every other code reads -inf."""
    composite = _composite("masked")
    lowering = composite._composite_states["history"]
    start = lowering.encode((0, 0, 0))
    v0 = np.asarray(composite.solve(params=_PARAMS, log_level="off").values[0]["alive"])
    other = np.delete(v0, start, axis=0)
    assert (np.isneginf(other).all(), np.isfinite(v0[start]).all()) == (True, True)


def test_initial_member_tuple_outside_the_code_set_is_refused() -> None:
    """A start the declaration never reaches has no composite code."""
    conditions = _initial_conditions() | {
        "count_c": jnp.full(40, _CAP, dtype=jnp.int32)
    }
    with pytest.raises(InvalidInitialConditionsError, match="count_c"):
        _composite("closed").simulate(
            params=_PARAMS, initial_conditions=conditions, log_level="off", seed=1
        )


def test_labelled_initial_frame_names_members_like_the_dense_model() -> None:
    """A DataFrame of member labels starts subjects at the same composite codes."""
    conditions = _initial_conditions()
    frame = pd.DataFrame(
        {
            "regime_name": "alive",
            "age": np.asarray(conditions["age"]),
            "wealth": np.asarray(conditions["wealth"]),
            "offer": np.where(np.asarray(conditions["offer"]) == 1, "yes", "no"),
        }
        | dict.fromkeys(_MEMBERS, "zero")
    )

    def run(model: Model) -> pd.DataFrame:
        return model.simulate(
            params=_PARAMS, initial_conditions=frame, log_level="off", seed=11
        ).to_dataframe(use_labels=False)[["subject_id", "period", "choice"]]

    pd.testing.assert_frame_equal(run(_composite("closed")), run(_model()))


def test_unknown_member_label_in_initial_frame_is_named() -> None:
    """A label that is no category of its member is reported as that label."""
    frame = pd.DataFrame(
        {
            "regime_name": ["alive"],
            "age": [0.0],
            "wealth": [2.0],
            "offer": ["no"],
            "count_a": ["zero"],
            "count_b": ["zero"],
            "count_c": ["bogus"],
        }
    )
    with pytest.raises(InvalidInitialConditionsError, match="bogus"):
        _composite("closed").simulate(
            params=_PARAMS, initial_conditions=frame, log_level="off", seed=1
        )


def test_composite_name_taken_by_an_action_is_refused() -> None:
    """The generated state may not reuse the name of a declared action."""
    with pytest.raises(ModelInitializationError, match="choice"):
        _model(
            composite_states={
                "choice": CompositeState(
                    members=_MEMBERS, start_codes=dict.fromkeys(_MEMBERS, (0,))
                )
            }
        )
