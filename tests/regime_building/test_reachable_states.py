"""Reachable discrete-state tuples follow from the declared laws and constraints.

Starting from declared start codes at the admissible initial nodes, the enumeration
pushes every tuple of the named states through their deterministic laws, at every
code of the other discrete variables those laws and the constraints read, keeping
only actions the evaluable constraints allow. The union of the per-node tuples is one
code set for all periods.
"""

import itertools
from collections.abc import Mapping

import jax.numpy as jnp
import pytest

from _lcm.regime_building.reachable_states import (
    ReachableStates,
    enumerate_reachable_states,
)
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import StateTransitionEntry
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    StateName,
)

_CAP = 2
_COUNTERS = ("count_a", "count_b", "count_c")


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


def _raise(*, count: DiscreteState, chosen: BoolND) -> DiscreteState:
    return jnp.where(chosen, jnp.minimum(count + 1, _CAP), count)


def _next_count_a(*, count_a: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return _raise(count=count_a, chosen=choice == _Choice.a)


def _next_count_b(*, count_b: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return _raise(count=count_b, chosen=choice == _Choice.b)


def _next_count_c(*, count_c: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return _raise(count=count_c, chosen=choice == _Choice.c)


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption + 1.0


def _c_needs_a(*, choice: DiscreteAction, count_a: DiscreteState) -> BoolND:
    return (choice != _Choice.c) | (count_a > 0)


def _b_needs_offer(*, choice: DiscreteAction, offer: DiscreteState) -> BoolND:
    return (choice != _Choice.b) | (offer == _Offer.yes)


def _budget(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _offer_probs(count_a: DiscreteState) -> FloatND:
    p = 0.2 + 0.3 * count_a
    return jnp.stack([1 - p, p])


def _utility(
    *, choice: DiscreteAction, count_a: DiscreteState, consumption: ContinuousAction
) -> FloatND:
    return jnp.log(consumption) + 0.1 * count_a - 0.05 * (choice != _Choice.rest)


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _retired_utility(
    *, count_a: DiscreteState, count_b: DiscreteState, count_c: DiscreteState
) -> FloatND:
    return 0.1 * (count_a + count_b + count_c)


def _alive_regime(
    *, state_transitions: Mapping[StateName, StateTransitionEntry] | None = None
) -> Regime:
    states = {name: DiscreteGrid(_Count) for name in _COUNTERS} | {
        "offer": DiscreteGrid(_Offer),
        "wealth": LinSpacedGrid(start=1, stop=4, n_points=4),
    }
    transitions = {
        "count_a": _next_count_a,
        "count_b": _next_count_b,
        "count_c": _next_count_c,
        "offer": StochasticTransition(func=_offer_probs),
        "wealth": _next_wealth,
    }
    return Regime(
        states=states,
        state_transitions=transitions | dict(state_transitions or {}),
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


def _model(*, alive: Regime | None = None, n_periods: int = 6) -> Model:
    """Counters move in `alive`, freeze in `retired`, and `dead` is terminal.

    `alive` covers periods `0 .. n_periods - 3`, `retired` the period after and
    `dead` the last.
    """
    retired = Regime(
        states={name: DiscreteGrid(_Count) for name in _COUNTERS},
        state_transitions={name: fixed_transition(name) for name in _COUNTERS},
        functions={"utility": _retired_utility},
    )
    return Model(
        regimes={
            "alive": alive or _alive_regime(),
            "retired": retired,
            "dead": Regime(functions={"utility": _zero_utility}),
        },
        edges={
            "alive": {"alive": range(n_periods - 3), "retired": n_periods - 3},
            "retired": {"dead": n_periods - 2},
        },
        ages=AgeGrid(start=0, inclusive_stop=n_periods - 1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        enable_jit=False,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _enumerate(
    *,
    model: Model,
    state_names: tuple[StateName, ...] = _COUNTERS,
    start_codes: Mapping[StateName, tuple[int, ...]] | None = None,
) -> ReachableStates:
    return enumerate_reachable_states(
        regimes=model._regimes,
        initial_nodes=model.graph.initial_nodes,
        ages=model._time,
        state_names=state_names,
        start_codes=dict.fromkeys(state_names, (0,))
        if start_codes is None
        else start_codes,
    )


def _oracle(n_periods: int) -> dict[tuple[int, str], frozenset[tuple[int, ...]]]:
    """Every counter tuple some admissible choice path reaches, per node."""
    expected: dict[tuple[int, str], set[tuple[int, ...]]] = {}
    for period in range(n_periods - 1):
        regime = "alive" if period <= n_periods - 3 else "retired"
        for choices in itertools.product(range(4), repeat=period):
            counts = [0, 0, 0]
            for choice in choices:
                if choice == _Choice.c and counts[0] == 0:
                    break
                if choice < len(counts):
                    counts[choice] = min(counts[choice] + 1, _CAP)
            else:
                expected.setdefault((period, regime), set()).add(tuple(counts))
    return {node: frozenset(tuples) for node, tuples in expected.items()}


def test_forward_sets_equal_itertools_oracle() -> None:
    """Every node's reachable tuples equal the brute-force set over choice paths."""
    reachable = _enumerate(model=_model(n_periods=6))
    assert dict(reachable.by_node) == _oracle(6)


def test_union_holds_every_reachable_tuple_once_in_sorted_order() -> None:
    """The code set is the sorted union of the per-node tuples."""
    reachable = _enumerate(model=_model(n_periods=6))
    expected = tuple(sorted(set().union(*_oracle(6).values())))
    assert reachable.union == expected


def test_every_code_as_start_code_gives_the_product_at_period_zero() -> None:
    """With every code as a start code, period 0 holds the whole product."""
    model = _model(n_periods=4)
    reachable = _enumerate(
        model=model, start_codes={name: tuple(range(_CAP + 1)) for name in _COUNTERS}
    )
    assert reachable.by_node[(0, "alive")] == frozenset(
        itertools.product(range(_CAP + 1), repeat=3)
    )


def test_constraint_reading_a_continuous_variable_is_skipped_and_reported() -> None:
    """The budget reads wealth and consumption, so it cannot prune tuples."""
    reachable = _enumerate(model=_model(n_periods=4))
    assert reachable.skipped_constraints == {
        ("alive", "budget"): ("consumption", "wealth")
    }


def test_non_member_discrete_state_is_enumerated_over_every_code() -> None:
    """The offer constraint never removes `b` because `offer=yes` is always tried."""
    reachable = _enumerate(model=_model(n_periods=4))
    assert (0, 1, 0) in reachable.by_node[(1, "alive")]


def _next_count_a_with_free_param(
    *, count_a: DiscreteState, choice: DiscreteAction, step: ScalarInt
) -> DiscreteState:
    return jnp.where(choice == _Choice.a, jnp.minimum(count_a + step, _CAP), count_a)


def test_member_law_reading_a_free_parameter_is_refused() -> None:
    """A law whose support depends on runtime params names itself in the error."""
    alive = _alive_regime(state_transitions={"count_a": _next_count_a_with_free_param})
    with pytest.raises(ModelInitializationError, match=r"next_count_a.*step"):
        _enumerate(model=_model(alive=alive, n_periods=4))


def _count_c_probs(count_c: DiscreteState) -> FloatND:
    return jnp.full(_CAP + 1, 1 / (_CAP + 1)) + 0.0 * count_c


def test_stochastic_member_law_is_refused() -> None:
    """A member must move deterministically; a lottery member is refused by name."""
    alive = _alive_regime(
        state_transitions={"count_c": StochasticTransition(func=_count_c_probs)}
    )
    with pytest.raises(ModelInitializationError, match=r"count_c.*stochastic"):
        _enumerate(model=_model(alive=alive, n_periods=4))


def test_continuous_member_is_refused() -> None:
    """Only discrete states can be enumerated."""
    with pytest.raises(ModelInitializationError, match=r"wealth.*discrete"):
        _enumerate(model=_model(n_periods=4), state_names=("count_a", "wealth"))
