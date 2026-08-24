"""Role-minimal continuation graphs for phase-invariant simulate Q."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import pytest

from _lcm.regime_building import processing
from lcm import AgeGrid, DiscreteGrid, Model, Phased, Regime, categorical
from lcm.transition import AgeSpecializedFunction
from lcm.typing import DiscreteAction, FloatND, Period, ScalarInt


@categorical(ordered=True)
class Move:
    stay: ScalarInt
    switch: ScalarInt


@categorical(ordered=True)
class Good:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    live: ScalarInt
    last: ScalarInt


def utility(good: DiscreteAction, move: DiscreteAction) -> FloatND:
    return 1.0 * good + 0.0 * move


def next_good(move: DiscreteAction) -> ScalarInt:
    return jnp.where(move == Move.stay, Good.good, Good.bad)


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


def _phase_invariant_model(*, functions: Mapping[str, Any] | None = None) -> Model:
    live = Regime(
        transition=_next_regime,
        state_transitions={"good": next_good},
        states={"good": DiscreteGrid(Good)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": utility} if functions is None else functions,
    ).replace(active=lambda age: age < 2)
    last = Regime(
        transition=None,
        state_transitions={},
        states={"good": DiscreteGrid(Good)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": utility},
    ).replace(active=lambda age: age >= 2)
    return Model(
        regimes={"live": live, "last": last},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="phase-invariant continuation-pool provenance witness",
    )


def _capture_dual_role_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Record actual Q builds that carry an explicit continuation role pool."""
    original = processing.get_Q_and_F
    observed: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

    def recording_get_Q_and_F(*args: Any, **kwargs: Any):
        functions = kwargs["functions"]
        continuation = kwargs["continuation_functions"]
        if continuation is not None:
            observed.append((functions, continuation))
        return original(*args, **kwargs)

    monkeypatch.setattr(processing, "get_Q_and_F", recording_get_Q_and_F)
    return observed


def _same_callable_provenance(
    role_pair: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> bool:
    functions, continuation = role_pair
    return functions.keys() == continuation.keys() and all(
        functions[name] is continuation[name] for name in functions
    )


def test_phase_invariant_simulation_q_reuses_one_role_graph(
    monkeypatch: pytest.MonkeyPatch,
):
    """Equivalent declarations must not enter Q as a second full role graph.

    The oracle watches the actual role inputs to `get_Q_and_F`. It independently
    establishes equivalence by matching every key and callable identity in the
    resolved pools; it does not consult production's sharing decision.
    """
    dual_roles = _capture_dual_role_graphs(monkeypatch)

    _phase_invariant_model()

    equivalent_dual_roles = [
        pair for pair in dual_roles if _same_callable_provenance(pair)
    ]
    assert not equivalent_dual_roles, (
        "phase-invariant simulate Q retained a second provenance-identical "
        "continuation graph"
    )


def test_phase_invariant_age_specialization_reuses_one_role_graph(
    monkeypatch: pytest.MonkeyPatch,
):
    """Equivalent periodized wrappers collapse by explicit signatures/provenance."""

    def build(age: float):
        def utility_at_age(good: DiscreteAction, move: DiscreteAction) -> FloatND:
            return 1.0 * good + 0.0 * move + 0.0 * age

        return utility_at_age

    dual_roles = _capture_dual_role_graphs(monkeypatch)
    _phase_invariant_model(
        functions={
            "utility": AgeSpecializedFunction(build=build, signature=lambda age: age)
        }
    )

    equivalent_dual_roles = [
        pair for pair in dual_roles if _same_callable_provenance(pair)
    ]
    assert not equivalent_dual_roles


def test_phased_helper_retains_a_distinct_solve_role_graph(
    monkeypatch: pytest.MonkeyPatch,
):
    """A genuine helper wedge must keep solve continuation separate."""

    def utility_with_bonus(
        bonus: FloatND, good: DiscreteAction, move: DiscreteAction
    ) -> FloatND:
        return bonus + 1.0 * good + 0.0 * move

    def solve_bonus() -> FloatND:
        return jnp.array(1.0)

    def simulate_bonus() -> FloatND:
        return jnp.array(-1.0)

    dual_roles = _capture_dual_role_graphs(monkeypatch)
    _phase_invariant_model(
        functions={
            "utility": utility_with_bonus,
            "bonus": Phased(solve=solve_bonus, simulate=simulate_bonus),
        }
    )

    assert dual_roles, "a genuine Phased helper must retain the solve role pool"
    assert any(
        functions["bonus"] is simulate_bonus and continuation["bonus"] is solve_bonus
        for functions, continuation in dual_roles
    )
