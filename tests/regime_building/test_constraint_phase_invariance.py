"""A constraint's feasible set must be phase-invariant through its whole ancestry.

`Regime.constraints` are contractually phase-invariant, and a *direct* `Phased`
constraint is rejected at regime init. But a bare constraint can reach a `Phased`
`next_<state>` or a `Phased` helper transitively; the solve and simulate
feasibility DAGs then resolve that dependency from different phase pools, so the
feasible set is phase-specific -- silently, exactly what the direct ban forbids.
Model build must reject it.

A carried state is deliberately NOT phase-varying: its imputation resolves to the
same `solve` variant in both phases, so a constraint reading its CURRENT value has
the same feasible set in both and must build. Reading `next_<carried>`, by contrast,
has no solve-phase producer and is rejected early. Both carried-state cases are
covered in `tests/test_carried_states.py`.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    Model,
    Phased,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.transition import AgeSpecializedFunction
from lcm.typing import DiscreteAction, DiscreteState, FloatND, Period, ScalarInt


@categorical(ordered=True)
class Move:
    stay: ScalarInt
    switch: ScalarInt


@categorical(ordered=True)
class Stock:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    live: ScalarInt
    last: ScalarInt


def _flat_utility(stock: FloatND, move: DiscreteAction) -> FloatND:
    return 0.0 * stock + 0.0 * move


def _next_stock_belief(move: DiscreteAction) -> FloatND:
    return jnp.where(move == Move.stay, Stock.good, Stock.bad)


def _next_stock_actual(move: DiscreteAction) -> FloatND:
    return jnp.where(move == Move.stay, Stock.bad, Stock.good)


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


def _last_regime() -> Regime:
    return Regime(
        transition=None,
        state_transitions={},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": _flat_utility},
    ).replace(active=lambda age: age >= 2)


def _model(*, live_functions, state_transitions, constraints) -> Model:
    live = Regime(
        transition=_next_regime,
        state_transitions=state_transitions,
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions=live_functions,
        constraints=constraints,
    ).replace(active=lambda age: age < 2)
    return Model(
        regimes={"live": live, "last": _last_regime()},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="constraint phase-invariance",
    )


def test_constraint_reading_a_phased_next_state_is_rejected():
    """A bare constraint whose `next_<state>` is `Phased` gets a phase-specific
    feasible set -- solve deems `stay` feasible, simulate deems `switch` feasible.
    Model build must reject it, naming the phase-varying dependency.
    """

    def capacity(next_stock: FloatND) -> FloatND:
        return next_stock == Stock.good

    with pytest.raises(ModelInitializationError, match="phase-varying"):
        _model(
            live_functions={"utility": _flat_utility},
            state_transitions={
                "stock": Phased(solve=_next_stock_belief, simulate=_next_stock_actual)
            },
            constraints={"capacity": capacity},
        )


def test_constraint_reading_a_phased_helper_is_rejected():
    """The same hazard through a `Phased` helper rather than the outer law."""

    def stay_ok_belief() -> FloatND:
        return jnp.array(True)  # noqa: FBT003

    def stay_ok_actual() -> FloatND:
        return jnp.array(False)  # noqa: FBT003

    def capacity(move: DiscreteAction, stay_ok: FloatND) -> FloatND:
        return jnp.where(move == Move.stay, stay_ok, jnp.array(True))  # noqa: FBT003

    with pytest.raises(ModelInitializationError, match="phase-varying"):
        _model(
            live_functions={
                "utility": _flat_utility,
                "stay_ok": Phased(solve=stay_ok_belief, simulate=stay_ok_actual),
            },
            state_transitions={"stock": _next_stock_actual},
            constraints={"capacity": capacity},
        )


def test_phase_invariant_constraint_still_builds():
    """A constraint reading only phase-invariant nodes must NOT be rejected: the
    guard is ancestry-specific, not a blanket ban whenever any `Phased` exists.
    """

    def capacity(move: DiscreteAction) -> FloatND:
        return jnp.where(move == Move.stay, jnp.array(True), jnp.array(True))  # noqa: FBT003

    # A `Phased` law is present, but the constraint does not reach it.
    model = _model(
        live_functions={"utility": _flat_utility},
        state_transitions={
            "stock": Phased(solve=_next_stock_belief, simulate=_next_stock_actual)
        },
        constraints={"capacity": capacity},
    )
    assert model is not None


def test_constraint_reaching_phased_helper_through_age_specialized_marker_is_rejected():
    """A phase-varying helper is caught behind an age-specialized one.

    The constraint `capacity` reads the age-specialized helper `policy_target`,
    whose per-age function reads the `Phased` helper `threshold` (`stay` when
    solving, `switch` when simulating), so the two feasible sets differ. An
    age-specialized function exposes only a generic `(*args, **kwargs)` signature
    until it is resolved, so the model is rejected only if the ancestry is walked
    on the resolved graph.
    """

    def threshold_solve() -> ScalarInt:
        return jnp.array(Move.stay)

    def threshold_simulate() -> ScalarInt:
        return jnp.array(Move.switch)

    def _policy_target_of_age(_age: float):
        def policy_target(threshold: ScalarInt) -> ScalarInt:
            return threshold

        return policy_target

    def capacity(move: DiscreteAction, policy_target: ScalarInt) -> FloatND:
        return move == policy_target

    with pytest.raises(ModelInitializationError, match="phase-varying"):
        _model(
            live_functions={
                "utility": _flat_utility,
                "threshold": Phased(solve=threshold_solve, simulate=threshold_simulate),
                "policy_target": AgeSpecializedFunction(
                    build=_policy_target_of_age, signature=lambda _age: 0
                ),
            },
            state_transitions={"stock": _next_stock_actual},
            constraints={"capacity": capacity},
        )


def test_per_target_phased_next_state_is_rejected():
    """A per-target law that varies by phase is rejected under its plain name.

    A per-target state law is keyed `next_stock__<target>`, while the constraint
    reads the unqualified `next_stock`. Declaring one belief law for every target
    and one truth law for every target keeps the target sets identical, but the
    feasible set still differs by phase, so the model must be rejected.
    """

    def capacity(next_stock: FloatND) -> FloatND:
        return next_stock == Stock.good

    with pytest.raises(ModelInitializationError, match="phase-varying"):
        _model(
            live_functions={"utility": _flat_utility},
            state_transitions={
                "stock": Phased(
                    solve={"live": _next_stock_belief, "last": _next_stock_belief},
                    simulate={"live": _next_stock_actual, "last": _next_stock_actual},
                )
            },
            constraints={"capacity": capacity},
        )


def test_constraint_reading_a_fixed_current_state_still_builds():
    """A constraint reading a state held fixed by `fixed_transition` still builds.

    `fixed_transition` rebuilds its identity law on every collection, so the two
    phases hold distinct objects standing for the same phase-invariant law. That
    must not be mistaken for phase variance.

    The constraint reads the current `stock`, not `next_stock`, because a `next_`
    name is reserved for a transition's output (see
    `test_next_prefix_is_reserved.py`). For a fixed state the two values are equal,
    so the question this pins down is unchanged.
    """

    def utility(stock: DiscreteState, move: DiscreteAction) -> FloatND:
        return 0.0 * stock + 0.0 * move

    def capacity(stock: DiscreteState) -> FloatND:
        return stock == Stock.good

    model = _model(
        live_functions={"utility": utility},
        state_transitions={"stock": fixed_transition("stock")},
        constraints={"capacity": capacity},
    )
    assert model is not None
