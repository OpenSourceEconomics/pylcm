"""A foreign categorical action must not erase a process code binding.

Regime ``a`` has a fixed, node-drawing Normal process ``z`` whose exact nodes are
8, 9, 10, 11, 12. Regime ``b`` has a binary categorical action. Naming that
unrelated action ``z`` must leave regime ``a``'s code reader unchanged. Renaming
the action ``choice`` is a positive control. Both regimes are terminal, so the
independent answer is the literal node-position table below, with no expectation
operator, interpolation, coordinate inversion, or production adapter in it.
The cases also keep categorical-code readers in the unrelated regime, so a
globally process-named variable is not treated as a process in every regime.

The value assertion distinguishes [8, 9, 10, 11, 12] from [0, 1, 2, 3, 4]; a
separate utility checks the required int32 dtype before converting its output
to floating point. The no-code control preserves the ordinary physical-value
result. Eager/JIT and regime insertion-order variants exercise the same public
inputs and literal answers.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, DiscreteGrid, Model, NormalIIDProcess, Regime, categorical
from lcm.typing import (
    ActionName,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserFunction,
)

# Independent oracle: ordinal means position in this declared node table.
# Every entry is exactly representable in either supported floating dtype.
PROCESS_NODES = (8.0, 9.0, 10.0, 11.0, 12.0)
PROCESS_POSITIONS = (0.0, 1.0, 2.0, 3.0, 4.0)


@categorical(ordered=False)
class RegimeIds:
    a: ScalarInt
    b: ScalarInt


@categorical(ordered=False)
class BinaryChoice:
    off: ScalarInt
    on: ScalarInt


def _position_payoff(z_code: DiscreteState) -> FloatND:
    """Allow the value assertion to expose a physical-value/code mix-up."""
    return jnp.asarray(z_code, dtype=float)


def _checked_position_payoff(z_code: DiscreteState) -> FloatND:
    """Check dtype before casting, using a trace-safe static dtype condition."""
    if z_code.dtype != jnp.dtype("int32"):
        raise TypeError(f"process z_code must be int32, received {z_code.dtype}")
    return jnp.asarray(z_code, dtype=float)


def _physical_payoff(z: ContinuousState) -> FloatND:
    return z


def _categorical_z_payoff(z: DiscreteAction) -> FloatND:
    return jnp.asarray(z, dtype=float)


def _categorical_choice_payoff(choice: DiscreteAction) -> FloatND:
    return jnp.asarray(choice, dtype=float)


def _categorical_z_code_payoff(z_code: DiscreteAction) -> FloatND:
    return jnp.asarray(z_code, dtype=float)


def _categorical_choice_code_payoff(choice_code: DiscreteAction) -> FloatND:
    return jnp.asarray(choice_code, dtype=float)


def _model(
    *,
    reader: UserFunction,
    foreign_action_name: ActionName,
    reverse: bool,
    enable_jit: bool,
    foreign_code: bool = False,
) -> Model:
    process_regime = Regime(
        states={
            "z": NormalIIDProcess(
                mu=10.0,
                sigma=1.0,
                n_std=2.0,
                n_points=5,
                gauss_hermite=False,
                simulate_draws="nodes",
            )
        },
        functions={"utility": reader},
    )
    foreign_utility = {
        ("z", False): _categorical_z_payoff,
        ("choice", False): _categorical_choice_payoff,
        ("z", True): _categorical_z_code_payoff,
        ("choice", True): _categorical_choice_code_payoff,
    }[foreign_action_name, foreign_code]
    categorical_regime = Regime(
        states={},
        actions={foreign_action_name: DiscreteGrid(category_class=BinaryChoice)},
        functions={"utility": foreign_utility},
    )
    entries = [("a", process_regime), ("b", categorical_regime)]
    return Model(
        regimes=dict(reversed(entries) if reverse else entries),
        edges={},
        ages=AgeGrid(exact_values=(0,)),
        initial_nodes={0: ("a", "b")},
        regime_id_class=RegimeIds,
        enable_jit=enable_jit,
    )


def _solve_and_check(*, model: Model, expected: tuple[float, ...]) -> None:
    # z_code must remain engine-bound even in the colliding-name model.
    assert "z_code" not in str(model.get_params_template())
    solution = model.solve(params={"a": {}, "b": {}}, log_level="off")
    values = solution.values[0]
    np.testing.assert_array_equal(np.asarray(values["a"]), np.asarray(expected))
    # In the unrelated categorical regime, maximizing over off/on yields one.
    np.testing.assert_array_equal(np.asarray(values["b"]), np.asarray(1.0))


@pytest.mark.parametrize(
    "foreign_action_name", ["z", "choice"], ids=["collision", "renamed"]
)
@pytest.mark.parametrize("reverse", [False, True], ids=["a-first", "b-first"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_process_code_binding_ignores_foreign_action_kind(
    *, foreign_action_name: ActionName, reverse: bool, enable_jit: bool
):
    model = _model(
        reader=_position_payoff,
        foreign_action_name=foreign_action_name,
        reverse=reverse,
        enable_jit=enable_jit,
    )
    _solve_and_check(model=model, expected=PROCESS_POSITIONS)


@pytest.mark.parametrize(
    "foreign_action_name", ["z", "choice"], ids=["collision", "renamed"]
)
@pytest.mark.parametrize("reverse", [False, True], ids=["a-first", "b-first"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_process_code_dtype_ignores_foreign_action_kind(
    *, foreign_action_name: ActionName, reverse: bool, enable_jit: bool
):
    model = _model(
        reader=_checked_position_payoff,
        foreign_action_name=foreign_action_name,
        reverse=reverse,
        enable_jit=enable_jit,
    )
    _solve_and_check(model=model, expected=PROCESS_POSITIONS)


@pytest.mark.parametrize(
    "foreign_action_name", ["z", "choice"], ids=["collision", "renamed"]
)
@pytest.mark.parametrize("reverse", [False, True], ids=["a-first", "b-first"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_no_process_code_read_preserves_physical_values(
    *, foreign_action_name: ActionName, reverse: bool, enable_jit: bool
):
    model = _model(
        reader=_physical_payoff,
        foreign_action_name=foreign_action_name,
        reverse=reverse,
        enable_jit=enable_jit,
    )
    _solve_and_check(model=model, expected=PROCESS_NODES)


@pytest.mark.parametrize(
    "foreign_action_name", ["z", "choice"], ids=["collision", "renamed"]
)
@pytest.mark.parametrize("reverse", [False, True], ids=["a-first", "b-first"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_categorical_code_reader_keeps_its_local_kind(
    *, foreign_action_name: ActionName, reverse: bool, enable_jit: bool
):
    # Only the categorical regime reads a code. A repair that sends the union
    # of process names through _local_process for every regime would reject
    # b's valid categorical z_code reader in the colliding-name case.
    model = _model(
        reader=_physical_payoff,
        foreign_action_name=foreign_action_name,
        reverse=reverse,
        enable_jit=enable_jit,
        foreign_code=True,
    )
    _solve_and_check(model=model, expected=PROCESS_NODES)
