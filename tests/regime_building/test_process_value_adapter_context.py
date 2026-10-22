"""Sharing a callable must preserve the reader's complete local binding.

Public-API regression, run at both --precision=32 and --precision=64.

Two terminal regimes use the same name z. Regime a holds the physical nodes
(8, 9, 10, 11, 12) of a Normal process; regime b chooses values (10, 20) from a
DiscreteGrid. The shared/distinct fixtures both declare categorical action k.
The plain and local-code controls omit that unused action. The shared utility reads
plain z and k_code. Reading k_code requires an adapter even where plain z
requires no decoding, so a cache keyed only by process-position conversions
conflates two different local bindings.

The independent oracle is the literal terminal payoff -(z - 10)**2: a's values
are (-4, -1, 0, -1, -4); b chooses physical value 10 and has value zero. If b's
codes (0, 1) are passed as values, it instead chooses code 1 and has value -81.
These numbers are exactly representable in both supported precisions. The
test does not catch arbitrary exceptions or reproduce coordinate inversion.

There are 16 cases: shared/distinct/plain readers x two regime orders x
eager/JIT (12), plus local process/discrete-code controls x order x eager/JIT (4).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, DiscreteGrid, Model, NormalIIDProcess, Regime, categorical
from lcm.typing import DiscreteState, FloatND, ScalarInt, ValueND


@categorical(ordered=False)
class RegimeIds:
    a: ScalarInt
    b: ScalarInt


@categorical(ordered=False)
class BinaryChoice:
    off: ScalarInt
    on: ScalarInt


def _shared_payoff(*, z: ValueND, k_code: DiscreteState) -> FloatND:
    # ValueND is deliberately dtype-polymorphic, so an incorrect integer input
    # reaches the numeric assertion rather than relying on a type-check error.
    return -((z - 10.0) ** 2) + 0.0 * k_code


def _distinct_payoff(*, z: ValueND, k_code: DiscreteState) -> FloatND:
    return -((z - 10.0) ** 2) + 0.0 * k_code


def _plain_payoff(z: ValueND) -> FloatND:
    return -((z - 10.0) ** 2)


def _process_code_payoff(z_code: DiscreteState) -> FloatND:
    if z_code.dtype != jnp.dtype("int32"):
        raise TypeError(f"process ordinal must be int32, received {z_code.dtype}")
    return jnp.asarray(z_code, dtype=float)


def _discrete_code_payoff(z_code: DiscreteState) -> FloatND:
    if z_code.dtype != jnp.dtype("int32"):
        raise TypeError(f"discrete code must be int32, received {z_code.dtype}")
    return jnp.asarray(z_code, dtype=float)


def _model(*, reader_mode: str, reverse: bool, enable_jit: bool) -> Model:
    readers = {
        "shared": (_shared_payoff, _shared_payoff),
        "distinct": (_shared_payoff, _distinct_payoff),
        "plain": (_plain_payoff, _plain_payoff),
        "local_codes": (_process_code_payoff, _discrete_code_payoff),
    }
    read_a, read_b = readers[reader_mode]
    needs_k = reader_mode in {"shared", "distinct"}
    a_actions = {"k": DiscreteGrid(category_class=BinaryChoice)} if needs_k else {}
    b_actions = {"z": DiscreteGrid(values=jnp.array([10.0, 20.0]))}
    if needs_k:
        b_actions["k"] = DiscreteGrid(category_class=BinaryChoice)
    a = Regime(
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
        actions=a_actions,
        functions={"utility": read_a},
    )
    b = Regime(
        actions=b_actions,
        functions={"utility": read_b},
    )
    entries = [("a", a), ("b", b)]
    return Model(
        regimes=dict(reversed(entries) if reverse else entries),
        edges={},
        ages=AgeGrid(exact_values=(0,)),
        initial_nodes={0: ("a", "b")},
        regime_id_class=RegimeIds,
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("reader_mode", ["shared", "distinct", "plain"])
@pytest.mark.parametrize("reverse", [False, True], ids=["process-first", "value-first"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_complete_read_context_survives_callable_reuse(
    *, reader_mode: str, reverse: bool, enable_jit: bool
):
    model = _model(reader_mode=reader_mode, reverse=reverse, enable_jit=enable_jit)
    template = str(model.get_params_template())
    assert "k_code" not in template
    assert "z_code" not in template
    values = model.solve(params={"a": {}, "b": {}}, log_level="off").values[0]
    np.testing.assert_array_equal(
        np.asarray(values["a"]), np.asarray((-4.0, -1.0, 0.0, -1.0, -4.0))
    )
    np.testing.assert_array_equal(np.asarray(values["b"]), np.asarray(0.0))


@pytest.mark.parametrize("reverse", [False, True], ids=["process-first", "value-first"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_local_process_positions_and_discrete_codes_remain_distinct(
    *, reverse: bool, enable_jit: bool
):
    model = _model(reader_mode="local_codes", reverse=reverse, enable_jit=enable_jit)
    assert "z_code" not in str(model.get_params_template())
    values = model.solve(params={"a": {}, "b": {}}, log_level="off").values[0]
    np.testing.assert_array_equal(
        np.asarray(values["a"]), np.asarray((0.0, 1.0, 2.0, 3.0, 4.0))
    )
    np.testing.assert_array_equal(np.asarray(values["b"]), np.asarray(1.0))
