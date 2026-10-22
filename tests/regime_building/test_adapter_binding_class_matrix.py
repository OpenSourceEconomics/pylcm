"""MT3: bounded declaration/binding population for R1's cache invariant.

Native-ready, not executed by the reviewer. Three translated node tables,
foreign value-grid states/actions, shared/distinct equivalent callables,
both declaration orders, and eager/JIT give 48 cases. The exact physical
payoffs are literal terminal answers; no production adapter is an oracle.

The categorical k_code argument intentionally requires adaptation even when
the process-valued z argument needs no conversion. State variants test every
represented z; action variants test the finite maximizing decision. RT8
separately retains plain-reader and local-int32-code controls. Neither suite
claims a global omitted-candidate bound or an observed native mutant kill.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, DiscreteGrid, Model, NormalIIDProcess, Regime
from lcm.typing import DiscreteState, FloatND, ValueND
from tests.regime_building.test_process_value_adapter_context import (
    BinaryChoice,
    RegimeIds,
)


def _make_payoff(center: float):
    def payoff(*, z: ValueND, k_code: DiscreteState) -> FloatND:
        return -((z - center) ** 2) + 0.0 * k_code

    return payoff


@pytest.mark.parametrize("center", [3.0, 10.0, 17.0])
@pytest.mark.parametrize("foreign_kind", ["state", "action"])
@pytest.mark.parametrize("shared", [False, True], ids=["distinct", "shared"])
@pytest.mark.parametrize("reverse", [False, True], ids=["process-first", "value-first"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_complete_binding_is_invariant_under_callable_reuse(
    *, center: float, foreign_kind: str, shared: bool, reverse: bool, enable_jit: bool
):
    process_reader = _make_payoff(center)
    foreign_reader = process_reader if shared else _make_payoff(center)
    a = Regime(
        states={
            "z": NormalIIDProcess(
                mu=center,
                sigma=1.0,
                n_std=2.0,
                n_points=5,
                gauss_hermite=False,
                simulate_draws="nodes",
            )
        },
        actions={"k": DiscreteGrid(category_class=BinaryChoice)},
        functions={"utility": process_reader},
    )
    value_grid = DiscreteGrid(values=jnp.array([center, center + 10.0]))
    foreign_states = {"z": value_grid} if foreign_kind == "state" else {}
    foreign_actions = {"k": DiscreteGrid(category_class=BinaryChoice)}
    if foreign_kind == "action":
        foreign_actions["z"] = value_grid
    b = Regime(
        states=foreign_states,
        actions=foreign_actions,
        functions={"utility": foreign_reader},
    )
    entries = [("a", a), ("b", b)]
    model = Model(
        regimes=dict(reversed(entries) if reverse else entries),
        edges={},
        ages=AgeGrid(exact_values=(0,)),
        initial_nodes={0: ("a", "b")},
        regime_id_class=RegimeIds,
        enable_jit=enable_jit,
    )
    assert "k_code" not in str(model.get_params_template())
    assert "z_code" not in str(model.get_params_template())
    values = model.solve(params={"a": {}, "b": {}}, log_level="off").values[0]
    np.testing.assert_array_equal(
        np.asarray(values["a"]), np.asarray((-4.0, -1.0, 0.0, -1.0, -4.0))
    )
    expected_foreign = (0.0, -100.0) if foreign_kind == "state" else 0.0
    np.testing.assert_array_equal(np.asarray(values["b"]), np.asarray(expected_foreign))
