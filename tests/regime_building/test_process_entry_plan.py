"""Process entry is decided over a target's states, never its actions.

Entering a process means placing a next-period value on the target's support, so
only a state can be entered: an action is chosen inside the target's own period
and has no axis in that regime's value function. A process declared as an action
therefore gets no entry machinery at all — it is an ordinary action grid.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.typing import ScalarFloat, ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _zero_utility() -> ScalarFloat:
    return jnp.float32(0)


def _shock_utility(shock: ScalarFloat) -> ScalarFloat:
    return shock


def _one_probability() -> ScalarFloat:
    return jnp.float32(1)


def _model_with_process_action(*, enable_jit: bool) -> Model:
    """A target whose `shock` is an action drawn from a process's node set."""
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=lambda age: age < 22,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                actions={
                    "shock": NormalIIDProcess(
                        n_points=3, gauss_hermite=True, mu=0.5, sigma=1.0
                    )
                },
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_a_process_declared_as_an_action_is_not_entered(*, enable_jit: bool) -> None:
    """A target's process-valued action solves as an action, with no entry law.

    The source hands nothing over and declares no entry law, so the only
    consistent reading is that `shock` is chosen in the target: the source's
    value is the target's best `shock`, which is that process's largest node.
    """
    model = _model_with_process_action(enable_jit=enable_jit)

    V = model.solve(
        params={"source": {"koopmans_aggregator": {"discount_factor": 1.0}}},
        log_level="debug",
    )

    largest_node = float(
        NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.5, sigma=1.0)
        .to_jax()
        .max()
    )
    assert float(jnp.ravel(V[0]["source"])[0]) == pytest.approx(largest_node, abs=1e-6)


def test_a_process_action_contributes_no_transition_to_the_source() -> None:
    """No `next_<action>` transition is synthesized toward a process action."""
    model = _model_with_process_action(enable_jit=False)

    solve_transitions = model._regimes["source"].solution.transitions

    assert "next_shock" not in dict(solve_transitions).get("target", ())
