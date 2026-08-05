"""One resolution rule decides where a process law may be written.

`Model(fixed_params=...)` accepts a parameter at any of the levels the params
template accepts it: exactly, under its regime, or at model level, with the most
specific declaration winning and a value at two levels an error. A process law is
an ordinary fixed parameter, so it obeys that rule too — the value that pins a
process must not have to be spelled differently from every other fixed value.

This matters beyond convenience. Entering a target's process requires its law to
be *fixed at construction*, and the check reads the process the model actually
built. A spelling the resolver accepts but the binder ignores would therefore
reject a model whose law is fully specified.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidNameError, ModelInitializationError
from lcm.typing import ScalarFloat, ScalarInt

# `mu=1, sigma=0.5, n_std=2` at three points puts equidistant nodes on
# `(0, 1, 2)`, so entering at the process's own law is worth `mu`.
_MU = 1.0
_LAW = {"mu": _MU, "sigma": 0.5, "n_std": 2.0}
_SOLVE_PARAMS = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}}


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


def _entered_process_model(fixed_params: dict) -> Model:
    """Build a source entering a target process whose law it does not carry."""
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=lambda age: age < 22,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states={"shock": NormalIIDProcess(n_points=3, gauss_hermite=False)},
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        fixed_params=fixed_params,
        enable_jit=False,
    )


_SPELLINGS = {
    "state_level": {"target": {"shock": _LAW}},
    "regime_level": {"target": dict(_LAW)},
    "model_level": dict(_LAW),
    "jax_scalar_leaves": {
        "target": {"shock": {name: jnp.asarray(v) for name, v in _LAW.items()}}
    },
}


@pytest.mark.parametrize("spelling", sorted(_SPELLINGS))
def test_every_level_pins_an_entered_process_law(spelling: str) -> None:
    """A law written at any accepted level fixes the process at construction.

    With the target's payoff equal to the shock and no discounting, entering at
    the process's own law is worth `mu`. Each spelling must build *and* reach
    that value — building alone would not show the law was actually bound.
    """
    solution = _entered_process_model(_SPELLINGS[spelling]).solve(
        params=_SOLVE_PARAMS, log_level="debug"
    )

    np.testing.assert_allclose(np.asarray(solution[0]["source"]), _MU, atol=1e-6)


@pytest.mark.parametrize("spelling", sorted(_SPELLINGS))
def test_a_bound_law_leaves_the_params_template(spelling: str) -> None:
    """However it is written, a bound law stops being a runtime parameter."""
    template = _entered_process_model(_SPELLINGS[spelling]).get_params_template()

    assert "shock" not in template["target"]


def test_one_law_written_at_two_levels_is_an_error() -> None:
    """Naming the same parameter twice is ambiguous, not silently resolved.

    The params template rejects a value given at two levels; a process law is an
    ordinary fixed parameter and gets the same answer.
    """
    with pytest.raises(InvalidNameError, match="Ambiguous"):
        _entered_process_model({"mu": _MU, "target": {"shock": _LAW}})


def test_a_broadcast_that_binds_a_law_still_reaches_a_function() -> None:
    """A model-level value serves every slot that asks for it, process or not.

    `mu` here pins the target's process *and* is an argument of the source's
    utility. Consuming it for the process alone would leave the function without
    its parameter; keeping it out of the process would leave the entry law
    unfixed. The source's payoff is `mu`, and the target contributes `mu` too, so
    the solved value is `2 * mu` exactly when both readings hold.
    """

    def _mu_utility(mu: ScalarFloat) -> ScalarFloat:
        return mu

    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=lambda age: age < 22,
                functions={"utility": _mu_utility},
            ),
            "target": Regime(
                transition=None,
                states={"shock": NormalIIDProcess(n_points=3, gauss_hermite=False)},
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        fixed_params=dict(_LAW),
        enable_jit=False,
    )

    solution = model.solve(params=_SOLVE_PARAMS, log_level="debug")

    np.testing.assert_allclose(np.asarray(solution[0]["source"]), 2 * _MU, atol=1e-6)


def test_a_law_left_to_runtime_still_cannot_be_entered() -> None:
    """Widening where a law may be written does not make an unwritten law legal.

    Nothing pins this process, so its law genuinely arrives at runtime and the
    source has no value to read. The message names the state and the parameters
    that block it.
    """
    with pytest.raises(
        ModelInitializationError, match="passes 'mu', 'sigma', 'n_std' at runtime"
    ):
        _entered_process_model({})
