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

import math

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LogNormalIIDProcess,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.exceptions import (
    InvalidNameError,
    InvalidParamsError,
    ModelInitializationError,
)
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


def _entered_process_model(fixed_params: dict, *, enable_jit: bool = False) -> Model:
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
        enable_jit=enable_jit,
    )


_SPELLINGS = {
    "state_level": {"target": {"shock": _LAW}},
    "regime_level": {"target": dict(_LAW)},
    "model_level": dict(_LAW),
    "jax_scalar_leaves": {
        "target": {"shock": {name: jnp.asarray(v) for name, v in _LAW.items()}}
    },
    "numpy_scalar_leaves": {
        "target": {"shock": {name: np.float64(v) for name, v in _LAW.items()}}
    },
    "numpy_zero_d_leaves": {
        "target": {"shock": {name: np.array(v) for name, v in _LAW.items()}}
    },
    "python_int_leaves": {"target": {"shock": {**_LAW, "n_std": 2}}},
}


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("spelling", sorted(_SPELLINGS))
def test_every_level_pins_an_entered_process_law(
    spelling: str,
    enable_jit: bool,  # noqa: FBT001
) -> None:
    """A law written at any accepted level fixes the process at construction.

    With the target's payoff equal to the shock and no discounting, entering at
    the process's own law is worth `mu`. Each spelling must build *and* reach
    that value — building alone would not show the law was actually bound.
    """
    solution = _entered_process_model(
        _SPELLINGS[spelling], enable_jit=enable_jit
    ).solve(params=_SOLVE_PARAMS, log_level="debug")

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


def test_an_unknown_fixed_param_is_rejected_as_it_always_was() -> None:
    """Widening where a law may be written does not widen what may be written.

    A key matching no template slot and no process field is still unknown, and
    binding a process must not turn the resolver's unknown-key check off for
    everything else in the same call.
    """
    with pytest.raises(InvalidParamsError, match=r"(?i)unknown"):
        _entered_process_model({**_LAW, "not_a_parameter": 1.0})


def test_a_non_scalar_law_names_the_parameter_it_cannot_pin() -> None:
    """A process's law is one number per field, so an array has no field to be.

    The rejection names the qualified parameter rather than surfacing later as a
    generic entry failure, because the two have different fixes.
    """
    with pytest.raises(InvalidParamsError, match="target__shock__mu"):
        _entered_process_model({"target": {"shock": {**_LAW, "mu": jnp.zeros(3)}}})


def test_a_lognormal_law_pins_from_a_broadcast_too() -> None:
    """The grammar belongs to `fixed_params`, not to one process class.

    A log-normal entry is priced at the mean of the exponentiated nodes, which
    differs from `exp(mu)`, so the value also shows the bound law reached the
    nodes rather than merely satisfying the build.
    """
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=lambda age: age < 22,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states={"shock": LogNormalIIDProcess(n_points=3, gauss_hermite=True)},
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        fixed_params={"mu": 0.0, "sigma": 1.0},
        enable_jit=False,
    )

    solution = model.solve(params=_SOLVE_PARAMS, log_level="debug")

    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(3)
    nodes = math.sqrt(2.0) * raw_nodes
    weights = raw_weights / math.sqrt(math.pi)
    expected = float(np.dot(np.exp(nodes), weights))
    np.testing.assert_allclose(np.asarray(solution[0]["source"]), expected, atol=1e-6)


def test_a_coarse_regime_transition_pins_the_same_law() -> None:
    """The binder does not depend on how the source names its target.

    A bare callable makes every regime reachable, where a per-target dict names
    one; the law is a property of the target's process either way.
    """

    def _always_target() -> ScalarInt:
        return RegimeId.target

    model = Model(
        regimes={
            "source": Regime(
                transition=_always_target,
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
        fixed_params=dict(_LAW),
        enable_jit=False,
    )

    solution = model.solve(params=_SOLVE_PARAMS, log_level="debug")

    np.testing.assert_allclose(np.asarray(solution[0]["source"]), _MU, atol=1e-6)


def test_one_resolved_process_object_reaches_every_consumer() -> None:
    """The model runs the bound process, not a half-specified twin.

    Every downstream stage — handoff validation, the target's own nodes,
    diagnostics, simulation — reads `user_regimes`, so pinning that the process
    there is fully specified pins that they all see the same resolved object.
    """
    model = _entered_process_model({"target": {"shock": _LAW}})

    process = model.user_regimes["target"].states["shock"]

    assert isinstance(process, NormalIIDProcess)
    assert process.params_to_pass_at_runtime == ()
    assert process.mu == _MU
    np.testing.assert_allclose(
        np.asarray(process.to_jax()), np.array([0.0, _MU, 2.0]), atol=1e-6
    )


def test_a_law_left_to_runtime_still_cannot_be_entered() -> None:
    """Widening where a law may be written does not make an unwritten law legal.

    Nothing pins this process, so its law genuinely arrives at runtime and the
    source has no value to read. The message names the state and the parameters
    that block it.
    """
    with pytest.raises(ModelInitializationError) as excinfo:
        _entered_process_model({})

    message = str(excinfo.value)
    assert "stochastic process 'shock'" in message
    assert "at runtime" in message
    for param_name in _LAW:
        assert f"'{param_name}'" in message
