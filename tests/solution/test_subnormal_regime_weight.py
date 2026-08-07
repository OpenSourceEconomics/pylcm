"""A weight too small for the dtype drops, unless dropping would change the answer.

A regime transition may place a strictly positive probability on a target that
the active precision cannot hold as a normal number. What the arithmetic then
does with it belongs to the backend: XLA:CPU flushes it, CUDA represents it.

Where the target's continuation is finite, dropping such a node is exact for
every purpose. The error is at most `tiny * |V|`, and reaching even the loosest
declared tolerance would take a value function above `1e32` in single precision
and above `1e295` in double. Refusing there would fail a model whose answer is
correct to its last bit.

Where the continuation is infinite, dropping is unbounded. `-inf` is the
ordinary value of a state at which no action is feasible, and a state reachable
with any strictly positive probability makes the expectation `-inf` however
small that probability is. The node keeps its infinity instead.

A weight of exactly zero is a different thing entirely — a target that cannot be
reached — and contributes nothing whatever value stands at it.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.utils.logging import LogLevel
from _lcm.zero_safe import zero_safe_weighted_term
from lcm import (
    AgeGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    categorical,
)
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.typing import FloatND, ScalarFloat, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    common: ScalarInt
    rare: ScalarInt


def _active_dtype() -> np.dtype:
    """The precision the suite is running at."""
    return np.dtype(jnp.zeros(()).dtype)


def _largest_subnormal() -> ScalarFloat:
    """The largest probability the dtype cannot hold as a normal number."""
    dtype = _active_dtype()
    tiny = np.asarray(np.finfo(dtype).tiny, dtype=dtype)
    return jnp.asarray(np.nextafter(tiny, np.asarray(0.0, dtype=dtype)), dtype=dtype)


def _smallest_normal() -> ScalarFloat:
    """The smallest probability the dtype holds as a normal number."""
    return jnp.asarray(np.finfo(_active_dtype()).tiny, dtype=_active_dtype())


def _certain() -> ScalarFloat:
    return jnp.asarray(1.0, dtype=_active_dtype())


def _impossible() -> ScalarFloat:
    return jnp.asarray(0.0, dtype=_active_dtype())


def _no_utility() -> ScalarFloat:
    return jnp.asarray(0.0, dtype=_active_dtype())


def _common_payoff() -> ScalarFloat:
    return jnp.asarray(1.0, dtype=_active_dtype())


def _finite_rare_payoff(shock: ScalarFloat) -> FloatND:
    """A payoff differing from the common target's, read at every shock node."""
    return jnp.asarray(5.0, dtype=_active_dtype()) + 0.0 * shock


def _infeasible_rare_payoff(shock: ScalarFloat) -> FloatND:
    """The ordinary value of a state at which no action is feasible."""
    return jnp.asarray(-jnp.inf, dtype=_active_dtype()) + 0.0 * shock


def _model(
    rare_probability: Callable[[], ScalarFloat],
    *,
    rare_payoff: Callable[..., FloatND] = _finite_rare_payoff,
    certainty_equivalent: CertaintyEquivalent | None = None,
    rare_carries_a_process: bool = True,
) -> Model:
    """A source choosing between a common target and a rare one.

    The rare target carries a target-only IID process, so the witness runs
    through the feature under audit; `rare_carries_a_process=False` removes it
    to cover the route with no stochastic node to multiply against.
    """
    rare_states = (
        {"shock": NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)}
        if rare_carries_a_process
        else {}
    )
    rare_utility = rare_payoff if rare_carries_a_process else _common_payoff
    return Model(
        regimes={
            "source": Regime(
                transition={
                    "common": MarkovTransition(_certain),
                    "rare": MarkovTransition(rare_probability),
                },
                active=lambda age: age < 21,
                functions={"utility": _no_utility},
                certainty_equivalent=certainty_equivalent,
            ),
            "common": Regime(transition=None, functions={"utility": _common_payoff}),
            "rare": Regime(
                transition=None,
                states=rare_states,
                functions={"utility": rare_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=_RegimeId,
    )


_PARAMS = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}}


def _source_value(model: Model, log_level: LogLevel = "off") -> FloatND:
    return jnp.asarray(model.solve(params=_PARAMS, log_level=log_level)[0]["source"])


def test_a_subnormal_weight_on_a_finite_value_drops() -> None:
    """The rare target's contribution is below the last bit, so the answer stands."""
    np.testing.assert_allclose(
        np.asarray(_source_value(_model(_largest_subnormal))), 1.0
    )


def test_a_subnormal_weight_drops_for_a_target_without_stochastic_nodes() -> None:
    """The rule does not depend on the target carrying a lottery to multiply."""
    model = _model(_largest_subnormal, rare_carries_a_process=False)
    np.testing.assert_allclose(np.asarray(_source_value(model)), 1.0)


def test_a_subnormal_weight_drops_under_a_nonlinear_aggregator() -> None:
    """Both continuation routes drop, not only the linear one."""
    model = _model(_largest_subnormal, certainty_equivalent=PowerMean())
    params = {
        "source": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "certainty_equivalent": {"risk_aversion": 2.0},
        }
    }
    value = jnp.asarray(model.solve(params=params, log_level="off")[0]["source"])

    np.testing.assert_allclose(np.asarray(value), 1.0)


def test_a_subnormal_weight_on_an_infinite_value_keeps_the_infinity() -> None:
    """A reachable state at which no action is feasible is worth `-inf`."""
    model = _model(_largest_subnormal, rare_payoff=_infeasible_rare_payoff)

    assert bool(jnp.all(jnp.isneginf(_source_value(model))))


def test_a_normal_weight_on_an_infinite_value_keeps_the_infinity() -> None:
    """The infinite branch is a rule about weights, not about small weights."""
    model = _model(_smallest_normal, rare_payoff=_infeasible_rare_payoff)

    assert bool(jnp.all(jnp.isneginf(_source_value(model))))


def test_a_zero_weight_on_an_infinite_value_contributes_nothing() -> None:
    """A target that cannot be reached is not made infinite by what stands at it."""
    model = _model(_impossible, rare_payoff=_infeasible_rare_payoff)

    np.testing.assert_allclose(np.asarray(_source_value(model)), 1.0)


def test_a_smallest_normal_weight_is_priced() -> None:
    """One representable step above the subnormal range, nothing is special."""
    assert bool(jnp.all(jnp.isfinite(_source_value(_model(_smallest_normal)))))


def test_a_zero_weight_leaves_the_common_target_alone() -> None:
    """A target that cannot be reached contributes nothing, and poisons nothing."""
    np.testing.assert_allclose(np.asarray(_source_value(_model(_impossible))), 1.0)


@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_weighted_term_classifies_by_bits_not_by_comparison(
    *, compile_it: bool
) -> None:
    """Zero-ness is read from the bits, so the verdict does not depend on the backend.

    A subnormal weight compares equal to zero where the backend flushes it and
    does not where it represents it. Against an infinite value the two would
    otherwise disagree: one drops the node, the other keeps the infinity.
    """
    dtype = _active_dtype()
    subnormal = _largest_subnormal()
    func = jax.jit(zero_safe_weighted_term) if compile_it else zero_safe_weighted_term
    negative_infinity = jnp.asarray(-jnp.inf, dtype=dtype)

    assert bool(jnp.isneginf(func(subnormal, negative_infinity)))
    assert float(func(jnp.asarray(0.0, dtype=dtype), negative_infinity)) == 0.0
    np.testing.assert_allclose(
        float(func(subnormal, jnp.asarray(5.0, dtype=dtype))), 0.0, atol=1e-30
    )
