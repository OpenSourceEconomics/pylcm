"""A target reachable with a representable probability may not be priced as impossible.

A regime transition can place a strictly positive probability on a target that
is too small for the dtype to hold as a normal number. The value survives in
memory, but XLA flushes it in both the comparison that decides whether the
target carries mass and the multiplication that would form its contribution —
so a target worth almost four units of a five-unit continuation is silently
priced at zero, and the remaining targets answer alone.

Refusing the probability is the only honest option: normalized against a
sibling of ordinary size, the weight really is that small, so no rescaling
recovers it. The refusal is arithmetic, so it holds with logging off; runtime
validation names it separately when it is on.

A probability of exactly zero is a different thing entirely — a target that
cannot be reached — and still contributes nothing.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.utils.logging import LogLevel
from _lcm.zero_safe import probability_or_nan
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
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
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
    """The largest representable probability the dtype cannot hold as a normal."""
    dtype = _active_dtype()
    tiny = np.asarray(np.finfo(dtype).tiny, dtype=dtype)
    return jnp.asarray(np.nextafter(tiny, np.asarray(0.0, dtype=dtype)), dtype=dtype)


def _smallest_normal() -> ScalarFloat:
    """The smallest probability the dtype holds exactly — one step above refusal."""
    dtype = _active_dtype()
    return jnp.asarray(np.finfo(dtype).tiny, dtype=dtype)


def _certain() -> ScalarFloat:
    return jnp.asarray(1.0, dtype=_active_dtype())


def _impossible() -> ScalarFloat:
    return jnp.asarray(0.0, dtype=_active_dtype())


def _no_utility() -> ScalarFloat:
    return jnp.asarray(0.0, dtype=_active_dtype())


def _common_payoff() -> ScalarFloat:
    return jnp.asarray(1.0, dtype=_active_dtype())


def _rare_payoff(shock: ScalarFloat) -> FloatND:
    """A payoff differing from the common target's, read at every shock node."""
    return jnp.asarray(5.0, dtype=_active_dtype()) + 0.0 * shock


def _model(
    rare_probability: Callable[[], ScalarFloat],
    *,
    certainty_equivalent: CertaintyEquivalent | None = None,
    rare_carries_a_process: bool = True,
) -> Model:
    """A source choosing between a common target and a rare, valuable one.

    The rare target carries a target-only IID process, so the witness runs
    through the feature under audit; `rare_carries_a_process=False` removes it
    to cover the route that has no stochastic node to multiply against.
    """
    rare_states = (
        {"shock": NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)}
        if rare_carries_a_process
        else {}
    )
    rare_utility = _rare_payoff if rare_carries_a_process else _common_payoff
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


def test_subnormal_regime_probability_refuses_the_continuation() -> None:
    """A target too rare for the dtype poisons the value rather than dropping out."""
    assert bool(jnp.all(jnp.isnan(_source_value(_model(_largest_subnormal)))))


def test_subnormal_regime_probability_refuses_a_target_without_stochastic_nodes() -> (
    None
):
    """The refusal does not depend on the target carrying a lottery to multiply."""
    model = _model(_largest_subnormal, rare_carries_a_process=False)
    assert bool(jnp.all(jnp.isnan(_source_value(model))))


def test_subnormal_regime_probability_refuses_under_a_nonlinear_aggregator() -> None:
    """Both continuation routes refuse, not only the linear one."""
    model = _model(_largest_subnormal, certainty_equivalent=PowerMean())
    params = {
        "source": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "certainty_equivalent": {"risk_aversion": 2.0},
        }
    }
    value = model.solve(params=params, log_level="off")[0]["source"]
    assert bool(jnp.all(jnp.isnan(jnp.asarray(value))))


def test_smallest_normal_regime_probability_is_priced_rather_than_refused() -> None:
    """One representable step above refusal, the rare target still contributes."""
    value = _source_value(_model(_smallest_normal))
    assert bool(jnp.all(jnp.isfinite(value)))


def test_zero_regime_probability_leaves_the_common_target_alone() -> None:
    """A target that cannot be reached contributes nothing, and nothing is poisoned."""
    np.testing.assert_allclose(np.asarray(_source_value(_model(_impossible))), 1.0)


def test_subnormal_regime_probability_is_named_by_validation() -> None:
    """Debug validation identifies the probability as subnormal."""
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError, match="subnormal"):
        _model(_largest_subnormal).solve(params=_PARAMS, log_level="debug")


def test_probability_or_nan_refuses_a_subnormal_and_keeps_everything_else() -> None:
    """Only a represented nonzero subnormal is refused; the other classes survive."""
    dtype = _active_dtype()
    tiny = np.asarray(np.finfo(dtype).tiny, dtype=dtype)
    values = jnp.asarray(
        np.array(
            [
                0.0,
                -0.0,
                np.nextafter(tiny, np.asarray(0.0, dtype=dtype)),
                -np.nextafter(tiny, np.asarray(0.0, dtype=dtype)),
                tiny,
                0.5,
                -0.5,
                np.nan,
            ],
            dtype=dtype,
        )
    )
    result = probability_or_nan(values)
    np.testing.assert_array_equal(
        np.asarray(jnp.isnan(result)),
        np.array([False, False, True, True, False, False, False, True]),
    )
    np.testing.assert_array_equal(
        np.asarray(result)[[0, 1, 4, 5, 6]], np.asarray(values)[[0, 1, 4, 5, 6]]
    )


@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_probability_or_nan_reads_bits_rather_than_arithmetic(
    *, compile_it: bool
) -> None:
    """A subnormal is invisible to comparison, so the refusal survives compilation."""
    subnormal = _largest_subnormal()
    assert bool(subnormal == 0.0), "the flush this test exists for is not happening"
    func = jax.jit(probability_or_nan) if compile_it else probability_or_nan
    assert bool(jnp.isnan(func(subnormal)))
