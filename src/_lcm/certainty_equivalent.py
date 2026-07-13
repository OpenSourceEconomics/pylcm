"""Certainty-equivalent classes and engine helpers.

The public `lcm.certainty_equivalent` module re-exports the three classes
(`CertaintyEquivalent`, `QuasiArithmeticMean`, `PowerMean`) for user code.
Engine modules may import directly from here.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.utils.functools import get_union_of_args
from lcm.exceptions import RegimeInitializationError
from lcm.typing import FloatND

# Reserved argument name through which transform callables receive values.
CE_VALUE_ARG = "value"


class CertaintyEquivalent(ABC):
    """Base class for certainty-equivalent specifications.

    Declared on a non-terminal `Regime` via `certainty_equivalent=...`. The
    engine dispatches on the concrete subclass; `QuasiArithmeticMean` is
    the shipped implementation. When the field is `None` (the default), the
    continuation is aggregated as the linear expectation `E[V']`.
    `GridSearch` and `NBEGM` support a nonlinear certainty equivalent.
    """

    @property
    @abstractmethod
    def param_names(self) -> frozenset[str]:
        """Names of the certainty equivalent's runtime parameters."""


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class QuasiArithmeticMean(CertaintyEquivalent):
    """Certainty equivalent `CE = g⁻¹(Σ_r p_r · E_w[g(V'_r)])`.

    A quasi-arithmetic (Kolmogorov) mean: `transform` (`g`) is applied
    elementwise to next-period values before every expectation - over
    stochastic state transitions and over regime transitions - and
    `inverse` (`g⁻¹`) once, after the regime-probability-weighted sum. Both
    callables take the value array as the reserved first argument `value`;
    every further signature argument becomes a runtime parameter under the
    pseudo-function name `certainty_equivalent` in the regime's params
    (`{"certainty_equivalent": {"<arg>": ...}}`).

    Combined with a user-supplied Bellman aggregator `H` this expresses
    Epstein-Zin and other transformed-expectation recursive preferences.
    The parameters are read from the params template only, not from DAG
    function outputs.
    """

    transform: Callable[..., FloatND]
    """`g` — applied elementwise to next-period values before every expectation."""

    inverse: Callable[..., FloatND]
    """`g⁻¹` — applied once, after the regime-probability-weighted sum."""

    def __post_init__(self) -> None:
        for name in ("transform", "inverse"):
            func = getattr(self, name)
            if CE_VALUE_ARG not in get_union_of_args([func]):
                msg = (
                    f"The `{name}` callable of a `QuasiArithmeticMean` must "
                    f"take the value array via an argument named "
                    f"'{CE_VALUE_ARG}'."
                )
                raise RegimeInitializationError(msg)

    @property
    def param_names(self) -> frozenset[str]:
        """Names of the runtime parameters of `transform` and `inverse`."""
        return frozenset(
            get_union_of_args([self.transform, self.inverse]) - {CE_VALUE_ARG}
        )


def power_transform(value: FloatND, risk_aversion: FloatND) -> FloatND:
    """Apply `g(v) = v^(1 - risk_aversion)`, or `log(v)` at `risk_aversion = 1`."""
    return jnp.where(
        risk_aversion == 1.0, jnp.log(value), value ** (1.0 - risk_aversion)
    )


def power_inverse(value: FloatND, risk_aversion: FloatND) -> FloatND:
    """Apply `g^(-1)(v) = v^(1 / (1 - risk_aversion))`; `exp(v)` in the log case."""
    # The unselected power branch must not divide by zero at `risk_aversion = 1`.
    safe_risk_aversion = jnp.where(risk_aversion == 1.0, 0.0, risk_aversion)
    return jnp.where(
        risk_aversion == 1.0,
        jnp.exp(value),
        value ** (1.0 / (1.0 - safe_risk_aversion)),
    )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class PowerMean(QuasiArithmeticMean):
    """Epstein-Zin power-mean certainty equivalent.

    `CE = (E[V'^(1 - risk_aversion)])^(1 / (1 - risk_aversion))` with the
    runtime parameter `{"certainty_equivalent": {"risk_aversion": ...}}`.
    Requires strictly positive continuation values. `risk_aversion = 1` is
    the geometric-mean (log) limit, `CE = exp(E[log V'])`; `risk_aversion
    = 0` reduces to the linear expectation.
    """

    transform: Callable[..., FloatND] = power_transform
    inverse: Callable[..., FloatND] = power_inverse

    def aggregate(
        self, *, values: FloatND, weights: FloatND, risk_aversion: FloatND
    ) -> FloatND:
        """Return the weighted power mean `(E[v^(1-ra)])^(1/(1-ra))`, stably.

        `ra` is `risk_aversion`. The naive `inverse(sum(w · transform(v)))`
        overflows when `risk_aversion > 1` and `v` is near the borrowing
        constraint: the intermediate `v^(1-ra)` exceeds the dtype's range and the
        certainty equivalent collapses to zero or infinity. The aggregation
        evaluates in an anchored weight/deviation log form —
        `log CE = a + [log(W) + log1p(E/W)] / (1-ra)` with `a` the extremal
        log value, `W` the weight sum, and `E` the `expm1`-deviation sum —
        which stays finite wherever the mathematical value is and keeps the
        geometric-mean limit exact arbitrarily close to `risk_aversion = 1`.
        `risk_aversion = 1` is the weighted geometric mean `exp(E[log v])`.

        Args:
            values: Strictly positive continuation values along the last axis.
            weights: Nonnegative probabilities over `values`, summing to one.
                The generator-weighted sum is not renormalized — scaling the
                weights by `k` scales the result by `k^(1/(1-ra))` — so only a
                unit-mass lottery yields a certainty equivalent. Zero-weight
                entries drop out exactly.
            risk_aversion: The Epstein-Zin risk-aversion coefficient.

        Returns:
            The certainty equivalent, reduced over the last axis.

        """
        log_v = jnp.log(values)
        positive = weights > 0.0
        exponent = 1.0 - risk_aversion
        # The `risk_aversion == 1` power branch must not divide by zero.
        safe_exponent = jnp.where(exponent == 0.0, 1.0, exponent)
        # Anchored weight/deviation form: with `a` the extremal log value on
        # the side that keeps every exponent nonpositive,
        # `log CE = a + [log(W) + log1p(E / W)] / (1-ra)` where `W = sum w`
        # and `E = sum w expm1((1-ra)(log v - a))`. The deviation ratio keeps
        # the quotient exact arbitrarily close to `ra = 1` — a rounded
        # log-sum divided by a near-zero exponent loses the geometric-mean
        # limit to cancellation — while the exact mass term `log(W)` (zero
        # for a unit-mass lottery) preserves the unnormalized sum's value.
        anchor_high = jnp.max(jnp.where(positive, log_v, -jnp.inf), axis=-1)
        anchor_low = jnp.min(jnp.where(positive, log_v, jnp.inf), axis=-1)
        anchor = jnp.where(exponent >= 0.0, anchor_high, anchor_low)
        anchor = jnp.where(exponent == 0.0, 0.0, anchor)
        centered = log_v - anchor[..., None]
        masked_weights = jnp.where(positive, weights, weights * 0.0)
        weight_sum = jnp.sum(jnp.broadcast_to(masked_weights, centered.shape), axis=-1)
        deviation_sum = jnp.sum(
            jnp.where(
                positive,
                weights * jnp.expm1(exponent * centered),
                weights * 0.0,
            ),
            axis=-1,
        )
        safe_weight = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
        log_ce_power = (
            anchor
            + (jnp.log(safe_weight) + jnp.log1p(deviation_sum / safe_weight))
            / safe_exponent
        )
        log_ce_geometric = jnp.sum(
            jnp.where(positive, weights * log_v, weights * 0.0), axis=-1
        )
        return jnp.exp(jnp.where(exponent == 0.0, log_ce_geometric, log_ce_power))


def resolve_certainty_equivalent(
    certainty_equivalent: CertaintyEquivalent | None,
) -> tuple[
    QuasiArithmeticMean | None,
    MappingProxyType[str, str],
    MappingProxyType[str, str],
]:
    """Narrow the certainty equivalent and map its args to flat param names.

    The runtime parameters live under the pseudo-function name
    `certainty_equivalent` in the regime's flat params
    (`certainty_equivalent__<arg>`); the returned mappings let the Q-and-F
    closure pull each callable's kwargs from `states_actions_params`.

    Returns:
        Tuple of the narrowed quasi-arithmetic-mean CE (or `None`), the
        transform's arg-to-flat-name mapping, and the inverse's
        arg-to-flat-name mapping.

    """
    if certainty_equivalent is None:
        return None, MappingProxyType({}), MappingProxyType({})
    if not isinstance(certainty_equivalent, QuasiArithmeticMean):
        msg = (
            "Only `QuasiArithmeticMean` certainty equivalents are "
            f"supported, got {type(certainty_equivalent).__name__}."
        )
        raise NotImplementedError(msg)

    def flat_names(func: Callable[..., FloatND]) -> MappingProxyType[str, str]:
        return MappingProxyType(
            {
                arg: f"certainty_equivalent__{arg}"
                for arg in get_union_of_args([func]) - {CE_VALUE_ARG}
            }
        )

    return (
        certainty_equivalent,
        flat_names(certainty_equivalent.transform),
        flat_names(certainty_equivalent.inverse),
    )
