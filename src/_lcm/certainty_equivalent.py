"""Certainty-equivalent classes and engine helpers.

The public `lcm.certainty_equivalent` module re-exports the three classes
(`CertaintyEquivalent`, `QuasiArithmeticMean`, `PowerMean`) for user code.
Engine modules may import directly from here.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
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

# Deviation ratio at which `PowerMean.aggregate` switches from the `log1p`
# moment representation to the `log` one. Either side of it both are accurate
# to a couple of ulps: `log1p` degrades only as the ratio approaches `-1` and
# `log` only as it approaches `0`, so any interior crossover works.
_DEVIATION_RATIO_CROSSOVER = -0.5


class CertaintyEquivalent(ABC):
    """Base class for certainty-equivalent specifications.

    Declared on a non-terminal `Regime` via `certainty_equivalent=...`. The
    engine dispatches on the concrete subclass; `QuasiArithmeticMean` is
    the shipped implementation. When the field is `None` (the default), the
    continuation is aggregated as the linear expectation `E[V']`. `GridSearch` is
    the only solver that supports a nonlinear certainty equivalent; declaring one
    on any other solver is rejected when the model builds.
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

    def aggregate(
        self,
        *,
        values: FloatND,
        weights: FloatND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        """Return the certainty equivalent `g⁻¹(Σ w · g(v))` over the last axis.

        The single aggregation entry point of the engine: the whole
        continuation lottery — every stochastic node of every reachable
        target regime, weighted by the regime probability — arrives here
        flattened, so `transform` is applied before every expectation and
        `inverse` exactly once.

        Args:
            values: Continuation values of the lottery along the last axis.
            weights: Nonnegative probabilities over `values`. A unit-mass
                lottery yields a certainty equivalent; a smaller mass carries
                through as the correspondingly smaller aggregate.
            params: Mapping of runtime parameter names to their values.
                `transform` and `inverse` each receive the subset their
                signature declares.

        Returns:
            The certainty equivalent, reduced over the last axis.

        """
        transformed = self.transform(value=values, **_args_for(self.transform, params))
        return self.inverse(
            value=jnp.sum(weights * transformed, axis=-1),
            **_args_for(self.inverse, params),
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

    The aggregation is evaluated in an anchored log form, so the result stays
    finite wherever the mathematical power mean is — including high risk
    aversion at continuation values near the borrowing constraint, where
    `V'^(1 - risk_aversion)` alone would overflow the dtype, and a lottery
    whose lowest value carries almost none of the probability mass.
    """

    transform: Callable[..., FloatND] = power_transform
    inverse: Callable[..., FloatND] = power_inverse

    def aggregate(
        self,
        *,
        values: FloatND,
        weights: FloatND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        """Return the weighted power mean `(Σ w · v^(1-ra))^(1/(1-ra))`, stably.

        `ra` is `risk_aversion`. The naive `inverse(Σ w · transform(v))`
        overflows when `risk_aversion > 1` and `v` is near the borrowing
        constraint: the intermediate `v^(1-ra)` exceeds the dtype's range and
        the certainty equivalent collapses to zero or infinity. The
        aggregation evaluates in the anchored log form
        `log CE = a + [log(W) + log M] / (1-ra)`, with `a` the extremal log
        value, `W` the weight sum, and `M` the moment of the anchored lottery.
        `M` has two representations — an `expm1`-deviation sum that survives
        `risk_aversion -> 1` and an `exp` sum that survives a lottery carrying
        near-zero mass on the anchor — and the aggregation takes whichever one
        the lottery does not cancel. The result therefore stays finite
        wherever the mathematical value is, and the geometric-mean limit stays
        exact arbitrarily close to `risk_aversion = 1`. `risk_aversion = 1` is
        the weighted geometric mean `exp(E[log v])`.

        Args:
            values: Strictly positive continuation values along the last axis.
            weights: Nonnegative probabilities over `values`, summing to one.
                A weight sum within sqrt(eps) of one is floating summation
                roundoff on a unit-mass lottery and aggregates as exactly
                normalized — the power mean has a finite `ra -> 1` limit only
                at unit mass. Scaling the weights by a materially non-unit
                `k` scales the result by `k^(1/(1-ra))` (with no `ra -> 1`
                limit; `ra = 1` publishes the normalized geometric mean), so
                only a unit-mass lottery yields a certainty equivalent.
                Zero-weight entries drop out exactly.
            params: Mapping carrying the `risk_aversion` runtime parameter.

        Returns:
            The certainty equivalent, reduced over the last axis.

        """
        risk_aversion = params["risk_aversion"]
        log_v = jnp.log(values)
        positive = weights > 0.0
        exponent = 1.0 - risk_aversion
        # The `risk_aversion == 1` power branch must not divide by zero.
        safe_exponent = jnp.where(exponent == 0.0, 1.0, exponent)
        # Anchored form: with `a` the extremal log value on the side that
        # keeps every scaled exponent nonpositive,
        # `log CE = a + [log(W) + log M] / (1-ra)` where `W = sum w` and
        # `M = sum (w/W) exp((1-ra)(log v - a))`. The mass term `log(W)`
        # carries a materially non-unit weight sum exactly and drops out for a
        # unit-mass lottery (up to summation roundoff; see below).
        anchor_high = jnp.max(jnp.where(positive, log_v, -jnp.inf), axis=-1)
        anchor_low = jnp.min(jnp.where(positive, log_v, jnp.inf), axis=-1)
        anchor = jnp.where(exponent >= 0.0, anchor_high, anchor_low)
        anchor = jnp.where(exponent == 0.0, 0.0, anchor)
        centered = log_v - anchor[..., None]
        broadcast_positive = jnp.broadcast_to(positive, centered.shape)
        broadcast_weights = jnp.broadcast_to(weights, centered.shape)
        masked_weights = jnp.where(
            broadcast_positive, broadcast_weights, broadcast_weights * 0.0
        )
        weight_sum = jnp.sum(masked_weights, axis=-1)
        safe_weight = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
        normalized_weights = masked_weights / safe_weight[..., None]
        scaled = exponent * centered
        # The moment `M = Σ w̃ exp((1-ra)(log v - a))` has two representations,
        # each exact where the other cancels. Both reduce the same anchored
        # lottery, whose terms all lie in `[0, 1]` with the anchor's own term
        # at exactly its weight, so neither can overflow:
        # - `log1p(Σ w̃ expm1(...))` sums small negatives, so it survives
        #   `1-ra -> 0`, where `log(M)` would be a rounded `0/0` against the
        #   exponent and the geometric-mean limit would be lost.
        # - `log(Σ w̃ exp(...))` sums strict positives, so it survives a wide
        #   lottery carrying near-zero mass on the anchor, where every other
        #   `expm1` rounds to `-1`, the deviation ratio rounds to `-1` too,
        #   and `log1p` would report a mathematically positive moment as zero.
        deviation_ratio = jnp.sum(
            jnp.where(
                broadcast_positive,
                normalized_weights * jnp.expm1(scaled),
                normalized_weights * 0.0,
            ),
            axis=-1,
        )
        moment = jnp.sum(
            jnp.where(
                broadcast_positive,
                normalized_weights * jnp.exp(scaled),
                normalized_weights * 0.0,
            ),
            axis=-1,
        )
        # `jnp.where` evaluates both branches, so each has to stay finite even
        # where it is discarded — an infinity in the dead branch would still
        # reach a gradient.
        near_geometric = deviation_ratio > _DEVIATION_RATIO_CROSSOVER
        safe_deviation_ratio = jnp.where(near_geometric, deviation_ratio, 0.0)
        safe_moment = jnp.where(near_geometric, 1.0, moment)
        log_moment = jnp.where(
            near_geometric,
            jnp.log1p(safe_deviation_ratio),
            jnp.log(safe_moment),
        )
        # A mass gap below sqrt(eps) is floating summation roundoff on a
        # mathematically unit-mass lottery (quadrature weights rarely sum to
        # one bit-exactly): `log(W)/(1-ra)` would amplify it to an order-one
        # error near `ra = 1`, so such lotteries aggregate as exactly
        # normalized. A materially non-unit mass keeps its exact `log(W)`
        # contribution (the documented `k^(1/(1-ra))` scaling).
        roundoff_mass = jnp.abs(weight_sum - 1.0) <= jnp.sqrt(
            jnp.finfo(safe_weight.dtype).eps
        )
        log_mass = jnp.where(roundoff_mass, 0.0, jnp.log(safe_weight))
        log_ce_power = anchor + (log_mass + log_moment) / safe_exponent
        log_ce_geometric = jnp.sum(
            jnp.where(
                broadcast_positive,
                normalized_weights * log_v,
                normalized_weights * 0.0,
            ),
            axis=-1,
        )
        return jnp.exp(jnp.where(exponent == 0.0, log_ce_geometric, log_ce_power))


def resolve_certainty_equivalent(
    certainty_equivalent: CertaintyEquivalent | None,
) -> tuple[
    QuasiArithmeticMean | None,
    MappingProxyType[str, str],
]:
    """Narrow the certainty equivalent and map its args to flat param names.

    The runtime parameters live under the pseudo-function name
    `certainty_equivalent` in the regime's flat params
    (`certainty_equivalent__<arg>`); the returned mapping lets the Q-and-F
    closure assemble `aggregate`'s `params` from `states_actions_params`.

    Returns:
        Tuple of the narrowed quasi-arithmetic-mean CE (or `None`) and its
        arg-to-flat-name mapping.

    """
    if certainty_equivalent is None:
        return None, MappingProxyType({})
    if not isinstance(certainty_equivalent, QuasiArithmeticMean):
        msg = (
            "Only `QuasiArithmeticMean` certainty equivalents are "
            f"supported, got {type(certainty_equivalent).__name__}."
        )
        raise NotImplementedError(msg)

    return (
        certainty_equivalent,
        MappingProxyType(
            {
                arg: f"certainty_equivalent__{arg}"
                for arg in certainty_equivalent.param_names
            }
        ),
    )


def _args_for(
    func: Callable[..., FloatND], params: Mapping[str, FloatND]
) -> dict[str, FloatND]:
    """Pick the entries of `params` that `func`'s signature declares."""
    return {name: params[name] for name in get_union_of_args([func]) - {CE_VALUE_ARG}}
