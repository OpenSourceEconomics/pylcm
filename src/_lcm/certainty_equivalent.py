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

from _lcm.beartype_conf import PARAMS_CONF, REGIME_CONF
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
    continuation is aggregated as the linear expectation `E[V']`. Only
    `GridSearch` supports a nonlinear certainty equivalent.
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

    @beartype(conf=PARAMS_CONF)
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
            weights: Nonnegative weights over `values`. The lottery is a
                probability distribution, so the weights are normalized by
                their sum; scaling them all by a constant leaves the result
                unchanged, and a lottery carrying no mass aggregates to NaN.
            params: Mapping of runtime parameter names to their values.
                `transform` and `inverse` each receive the subset their
                signature declares.

        Returns:
            The certainty equivalent, reduced over the last axis.

        """
        transformed = self.transform(value=values, **_args_for(self.transform, params))
        weight_sum = jnp.sum(weights, axis=-1)
        return self.inverse(
            value=jnp.sum(weights * transformed, axis=-1) / weight_sum,
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
    `risk_aversion = 1` is the geometric-mean (log) limit,
    `CE = exp(E[log V'])`; `risk_aversion = 0` reduces to the linear
    expectation. Continuation values must be positive, except that a value of
    exactly zero is admitted as the limiting case.

    The aggregation is evaluated in an anchored log form, so the result stays
    finite wherever the mathematical power mean is — including high risk
    aversion at continuation values near the borrowing constraint, where
    `V'^(1 - risk_aversion)` alone would overflow the dtype, and a lottery
    whose lowest value carries almost none of the probability mass.
    """

    transform: Callable[..., FloatND] = power_transform
    inverse: Callable[..., FloatND] = power_inverse

    def __post_init__(self) -> None:
        for name, expected in (
            ("transform", power_transform),
            ("inverse", power_inverse),
        ):
            if getattr(self, name) is not expected:
                msg = (
                    f"`PowerMean` aggregates the power transform "
                    f"`v^(1 - risk_aversion)` in a form specific to it, so it "
                    f"cannot honour a custom `{name}`. Use "
                    f"`QuasiArithmeticMean` for other transform pairs."
                )
                raise RegimeInitializationError(msg)
        super().__post_init__()

    @beartype(conf=PARAMS_CONF)
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
        `log CE = a + log M / (1-ra)`, with `a` the extremal log value and `M`
        the moment of the anchored, mass-normalized lottery.
        `M` has two representations — an `expm1`-deviation sum that survives
        `risk_aversion -> 1` and an `exp` sum that survives a lottery carrying
        near-zero mass on the anchor — and the aggregation takes whichever one
        the lottery does not cancel. The result therefore stays finite
        wherever the mathematical value is, and the geometric-mean limit stays
        exact arbitrarily close to `risk_aversion = 1`. `risk_aversion = 1` is
        the weighted geometric mean `exp(E[log v])`.

        The geometric mean is selected by an exact `risk_aversion == 1` test,
        so at that one point the result carries no dependence on
        `risk_aversion` and its derivative there reads as zero rather than the
        true finite value. Gradient-based work that starts exactly at unit risk
        aversion should offset the starting value.

        Args:
            values: Strictly positive continuation values along the last axis.
                A value of exactly zero is admitted as the limiting case: it
                sends the certainty equivalent to zero above unit risk
                aversion and contributes nothing below it.
            weights: Nonnegative weights over `values`. The lottery is a
                probability distribution, so the weights are normalized by
                their sum; scaling them all by a constant leaves the result
                unchanged, and a lottery carrying no mass aggregates to NaN.
                Zero-weight entries drop out exactly, while a NaN weight
                propagates.
            params: Mapping carrying the `risk_aversion` runtime parameter.

        Returns:
            The certainty equivalent, reduced over the last axis.

        """
        risk_aversion = params["risk_aversion"]
        live = weights > 0.0
        # A node carrying no weight must not be able to inject a non-finite
        # quantity into the reductions. `log(0)` is `-inf` and its derivative
        # `1/0` is infinite, and a weight of exactly zero cancels neither —
        # forward it would take a `jnp.where` in every reduction, and in
        # reverse mode the masked cotangent would still be `0 * inf`. Replacing
        # the value itself keeps both directions finite, and leaves the result
        # unchanged because the node's normalized weight is exactly zero.
        log_v = jnp.log(jnp.where(live, values, 1.0))
        exponent = 1.0 - risk_aversion
        # The `risk_aversion == 1` power branch must not divide by zero.
        safe_exponent = jnp.where(exponent == 0.0, 1.0, exponent)
        # Anchored form: with `a` the extremal log value on the side that
        # keeps every scaled exponent nonpositive,
        # `log CE = a + log M / (1-ra)` where `M = sum (w/W) exp((1-ra)(log v - a))`
        # over the mass-normalized lottery.
        #
        # Only a node with a finite log value may anchor. A live node at value
        # zero has `log v = -inf`, and anchoring there would make its own
        # `log v - a` the indeterminate `-inf - (-inf)`. Anchored to a finite
        # node instead, that value's scaled exponent is `+inf` below unit risk
        # aversion, the moment diverges, and the certainty equivalent goes to
        # zero — which is what `v^(1-ra) -> inf` gives.
        anchorable = live & jnp.isfinite(log_v)
        anchor_high = jnp.max(jnp.where(anchorable, log_v, -jnp.inf), axis=-1)
        anchor_low = jnp.min(jnp.where(anchorable, log_v, jnp.inf), axis=-1)
        anchor = jnp.where(exponent >= 0.0, anchor_high, anchor_low)
        # With no anchorable node the reductions sit at their sentinels; any
        # finite anchor will do, and zero keeps `centered` well defined.
        anchor = jnp.where(jnp.isfinite(anchor), anchor, 0.0)
        anchor = jnp.where(exponent == 0.0, 0.0, anchor)
        centered = log_v - anchor[..., None]
        broadcast_live = jnp.broadcast_to(live, centered.shape)
        broadcast_weights = jnp.broadcast_to(weights, centered.shape)
        # A NaN weight is deliberately kept rather than masked away: a
        # malformed lottery must surface as NaN, not silently lose a branch.
        masked_weights = jnp.where(
            broadcast_live | jnp.isnan(broadcast_weights), broadcast_weights, 0.0
        )
        weight_sum = jnp.sum(masked_weights, axis=-1)
        # A lottery carrying no mass has no certainty equivalent. The linear
        # path's `jnp.average` reports the same NaN, so either aggregation
        # reaches the NaN diagnostics on the same model.
        safe_weight = jnp.where(weight_sum > 0.0, weight_sum, jnp.nan)
        normalized_weights = masked_weights / safe_weight[..., None]
        scaled = exponent * centered
        # The moment `M = Σ w̃ exp((1-ra)(log v - a))` has two representations,
        # each exact where the other cancels. Both reduce the same anchored
        # lottery, whose terms all lie in `[0, 1]` with the anchor's own term
        # at exactly its weight — so neither can overflow, and only a value of
        # exactly zero can send a term to infinity:
        # - `log1p(Σ w̃ expm1(...))` sums small negatives, so it survives
        #   `1-ra -> 0`, where `log(M)` would be a rounded `0/0` against the
        #   exponent and the geometric-mean limit would be lost.
        # - `log(Σ w̃ exp(...))` sums strict positives, so it survives a wide
        #   lottery carrying near-zero mass on the anchor, where every other
        #   `expm1` rounds to `-1`, the deviation ratio rounds to `-1` too,
        #   and `log1p` would report a mathematically positive moment as zero.
        deviation_ratio = jnp.sum(normalized_weights * jnp.expm1(scaled), axis=-1)
        moment = jnp.sum(normalized_weights * jnp.exp(scaled), axis=-1)
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
        log_ce_power = anchor + log_moment / safe_exponent
        log_ce_geometric = jnp.sum(normalized_weights * log_v, axis=-1)
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
