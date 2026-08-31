"""Certainty-equivalent classes and engine helpers.

The public `lcm.certainty_equivalent` module re-exports the three classes
(`CertaintyEquivalent`, `QuasiArithmeticMean`, `PowerMean`) for user code.
Engine modules may import directly from here.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import PARAMS_CONF, REGIME_CONF
from _lcm.power_mean import weighted_power_mean
from _lcm.probability import (
    binades_above_smallest_normal,
    binades_to_fit_product,
    is_live,
    is_negative,
    is_represented_zero,
    rescaled_lottery_weights,
    scaled_down_by_power_of_two,
)
from _lcm.utils.functools import get_union_of_args
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.exceptions import (
    RegimeInitializationError,
    ScaledLotteryDifferentiationError,
)
from lcm.typing import FloatND, IntND

# Reserved argument name through which transform callables receive values.
CE_VALUE_ARG = "value"


class CertaintyEquivalent(ABC):
    """Base class for certainty-equivalent specifications.

    Declared on a non-terminal `Regime` via `certainty_equivalent=...`. The
    shipped implementations are `LinearExpectation` (expected utility, the
    default) and `QuasiArithmeticMean` with its `PowerMean` specialization.

    `aggregate` reduces the whole joint continuation lottery in one piece,
    because a transform has to be applied before any expectation is taken.
    `LinearExpectation` needs no transform, so the engine reduces each target
    regime on its own instead and never materializes the joint lottery; its
    `aggregate` states the same quantity and serves as the reference that
    route is tested against. Only `GridSearch` supports a nonlinear certainty
    equivalent.
    """

    @property
    @abstractmethod
    def param_names(self) -> frozenset[str]:
        """Names of the certainty equivalent's runtime parameters."""

    @abstractmethod
    def aggregate(
        self,
        *,
        values: FloatND,
        weights: FloatND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        """Reduce the continuation lottery over its last axis.

        The whole lottery — every stochastic node of every reachable target
        regime, weighted by that target's regime-transition probability —
        arrives flattened, so a transform is applied before every expectation
        and inverted exactly once.

        Args:
            values: Continuation values of the lottery along the last axis.
            weights: Nonnegative weights over `values`.
            params: Mapping of the runtime parameter names in `param_names`
                to their values.

        Returns:
            The certainty equivalent, reduced over the last axis.

        """

    def aggregate_scaled(
        self,
        *,
        values: FloatND,
        coefficients: FloatND,
        shifts: IntND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        """Reduce a continuation lottery whose weights carry per-entry scales.

        Node `i` carries probability `coefficients[i] * 2**-shifts[i]`. A joint
        probability formed from several rare factors can sit further below the
        likeliest node than the exponent field spans, and the pair is then the
        only form in which the lottery is exact — on any single scale the
        rarest node has to be rounded.

        This base is a capability boundary rather than an implementation. There
        is no reduction it could perform on an arbitrary subclass's behalf: the
        only thing it knows about that subclass is `aggregate`, which takes
        ordinary numbers, and no ordinary number states a probability below the
        smallest positive float. Flattening the pairs first and deferring would
        understate exactly the node the pair exists to carry, and a node whose
        value is large enough to matter is precisely the one that survives
        being rare — so the understatement is not small in the answer.

        Every shipped certainty equivalent overrides this. A model whose
        certainty equivalent reaches this method is rejected when it is built,
        so the failure is a construction error naming the remedy rather than
        this exception; the raise is the backstop for a lottery that reaches it
        another way.

        Args:
            values: Continuation values of the lottery along the last axis.
            coefficients: The weights' significands, over the same axis.
            shifts: Each weight's own base-two scale, broadcast against them.
            params: Mapping of the runtime parameter names in `param_names`
                to their values.

        Raises:
            NotImplementedError: Always.

        """
        msg = (
            f"`{type(self).__name__}` has no `aggregate_scaled`, so it cannot "
            "reduce a continuation lottery whose weights carry their own "
            "scales. Implement `aggregate_scaled`, or use one of the shipped "
            "certainty equivalents (`LinearExpectation`, `QuasiArithmeticMean`, "
            "`PowerMean`)."
        )
        raise NotImplementedError(msg)

    @property
    def flat_param_names(self) -> MappingProxyType[str, str]:
        """Immutable mapping of each runtime parameter to its flat params name.

        The parameters live under the pseudo-function name
        `certainty_equivalent` in the regime's flat params, so the Q-and-F
        closure can assemble `aggregate`'s `params` straight from
        `states_actions_params`.
        """
        return MappingProxyType(
            {arg: f"certainty_equivalent__{arg}" for arg in self.param_names}
        )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class LinearExpectation(CertaintyEquivalent):
    """Certainty equivalent of expected utility: the plain expectation `E[V']`.

    The engine recognizes this specification and reduces each target regime on
    its own rather than flattening the joint lottery, which is cheaper by
    roughly a factor of two on any lottery past a couple of nodes. `aggregate`
    below states the same quantity over the flattened lottery; it is the
    reference the engine's route is tested against, not the route it takes.
    """

    @property
    def param_names(self) -> frozenset[str]:
        """The plain expectation has no runtime parameters."""
        return frozenset()

    @beartype(conf=PARAMS_CONF)
    def aggregate(
        self,
        *,
        values: FloatND,
        weights: FloatND,
        params: Mapping[str, FloatND],  # noqa: ARG002
    ) -> FloatND:
        """Return the probability-weighted mean of the lottery.

        Args:
            values: Continuation values of the lottery along the last axis.
            weights: Nonnegative weights over `values`, normalized by their
                sum; a lottery carrying no mass aggregates to NaN.
            params: Unused — the plain expectation has no parameters.

        Returns:
            The expectation, reduced over the last axis.

        """
        # Rescaling by a common power of two leaves the mean unchanged and
        # keeps a weight below the normal range out of the multiplication,
        # where a backend that flushes it would turn a rare node's `-inf` into
        # `0 * -inf` and take the whole lottery down as NaN. No power of two
        # reaches a weight of exactly zero, so the node that cannot occur is
        # still annihilated by the term itself — the scale is accounted for by
        # that point, so this is its cheap branch.
        weights = rescaled_lottery_weights(weights=weights)
        return jnp.sum(
            zero_safe_weighted_term(
                weight=weights, value=values, subnormal_is_accounted_for=True
            ),
            axis=-1,
        ) / jnp.sum(weights, axis=-1)

    @beartype(conf=PARAMS_CONF)
    def aggregate_scaled(
        self,
        *,
        values: FloatND,
        coefficients: FloatND,
        shifts: IntND,
        params: Mapping[str, FloatND],  # noqa: ARG002
    ) -> FloatND:
        """Return the expectation of a lottery whose weights carry their scales.

        The scale is applied to each node's `weight * value` rather than to its
        weight, so a node too rare to state on any one scale still contributes
        what it is worth. That is the whole of the difference: a node whose
        probability sits `2**-1024` below the likeliest one and whose value is
        the largest the format holds contributes a term of order one-half, and
        naming its weight first would report the mean as zero.

        Args:
            values: Continuation values of the lottery along the last axis.
            coefficients: The weights' significands, over the same axis.
            shifts: Each weight's own base-two scale, broadcast against them.
            params: Unused — the plain expectation has no parameters.

        Returns:
            The expectation, reduced over the last axis.

        """
        value_terms, weight_terms = _scaled_lottery_terms(
            values=values, coefficients=coefficients, shifts=shifts
        )
        return _states_no_derivative(
            jnp.sum(value_terms, axis=-1) / jnp.sum(weight_terms, axis=-1)
        )


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

    Combined with a user-supplied Koopmans aggregator `W` this expresses
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
                Because that normalization divides any lost mass back out
                without leaving a trace, the caller — not this method — is
                responsible for the weights summing to one; the engine checks
                it in `_lcm.regime_building.Q_and_F`.
            params: Mapping of runtime parameter names to their values.
                `transform` and `inverse` each receive the subset their
                signature declares.

        Returns:
            The certainty equivalent, reduced over the last axis.

        """
        # A node carrying no weight must not be able to inject a non-finite
        # quantity into the reduction. `transform` is arbitrary user code and
        # may be unbounded at the edge of its domain — `log` at zero is the
        # ordinary case — and a weight of exactly zero does not cancel an
        # infinity: `0 * inf` is NaN, which would take the well-specified nodes
        # down with it. Transforming a stand-in value instead keeps the
        # reduction finite and changes nothing, the node's weight being zero.
        # The stand-in is copied from the heaviest node rather than being a
        # constant, because an arbitrary constant need not lie in `transform`'s
        # domain while a value already in the lottery always does. A constant
        # `transform` is unbounded at would leave `0 * inf` on the branch the
        # mask discards — absent from the value, NaN in its derivative.
        # `transform` can be unbounded, so a weight the dtype cannot multiply
        # is not a negligible term here: `g(v)` at a near-zero value can be
        # large enough that the product is of order one. Rescaling the lottery
        # by a common power of two leaves every ratio exactly as supplied and
        # puts every live weight where the arithmetic can use it.
        # A negative weight is not a lottery. Mapping it to NaN makes it
        # propagate like any other malformed weight instead of being read as
        # "dead" by the liveness test below and silently dropped — which would
        # return the surviving nodes' mean as though the caller had asked for
        # it. `weighted_power_mean` opens the same way, for the same reason.
        weights = rescaled_lottery_weights(weights=weights)
        weights = jnp.where(is_negative(weights), jnp.nan, weights)
        live = is_live(weights)
        stand_in = jnp.take_along_axis(
            values, jnp.argmax(weights, axis=-1, keepdims=True), axis=-1
        )
        safe_values = jnp.where(live, values, stand_in)
        transformed = self.transform(
            value=safe_values, **_args_for(func=self.transform, params=params)
        )
        weight_sum = jnp.sum(weights, axis=-1)
        return self.inverse(
            value=jnp.sum(jnp.where(live, weights * transformed, 0.0), axis=-1)
            / weight_sum,
            **_args_for(func=self.inverse, params=params),
        )

    @beartype(conf=PARAMS_CONF)
    def aggregate_scaled(
        self,
        *,
        values: FloatND,
        coefficients: FloatND,
        shifts: IntND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        """Reduce the scaled lottery, transforming before the scales are spent.

        `transform` is arbitrary user code and may be unbounded inside its
        domain: `1/v` at a value of zero, `log v` likewise. So the quantity being
        averaged is `g(v)`, not `v`, and a node whose value is ordinary while its
        transformed value is enormous carries the whole certainty equivalent. Its
        probability is exactly the kind a single scale cannot state, which is why
        the transform happens first and the scales are still present when the
        reduction takes them: `weight * g(value)` is an ordinary number where
        `weight` alone is not.

        The order is therefore: replace the values that genuinely cannot occur,
        transform, reduce with the scales in hand, invert once.

        Args:
            values: Continuation values of the lottery along the last axis.
            coefficients: The weights' significands, over the same axis.
            shifts: Each weight's own base-two scale, broadcast against them.
            params: Mapping of the runtime parameter names in `param_names`
                to their values.

        Returns:
            The certainty equivalent, reduced over the last axis.

        """
        coefficients = jnp.asarray(coefficients)
        # A negative coefficient is not a probability. It reaches the result
        # through the mass rather than the numerator, exactly as in `aggregate`:
        # the liveness test would read a NaN weight as a node that cannot occur.
        coefficients = jnp.where(is_negative(coefficients), jnp.nan, coefficients)
        # The null events are the represented zeros of the coefficient, which no
        # scale can change. The stand-in comes from the likeliest node that can
        # occur — the one carrying the smallest shift — because an arbitrary
        # constant need not lie in `transform`'s domain while a value already in
        # the lottery always does.
        live = is_live(coefficients)
        unusable = jnp.max(shifts, axis=-1, keepdims=True)
        stand_in = jnp.take_along_axis(
            values,
            jnp.argmin(jnp.where(live, shifts, unusable), axis=-1, keepdims=True),
            axis=-1,
        )
        safe_values = jnp.where(is_represented_zero(coefficients), stand_in, values)
        transformed = self.transform(
            value=safe_values, **_args_for(func=self.transform, params=params)
        )
        value_terms, weight_terms = _scaled_lottery_terms(
            values=transformed, coefficients=coefficients, shifts=shifts
        )
        return _states_no_derivative(
            self.inverse(
                value=jnp.sum(value_terms, axis=-1) / jnp.sum(weight_terms, axis=-1),
                **_args_for(func=self.inverse, params=params),
            )
        )


def power_transform(*, value: FloatND, risk_aversion: FloatND) -> FloatND:
    """Apply `g(v) = v^(1 - risk_aversion)`, or `log(v)` at `risk_aversion = 1`."""
    return jnp.where(
        risk_aversion == 1.0, jnp.log(value), value ** (1.0 - risk_aversion)
    )


def power_inverse(*, value: FloatND, risk_aversion: FloatND) -> FloatND:
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
    """`g` — fixed to the power transform; see `inverse`."""

    inverse: Callable[..., FloatND] = power_inverse
    """`g⁻¹` — fixed to the power inverse.

    The pair defines the mean, and `aggregate` evaluates it by a route that
    survives ranges where applying them directly overflows. They are therefore
    the reference the anchored form is tested against rather than the code path
    it takes, and neither may be replaced: swapping one would leave `aggregate`
    computing something else entirely.
    """

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
        """Return the weighted power mean `(Σ w̃ · v^(1-ra))^(1/(1-ra))`, stably.

        `ra` is `risk_aversion` and `w̃` the mass-normalized weights. The
        evaluation is `weighted_power_mean`, which the Koopmans aggregator
        `CESAggregator` shares: the naive `inverse(Σ w · transform(v))`
        overflows when `risk_aversion > 1` and `v` is near the borrowing
        constraint, so the mean is taken in an anchored log form instead. It
        stays finite wherever the mathematical value is, and the
        geometric-mean limit stays exact arbitrarily close to
        `risk_aversion = 1`, where the result is `exp(E[log v])`.

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
                propagates. Because that normalization divides any lost mass
                back out without leaving a trace, the caller — not this method
                — is responsible for the weights summing to one; the engine
                checks it in `_lcm.regime_building.Q_and_F`.
            params: Mapping carrying the `risk_aversion` runtime parameter.

        Returns:
            The certainty equivalent, reduced over the last axis.

        """
        return weighted_power_mean(
            values=values,
            weights=weights,
            exponent=1.0 - params["risk_aversion"],
            # Weights given as plain numbers already share one scale.
            shifts=jnp.zeros((), jnp.int32),
        )

    @beartype(conf=PARAMS_CONF)
    def aggregate_scaled(
        self,
        *,
        values: FloatND,
        coefficients: FloatND,
        shifts: IntND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        """Return the power mean of a lottery carrying per-entry scales, exactly.

        `weighted_power_mean` reduces in the log domain, where a node's scale
        is a subtraction rather than a magnitude the format has to hold. The
        lottery is therefore priced at whatever spread it arrives with, and
        nothing is understated.
        """
        return _states_no_derivative(
            weighted_power_mean(
                values=values,
                weights=coefficients,
                exponent=1.0 - params["risk_aversion"],
                shifts=shifts,
            )
        )


def _identity(value: FloatND) -> FloatND:
    """Return `value` unchanged, and refuse to be differentiated.

    A scaled reduction's whole reason to exist is that the format cannot state
    its weights as ordinary numbers. A derivative with respect to such a weight
    is in the same position, and the derivative machinery has no scale to work
    on: it would materialize the rare quantity as an ordinary float, get zero,
    and hand back a gradient indistinguishable from a genuinely flat objective.

    Refusing at trace time is what makes that visible. It costs the primal
    nothing — under `jit` or plain evaluation this is the identity — and it
    leaves the ordinary `aggregate` route, where every weight is a number the
    format holds, differentiable as before.
    """
    return value


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _scaled_reduction_jvp(
    primals: tuple[FloatND, ...],
    tangents: tuple[FloatND, ...],
) -> tuple[FloatND, FloatND]:
    """Raise rather than report a derivative the scale was needed to state."""
    del primals, tangents
    msg = (
        "`aggregate_scaled` states no derivative: its lottery carries weights "
        "as `(coefficient, shift)` pairs precisely because no ordinary float "
        "states the probability, and the same holds of a derivative with "
        "respect to one. Differentiate the model through a route whose weights "
        "are ordinary numbers, or reduce the lottery with `aggregate`."
    )
    raise ScaledLotteryDifferentiationError(msg)


# Built by call rather than by decorator: `@jax.custom_jvp` produces a callable
# instance, which the package claw rebinds to a bound method of its `__call__`,
# losing `defjvp` along with everything else the object knows.
_states_no_derivative = jax.custom_jvp(_identity)
_states_no_derivative.defjvp(_scaled_reduction_jvp)


def _args_for(
    *, func: Callable[..., FloatND], params: Mapping[str, FloatND]
) -> dict[str, FloatND]:
    """Pick the entries of `params` that `func`'s signature declares."""
    return {name: params[name] for name in get_union_of_args([func]) - {CE_VALUE_ARG}}


def aggregates_nonlinearly(certainty_equivalent: CertaintyEquivalent | None) -> bool:
    """Whether this certainty equivalent is anything other than the linear one.

    The cut is the type, not whether a value is attached. `LinearExpectation`
    is a real class with a real `aggregate` and is what a regime that declared
    nothing receives, so every non-terminal regime carries a certainty
    equivalent and presence separates nothing. A guard that means "this path
    does not implement a nonlinear CE" therefore asks whether the certainty
    equivalent is `LinearExpectation`; an unattached slot carries no nonlinear
    aggregation of its own and answers False as well.
    """
    return certainty_equivalent is not None and not isinstance(
        certainty_equivalent, LinearExpectation
    )


def _scaled_lottery_terms(
    *, values: FloatND, coefficients: FloatND, shifts: IntND
) -> tuple[FloatND, FloatND]:
    """Return a lottery's value terms and weight terms on one shared scale.

    A node's contribution to a mean is `c * 2**-s * v`, and the order those
    three are combined in decides whether the node survives. Two orders each
    lose it at one end of the range:

    - forming the weight `c * 2**-s` first loses a rare node, because the
      weight of a node many binades below the likeliest one is not
      representable and becomes zero, taking its value with it — although the
      *product* it was heading for may be an ordinary number, which is
      precisely the case where the node changes the answer;
    - forming `c * v` first loses a node whose coefficient has been normalized
      above one and whose value sits near the top of the range, because that
      intermediate overflows to infinity before any scale is applied to bring
      it back.

    Splitting the scale covers both. The coefficient absorbs as much of the
    downward shift as it can while staying normal, the product absorbs the
    remainder. Neither an unrepresentable weight nor an overflowing product is
    ever materialized, and a rare node earns its place in the mean exactly when
    its value is large enough to offset its probability.

    Both returned terms carry the same shift, so the scale cancels in their
    ratio and the caller divides them directly. The shift is the smallest the
    row carries, per lottery rather than per batch, so the likeliest node's
    weight lands at its own magnitude and no unrelated lottery beside it can
    move it. A dead node takes no part in that choice: an unreachable node
    carrying a large shift would otherwise push every live weight down with it.

    The row's own weight total joins that shift, scaled against the largest
    value the row carries, so that the weighted sum cannot reach infinity while
    the mean it states is an ordinary number. Two equally likely nodes near the
    top of the range are the case: their mean is comfortably representable, but
    the numerator alone is not. That correction is the smallest one that fits,
    because every binade spent at the top is one the rarest term of the same
    sum loses at the bottom.

    A coefficient that is not a probability at all — negative, infinite, or
    NaN — is not a dead node and is not silently dropped. It poisons its whole
    row, so a malformed lottery is visible in the result rather than reduced to
    a plausible number.

    Args:
        values: Continuation values of the lottery along the last axis.
        coefficients: The weights' significands, over the same axis.
        shifts: Each weight's own base-two scale, broadcast against them.

    Returns:
        Tuple of the value terms and the weight terms, both over the last axis
        and both carrying the row's shared scale.

    """
    coefficients = jnp.asarray(coefficients)
    values = jnp.asarray(values)
    shifts = jnp.broadcast_to(jnp.asarray(shifts), jnp.shape(coefficients))

    live = is_live(coefficients) & jnp.isfinite(coefficients)
    invalid = is_negative(coefficients) | (
        ~jnp.isfinite(coefficients) & ~is_represented_zero(coefficients)
    )
    unusable = jnp.max(shifts, axis=-1, keepdims=True)
    common = jnp.min(jnp.where(live, shifts, unusable), axis=-1, keepdims=True)
    scale = (common - shifts).astype(jnp.int32)

    on_common_scale = jnp.where(
        live,
        scaled_down_by_power_of_two(values=coefficients, shift=scale),
        jnp.zeros_like(coefficients),
    )
    total = jnp.sum(on_common_scale, axis=-1, keepdims=True)
    largest = jnp.max(
        jnp.where(live, jnp.abs(values), jnp.zeros_like(values)),
        axis=-1,
        keepdims=True,
    )
    full = scale - binades_to_fit_product(left=total, right=largest)

    room = binades_above_smallest_normal(coefficients)
    on_weight = jnp.maximum(full, -room)
    scaled_coefficients = scaled_down_by_power_of_two(
        values=coefficients, shift=on_weight
    )
    products = zero_safe_weighted_term(
        weight=scaled_coefficients,
        value=values,
        subnormal_is_accounted_for=False,
    )
    value_terms = scaled_down_by_power_of_two(values=products, shift=full - on_weight)
    weight_terms = scaled_down_by_power_of_two(values=coefficients, shift=full)

    value_terms = jnp.where(live, value_terms, jnp.zeros_like(value_terms))
    weight_terms = jnp.where(live, weight_terms, jnp.zeros_like(weight_terms))
    invalid_row = jnp.any(invalid, axis=-1, keepdims=True)
    nan = jnp.asarray(jnp.nan, dtype=coefficients.dtype)
    return (
        jnp.where(invalid_row, nan, value_terms),
        jnp.where(invalid_row, nan, weight_terms),
    )
