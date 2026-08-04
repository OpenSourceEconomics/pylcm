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
from _lcm.power_mean import weighted_power_mean
from _lcm.utils.functools import get_union_of_args
from lcm.exceptions import RegimeInitializationError
from lcm.typing import FloatND

# Reserved argument name through which transform callables receive values.
CE_VALUE_ARG = "value"


class CertaintyEquivalent(ABC):
    """Base class for certainty-equivalent specifications.

    Declared on a non-terminal `Regime` via `certainty_equivalent=...`. The
    shipped implementations are `LinearExpectation` — expected utility, and
    what a regime that declares nothing gets — and `QuasiArithmeticMean` with
    its `PowerMean` specialization.

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
        return jnp.sum(weights * values, axis=-1) / jnp.sum(weights, axis=-1)


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
        `W_epstein_zin` shares: the naive `inverse(Σ w · transform(v))`
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
                propagates.
            params: Mapping carrying the `risk_aversion` runtime parameter.

        Returns:
            The certainty equivalent, reduced over the last axis.

        """
        return weighted_power_mean(
            values=values,
            weights=weights,
            exponent=1.0 - params["risk_aversion"],
        )


def _args_for(
    func: Callable[..., FloatND], params: Mapping[str, FloatND]
) -> dict[str, FloatND]:
    """Pick the entries of `params` that `func`'s signature declares."""
    return {name: params[name] for name in get_union_of_args([func]) - {CE_VALUE_ARG}}


def aggregates_nonlinearly(certainty_equivalent: CertaintyEquivalent | None) -> bool:
    """Whether this certainty equivalent is anything other than the linear one.

    Every non-terminal regime now carries a certainty equivalent, so PRESENCE
    no longer distinguishes a nonlinear aggregation from the expected-utility
    default: `LinearExpectation` is a real class with a real `aggregate`, and it
    is what a regime that declared nothing receives. A guard that means "this
    path does not implement a nonlinear CE" must therefore ask for the property,
    not for presence.
    """
    return certainty_equivalent is not None and not isinstance(
        certainty_equivalent, LinearExpectation
    )
