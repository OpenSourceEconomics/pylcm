"""Nonlinear certainty equivalents over the next-period value distribution."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

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
    engine dispatches on the concrete subclass; `TransformedExpectation` is
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
class TransformedExpectation(CertaintyEquivalent):
    """Certainty equivalent `CE = g⁻¹(Σ_r p_r · E_w[g(V'_r)])`.

    `transform` (`g`) is applied elementwise to next-period values before
    every expectation - over stochastic state transitions and over regime
    transitions - and `inverse` (`g⁻¹`) once, after the regime-probability-
    weighted sum. Both callables take the value array as the reserved first
    argument `value`; every further signature argument becomes a runtime
    parameter under the pseudo-function name `certainty_equivalent` in the
    regime's params (`{"certainty_equivalent": {"<arg>": ...}}`).

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
                    f"The `{name}` callable of a `TransformedExpectation` must "
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


def _power_transform(value: FloatND, risk_aversion: FloatND) -> FloatND:
    return value ** (1.0 - risk_aversion)


def _power_inverse(value: FloatND, risk_aversion: FloatND) -> FloatND:
    return value ** (1.0 / (1.0 - risk_aversion))


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class PowerCertaintyEquivalent(TransformedExpectation):
    """Epstein-Zin power certainty equivalent.

    `CE = (E[V'^(1 - risk_aversion)])^(1 / (1 - risk_aversion))` with the
    runtime parameter `{"certainty_equivalent": {"risk_aversion": ...}}`.
    Requires strictly positive continuation values; `risk_aversion = 1`
    (the log case) is not representable. `risk_aversion = 0` reduces to
    the linear expectation.
    """

    transform: Callable[..., FloatND] = _power_transform
    inverse: Callable[..., FloatND] = _power_inverse
