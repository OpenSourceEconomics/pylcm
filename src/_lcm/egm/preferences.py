"""The preference maps an EGM step reads off a regime's own `functions`.

Every endogenous-grid step needs the same three maps of the consumption action —
felicity $u$, its marginal $u'$, and the inverse marginal $(u')^{-1}$ — plus the
discount factor $\\beta$ the Koopmans aggregator applies to the continuation. All
of them come from the regime the modeller wrote, never from a preference family
the solver assumes:

- $u$ is the regime's own `utility` DAG target, bound to one parameter set;
- $u'$ is its action-derivative, taken by automatic differentiation;
- $(u')^{-1}$ is the regime's `inverse_marginal_utility` target when it declares
  one, and a bracketed Newton solve of $u'$ otherwise (the iEGM path).

`Preferences` bundles the three as unary callables the steps apply elementwise,
so a step may hand them a single node or a whole mesh. The discount factor stays
a separate runtime scalar: it multiplies a continuation, not an action, and the
steps read it where they read the continuation.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from dags import concatenate_functions

from _lcm.egm.numeric_inverse import numeric_inverse_marginal_utility
from _lcm.regime_building.w_dag import _get_build_W_kwargs
from _lcm.typing import (
    EconFunction,
    EconFunctionsMapping,
    FunctionName,
)
from lcm.typing import ActionName, Float1D, FloatND, ScalarFloat, UserFunction

# Name of the regime function supplying the analytic inverse marginal utility.
INVERSE_MARGINAL_UTILITY = "inverse_marginal_utility"
# Lower bracket on the action for the Newton inverse — a small positive floor.
NEWTON_ACTION_FLOOR = 1e-8


def newton_action_ceiling(scale_grid: Float1D) -> ScalarFloat:
    """Return the initial upper action bracket for the numerical inverse.

    The post-decision grid supplies a stable model-specific scale, not a hard
    consumption bound. `numeric_inverse_marginal_utility` expands this initial
    endpoint geometrically when the Euler root lies above it, so income or
    current resources may legitimately exceed the savings grid's scale.
    """
    return scale_grid[-1] * 1000.0 + 1000.0


def get_numeric_inverse_marginal_utility(
    *,
    marginal_utility: Callable[[ScalarFloat], ScalarFloat],
    action_lower: ScalarFloat | float,
    action_upper: ScalarFloat | float,
) -> Callable[[ScalarFloat], ScalarFloat]:
    """Return the bracketed Newton inverse of a marginal felicity — the iEGM path.

    A felicity composed from a regime's DAG has no closed-form Euler inversion, so
    the root is found numerically inside a bracket that dominates every feasible
    action. The returned map is scalar-in, scalar-out: callers that carry whole
    meshes either `jax.vmap` it or lift it elementwise.

    Args:
        marginal_utility: Marginal felicity `u'(c)` of a scalar action, bound to
            one parameter set.
        action_lower: Lower bracket on the action — a small positive floor.
        action_upper: Upper bracket on the action; must dominate every feasible
            action.

    Returns:
        Callable mapping a marginal continuation to the action that equates
        `u'(c)` with it.

    """
    return _NumericInverseMarginalUtility(
        marginal_utility=marginal_utility,
        action_lower=action_lower,
        action_upper=action_upper,
    )


@dataclass(frozen=True)
class Preferences:
    """Felicity, marginal felicity, and inverse marginal felicity of the action.

    Each callable is bound to one parameter set and applies elementwise, so the
    same bundle serves a per-node scalar solve and a whole-mesh step.
    """

    utility: Callable[[FloatND], FloatND]
    """Felicity `u(c)` of the consumption action."""

    marginal_utility: Callable[[FloatND], FloatND]
    """Marginal felicity `u'(c)`."""

    inverse_marginal_utility: Callable[[FloatND], FloatND]
    """Inverse marginal felicity `(u')^{-1}(m)` — the Euler inversion."""


def preferences_from_utility(
    *,
    utility_of_action: Callable[[FloatND], FloatND],
    action_lower: ScalarFloat | float,
    action_upper: ScalarFloat | float,
) -> Preferences:
    """Bundle a felicity with its autodiff marginal and its bracketed Newton inverse.

    For a felicity already bound to its parameters and to whatever states the
    caller holds fixed — a cell-bound period utility, say — rather than one
    concatenated from a regime's `functions`. Both maps are lifted elementwise,
    so the bundle applies to a single node or a whole mesh alike.

    Args:
        utility_of_action: Felicity `u(c)` of the consumption action, bound.
        action_lower: Lower bracket on the action — a small positive floor.
        action_upper: Upper bracket on the action; must dominate every feasible
            action.

    Returns:
        The bound `Preferences` bundle.

    """
    marginal_utility = jax.grad(utility_of_action)
    return Preferences(
        utility=_elementwise(utility_of_action),
        marginal_utility=_elementwise(marginal_utility),
        inverse_marginal_utility=_elementwise(
            get_numeric_inverse_marginal_utility(
                marginal_utility=marginal_utility,
                action_lower=action_lower,
                action_upper=action_upper,
            )
        ),
    )


def get_preferences_builder(
    *,
    functions: EconFunctionsMapping,
    action_name: ActionName,
    action_lower: ScalarFloat | float,
    action_upper: ScalarFloat | float,
) -> Callable[[Mapping[str, Any]], Preferences]:
    """Return a closure binding a regime's preference maps to a parameter set.

    The concatenation happens once, at kernel-build time; the returned closure
    binds the regime's flat parameters into the three maps at each call, so a
    solve may vary parameters without recompiling the DAG.

    Args:
        functions: The regime's processed functions (parameters carry their
            qualified names). Must contain a `utility` target;
            `inverse_marginal_utility` is optional.
        action_name: The regime's own name for the consumption action, the
            argument the three maps are functions of.
        action_lower: Lower bracket for the Newton inverse — a small positive
            floor on the action. Unused when the regime declares an analytic
            inverse.
        action_upper: Initial upper bracket for the Newton inverse. The numerical
            solver expands it when necessary.

    Returns:
        Callable mapping the regime's flat parameters to the bound
        `Preferences` bundle.

    """
    utility_func = concatenate_regime_function(functions=functions, target="utility")
    analytic_inverse = (
        concatenate_regime_function(
            functions=functions, target=INVERSE_MARGINAL_UTILITY
        )
        if INVERSE_MARGINAL_UTILITY in functions
        else None
    )

    return _PreferencesBuilder(
        utility_func=utility_func,
        analytic_inverse=analytic_inverse,
        action_name=action_name,
        action_lower=action_lower,
        action_upper=action_upper,
    )


def get_discount_factor_reader(
    *,
    functions: EconFunctionsMapping,
    koopmans_aggregator: EconFunction,
) -> Callable[[Mapping[str, Any]], ScalarFloat]:
    """Return a closure reading `beta` off the aggregator's own signature.

    The aggregator declares which parameters it consumes beyond `utility` and
    `CE`, and `fail_if_custom_koopmans_aggregator` pins that set to the single
    discount factor. Reading it through the aggregator rather than by a
    hard-coded qualified name keeps the modeller's spelling — including a
    `discount_factor` computed by a DAG function, e.g. one indexing a per-type
    series — the single source of the value.

    Args:
        functions: The regime's processed functions, so an aggregator parameter
            that is itself a regime function is computed rather than demanded.
        koopmans_aggregator: The regime's processed Koopmans aggregator.

    Returns:
        Callable mapping the regime's flat parameters to the discount factor.

    """
    return _DiscountFactorReader(
        build_W_kwargs=_get_build_W_kwargs(
            functions=functions, koopmans_aggregator=koopmans_aggregator
        )
    )


def concatenate_regime_function(
    *,
    functions: EconFunctionsMapping,
    target: FunctionName,
) -> UserFunction:
    """Concatenate one regime-function target from the regime DAG."""
    return concatenate_functions(
        functions=dict(functions),
        targets=target,
        enforce_signature=False,
        set_annotations=True,
    )


def _elementwise(
    func: Callable[[ScalarFloat], ScalarFloat],
) -> Callable[[FloatND], FloatND]:
    """Lift a scalar-in, scalar-out map to any array shape.

    `jax.grad` and the Newton inverse are defined on a scalar action, while the
    EGM steps carry whole grids and meshes. Flattening, mapping, and restoring
    the shape covers both — a 0-d input maps through a length-one batch — and
    keeps the lifted map jittable and itself vmappable.
    """
    return _Elementwise(func=func)


@dataclass(frozen=True, kw_only=True, eq=False)
class _Elementwise:
    """A scalar-in, scalar-out map lifted to any array shape."""

    func: Callable[[ScalarFloat], ScalarFloat]
    """The scalar map applied at every element."""

    def __call__(self, values: FloatND) -> FloatND:
        array = jnp.asarray(values)
        return jax.vmap(self.func)(array.reshape(-1)).reshape(array.shape)


@dataclass(frozen=True, kw_only=True, eq=False)
class BoundUtilityOfAction:
    """Felicity `u(c)` of the consumption action, everything else bound.

    Calls the regime's concatenated utility with the action under the regime's
    own name and every other argument — states, discrete codes, flat params —
    from `bound`.
    """

    utility_func: UserFunction
    """The regime's concatenated utility function."""

    action_name: ActionName
    """The regime's name for the consumption action."""

    bound: Mapping[str, Any]
    """Every other argument of `utility_func`, by name."""

    def __call__(self, action_value: FloatND) -> FloatND:
        return self.utility_func(**{self.action_name: action_value}, **self.bound)


@dataclass(frozen=True, kw_only=True, eq=False)
class AnalyticInverseMarginalUtility:
    """The regime's declared `inverse_marginal_utility`, parameters bound."""

    analytic_inverse: UserFunction
    """The regime's concatenated inverse-marginal-utility function."""

    bound: Mapping[str, Any]
    """Every argument of `analytic_inverse` but `marginal_continuation`."""

    def __call__(self, marginal_continuation: FloatND) -> FloatND:
        return self.analytic_inverse(
            marginal_continuation=marginal_continuation, **self.bound
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class _NumericInverseMarginalUtility:
    """The bracketed Newton inverse of a marginal felicity — the iEGM path."""

    marginal_utility: Callable[[ScalarFloat], ScalarFloat]
    """Marginal felicity `u'(c)` of a scalar action, bound to one parameter set."""

    action_lower: ScalarFloat | float
    """Lower bracket on the action — a small positive floor."""

    action_upper: ScalarFloat | float
    """Initial upper bracket on the action, expanded when the root lies above it."""

    def __call__(self, marginal_continuation: ScalarFloat) -> ScalarFloat:
        return numeric_inverse_marginal_utility(
            marginal_continuation=marginal_continuation,
            marginal_utility=self.marginal_utility,
            c_lower=jnp.asarray(self.action_lower),
            c_upper=jnp.asarray(self.action_upper),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class _PreferencesBuilder:
    """Bind a regime's concatenated preference maps to one parameter set.

    The concatenation happens once, at kernel-build time; each call binds the
    regime's flat parameters into the three maps, so a solve may vary
    parameters without recompiling the DAG.
    """

    utility_func: UserFunction
    """The regime's concatenated `utility` target."""

    analytic_inverse: UserFunction | None
    """The regime's concatenated `inverse_marginal_utility` target, if declared."""

    action_name: ActionName
    """The regime's own name for the consumption action."""

    action_lower: ScalarFloat | float
    """Lower bracket for the Newton inverse (unused with an analytic inverse)."""

    action_upper: ScalarFloat | float
    """Initial upper bracket for the Newton inverse."""

    def __call__(self, params: Mapping[str, Any]) -> Preferences:
        utility_of_action = BoundUtilityOfAction(
            utility_func=self.utility_func, action_name=self.action_name, bound=params
        )
        marginal_utility = _elementwise(jax.grad(utility_of_action))
        inverse_marginal_utility: Callable[[FloatND], FloatND]
        if self.analytic_inverse is not None:
            inverse_marginal_utility = AnalyticInverseMarginalUtility(
                analytic_inverse=self.analytic_inverse, bound=params
            )
        else:
            inverse_marginal_utility = _elementwise(
                get_numeric_inverse_marginal_utility(
                    marginal_utility=jax.grad(utility_of_action),
                    action_lower=self.action_lower,
                    action_upper=self.action_upper,
                )
            )
        return Preferences(
            utility=utility_of_action,
            marginal_utility=marginal_utility,
            inverse_marginal_utility=inverse_marginal_utility,
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class _DiscountFactorReader:
    """Read the discount factor off the Koopmans aggregator's own signature."""

    build_W_kwargs: Callable[[Mapping[str, Any]], dict[str, Any]]
    """Assembles the aggregator's keyword arguments beyond `utility` and `CE`."""

    def __call__(self, params: Mapping[str, Any]) -> ScalarFloat:
        (discount_factor,) = tuple(self.build_W_kwargs(params).values())
        return discount_factor
