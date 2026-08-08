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
    RegimeName,
)
from lcm.exceptions import ModelInitializationError
from lcm.koopmans_aggregation import LinearAggregator
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.typing import ActionName, Float1D, FloatND, ScalarFloat, UserFunction

# Name of the regime function supplying the analytic inverse marginal utility.
INVERSE_MARGINAL_UTILITY = "inverse_marginal_utility"
# Lower bracket on the action for the Newton inverse — a small positive floor.
NEWTON_ACTION_FLOOR = 1e-8


def newton_action_ceiling(scale_grid: Float1D) -> ScalarFloat:
    """Return an upper action bracket dominating every feasible action.

    The bound is derived from a post-decision grid's top node — the model's own
    resources scale — rather than from a resources value, since the Euler
    inversion runs per post-decision node with no single current-state
    resources to read. A model whose optimal action can genuinely exceed this
    multiple of its own savings scale (income far above the savings scale) is
    mis-scaled for its grid: the root is clamped to the bound and reported with
    a zero derivative, as a binding corner is. Widen the grid or declare an
    analytic `inverse_marginal_utility` in that case.
    """
    return scale_grid[-1] * 1000.0 + 1000.0


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
        action_upper: Upper bracket for the Newton inverse; must dominate every
            feasible action.

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

    def build(params: Mapping[str, Any]) -> Preferences:
        def utility_of_action(action_value: FloatND) -> FloatND:
            return utility_func(**{action_name: action_value}, **params)

        marginal_utility = _elementwise(jax.grad(utility_of_action))

        if analytic_inverse is not None:

            def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
                return analytic_inverse(
                    marginal_continuation=marginal_continuation, **params
                )
        else:

            def scalar_inverse(marginal_continuation: ScalarFloat) -> ScalarFloat:
                return numeric_inverse_marginal_utility(
                    marginal_continuation=marginal_continuation,
                    marginal_utility=jax.grad(utility_of_action),
                    c_lower=jnp.asarray(action_lower),
                    c_upper=jnp.asarray(action_upper),
                )

            inverse_marginal_utility = _elementwise(scalar_inverse)

        return Preferences(
            utility=utility_of_action,
            marginal_utility=marginal_utility,
            inverse_marginal_utility=inverse_marginal_utility,
        )

    return build


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
    build_W_kwargs = _get_build_W_kwargs(functions, koopmans_aggregator)

    def read(params: Mapping[str, Any]) -> ScalarFloat:
        (discount_factor,) = tuple(build_W_kwargs(params).values())
        return discount_factor

    return read


def fail_if_custom_koopmans_aggregator(
    *, regime_name: RegimeName, user_regime: UserRegime, solver_name: str
) -> None:
    """Require the default Koopmans aggregator `W` at solve time.

    An Euler inversion hard-codes `W = utility + discount_factor * CE`, so a
    custom *solve-phase* aggregator would silently change the meaning of the
    solution. A `Phased` aggregator whose solve variant is the default is
    accepted — an Euler solver never reads the simulate variant, so a naive
    present-bias regime (`koopmans_aggregator=Phased(solve=LinearAggregator(),
    simulate=beta_delta_W)`) is admissible: the present bias enters only the
    simulate-phase re-optimization, outside the Euler inversion.

    A regime that declares nothing takes the model-level default, which this
    check sees as `None` because it runs on the user regime before the model
    fills the slot.

    Args:
        regime_name: Name of the regime being validated.
        user_regime: The user-facing regime, before the model fills its slots.
        solver_name: Name of the solver the message names as the constraint.

    Raises:
        ModelInitializationError: If the regime declares a custom solve-phase
            Koopmans aggregator.

    """
    declared = user_regime.koopmans_aggregator
    solve_W = declared.solve if isinstance(declared, Phased) else declared
    if solve_W is not None and not isinstance(solve_W, LinearAggregator):
        msg = (
            f"Regime '{regime_name}' declares a custom solve-phase Koopmans "
            f"aggregator. The {solver_name} solver hard-codes the default "
            "aggregator `W = utility + discount_factor * CE` at solve time; "
            "remove the custom `koopmans_aggregator` (a `Phased` one whose "
            "solve variant is `LinearAggregator()` is accepted) or use the "
            "brute-force solver."
        )
        raise ModelInitializationError(msg)


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

    def lifted(values: FloatND) -> FloatND:
        array = jnp.asarray(values)
        return jax.vmap(func)(array.reshape(-1)).reshape(array.shape)

    return lifted
