"""Recover and admit the outer action, identically in solve and simulation.

N-NB-EGM searches the outer margin over post-decision targets and retains those,
so both phases have to recover the action that reaches a retained target. They
must recover it the same way: a solve that certifies a candidate and a simulation
that replays a different action for it disagree about which stock was chosen.

The recovery reads the map's coefficient from its structure rather than measuring
it (see `outer_affine_structure`), then divides only when the coefficient is not
one. Admission is tolerance-free and two-tier:

- every candidate's image must lie inside the relevant branch domain — the outer
  state's declared domain for a keeper, and the published outer mesh for an
  adjuster; and
- a candidate whose target *is* a domain endpoint must reproduce that endpoint
  bit for bit, because the endpoint is where an ULP of error leaves the domain.

Interior candidates are admitted on containment alone. Requiring target-local
reproduction there rejects ordinary candidate banks whose inverse round trip
accumulates more than one target ULP while remaining inside the represented
domain. The optional stricter rule therefore remains outside this merge repair.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

import jax
import jax.numpy as jnp

from _lcm.egm.outer_affine_structure import certify_outer_coefficient
from lcm.exceptions import RegimeInitializationError
from lcm.solver_api import _register_artifact_static_metadata_dataclass
from lcm.typing import BoolND, FloatND, ScalarFloat

__all__ = [
    "DeclaredOuterInverse",
    "OuterInversion",
    "abstract_like",
    "certify_declared_outer_inverse",
    "coefficient_is_exactly_invertible",
    "invert_declared_outer_target",
    "outer_candidate_is_admissible",
    "recover_outer_action",
]


@dataclass(frozen=True)
class DeclaredOuterInverse:
    """The certified inverse of one regime's outer post-decision map."""

    coefficient: Fraction
    """The exact coefficient of the outer action, read off the map's structure."""

    low: float
    """Lower endpoint of the outer state's declared domain."""

    high: float
    """Upper endpoint of the outer state's declared domain."""


_register_artifact_static_metadata_dataclass(
    cls=DeclaredOuterInverse,
    field_names=("coefficient", "low", "high"),
)


@dataclass(frozen=True)
class OuterInversion:
    """One recovered outer action, its forward image, and its admission."""

    action: FloatND
    """The outer action recovered from the retained target."""

    image: FloatND
    """The declared map re-evaluated at `action` -- the stock actually reached."""

    admissible: BoolND
    """Which candidates may be published."""


def certify_declared_outer_inverse(
    *,
    func: Callable[..., Mapping[str, FloatND]],
    arg_names: tuple[str, ...],
    abstract_args: Iterable[object],
    outer_action_name: str,
    outer_post_decision_name: str,
    outer_state_domain: tuple[float, float],
    regime_name: str,
) -> DeclaredOuterInverse:
    """Return the inverse of `func`, refusing a map it cannot invert exactly.

    Args:
        func: The resolved post-decision DAG, returning a mapping of targets.
        arg_names: Names of the arguments `func` is called with.
        abstract_args: Shapes and dtypes for those arguments, in the same order.
        outer_action_name: The action the map is inverted in.
        outer_post_decision_name: The target whose identity the solve retains.
        outer_state_domain: The outer state's declared `(low, high)` endpoints.
        regime_name: Named in the refusal so the modeller knows where to look.

    Returns:
        A `DeclaredOuterInverse` carrying the exact coefficient and the domain
        every recovered stock must land in.

    Raises:
        RegimeInitializationError: If the outer action does not enter the map
            affinely, does not enter it at all, or enters it with a coefficient
            that cannot be divided out exactly.
    """
    names = tuple(arg_names)
    certificate = certify_outer_coefficient(
        func=_OuterPostDecisionTarget(
            func=func,
            arg_names=names,
            outer_post_decision_name=outer_post_decision_name,
        ),
        outer_action_name=outer_action_name,
        abstract_args=abstract_args,
        arg_names=names,
    )
    remedy = (
        "Declare the outer post-decision target as an affine function of the "
        f"outer action {outer_action_name!r} whose coefficient is a power of "
        "two -- `new = old + action` and `new = old + 2 * action` both "
        "qualify, and the offset may be any function of states and params. A "
        "map outside that form is solved by `GridSearch`, which searches the "
        "outer action directly and so never inverts it."
    )
    if certificate.coefficient is None:
        msg = (
            f"Regime {regime_name!r} declares an outer post-decision target "
            f"N-NB-EGM cannot invert. {certificate.violation} {remedy}"
        )
        raise RegimeInitializationError(msg)
    if certificate.coefficient == 0:
        msg = (
            f"Regime {regime_name!r} declares an outer post-decision target "
            f"that does not depend on the outer action {outer_action_name!r}. "
            "A constant map retains no information about the action that "
            f"reached it, so no action can be recovered. {remedy}"
        )
        raise RegimeInitializationError(msg)
    if not coefficient_is_exactly_invertible(certificate.coefficient):
        msg = (
            f"Regime {regime_name!r} moves {certificate.coefficient} units of "
            f"outer stock per unit of {outer_action_name!r}. Dividing that out "
            "rounds, and a rounded outer action reaches a stock away from the "
            "node the solve ranked -- at a domain endpoint, away from the "
            f"declared domain entirely. {remedy}"
        )
        raise RegimeInitializationError(msg)
    low, high = outer_state_domain
    return DeclaredOuterInverse(coefficient=certificate.coefficient, low=low, high=high)


@dataclass(frozen=True, kw_only=True, eq=False)
class _OuterPostDecisionTarget:
    """One target of a post-decision DAG, callable positionally in argument order.

    The coefficient certificate traces a positional map, while the resolved DAG
    binds by name and returns every target at once; this binds the two forms.
    Its `__name__` is the target's, so the certificate's diagnostics name it.
    """

    func: Callable[..., Mapping[str, FloatND]]
    """The resolved post-decision DAG, returning a mapping of targets."""

    arg_names: tuple[str, ...]
    """Names of `func`'s arguments, in the positional order it is called with."""

    outer_post_decision_name: str
    """The target read off the DAG's output mapping."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "__name__", self.outer_post_decision_name)

    def __call__(self, *values: object) -> FloatND:
        bound = dict(zip(self.arg_names, values, strict=True))
        return jnp.asarray(self.func(**bound)[self.outer_post_decision_name])


def invert_declared_outer_target(
    *,
    inverse: DeclaredOuterInverse,
    target: FloatND,
    at_zero: FloatND,
    forward: Callable[[FloatND], FloatND],
) -> OuterInversion:
    """Recover the outer action reaching `target`, and admit it or drop it.

    Args:
        inverse: The regime's certified inverse.
        target: The post-decision target the solve retained.
        at_zero: The declared map evaluated at a zero outer action.
        forward: The declared map, to re-evaluate at the recovered action.

    Returns:
        An `OuterInversion` carrying the action, the stock it actually reaches,
        and which candidates may be published. The image is returned alongside
        the action rather than recomputed by the caller: every downstream reader
        needs the stock the action reaches, and evaluating the DAG twice both
        costs and invites the two to drift apart.
    """
    action = recover_outer_action(
        target=target, at_zero=at_zero, coefficient=inverse.coefficient
    )
    image = forward(action)
    admissible = outer_candidate_is_admissible(
        image=image,
        target=target,
        low=jnp.asarray(inverse.low, dtype=image.dtype),
        high=jnp.asarray(inverse.high, dtype=image.dtype),
    )
    return OuterInversion(action=action, image=image, admissible=admissible)


def abstract_like(values: Iterable[object]) -> tuple[object, ...]:
    """Return shape/dtype stand-ins for values, so tracing never runs the map.

    The certificate reads structure, and structure does not depend on the
    numbers flowing through it. Handing `make_jaxpr` abstract stand-ins also
    keeps certification safe to perform while the caller is itself being traced.
    """
    return tuple(
        jax.ShapeDtypeStruct(array.shape, array.dtype)
        for array in (jnp.asarray(value) for value in values)
    )


def coefficient_is_exactly_invertible(coefficient: Fraction) -> bool:
    """Return whether dividing by `coefficient` is exact in binary floating point.

    A power of two scales the exponent and leaves the significand alone, so the
    quotient is exact. Any other factor rounds it however exactly the factor
    itself is known -- measured, a coefficient of three leaves hundreds of
    thousands of states off their endpoint even with a perfectly declared slope.
    Such a map is refused where it is declared rather than left to drop
    candidates one at a time.

    This is necessary structural support, never sufficient pointwise
    certification. Exact division by a power of two does not make
    `target - offset` followed by the forward addition exact for every
    representable offset -- on a signed domain it is not -- so the runtime
    endpoint and containment predicate stays active for a dyadic coefficient
    exactly as it does for any other.
    """
    if coefficient == 0:
        return False
    magnitude = abs(coefficient)
    numerator, denominator = magnitude.numerator, magnitude.denominator
    return _is_power_of_two(numerator) and _is_power_of_two(denominator)


def recover_outer_action(
    *, target: FloatND, at_zero: FloatND, coefficient: Fraction
) -> FloatND:
    """Return the outer action whose post-decision image is `target`.

    `at_zero` is the map evaluated at a zero outer action, so the affine map is
    `image = at_zero + coefficient * action`. At unit coefficient the inverse is
    a subtraction and no division is performed at all, which is what removes the
    rounding the estimated-slope route introduced.
    """
    difference = target - at_zero
    if coefficient == 1:
        return difference
    divisor = jnp.asarray(
        coefficient.numerator / coefficient.denominator, dtype=difference.dtype
    )
    return difference / divisor


def outer_candidate_is_admissible(
    *,
    image: FloatND,
    target: FloatND,
    low: ScalarFloat,
    high: ScalarFloat,
) -> BoolND:
    """Return which candidates may be published, deciding without a tolerance.

    Args:
        image: The declared map re-evaluated at the recovered action.
        target: The post-decision target the solve retained.
        low: The relevant domain's lower bound.
        high: The relevant domain's upper bound.

    Returns:
        A boolean array over candidates. Every comparison is exact, so no scale
        constant appears anywhere on the acceptance path and there is nothing an
        out-of-domain image can pass by being small enough.
    """
    finite = jnp.isfinite(image) & jnp.isfinite(target)
    inside = (image >= low) & (image <= high)
    at_endpoint = (target == low) | (target == high)
    reproduces_endpoint = image == target
    return finite & inside & (~at_endpoint | reproduces_endpoint)


def _is_power_of_two(value: int) -> bool:
    """Return whether a positive integer is a power of two."""
    return value > 0 and value & (value - 1) == 0
