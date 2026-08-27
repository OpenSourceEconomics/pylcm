"""Certify the exact coefficient of an outer post-decision map in its action.

N-NB-EGM retains post-decision targets and must recover the action that reaches
one. That recovery needs the map's slope in the outer action, and measuring the
slope by differencing two rounded evaluations is not accurate enough to use: for
`target = state + action` the secant is one ULP away from one, and dividing by it
carries the reassembled stock off the outer state's declared domain.

The coefficient is therefore read off the traced structure, in exact rational
arithmetic. A forward walk over the jaxpr seeds the outer action's variable with
coefficient one and propagates it through the affine primitives; a variable that
carries a coefficient is *tainted*, and any primitive that consumes a tainted
variable without an exact affine rule refuses the map.

Automatic differentiation cannot do this job. `stop_gradient` is invisible to it,
so `state + action + stop_gradient(2 * action)` linearizes as unit slope while the
map actually moves three units of stock per unit of action. The walk reads the
arithmetic rather than its derivative, so it sees the term AD drops.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

import jax

__all__ = ["OuterAffineCertificate", "certify_outer_coefficient"]

# Sub-jaxprs hang off different param keys per primitive: `closed_call` and
# `pjit` use `jaxpr`, `custom_jvp_call` uses `call_jaxpr`. Naming them keeps a
# primitive whose payload sits under an unlisted key from being walked as if it
# had no body -- it is refused instead.
_SUB_JAXPR_PARAM_KEYS = ("jaxpr", "call_jaxpr", "fun_jaxpr")

# Primitives with an exact affine rule in a tainted operand.
_ADD_PRIMS = frozenset({"add", "add_any"})
_PASS_THROUGH_PRIMS = frozenset(
    {
        "convert_element_type",
        "reshape",
        "broadcast_in_dim",
        "squeeze",
        "copy",
        "stop_gradient",
    }
)


@dataclass(frozen=True)
class OuterAffineCertificate:
    """The outcome of certifying one outer post-decision map."""

    coefficient: Fraction | None
    """The exact coefficient of the outer action, or `None` when refused."""

    violation: str | None
    """Why the map was refused, or `None` when it certified."""


def certify_outer_coefficient(
    *,
    func: Callable[..., object],
    outer_action_name: str,
    abstract_args: Iterable[object],
    arg_names: tuple[str, ...],
) -> OuterAffineCertificate:
    """Return the exact coefficient of `outer_action_name` in `func`.

    Args:
        func: The traced outer post-decision map.
        outer_action_name: The argument whose coefficient is certified.
        abstract_args: Positional sample arguments to trace `func` against.
        arg_names: Positional names of `func`'s arguments, aligned with
            `abstract_args`, used to locate the outer action's input variable.

    Returns:
        An `OuterAffineCertificate` carrying either the exact coefficient or the
        reason the map was refused. A map that never reads the outer action
        certifies coefficient zero: whether a zero slope is usable belongs to the
        caller, and is a different complaint from an unreadable structure.
    """
    name = getattr(func, "__name__", "<unknown>")
    if outer_action_name not in arg_names:
        return OuterAffineCertificate(
            coefficient=None,
            violation=(
                f"{name!r} does not take the outer action {outer_action_name!r} "
                f"among its arguments {arg_names!r}, so no coefficient can be read."
            ),
        )
    try:
        closed = jax.make_jaxpr(func)(*abstract_args)
    except Exception as probe_error:  # noqa: BLE001 - any trace failure is unverified
        return OuterAffineCertificate(
            coefficient=None,
            violation=(
                f"{name!r} could not be traced on the build-time fills "
                f"({type(probe_error).__name__}: {probe_error}), so its dependence "
                "on the outer action is unverified."
            ),
        )

    position = arg_names.index(outer_action_name)
    jaxpr = closed.jaxpr
    if position >= len(jaxpr.invars):
        return OuterAffineCertificate(
            coefficient=None,
            violation=(
                f"{name!r} traced to {len(jaxpr.invars)} inputs, so the outer "
                f"action at position {position} could not be located."
            ),
        )

    seed: dict[object, Fraction] = {jaxpr.invars[position]: Fraction(1)}
    try:
        coefficients = _walk(jaxpr=jaxpr, tainted=seed, func_name=name)
    except _RefusedError as refusal:
        return OuterAffineCertificate(coefficient=None, violation=str(refusal))

    outvars = list(jaxpr.outvars)
    if len(outvars) != 1:
        return OuterAffineCertificate(
            coefficient=None,
            violation=(
                f"{name!r} returns {len(outvars)} values; the outer post-decision "
                "map must return exactly one."
            ),
        )
    return OuterAffineCertificate(
        coefficient=coefficients.get(outvars[0], Fraction(0)), violation=None
    )


def _coefficient_of(var: object, carried: Mapping[object, Fraction]) -> Fraction | None:
    """Return a variable's coefficient, treating a jaxpr `Literal` as untainted.

    Literals are unhashable, so they cannot be dict keys at all -- and a literal
    is a constant with respect to the outer action by definition.
    """
    if type(var).__name__ == "Literal":
        return None
    return carried.get(var)


class _RefusedError(Exception):
    """Raised when a primitive consumes the outer action without an exact rule."""


def _walk(
    *, jaxpr: object, tainted: Mapping[object, Fraction], func_name: str
) -> dict[object, Fraction]:
    """Propagate exact coefficients forward through one jaxpr's equations."""
    carried: dict[object, Fraction] = dict(tainted)

    for eqn in jaxpr.eqns:  # ty: ignore[unresolved-attribute]
        prim = eqn.primitive.name
        operands = [_coefficient_of(var, carried) for var in eqn.invars]
        if all(coefficient is None for coefficient in operands):
            # Nothing tainted flows in, so whatever this computes is a constant
            # with respect to the outer action -- however nonlinear it is.
            continue

        result = _apply(prim=prim, eqn=eqn, operands=operands, func_name=func_name)
        if result is None:
            raise _RefusedError(
                f"JAX primitive `{prim}` in {func_name!r} consumes the outer "
                "action, so the outer post-decision map is not affine in it. "
                "N-NB-EGM inverts that map exactly and cannot invert this one; "
                "reparametrize the law so the action enters affinely, or choose "
                "a solver that does not invert the outer margin."
            )
        for var in eqn.outvars:
            carried[var] = result
    return carried


def _apply(
    *,
    prim: str,
    eqn: object,
    operands: list[Fraction | None],
    func_name: str,
) -> Fraction | None:
    """Return the outgoing coefficient for one equation, or `None` to refuse."""
    if prim in _ADD_PRIMS:
        return sum((c for c in operands if c is not None), Fraction(0))

    if prim == "sub":
        left, right = operands[0] or Fraction(0), operands[1] or Fraction(0)
        return left - right

    if prim == "neg":
        return -(operands[0] or Fraction(0))

    if prim in _PASS_THROUGH_PRIMS:
        # `stop_gradient` belongs here deliberately: it changes what AD reports
        # and changes nothing about the value, so the term it hides still moves
        # the stock and still counts toward the coefficient.
        return operands[0]

    return _apply_structural(prim=prim, eqn=eqn, operands=operands, func_name=func_name)


def _apply_structural(
    *,
    prim: str,
    eqn: object,
    operands: list[Fraction | None],
    func_name: str,
) -> Fraction | None:
    """Handle the primitives whose rule needs the equation's own structure."""
    if prim in {"mul", "div"}:
        return _scale(prim=prim, eqn=eqn, operands=operands)
    if prim in {"jit", "pjit", "closed_call", "custom_jvp_call", "custom_vjp_call"}:
        return _descend(eqn=eqn, operands=operands, func_name=func_name)
    return None


def _scale(
    *, prim: str, eqn: object, operands: list[Fraction | None]
) -> Fraction | None:
    """Scale a tainted operand by a literal, refusing a non-literal factor.

    A factor that is itself a traced value -- a state, a parameter, another
    action -- makes the slope state-dependent, and a state-dependent slope has no
    single exact coefficient to certify.
    """
    tainted = [i for i, c in enumerate(operands) if c is not None]
    if len(tainted) != 1:
        return None
    position = tainted[0]
    coefficient = operands[position]
    # Dividing BY the action is not affine in it.
    if coefficient is None or (prim == "div" and position != 0):
        return None
    factor = _literal_factor(eqn=eqn, position=1 - position)
    if factor is None:
        return None
    if prim == "div":
        return None if factor == 0 else coefficient / factor
    return coefficient * factor


def _literal_factor(*, eqn: object, position: int) -> Fraction | None:
    """Return the exact rational value of an equation's literal operand."""
    literal = _literal_value(eqn.invars[position])  # ty: ignore[unresolved-attribute]
    if literal is None:
        return None
    try:
        return Fraction(literal)
    except TypeError, ValueError, OverflowError:
        return None


def _literal_value(var: object) -> float | int | None:
    """Return a jaxpr `Literal`'s Python value, or `None` if it is not one."""
    value = getattr(var, "val", None)
    if value is None:
        return None
    try:
        item = value.item() if hasattr(value, "item") else value
    except AttributeError, ValueError:
        return None
    return item if isinstance(item, float | int) else None


def _descend(
    *, eqn: object, operands: list[Fraction | None], func_name: str
) -> Fraction | None:
    """Walk a nested call's body, threading coefficients across its inputs."""
    body = None
    for key in _SUB_JAXPR_PARAM_KEYS:
        candidate = eqn.params.get(key)  # ty: ignore[unresolved-attribute]
        if candidate is not None:
            body = getattr(candidate, "jaxpr", candidate)
            break
    if body is None or not hasattr(body, "invars"):
        # A call whose body sits under a param key this walk does not know is
        # refused rather than treated as empty: an unwalked body is an unread one.
        return None

    inner_seed = {
        invar: coefficient
        for invar, coefficient in zip(body.invars, operands, strict=False)
        if coefficient is not None and type(invar).__name__ != "Literal"
    }
    inner = _walk(jaxpr=body, tainted=inner_seed, func_name=func_name)
    outvars = list(body.outvars)
    if len(outvars) != 1:
        return None
    return inner.get(outvars[0], Fraction(0))
