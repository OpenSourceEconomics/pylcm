"""Authenticate the full unbound scalar log-utility graph used by Case R."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var

from _lcm.typing import EconFunctionArg, EconFunctionKwargs
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import (
    ActionName,
    FloatND,
    ReferenceName,
    ScalarBool,
    ScalarFloat,
    ValueND,
)


@dataclass(frozen=True, kw_only=True)
class LogUtilityProgram:
    """Canonical original callable and authenticated action-independent graph.

    The structural walk admits one unmodified log(action), scaled/shifted by
    action-independent operands. Coefficients are read from this invocation's
    original pool; they never replace the callable used for candidate scores.
    """

    function: Callable[..., FloatND]
    """Canonical score function, retained without algebraic replacement."""
    action_name: ActionName
    """Original scalar consumption/action input."""
    names: tuple[ReferenceName, ...]
    """Unbound argument order, including context and runtime parameters."""
    closed: ClosedJaxpr
    """Authenticated unbound graph; invocation operands remain dynamic."""

    @classmethod
    def authenticate(
        cls,
        *,
        function: Callable[..., FloatND],
        action_name: ActionName,
        abstract_pool: Mapping[ReferenceName, jax.ShapeDtypeStruct | EconFunctionArg],
    ) -> LogUtilityProgram:
        """Refuse unproved action primitives, branches and nested utility classes."""
        names = tuple(inspect.signature(function).parameters)
        if action_name not in names:
            raise UnsupportedOperationError(
                "Balanced Case-R utility must read its action."
            )

        def original(*values: FloatND) -> FloatND:
            return function(**dict(zip(names, values, strict=True)))

        arguments = tuple(abstract_pool[name] for name in names)
        if any(
            not isinstance(value, (jax.Array, jax.ShapeDtypeStruct))
            or value.shape != ()
            for value in arguments
        ):
            raise UnsupportedOperationError(
                "Balanced log utility requires actual scalar operand schemas."
            )
        closed = jax.make_jaxpr(original)(*arguments)
        if (
            len(closed.jaxpr.outvars) != 1
            or getattr(closed.jaxpr.outvars[0].aval, "shape", None) != ()
        ):
            raise UnsupportedOperationError(
                "Balanced utility requires its original scalar output."
            )
        marks, logs = _authenticate_log_graph(
            jaxpr=closed.jaxpr,
            inputs=tuple(int(name == action_name) for name in names),
        )
        if marks not in {(2,), (3,)} or logs != 1:
            raise UnsupportedOperationError(
                "Balanced Case-R requires one authenticated affine log(action) graph."
            )
        return cls(
            function=function, action_name=action_name, names=names, closed=closed
        )

    def coefficients(
        self, *, pool: EconFunctionKwargs, dtype: jnp.dtype
    ) -> tuple[ScalarFloat, ScalarFloat, ScalarBool]:
        """Read actual action-independent scale/shift operands without AD.

        The graph interpreter propagates the admitted single log basis and
        evaluates every action-independent primitive in its original order.
        A scale is taken directly from that subgraph, rather than estimated
        from samples or differentiated through a rounded interpolation.
        """
        one = jnp.ones((), dtype=dtype)
        inputs = tuple(
            _LogOperand(
                value=one
                if name == self.action_name
                else _scalar_utility_operand(value=pool[name]),
                mark=int(name == self.action_name),
                scale=jnp.zeros_like(one),
            )
            for name in self.names
        )
        output = _coefficient_outputs(
            jaxpr=self.closed.jaxpr, consts=tuple(self.closed.consts), inputs=inputs
        )[0]
        scale, shift = output.scale, output.value
        return scale, shift, jnp.isfinite(scale) & (scale > 0.0) & jnp.isfinite(shift)


@dataclass(frozen=True, kw_only=True)
class _LogOperand:
    value: ValueND
    """Original independent value or shift of the admitted log basis."""
    mark: int
    """Independent, raw action, unit log or scaled-log structural class."""
    scale: FloatND
    """Actual independent multiplier of the unique log basis."""


def _coefficient_outputs(
    *, jaxpr: Jaxpr, consts: tuple[ValueND, ...], inputs: tuple[_LogOperand, ...]
) -> tuple[_LogOperand, ...]:
    values = dict(zip(jaxpr.invars, inputs, strict=True))
    zero = jnp.zeros_like(inputs[0].scale)
    for var, value in zip(jaxpr.constvars, consts, strict=True):
        values[var] = _LogOperand(value=value, mark=0, scale=zero)

    def read(var: Var | Literal) -> _LogOperand:
        return (
            _LogOperand(value=jnp.asarray(var.val), mark=0, scale=zero)
            if isinstance(var, Literal)
            else values[var]
        )

    for equation in jaxpr.eqns:
        args = tuple(read(var) for var in equation.invars)
        outputs = _coefficient_equation(equation=equation, args=args, zero=zero)
        values.update(zip(equation.outvars, outputs, strict=True))
    return tuple(read(var) for var in jaxpr.outvars)


def _coefficient_equation(
    *, equation: JaxprEqn, args: tuple[_LogOperand, ...], zero: FloatND
) -> tuple[_LogOperand, ...]:
    marks = tuple(arg.mark for arg in args)
    name = equation.primitive.name
    nested = tuple(
        equation.params[key]
        for key in ("jaxpr", "call_jaxpr", "fun_jaxpr")
        if key in equation.params
    )
    if any(marks) and nested:
        graph = nested[0]
        if isinstance(graph, ClosedJaxpr):
            return _coefficient_outputs(
                jaxpr=graph.jaxpr, consts=tuple(graph.consts), inputs=args
            )
        if isinstance(graph, Jaxpr):
            return _coefficient_outputs(jaxpr=graph, consts=(), inputs=args)
        raise UnsupportedOperationError(
            "Balanced utility lost its authenticated nested graph."
        )
    if name == "log" and marks == (1,):
        return (_LogOperand(value=zero, mark=2, scale=jnp.ones_like(zero)),)
    result = equation.primitive.bind(*(arg.value for arg in args), **equation.params)
    numerical = cast(
        "tuple[ValueND,...]",
        tuple(result) if equation.primitive.multiple_results else (result,),
    )
    if not any(marks):
        return tuple(
            _LogOperand(value=value, mark=0, scale=zero) for value in numerical
        )
    return (_affine_log_output(equation=equation, args=args, value=numerical[0]),)


def _affine_log_output(
    *, equation: JaxprEqn, args: tuple[_LogOperand, ...], value: ValueND
) -> _LogOperand:
    name = equation.primitive.name
    if name in {"copy", "reshape", "squeeze", "broadcast_in_dim"}:
        return _LogOperand(
            value=value,
            mark=args[0].mark,
            scale=equation.primitive.bind(args[0].scale, **equation.params),
        )
    if name == "neg":
        return _LogOperand(value=value, mark=3, scale=-args[0].scale)
    index = 0 if args[0].mark else 1
    if name in {"add", "add_any", "sub"}:
        negate = name == "sub" and index == 1
        return _LogOperand(
            value=value,
            mark=3 if negate else args[index].mark,
            scale=-args[index].scale if negate else args[index].scale,
        )
    if name == "mul":
        return _LogOperand(value=value, mark=3, scale=args[1 - index].value)
    raise UnsupportedOperationError(
        "Balanced coefficient reader disagrees with its proof."
    )


def _authenticate_log_graph(
    *, jaxpr: Jaxpr, inputs: tuple[int, ...]
) -> tuple[tuple[int, ...], int]:
    if jaxpr.effects:
        raise UnsupportedOperationError(
            "Balanced utility requires an effect-free graph."
        )
    marks: dict[Var, int] = dict(zip(jaxpr.invars, inputs, strict=True))
    logs = 0
    for equation in jaxpr.eqns:
        args = tuple(
            0 if isinstance(var, Literal) else marks.get(var, 0)
            for var in equation.invars
        )
        name = equation.primitive.name
        nested = tuple(
            equation.params[key]
            for key in ("jaxpr", "call_jaxpr", "fun_jaxpr")
            if key in equation.params
        )
        if not any(args):
            outputs = (0,) * len(equation.outvars)
        elif nested:
            if len(nested) != 1:
                raise UnsupportedOperationError(
                    "Balanced log utility has an unsupported nested call."
                )
            graph = nested[0]
            if isinstance(graph, ClosedJaxpr):
                graph = graph.jaxpr
            if not isinstance(graph, Jaxpr):
                raise UnsupportedOperationError(
                    "Balanced log utility has no inspectable nested graph."
                )
            outputs, inner_logs = _authenticate_log_graph(jaxpr=graph, inputs=args)
            logs += inner_logs
        else:
            outputs = _log_primitive_marks(name=name, args=args)
            logs += int(name == "log")
        marks.update(zip(equation.outvars, outputs, strict=True))
    return tuple(
        0 if isinstance(var, Literal) else marks.get(var, 0) for var in jaxpr.outvars
    ), logs


def _log_primitive_marks(*, name: str, args: tuple[int, ...]) -> tuple[int, ...]:
    if name == "log" and args == (1,):
        return (2,)
    if name in {"copy", "reshape", "squeeze", "broadcast_in_dim"} and len(args) == 1:
        return args
    if name == "neg" and args in {(2,), (3,)}:
        return (3,)
    if name in {"add", "add_any", "sub"} and sorted(args) in ([0, 2], [0, 3]):
        return (3 if name == "sub" and args[0] == 0 else max(args),)
    if name == "mul" and sorted(args) == [0, 2]:
        return (3,)
    raise UnsupportedOperationError(
        f"Balanced log utility refuses action-dependent primitive '{name}'."
    )


def _scalar_utility_operand(*, value: EconFunctionArg) -> ValueND:
    if not isinstance(value, jax.Array) or value.ndim != 0:
        raise UnsupportedOperationError(
            "Balanced log utility requires original scalar array operands."
        )
    return value
