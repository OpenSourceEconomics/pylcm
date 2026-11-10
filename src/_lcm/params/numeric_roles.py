"""The numeric role of a parameter slot, which decides how its value is cast.

A slot's role comes from what consumes it, never from its value:

- `FLOAT`: a library-owned coefficient (a process law, a correlation, a discount
  factor) or a callback argument annotated as a float. An `int` value, scalar or
  array, is converted to the active float dtype.
- `INT`: a callback argument annotated as an integer — a category code, an index.
  Its value keeps its kind.
- `AS_GIVEN`: anything else, including an unannotated argument. Its value keeps its
  kind.

A Boolean value keeps its kind in every role.

A params-template leaf is a `TemplateAnnotation`: the display text a user sees in
`Model.get_params_template()`, carrying the role the boundary cast reads. Every
template producer builds its leaves through this module, and `template_roles`
refuses a leaf without a role, so no slot can fall back to a guess silently.
"""

import inspect
import types
import typing
from annotationlib import Format, ForwardRef
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Self, TypeAliasType, cast

import jaxtyping

import lcm.typing
from _lcm.typing import ParamsTemplate, QualifiedName
from _lcm.utils.namespace import flatten_regime_namespace
from lcm.exceptions import ModelInitializationError

# The dtype-name prefixes of each array role.
_FLOAT_DTYPE_PREFIXES = ("float", "bfloat")
_INT_DTYPE_PREFIXES = ("int", "uint")


class NumericRole(Enum):
    """How the boundary cast treats a parameter slot's value."""

    FLOAT = "float"
    INT = "int"
    AS_GIVEN = "as_given"

    @property
    def converts_to_float(self) -> bool:
        """Whether an integer value of this slot becomes the active float dtype."""
        return self is NumericRole.FLOAT


class TemplateAnnotation(str):  # noqa: SLOT000 - a str subclass holds no non-empty slots
    """A params-template leaf: the slot's display text and its numeric role."""

    role: NumericRole

    # keyword-only-exempt: primary-argument=text
    def __new__(cls, text: str, *, role: NumericRole) -> Self:
        leaf = super().__new__(cls, text)
        leaf.role = role
        return leaf

    def __getnewargs_ex__(self) -> tuple[tuple[str], dict[str, NumericRole]]:
        return (str(self),), {"role": self.role}


def role_of_annotation(annotation: object) -> NumericRole:  # noqa: PAN001 - any annotation object
    """Return the numeric role a resolved annotation declares.

    A union, `Annotated` metadata and `type` aliases are seen through. A union whose
    arms disagree, and anything that is not a float or integer kind, keeps the
    value's kind.
    """
    roles = {_arm_role(arm) for arm in _arms(annotation)}
    return roles.pop() if len(roles) == 1 else NumericRole.AS_GIVEN


def callable_param_roles(
    func: Callable[..., object],  # noqa: PAN001 - any user callable
) -> dict[str, NumericRole]:
    """Return the numeric role of each parameter of `func`, from its annotations.

    String annotations are resolved against the callable's module and the public
    `lcm.typing` aliases; one that does not resolve keeps the value's kind.
    """
    try:
        signature = inspect.signature(func, annotation_format=Format.FORWARDREF)
    except TypeError, ValueError:
        return {}
    namespace = {
        **vars(lcm.typing),
        **getattr(inspect.unwrap(func), "__globals__", {}),
    }
    return {
        name: role_of_annotation(
            _resolve(annotation=parameter.annotation, namespace=namespace)
        )
        for name, parameter in signature.parameters.items()
    }


def annotated_params(
    *, params: Mapping[str, str], roles: Mapping[str, NumericRole]
) -> dict[str, TemplateAnnotation]:
    """Attach each parameter's role to its display text."""
    return {
        name: TemplateAnnotation(text, role=roles.get(name, NumericRole.AS_GIVEN))
        for name, text in params.items()
    }


def merge_roles(*, name: str, left: NumericRole, right: NumericRole) -> NumericRole:
    """Combine two consumers' roles for one slot, in either order.

    A slot no consumer types keeps the value's kind; one typed by a single kind
    takes it. A slot read as a float by one consumer and as an integer by another
    has no single cast, so it is refused.
    """
    if left is right or right is NumericRole.AS_GIVEN:
        return left
    if left is NumericRole.AS_GIVEN:
        return right
    msg = (
        f"Parameter {name!r} is read as a float by one function and as an integer "
        "by another, so one value cannot serve both. Give the two readings "
        "distinct parameter names."
    )
    raise ModelInitializationError(msg)


def merge_template_params[Node](
    *, left: Mapping[str, Node], right: Mapping[str, Node]
) -> dict[str, Node]:
    """Union two consumers' parameters, merging the roles of a shared leaf.

    The display text of a shared leaf, and a shared branch, are `right`'s.
    """
    merged = dict(left)
    for name, node in right.items():
        previous = merged.get(name)
        if isinstance(previous, str) and isinstance(node, str):
            role = merge_roles(
                name=name, left=role_of_leaf(previous), right=role_of_leaf(node)
            )
            node = cast("Node", TemplateAnnotation(str(node), role=role))  # noqa: PLW2901
        merged[name] = node
    return merged


def role_of_leaf(leaf: str) -> NumericRole:
    """Return a template leaf's role; a plain string has no role to read."""
    if not isinstance(leaf, TemplateAnnotation):
        msg = f"Params-template leaf {leaf!r} carries no numeric role."
        raise TypeError(msg)
    return leaf.role


def template_roles(
    template: ParamsTemplate,
) -> types.MappingProxyType[QualifiedName, NumericRole]:
    """Return the role of every slot of a params template, keyed by qualified name."""
    return types.MappingProxyType(
        {
            qname: role_of_leaf(leaf)
            for qname, leaf in flatten_regime_namespace(template).items()
        }
    )


def _arms(annotation: object) -> tuple[object, ...]:  # noqa: PAN001 - any annotation object
    if isinstance(annotation, TypeAliasType):
        return _arms(annotation.__value__)
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _arms(typing.get_args(annotation)[0])
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        return tuple(
            arm
            for member in typing.get_args(annotation)
            if member is not type(None)
            for arm in _arms(member)
        )
    return (annotation,)


def _arm_role(arm: object) -> NumericRole:  # noqa: PAN001 - any annotation object
    if arm is float:
        return NumericRole.FLOAT
    if arm is int:
        return NumericRole.INT
    if isinstance(arm, type) and issubclass(arm, jaxtyping.AbstractArray):
        # `Shaped` admits any dtype and stores a sentinel, not a list of names.
        dtypes = arm.dtypes if isinstance(arm.dtypes, list | tuple) else ()
        if dtypes and all(
            _has_prefix(dtype=dtype, prefixes=_FLOAT_DTYPE_PREFIXES) for dtype in dtypes
        ):
            return NumericRole.FLOAT
        if dtypes and all(
            _has_prefix(dtype=dtype, prefixes=_INT_DTYPE_PREFIXES) for dtype in dtypes
        ):
            return NumericRole.INT
    return NumericRole.AS_GIVEN


def _has_prefix(*, dtype: object, prefixes: tuple[str, ...]) -> bool:  # noqa: PAN001 - a dtype name or pattern
    return isinstance(dtype, str) and dtype.startswith(prefixes)


def _resolve(
    *,
    annotation: object,  # noqa: PAN001 - any annotation object
    namespace: Mapping[str, object],  # noqa: PAN001 - a module namespace
) -> object:  # noqa: PAN001 - any annotation object
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)
    if isinstance(annotation, ForwardRef):
        try:
            return annotation.evaluate(globals=dict(namespace))
        except Exception:  # noqa: BLE001 - an unresolvable annotation types nothing
            return inspect.Parameter.empty
    return annotation
