"""Bounded library contracts for GETTSIM's generated JAX callable graphs.

The semantic walk separates installed implementations from constructed model data:

- Reviewed exports of dags, flatten-dict, ttsim and NumPy-groupies are sealed by their
  executable code, defaults and the versions of both numerical backend stacks and
  of pint, from which ttsim derives its time-conversion factors.
  A function must resolve by identity at its installed source location and carry
  no closure or instance state beyond typing caches. Generated functions and
  GETTSIM policy bodies retain the full recursive walk.
- ttsim guards its functions with an import-time beartype claw by default. A guard
  of a reviewed export is transparent only if beartype regenerates it.
- Exact ttsim column and unit declarations and installed GETTSIM frozen policy
  records bind every field. The foreign-key, aggregation and quantity-kind enums
  bind their complete member schema and selected member.
- ttsim's generated column-typed forwarder must match its one-line template; it is
  identified by its signature and the callable it forwards to. Its beartype guard
  must regenerate from the hints it recorded.
- Exact JAX lookup/polynomial parameter carriers bind their arrays and operation
  code. JAX NumPy has a versioned meaning only in a reviewed TTSIM backend binding.
  Other module-valued data, custom classes, carrier subclasses and opaque/mutable
  parameter objects stay closed.
- The exact rounding-wrapper body may carry copied introspection metadata from a
  column declaration. That metadata is represented by the wrapped declaration;
  the wrapper's executable body, real closure and rounding inputs remain hashed.

The implementation seal assumes unmodified installed libraries, as do the JAX
primitives. Runtime library monkeypatching and in-place mutation of constructed
model data are outside the contract. No library name or repr alone admits an object.
GETTSIM and ttsim remain optional dependencies.
"""

import dataclasses
import importlib
import importlib.metadata
import inspect
import json
import pathlib
import sys
import types
from enum import Enum
from functools import cache, partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from beartype import BeartypeConf, beartype


def external_enum_record(value: object) -> tuple[object, ...] | None:
    """Return the versioned schema and value of a genuine reviewed ttsim enum member."""
    contract = _ttsim_contract_for(value)
    if contract is None:
        return None
    schema = next(
        (members for enum_type, members in contract.enums if type(value) is enum_type),
        None,
    )
    if schema is None:
        return None
    _require_supported_version(distribution="ttsim-backend", version=contract.version)
    member = cast("Enum", value)
    enum_members = type(member).__members__
    if (
        any(
            type(item.value) not in (str, int)
            or set(vars(item)) != {"_value_", "_name_", "__objclass__", "_sort_order_"}
            for item in enum_members.values()
        )
        or tuple((name, item.value) for name, item in enum_members.items()) != schema
    ):
        raise TypeError("Cannot durably fingerprint a modified ttsim enum declaration.")
    return (
        "ttsim-enum",
        contract.version,
        type(member).__qualname__,
        schema,
        member.name,
        member.value,
    )


def external_parameter_record(value: object) -> tuple[object, ...] | None:
    """Project exact ttsim parameter carriers into closed data."""
    contract = _ttsim_contract_for(value)
    if contract is None:
        return None
    for value_type, fields, operation_name in contract.parameters:
        if type(value) is not value_type:
            continue
        _require_supported_version(
            distribution="ttsim-backend", version=contract.version
        )
        if operation_name == "look_up":
            if cast("Any", value).xnp is not jnp or tuple(
                cast("Any", value_type).__slots__
            ) != (*fields, "xnp"):
                raise TypeError(
                    "Cannot durably fingerprint a non-JAX ttsim lookup table."
                )
        elif set(vars(value)) != set(fields):
            raise TypeError("Cannot durably fingerprint extra ttsim parameter state.")
        arrays = tuple(getattr(value, name) for name in fields)
        if any(not isinstance(array, jax.Array) for array in arrays):
            raise TypeError(
                "Cannot durably fingerprint non-array ttsim parameter data."
            )
        operation = unwrap_beartype_guard(
            inspect.getattr_static(value_type, operation_name)
        )
        if not isinstance(operation, types.FunctionType):
            raise TypeError(
                "Cannot durably fingerprint opaque ttsim parameter behavior."
            )
        return (
            "ttsim-JAX-parameter",
            contract.version,
            value_type.__qualname__,
            fields,
            arrays,
            operation.__code__,
            jax.__version__,
        )
    return None


def external_backend_binding(value: object) -> tuple[str, str] | None:
    """Identify JAX bound to a reviewed TTSIM policy declaration."""
    if (
        not isinstance(value, partial)
        or value.keywords is None
        or value.keywords.get("xnp") is not jnp
    ):
        return None
    contract = _ttsim_contract_for(value.func)
    if contract is None or type(value.func) is not contract.policy_function:
        return None
    return "jax.numpy", jax.__version__


def is_external_declaration(value: object) -> bool:
    """Admit genuine frozen library declarations to the full field traversal."""
    if external_policy_record_version(value) is not None:
        metadata = frozenset()
    elif (contract := _ttsim_contract_for(value)) is not None and any(
        type(value) is cls for cls in (*contract.columns, *contract.units)
    ):
        _require_supported_version(
            distribution="ttsim-backend", version=contract.version
        )
        metadata = (
            _COLUMN_METADATA | _copied_function_marks(value)
            if type(value) in contract.columns
            else frozenset()
        )
    else:
        return False
    fields = {field.name for field in dataclasses.fields(cast("Any", value))}
    if extra := set(vars(value)) - fields - metadata:
        raise TypeError(
            "Cannot durably fingerprint extra external declaration state "
            f"{sorted(extra)}."
        )
    return True


def external_wrapper_metadata(function: types.FunctionType) -> frozenset[str]:
    """Identify redundant copied fields on the exact ttsim rounding wrapper body."""
    if function.__code__.co_name != "wrapper":
        return frozenset()
    wrapped = function.__dict__.get("__wrapped__")
    contract = _ttsim_contract_for(wrapped)
    if contract is None:
        return frozenset()
    if not any(
        _normalized_code(function.__code__) == code for code in contract.rounding_code
    ):
        return frozenset()
    _require_supported_version(distribution="ttsim-backend", version=contract.version)
    if not is_external_declaration(wrapped):
        return frozenset()
    names = frozenset({"__code__", "__closure__", "__globals__"})
    for name in names & function.__dict__.keys():
        if function.__dict__[name] is not vars(wrapped).get(name):
            raise TypeError("Cannot durably fingerprint modified rounding metadata.")
    return names


def external_typed_forwarder(
    value: object,
) -> tuple[types.FunctionType, object] | None:
    """Recognize ttsim's synthesized column-typed forwarder and its callee.

    ttsim wraps vectorized and rounded policy callables in a generated function whose
    body only forwards its parameters to `_ttsim_wrapped_impl`, then guards it with
    beartype and resets its annotations. The guard is transparent only if beartype
    regenerates its code from the annotations it recorded. The forwarder is
    identified by its signature and callee.
    """
    if not isinstance(value, types.FunctionType):
        return None
    guarded = value.__dict__.get("__beartype_wrapper") is True
    forwarder = value.__dict__.get("__wrapped__") if guarded else value
    if (
        not isinstance(forwarder, types.FunctionType)
        or forwarder.__module__ != "ttsim.typing"
        or forwarder.__code__.co_filename != _TYPED_FORWARDER_FILENAME
    ):
        return None
    contract = _capture_ttsim_contract()
    if contract is None:
        return None
    _require_supported_version(distribution="ttsim-backend", version=contract.version)
    if guarded:
        _fail_if_guard_is_not_regenerated(guard=value, callee=forwarder)
    code = forwarder.__code__
    parameters = ", ".join(code.co_varnames[: code.co_argcount])
    source = (
        f"def {code.co_name}({parameters}):\n"
        f"    return _ttsim_wrapped_impl({parameters})\n"
    )
    reference = next(
        constant
        for constant in compile(source, _TYPED_FORWARDER_FILENAME, "exec").co_consts
        if isinstance(constant, types.CodeType)
    )
    if (
        _normalized_code(code) != _normalized_code(reference)
        or forwarder.__closure__
        or forwarder.__defaults__
        or forwarder.__kwdefaults__
        or set(forwarder.__dict__) - _ANNOTATION_CACHES - {"__signature__"}
        or "_ttsim_wrapped_impl" not in forwarder.__globals__
    ):
        raise TypeError("Cannot durably fingerprint a modified ttsim typed forwarder.")
    return forwarder, forwarder.__globals__["_ttsim_wrapped_impl"]


def external_policy_record_version(value: object) -> str | None:
    """Recognize frozen records sourced from the installed GETTSIM policy tree.

    GETTSIM loads modules under domain aliases such as wohngeld.wohngeld. Resolve
    the actual class in its module and check that module's installed source path.
    """
    if isinstance(value, type) or not dataclasses.is_dataclass(value):
        return None
    value_type = type(value)
    if not cast("Any", value_type).__dataclass_params__.frozen:
        return None
    origin = _resolved_source(value=value_type)
    root, version = _installed_package(
        distribution_name="gettsim", package="gettsim/germany"
    )
    if origin is None or root is None or not origin.is_relative_to(root):
        return None
    _require_supported_version(distribution="gettsim", version=version)
    return version


def external_function_versions(value: object) -> tuple[tuple[str, str], ...] | None:
    """Seal a fixed export of the bounded graph-building infrastructure stack."""
    if not isinstance(value, types.FunctionType):
        return None
    package = value.__module__.split(".", 1)[0]
    distribution = _INFRASTRUCTURE_PACKAGES.get(package)
    if (
        distribution is None
        or value.__closure__
        or not _carries_only_annotation_caches(value)
    ):
        return None
    source = _resolved_source(value=value)
    root, _version = _installed_package(distribution_name=distribution, package=package)
    if source is None or root is None or not source.is_relative_to(root):
        return None
    if pathlib.Path(value.__code__.co_filename).resolve() != source:
        return None
    operation = f"{value.__module__}.{value.__qualname__}"
    if operation not in _SUPPORTED_OPERATIONS:
        raise TypeError(f"Unreviewed external operation {operation}.")
    _require_supported_version(
        distribution=distribution,
        version=_installed_package(distribution_name=distribution, package=package)[1],
    )
    return _infrastructure_versions()


def unwrap_beartype_guard(function: object) -> object:
    """Return the bound callee of an exact beartype guard, else the callable itself.

    ttsim guards its functions with an import-time beartype claw by default. The
    guard only checks argument types, so the reviewed implementation is its callee.
    """
    if not isinstance(function, types.FunctionType):
        return function
    state = function.__dict__
    wrapped = state.get("__wrapped__")
    if (
        state.get("__beartype_wrapper") is True
        and isinstance(wrapped, types.FunctionType)
        and (function.__kwdefaults__ or {}).get("__beartype_func") is wrapped
        and function.__code__.co_filename.startswith(_BEARTYPE_BODY_FILENAME_PREFIX)
    ):
        return wrapped
    return function


def _require_supported_version(*, distribution: str, version: str) -> None:
    """Reject integrations whose implementation contract has not been reviewed."""
    if version != _SUPPORTED_VERSIONS[distribution]:
        raise TypeError(
            f"Unsupported {distribution} version {version!r} for durable identity."
        )
    direct_url = importlib.metadata.distribution(distribution).read_text(
        "direct_url.json"
    )
    if direct_url is not None:
        try:
            installation = json.loads(direct_url)
        except json.JSONDecodeError as error:
            raise TypeError(
                f"Cannot inspect {distribution} installation for durable identity."
            ) from error
        if installation.get("dir_info", {}).get("editable") is True:
            raise TypeError(
                f"Editable {distribution} installation has no durable identity."
            )


def _fail_if_guard_is_not_regenerated(
    *, guard: types.FunctionType, callee: types.FunctionType
) -> None:
    """Require beartype to reproduce a guard from its callee and recorded hints."""
    state = guard.__dict__
    keyword_defaults = guard.__kwdefaults__
    conf = (
        keyword_defaults.get("__beartype_conf")
        if type(keyword_defaults) is dict
        else None
    )
    annotations = state.get("__beartype_annotations")
    if not (
        type(keyword_defaults) is dict
        and keyword_defaults.get("__beartype_func") is callee
        and isinstance(conf, BeartypeConf)
        and type(annotations) is dict
        and guard.__code__.co_filename.startswith(_BEARTYPE_BODY_FILENAME_PREFIX)
    ):
        raise TypeError("Cannot durably fingerprint an inexact ttsim beartype guard.")
    clone = types.FunctionType(callee.__code__, callee.__globals__, callee.__name__)
    clone.__qualname__ = callee.__qualname__
    clone.__module__ = callee.__module__
    clone.__annotations__ = dict(annotations)
    regenerated = beartype(conf=conf)(clone)
    code = guard.__code__
    if (
        not isinstance(regenerated, types.FunctionType)
        or regenerated.__code__.co_code != code.co_code
        or regenerated.__code__.co_consts != code.co_consts
        or regenerated.__code__.co_names != code.co_names
    ):
        raise TypeError(
            "Cannot durably fingerprint a ttsim beartype guard that beartype does not "
            "regenerate from its callee."
        )


def _carries_only_annotation_caches(function: types.FunctionType) -> bool:
    """Whether a function's own state is limited to typing and beartype caches."""
    state = function.__dict__
    return (
        set(state) <= _ANNOTATION_CACHES
        and state.get("__no_type_check__", True) is True
    )


def _copied_function_marks(declaration: object) -> frozenset[str]:
    """Name typing and beartype marks a declaration copied from its function field.

    ttsim copies the function's `__dict__` onto the declaration. The function field
    is hashed itself, so marks identical to its own add no state.
    """
    source = getattr(getattr(declaration, "function", None), "__dict__", {})
    state = vars(declaration)
    return frozenset(
        name
        for name in _COPIED_FUNCTION_MARKS
        if name in state and name in source and state[name] is source[name]
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TTSimContract:
    """Genuine optional backend declarations captured when an integration is used."""

    version: str
    """Installed backend release."""
    enums: tuple[tuple[type[Enum], tuple[tuple[str, object], ...]], ...]
    """Exact foreign-key, aggregation and quantity-kind enums with member schemas."""
    columns: tuple[type, ...]
    """Exact frozen column and rounding declaration types."""
    policy_function: type
    """Declaration type that may bind the JAX backend in a generated partial."""
    units: tuple[type, ...]
    """Exact frozen unit declaration types bound as fields of column declarations."""
    parameters: tuple[tuple[type, tuple[str, ...], str], ...]
    """Array carriers, their fields and executable operation names."""
    rounding_code: tuple[types.CodeType, ...]
    """Normalized bodies of generated rounding wrappers."""


@cache
def _capture_ttsim_contract() -> _TTSimContract | None:
    """Capture declarations without making the backend a required dependency."""
    root, version = _installed_package(
        distribution_name="ttsim-backend", package="ttsim"
    )
    if root is None:
        return None
    _require_supported_version(distribution="ttsim-backend", version=version)
    columns = importlib.import_module("ttsim.tt.column_objects_param_function")
    params = importlib.import_module("ttsim.tt.param_objects")
    rounding = importlib.import_module("ttsim.tt.rounding")
    units = importlib.import_module("ttsim.tt.units")
    return _TTSimContract(
        version=version,
        enums=tuple(
            (
                enum_type,
                tuple(
                    (name, member.value)
                    for name, member in enum_type.__members__.items()
                ),
            )
            for enum_type in (columns.FKType, columns.AggType, units.QuantityKind)
        ),
        columns=(
            columns.ColumnObject,
            columns.PolicyInput,
            columns.ColumnFunction,
            columns.PolicyFunction,
            columns.GroupCreationFunction,
            columns.AggByGroupFunction,
            columns.AggByPIDFunction,
            columns.TimeConversionFunction,
            rounding.RoundingSpec,
        ),
        policy_function=columns.PolicyFunction,
        units=(units.CompositeUnit, units.UnsetUnit, units.InputOutputUnits),
        parameters=(
            (
                params.ConsecutiveIntLookupTableParamValue,
                ("bases_to_subtract", "lookup_multipliers", "values_to_look_up"),
                "look_up",
            ),
            (
                params.PiecewisePolynomialParamValue,
                ("thresholds", "intercepts", "coefficients"),
                "__getitem__",
            ),
        ),
        rounding_code=tuple(
            _normalized_code(code)
            for code in cast(
                "types.FunctionType",
                unwrap_beartype_guard(rounding.RoundingSpec.apply_rounding),
            ).__code__.co_consts
            if isinstance(code, types.CodeType) and code.co_name == "wrapper"
        ),
    )


def _ttsim_contract_for(value: object) -> _TTSimContract | None:
    """Load optional TTSIM declarations only for values claiming that namespace."""
    if not type(value).__module__.startswith("ttsim."):
        return None
    return _capture_ttsim_contract()


def _resolved_source(*, value: type | types.FunctionType) -> pathlib.Path | None:
    """Verify module ownership and source location independently of aliases."""
    module = sys.modules.get(value.__module__)
    if not isinstance(module, types.ModuleType):
        return None
    resolved: object = module
    for name in value.__qualname__.split("."):
        resolved = inspect.getattr_static(resolved, name, None)
    if resolved is not value and unwrap_beartype_guard(resolved) is not value:
        return None
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    return pathlib.Path(origin).resolve() if isinstance(origin, str) else None


@cache
def _normalized_code(code: types.CodeType) -> types.CodeType:
    """Compare wrapper bodies across serialization and installation locations."""
    return code.replace(co_filename="", co_firstlineno=0, co_linetable=b"")


@cache
def _installed_package(
    *, distribution_name: str, package: str
) -> tuple[pathlib.Path | None, str]:
    """Locate an optional installed package independently of mutable module names."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None, "absent"
    return pathlib.Path(
        str(distribution.locate_file(package))
    ).resolve(), distribution.version


@cache
def _infrastructure_versions() -> tuple[tuple[str, str], ...]:
    """Seal graph construction and both numerical backend implementations."""
    names = (
        "dags",
        "flatten-dict",
        "ttsim-backend",
        "numpy-groupies",
        "numpy",
        "jax",
        "jaxlib",
        "numba",
        "llvmlite",
        "pint",
    )
    return tuple(
        (name, _installed_package(distribution_name=name, package="")[1])
        for name in names
    )


_INFRASTRUCTURE_PACKAGES = {
    "dags": "dags",
    "flatten_dict": "flatten-dict",
    "ttsim": "ttsim-backend",
    "numpy_groupies": "numpy-groupies",
}
_SUPPORTED_VERSIONS = {
    "dags": "0.6.0",
    "flatten-dict": "0.5.0",
    "gettsim": "1.3.1",
    "numpy-groupies": "0.11.3",
    "ttsim-backend": "1.3.2",
}
_SUPPORTED_OPERATIONS = frozenset(
    {
        "dags.tree.tree_utils.flatten_to_qnames",
        "ttsim.tt.aggregation.grouped_sum",
        "ttsim.tt.aggregation.sum_by_p_id",
        "ttsim.tt.column_objects_param_function.ColumnFunction.__call__",
        "ttsim.tt.piecewise_polynomial.piecewise_polynomial",
        "ttsim.time_converters.per_m_to_per_y",
        "ttsim.time_converters.per_y_to_per_m",
        "ttsim.tt.units.cast_ttsim_unit",
    }
)
_TYPED_FORWARDER_FILENAME = "<ttsim-typed-wrapper>"
_BEARTYPE_BODY_FILENAME_PREFIX = "<@beartype("
_ANNOTATION_CACHES = frozenset(
    {"__beartype_annotations", "__beartype_args_lens", "__no_type_check__"}
)
_COPIED_FUNCTION_MARKS = frozenset(
    {
        "__beartype_annotations",
        "__beartype_args_lens",
        "__beartype_wrapper",
        "__no_type_check__",
    }
)
_COLUMN_METADATA = frozenset(
    {
        "__signature__",
        "__globals__",
        "__closure__",
        "__code__",
        "__doc__",
        "__name__",
        "__QName__",
        "__module__",
        "__annotations__",
        "__type_params__",
        "__wrapped__",
        "__qualname__",
    }
)
