"""Bounded library contracts for GETTSIM's generated JAX callable graphs.

The semantic walk separates installed implementations from constructed model data:

- Reviewed exports of dags, flatten-dict, ttsim and NumPy-groupies are sealed by their
  executable code, defaults and operation-specific implementation dependencies.
  Every required distribution must be installed at its reviewed non-editable release.
  Aggregation covers the selected NumPy-groupies backend and its compiler libraries;
  time conversion covers Pint's interpretation of the captured period factors.
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
from types import MappingProxyType
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
            _require_operation_versions(operation="ttsim-JAX-parameter"),
        )
    return None


def external_backend_binding(value: object) -> tuple[object, ...] | None:
    """Identify JAX bound to a reviewed TTSIM policy declaration or its forwarder.

    ttsim binds its backend into the final column callable: the policy declaration
    itself, the generated typed forwarder of a rounded declaration, or the wrapper that
    broadcasts scalar columns for a declaration that is not auto-vectorized.
    """
    if (
        not isinstance(value, partial)
        or value.keywords is None
        or value.keywords.get("xnp") is not jnp
    ):
        return None
    contract = _ttsim_contract_for(value.func)
    if (
        (contract is None or type(value.func) is not contract.policy_function)
        and external_typed_forwarder(value.func) is None
        and not _is_broadcast_wrapper(value.func)
    ):
        return None
    return (
        "jax.numpy",
        jax.__version__,
        _require_operation_versions(operation="ttsim-JAX-backend"),
    )


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


def external_unit_reference(*, value: object, path: tuple[str, ...]) -> object | None:
    """Resolve a unit spelled off ttsim's `TTSIMUnit` builder namespace.

    Policy code spells units as attribute chains such as
    `TTSIMUnit.CURRENCY.PER_MONTH`. Each step is an installed builder attribute that
    returns a new unit declaration, so the spelled unit is bound by value.
    """
    if not isinstance(value, type) or not path:
        return None
    contract = _ttsim_contract_for(value)
    if contract is None or value is not contract.unit_namespace:
        return None
    _require_supported_version(distribution="ttsim-backend", version=contract.version)
    composite = contract.units[0]
    unit = inspect.getattr_static(value, path[0], None)
    for name in path[1:]:
        if type(unit) is not composite:
            return None
        unit = getattr(unit, name)
    return unit if type(unit) is composite else None


def external_annotation_record(annotation: object) -> tuple[object, ...] | None:
    """Identify ttsim's validator-carrying column alias by identity.

    `DatetimeColumn` pairs a column type with an opaque beartype predicate on the
    dtype. The alias is bound by name and the backend version instead.
    """
    typing_module = sys.modules.get("ttsim.typing")
    if typing_module is None or annotation is not vars(typing_module).get(
        "DatetimeColumn"
    ):
        return None
    contract = _capture_ttsim_contract()
    if contract is None:
        return None
    _require_supported_version(distribution="ttsim-backend", version=contract.version)
    return "ttsim-annotation", contract.version, "ttsim.typing.DatetimeColumn"


def external_policy_record_version(value: object) -> str | None:
    """Recognize frozen records sourced from the installed GETTSIM policy tree.

    GETTSIM registers policy modules under their canonical name and re-executes them
    on every environment load. Resolve the class in its registered module, or its
    exact reload there, and check that module's installed source path.
    """
    if isinstance(value, type) or not dataclasses.is_dataclass(value):
        return None
    value_type = type(value)
    if (
        not value_type.__module__.startswith("gettsim.germany.")
        or not cast("Any", value_type).__dataclass_params__.frozen
    ):
        return None
    origin = _resolved_source(value=value_type)
    if origin is None:
        origin = _reloaded_record_source(value_type)
    if origin is None:
        return None
    root, version = _installed_package(
        distribution_name="gettsim", package="gettsim/germany"
    )
    if root is None:
        _require_supported_version(distribution="gettsim", version=version)
        return None
    if not origin.is_relative_to(root):
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
    root, version = _installed_package(distribution_name=distribution, package=package)
    if source is None or pathlib.Path(value.__code__.co_filename).resolve() != source:
        return None
    if root is None:
        _require_supported_version(distribution=distribution, version=version)
        return None
    if not source.is_relative_to(root):
        return None
    operation = f"{value.__module__}.{value.__qualname__}"
    if operation not in _SUPPORTED_OPERATIONS:
        raise TypeError(f"Unreviewed external operation {operation}.")
    return _require_operation_versions(operation=operation)


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
    if version == "absent":
        raise TypeError(
            f"Missing {distribution} installation required for durable identity."
        )
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


def _is_broadcast_wrapper(value: object) -> bool:
    """Whether a signature wrapper forwards to ttsim's scalar-column broadcaster."""
    wrapped = getattr(value, "__dict__", {}).get("__wrapped__")
    contract = (
        _capture_ttsim_contract() if isinstance(wrapped, types.FunctionType) else None
    )
    return contract is not None and any(
        _normalized_code(cast("types.FunctionType", wrapped).__code__) == code
        for code in contract.broadcast_code
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
    unit_namespace: type
    """Builder namespace from which policy code spells unit declarations."""
    parameters: tuple[tuple[type, tuple[str, ...], str], ...]
    """Array carriers, their fields and executable operation names."""
    rounding_code: tuple[types.CodeType, ...]
    """Normalized bodies of generated rounding wrappers."""
    broadcast_code: tuple[types.CodeType, ...]
    """Normalized bodies of generated scalar-column broadcast wrappers."""


@cache
def _capture_ttsim_contract() -> _TTSimContract | None:
    """Capture declarations without making the backend a required dependency."""
    root, version = _installed_package(
        distribution_name="ttsim-backend", package="ttsim"
    )
    if root is None:
        if "ttsim" in sys.modules:
            _require_supported_version(distribution="ttsim-backend", version=version)
        return None
    _require_supported_version(distribution="ttsim-backend", version=version)
    columns = importlib.import_module("ttsim.tt.column_objects_param_function")
    params = importlib.import_module("ttsim.tt.param_objects")
    rounding = importlib.import_module("ttsim.tt.rounding")
    environment = importlib.import_module(
        "ttsim.interface_dag_elements.specialized_environment"
    )
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
        unit_namespace=units.TTSIMUnit,
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
        broadcast_code=tuple(
            _normalized_code(code)
            for code in cast(
                "types.FunctionType",
                unwrap_beartype_guard(
                    environment._broadcast_scalar_columns_at_call_time  # noqa: SLF001
                ),
            ).__code__.co_consts
            if isinstance(code, types.CodeType)
            and code.co_name == "broadcast_scalar_columns"
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


def _reloaded_record_source(record_type: type) -> pathlib.Path | None:
    """Locate a record class whose defining module GETTSIM has since re-executed.

    A class from an earlier environment load belongs to a replaced module object. It
    is accepted only if the class of the same name in the registered module is its
    exact reload: the same class chain by name, attribute names, dataclass fields and
    function code.
    """
    module = sys.modules.get(record_type.__module__)
    current: object = module
    for name in record_type.__qualname__.split("."):
        current = inspect.getattr_static(current, name, None)
    if not isinstance(current, type) or not _is_exact_reload(
        stale=record_type, current=current
    ):
        return None
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    return pathlib.Path(origin).resolve() if isinstance(origin, str) else None


def _is_exact_reload(*, stale: type, current: type) -> bool:
    """Whether two classes are the same source executed twice."""
    if len(stale.__mro__) != len(current.__mro__):
        return False
    for old, new in zip(stale.__mro__, current.__mro__, strict=True):
        if old is new:
            continue
        old_state, new_state = vars(old), vars(new)
        if (
            (old.__module__, old.__qualname__) != (new.__module__, new.__qualname__)
            or old_state.keys() != new_state.keys()
            or _dataclass_field_spec(old) != _dataclass_field_spec(new)
        ):
            return False
        for name, member in old_state.items():
            other = new_state[name]
            if isinstance(member, types.FunctionType) != isinstance(
                other, types.FunctionType
            ):
                return False
            if isinstance(member, types.FunctionType) and _normalized_code(
                member.__code__
            ) != _normalized_code(cast("types.FunctionType", other).__code__):
                return False
    return True


def _dataclass_field_spec(cls: type) -> tuple[tuple[str, str], ...] | None:
    """Field names and declared types of a dataclass, else `None`."""
    if not dataclasses.is_dataclass(cls):
        return None
    return tuple((field.name, str(field.type)) for field in dataclasses.fields(cls))


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
def _require_operation_versions(*, operation: str) -> tuple[tuple[str, str], ...]:
    """Admit and seal only the dependencies of a reviewed implementation.

    Aggregation exports include both backend branches. NumPy-groupies selects
    its Numba implementation at import when available, so that selection also
    requires the reviewed compiler releases. Its NumPy fallback keeps Numba
    optional. Generated code and captured model data retain the recursive walk.
    """
    names = (
        _OPERATION_DISTRIBUTIONS[operation]
        if operation in _OPERATION_DISTRIBUTIONS
        else _ADAPTER_DISTRIBUTIONS[operation]
    )
    if "numpy-groupies" in names:
        version = _installed_package(distribution_name="numpy-groupies", package="")[1]
        _require_supported_version(distribution="numpy-groupies", version=version)
        groupies = importlib.import_module("numpy_groupies")
        selected = getattr(groupies.aggregate, "__module__", None)
        if selected == "numpy_groupies.aggregate_numba":
            names = (*names, "numba", "llvmlite")
        elif selected != "numpy_groupies.aggregate_numpy":
            raise TypeError(
                "Unsupported numpy-groupies implementation for durable identity."
            )
    versions = tuple(
        (name, _installed_package(distribution_name=name, package="")[1])
        for name in names
    )
    for name, version in versions:
        _require_supported_version(distribution=name, version=version)
    return versions


_INFRASTRUCTURE_PACKAGES = {
    "dags": "dags",
    "flatten_dict": "flatten-dict",
    "ttsim": "ttsim-backend",
    "numpy_groupies": "numpy-groupies",
}
_SUPPORTED_VERSIONS = MappingProxyType(
    {
        "dags": "0.6.0",
        "flatten-dict": "0.5.0",
        "gettsim": "1.3.1",
        "numpy-groupies": "0.12.3",
        "ttsim-backend": "1.3.2",
        "numpy": "2.4.6",
        "jax": "0.11.2",
        "jaxlib": "0.11.2",
        "numba": "0.68.0",
        "llvmlite": "0.50.0",
        "pint": "0.26.1",
    }
)
_NUMERICAL_BACKEND_DISTRIBUTIONS = ("ttsim-backend", "numpy", "jax", "jaxlib")
_OPERATION_DISTRIBUTIONS = MappingProxyType(
    {
        "dags.tree.tree_utils.flatten_to_qnames": ("dags", "flatten-dict"),
        **{
            f"dags.signature.{name}": ("dags",)
            for name in (
                "_fail_if_too_many_positional_arguments",
                "_fail_if_duplicated_arguments",
                "_fail_if_invalid_keyword_arguments",
                "_fail_if_missing_arguments",
            )
        },
        **{
            f"ttsim.tt.aggregation.{name}": (
                *_NUMERICAL_BACKEND_DISTRIBUTIONS,
                "numpy-groupies",
            )
            for name in (
                "grouped_any",
                "grouped_count",
                "grouped_min",
                "grouped_sum",
                "sum_by_p_id",
            )
        },
        "ttsim.tt.column_objects_param_function.ColumnFunction.__call__": (
            "ttsim-backend",
        ),
        "ttsim.tt.column_objects_param_function.reorder_ids": (
            _NUMERICAL_BACKEND_DISTRIBUTIONS
        ),
        "ttsim.tt.piecewise_polynomial.piecewise_polynomial": (
            _NUMERICAL_BACKEND_DISTRIBUTIONS
        ),
        "ttsim.tt.shared.join": _NUMERICAL_BACKEND_DISTRIBUTIONS,
        **{
            f"ttsim.time_converters.{name}": ("ttsim-backend", "pint")
            for name in (
                "m_to_y",
                "per_m_to_per_y",
                "per_w_to_per_y",
                "per_y_to_per_m",
                "y_to_m",
            )
        },
        "ttsim.tt.units.cast_ttsim_unit": ("ttsim-backend",),
    }
)
_SUPPORTED_OPERATIONS = frozenset(_OPERATION_DISTRIBUTIONS)
_ADAPTER_DISTRIBUTIONS = MappingProxyType(
    {
        "ttsim-JAX-parameter": _NUMERICAL_BACKEND_DISTRIBUTIONS,
        "ttsim-JAX-backend": _NUMERICAL_BACKEND_DISTRIBUTIONS,
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
