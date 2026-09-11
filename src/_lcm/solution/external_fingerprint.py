"""Bounded library contracts for GETTSIM's generated JAX callable graphs.

The semantic walk separates installed implementations from constructed model data:

- Fixed exports of dags, flatten-dict, ttsim and NumPy-groupies are sealed by their
  executable code, defaults and the versions of both numerical backend stacks.
  A function must resolve by identity at its installed source location and carry
  no closure or instance state. Generated functions and GETTSIM policy bodies
  retain the full recursive walk.
- Exact ttsim column declarations and installed GETTSIM frozen policy records
  bind every field. FKType binds its complete member schema and selected member.
- Exact JAX lookup/polynomial parameter carriers bind their arrays and operation
  code. JAX NumPy is a versioned backend token. Other module-valued data, custom
  classes, carrier subclasses and opaque/mutable parameter objects stay closed.
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
import pathlib
import sys
import types
from enum import Enum
from functools import cache
from typing import Any, cast

import jax
import jax.numpy as jnp


def external_enum_record(value: object) -> tuple[object, ...] | None:
    """Return the versioned schema and value of a genuine ttsim FKType member."""
    if _TTSIM is None or type(value) is not _TTSIM.enum_type:
        return None
    member = value
    enum_members = _TTSIM.enum_type.__members__
    if (
        any(
            type(item.value) is not str
            or set(vars(item)) != {"_value_", "_name_", "__objclass__", "_sort_order_"}
            for item in enum_members.values()
        )
        or tuple((name, item.value) for name, item in enum_members.items())
        != _TTSIM.members
    ):
        raise TypeError(
            "Cannot durably fingerprint a modified ttsim FKType declaration."
        )
    return "ttsim-FKType", _TTSIM.version, _TTSIM.members, member.name, member.value


def external_parameter_record(value: object) -> tuple[object, ...] | None:
    """Project exact JAX backend and ttsim parameter carriers into closed data."""
    if value is jnp:
        return "numerical-backend", "jax.numpy", jax.__version__
    if _TTSIM is None:
        return None
    for value_type, fields, operation_name in _TTSIM.parameters:
        if type(value) is not value_type:
            continue
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
        operation = inspect.getattr_static(value_type, operation_name)
        if not isinstance(operation, types.FunctionType):
            raise TypeError(
                "Cannot durably fingerprint opaque ttsim parameter behavior."
            )
        return (
            "ttsim-JAX-parameter",
            _TTSIM.version,
            value_type.__qualname__,
            fields,
            arrays,
            operation.__code__,
            jax.__version__,
        )
    return None


def is_external_declaration(value: object) -> bool:
    """Admit genuine frozen library declarations to the full field traversal."""
    if external_policy_record_version(value) is not None:
        metadata = frozenset()
    elif _TTSIM is not None and any(type(value) is cls for cls in _TTSIM.columns):
        metadata = _COLUMN_METADATA
    else:
        return False
    fields = {field.name for field in dataclasses.fields(cast("Any", value))}
    if set(vars(value)) - fields - metadata:
        raise TypeError("Cannot durably fingerprint extra external declaration state.")
    return True


def external_wrapper_metadata(function: types.FunctionType) -> frozenset[str]:
    """Identify redundant copied fields on the exact ttsim rounding wrapper body."""
    if _TTSIM is None or function.__code__.co_name != "wrapper":
        return frozenset()
    if not any(
        _normalized_code(function.__code__) == code for code in _TTSIM.rounding_code
    ):
        return frozenset()
    wrapped = function.__dict__.get("__wrapped__")
    if not is_external_declaration(wrapped):
        return frozenset()
    names = frozenset({"__code__", "__closure__", "__globals__"})
    for name in names & function.__dict__.keys():
        if function.__dict__[name] is not vars(wrapped).get(name):
            raise TypeError("Cannot durably fingerprint modified rounding metadata.")
    return names


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
    return version


def external_function_versions(value: object) -> tuple[tuple[str, str], ...] | None:
    """Seal a fixed export of the bounded graph-building infrastructure stack."""
    if not isinstance(value, types.FunctionType):
        return None
    package = value.__module__.split(".", 1)[0]
    distribution = _INFRASTRUCTURE_PACKAGES.get(package)
    if distribution is None or value.__closure__ or value.__dict__:
        return None
    source = _resolved_source(value=value)
    root, _version = _installed_package(distribution_name=distribution, package=package)
    if source is None or root is None or not source.is_relative_to(root):
        return None
    if pathlib.Path(value.__code__.co_filename).resolve() != source:
        return None
    return _infrastructure_versions()


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TTSimContract:
    """Genuine optional backend declarations captured at import."""

    version: str
    """Installed backend release."""
    enum_type: type[Enum]
    """Exact foreign-key enum class."""
    members: tuple[tuple[str, str], ...]
    """Ordered foreign-key member schema."""
    columns: tuple[type, ...]
    """Exact frozen column and rounding declaration types."""
    parameters: tuple[tuple[type, tuple[str, ...], str], ...]
    """Array carriers, their fields and executable operation names."""
    rounding_code: tuple[types.CodeType, ...]
    """Normalized bodies of generated rounding wrappers."""


def _capture_ttsim_contract() -> _TTSimContract | None:
    """Capture declarations without making the backend a required dependency."""
    root, version = _installed_package(
        distribution_name="ttsim-backend", package="ttsim"
    )
    if root is None:
        return None
    columns = importlib.import_module("ttsim.tt.column_objects_param_function")
    params = importlib.import_module("ttsim.tt.param_objects")
    rounding = importlib.import_module("ttsim.tt.rounding")
    return _TTSimContract(
        version=version,
        enum_type=columns.FKType,
        members=tuple(
            (name, member.value) for name, member in columns.FKType.__members__.items()
        ),
        columns=(
            *(
                value
                for value in vars(columns).values()
                if isinstance(value, type)
                and issubclass(value, columns.ColumnObject)
                and dataclasses.is_dataclass(value)
                and value.__dataclass_params__.frozen
            ),
            rounding.RoundingSpec,
        ),
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
            for code in rounding.RoundingSpec.apply_rounding.__code__.co_consts
            if isinstance(code, types.CodeType) and code.co_name == "wrapper"
        ),
    )


def _resolved_source(*, value: type | types.FunctionType) -> pathlib.Path | None:
    """Verify module ownership and source location independently of aliases."""
    module = sys.modules.get(value.__module__)
    if not isinstance(module, types.ModuleType):
        return None
    resolved: object = module
    for name in value.__qualname__.split("."):
        resolved = inspect.getattr_static(resolved, name, None)
    if resolved is not value:
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
_TTSIM = _capture_ttsim_contract()
