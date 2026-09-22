"""Bounded library contracts for GETTSIM's generated JAX callable graphs.

The semantic walk separates installed implementations from constructed model data:

- Reviewed exports of dags, flatten-dict, ttsim and NumPy-groupies are sealed by their
  executable code, defaults and the versions of both numerical backend stacks.
  A function must resolve by identity at its installed source location and carry
  no closure or instance state. Generated functions and GETTSIM policy bodies
  retain the full recursive walk.
- Exact ttsim column declarations and installed GETTSIM frozen policy records
  bind every field. FKType binds its complete member schema and selected member.
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


def external_enum_record(value: object) -> tuple[object, ...] | None:
    """Return the versioned schema and value of a genuine ttsim FKType member."""
    contract = _ttsim_contract_for(value)
    if contract is None or type(value) is not contract.enum_type:
        return None
    _require_supported_version(distribution="ttsim-backend", version=contract.version)
    member = value
    enum_members = contract.enum_type.__members__
    if (
        any(
            type(item.value) is not str
            or set(vars(item)) != {"_value_", "_name_", "__objclass__", "_sort_order_"}
            for item in enum_members.values()
        )
        or tuple((name, item.value) for name, item in enum_members.items())
        != contract.members
    ):
        raise TypeError(
            "Cannot durably fingerprint a modified ttsim FKType declaration."
        )
    return "ttsim-FKType", contract.version, contract.members, member.name, member.value


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
        operation = inspect.getattr_static(value_type, operation_name)
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
        type(value) is cls for cls in contract.columns
    ):
        _require_supported_version(
            distribution="ttsim-backend", version=contract.version
        )
        metadata = _COLUMN_METADATA
    else:
        return False
    fields = {field.name for field in dataclasses.fields(cast("Any", value))}
    if set(vars(value)) - fields - metadata:
        raise TypeError("Cannot durably fingerprint extra external declaration state.")
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
    if distribution is None or value.__closure__ or value.__dict__:
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TTSimContract:
    """Genuine optional backend declarations captured when an integration is used."""

    version: str
    """Installed backend release."""
    enum_type: type[Enum]
    """Exact foreign-key enum class."""
    members: tuple[tuple[str, str], ...]
    """Ordered foreign-key member schema."""
    columns: tuple[type, ...]
    """Exact frozen column and rounding declaration types."""
    policy_function: type
    """Declaration type that may bind the JAX backend in a generated partial."""
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
    return _TTSimContract(
        version=version,
        enum_type=columns.FKType,
        members=tuple(
            (name, member.value) for name, member in columns.FKType.__members__.items()
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
_SUPPORTED_VERSIONS = {
    "dags": "0.6.0",
    "flatten-dict": "0.5.0",
    "gettsim": "1.2",
    "numpy-groupies": "0.11.3",
    "ttsim-backend": "1.2.1",
}
_SUPPORTED_OPERATIONS = frozenset(
    {
        "dags.tree.tree_utils.flatten_to_qnames",
        "ttsim.tt.aggregation.grouped_sum",
        "ttsim.tt.aggregation.sum_by_p_id",
        "ttsim.tt.column_objects_param_function.ColumnFunction.__call__",
        "ttsim.tt.piecewise_polynomial.piecewise_polynomial",
        "ttsim.unit_converters.per_m_to_per_y",
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
