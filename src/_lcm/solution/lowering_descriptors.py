"""Copy fail-closed lowering descriptors and diagnostic-only byte identities.

No collector runs on ordinary solve. CPU and bounded single-device CUDA12
profiles require a source checkout and an identifiable installed native payload.
"""

import dataclasses
import enum
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from importlib.metadata import distribution
from pathlib import Path
from types import MappingProxyType, ModuleType

import jax
import numpy as np

from _lcm.egm.upper_envelope._exact_affine.ffi import _installed_native_directory
from _lcm.regime_building.age_specialization import INVARIANT
from _lcm.solution.cuda_lowering_identity import capture_cuda_lowering_identity
from _lcm.solution.fingerprint import _semantic_fingerprint
from _lcm.typing import HostArray, JSONValue
from lcm.exceptions import ExecutionPlanningError

# A copied lowering descriptor: strings, integers, Booleans, bytes and `None` at
# the leaves, nested in tuples, frozensets and read-only mappings. It retains no
# live payload.
type LoweringDescriptor = (
    str
    | int
    | bool
    | bytes
    | tuple[LoweringDescriptor, ...]
    | frozenset[LoweringDescriptor]
    | MappingProxyType[LoweringDescriptor, LoweringDescriptor]
    | None
)


def describe_lowering_value(
    value: object,  # noqa: PAN001 - copies JAX tree-definition node data, whose auxiliary part JAX leaves untyped
) -> LoweringDescriptor:
    """Copy descriptor data; reject unknown live objects instead of retaining them."""
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__module__, type(value).__qualname__, value.name)
    if value is None or (
        isinstance(value, str | int | bytes) and type(value) in (str, bool, int, bytes)
    ):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, np.dtype):
        return ("dtype", value.str)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    return _describe_tree(value)


def _describe_jax(
    value: jax.Array
    | jax.ShapeDtypeStruct
    | HostArray
    | jax.tree_util.PyTreeDef
    | jax.sharding.Sharding
    | jax.sharding.AbstractMesh,
) -> LoweringDescriptor:
    """Copy supported JAX descriptors without retaining their live payloads."""
    if isinstance(value, (jax.Array, jax.ShapeDtypeStruct, np.ndarray)):
        return (
            "array",
            tuple(value.shape),
            str(value.dtype),
            bool(getattr(value, "weak_type", False)),
            describe_lowering_value(getattr(value, "sharding", None)),
        )
    if isinstance(value, jax.tree_util.PyTreeDef):
        return (
            "pytree",
            describe_lowering_value(value.node_data()),
            tuple(describe_lowering_value(child) for child in value.children()),
        )
    if isinstance(value, jax.sharding.Sharding):
        if not isinstance(
            value, (jax.sharding.NamedSharding, jax.sharding.SingleDeviceSharding)
        ):
            raise ExecutionPlanningError("Unsupported lowering sharding descriptor.")
        mesh = getattr(value, "mesh", None)
        return (
            type(value).__qualname__,
            value.memory_kind,
            tuple(
                sorted((d.platform, d.process_index, d.id) for d in value.device_set)
            ),
            None if mesh is None else tuple(mesh.shape.items()),
            None if mesh is None else describe_lowering_value(mesh.axis_types),
            None if mesh is None else tuple(d.id for d in mesh.devices.flat),
            None
            if not hasattr(value, "spec")
            else (
                describe_lowering_value(tuple(value.spec)),
                describe_lowering_value(value.spec.reduced),
                describe_lowering_value(value.spec.unreduced),
            ),
        )
    if isinstance(value, jax.sharding.AbstractMesh):
        # JAX's trace context carries the ambient abstract mesh, which is empty
        # unless `jax.set_mesh` is active.
        return (
            "abstract_mesh",
            describe_lowering_value(value.shape_tuple),
            describe_lowering_value(value.axis_types),
            describe_lowering_value(value.abstract_device),
        )
    raise ExecutionPlanningError(f"Unspecified JAX descriptor type: {type(value)}")


def _describe_tree(
    value: object,  # noqa: PAN001 - copies JAX tree-definition node data, whose auxiliary part JAX leaves untyped
) -> LoweringDescriptor:
    """Copy structural containers without saving their live leaves."""
    if value is INVARIANT:
        return ("singleton", "_lcm.regime_building.age_specialization", "INVARIANT")
    if isinstance(
        value,
        (
            jax.Array,
            jax.ShapeDtypeStruct,
            np.ndarray,
            jax.tree_util.PyTreeDef,
            jax.sharding.Sharding,
            jax.sharding.AbstractMesh,
        ),
    ):
        return _describe_jax(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                describe_lowering_value(k): describe_lowering_value(v)
                for k, v in value.items()
            }
        )
    if dataclasses.is_dataclass(value):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (f.name, describe_lowering_value(getattr(value, f.name)))
                for f in dataclasses.fields(value)
            ),
        )
    if isinstance(value, (tuple, list)):
        return tuple(describe_lowering_value(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(describe_lowering_value(v) for v in value)
    raise ExecutionPlanningError(f"Unspecified descriptor type: {type(value)}")


def capture_lowering_identity() -> Mapping[str, JSONValue]:
    """Read exact bytes before observation; reuse existing native/source seals."""
    # This diagnostic requires an identifiable source checkout and installed native
    # payload. A wheel without source inputs cannot satisfy this schema.
    root = Path(__file__).resolve().parents[3]
    if not (root / "hatch_build.py").is_file():
        raise ExecutionPlanningError("Lowering identity requires the source checkout.")
    hatch_build = _load_native_fingerprint_utility(root=root)
    sources = tuple(
        (
            path.relative_to(root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for package in ("lcm", "_lcm")
        for path in sorted((root / "src" / package).rglob("*.py"))
    )
    if not sources:
        raise ExecutionPlanningError("No package source identity was collected.")
    directory = _installed_native_directory()
    manifest_bytes = (directory / hatch_build.NATIVE_MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    encoded_inputs = json.dumps(
        manifest["inputs"], sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(encoded_inputs).hexdigest() != manifest["fingerprint"]:
        raise ExecutionPlanningError("Native manifest fingerprint is inconsistent.")
    native_source = hatch_build.native_source_fingerprint(root=root)
    if manifest["inputs"]["source"] != native_source:
        raise ExecutionPlanningError(
            "Installed native source does not match this checkout."
        )
    libraries = []
    for name in manifest["libraries"]:
        if Path(name).name != name:
            raise ExecutionPlanningError("Native library name is not a basename.")
        libraries.append(
            (name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
        )
    if not libraries:
        raise ExecutionPlanningError("Native manifest declares no libraries.")
    distributions = _capture_runtime_files()
    devices = tuple(
        (
            device.id,
            device.process_index,
            device.platform,
            device.device_kind,
            device.client.platform_version,
        )
        for device in jax.devices()
    )
    if not devices:
        raise ExecutionPlanningError("Lowering identity requires a device.")
    extra_identity: Mapping[str, JSONValue] = {}
    if not all(device[2] == "cpu" for device in devices):
        extra_identity = capture_cuda_lowering_identity(
            root=root,
            native_directory=directory,
            manifest=manifest,
            hatch_build=hatch_build,
        )
    return MappingProxyType(
        {
            "source_files": sources,
            "source_identity": _semantic_fingerprint(sources),
            "native_source_identity": native_source,
            "native_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "native_build_identity": manifest["fingerprint"],
            "native_library_bytes": tuple(libraries),
            "runtime_files": tuple(distributions),
            "runtime_identity": _semantic_fingerprint(tuple(distributions)),
            "python_binary_sha256": hashlib.sha256(
                Path(sys.executable).read_bytes()
            ).hexdigest(),
            "python_abi": sys.implementation.cache_tag,
            "devices": devices,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            **extra_identity,
        }
    )


def _capture_runtime_files() -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Hash the common installed runtime source and shared-library inventory."""
    distributions = []
    # CUDA plugin/driver inventories are bound separately by its profile.
    for name in ("jax", "jaxlib", "numpy"):
        package = distribution(name)
        if not package.files:
            raise ExecutionPlanningError("Runtime package has no file inventory.")
        files = tuple(
            (
                str(path),
                hashlib.sha256(
                    Path(str(package.locate_file(path))).read_bytes()
                ).hexdigest(),
            )
            for path in sorted(package.files, key=str)
            if Path(str(path)).suffix in (".py", ".so", ".pyd", ".dll", ".dylib")
        )
        if not files:
            raise ExecutionPlanningError(
                "Runtime package has no source or binary files."
            )
        distributions.append((name, files))
    return tuple(distributions)


def _load_native_fingerprint_utility(*, root: Path) -> ModuleType:
    """Load the checkout's existing source fingerprint utility without building."""
    spec = importlib.util.spec_from_file_location(
        "_lcm_lowering_native_fingerprint", root / "hatch_build.py"
    )
    if spec is None or spec.loader is None:
        raise ExecutionPlanningError(
            "Native source fingerprint utility is unavailable."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
