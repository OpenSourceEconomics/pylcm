"""Collect the bounded CUDA12 source-checkout lowering identity.

Observed headers and tools identify the current environment. They do not attest
which header bytes produced a previously built or cached native library.
"""

import csv
import hashlib
import os
import shutil
import subprocess
from collections.abc import Mapping
from importlib.metadata import Distribution, distributions
from pathlib import Path
from types import MappingProxyType, ModuleType

import jax

from _lcm.solution.fingerprint import _semantic_fingerprint
from _lcm.typing import JSONValue
from lcm.exceptions import ExecutionPlanningError

_PROC_MAPS_FIELD_COUNT = 6
_NVIDIA_SMI_FIELD_COUNT = 5

#: CUDA package digests per installed inventory of `(name, version, location)`.
_CUDA_PACKAGE_DIGESTS: dict[
    tuple[tuple[str, str, str], ...],
    tuple[MappingProxyType[str, tuple[tuple[str, str], ...]], frozenset[Path]],
] = {}


def capture_cuda_lowering_identity(
    *,
    root: Path,
    native_directory: Path,
    manifest: Mapping[str, JSONValue],
    hatch_build: ModuleType,
) -> Mapping[str, JSONValue]:
    """Bind actual single-device CUDA placement, loaded plugin and native bytes."""
    try:
        devices = tuple(jax.devices())
        if (
            len(devices) != 1
            or devices[0].platform != "gpu"
            or devices != tuple(jax.devices("cuda"))
            or devices[0].local_hardware_id != 0
            or not jax.devices("cpu")
        ):
            raise ExecutionPlanningError(
                "CUDA lowering requires one CUDA device and CPU retention support."
            )
        if manifest["inputs"] != hatch_build.native_build_inputs(root=root):
            raise ExecutionPlanningError("Current native build inputs do not match.")
        libraries = manifest["libraries"]
        if not isinstance(libraries, list) or not all(
            isinstance(name, str) for name in libraries
        ):
            raise ExecutionPlanningError("Native library inventory is invalid.")
        if "libcertified_affine_ffi_cuda.so" not in libraries:
            raise ExecutionPlanningError("Native payload has no CUDA library.")
        inputs = manifest["inputs"]
        if not isinstance(inputs, Mapping):
            raise ExecutionPlanningError("Native build inputs are not a mapping.")
        return MappingProxyType(
            {
                "gpu_identity": _capture_cuda_runtime(
                    native_directory=native_directory
                ),
                "native_header_files": _headers(
                    files=tuple(sorted((root / hatch_build.PACKAGE_DIR).glob("*.h"))),
                    relative_to=root,
                ),
                "jax_ffi_header_files": _headers(
                    files=tuple(sorted(Path(jax.ffi.include_dir()).rglob("*.h"))),
                    relative_to=Path(jax.ffi.include_dir()),
                ),
                "native_tool_files": tuple(
                    (name, _tool_identity(inputs[name]))
                    for name in ("compiler", "nvcc")
                ),
            }
        )
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        raise ExecutionPlanningError(
            "CUDA lowering identity is unavailable or incompatible."
        ) from exc


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _headers(
    *, files: tuple[Path, ...], relative_to: Path
) -> tuple[tuple[str, str], ...]:
    identities = tuple(
        (path.relative_to(relative_to).as_posix(), _sha256(path)) for path in files
    )
    if not identities:
        raise ExecutionPlanningError("Lowering header identity is unavailable.")
    return identities


def _tool_identity(command: object) -> tuple[str, str, str, str]:
    if not isinstance(command, str) or not command:
        raise ExecutionPlanningError("Native tool command must be a nonempty string.")
    declared = Path(command)
    if not declared.is_absolute() and declared.name != command:
        raise ExecutionPlanningError("Relative native tool paths are unsupported.")
    selected = shutil.which(command)
    if selected is None:
        raise ExecutionPlanningError("Native tool is absent or not executable.")
    origin = Path(selected).absolute()
    real_file = origin.resolve(strict=True)
    if not real_file.is_file() or not os.access(real_file, os.X_OK):
        raise ExecutionPlanningError("Native tool is not a regular executable file.")
    return command, str(origin), str(real_file), _sha256(real_file)


def _capture_cuda_runtime(*, native_directory: Path) -> Mapping[str, JSONValue]:
    packages, pjrt_libraries = _capture_cuda_packages()
    loaded_paths = _loaded_libraries()
    native_cuda = (native_directory / "libcertified_affine_ffi_cuda.so").resolve(
        strict=True
    )
    if native_cuda not in loaded_paths:
        raise ExecutionPlanningError(
            "The identified native CUDA library is not loaded."
        )
    if not pjrt_libraries.intersection(loaded_paths):
        raise ExecutionPlanningError(
            "The identified CUDA12 PJRT library is not loaded."
        )
    driver_paths = {
        path
        for path in loaded_paths
        if path.name.startswith(("libcuda.so", "libnvidia-"))
    }
    if not any(path.name.startswith("libcuda.so") for path in driver_paths):
        raise ExecutionPlanningError("No identifiable loaded CUDA driver exists.")
    rows = _physical_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "0", rows[0][0]):
        raise ExecutionPlanningError("CUDA device visibility is outside this profile.")
    package_bytes = tuple(sorted(packages.items()))
    kernel_version = Path("/proc/driver/nvidia/version").read_text()
    if not kernel_version.strip():
        raise ExecutionPlanningError("NVIDIA kernel driver identity is empty.")
    return MappingProxyType(
        {
            "profile": "cuda12-source-checkout-v1",
            "package_files": package_bytes,
            "package_identity": _semantic_fingerprint(package_bytes),
            "loaded_driver_files": tuple(
                (str(path), _sha256(path)) for path in sorted(driver_paths)
            ),
            "driver_kernel_version": kernel_version,
            "physical_devices": rows,
            "cuda_visible_devices": visible,
            "precision": 64 if jax.config.x64_enabled else 32,
            "compilation_cache_enabled": jax.config.jax_enable_compilation_cache,
        }
    )


def _loaded_libraries() -> frozenset[Path]:
    paths = set()
    for row in Path("/proc/self/maps").read_text().splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) != _PROC_MAPS_FIELD_COUNT or not fields[5].startswith("/"):
            continue
        path = Path(fields[5])
        if ".so" not in path.name:
            continue
        if not path.is_file():
            raise ExecutionPlanningError("A loaded library has no identifiable file.")
        paths.add(path.resolve(strict=True))
    return frozenset(paths)


def _physical_device() -> tuple[tuple[str, ...], ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise ExecutionPlanningError("NVIDIA device identity query is unavailable.")
    result = subprocess.run(  # noqa: S603 - fixed read-only query to installed tool
        [
            executable,
            "--query-gpu=uuid,pci.bus_id,name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = tuple(
        tuple(item.strip() for item in row)
        for row in csv.reader(result.stdout.splitlines())
    )
    if len(rows) != 1 or len(rows[0]) != _NVIDIA_SMI_FIELD_COUNT or not all(rows[0]):
        raise ExecutionPlanningError("Physical CUDA device identity is ambiguous.")
    return rows


def _capture_cuda_packages() -> tuple[
    dict[str, tuple[tuple[str, str], ...]], set[Path]
]:
    """Identify installed CUDA package bytes and actual PJRT library paths.

    File digests are computed once per process for each installed inventory of
    distributions, identified by name, version and location.
    """
    installed = tuple(distributions())
    inventory = tuple(
        (str(package.metadata["Name"]), package.version, str(package.locate_file("")))
        for package in installed
    )
    if inventory not in _CUDA_PACKAGE_DIGESTS:
        _CUDA_PACKAGE_DIGESTS[inventory] = _hash_cuda_packages(packages=installed)
    packages, pjrt_libraries = _CUDA_PACKAGE_DIGESTS[inventory]
    return dict(packages), set(pjrt_libraries)


def _hash_cuda_packages(
    *, packages: tuple[Distribution, ...]
) -> tuple[MappingProxyType[str, tuple[tuple[str, str], ...]], frozenset[Path]]:
    """Hash the files of the installed CUDA packages."""
    digests = {}
    pjrt_libraries = set()
    for package in packages:
        declared_name = package.metadata["Name"]
        if not isinstance(declared_name, str) or not declared_name:
            raise ExecutionPlanningError("Runtime package has no name identity.")
        name = declared_name.lower().replace("_", "-")
        if not name.startswith(("jax-cuda", "nvidia-")):
            continue
        if name in digests or not package.files:
            raise ExecutionPlanningError(
                "CUDA package inventory is ambiguous or empty."
            )
        files = []
        for path in sorted(package.files, key=str):
            if (
                Path(str(path)).suffix == ".pyc"
                or "__pycache__" in Path(str(path)).parts
            ):
                continue
            installed = Path(str(package.locate_file(path)))
            files.append((str(path), _sha256(installed)))
            if name == "jax-cuda12-pjrt" and ".so" in installed.name:
                pjrt_libraries.add(installed.resolve(strict=True))
        if not files:
            raise ExecutionPlanningError("CUDA package has no identifiable files.")
        digests[name] = tuple(files)
    if not {"jax-cuda12-plugin", "jax-cuda12-pjrt"} <= digests.keys() or any(
        name.startswith("jax-cuda")
        and name not in {"jax-cuda12-plugin", "jax-cuda12-pjrt"}
        for name in digests
    ):
        raise ExecutionPlanningError("Lowering requires the CUDA12 plugin profile.")
    return MappingProxyType(digests), frozenset(pjrt_libraries)
