"""Compile the exact-affine FFI shared objects into the package tree.

The certified upper-envelope path decides candidate ownership with exact integer
arithmetic over the stored IEEE operands rather than with backend floating
arithmetic. That kernel is C++ (and CUDA), so a wheel — including the editable
wheel a development environment installs — has to carry the compiled libraries
beside `_lcm/egm/upper_envelope/_exact_affine/`.

Compiling is a build-time step on purpose. Importing pylcm never invokes a
compiler: a missing library is reported when an exact verdict is requested,
naming the task that builds it, rather than silently falling back to arithmetic
that cannot make the guarantee.

Run standalone during development, after any change to the C++ or CUDA sources:

    pixi run build-exact-affine

CUDA is optional. Where `nvcc` is absent the CPU library is built alone and the
certified path runs on CPU only. Where it is present the build emits ready code
for several architectures plus a virtual target; `NVCCFLAGS='-arch=sm_80'`
replaces that set with one architecture, which builds faster and runs only
there.

Which libraries come out therefore depends on the environment the build ran in,
and every combination exits successfully. The build states the toolchain it
found on stdout so that a CPU-only result is read at the time rather than
inferred later from a library that is not there.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:
    # Run standalone — `pixi run build-exact-affine` — from a runtime environment,
    # which carries no build backend. Only the hook class needs hatchling, and
    # nothing instantiates it here.
    BuildHookInterface = object

# Location of the FFI sources, relative to the project root.
PACKAGE_DIR = Path("src/_lcm/egm/upper_envelope/_exact_affine")

# Names the Python wrapper loads, per platform. The CUDA one may be absent.
CPU_LIBRARY = "libcertified_affine_ffi_cpu.so"
CUDA_LIBRARY = "libcertified_affine_ffi_cuda.so"

# Architectures the CUDA build emits ready code for, plus one virtual target so
# an architecture not listed can still be translated at load. Naming none would
# leave translation the only route, and a driver older than the toolchain that
# emitted the intermediate form refuses it — the kernel fails to launch rather
# than running slowly.
_DEFAULT_CUDA_TARGETS = (
    "arch=compute_75,code=sm_75",
    "arch=compute_80,code=sm_80",
    "arch=compute_86,code=sm_86",
    "arch=compute_90,code=sm_90",
    "arch=compute_90,code=compute_90",
)


def cuda_arch_flags(*, nvcc_flags: tuple[str, ...]) -> list[str]:
    """Return the `-gencode` targets to add to a CUDA compile.

    Args:
        nvcc_flags: Flags the caller supplied, via `NVCCFLAGS`.

    Returns:
        List of `-gencode` argument pairs, flattened, and empty where the caller
        already names an architecture — two sets of targets would conflict.

    """
    if any(
        flag.startswith(("-arch", "--gpu-architecture", "-gencode"))
        for flag in nvcc_flags
    ):
        return []
    flags = []
    for target in _DEFAULT_CUDA_TARGETS:
        flags += ["-gencode", target]
    return flags


def toolchain_report(*, compiler: str, nvcc: str | None) -> str:
    """Return the line a build prints to state which toolchain it found.

    Which libraries a build produces depends on the environment it ran in, and a
    build that skipped CUDA exits successfully. Stating the toolchain makes the
    two cases distinguishable at the time rather than inferable afterwards from a
    library that is not there.

    Args:
        compiler: Path to the C++ compiler the CPU library is built with.
        nvcc: Path to `nvcc`, or `None` where none is on the path.

    Returns:
        One line naming the compilers found, and — where `nvcc` is absent — what
        that means for the certified path.

    """
    if nvcc is None:
        return (
            f"exact-affine: building with c++ at {compiler}; no nvcc on this "
            "path, so the CUDA library is not built and the certified upper "
            "envelope runs on CPU only."
        )
    return f"exact-affine: building with c++ at {compiler} and nvcc at {nvcc}."


def build_exact_affine(*, root: Path, jax_include_dir: str | None = None) -> list[Path]:
    """Compile the FFI libraries and return the paths that were written.

    Args:
        root: Project root containing `src/`.
        jax_include_dir: Directory holding the XLA FFI headers. Resolved from the
            importable `jax` when omitted.

    Returns:
        List of shared-object paths written, CPU first. The CUDA entry is absent
        where `nvcc` is not on the path, and the list is empty on Windows.

    Raises:
        RuntimeError: If no C++ compiler is available, if the FFI headers cannot
            be located, or if a compile fails.

    """
    if sys.platform == "win32":
        # The compile flags and library names here are the Unix ones, and a
        # MinGW artifact under a `.so` name loads only where its toolchain's
        # runtime is present. Building nothing leaves the certified path to
        # report its own absence, which names what is missing.
        sys.stdout.write(
            "exact-affine: no kernel is built on this platform, so the certified "
            "upper envelope is unavailable here.\n"
        )
        return []

    source_dir = root / PACKAGE_DIR
    include_dir = jax_include_dir or _find_jax_include_dir()

    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        msg = (
            "No C++ compiler found. pylcm's certified upper envelope needs one to "
            "build the exact-affine kernel; set CXX or install a compiler."
        )
        raise RuntimeError(msg)

    nvcc = shutil.which("nvcc")
    sys.stdout.write(f"{toolchain_report(compiler=compiler, nvcc=nvcc)}\n")

    written = [
        _compile(
            command=[
                compiler,
                "-std=c++20",
                "-O3",
                "-DNDEBUG",
                "-fPIC",
                "-shared",
                f"-I{include_dir}",
                str(source_dir / "certified_affine_ffi_cpu.cc"),
                "-o",
                str(source_dir / CPU_LIBRARY),
            ],
            target=source_dir / CPU_LIBRARY,
        )
    ]

    if nvcc is not None:
        nvcc_flags = tuple(os.environ.get("NVCCFLAGS", "").split())
        written.append(
            _compile(
                command=[
                    nvcc,
                    "-std=c++17",
                    "-O3",
                    "--shared",
                    "-Xcompiler=-fPIC",
                    *cuda_arch_flags(nvcc_flags=nvcc_flags),
                    *nvcc_flags,
                    f"-I{include_dir}",
                    str(source_dir / "certified_affine_ffi_cuda.cu"),
                    "-o",
                    str(source_dir / CUDA_LIBRARY),
                ],
                target=source_dir / CUDA_LIBRARY,
            )
        )
    return written


class CustomBuildHook(BuildHookInterface):
    """Build the FFI libraries before hatchling collects wheel contents."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:  # noqa: ARG002
        """Compile the libraries and mark the wheel as platform-specific."""
        build_exact_affine(root=Path(self.root))
        build_data["infer_tag"] = True
        build_data["pure_python"] = False


def _find_jax_include_dir() -> str:
    """Return the directory holding the XLA FFI headers."""
    try:
        # Deliberately local: this module is imported by hatchling in a build
        # environment that may not have jax, and only this one function needs it.
        import jax.ffi  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the build env
        msg = (
            "jax is required to locate the XLA FFI headers when building pylcm's "
            "exact-affine kernel. Install jax in the build environment, or pass "
            "jax_include_dir explicitly."
        )
        raise RuntimeError(msg) from error
    return jax.ffi.include_dir()


def _compile(*, command: list[str], target: Path) -> Path:
    """Run one compile command and return the file it produced.

    The compiler's own report is written beside the library as `<name>.build.log`,
    together with the exact command. That report is the only place a CUDA build
    states its register, stack-frame, spill and local-memory usage — `ptxas`
    writes it to stderr, where a successful build would otherwise discard it —
    and those numbers decide whether the fixed-width accumulators fit a device.
    """
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    target.with_suffix(target.suffix + ".build.log").write_text(
        f"$ {' '.join(command)}\nexit {result.returncode}\n{result.stderr}"
    )
    if result.returncode != 0:
        msg = (
            f"Failed to build {target.name} (exit {result.returncode}):\n"
            f"{' '.join(command)}\n{result.stderr}"
        )
        raise RuntimeError(msg)
    if not target.is_file():
        msg = f"Compiler reported success but {target} was not written."
        raise RuntimeError(msg)
    return target


if __name__ == "__main__":
    for path in build_exact_affine(root=Path(__file__).parent):
        sys.stdout.write(f"built {path}\n")
