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
import re
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
WINDOWS_CPU_LIBRARY = "certified_affine_ffi_cpu.dll"

# XLA_FFI_DEFINE_HANDLER_SYMBOL expands to a plain extern-C definition. MSVC
# exports no such function unless it is marked __declspec(dllexport) or named in
# a module-definition file. Keep the export surface in one authoritative tuple;
# the Windows build writes it to a .def file consumed by link.exe.
EXACT_AFFINE_HANDLER_SYMBOLS = (
    "CertifiedAffineCompareF32",
    "CertifiedAffineCompareF64",
    "ExactAffineReadF32",
    "ExactAffineReadF64",
    "ExactQueryWinnerF32",
    "ExactQueryWinnerF64",
    "ExactQueryWinnerBatchedF32",
    "ExactQueryWinnerBatchedF64",
    "ExactAffineHandoverF32",
    "ExactAffineHandoverF64",
    "ExactCellHullF32",
    "ExactCellHullF64",
)

# Architectures the CUDA build emits ready code for. The oldest entry is the
# oldest card the project supports; a toolchain that has retired it drops it
# from this set rather than failing the compile over a target it rejects.
_DEFAULT_CUDA_ARCHITECTURES = (70, 75, 80, 86, 90)


def parse_gpu_architectures(*, listing: str) -> frozenset[int] | None:
    """Return the capabilities named in an `nvcc --list-gpu-arch` listing.

    Args:
        listing: The compiler's own output, one `compute_NN` per line.

    Returns:
        Frozenset of capability numbers, or `None` where the listing names none
        — an unreadable probe means "unknown", not "supports nothing", so the
        caller emits its full target set rather than an empty one.

    """
    found = {int(number) for number in re.findall(r"compute_(\d+)", listing)}
    return frozenset(found) or None


def probe_gpu_architectures(*, nvcc: str) -> frozenset[int] | None:
    """Return the capabilities a CUDA toolchain can target, by asking it.

    Args:
        nvcc: Path to the CUDA compiler to interrogate.

    Returns:
        Frozenset of capability numbers, or `None` where the compiler could not
        be asked — one predating the flag, or one that fails to run.

    """
    try:
        result = subprocess.run(  # noqa: S603
            [nvcc, "--list-gpu-arch"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return parse_gpu_architectures(listing=result.stdout)


def cuda_arch_flags(
    *,
    nvcc_flags: tuple[str, ...],
    supported_architectures: frozenset[int] | None = None,
) -> list[str]:
    """Return the `-gencode` targets to add to a CUDA compile.

    The last target is virtual, so a card matching no ready-code entry can still
    have one translated for it at load. It sits at the *oldest* architecture
    emitted, not the newest, because intermediate code is forward-compatible
    only: emitted for capability `X` it translates onto any device at `X` or
    above and onto nothing below. At the newest architecture it would cover only
    hardware newer than every ready entry — the one case ready code already
    handles — leaving every older unlisted card with no image at all.

    Args:
        nvcc_flags: Flags the caller supplied, via `NVCCFLAGS`.
        supported_architectures: Capabilities the toolchain can target. `None`
            means unknown, and every default target is emitted.

    Returns:
        List of `-gencode` argument pairs, flattened. Empty where the caller
        already names an architecture — two sets of targets would conflict — or
        where the toolchain supports none of the defaults, which leaves `nvcc`
        to pick its own rather than failing over a target it rejects.

    """
    if any(
        flag.startswith(("-arch", "--gpu-architecture", "-gencode"))
        for flag in nvcc_flags
    ):
        return []
    architectures = sorted(
        architecture
        for architecture in _DEFAULT_CUDA_ARCHITECTURES
        if supported_architectures is None or architecture in supported_architectures
    )
    if not architectures:
        return []
    flags = []
    for architecture in architectures:
        flags += ["-gencode", f"arch=compute_{architecture},code=sm_{architecture}"]
    oldest = architectures[0]
    flags += ["-gencode", f"arch=compute_{oldest},code=compute_{oldest}"]
    return flags


def windows_module_definition() -> str:
    """Return the MSVC module-definition file exporting every FFI handler."""
    lines = ["LIBRARY certified_affine_ffi_cpu", "EXPORTS"]
    lines.extend(f"    {name}" for name in EXACT_AFFINE_HANDLER_SYMBOLS)
    return "\n".join(lines) + "\n"


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
    if os.environ.get("LCM_SKIP_EXACT_AFFINE", "") not in ("", "0"):
        # An install without a C++ compiler is a deliberate choice, not a fallback:
        # `"exact"` is the default upper envelope, so an install that skips the
        # kernel cannot run DC-EGM on its defaults. Missing the compiler by
        # accident still raises below, so the capability is never dropped quietly.
        sys.stdout.write(
            "exact-affine: LCM_SKIP_EXACT_AFFINE is set, so no kernel is built. "
            "The certified upper envelope is unavailable in this install, and "
            "with it the default DC-EGM envelope; brute-force backward induction "
            "is unaffected. Unset the variable and reinstall to get it back.\n"
        )
        return []

    source_dir = root / PACKAGE_DIR
    include_dir = jax_include_dir or _find_jax_include_dir()

    if sys.platform == "win32":
        compiler = os.environ.get("CXX") or shutil.which("cl")
        if compiler is None:
            msg = (
                "No MSVC C++ compiler found. pylcm's certified upper envelope "
                "needs cl.exe to build the Windows exact-affine kernel; activate "
                "an MSVC developer environment or set CXX."
            )
            raise RuntimeError(msg)
        target = source_dir / WINDOWS_CPU_LIBRARY
        definition = source_dir / "certified_affine_ffi_cpu.def"
        definition.write_text(windows_module_definition())
        sys.stdout.write(
            f"exact-affine: building Windows CPU library with MSVC at {compiler}.\n"
        )
        return [
            _compile(
                command=[
                    compiler,
                    "/nologo",
                    "/std:c++20",
                    "/O2",
                    "/DNDEBUG",
                    "/EHsc",
                    "/LD",
                    f"/I{include_dir}",
                    str(source_dir / "certified_affine_ffi_cpu.cc"),
                    f"/Fe:{target}",
                    "/link",
                    f"/DEF:{definition}",
                ],
                target=target,
            )
        ]

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
                "-pthread",
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
                    *cuda_arch_flags(
                        nvcc_flags=nvcc_flags,
                        supported_architectures=probe_gpu_architectures(nvcc=nvcc),
                    ),
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

    Both streams are kept, because which one carries the diagnostics is a property
    of the toolchain: `gcc` and `clang` write errors to stderr, while MSVC writes
    them to stdout. Reporting one stream leaves the other toolchain announcing an
    exit code and nothing else, at the one moment the message is needed.
    """
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    report = "\n".join(part for part in (result.stdout, result.stderr) if part.strip())
    target.with_suffix(target.suffix + ".build.log").write_text(
        f"$ {' '.join(command)}\nexit {result.returncode}\n{report}"
    )
    if result.returncode != 0:
        msg = (
            f"Failed to build {target.name} (exit {result.returncode}):\n"
            f"{' '.join(command)}\n{report}"
        )
        raise RuntimeError(msg)
    if not target.is_file():
        msg = f"Compiler reported success but {target} was not written."
        raise RuntimeError(msg)
    return target


if __name__ == "__main__":
    for path in build_exact_affine(root=Path(__file__).parent):
        sys.stdout.write(f"built {path}\n")
