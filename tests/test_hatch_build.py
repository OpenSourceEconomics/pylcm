"""What the exact-affine build emits, and what it says it emitted.

Two properties the build owes its caller:

- The CUDA build targets real architectures, not PTX alone. A PTX-only artifact
  has to be translated by the driver at load, and a driver older than the
  toolchain that emitted the PTX refuses it outright — the kernel does not
  launch. Naming architectures embeds ready code for them and leaves the
  translation path as a fallback rather than the only route.
- The build states which toolchain it found. Whether a CUDA library is produced
  depends on the environment the build ran in, so a build that skipped CUDA and
  a build that had no CUDA to skip are indistinguishable from an exit status —
  the difference has to be read off a missing file afterwards unless the build
  says so at the time.
"""

import subprocess
from pathlib import Path

import pytest

import hatch_build
from _lcm.egm.upper_envelope._exact_affine import ffi


def test_the_default_cuda_build_names_architectures():
    """Without an override, the build emits code for real architectures."""
    flags = hatch_build.cuda_arch_flags(nvcc_flags=())

    assert any(flag.startswith("arch=compute_") for flag in flags)


def test_the_default_cuda_build_keeps_a_forward_compatible_fallback():
    """A virtual target is kept, so an architecture not listed can still run."""
    flags = hatch_build.cuda_arch_flags(nvcc_flags=())

    virtual = [flag for flag in flags if ",code=compute_" in flag]
    assert virtual, f"no virtual target among {flags}"


def _architectures(*, flags: list[str], kind: str) -> list[int]:
    """Return the capability numbers of every `code={kind}_NN` target, ascending."""
    return sorted(
        int(flag.rsplit(f"code={kind}_", 1)[1])
        for flag in flags
        if f",code={kind}_" in flag
    )


def test_the_cuda_build_emits_ready_code_for_the_oldest_supported_card():
    """A Volta card gets ready code, not only a translation target.

    Its capability is below every later architecture, so a virtual target emitted
    for a newer one cannot be translated onto it and the kernel does not launch.
    """
    flags = hatch_build.cuda_arch_flags(nvcc_flags=())

    assert 70 in _architectures(flags=flags, kind="sm")


def test_the_virtual_target_sits_at_the_lowest_architecture():
    """The translation target names the oldest architecture the build supports.

    Intermediate code is forward-compatible only: emitted for capability `X` it
    can be translated onto any device at `X` or above, and onto nothing below. A
    virtual target at the newest architecture therefore serves only hardware newer
    than everything already listed, which is the one case ready code covers — and
    leaves every unlisted older card with no image at all.
    """
    flags = hatch_build.cuda_arch_flags(nvcc_flags=())

    virtual = _architectures(flags=flags, kind="compute")
    assert virtual == [min(_architectures(flags=flags, kind="sm"))]


def test_an_architecture_the_toolkit_cannot_target_is_not_emitted():
    """Targets the installed CUDA toolkit rejects are dropped, not passed to it.

    A toolkit that has retired an architecture fails the whole compile on the
    first `-gencode` naming it, so emitting one target no compiler can build
    costs every other target too.
    """
    flags = hatch_build.cuda_arch_flags(
        nvcc_flags=(), supported_architectures=frozenset({75, 80, 86, 90})
    )

    assert 70 not in _architectures(flags=flags, kind="sm")


def test_the_virtual_target_follows_the_toolkit_floor():
    """The translation target tracks the oldest architecture actually emitted.

    Pinning it to an architecture the toolkit dropped would fail the build; the
    fallback has to be the lowest one this compiler can still produce.
    """
    flags = hatch_build.cuda_arch_flags(
        nvcc_flags=(), supported_architectures=frozenset({75, 80, 86, 90})
    )

    assert _architectures(flags=flags, kind="compute") == [75]


def test_a_toolkit_that_keeps_the_oldest_card_still_gets_it():
    """Where the toolkit supports the oldest card, its ready code is emitted."""
    flags = hatch_build.cuda_arch_flags(
        nvcc_flags=(), supported_architectures=frozenset({70, 75, 80, 86, 90})
    )

    assert 70 in _architectures(flags=flags, kind="sm")
    assert _architectures(flags=flags, kind="compute") == [70]


def test_the_supported_architectures_are_read_from_the_compiler():
    """The supported set is what the compiler reports, not a hardcoded list."""
    listing = "compute_75\ncompute_80\ncompute_86\n"

    assert hatch_build.parse_gpu_architectures(listing=listing) == frozenset(
        {75, 80, 86}
    )


def test_a_compiler_that_reports_nothing_leaves_the_targets_alone():
    """An unreadable listing emits the full target set rather than none.

    Dropping every target because a probe failed would silently produce a
    library with no ready code at all, which fails at launch rather than here.
    """
    assert hatch_build.parse_gpu_architectures(listing="") is None


def test_an_explicit_arch_suppresses_the_defaults():
    """A caller naming its own architecture is not given conflicting targets."""
    flags = hatch_build.cuda_arch_flags(nvcc_flags=("-arch=sm_80",))

    assert flags == []


def test_an_explicit_gencode_suppresses_the_defaults():
    """`-gencode` counts as naming an architecture, like `-arch` does."""
    flags = hatch_build.cuda_arch_flags(
        nvcc_flags=("-gencode", "arch=compute_70,code=sm_70")
    )

    assert flags == []


def test_the_build_names_the_cuda_compiler_it_found():
    """A build that produces the CUDA library says which `nvcc` produced it."""
    report = hatch_build.toolchain_report(compiler="/usr/bin/c++", nvcc="/opt/bin/nvcc")

    assert "/opt/bin/nvcc" in report


def test_the_build_states_when_it_found_no_cuda_compiler():
    """A build that produces no CUDA library says so, rather than exiting quietly."""
    report = hatch_build.toolchain_report(compiler="/usr/bin/c++", nvcc=None)

    assert "no nvcc" in report


def test_native_libraries_are_built_outside_the_checkout(*, monkeypatch, tmp_path):
    """A native build writes only to its declared payload directory."""
    source_dir = tmp_path / hatch_build.PACKAGE_DIR
    source_dir.mkdir(parents=True)
    (source_dir / "certified_affine_ffi_cpu.cc").write_text("// source")
    payload_dir = tmp_path / "wheel-payload"

    def fake_compile(*, command, target):
        assert str(target) in command
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"shared object")
        return target

    monkeypatch.setattr(hatch_build.sys, "platform", "linux")
    monkeypatch.delenv("LCM_SKIP_EXACT_AFFINE", raising=False)
    monkeypatch.setenv("CXX", "/usr/bin/c++")
    monkeypatch.setattr(hatch_build.shutil, "which", lambda _name: None)
    monkeypatch.setattr(hatch_build, "_compile", fake_compile)

    written = hatch_build.build_exact_affine(
        root=tmp_path,
        output_dir=payload_dir,
        jax_include_dir="/jax/include",
    )

    assert written == [payload_dir / hatch_build.CPU_LIBRARY]
    assert not list(source_dir.glob("*.so"))


def test_native_payload_is_force_included_for_wheels_and_editable_installs(tmp_path):
    """Both install modes receive the same checkout-independent payload."""
    library = tmp_path / hatch_build.CPU_LIBRARY
    library.write_bytes(b"shared object")

    build_data = {}
    hatch_build.include_native_payload(build_data=build_data, libraries=[library])

    expected = {
        str(library): f"_pylcm_native/{hatch_build.CPU_LIBRARY}",
    }
    assert build_data["force_include"] == expected
    assert build_data["force_include_editable"] == expected


def test_native_source_fingerprint_changes_with_the_kernel_source(tmp_path):
    """A changed kernel source cannot reuse an installed native payload."""
    source_dir = tmp_path / hatch_build.PACKAGE_DIR
    source_dir.mkdir(parents=True)
    source = source_dir / "certified_affine_ffi_cpu.cc"
    source.write_text("first")
    (source_dir / "handler_symbols.py").write_text("symbols")
    (tmp_path / "hatch_build.py").write_text("hook")

    before = hatch_build.native_source_fingerprint(root=tmp_path)
    source.write_text("second")
    after = hatch_build.native_source_fingerprint(root=tmp_path)

    assert before != after


def test_windows_module_definition_exports_every_handler():
    """Every name in the export tuple reaches the .def file, in order.

    This compares the generated file against the tuple it is generated from, so
    it catches a broken generator and nothing else. That the tuple holds the
    right names is a separate property, pinned against `ffi._TARGETS` below.
    """
    definition = hatch_build.windows_module_definition().splitlines()

    assert definition[:2] == ["LIBRARY certified_affine_ffi_cpu", "EXPORTS"]
    assert definition[2:] == [
        f"    {name}" for name in hatch_build.EXACT_AFFINE_HANDLER_SYMBOLS
    ]


def test_the_windows_build_uses_msvc_and_the_export_definition(
    *, monkeypatch, tmp_path, capsys
):
    """Windows emits a DLL through cl.exe and passes the generated .def file."""
    source_dir = tmp_path / hatch_build.PACKAGE_DIR
    source_dir.mkdir(parents=True)
    (source_dir / "certified_affine_ffi_cpu.cc").write_text("// source")
    commands = []

    def fake_compile(*, command, target):
        commands.append(command)
        target.write_bytes(b"dll")
        return target

    monkeypatch.setattr(hatch_build.sys, "platform", "win32")
    monkeypatch.delenv("LCM_SKIP_EXACT_AFFINE", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(
        hatch_build.shutil,
        "which",
        lambda name: "C:/VC/cl.exe" if name == "cl" else None,
    )
    monkeypatch.setattr(hatch_build, "_compile", fake_compile)

    payload_dir = tmp_path / ".pylcm-native-build"
    written = hatch_build.build_exact_affine(
        root=tmp_path, jax_include_dir="C:/jax/include"
    )

    assert written == [payload_dir / hatch_build.WINDOWS_CPU_LIBRARY]
    assert commands
    assert commands[0][0] == "C:/VC/cl.exe"
    assert "/LD" in commands[0]
    assert f"/Fe:{written[0]}" in commands[0]
    definition = payload_dir / "certified_affine_ffi_cpu.def"
    assert f"/DEF:{definition}" in commands[0]
    assert definition.read_text() == hatch_build.windows_module_definition()
    assert "MSVC" in capsys.readouterr().out


def test_the_windows_build_writes_its_object_file_beside_the_library(
    *, monkeypatch, tmp_path
):
    """The intermediate `.obj` lands beside the install payload."""
    source_dir = tmp_path / hatch_build.PACKAGE_DIR
    source_dir.mkdir(parents=True)
    (source_dir / "certified_affine_ffi_cpu.cc").write_text("// source")
    commands = []

    def fake_compile(*, command, target):
        commands.append(command)
        target.write_bytes(b"dll")
        return target

    monkeypatch.setattr(hatch_build.sys, "platform", "win32")
    monkeypatch.delenv("LCM_SKIP_EXACT_AFFINE", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(
        hatch_build.shutil,
        "which",
        lambda name: "C:/VC/cl.exe" if name == "cl" else None,
    )
    monkeypatch.setattr(hatch_build, "_compile", fake_compile)

    hatch_build.build_exact_affine(root=tmp_path, jax_include_dir="C:/jax/include")

    payload_dir = tmp_path / ".pylcm-native-build"
    assert f"/Fo{payload_dir}\\" in commands[0]  # codespell:ignore


def test_a_non_msvc_cxx_is_refused_on_windows(*, monkeypatch, tmp_path):
    """`CXX` naming a compiler that cannot parse MSVC flags fails with a reason."""
    source_dir = tmp_path / hatch_build.PACKAGE_DIR
    source_dir.mkdir(parents=True)
    (source_dir / "certified_affine_ffi_cpu.cc").write_text("// source")

    monkeypatch.setattr(hatch_build.sys, "platform", "win32")
    monkeypatch.delenv("LCM_SKIP_EXACT_AFFINE", raising=False)
    monkeypatch.setenv("CXX", "C:/msys64/mingw64/bin/g++.exe")

    with pytest.raises(RuntimeError, match="does not take MSVC flags"):
        hatch_build.build_exact_affine(root=tmp_path, jax_include_dir="C:/jax/include")


def test_an_msvc_compatible_cxx_is_accepted_on_windows(*, monkeypatch, tmp_path):
    """`clang-cl` takes MSVC flags, so it builds like `cl` does."""
    source_dir = tmp_path / hatch_build.PACKAGE_DIR
    source_dir.mkdir(parents=True)
    (source_dir / "certified_affine_ffi_cpu.cc").write_text("// source")

    def fake_compile(*, command, target):  # noqa: ARG001
        target.write_bytes(b"dll")
        return target

    monkeypatch.setattr(hatch_build.sys, "platform", "win32")
    monkeypatch.delenv("LCM_SKIP_EXACT_AFFINE", raising=False)
    monkeypatch.setenv("CXX", "C:/LLVM/bin/clang-cl.exe")
    monkeypatch.setattr(hatch_build, "_compile", fake_compile)

    written = hatch_build.build_exact_affine(
        root=tmp_path, jax_include_dir="C:/jax/include"
    )

    assert written == [
        tmp_path / ".pylcm-native-build" / hatch_build.WINDOWS_CPU_LIBRARY
    ]


def test_the_opt_out_builds_no_kernel(monkeypatch):
    """`LCM_SKIP_EXACT_AFFINE=1` installs pylcm without compiling anything."""
    monkeypatch.setenv("LCM_SKIP_EXACT_AFFINE", "1")

    assert hatch_build.build_exact_affine(root=Path("/nowhere")) == []


def test_the_opt_out_names_what_it_gave_up(*, monkeypatch, capsys):
    """Skipping the build says which capability the install will not have."""
    monkeypatch.setenv("LCM_SKIP_EXACT_AFFINE", "1")

    hatch_build.build_exact_affine(root=Path("/nowhere"))

    assert "LCM_SKIP_EXACT_AFFINE" in capsys.readouterr().out


def test_the_opt_out_is_off_unless_asked_for(monkeypatch):
    """`LCM_SKIP_EXACT_AFFINE=0` builds the kernel; the variable is not a mere flag."""
    monkeypatch.setattr(hatch_build.sys, "platform", "linux")
    monkeypatch.setenv("LCM_SKIP_EXACT_AFFINE", "0")
    monkeypatch.setattr(hatch_build.shutil, "which", lambda _name: None)
    monkeypatch.delenv("CXX", raising=False)

    with pytest.raises(RuntimeError, match="No C\\+\\+ compiler"):
        hatch_build.build_exact_affine(
            root=Path("/nowhere"), jax_include_dir="/nowhere"
        )


def test_a_missing_compiler_fails_loudly_without_the_opt_out(monkeypatch):
    """An install that cannot build the kernel fails at build time, not at solve."""
    monkeypatch.setattr(hatch_build.sys, "platform", "linux")
    monkeypatch.delenv("LCM_SKIP_EXACT_AFFINE", raising=False)
    monkeypatch.setattr(hatch_build.shutil, "which", lambda _name: None)
    monkeypatch.delenv("CXX", raising=False)

    with pytest.raises(RuntimeError, match="No C\\+\\+ compiler"):
        hatch_build.build_exact_affine(
            root=Path("/nowhere"), jax_include_dir="/nowhere"
        )


def test_the_opt_out_is_read_before_the_platform_is_consulted(*, monkeypatch, capsys):
    """On a platform that builds nothing anyway, the opt-out still names itself.

    The two early returns are not interchangeable: one reports a deliberate choice
    and the other reports a platform limit, and an install that asked to skip should
    be told that is why, wherever it runs.
    """
    monkeypatch.setattr(hatch_build.sys, "platform", "win32")
    monkeypatch.setenv("LCM_SKIP_EXACT_AFFINE", "1")

    hatch_build.build_exact_affine(root=Path("/nowhere"))

    assert "LCM_SKIP_EXACT_AFFINE" in capsys.readouterr().out


def test_a_compile_failure_reports_diagnostics_written_to_stdout(
    *, tmp_path, monkeypatch
):
    """A failing compile names what the compiler said, whichever stream it used.

    MSVC writes its diagnostics to stdout rather than stderr, so a report built
    only from stderr states an exit code and nothing else — the one moment the
    message is needed.
    """
    monkeypatch.setattr(
        hatch_build.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["cl"],
            returncode=2,
            stdout="kernel.cc(9): error C2065: undeclared\n",
            stderr="",
        ),
    )
    target = tmp_path / "certified_affine_ffi_cpu.dll"

    with pytest.raises(RuntimeError, match="error C2065: undeclared"):
        hatch_build._compile(command=["cl", "/nologo"], target=target)

    assert "error C2065: undeclared" in target.with_suffix(".dll.build.log").read_text()


def test_the_export_manifest_names_every_registered_ffi_target():
    """Every handler the Python wrapper registers is in the Windows export list.

    The two lists are maintained by hand in different files: `hatch_build` owns
    the export surface the MSVC linker is given, `ffi` owns the targets JAX is
    told about. A handler added to one and not the other links a DLL missing
    that export, and no test comparing either manifest to itself can see it.
    """
    assert set(hatch_build.EXACT_AFFINE_HANDLER_SYMBOLS) == set(ffi._TARGETS)
