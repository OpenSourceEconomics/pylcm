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

from pathlib import Path

import pytest

import hatch_build


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


def test_windows_module_definition_exports_every_handler():
    """The MSVC linker receives all ten symbols the Python wrapper registers."""
    definition = hatch_build.windows_module_definition().splitlines()

    assert definition[:2] == ["LIBRARY certified_affine_ffi_cpu", "EXPORTS"]
    assert definition[2:] == [
        f"    {name}" for name in hatch_build.EXACT_AFFINE_HANDLER_SYMBOLS
    ]
    assert len(definition[2:]) == 10


def test_the_windows_build_uses_msvc_and_the_export_definition(
    monkeypatch, tmp_path, capsys
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

    written = hatch_build.build_exact_affine(
        root=tmp_path, jax_include_dir="C:/jax/include"
    )

    assert written == [source_dir / hatch_build.WINDOWS_CPU_LIBRARY]
    assert commands
    assert commands[0][0] == "C:/VC/cl.exe"
    assert "/LD" in commands[0]
    assert f"/Fe:{written[0]}" in commands[0]
    definition = source_dir / "certified_affine_ffi_cpu.def"
    assert f"/DEF:{definition}" in commands[0]
    assert definition.read_text() == hatch_build.windows_module_definition()
    assert "MSVC" in capsys.readouterr().out


def test_the_opt_out_builds_no_kernel(monkeypatch):
    """`LCM_SKIP_EXACT_AFFINE=1` installs pylcm without compiling anything."""
    monkeypatch.setenv("LCM_SKIP_EXACT_AFFINE", "1")

    assert hatch_build.build_exact_affine(root=Path("/nowhere")) == []


def test_the_opt_out_names_what_it_gave_up(monkeypatch, capsys):
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


def test_the_opt_out_is_read_before_the_platform_is_consulted(monkeypatch, capsys):
    """On a platform that builds nothing anyway, the opt-out still names itself.

    The two early returns are not interchangeable: one reports a deliberate choice
    and the other reports a platform limit, and an install that asked to skip should
    be told that is why, wherever it runs.
    """
    monkeypatch.setattr(hatch_build.sys, "platform", "win32")
    monkeypatch.setenv("LCM_SKIP_EXACT_AFFINE", "1")

    hatch_build.build_exact_affine(root=Path("/nowhere"))

    assert "LCM_SKIP_EXACT_AFFINE" in capsys.readouterr().out
