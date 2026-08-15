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


def test_the_build_states_that_it_produced_nothing_on_windows(monkeypatch, capsys):
    """A platform that builds no kernel at all says so, rather than exiting quietly."""
    monkeypatch.setattr(hatch_build.sys, "platform", "win32")

    hatch_build.build_exact_affine(root=Path("/nowhere"))

    assert "no kernel" in capsys.readouterr().out


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
