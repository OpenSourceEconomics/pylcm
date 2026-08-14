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

import hatch_build


def test_the_default_cuda_build_names_architectures():
    """Without an override, the build emits code for real architectures."""
    flags = hatch_build.cuda_arch_flags(nvcc_flags=())

    assert any(flag.startswith("arch=compute_") for flag in flags)


def test_the_default_cuda_build_keeps_a_forward_compatible_fallback():
    """A virtual target is kept, so an architecture not listed can still run."""
    flags = hatch_build.cuda_arch_flags(nvcc_flags=())

    virtual = [flag for flag in flags if flag.endswith(",code=compute_90")]
    assert virtual, f"no virtual target among {flags}"


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
