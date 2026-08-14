"""The CUDA build targets real architectures, not PTX alone.

A PTX-only artifact has to be translated by the driver at load, and a driver
older than the toolchain that emitted the PTX refuses it outright — the kernel
does not launch. Naming architectures embeds ready code for them and leaves the
translation path as a fallback rather than the only route.
"""

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
