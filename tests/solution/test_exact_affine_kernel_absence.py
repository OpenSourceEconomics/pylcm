"""Certified exact-affine capability fails at each route's documented boundary.

Importing pylcm remains possible without a compiled kernel. A semantically valid model
selecting ``ExactEnvelope`` fails during ``Model(...)``; certified NBEGM requires the
same payload before returning a certified result, while ordinary NBEGM avoids it.
Low-level verdict entry points retain their own fail-loud guard. File presence,
rather than loadability, distinguishes a supported absence from a
present-but-broken build.
"""

import re
from pathlib import Path

import jax.numpy as jnp
import pytest

from _lcm.egm.upper_envelope._exact_affine import ffi
from lcm.exceptions import ExactAffineKernelUnavailableError, ModelInitializationError
from tests.conftest import EXACT_KERNEL_SKIP_REASON, X64_ENABLED
from tests.test_models import nbegm_tax_toy, negm_kinked_toy
from tests.test_models.dcegm_paper_twin import build_dcegm_model


def test_a_selected_exact_backend_fails_during_model_construction(monkeypatch):
    """A valid model cannot defer an unavailable selected backend until ``solve``."""
    monkeypatch.setattr(ffi, "kernel_available_for_current_backend", lambda: False)

    with pytest.raises(ExactAffineKernelUnavailableError, match="ExactEnvelope"):
        build_dcegm_model()


def test_construction_on_a_kernelless_platform_reports_the_absence_as_such(
    monkeypatch,
):
    """An unavailable exact backend raises the dedicated capability error."""
    monkeypatch.setattr(ffi, "kernel_available_for_current_backend", lambda: False)

    with pytest.raises(ExactAffineKernelUnavailableError) as excinfo:
        build_dcegm_model()

    assert excinfo.errisinstance(ExactAffineKernelUnavailableError)
    assert not excinfo.errisinstance(ModelInitializationError)


def test_ordinary_nbegm_does_not_request_the_installed_kernel(monkeypatch):
    """Ordinary NBEGM remains available without the exact-affine payload."""
    calls = 0

    def record_request() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(ffi, "_ensure_registered", record_request)
    model = nbegm_tax_toy.build_model(
        variant="nbegm",
        envelope_arithmetic="ordinary",
        n_periods=2,
        n_liquid=8,
        n_savings=8,
        savings_max=20.0,
    )

    model.solve(
        params=nbegm_tax_toy.build_params(final_age_alive=1.0), log_level="debug"
    )

    assert calls == 0


def test_current_cpu_backend_requires_the_cpu_library_file(monkeypatch, tmp_path):
    """The skip predicate is false until the selected CPU kernel file exists."""
    library = tmp_path / "libcertified_affine_ffi_cpu.so"
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", library)
    monkeypatch.setattr(ffi.jax, "default_backend", lambda: "cpu")

    assert ffi.kernel_built_for_current_backend() is False
    library.touch()
    assert ffi.kernel_built_for_current_backend() is True


def test_current_gpu_backend_requires_both_cpu_and_cuda_library_files(
    monkeypatch, tmp_path
):
    """A CPU library alone cannot satisfy an exact verdict on a GPU backend."""
    cpu_library = tmp_path / "libcertified_affine_ffi_cpu.so"
    cuda_library = tmp_path / "libcertified_affine_ffi_cuda.so"
    cpu_library.touch()
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", cpu_library)
    monkeypatch.setattr(ffi, "_CUDA_LIBRARY", cuda_library)
    monkeypatch.setattr(ffi.jax, "default_backend", lambda: "gpu")

    assert ffi.kernel_built_for_current_backend() is False
    cuda_library.touch()
    assert ffi.kernel_built_for_current_backend() is True


def test_a_gpu_backend_requires_a_loadable_cuda_exact_kernel(monkeypatch):
    """A loadable CPU library alone does not satisfy an exact GPU request."""
    monkeypatch.setattr(ffi, "kernel_available", lambda: True)
    monkeypatch.setattr(ffi, "cuda_kernel_built", lambda: False)
    monkeypatch.setattr(ffi.jax, "default_backend", lambda: "gpu")

    assert ffi.kernel_available_for_current_backend() is False


def test_a_missing_kernel_is_reported_when_a_verdict_is_requested(monkeypatch):
    """Asking for a verdict without a kernel raises, naming the build task."""
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", Path("/nowhere/libcertified.so"))
    monkeypatch.setattr(ffi, "_REGISTERED", False)

    with pytest.raises(ExactAffineKernelUnavailableError, match="reinstall pylcm"):
        ffi._ensure_registered()


def test_a_missing_kernel_names_the_path_it_looked_for(monkeypatch):
    """The absence message states which file was missing."""
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", Path("/nowhere/libcertified.so"))
    monkeypatch.setattr(ffi, "_REGISTERED", False)

    with pytest.raises(
        ExactAffineKernelUnavailableError, match=re.escape("/nowhere/libcertified.so")
    ):
        ffi._ensure_registered()


def test_availability_is_false_where_the_kernel_file_is_absent(monkeypatch):
    """``kernel_available`` answers rather than raising on supported absence."""
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", Path("/nowhere/libcertified.so"))
    monkeypatch.setattr(ffi, "_REGISTERED", False)

    assert ffi.kernel_available() is False


def test_a_present_kernel_missing_a_target_fails_as_a_broken_build(
    monkeypatch, tmp_path
):
    """A stale shared object is present, not skippable, and names the rebuild."""
    library = tmp_path / "libcertified_affine_ffi_cpu.so"
    library.touch()

    def raise_missing_symbol(**kwargs):  # noqa: ARG001
        msg = "undefined symbol: ExactCellHullF64"
        raise AttributeError(msg)

    monkeypatch.setattr(ffi, "_CPU_LIBRARY", library)
    monkeypatch.setattr(ffi, "_REGISTERED", False)
    monkeypatch.setattr(ffi, "cuda_kernel_built", lambda: False)
    monkeypatch.setattr(ffi, "_register_platform", raise_missing_symbol)
    monkeypatch.setattr(ffi.jax, "default_backend", lambda: "cpu")

    assert ffi.kernel_built_for_current_backend() is True
    with pytest.raises(ExactAffineKernelUnavailableError, match="reinstall pylcm"):
        ffi._ensure_registered()


def test_a_present_unloadable_kernel_is_unavailable_but_not_absent(
    monkeypatch, tmp_path
):
    """Load failure remains a hard build failure rather than a skip condition."""
    library = tmp_path / "libcertified_affine_ffi_cpu.so"
    library.touch()

    def raise_load_error(**kwargs):  # noqa: ARG001
        msg = "missing compiler runtime"
        raise OSError(msg)

    monkeypatch.setattr(ffi, "_CPU_LIBRARY", library)
    monkeypatch.setattr(ffi, "_REGISTERED", False)
    monkeypatch.setattr(ffi, "cuda_kernel_built", lambda: False)
    monkeypatch.setattr(ffi, "_register_platform", raise_load_error)
    monkeypatch.setattr(ffi.jax, "default_backend", lambda: "cpu")

    assert ffi.kernel_built_for_current_backend() is True
    assert ffi.kernel_available() is False


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_availability_is_true_where_the_kernel_loads():
    """A built kernel reports available in the selected backend."""
    assert ffi.kernel_available_for_current_backend() is True


def test_a_platform_with_no_kernel_file_reports_it_is_not_built(monkeypatch):
    """A platform that never built a CPU library reports capability absence."""
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", Path("/nowhere/libcertified.so"))

    assert ffi.kernel_built() is False


def test_a_present_kernel_counts_as_built_before_it_is_loaded(monkeypatch, tmp_path):
    """Presence alone distinguishes a potentially broken build from absence."""
    library = tmp_path / "libcertified_affine_ffi_cpu.so"
    library.touch()
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", library)

    assert ffi.kernel_built() is True


def _f(value):
    """Return ``value`` in the floating precision selected for this suite."""
    return jnp.asarray(value, dtype=jnp.float64 if X64_ENABLED else jnp.float32)


@pytest.mark.parametrize(
    "request_a_verdict",
    [
        lambda: ffi.certified_affine_compare(
            a_x0=_f(0.0),
            a_x1=_f(1.0),
            a_v0=_f(0.0),
            a_v1=_f(1.0),
            b_x0=_f(0.0),
            b_x1=_f(1.0),
            b_v0=_f(1.0),
            b_v1=_f(0.0),
            x_query=_f(0.5),
        ),
        lambda: ffi.exact_affine_read(
            x0=_f(0.0), x1=_f(1.0), v0=_f(0.0), v1=_f(1.0), x_query=_f(0.5)
        ),
        lambda: ffi.exact_affine_handover(
            a_x0=_f(0.0),
            a_x1=_f(1.0),
            a_v0=_f(0.0),
            a_v1=_f(1.0),
            b_x0=_f(0.0),
            b_x1=_f(1.0),
            b_v0=_f(1.0),
            b_v1=_f(0.0),
            left=_f(0.0),
            right=_f(1.0),
        ),
        lambda: ffi.exact_cell_hull(
            left=_f(0.0),
            right=_f(1.0),
            live=jnp.asarray([True]),
            low=jnp.asarray([0], dtype=jnp.int32),
            high=jnp.asarray([1], dtype=jnp.int32),
            endog_grid=_f([0.0, 1.0]),
            value=_f([0.0, 1.0]),
            max_runs=1,
        ),
    ],
    ids=["compare", "read", "handover", "cell_hull"],
)
def test_every_entry_point_reports_a_missing_kernel(monkeypatch, request_a_verdict):
    """No exact entry point reaches XLA without a kernel behind it."""
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", Path("/nowhere/libcertified.so"))
    monkeypatch.setattr(ffi, "_REGISTERED", False)

    with pytest.raises(ExactAffineKernelUnavailableError, match="reinstall pylcm"):
        request_a_verdict()


def test_registration_is_reported_as_done_only_once(monkeypatch, tmp_path):
    """A second verdict request does not re-register existing targets."""
    library = tmp_path / "libcertified_affine_ffi_cpu.so"
    library.touch()
    calls = []
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", library)
    monkeypatch.setattr(ffi, "cuda_kernel_built", lambda: False)
    monkeypatch.setattr(ffi, "_REGISTERED", False)
    monkeypatch.setattr(
        ffi, "_register_platform", lambda **kwargs: calls.append(kwargs["platform"])
    )

    ffi._ensure_registered()
    ffi._ensure_registered()

    assert calls == ["cpu"]


def test_a_nested_regime_reports_its_inner_exact_backend_as_unavailable(monkeypatch):
    """A NEGM regime carries its inner solver's capability gate to model build."""
    monkeypatch.setattr(ffi, "kernel_available_for_current_backend", lambda: False)

    with pytest.raises(ExactAffineKernelUnavailableError, match="ExactEnvelope"):
        negm_kinked_toy.build_model()


def test_a_cuda_backend_without_a_cuda_kernel_is_reported_at_registration(monkeypatch):
    """Registering on a CUDA backend with no CUDA kernel raises, naming the build.

    The CPU kernel alone satisfies registration, so a CUDA-backed solve reaches
    the first compile that needs the device target before anything objects — and
    that compile can be hours into a job. The refusal states what to build while
    the process is still cheap to restart.
    """
    monkeypatch.setattr(ffi, "_REGISTERED", False)
    monkeypatch.setattr(ffi, "cuda_kernel_built", lambda: False)
    monkeypatch.setattr(ffi, "_default_backend", lambda: "gpu")

    with pytest.raises(ExactAffineKernelUnavailableError, match="reinstall pylcm"):
        ffi._ensure_registered()


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_a_cpu_backend_without_a_cuda_kernel_registers(monkeypatch):
    """A CPU-backed solve needs no CUDA kernel, including on a machine with a GPU.

    Registration is asserted to succeed, so the CPU library has to be loadable —
    the one test here that needs a build rather than characterising its absence.
    """
    monkeypatch.setattr(ffi, "_REGISTERED", False)
    monkeypatch.setattr(ffi, "cuda_kernel_built", lambda: False)
    monkeypatch.setattr(ffi, "_default_backend", lambda: "cpu")

    ffi._ensure_registered()

    assert ffi._REGISTERED is True
