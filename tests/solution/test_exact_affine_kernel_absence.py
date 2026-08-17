"""Native exact-kernel availability is checked at the earliest owning boundary.

Importing pylcm remains possible without a compiled kernel. Selecting
`ExactEnvelope` for a model fails during `Model(...)`; low-level verdict entry
points retain their own fail-loud guard.
"""

import re
from pathlib import Path

import jax.numpy as jnp
import pytest

from _lcm.egm.upper_envelope._exact_affine import ffi
from lcm.exceptions import ExactAffineKernelUnavailableError, ModelInitializationError
from tests import conftest
from tests.conftest import X64_ENABLED
from tests.test_models.dcegm_paper_twin import build_dcegm_model


def test_a_selected_exact_backend_fails_during_model_construction(monkeypatch):
    """A model cannot defer an unavailable selected backend until `solve()`."""
    monkeypatch.setattr(ffi, "kernel_available_for_current_backend", lambda: False)

    with pytest.raises(ModelInitializationError, match="ExactEnvelope"):
        build_dcegm_model()


def test_a_gpu_backend_requires_the_cuda_exact_kernel(monkeypatch):
    """A CPU library alone cannot satisfy a selected GPU exact backend."""
    monkeypatch.setattr(ffi, "kernel_available", lambda: True)
    monkeypatch.setattr(ffi, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(ffi.jax, "default_backend", lambda: "gpu")

    assert ffi.kernel_available_for_current_backend() is False


def test_a_missing_kernel_is_reported_when_a_verdict_is_requested(monkeypatch):
    """Asking for a verdict without a kernel raises, naming the build task."""
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", Path("/nowhere/libcertified.so"))
    monkeypatch.setattr(ffi, "_REGISTERED", False)

    with pytest.raises(ExactAffineKernelUnavailableError, match="build-exact-affine"):
        ffi._ensure_registered()


def test_a_missing_kernel_names_the_path_it_looked_for(monkeypatch):
    """The message states which file was absent, not merely that one was."""
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", Path("/nowhere/libcertified.so"))
    monkeypatch.setattr(ffi, "_REGISTERED", False)

    with pytest.raises(
        ExactAffineKernelUnavailableError, match=re.escape("/nowhere/libcertified.so")
    ):
        ffi._ensure_registered()


def test_availability_is_false_where_the_kernel_cannot_be_loaded(monkeypatch):
    """`kernel_available` answers without raising, so callers can branch on it."""
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", Path("/nowhere/libcertified.so"))
    monkeypatch.setattr(ffi, "_REGISTERED", False)

    assert ffi.kernel_available() is False


def test_a_kernel_missing_a_target_is_reported_like_a_missing_kernel(monkeypatch):
    """A library built before a target existed names the rebuild, not a raw symbol.

    A shared object left over from an earlier build loads successfully and only
    fails when a target added since is looked up in it. That is a stale build,
    and it is reported as one.
    """

    def raise_missing_symbol(**kwargs):  # noqa: ARG001
        msg = "undefined symbol: ExactCellHullF64"
        raise AttributeError(msg)

    monkeypatch.setattr(ffi, "_REGISTERED", False)
    monkeypatch.setattr(ffi, "_register_platform", raise_missing_symbol)

    with pytest.raises(ExactAffineKernelUnavailableError, match="build-exact-affine"):
        ffi._ensure_registered()


def test_availability_is_true_where_the_kernel_loads():
    """A built kernel reports available, so the suite is not skipped wholesale."""
    assert ffi.kernel_available() is True


@pytest.mark.parametrize(
    ("failed", "asked_for_kernel", "built", "expected"),
    [
        (True, True, False, True),
        (True, True, True, False),
        (True, False, False, False),
        (False, True, False, False),
    ],
)
def test_only_a_kernel_request_on_a_kernelless_host_is_skipped(
    failed, asked_for_kernel, built, expected
):
    """A failure is reported as skipped only when the kernel is truly absent."""
    got = conftest.is_missing_kernel_failure(
        failed=failed, asked_for_kernel=asked_for_kernel, kernel_built=built
    )

    assert got is expected


def test_a_platform_with_no_kernel_file_reports_it_is_not_built(monkeypatch):
    """Absence is decided by the file, so a platform that never built one skips."""
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", Path("/nowhere/libcertified.so"))

    assert ffi.kernel_built() is False


def test_a_kernel_that_cannot_answer_still_counts_as_built():
    """A present-but-unusable library is a broken build, not an absent one.

    Skipping exists for a platform that has no kernel at all. A build that is
    there and cannot answer has to fail: were it skipped instead, a shared
    object left over from an earlier build would turn the entire certified
    suite green by removing every test that could have caught it.
    """
    assert ffi.kernel_built() is True


def _f(value):
    """Return `value` as an array of the precision the suite is running at."""
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
    """No entry point reaches XLA without a kernel behind it."""
    monkeypatch.setattr(ffi, "_CPU_LIBRARY", Path("/nowhere/libcertified.so"))
    monkeypatch.setattr(ffi, "_REGISTERED", False)

    with pytest.raises(ExactAffineKernelUnavailableError, match="build-exact-affine"):
        request_a_verdict()


def test_registration_is_reported_as_done_only_once(monkeypatch):
    """A second request does not re-register targets already registered."""
    calls = []
    monkeypatch.setattr(ffi, "_REGISTERED", False)
    monkeypatch.setattr(
        ffi, "_register_platform", lambda **kwargs: calls.append(kwargs["platform"])
    )

    ffi._ensure_registered()
    ffi._ensure_registered()

    assert calls.count("cpu") == 1


def test_a_cuda_backend_without_a_cuda_kernel_is_reported_at_registration(monkeypatch):
    """Registering on a CUDA backend with no CUDA kernel raises, naming the build.

    The CPU kernel alone satisfies registration, so a CUDA-backed solve reaches
    the first compile that needs the device target before anything objects — and
    that compile can be hours into a job. The refusal states what to build while
    the process is still cheap to restart.
    """
    monkeypatch.setattr(ffi, "_REGISTERED", False)
    monkeypatch.setattr(ffi, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(ffi, "_default_backend", lambda: "gpu")

    with pytest.raises(ExactAffineKernelUnavailableError, match="build-exact-affine"):
        ffi._ensure_registered()


def test_a_cpu_backend_without_a_cuda_kernel_registers(monkeypatch):
    """A CPU-backed solve needs no CUDA kernel, including on a machine with a GPU."""
    monkeypatch.setattr(ffi, "_REGISTERED", False)
    monkeypatch.setattr(ffi, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(ffi, "_default_backend", lambda: "cpu")

    ffi._ensure_registered()

    assert ffi._REGISTERED is True
