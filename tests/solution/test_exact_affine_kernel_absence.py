"""The certified path reports a missing kernel when used, not when imported.

Importing the upper envelope must not require a compiled kernel: a platform
without one still runs every part of pylcm that does not ask for an exact
verdict. The refusal belongs at the point a verdict is requested, where it can
name what to build.
"""

import re
from pathlib import Path

import pytest

from _lcm.egm.upper_envelope._exact_affine import ffi
from lcm.exceptions import ExactAffineKernelUnavailableError
from tests import conftest


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


def test_availability_is_true_where_the_kernel_loads():
    """A built kernel reports available, so the suite is not skipped wholesale."""
    assert ffi.kernel_available() is True


@pytest.mark.parametrize(
    ("failed", "asked_for_kernel", "available", "expected"),
    [
        (True, True, False, True),
        (True, True, True, False),
        (True, False, False, False),
        (False, True, False, False),
    ],
)
def test_only_a_kernel_request_on_a_kernelless_host_is_skipped(
    failed, asked_for_kernel, available, expected
):
    """A failure is reported as skipped only when the kernel is truly absent."""
    got = conftest.is_missing_kernel_failure(
        failed=failed, asked_for_kernel=asked_for_kernel, kernel_available=available
    )

    assert got is expected


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
