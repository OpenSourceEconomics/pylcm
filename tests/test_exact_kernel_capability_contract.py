"""The test harness distinguishes exact-kernel absence from a broken build."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope._exact_affine import ffi
from tests import conftest
from tests.conftest import EXACT_KERNEL_SKIP_REASON, X64_ENABLED


def test_native_payload_is_resolved_from_the_installed_distribution(
    monkeypatch, tmp_path
):
    """The exact kernel remains available when its source checkout disappears."""
    installed_root = tmp_path / "site-packages"

    class InstalledPylcm:
        def locate_file(self, path):
            return installed_root / path

    monkeypatch.setattr(ffi, "distribution", lambda _name: InstalledPylcm())

    assert ffi._installed_native_directory() == installed_root / "_pylcm_native"


@pytest.mark.coverage(backends=("cpu", "gpu-small"), precisions="both")
@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_exact_kernel_answers_in_the_active_precision():
    """The native comparator answers one finite strict ordering in each profile."""
    dtype = jnp.float64 if X64_ENABLED else jnp.float32
    verdict = ffi.certified_affine_compare(
        a_x0=jnp.asarray(0.0, dtype=dtype),
        a_x1=jnp.asarray(1.0, dtype=dtype),
        a_v0=jnp.asarray(2.0, dtype=dtype),
        a_v1=jnp.asarray(2.0, dtype=dtype),
        b_x0=jnp.asarray(0.0, dtype=dtype),
        b_x1=jnp.asarray(1.0, dtype=dtype),
        b_v0=jnp.asarray(1.0, dtype=dtype),
        b_v1=jnp.asarray(1.0, dtype=dtype),
        x_query=jnp.asarray(0.5, dtype=dtype),
    )

    assert int(np.asarray(verdict)) == 1


def test_exact_skip_records_are_sorted_and_duplicate_free():
    """Inventory equality is independent of worker completion order."""
    reason = EXACT_KERNEL_SKIP_REASON
    records = [
        {"nodeid": "tests/z.py::test_z", "reason": reason},
        {"nodeid": "tests/a.py::test_a", "reason": reason},
        {"nodeid": "tests/z.py::test_z", "reason": reason},
    ]

    got = conftest._normalise_exact_kernel_skip_records(records)

    assert got == [
        {"nodeid": "tests/a.py::test_a", "reason": reason},
        {"nodeid": "tests/z.py::test_z", "reason": reason},
    ]


def test_expected_inventory_rejects_unstructured_records(tmp_path: Path):
    """A malformed checked-in inventory fails instead of weakening the gate."""
    path = tmp_path / "inventory.json"
    path.write_text('{"schema_version": 1, "skipped": [{"nodeid": 1}]}')

    with pytest.raises(ValueError, match="Malformed exact-kernel skip records"):
        conftest._read_exact_kernel_skip_inventory(path=path)
