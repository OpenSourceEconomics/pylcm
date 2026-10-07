"""CUDA package identity is hashed once per process for one installed inventory."""

from importlib.metadata import PathDistribution
from pathlib import Path

import pytest

from _lcm.solution import cuda_lowering_identity


def _install_fake_distribution(*, site: Path, name: str, version: str) -> Path:
    """Write a minimal installed distribution with one shared library."""
    package = name.replace("-", "_")
    library = site / package / "lib.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(name.encode())
    dist_info = site / f"{package}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    (dist_info / "RECORD").write_text(f"{package}/lib.so,,\n")
    return dist_info


def test_capture_cuda_packages_hashes_an_unchanged_inventory_once(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated identity captures reuse the file digests of the same distributions."""
    installed = tuple(
        PathDistribution(
            _install_fake_distribution(site=tmp_path, name=name, version="1.0")
        )
        for name in ("jax-cuda12-plugin", "jax-cuda12-pjrt")
    )
    monkeypatch.setattr(cuda_lowering_identity, "distributions", lambda: installed)
    hashed: list[Path] = []
    sha256 = cuda_lowering_identity._sha256

    def counting_sha256(path: Path) -> str:
        hashed.append(path)
        return sha256(path)

    monkeypatch.setattr(cuda_lowering_identity, "_sha256", counting_sha256)
    first = cuda_lowering_identity._capture_cuda_packages()
    second = cuda_lowering_identity._capture_cuda_packages()
    assert (len(hashed), first == second) == (2, True)
