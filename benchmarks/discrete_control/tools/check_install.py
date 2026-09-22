"""Verify a normal installation imports from the exact pulled Git revision."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


def check_install(*, root: Path, output: Path) -> None:
    """Record version imports and reject missing generated or foreign modules."""
    import _lcm
    import _lcm.version
    import lcm
    import lcm.version

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    assert head == os.environ["PYLCM_SCALING_EXPECTED_HEAD"]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    )
    assert not dirty, dirty
    versions = {
        "lcm": lcm.__version__,
        "public": lcm.version.__version__,
        "internal": _lcm.version.__version__,
    }
    assert len(set(versions.values())) == 1
    modules = {
        name: Path(module.__file__).resolve()
        for name, module in (
            ("lcm", lcm),
            ("_lcm", _lcm),
            ("lcm.version", lcm.version),
            ("_lcm.version", _lcm.version),
        )
    }
    assert all(path.is_relative_to(root / "src") for path in modules.values()), modules
    assert modules["_lcm.version"] == root / "src/_lcm/version.py"
    distribution = importlib.metadata.distribution("pylcm")
    prefix = Path(sys.prefix).resolve()
    assert prefix == root / ".pixi/envs/tests-cuda13"
    direct_url = json.loads(distribution.read_text("direct_url.json"))
    assert Path(unquote(urlparse(direct_url["url"]).path)).resolve() == root
    assert direct_url.get("dir_info", {}).get("editable") is True
    from jax._src import compiler, dispatch
    from jax._src.interpreters import pxla

    from _lcm.egm.upper_envelope._exact_affine import ffi

    native = ffi._DIRECTORY.resolve()  # noqa: SLF001 - installed payload probe
    assert native.is_relative_to(prefix), native
    fingerprints = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in native.iterdir()
        if path.is_file()
    }
    for module in (compiler, dispatch, pxla):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(prefix), path
        fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert distribution.version == lcm.__version__
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "head": head,
                "versions": versions,
                "imports": {name: str(path) for name, path in modules.items()},
                "generated_version_sha256": hashlib.sha256(
                    modules["_lcm.version"].read_bytes()
                ).hexdigest(),
                "python": sys.version,
                "executable": sys.executable,
                "distribution": distribution.version,
                "direct_url": direct_url,
                "native_and_instrumentation_sha256": fingerprints,
                "prefix": str(prefix),
                "lock_sha256": hashlib.sha256(
                    (root / "pixi.lock").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    options = parser.parse_args()
    check_install(root=options.root.resolve(), output=options.output)
