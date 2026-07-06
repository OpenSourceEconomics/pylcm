"""The NBEGM continuation core reads only its device-local child carry.

A fixed, distributed ride-along state (a permanent `kind` sharded one block per
device) never transitions, so a ride cell's continuation depends only on its own
`kind` slice of the next-period child carry. Sharded on that axis, the
continuation interpolation must run device-locally: the optimized
`continuation_core` module contains no `all-gather` collective assembling every
`kind` slice of the child carry onto every device.

The check solves the distributed ride-along toy under NBEGM on two forced host
devices and inspects the XLA dump of the compiled continuation core.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent(
    """
    import glob
    import os
    import sys

    import jax

    assert jax.device_count() == 2, jax.devices()

    from tests.test_models import nbegm_ride_along_toy as toy

    model = toy.build_model(
        variant="nbegm", n_periods=4, n_liquid=24, n_savings=32, distributed_kind=True
    )
    model.solve(params=toy.build_params(), log_level="off")

    dump_dir = os.environ["XLA_DUMP_DIR"]
    hits = glob.glob(
        os.path.join(dump_dir, "*continuation_core*after_optimizations.txt")
    )
    assert hits, f"no continuation_core dump in {dump_dir}"
    gathered = [p for p in hits if "all-gather" in open(p).read()]
    if gathered:
        print("ALL-GATHER-PRESENT", os.path.basename(gathered[0]))
    else:
        print("DEVICE-LOCAL-OK")
    """
)


@pytest.mark.xfail(
    strict=True,
    reason="The NBEGM continuation core interpolates the full child carry; the "
    "device-local co-map of fixed distributed ride states is not yet wired in.",
)
def test_nbegm_continuation_core_does_not_all_gather_child_carry(
    tmp_path: Path,
) -> None:
    """The compiled NBEGM continuation core reads only its device-local carry.

    With `kind` a fixed distributed ride state, the continuation interpolation
    slices the child carry per device — the optimized `continuation_core` module
    inserts no `all-gather` of the full carry onto every device.
    """
    dump_dir = tmp_path / "xla-dump"
    dump_dir.mkdir()
    env = {
        **os.environ,
        "XLA_FLAGS": (
            "--xla_force_host_platform_device_count=2 "
            f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text "
            "--xla_gpu_autotune_level=0"
        ),
        "JAX_PLATFORMS": "cpu",
        "XLA_DUMP_DIR": str(dump_dir),
    }
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "DEVICE-LOCAL-OK" in result.stdout, result.stdout[-2000:]
