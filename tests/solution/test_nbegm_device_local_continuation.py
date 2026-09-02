"""The NBEGM tile-local core reads only its device-local child carry.

A fixed, distributed ride-along state (a permanent `kind` sharded one block per
device) never transitions, so a ride cell's continuation depends only on its own
`kind` slice of the next-period child carry. Sharded on that axis, the
continuation read must run device-locally: the optimized `tiled_core` module
performs no collective the envelope step alone does not — in particular no
`all-gather` assembling every `kind` slice of the child carry onto every device.
The envelope step's own collectives (it assembles the published arrays over the
ride cells) are the reference: the oracle envelope core, lowered against the same
period's arguments in the same process, sets the multiset of collectives the
tile-local core may contain.

The check solves the distributed ride-along toy under NBEGM on two forced host
devices and inspects the XLA dumps of the compiled tile-local core and of the
oracle envelope core.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent(
    """
    import collections
    import glob
    import os
    import re

    import jax

    assert jax.device_count() == 2, jax.devices()

    from _lcm.solution import nbegm as solvers
    from tests.test_models import nbegm_ride_along_toy as toy

    captured = []
    original_build_lower_args = solvers._RideAlongNBEGMPeriodKernel.build_lower_args

    def capture_lower_args(self, **kwargs):
        lower_args = original_build_lower_args(self, **kwargs)
        if kwargs.get("core_key", "main") == "main":
            envelope_args = original_build_lower_args(
                self, **{**kwargs, "core_key": "envelope"}
            )
            captured.append((self, envelope_args))
        return lower_args

    solvers._RideAlongNBEGMPeriodKernel.build_lower_args = capture_lower_args

    model = toy.build_model(
        variant="nbegm", n_periods=4, n_liquid=24, n_savings=32, distributed_kind=True
    )
    model.solve(params=toy.build_params(), log_level="debug")
    assert captured, "no production core was lowered"
    for kernel, envelope_args in captured:
        jax.jit(kernel.split_cores()["envelope"]).lower(**envelope_args).compile()

    dump_dir = os.environ["XLA_DUMP_DIR"]
    collective = re.compile(
        r"= (\\S+) (all-gather|all-reduce|all-to-all|collective-permute"
        r"|reduce-scatter)\\("
    )

    def collectives(pattern):
        paths = glob.glob(os.path.join(dump_dir, pattern))
        found = collections.Counter()
        for path in paths:
            with open(path) as handle:
                for match in collective.finditer(handle.read()):
                    found[(match.group(2), match.group(1))] += 1
        return paths, found

    tiled_paths, tiled = collectives("*tiled_core*after_optimizations.txt")
    envelope_paths, envelope = collectives("*envelope_core*after_optimizations.txt")
    assert tiled_paths, f"no tiled_core dump in {dump_dir}"
    assert len(tiled_paths) == len(envelope_paths), (tiled_paths, envelope_paths)
    # The instrument has to be shown firing in this run: the envelope step's own
    # output assembly is where the reference collectives come from.
    assert envelope, "no collective found in the oracle envelope core dumps"
    extra = tiled - envelope
    if extra:
        print("EXTRA-COLLECTIVES", sorted(extra.items()))
    else:
        print("DEVICE-LOCAL-OK", sorted(tiled.items()))
    """
)


def test_nbegm_tile_local_core_does_not_all_gather_child_carry(
    tmp_path: Path,
) -> None:
    """The compiled NBEGM tile-local core reads only its device-local carry.

    With `kind` a fixed distributed ride state, the continuation interpolation
    slices the child carry per device: the optimized `tiled_core` module contains
    no collective beyond those of the oracle envelope core lowered against the
    same arguments, so no `all-gather` of the full carry onto every device.
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
