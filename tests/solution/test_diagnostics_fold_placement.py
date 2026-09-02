"""The NaN/Inf diagnostics fold accepts value arrays on any device placement.

A solved value array is born wherever its program's planned output layout puts
it: a terminal regime on a single committed device, a regime with a distributed
state sharded across the mesh. The running NaN/Inf flags fold every such array
into one scalar, so the fold must not require all value arrays of a solve to
share one placement. The check runs in a subprocess with two forced host
devices so a genuinely sharded mesh exists on CPU.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp

    from _lcm.solution.diagnostics import (
        _fold_period_diagnostics,
        _init_diagnostic_accumulators,
    )
    from lcm import AgeGrid

    assert jax.device_count() == 2, jax.devices()
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("kind",))
    sharded = jax.NamedSharding(mesh, jax.P("kind", None))
    single = jax.sharding.SingleDeviceSharding(devices[0])

    committed_single = jax.device_put(jnp.zeros((4, 3)), single)
    committed_sharded = jax.device_put(jnp.zeros((4, 3)), sharded)
    committed_sharded = committed_sharded.at[1, 2].set(jnp.nan)
    assert committed_single.committed and committed_sharded.committed

    rows, mins, maxs, means, any_nan, any_inf = _init_diagnostic_accumulators()
    ages = AgeGrid(start=0, stop=2, step="Y")
    common = dict(
        ages=ages,
        diagnostics_enabled=True,
        stats_enabled=True,
        diagnostic_rows=rows,
        diagnostic_min=mins,
        diagnostic_max=maxs,
        diagnostic_mean=means,
    )
    any_nan, any_inf = _fold_period_diagnostics(
        V_arr=committed_single, regime_name="dead", period=1,
        running_any_nan=any_nan, running_any_inf=any_inf, **common,
    )
    any_nan, any_inf = _fold_period_diagnostics(
        V_arr=committed_sharded, regime_name="alive", period=0,
        running_any_nan=any_nan, running_any_inf=any_inf, **common,
    )
    assert bool(any_nan) is True, any_nan
    assert bool(any_inf) is False, any_inf
    assert jnp.stack(mins).shape == (2,)
    assert jnp.stack(maxs).shape == (2,)
    assert jnp.stack(means).shape == (2,)
    print("FOLD-PLACEMENT-OK")
    """
)


def test_diagnostics_fold_accepts_mixed_committed_placements() -> None:
    """Folding a committed single-device value and then a mesh-sharded value
    yields the correct NaN flag, and the per-period stats still stack, instead
    of raising a device-mismatch error."""
    env = {
        **os.environ,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        "JAX_PLATFORMS": "cpu",
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
    assert "FOLD-PLACEMENT-OK" in result.stdout
