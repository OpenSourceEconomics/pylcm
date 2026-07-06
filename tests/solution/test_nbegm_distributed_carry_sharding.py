"""The ride-along carry template carries the regime's device sharding.

The compiled ride-along cores accept one carry pytree layout across every
period. The envelope core publishes carries whose ride axes inherit the
regime's state sharding, so the template — the compile-time sample and the
terminal-period input — must be placed on the same sharding: a replicated
template would compile the cores for replicated carries and reject every
later period's sharded inputs. The check runs in a subprocess with two forced
host devices so a genuinely sharded mesh exists on CPU.
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

    assert jax.device_count() == 2, jax.devices()

    from _lcm.grids import DiscreteGrid
    from _lcm.solution.solvers import (
        _build_ride_along_carry_template,
        _shard_ride_carry_template,
    )
    from tests.test_models.nbegm_ride_along_toy import ConsumerKind

    grids = {"kind": DiscreteGrid(ConsumerKind, distributed=True)}
    liquid_grid = jnp.linspace(0.1, 30.0, 24)
    template = _build_ride_along_carry_template(
        liquid_grid=liquid_grid, ride_shape=(2,), n_breakpoints=1
    )
    sharded = _shard_ride_carry_template(
        template=template,
        grids=grids,
        ride_along_state_names=("kind",),
    )

    spec = sharded.value.sharding.spec
    assert spec == jax.P("kind"), spec
    assert sharded.endog_grid.sharding.spec == jax.P("kind")
    assert sharded.breakpoints.sharding.spec == jax.P("kind")
    # The scalar taste-shock slot replicates across the mesh.
    assert sharded.taste_shock_scale.sharding.spec == jax.P()
    # Each device holds one `kind` slice.
    assert not sharded.value.sharding.is_fully_replicated

    # Without a distributed ride state the template is returned unchanged.
    plain = _shard_ride_carry_template(
        template=template,
        grids={"kind": DiscreteGrid(ConsumerKind)},
        ride_along_state_names=("kind",),
    )
    assert plain is template
    print("SHARDING-OK")
    """
)


def test_ride_carry_template_is_sharded_like_the_distributed_ride_state() -> None:
    """With a distributed ride state, every carry row array shards its ride
    axis across the mesh while scalars replicate; without one, the template
    passes through untouched."""
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
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "SHARDING-OK" in result.stdout
