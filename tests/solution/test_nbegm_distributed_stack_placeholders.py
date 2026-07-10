"""The envelope core's continuation-stack placeholders carry the co-map sharding.

The ride-along envelope core is AOT-lowered against zero placeholders standing in
for the continuation stacks the continuation core emits at runtime. With a
distributed ride state, those runtime stacks arrive sharded along the flattened
ride-cell axis (one block per device). The placeholders must be committed to that
same sharding: an uncommitted placeholder leaves the compiled-for input sharding
to backend-specific propagation, which may compile the core for replicated stacks
and reject every runtime call. The check runs in a subprocess with two forced
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

    assert jax.device_count() == 2, jax.devices()

    from _lcm.solution import solvers
    from tests.test_models import nbegm_ride_along_toy as toy

    captured = []
    original_build_lower_args = solvers._RideAlongNBEGMPeriodKernel.build_lower_args

    def capture_envelope_lower_args(self, **kwargs):
        lower_args = original_build_lower_args(self, **kwargs)
        if kwargs.get("core_key") == "envelope":
            captured.append(lower_args)
        return lower_args

    solvers._RideAlongNBEGMPeriodKernel.build_lower_args = (
        capture_envelope_lower_args
    )

    model = toy.build_model(
        variant="nbegm", n_periods=4, n_liquid=24, n_savings=32,
        distributed_kind=True,
    )
    model.solve(params=toy.build_params(), log_level="off")

    assert captured, "no envelope core was lowered"
    for lower_args in captured:
        mesh = lower_args["kind"].sharding.mesh
        expected = jax.NamedSharding(mesh=mesh, spec=jax.P("kind"))
        value_sharding = lower_args["cont_value_stack"].sharding
        marginal_sharding = lower_args["cont_marginal_stack"].sharding
        assert value_sharding == expected, (
            f"cont_value_stack placeholder sharding {value_sharding} != {expected}"
        )
        assert marginal_sharding == expected, (
            f"cont_marginal_stack placeholder sharding {marginal_sharding} "
            f"!= {expected}"
        )
    print("PLACEHOLDER-SHARDING-OK")
    """
)


def test_envelope_stack_placeholders_carry_the_co_map_sharding() -> None:
    """With a distributed ride state, the envelope core's lowering placeholders
    for `cont_value_stack` / `cont_marginal_stack` are committed to the same
    ride-cell sharding the continuation core's runtime stacks arrive with."""
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
    assert "PLACEHOLDER-SHARDING-OK" in result.stdout
