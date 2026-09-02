"""Co-mapped ride states place every lowering argument on the co-map mesh.

With a distributed ride state the ride-along cores run on that state's mesh. Two
arguments do not arrive there on their own and are committed explicitly:

- the production core's child carry for a target that does not depend on the
  distributed state (a terminal carry), produced on a single device and
  replicated onto the mesh for lowering and at runtime alike;
- the oracle envelope core's zero placeholders for the continuation stacks, which
  stand in for stacks the oracle continuation core emits sharded along the
  flattened ride-cell axis (one block per device).

An uncommitted argument leaves the compiled-for input sharding to backend-specific
propagation, which can compile a core for a sharding the runtime argument does not
have and reject every call. The check runs in a subprocess with two forced host
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
            captured.append((lower_args, envelope_args))
        return lower_args

    solvers._RideAlongNBEGMPeriodKernel.build_lower_args = capture_lower_args

    model = toy.build_model(
        variant="nbegm", n_periods=4, n_liquid=24, n_savings=32,
        distributed_kind=True,
    )
    model.solve(params=toy.build_params(), log_level="debug")

    assert captured, "no production core was lowered"
    for main_args, envelope_args in captured:
        mesh = main_args["kind"].sharding.mesh
        for leaf in jax.tree.leaves(main_args["next_regime_to_continuation"]):
            assert isinstance(leaf.sharding, jax.NamedSharding), leaf.sharding
            assert leaf.sharding.mesh == mesh, f"{leaf.sharding} not on {mesh}"
        expected = jax.NamedSharding(mesh=mesh, spec=jax.P("kind"))
        value_sharding = envelope_args["cont_value_stack"].sharding
        marginal_sharding = envelope_args["cont_marginal_stack"].sharding
        assert value_sharding == expected, (
            f"cont_value_stack placeholder sharding {value_sharding} != {expected}"
        )
        assert marginal_sharding == expected, (
            f"cont_marginal_stack placeholder sharding {marginal_sharding} "
            f"!= {expected}"
        )
    print("CO-MAP-SHARDING-OK")
    """
)


def test_co_map_lowering_arguments_live_on_the_co_map_mesh() -> None:
    """With a distributed ride state, the production core's child carry is
    committed to the co-map mesh at every period — including the terminal carry
    that does not depend on the distributed state — and the oracle envelope core's
    stack placeholders carry the ride-cell sharding of the oracle's runtime
    stacks; the distributed solve runs to completion."""
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
    assert "CO-MAP-SHARDING-OK" in result.stdout
