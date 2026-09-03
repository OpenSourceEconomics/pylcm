"""A ride-along NB-EGM period's outputs are born in their planned layouts.

With a distributed ride state, the value array, every carry row, and the policy
leave the compiled program on the placement the engine lowered it for, so the
solve loop publishes them without re-placing anything: the value never reaches the
legacy repair, and the continuation roll places every carry leaf on a template
sharding it already has. The check runs on two forced host devices.
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

    from _lcm.solution import backward_induction
    from _lcm.solution import nbegm as solvers
    from tests.test_models import nbegm_ride_along_toy as toy

    def refuse_repair(**_kwargs):
        raise AssertionError("an unplanned value reached the repair path")

    original_match = backward_induction._match_continuation_template_sharding

    def match_is_identity(*, continuation, template):
        placed = original_match(continuation=continuation, template=template)
        for got, want in zip(
            jax.tree.leaves(placed), jax.tree.leaves(continuation), strict=True
        ):
            assert got is want, "a carry leaf was re-placed on its template"
        return placed

    recorded = []
    original_call = solvers._RideAlongArgumentBuilder.__call__

    def record_arguments(self, context):
        arguments = original_call(self, context)
        recorded.append(arguments)
        return arguments

    backward_induction._repair_unplanned_kernel_value = refuse_repair
    backward_induction._match_continuation_template_sharding = match_is_identity
    solvers._RideAlongArgumentBuilder.__call__ = record_arguments

    model = toy.build_model(
        variant="nbegm", n_periods=4, n_liquid=24, n_savings=32, distributed_kind=True
    )
    solution = model.solve(params=toy.build_params(), log_level="debug")
    alive_values = [
        regime_to_value["alive"]
        for regime_to_value in solution.values.values()
        if "alive" in regime_to_value
    ]
    assert len(alive_values) == 3, sorted(solution.values)
    for value in alive_values:
        assert isinstance(value.sharding, jax.NamedSharding), value.sharding
        assert "kind" in tuple(value.sharding.spec), value.sharding.spec

    assert recorded, "no ride-along program was lowered or run"
    for arguments in recorded:
        mesh = arguments["kind"].sharding.mesh
        for leaf in jax.tree.leaves(arguments["next_regime_to_continuation"]):
            assert isinstance(leaf.sharding, jax.NamedSharding), leaf.sharding
            assert leaf.sharding.mesh == mesh
    print("PLANNED-OUTPUTS-OK", len(recorded))
    """
)


def test_ride_along_outputs_are_published_without_any_placement_repair() -> None:
    """Value, carry, and policy leave the program on their planned placement.

    The legacy value repair is never reached, the continuation roll finds every
    carry leaf already on its template's sharding, every published value array of
    the ride-along regime is sharded over `kind`, and the builder places every
    carry leaf it hands the program on the co-map mesh.
    """
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
    assert "PLANNED-OUTPUTS-OK" in result.stdout, result.stdout[-2000:]
