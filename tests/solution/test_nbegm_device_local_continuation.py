"""The NBEGM tile-local core reads only its device-local child carry.

A fixed, distributed ride-along state (a permanent `kind` sharded one block per
device) never transitions, so a ride cell's continuation depends only on its own
`kind` slice of the next-period child carry. Sharded on that axis, the
continuation read must run device-locally: the optimized tile-local program
performs no collective beyond those it performs when the same carry arrives
replicated on every device — in particular no `all-gather` assembling every
`kind` slice of the child carry onto every device. The replicated-carry compile
is the reference: it cannot contain a carry gather, and every collective it does
contain belongs to the program's own output assembly over the ride cells.

The check solves the distributed ride-along toy under NBEGM on two forced host
devices, records the live arguments the engine hands the ride-along programs,
and compares the optimized HLO of the program lowered against those arguments
with the HLO of the same program lowered against a replicated copy of the carry.
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
    import re

    import jax

    assert jax.device_count() == 2, jax.devices()

    from _lcm.execution.core_program import core_program_graph
    from _lcm.solution import nbegm as solvers
    from tests.test_models import nbegm_ride_along_toy as toy

    recorded = []
    original_call = solvers._RideAlongArgumentBuilder.__call__

    def record_arguments(self, context):
        arguments = original_call(self, context)
        recorded.append((self, arguments))
        return arguments

    solvers._RideAlongArgumentBuilder.__call__ = record_arguments

    model = toy.build_model(
        variant="nbegm", n_periods=4, n_liquid=24, n_savings=32, distributed_kind=True
    )
    model.solve(params=toy.build_params(), log_level="debug")
    assert recorded, "no ride-along program was lowered"

    programs_by_builder = {}
    for kernel in model._regimes["alive"].solution.period_kernels.values():
        for program in core_program_graph(kernel=kernel).values():
            programs_by_builder.setdefault(id(program.argument_builder), []).append(
                program
            )

    collective = re.compile(
        r"= (\\S+) (all-gather|all-reduce|all-to-all|collective-permute"
        r"|reduce-scatter)\\("
    )

    def collectives(*, function, arguments):
        text = jax.jit(function).lower(**arguments).compile().as_text()
        found = collections.Counter()
        for match in collective.finditer(text):
            found[(match.group(2), match.group(1))] += 1
        return found

    def replicated_carry(arguments):
        carry = arguments["next_regime_to_continuation"]
        mesh = next(
            leaf.sharding.mesh
            for leaf in jax.tree.leaves(carry)
            if isinstance(leaf.sharding, jax.NamedSharding)
        )
        replicated = jax.NamedSharding(mesh, jax.P())
        return {
            **arguments,
            "next_regime_to_continuation": jax.tree.map(
                lambda leaf: jax.device_put(leaf, replicated), carry
            ),
        }

    compared = 0
    sharded_total = collections.Counter()
    for builder, arguments in recorded:
        carry = arguments["next_regime_to_continuation"]
        carry_shardings = {leaf.sharding for leaf in jax.tree.leaves(carry)}
        if not any(
            isinstance(s, jax.NamedSharding) and s.spec != jax.P()
            for s in carry_shardings
        ):
            continue
        for program in programs_by_builder[id(builder)]:
            sharded = collectives(function=program.function, arguments=arguments)
            replicated = collectives(
                function=program.function, arguments=replicated_carry(arguments)
            )
            extra = sharded - replicated
            if extra:
                print("EXTRA-COLLECTIVES", program.name, sorted(extra.items()))
                raise SystemExit(1)
            sharded_total.update(sharded)
            compared += 1
    assert compared, "no program was lowered against a kind-sharded carry"
    # The instrument has to be shown firing in this run: the program's own output
    # assembly over the ride cells is where the reference collectives come from.
    assert sharded_total, "no collective found in any sharded-carry compile"
    print("DEVICE-LOCAL-OK", compared, sorted(sharded_total.items()))
    """
)


def test_nbegm_tile_local_core_does_not_all_gather_child_carry() -> None:
    """The compiled NBEGM tile-local core reads only its device-local carry.

    With `kind` a fixed distributed ride state, the continuation interpolation
    slices the child carry per device: each ride-along program lowered against the
    kind-sharded carry contains no collective beyond those of the same program
    lowered against a replicated copy of that carry, so no `all-gather` of the
    full carry onto every device.
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
    assert result.returncode == 0, result.stderr[-4000:] + result.stdout[-2000:]
    assert "DEVICE-LOCAL-OK" in result.stdout, result.stdout[-2000:]
