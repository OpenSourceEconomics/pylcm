"""Subject sharding sees a regime's capability through its fixed-param binding.

A regime with reachable fixed params reaches the runtime with every simulation
program's body wrapped in a `functools.partial` that binds them. A partial
proxies no attribute of the callable it wraps, so a structural read of the
subject-sharding declaration taken off the wrapper reports the model as
incapable and refuses the run. The declaration is a property of the body, and
binding a shared scalar into it partitions nothing, so the same model must
simulate identically on one, two and four devices.

Each device count runs in its own subprocess with forced host devices, the way
the topology lanes do, so the witness does not depend on the ambient topology.
"""

import dataclasses
import functools
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from _lcm.simulation.subject_parallel import declared_subject_shard_arg_names

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent(
    """
    import sys

    import jax
    import jax.numpy as jnp
    import pandas as pd

    from lcm import AgeGrid, ExecutionConfig, LinSpacedGrid, Model
    from tests.test_models.deterministic.regression import (
        START_AGE,
        RegimeId,
        dead,
        get_params,
        working_life,
    )

    n_devices = int(sys.argv[1])
    assert jax.device_count() == n_devices, jax.devices()

    N_PERIODS = 2
    N_SUBJECTS = 4
    final_age_alive = START_AGE + N_PERIODS - 2
    # A regime-level fixed param makes `_partial_fixed_params_into_regimes`
    # rebuild every decision, transition and route program of this regime.
    FIXED = {"working_life": {"next_wealth": {"interest_rate": 0.05}}}


    def build(*, execution_config, fixed_params):
        return Model(
            regimes={
                "working_life": working_life.replace(
                    active=lambda age: age <= final_age_alive,
                    states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
                    actions={
                        "labor_supply": working_life.actions["labor_supply"],
                        "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                    },
                ),
                "dead": dead,
            },
            ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
            regime_id_class=RegimeId,
            fixed_params=fixed_params,
            execution_config=execution_config,
        )


    supplied = get_params(n_periods=N_PERIODS)
    free = {
        "discount_factor": supplied["discount_factor"],
        "working_life": {
            "utility": supplied["working_life"]["utility"],
            "next_regime": supplied["working_life"]["next_regime"],
        },
    }
    ids = tuple(device.id for device in jax.devices()[:n_devices])
    initial = {
        "wealth": jnp.linspace(1.0, 3.0, N_SUBJECTS),
        "age": jnp.full(N_SUBJECTS, float(START_AGE)),
        "regime_id": jnp.full(N_SUBJECTS, RegimeId.working_life, dtype=jnp.int32),
    }


    def simulate(model):
        solution = model.solve(params=free, log_level="off")
        return model.simulate(
            params=free,
            solution=solution,
            initial_conditions=initial,
            seed=17,
            log_level="off",
        ).to_dataframe()


    # The same fixed-param model, once unsharded on a single device and once
    # partitioned over every visible device.
    reference = simulate(
        build(
            execution_config=ExecutionConfig(devices=(ids[0],)),
            fixed_params=FIXED,
        )
    )
    actual = simulate(
        build(
            execution_config=ExecutionConfig(
                devices=ids,
                sharded_states=(),
                simulation_sharding="subjects",
            ),
            fixed_params=FIXED,
        )
    )
    pd.testing.assert_frame_equal(
        actual, reference, check_exact=False, rtol=2e-6, atol=2e-6
    )
    print("SUBJECT-SHARD-FIXED-PARAMS-OK")
    """
)


@pytest.mark.parametrize("n_devices", [1, 2, 4])
def test_subject_sharding_accepts_a_regime_with_fixed_params(*, n_devices: int) -> None:
    """A fixed-param regime simulates under subject sharding, matching the
    unsharded rows, instead of being refused as not declaring independent
    leading-axis subject outputs."""
    env = {
        **os.environ,
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={n_devices}",
        "JAX_PLATFORMS": "cpu",
    }
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SCRIPT, str(n_devices)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "SUBJECT-SHARD-FIXED-PARAMS-OK" in result.stdout


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Declared:
    """A minimal body declaring the subject-shardable capability."""

    names: tuple[str, ...]

    @property
    def subject_shard_arg_names(self) -> tuple[str, ...]:
        return self.names

    def __call__(self, **kwargs: object) -> object:
        return kwargs


def test_a_keyword_binding_does_not_hide_the_declaration() -> None:
    """The names survive one and several layers of keyword binding."""
    body = _Declared(names=("states", "keys"))
    once = functools.partial(body, interest_rate=0.05)
    twice = functools.partial(once, disutility_of_work=0.5)
    assert declared_subject_shard_arg_names(function=body) == ("states", "keys")
    assert declared_subject_shard_arg_names(function=once) == ("states", "keys")
    assert declared_subject_shard_arg_names(function=twice) == ("states", "keys")


def test_a_bound_subject_argument_is_no_longer_offered_for_partitioning() -> None:
    """A name a binding already supplies is not a per-dispatch operand."""
    bound = functools.partial(_Declared(names=("states", "keys")), keys=object())
    assert declared_subject_shard_arg_names(function=bound) == ("states",)


def test_an_undeclared_body_and_a_positional_binding_are_both_refused() -> None:
    """Absent capability and positional binding both report no declaration."""

    def plain(**kwargs: object) -> object:
        return kwargs

    declared = cast("Callable[..., object]", _Declared(names=("states",)))
    positional = functools.partial(declared, object())
    assert declared_subject_shard_arg_names(function=plain) is None
    assert declared_subject_shard_arg_names(function=positional) is None
