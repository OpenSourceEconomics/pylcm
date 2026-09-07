"""A warm simulate call performs no JAX trace, lowering, or compile."""

import jax
import jax.numpy as jnp
import pytest

from _lcm.utils.logging import LogLevel
from benchmarks.asv._dispatch_counters import count_dispatches
from benchmarks.asv._simulation_witnesses import (
    MULTI_INITIAL_CONDITIONS,
    WITNESSES,
)
from lcm import Model
from lcm.typing import FloatND
from tests.test_models.processes import (
    MultiRegimeId,
    get_multi_regime_model,
    get_multi_regime_params,
)

LOG_LEVELS: tuple[LogLevel, ...] = ("off", "warning", "progress", "debug")


def _double(x: FloatND) -> FloatND:
    """Return twice the input."""
    return 2.0 * x


def _warm_then_count(*, witness: str, log_level: LogLevel) -> tuple[int, int, int]:
    """Return the traces, lowerings, and compiles of a second simulate call."""
    model, params, initial_conditions = WITNESSES[witness]()
    solution = model.solve(params=params, log_level="off")
    model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level=log_level,
        seed=0,
    )
    with count_dispatches() as counts:
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level=log_level,
            seed=0,
        )
    return counts.traces, counts.lowerings, counts.compiles


@pytest.mark.parametrize("witness", sorted(WITNESSES))
@pytest.mark.parametrize("log_level", LOG_LEVELS)
def test_second_simulate_call_dispatches_nothing(
    *, witness: str, log_level: LogLevel
) -> None:
    """After one warm call, a simulate call at any log level performs no compile."""
    assert _warm_then_count(witness=witness, log_level=log_level) == (0, 0, 0)


def test_counters_report_one_of_each_for_a_freshly_compiled_function() -> None:
    """Compiling a function inside the block is reported once per stage."""
    argument = jnp.arange(3.0)
    with count_dispatches() as counts:
        jax.jit(_double)(argument)
    assert (counts.traces, counts.lowerings, counts.compiles) == (1, 1, 1)


def test_cold_simulate_call_compiles() -> None:
    """A simulate call on a freshly built model issues backend compilations."""
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    model = Model(
        regimes=dict(base.user_regimes),
        regime_id_class=MultiRegimeId,
        ages=base.ages,
        fixed_params=dict(base.fixed_params),
    )
    params = get_multi_regime_params("normal")
    solution = model.solve(params=params, log_level="off")
    with count_dispatches() as counts:
        model.simulate(
            params=params,
            initial_conditions=MULTI_INITIAL_CONDITIONS,
            solution=solution,
            log_level="off",
            seed=0,
        )
    assert counts.compiles > 0
