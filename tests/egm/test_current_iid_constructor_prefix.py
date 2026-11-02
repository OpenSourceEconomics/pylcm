"""Original constructor prefix shares entry admission; dated factories stay closed."""

from types import MappingProxyType
from typing import Never

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.execution_plan import (
    resolve_execution_config,
)
from _lcm.regime_building.fixed_regime_support import _evaluate_fixed_function
from _lcm.typing import PytreeValue
from lcm import ExecutionConfig, Regime
from lcm.exceptions import (
    ExecutionPlanningError,
)
from lcm.typing import FloatND
from tests.egm.test_current_iid_generation_materialization import _owner

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def _original_log1p_declaration(*, value: FloatND) -> FloatND:
    return jnp.log1p(value)


def test_passive_canonical_jit_does_not_admit_executable_validator_early_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unchanged canonical callback still has its executable-stage refusal."""
    from _lcm.egm import validation_stages  # noqa: PLC0415

    def must_not_trace(**_: PytreeValue) -> Never:
        raise AssertionError("Canonical EARLY validator traced before refusal.")

    monkeypatch.setattr(validation_stages, "prepare_validation_recipe", must_not_trace)
    value = jnp.asarray(0.25)
    owner = _owner(roots=(value,))
    try:
        with pytest.raises(ExecutionPlanningError, match="JAX_EARLY"):
            owner.evaluate_validation(
                function=_original_log1p_declaration, arguments={"value": value}
            )
        assert not owner.operations.cache
        assert not owner._pending
    finally:
        owner.close()


def _unsupported_fixed_probability(*, value: FloatND) -> FloatND:
    return jnp.sin(value)


def _invalid_fixed_probability(*, value: FloatND) -> FloatND:
    del value
    raise ValueError("An ordinary invalid fixed law proves no structural zero.")


def _terminal_utility() -> float:
    return 0.0


def test_fixed_zero_pruning_preserves_ordinary_error_but_propagates_admission() -> None:
    regime = Regime(functions={"utility": _terminal_utility})
    fixed = MappingProxyType({"r__law__value": np.asarray(0.5)})
    owner = _owner()
    try:
        assert (
            _evaluate_fixed_function(
                func=_invalid_fixed_probability,
                path=("r", "law"),
                side="solve",
                regime=regime,
                fixed_flat=fixed,
                ancestors=(),
            )
            is None
        )
        assert (
            _evaluate_fixed_function(
                func=_invalid_fixed_probability,
                materializer=owner,
                path=("r", "law"),
                side="solve",
                regime=regime,
                fixed_flat=fixed,
                ancestors=(),
            )
            is None
        )
        with pytest.raises(ExecutionPlanningError, match="JAX_EARLY"):
            _evaluate_fixed_function(
                func=_unsupported_fixed_probability,
                materializer=owner,
                path=("r", "law"),
                side="solve",
                regime=regime,
                fixed_flat=fixed,
                ancestors=(),
            )
    finally:
        owner.close()


def test_hardware_prefix_reuses_pool_rule_and_final_name_validation_order() -> None:
    config = ExecutionConfig(
        devices=(3,),
        device_memory_bytes=2048,
        device_memory_headroom_fraction=0.25,
    )
    plain = resolve_execution_config(
        config=config,
        visible_device_ids=(7, 3),
        device_pool_limit_bytes={3: 1024},
        state_names=frozenset(),
    )
    assert plain.device_memory_bytes == 768
    bad = ExecutionConfig(devices=(9,), sharded_states=("missing",))
    with pytest.raises(ExecutionPlanningError, match="sharded_states"):
        resolve_execution_config(
            config=bad, visible_device_ids=(3,), state_names=frozenset()
        )
