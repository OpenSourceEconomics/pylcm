"""Public execution-policy configuration."""

from dataclasses import FrozenInstanceError
from inspect import signature

import jax
import pytest
from beartype.roar import BeartypeCallHintViolation

import lcm
from lcm import Model
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from tests.test_models.processes import MultiRegimeId, get_multi_regime_model


def test_execution_config_is_a_public_frozen_keyword_only_value() -> None:
    config = ExecutionConfig(device_memory_bytes=1024)

    assert lcm.ExecutionConfig is ExecutionConfig
    assert config.device_memory_bytes == 1024
    with pytest.raises(TypeError):
        ExecutionConfig(1024)  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(FrozenInstanceError):
        config.device_memory_bytes = 2048  # ty: ignore[invalid-assignment]


def test_execution_config_defaults_to_no_device_memory_budget() -> None:
    assert ExecutionConfig().device_memory_bytes is None


@pytest.mark.parametrize("device_memory_bytes", [True, 0, -1])
def test_execution_config_rejects_nonpositive_or_boolean_budgets(
    *, device_memory_bytes: int
) -> None:
    with pytest.raises((TypeError, ValueError), match="device_memory_bytes"):
        ExecutionConfig(device_memory_bytes=device_memory_bytes)


@pytest.mark.parametrize("device_memory_bytes", [1.5, "1024"])
def test_execution_config_rejects_noninteger_budgets(
    *, device_memory_bytes: object
) -> None:
    with pytest.raises(BeartypeCallHintViolation):
        ExecutionConfig(device_memory_bytes=device_memory_bytes)  # ty: ignore[invalid-argument-type]


def test_sharded_states_default_is_empty() -> None:
    """Without a declaration no state carries a device axis."""
    assert ExecutionConfig().sharded_states == ()


def test_devices_default_means_every_visible_device() -> None:
    """`devices=None` is the documented default and resolves at model build."""
    assert ExecutionConfig().devices is None


def test_devices_rejects_duplicate_ids() -> None:
    """A device id may appear once."""
    with pytest.raises(ValueError, match="devices must be distinct"):
        ExecutionConfig(devices=(0, 0))


def test_sharded_states_rejects_duplicate_names() -> None:
    """A state name may appear once."""
    with pytest.raises(ValueError, match="sharded_states must be distinct"):
        ExecutionConfig(sharded_states=("wealth", "wealth"))


def test_model_rejects_an_unknown_sharded_state() -> None:
    """A sharded state that no regime declares is refused at model build."""
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    with pytest.raises(ExecutionPlanningError, match="sharded_states names 'nope'"):
        Model(
            regimes=base.user_regimes,
            ages=base.ages,
            regime_id_class=MultiRegimeId,
            fixed_params=dict(base.fixed_params),
            execution_config=ExecutionConfig(sharded_states=("nope",)),
        )


def test_model_rejects_an_unknown_axis_width() -> None:
    """An axis width for a name no program declares is refused at model build."""
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    with pytest.raises(ExecutionPlanningError, match="axis_widths names 'nope'"):
        Model(
            regimes=base.user_regimes,
            ages=base.ages,
            regime_id_class=MultiRegimeId,
            fixed_params=dict(base.fixed_params),
            execution_config=ExecutionConfig(axis_widths={"nope": 4}),
        )


def test_model_rejects_an_invisible_device_id() -> None:
    """A device id JAX does not report is refused at model build."""
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    with pytest.raises(ExecutionPlanningError, match="device id 999 is not visible"):
        Model(
            regimes=base.user_regimes,
            ages=base.ages,
            regime_id_class=MultiRegimeId,
            fixed_params=dict(base.fixed_params),
            execution_config=ExecutionConfig(devices=(999,)),
        )


def test_solve_has_no_execution_config_parameter() -> None:
    """Hardware-local configuration is fixed when the model is built."""
    assert "execution_config" not in signature(Model.solve).parameters


def test_simulate_has_no_execution_config_parameter() -> None:
    """Hardware-local configuration is fixed when the model is built."""
    assert "execution_config" not in signature(Model.simulate).parameters


def test_model_takes_an_execution_config_parameter() -> None:
    """Hardware-local configuration is declared where the model is built."""
    assert "execution_config" in signature(Model.__init__).parameters


def test_execution_config_does_not_change_the_structure_fingerprint() -> None:
    """Two models differing only in `ExecutionConfig` share a fingerprint."""
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    tuned = Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=MultiRegimeId,
        fixed_params=dict(base.fixed_params),
        execution_config=ExecutionConfig(
            device_memory_bytes=1 << 30, axis_widths={"action_product": 2}
        ),
    )

    assert tuned._model_structure_fingerprint == base._model_structure_fingerprint


def test_execution_devices_defaults_to_every_visible_device() -> None:
    """A model that names no device may use every device JAX reports."""
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")

    assert base.execution_devices == tuple(
        sorted(device.id for device in jax.devices())
    )
