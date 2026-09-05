"""Public execution-policy configuration."""

from dataclasses import FrozenInstanceError
from inspect import signature

import pytest
from beartype.roar import BeartypeCallHintViolation

import lcm
from lcm import Model
from lcm.execution import ExecutionConfig


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


def test_public_solve_and_simulate_default_to_inert_execution_config() -> None:
    expected = ExecutionConfig()

    assert signature(Model.solve).parameters["execution_config"].default == expected
    assert signature(Model.simulate).parameters["execution_config"].default == expected


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
