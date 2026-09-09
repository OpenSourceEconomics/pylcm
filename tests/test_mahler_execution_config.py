"""Execution configuration of the public Mahler and Yum model builder."""

from lcm import ExecutionConfig
from lcm_examples import mahler_yum_2024


def test_create_model_applies_explicit_execution_configuration() -> None:
    """The builder configures the action and cell widths and memory budget."""
    builder = getattr(mahler_yum_2024, "create_model", None)
    assert callable(builder), "The Mahler example needs a configurable model builder."
    model = builder(
        execution_config=ExecutionConfig(
            axis_widths={"action_product": 64, "cell": 4096},
            device_memory_bytes=2**30,
        )
    )

    assert (
        dict(model._execution.axis_widths),
        model._execution.device_memory_bytes,
    ) == ({"action_product": 64, "cell": 4096}, 2**30)


def test_create_model_preserves_default_execution_configuration() -> None:
    """Omitted controls keep unbudgeted planning and automatic width selection."""
    model = mahler_yum_2024.create_model()

    assert (
        dict(model._execution.axis_widths),
        model._execution.device_memory_bytes,
        model._execution.donate_buffers,
        model.execution_devices,
    ) == ({}, None, True, mahler_yum_2024.MAHLER_YUM_MODEL.execution_devices)


def test_create_model_execution_configuration_preserves_economic_identity() -> None:
    """Explicit widths and a budget preserve the exported model's economics."""
    model = mahler_yum_2024.create_model(
        execution_config=ExecutionConfig(
            axis_widths={"action_product": 64, "cell": 4096},
            device_memory_bytes=2**30,
        )
    )

    assert (
        model._model_structure_fingerprint
        == mahler_yum_2024.MAHLER_YUM_MODEL._model_structure_fingerprint
    )
