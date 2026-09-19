"""Public execution-policy configuration."""

from dataclasses import FrozenInstanceError
from inspect import signature
from typing import Any, cast

import cloudpickle
import jax
import pytest
from beartype.roar import BeartypeCallHintViolation

import lcm
from _lcm.execution.execution_plan import resolve_execution_config
from lcm import Model
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig, WidthSearch, WidthSearchPolicy
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
            device_memory_bytes=1 << 30,
            axis_widths={"action_product": 2, "subject": 2},
        ),
    )

    assert tuned._model_structure_fingerprint == base._model_structure_fingerprint


def test_execution_devices_defaults_to_every_visible_device() -> None:
    """A model that names no device may use every device JAX reports."""
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")

    assert base.execution_devices == tuple(
        sorted(device.id for device in jax.devices())
    )


def test_a_budget_alone_is_a_complete_configuration() -> None:
    """The single budgeted outer-cohort planner needs no explicit subject pin."""
    config = ExecutionConfig(device_memory_bytes=40 * 1024**3)
    assert config.device_memory_bytes == 40 * 1024**3
    assert dict(config.axis_widths) == {}


def test_budgeted_config_resolves_and_roundtrips() -> None:
    config = ExecutionConfig(
        device_memory_bytes=1024,
        axis_widths={"subject": 2, "action_product": 3},
    )
    restored = cloudpickle.loads(cloudpickle.dumps(config))
    assert restored == config
    resolved = resolve_execution_config(
        config=restored, visible_device_ids=(0,), state_names=frozenset()
    )
    assert resolved.axis_widths == {"subject": 2, "action_product": 3}
    assert resolved.device_memory_bytes == 1024


def test_simulation_sharding_is_an_explicit_legacy_preserving_opt_in() -> None:
    assert ExecutionConfig().simulation_sharding == "legacy"
    config = ExecutionConfig(devices=(0, 1), simulation_sharding="subjects")
    restored = cloudpickle.loads(cloudpickle.dumps(config))
    assert restored == config
    resolved = resolve_execution_config(
        config=restored, visible_device_ids=(0, 1), state_names=frozenset()
    )
    assert resolved.simulation_sharding == "subjects"
    assert resolved.sharded_states == frozenset()
    assert resolved.axis_widths == {}


@pytest.mark.parametrize("mode", ["automatic", "", None, True, 1])
def test_simulation_sharding_rejects_unknown_modes(mode: object) -> None:
    with pytest.raises((BeartypeCallHintViolation, TypeError, ValueError)):
        ExecutionConfig(simulation_sharding=mode)  # ty: ignore[invalid-argument-type]


def test_device_memory_headroom_fraction_defaults_to_a_fifteen_percent_margin() -> None:
    """Budgets are admitted against the pool less an operational safety margin."""
    assert ExecutionConfig().device_memory_headroom_fraction == 0.15


def test_device_memory_headroom_fraction_is_configurable() -> None:
    """A caller who has measured their own envelope sets their own margin."""
    config = ExecutionConfig(
        device_memory_bytes=1024, device_memory_headroom_fraction=0.05
    )

    assert config.device_memory_headroom_fraction == 0.05


def test_device_memory_headroom_fraction_survives_a_roundtrip() -> None:
    """The margin travels with the configuration to a worker process."""
    config = ExecutionConfig(device_memory_headroom_fraction=0.4)

    assert cloudpickle.loads(cloudpickle.dumps(config)) == config


def test_device_memory_headroom_fraction_reaches_the_resolution() -> None:
    """Every phase can read the margin its model was planned under."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(
            device_memory_bytes=1024, device_memory_headroom_fraction=0.2
        ),
        visible_device_ids=(0,),
        state_names=frozenset(),
    )

    assert resolved.device_memory_headroom_fraction == 0.2


def test_execution_config_defaults_to_the_exhaustive_width_search() -> None:
    """A model that declares no policy walks the ranked frontier as before."""
    assert ExecutionConfig().width_search.kind is WidthSearch.EXHAUSTIVE


def test_width_search_policy_rejects_a_refinement_share_above_the_budget() -> None:
    """A refinement share larger than the evaluation budget is unusable."""
    with pytest.raises(ValueError, match="refinement_share"):
        WidthSearchPolicy(max_evaluations=4, refinement_share=5)


@pytest.mark.parametrize("max_evaluations", [0, -1])
def test_width_search_policy_rejects_a_nonpositive_evaluation_budget(
    *, max_evaluations: int
) -> None:
    """A search that may evaluate nothing cannot admit anything."""
    with pytest.raises(ValueError, match="max_evaluations"):
        WidthSearchPolicy(max_evaluations=max_evaluations, refinement_share=0)


def test_width_search_policy_rejects_an_unknown_seed() -> None:
    """Only the two declared seed rules are accepted."""
    with pytest.raises((ValueError, BeartypeCallHintViolation), match="seed"):
        WidthSearchPolicy(seed=cast("Any", "widest-ish"))


def test_width_search_policy_freezes_its_hints() -> None:
    """A later mutation of the caller's hint mapping cannot reach the policy."""
    hints = {"working": {"action_product": 4}}
    policy = WidthSearchPolicy(hints=hints)
    hints["working"]["action_product"] = 8

    assert policy.hints["working"]["action_product"] == 4


def test_execution_config_accepts_the_bounded_width_search() -> None:
    """`ExecutionConfig` keeps a bounded width search the solver dispatches on."""
    config = ExecutionConfig(width_search=WidthSearchPolicy(kind=WidthSearch.BOUNDED))

    assert config.width_search.kind is WidthSearch.BOUNDED
