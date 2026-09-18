"""Resolution of the effective device-memory budget at device binding.

A caller may hand pylcm the whole allocator pool as `device_memory_bytes`. The
resolution keeps an operational headroom off that pool, so the ceiling the
admission inequalities are checked against is the pool limit less the headroom,
never the pool limit itself.
"""

import logging

import pytest
from beartype.roar import BeartypeCallHintViolation

from _lcm.execution.execution_plan import (
    resolve_execution_config,
    visible_device_ids,
    visible_device_pool_limits,
)
from lcm import Model
from lcm.execution import ExecutionConfig
from tests.test_models.processes import MultiRegimeId, get_multi_regime_model

_POOL_LIMIT = 48_000_000_000


class _FakeMemoryStatsDevice:
    """A device reporting one allocator pool limit, as JAX devices do."""

    def __init__(self, *, device_id: int, bytes_limit: int) -> None:
        self.id = device_id
        self._bytes_limit = bytes_limit

    def memory_stats(self) -> dict[str, int]:
        """Return the allocator counters keyed as the JAX backends key them."""
        return {"bytes_limit": self._bytes_limit, "bytes_in_use": 0}


class _FakeStatelessDevice:
    """A device whose backend reports no allocator counters, as CPU does."""

    def __init__(self, *, device_id: int) -> None:
        self.id = device_id

    def memory_stats(self) -> None:
        """Return the `None` the CPU backend returns."""
        return


class _FakeRaisingDevice:
    """A device whose allocator query fails."""

    def __init__(self, *, device_id: int) -> None:
        self.id = device_id

    def memory_stats(self) -> dict[str, int]:
        """Raise the way a backend without the query raises."""
        raise RuntimeError("memory_stats is unsupported on this backend")


def test_a_request_at_the_pool_limit_is_capped_by_the_headroom() -> None:
    """A budget equal to the pool keeps the default fraction of it free."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    assert resolved.device_memory_bytes == _POOL_LIMIT - 7_200_000_000


def test_the_capped_request_is_recorded_next_to_the_effective_budget() -> None:
    """The request a caller made stays readable after the cap applies."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    assert resolved.requested_device_memory_bytes == _POOL_LIMIT


def test_an_already_conservative_request_is_not_reduced_again() -> None:
    """A request below the capped pool limit is the effective budget."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=1_000_000),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    assert resolved.device_memory_bytes == 1_000_000


def test_the_cap_takes_the_smallest_selected_device_limit() -> None:
    """A heterogeneous selection is bounded by its smallest device."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT),
        visible_device_ids=(0, 1),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT, 1: 10_000_000_000},
    )

    assert resolved.device_memory_bytes == 8_500_000_000


def test_an_unselected_device_limit_does_not_bound_the_budget() -> None:
    """Only the devices the model runs on contribute a cap."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT, devices=(0,)),
        visible_device_ids=(0, 1),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT, 1: 10_000_000_000},
    )

    assert resolved.device_memory_bytes == _POOL_LIMIT - 7_200_000_000


def test_a_device_without_an_allocator_limit_contributes_no_cap() -> None:
    """A backend reporting no pool limit leaves the request as it stands."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: None},
    )

    assert resolved.device_memory_bytes == _POOL_LIMIT


def test_a_zero_headroom_fraction_keeps_the_full_pool_limit() -> None:
    """An explicit zero margin admits the whole pool, as before the policy."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(
            device_memory_bytes=_POOL_LIMIT, device_memory_headroom_fraction=0.0
        ),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    assert resolved.device_memory_bytes == _POOL_LIMIT


def test_an_unbudgeted_request_stays_unbudgeted() -> None:
    """`None` keeps its meaning: no ceiling, whatever the device reports."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    assert resolved.device_memory_bytes is None


def test_an_unbudgeted_request_records_no_pool_limit() -> None:
    """Nothing about the device is consulted when no budget is declared."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    assert dict(resolved.device_pool_limit_bytes) == {}


def test_the_selected_pool_limits_are_published_on_the_resolution() -> None:
    """A budgeted model publishes the limit it planned each device against."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT),
        visible_device_ids=(0, 1),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT, 1: None},
    )

    assert dict(resolved.device_pool_limit_bytes) == {0: _POOL_LIMIT, 1: None}


def test_the_resolution_publishes_the_headroom_fraction_it_applied() -> None:
    """The margin a model was planned under is readable from the resolution."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(
            device_memory_bytes=_POOL_LIMIT, device_memory_headroom_fraction=0.25
        ),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    assert resolved.device_memory_headroom_fraction == 0.25


@pytest.mark.parametrize("fraction", [1.0, 1.5, -0.1, float("nan")])
def test_execution_config_rejects_an_unusable_headroom_fraction(
    *, fraction: float
) -> None:
    """The margin is a fraction of the pool strictly below one, never NaN."""
    with pytest.raises(ValueError, match="device_memory_headroom_fraction"):
        ExecutionConfig(device_memory_headroom_fraction=fraction)


@pytest.mark.parametrize("fraction", [True, 0, 1])
def test_execution_config_rejects_a_non_float_headroom_fraction(
    *, fraction: object
) -> None:
    """An exact float keeps `0` from reading as "no margin" by accident."""
    with pytest.raises((TypeError, BeartypeCallHintViolation)):
        ExecutionConfig(device_memory_headroom_fraction=fraction)  # ty: ignore[invalid-argument-type]


def test_the_budget_summary_names_the_request_and_the_effective_ceiling() -> None:
    """The summary carries every number the cap decision was made from."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    summary = resolved.device_memory_budget_summary()

    assert summary == (
        "Device-memory budget: requested 48000000000 bytes; headroom fraction "
        "0.15; per-device pool limits: device 0: limit 48000000000 bytes, "
        "headroom 7200000000 bytes; effective 40800000000 bytes; capped by "
        "device headroom."
    )


def test_the_budget_summary_reports_an_uncapped_request_as_uncapped() -> None:
    """A request the devices can host is reported as passing through."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=1_000_000),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    assert resolved.device_memory_budget_summary().endswith("no cap applied.")


def test_the_budget_summary_reports_an_unbudgeted_model_as_unbudgeted() -> None:
    """Without a budget the summary consults and reports no device."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(), visible_device_ids=(0,), state_names=frozenset()
    )

    assert resolved.device_memory_budget_summary() == (
        "Device-memory budget: none requested; admission is unbudgeted."
    )


def test_a_capped_budget_is_logged_as_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silently planning against less memory than asked for is not acceptable."""
    with caplog.at_level(logging.INFO, logger="_lcm.execution.execution_plan"):
        resolve_execution_config(
            config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT),
            visible_device_ids=(0,),
            state_names=frozenset(),
            device_pool_limit_bytes={0: _POOL_LIMIT},
        )

    assert [record.levelno for record in caplog.records] == [logging.WARNING]


def test_an_uncapped_budget_is_logged_at_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A budget that passes through is still recorded, without alarming."""
    with caplog.at_level(logging.INFO, logger="_lcm.execution.execution_plan"):
        resolve_execution_config(
            config=ExecutionConfig(device_memory_bytes=1_000_000),
            visible_device_ids=(0,),
            state_names=frozenset(),
            device_pool_limit_bytes={0: _POOL_LIMIT},
        )

    assert [record.levelno for record in caplog.records] == [logging.INFO]


def test_an_unbudgeted_resolution_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No budget, no decision to report."""
    with caplog.at_level(logging.INFO, logger="_lcm.execution.execution_plan"):
        resolve_execution_config(
            config=ExecutionConfig(), visible_device_ids=(0,), state_names=frozenset()
        )

    assert caplog.records == []


def test_the_cap_note_is_empty_when_no_cap_applies() -> None:
    """A diagnostic gains nothing to say when the request was honoured."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=1_000_000),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    assert resolved.device_memory_cap_note() == ""


def test_the_cap_note_names_both_budgets_when_they_differ() -> None:
    """An admission refusal says which ceiling it was measured against."""
    resolved = resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT),
        visible_device_ids=(0,),
        state_names=frozenset(),
        device_pool_limit_bytes={0: _POOL_LIMIT},
    )

    assert resolved.device_memory_cap_note() == (
        " The effective budget is 40800000000 bytes of the requested "
        "48000000000 bytes after a device-memory headroom fraction of 0.15."
    )


def test_visible_device_pool_limits_reads_a_reported_limit() -> None:
    """A device reporting `bytes_limit` contributes it by id."""
    devices = (_FakeMemoryStatsDevice(device_id=3, bytes_limit=_POOL_LIMIT),)

    assert dict(visible_device_pool_limits(devices=devices)) == {3: _POOL_LIMIT}


def test_visible_device_pool_limits_tolerates_a_stateless_device() -> None:
    """A CPU device reports no counters and contributes `None`."""
    devices = (_FakeStatelessDevice(device_id=0),)

    assert dict(visible_device_pool_limits(devices=devices)) == {0: None}


def test_visible_device_pool_limits_tolerates_a_failing_query() -> None:
    """A backend that raises on the query contributes `None`, not an error."""
    devices = (_FakeRaisingDevice(device_id=0),)

    assert dict(visible_device_pool_limits(devices=devices)) == {0: None}


def test_visible_device_pool_limits_tolerates_a_missing_counter() -> None:
    """Counters without `bytes_limit` contribute `None`."""

    class _NoLimit(_FakeMemoryStatsDevice):
        def memory_stats(self) -> dict[str, int]:
            """Return counters that omit the pool limit."""
            return {"bytes_in_use": 17}

    devices = (_NoLimit(device_id=0, bytes_limit=_POOL_LIMIT),)

    assert dict(visible_device_pool_limits(devices=devices)) == {0: None}


def test_visible_device_pool_limits_reads_every_visible_device_by_default() -> None:
    """Called with no argument the helper reads what JAX reports."""
    limits = visible_device_pool_limits()

    assert set(limits) == set(visible_device_ids())


def test_model_construction_caps_the_budget_against_the_real_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model built with the whole pool as its budget plans below the pool."""
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    monkeypatch.setattr(
        "lcm.model.visible_device_pool_limits",
        lambda: dict.fromkeys(base.execution_devices, _POOL_LIMIT),
    )

    model = Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=MultiRegimeId,
        fixed_params=dict(base.fixed_params),
        execution_config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT),
    )

    assert model._execution.device_memory_bytes == _POOL_LIMIT - 7_200_000_000


def test_model_construction_records_the_requested_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model keeps the caller's request beside the ceiling it uses."""
    base = get_multi_regime_model(n_periods=6, distribution_type="normal")
    monkeypatch.setattr(
        "lcm.model.visible_device_pool_limits",
        lambda: dict.fromkeys(base.execution_devices, _POOL_LIMIT),
    )

    model = Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=MultiRegimeId,
        fixed_params=dict(base.fixed_params),
        execution_config=ExecutionConfig(device_memory_bytes=_POOL_LIMIT),
    )

    assert model._execution.requested_device_memory_bytes == _POOL_LIMIT
