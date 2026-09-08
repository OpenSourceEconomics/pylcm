"""Independent bit-packing and allocation-boundary controls for preflight."""

import dataclasses
from fractions import Fraction
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.initial_conditions as initial_module
from _lcm.dtypes import canonical_float_dtype
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.params.processing import process_params
from _lcm.simulation.initial_conditions import (
    _pack_initial_summary,
    _preflight_memory,
    _read_initial_cohorts,
    validate_simulation_inputs,
)
from _lcm.simulation.residency import DeviceBufferFootprint, resident_bytes_by_device
from _lcm.typing import FlatParams, InitialConditions
from _lcm.utils.logging import get_logger
from lcm import AgeGrid, Model
from lcm.exceptions import ExecutionPlanningError
from tests.simulation.test_initial_conditions import _make_asymmetric_state_model


def _case(*, n_subjects: int = 1) -> tuple[Model, FlatParams, InitialConditions]:
    """Build tiny real canonical inputs without a solve or replay payload."""
    model = _make_asymmetric_state_model()
    params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    initial = MappingProxyType(
        {
            "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
            "age": jnp.zeros(n_subjects, dtype=canonical_float_dtype()),
            "wealth": jnp.ones(n_subjects, dtype=canonical_float_dtype()),
            "health": jnp.zeros(n_subjects, dtype=jnp.int32),
        }
    )
    return model, params, initial


@pytest.mark.parametrize("relevant_bad_code", [False, True])
def test_metadata_words_preserve_exact_ages_and_filter_codes_by_cohort(
    *,
    relevant_bad_code: bool,
) -> None:
    """Preserve age bits and ignore invalid codes outside the owning cohort."""
    dtype = np.dtype(canonical_float_dtype())
    ages = np.array(
        [0.0, -0.0, np.nextafter(dtype.type(1), dtype.type(2))], dtype=dtype
    )
    # Canonical mapping order is deliberately different from subject order.
    ids = np.array([7, 2, 7], dtype=np.int32)
    codes = np.array([1, 999, -1 if relevant_bad_code else 0], dtype=np.int32)
    actual = _pack_initial_summary(
        regime_ids=jnp.asarray(ids),
        age_values=jnp.asarray(ages),
        canonical_ids=(jnp.int32(2), jnp.int32(7)),
        discrete_values=(jnp.asarray(codes),),
        discrete_specs=(("status", (0, 1), (1,)),),
    )
    expected = np.concatenate(
        (
            np.array([2, 7, 7, 2, 7], dtype=np.int32),
            ages.view(np.int32),
            np.array([int(relevant_bad_code)], dtype=np.int32),
        )
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual)[5:-1].view(dtype).view(np.uint8), ages.view(np.uint8)
    )


def test_fractional_age_lookup_preserves_existing_exactness() -> None:
    """Working-format membership cannot replace AgeGrid's exact host lookup."""
    model, _, initial = _case()
    ages = AgeGrid(exact_values=(0, Fraction(1, 3), 1))
    initial = {
        **initial,
        "age": jnp.array([float(Fraction(1, 3))], dtype=canonical_float_dtype()),
    }
    if np.dtype(canonical_float_dtype()).itemsize == 4:
        with pytest.raises(ValueError, match="not a valid grid point"):
            _read_initial_cohorts(
                initial_conditions=initial,
                regimes=model._regimes,
                regime_names_to_ids=model.regime_names_to_ids,
                ages=ages,
                memory=None,
            )
    else:
        metadata = _read_initial_cohorts(
            initial_conditions=initial,
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            ages=ages,
            memory=None,
        )
        assert metadata.periods == (1,)


@pytest.mark.parametrize(
    "error_type", [ExecutionPlanningError, MemoryError, jax.errors.JaxRuntimeError]
)
def test_resource_failures_never_select_serial_diagnostics(
    *, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """Refusal and resource exhaustion propagate by identity without a retry."""
    model, params, initial = _case()
    expected = error_type("resource sentinel")

    def denied(**_kwargs: object) -> None:
        raise expected

    def forbidden(**_kwargs: object) -> None:
        pytest.fail("Resource failure selected serial validation.")

    monkeypatch.setattr(initial_module, "_pack_initial_summary", denied)
    monkeypatch.setattr(initial_module, "validate_initial_conditions", forbidden)
    with pytest.raises(error_type) as caught:
        validate_simulation_inputs(
            initial_conditions=initial,
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=params,
            ages=model.ages,
            logger=get_logger(log_level="debug"),
        )
    assert caught.value is expected


def test_summary_allocation_is_refused_before_dispatch_and_inputs_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real admission refuses an unfunded summary and accepts generous headroom."""
    model, params, initial = _case(n_subjects=1024)
    generous = dataclasses.replace(model._execution, device_memory_bytes=2**25)
    empty = DeviceBufferFootprint(spans={})
    memory = _preflight_memory(
        execution=generous,
        retained_footprint=empty,
        initial_conditions=initial,
        flat_params=params,
        regimes=model._regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        ages=model.ages,
    )
    assert memory is not None
    resident = max(
        resident_bytes_by_device(
            live=memory.inputs,
            arguments=empty,
            devices=memory.devices,
        ).values()
    )
    packed_bytes = (2 + 1024 + 1024 * initial["age"].dtype.itemsize // 4 + 1) * 4
    tight = dataclasses.replace(
        generous, device_memory_bytes=resident + packed_bytes // 2
    )
    dispatched = []
    original = jax.stages.Compiled.__call__

    def observe(self: jax.stages.Compiled, *args: object, **kwargs: object) -> object:
        dispatched.append(self)
        return original(self, *args, **kwargs)

    def forbidden(**_kwargs: object) -> None:
        pytest.fail("A valid budget witness selected unprofiled serial diagnostics.")

    monkeypatch.setattr(jax.stages.Compiled, "__call__", observe)
    monkeypatch.setattr(initial_module, "validate_initial_conditions", forbidden)

    def validate(execution: ResolvedExecution) -> None:
        validate_simulation_inputs(
            initial_conditions=initial,
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=params,
            ages=model.ages,
            logger=get_logger(log_level="debug"),
            retained_footprint=empty,
            execution=execution,
        )

    with pytest.raises(ExecutionPlanningError):
        validate(tight)
    assert not dispatched
    validate(generous)
    assert dispatched
    for name, values in initial.items():
        assert not values.is_deleted()
        np.testing.assert_array_equal(
            values, np.full(1024, int(name == "wealth"), dtype=values.dtype)
        )
