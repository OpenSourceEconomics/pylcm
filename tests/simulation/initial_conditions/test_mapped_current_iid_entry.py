"""Public checked-entry validation retains each mapped producing context."""

from collections.abc import Callable, Mapping
from fractions import Fraction
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.params.edges import edge_params, flat_namespaces
from _lcm.policy_queries import _query_flat_params
from _lcm.simulation.current_iid_entry import _MappedEntryFrames
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.typing import FlatParams, PytreeValue, ShapeDtypePytree
from lcm import (
    AgeGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    ShockDeclarationError,
    UnsupportedOperationError,
)
from lcm.shocks import PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.typing import BoolND, FloatND, IntND, ScalarInt
from tests.simulation.test_current_exits import _hidden_offer_model


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt


def _stay(*, x: FloatND) -> FloatND:
    return 0.5 + 0.0 * x


def _mapped_y(*, x: FloatND) -> FloatND:
    return jnp.where(x > 0, x + 10, jnp.nan)


def _available(*, x: FloatND) -> BoolND:
    return x > 0


def _lower(*, y: FloatND, age: FloatND, period: IntND) -> FloatND:
    return jnp.where(period.dtype == jnp.int32, y + age + period, jnp.nan)


def _upper(*, y: FloatND, age: FloatND, period: IntND, width: FloatND) -> FloatND:
    return _lower(y=y, age=age, period=period) + width


def _target(*, signal: FloatND, y: FloatND) -> FloatND:
    return signal + 0.0 * y


def _model(*, budget_bytes: int | None = None) -> Model:
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": _stay},
                states={"x": LinSpacedGrid(start=0, stop=1, n_points=3)},
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target", projection={"y": _mapped_y}
                            )
                        },
                        rule="unilateral",
                        observes=("signal",),
                        available=_available,
                    )
                },
            ),
            "target": Regime(
                functions={"utility": _target},
                states={"y": LinSpacedGrid(start=10, stop=11, n_points=3)},
                shocks=("risk",),
            ),
        },
        shocks={
            "risk": ShockGroup(
                shocks={"signal": Uniform(start=_lower, stop=_upper)},
                nodes=UserNodes(
                    nodes=np.asarray([[0.25], [0.75]]),
                    weights=np.asarray([0.5, 0.5]),
                ),
            )
        },
        ages=AgeGrid(exact_values=(Fraction(41, 2),)),
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_age={20.5: "source"}),
        edges={},
        enable_jit=True,
        execution_config=ExecutionConfig(device_memory_bytes=budget_bytes),
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_observed_mapped_entry_uses_current_params_rows_and_canonical_clock() -> None:
    """Both helpers and checked replay validate the target law after settlement."""
    model = _model()
    initial = {
        "age": jnp.asarray([20.5 - 1e-4, 20.5 - 1e-4]),
        "regime_id": jnp.zeros(2, dtype=jnp.int32),
        "x": jnp.asarray([1.0, 0.5]),
        "signal": PhysicalValues(values=jnp.asarray([31.75, 31.25])),
    }
    params = {"risk": {"width": 0.5}}
    model.validate_initial_conditions(initial_conditions=initial, params=params)
    np.testing.assert_array_equal(
        model.initial_conditions_feasibility(initial_conditions=initial, params=params),
        [True, True],
    )
    frame = (
        model.simulate(
            params=params,
            initial_conditions=initial,
            seed=17,
            runtime_checks=True,
            log_level="off",
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values("subject_id")
    )
    np.testing.assert_array_equal(frame["regime_name"], "target")
    np.testing.assert_array_equal(frame["age"], [20.5, 20.5])
    np.testing.assert_array_equal(frame["y"], [11.0, 10.5])
    np.testing.assert_array_equal(frame["signal"], [31.75, 31.25])
    permuted = {
        name: PhysicalValues(values=value.values[::-1])
        if isinstance(value, PhysicalValues)
        else value[::-1]
        for name, value in initial.items()
    }
    model.validate_initial_conditions(initial_conditions=permuted, params=params)
    for method in (
        model.validate_initial_conditions,
        model.initial_conditions_feasibility,
    ):
        with pytest.raises(ShockDeclarationError, match="mapped entry"):
            method(initial_conditions=initial, params={"risk": {"width": 0.125}})


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_inactive_mapped_entry_does_not_evaluate_its_numerical_law() -> None:
    """An unavailable exit leaves its invalid projection and physical cell unread."""
    model = _model()
    initial = {
        "age": jnp.asarray([20.5]),
        "regime_id": jnp.zeros(1, dtype=jnp.int32),
        "x": jnp.zeros(1),
        "signal": PhysicalValues(values=jnp.asarray([jnp.nan])),
    }
    params = {"risk": {"width": 0.5}}
    model.validate_initial_conditions(initial_conditions=initial, params=params)
    np.testing.assert_array_equal(
        model.initial_conditions_feasibility(initial_conditions=initial, params=params),
        [True],
    )
    frame = model.simulate(
        params=params,
        initial_conditions=initial,
        seed=17,
        runtime_checks=True,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame["regime_name"], "source")
    np.testing.assert_array_equal(frame["value"], [0.5])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_seedless_helper_refuses_a_hidden_first_producer_without_selecting_policy() -> (
    None
):
    """A helper does not solve a hidden exit to guess a supplied value's law."""
    model = _hidden_offer_model(observed=False, enable_jit=True)
    initial = {
        "period": jnp.zeros(1, dtype=jnp.int32),
        "regime_id": jnp.zeros(1, dtype=jnp.int32),
        "offer": PhysicalValues(values=jnp.ones(1, dtype=jnp.int32)),
    }
    for method in (
        model.validate_initial_conditions,
        model.initial_conditions_feasibility,
    ):
        with pytest.raises(UnsupportedOperationError, match="first hidden producer"):
            method(initial_conditions=initial, params={})


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_checked_mapped_entry_uses_model_admission_and_preserves_budget_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine mapped producer is admitted by its Model; tiny residency refuses."""
    admitted = []
    original = ProfiledSimulationOperations.admit_producer

    # keyword-only-exempt: primary-argument=self
    def observe(
        self: ProfiledSimulationOperations,
        *,
        function: Callable[..., PytreeValue],
        arguments: Mapping[str, ShapeDtypePytree],
        devices: tuple[jax.Device, ...],
        output_sharding: jax.sharding.Sharding | None,
        budget_bytes: int,
        resident_bytes: int,
    ) -> jax.stages.Compiled:
        executable = original(
            self,
            function=function,
            arguments=arguments,
            devices=devices,
            output_sharding=output_sharding,
            budget_bytes=budget_bytes,
            resident_bytes=resident_bytes,
        )
        if isinstance(function, _MappedEntryFrames):
            admitted.append(self)
        return executable

    monkeypatch.setattr(ProfiledSimulationOperations, "admit_producer", observe)
    initial = {
        "age": jnp.asarray([20.5]),
        "regime_id": jnp.zeros(1, dtype=jnp.int32),
        "x": jnp.ones(1),
        "signal": PhysicalValues(values=jnp.asarray([31.75])),
    }
    model = _model(budget_bytes=2**26)
    model.simulate(
        params={"risk": {"width": 0.5}},
        initial_conditions=initial,
        seed=17,
        runtime_checks=True,
        log_level="off",
    )
    assert admitted
    assert all(owner is model._simulate_entry_operations for owner in admitted)
    before = tuple(admitted)
    limited = _model(budget_bytes=1)
    with pytest.raises(ExecutionPlanningError):
        limited.simulate(
            params={"risk": {"width": 0.5}},
            initial_conditions=initial,
            seed=17,
            runtime_checks=True,
            log_level="off",
        )
    assert tuple(admitted) == before


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_mapped_query_params_merge_fixed_and_runtime_edge_leaves() -> None:
    """A runtime edge slot must preserve other fixed slots of its source."""
    fixed: FlatParams = MappingProxyType(
        {
            "source": MappingProxyType(
                {
                    "utility__fixed": jnp.asarray(2.0),
                    "utility__shared": jnp.asarray(3.0),
                }
            ),
            "edges": MappingProxyType(
                {
                    "source": MappingProxyType(
                        {
                            "target__fixed": jnp.asarray(4.0),
                            "target__shared": jnp.asarray(5.0),
                        }
                    ),
                    "fixed_only": MappingProxyType({"rate": jnp.asarray(6.0)}),
                }
            ),
        }
    )
    runtime: FlatParams = MappingProxyType(
        {
            "source": MappingProxyType(
                {
                    "utility__runtime": jnp.asarray(7.0),
                    "utility__shared": jnp.asarray(8.0),
                }
            ),
            "edges": MappingProxyType(
                {
                    "source": MappingProxyType(
                        {
                            "target__runtime": jnp.asarray(9.0),
                            "target__shared": jnp.asarray(10.0),
                        }
                    ),
                    "runtime_only": MappingProxyType({"rate": jnp.asarray(11.0)}),
                }
            ),
        }
    )
    merged = _query_flat_params(flat_params=runtime, fixed_flat_params=fixed)
    own = dict(flat_namespaces(merged))[("source",)]
    np.testing.assert_equal(
        set(own), {"utility__fixed", "utility__runtime", "utility__shared"}
    )
    np.testing.assert_array_equal(own["utility__fixed"], 2.0)
    np.testing.assert_array_equal(own["utility__runtime"], 7.0)
    np.testing.assert_array_equal(own["utility__shared"], 8.0)
    source = edge_params(merged, source="source")
    np.testing.assert_equal(
        set(source), {"target__fixed", "target__runtime", "target__shared"}
    )
    np.testing.assert_array_equal(source["target__fixed"], 4.0)
    np.testing.assert_array_equal(source["target__runtime"], 9.0)
    np.testing.assert_array_equal(source["target__shared"], 10.0)
    np.testing.assert_array_equal(edge_params(merged, source="fixed_only")["rate"], 6.0)
    np.testing.assert_array_equal(
        edge_params(merged, source="runtime_only")["rate"], 11.0
    )
