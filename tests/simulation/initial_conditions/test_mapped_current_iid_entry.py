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


def _model(
    *,
    budget_bytes: int | None = None,
    ages: AgeGrid | None = None,
    initial_nodes: InitialNodes | None = None,
) -> Model:
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
        ages=AgeGrid(exact_values=(Fraction(41, 2),)) if ages is None else ages,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_age={20.5: "source"})
        if initial_nodes is None
        else initial_nodes,
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


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_mapped_entry_mixed_dates_keep_original_rows() -> None:
    """Interleaved entry cohorts use their own date and supplied physical frame."""
    model = _model(
        ages=AgeGrid(exact_values=(Fraction(41, 2), Fraction(95, 4))),
        initial_nodes=InitialNodes(by_age={20.5: "source", 23.75: "source"}),
    )
    initial = {
        "age": jnp.asarray([23.75 - 1e-4, 20.5 - 1e-4, 23.75 - 1e-4, 20.5 - 1e-4]),
        "regime_id": jnp.zeros(4, dtype=jnp.int32),
        "x": jnp.asarray([0.25, 1.0, 0.75, 0.5]),
        "signal": PhysicalValues(values=jnp.asarray([35.25, 31.75, 35.75, 31.25])),
    }
    params = {"risk": {"width": 0.5}}
    model.validate_initial_conditions(initial_conditions=initial, params=params)
    np.testing.assert_array_equal(
        model.initial_conditions_feasibility(initial_conditions=initial, params=params),
        [True, True, True, True],
    )
    # Every wrong-date replacement is outside that row's genuine physical law.
    swapped = {
        **initial,
        "signal": PhysicalValues(values=jnp.asarray([31.75, 35.25, 31.25, 35.75])),
    }
    with pytest.raises(ShockDeclarationError, match="mapped entry"):
        model.validate_initial_conditions(initial_conditions=swapped, params=params)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_dated_mapped_path_overrides_entry_in_original_row_order(
    *,
    runtime_checks: bool,
) -> None:
    """A subject-by-period physical path wins at each mapped entry date."""
    model = _model(
        ages=AgeGrid(exact_values=(Fraction(41, 2), Fraction(95, 4))),
        initial_nodes=InitialNodes(by_age={20.5: "source", 23.75: "source"}),
    )
    initial = {
        "age": np.asarray([23.75, 20.5, 23.75, 20.5]),
        "regime_id": np.zeros(4, dtype=np.int32),
        "x": jnp.asarray([0.25, 1.0, 0.75, 0.5]),
        "signal": PhysicalValues(values=jnp.full(4, -999.0)),
    }
    frame = model.simulate(
        params={"risk": {"width": 0.5}},
        initial_conditions=initial,
        shock_paths={
            "signal": PhysicalValues(
                values=jnp.asarray(
                    [[31.0, 35.25], [31.75, 36.0], [31.5, 35.75], [31.25, 35.5]]
                )
            )
        },
        seed=17,
        runtime_checks=runtime_checks,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    for subject, period, mapped_y, signal in (
        (0, 1, 10.25, 35.25),
        (1, 0, 11.0, 31.75),
        (2, 1, 10.75, 35.75),
        (3, 0, 10.5, 31.25),
    ):
        selected = frame.loc[
            (frame["subject_id"] == subject) & (frame["period"] == period)
        ]
        np.testing.assert_array_equal(selected["regime_name"], "target")
        np.testing.assert_array_equal(selected["y"], mapped_y)
        np.testing.assert_array_equal(selected["signal"], signal)
