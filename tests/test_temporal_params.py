"""Label alignment protects temporal parameters before numerical evaluation."""

import inspect
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from beartype import beartype
from dags.tree import flatten_to_qnames

import lcm
import lcm.exceptions
from _lcm.execution.core_program import core_program_graph
from _lcm.params.temporal import align_time_varying
from _lcm.solution.preconditions import check_solver_params
from _lcm.time import ModelTime
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.params import UserMappingLeaf, UserSequenceLeaf
from lcm.solvers import EGM, NBEGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    Period,
    ScalarInt,
    UserFunction,
    ValueND,
)

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@lcm.categorical(ordered=False)
class RegimeId:
    work: ScalarInt
    done: ScalarInt


def _flow(*, wage: FloatND) -> FloatND:
    return wage


def _manual(*, wage: FloatND, period: Period) -> FloatND:
    return wage[period]


def _wrong_age(*, wage: FloatND, age: FloatND) -> FloatND:
    return wage[age]


def _terminal() -> FloatND:
    return jnp.asarray(8.0)


def _model(
    *,
    fixed: ValueND | pd.Series | lcm.TimeVarying | float | None = None,
    manual: bool = False,
    age: bool = False,
    flow: UserFunction | lcm.Phased | None = None,
    budget: int | None = None,
) -> lcm.Model:
    flow = (
        flow
        if flow is not None
        else (_manual if manual else lcm.time_varying_params("wage")(_flow))
    )
    clock: dict[str, Any] = (
        {"ages": lcm.AgeGrid(start=40, inclusive_stop=42, step="Y")}
        if age
        else {"n_periods": 3}
    )
    return lcm.Model(
        **clock,
        regimes={
            "work": lcm.Regime(functions={"utility": flow}),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=((40, "work"),)
        if age
        else lcm.InitialNodes(by_period={0: "work"}),
        edges={"work": {"work": (40,), "done": (41,)}}
        if age
        else {
            "work": {"work": lcm.Periods(values=(0,)), "done": lcm.Periods(values=(1,))}
        },
        fixed_params={
            "discount_factor": 0.5,
            **({"wage": fixed} if fixed is not None else {}),
        },
        execution_config=lcm.ExecutionConfig(device_memory_bytes=budget),
    )


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("series", [False, True])
@pytest.mark.parametrize("age", [False, True])
def test_temporal_labels_align_and_ignore_surplus(
    *, fixed: bool, series: bool, age: bool
) -> None:
    labels = [41, 99, 40, 99] if age else [1, -3, 0, -3]
    value = (
        pd.Series(
            [2.0, 90.0, 1.0, 91.0],
            index=pd.Index(labels, name="age" if age else "period"),
        )
        if series
        else lcm.TimeVarying(
            values=jnp.array([2.0, 90.0, 1.0, 91.0]),
            **{"ages" if age else "periods": tuple(labels)},
        )
    )
    model = _model(fixed=value if fixed else None, age=age)
    result = model.solve(params={} if fixed else {"wage": value}, log_level="off")
    np.testing.assert_allclose(result.values[0]["work"], 4)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "wrong_kind",
        "fractional",
        "boolean",
        "length",
        "unlabelled",
    ],
)
@pytest.mark.parametrize("fixed", [False, True])
def test_temporal_invalid_inputs_fail_before_solving(*, kind: str, fixed: bool) -> None:
    factory = {
        "missing": lambda: lcm.TimeVarying(values=jnp.array([1.0]), periods=(0,)),
        "duplicate": lambda: lcm.TimeVarying(
            values=jnp.array([1.0, 2.0, 3.0]), periods=(0, 1, 1)
        ),
        "wrong_kind": lambda: lcm.TimeVarying(
            values=jnp.array([1.0, 2.0]), ages=(0, 1)
        ),
        "fractional": lambda: lcm.TimeVarying(
            values=jnp.array([1.0, 2.0, 3.0]), periods=(0, 1, 9.5)
        ),
        "boolean": lambda: lcm.TimeVarying(
            values=jnp.array([1.0, 2.0]), periods=(False, True)
        ),
        "length": lambda: lcm.TimeVarying(values=jnp.array([1.0]), periods=(0, 1)),
        "unlabelled": lambda: jnp.array([1.0, 2.0]),
    }
    with pytest.raises(  # noqa: PT012  (the public failure can occur at any boundary)
        (lcm.exceptions.InvalidParamsError, ValueError),
        match=r"wage|period|coordinate|length|label",
    ):
        value = factory[kind]()
        model = _model(fixed=value if fixed else None)
        model.solve(params={} if fixed else {"wage": value}, log_level="off")


def test_managed_scalar_constant_and_result_target() -> None:
    model = _model()
    result = model.simulate(
        params={"wage": 2.0},
        initial_conditions={
            "period": jnp.array([0]),
            "regime_id": jnp.array([RegimeId.work]),
        },
        seed=1,
        log_level="off",
    )
    frame = result.to_dataframe(additional_targets=["utility"])
    np.testing.assert_allclose(frame["utility"], [2.0, 2.0, 8.0])


def _flow_with_bonus(*, wage: FloatND, bonus: FloatND) -> FloatND:
    return wage + bonus


def _bonus(*, period: Period) -> FloatND:
    return jnp.asarray(period, dtype=float)


def test_temporal_function_shares_period_with_annotated_regime_function() -> None:
    """A temporal function without `period` composes with one annotating `period`."""
    model = lcm.Model(
        n_periods=3,
        regimes={
            "work": lcm.Regime(
                functions={
                    "utility": lcm.time_varying_params("wage")(_flow_with_bonus),
                    "bonus": _bonus,
                }
            ),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={
            "work": {"work": lcm.Periods(values=(0,)), "done": lcm.Periods(values=(1,))}
        },
        fixed_params={"discount_factor": 0.5},
    )
    result = model.solve(params={"wage": 2.0}, log_level="off")
    np.testing.assert_allclose(result.values[0]["work"], 5.5)


def test_raw_manual_period_array_is_rejected() -> None:
    with pytest.raises(lcm.exceptions.InvalidParamsError, match=r"label|TimeVarying"):
        _model(manual=True).solve(
            params={"wage": jnp.array([1.0, 2.0])}, log_level="off"
        )


def test_raw_manual_age_array_warns() -> None:
    with pytest.warns(lcm.UnlabelledTimeParameterWarning, match="wage"):
        result = _model(manual=True, age=True).solve(
            params={"wage": jnp.array([1.0, 2.0, 8.0])}, log_level="off"
        )
    np.testing.assert_allclose(result.values[0]["work"], 4)


def test_labelled_manual_period_array() -> None:
    result = _model(manual=True).solve(
        params={"wage": pd.Series([2.0, 1.0], index=pd.Index([1, 0], name="period"))},
        log_level="off",
    )
    np.testing.assert_allclose(result.values[0]["work"], 4)


def test_labelled_manual_age_index_is_rejected() -> None:
    model = _model(age=True, flow=_wrong_age)
    with pytest.raises(lcm.exceptions.InvalidParamsError, match=r"table\[age\]"):
        model.solve(
            params={
                "wage": pd.Series([1.0, 2.0], index=pd.Index([40, 41], name="age"))
            },
            log_level="off",
        )


def test_time_varying_values_remain_differentiable_leaves() -> None:
    def total(values: FloatND) -> FloatND:
        aligned = align_time_varying(
            value=lcm.TimeVarying(values=values, periods=(1, 9, 0)),
            ages=ModelTime(n_periods=3),
            required_periods=(0, 1),
            name="wage",
        )
        return aligned[0] + 2 * aligned[1]

    np.testing.assert_allclose(
        jax.grad(total)(jnp.array([3.0, 99.0, 5.0])), [2.0, 0.0, 1.0]
    )


def test_schedule_requires_only_periods_of_consuming_case() -> None:
    model = lcm.Model(
        n_periods=3,
        regime_id_class=RegimeId,
        regimes={
            "work": lcm.Regime(functions={"utility": _terminal}),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={
            "work": lcm.Transition(
                law=lcm.ByPeriod(
                    cases={
                        0: {
                            "work": lcm.StochasticTransition(
                                func=lcm.time_varying_params("wage")(_flow)
                            )
                        },
                        1: "done",
                    }
                )
            )
        },
        fixed_params={"discount_factor": 0.5},
    )
    result = model.solve(
        params={"wage": lcm.TimeVarying(values=jnp.array([1.0]), periods=(0,))},
        log_level="off",
    )
    np.testing.assert_allclose(result.values[0]["work"], 14.0)


def test_temporal_marker_cannot_name_a_dag_node() -> None:
    with pytest.raises(
        lcm.exceptions.ModelInitializationError,
        match=r"wage.*(node|parameter)|parameter.*wage",
    ):
        lcm.Model(
            n_periods=1,
            regime_id_class=RegimeId,
            regimes={
                "work": lcm.Regime(
                    functions={
                        "utility": lcm.time_varying_params("wage")(_flow),
                        "wage": _terminal,
                    }
                ),
                "done": lcm.Regime(functions={"utility": _terminal}),
            },
            initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
            edges={},
        )


def test_time_varying_requires_declared_slot() -> None:
    with pytest.raises(lcm.exceptions.InvalidParamsError, match="time_varying_params"):
        _model(manual=True).solve(
            params={
                "wage": lcm.TimeVarying(values=jnp.array([1.0, 2.0]), periods=(0, 1))
            },
            log_level="off",
        )


def test_fixed_parameter_cannot_be_overridden_at_runtime() -> None:
    model = _model(
        fixed=lcm.TimeVarying(values=jnp.array([90.0, 91.0]), periods=(0, 1))
    )
    with pytest.raises(lcm.exceptions.InvalidParamsError, match=r"Unknown.*wage"):
        model.solve(
            params={
                "wage": lcm.TimeVarying(values=jnp.array([2.0, 1.0]), periods=(1, 0))
            },
            log_level="off",
        )


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1.0, 2.0, 4.0], index=pd.Index([0.0, 1.0, 8.5], name="period")),
        pd.Series([1.0, 2.0], index=pd.Index([False, True], name="period")),
        pd.Series([1.0, 2.0], index=pd.Index([0, 1], name="age")),
        pd.Series([1.0, 2.0], index=pd.Index([0, 1])),
        pd.Series([1.0, 2.0, 3.0], index=pd.Index([0, 1, 1], name="period")),
        pd.Series([1.0], index=pd.Index([0], name="period")),
    ],
)
def test_series_schema_and_selected_keys_are_checked(series: pd.Series) -> None:
    with pytest.raises(
        (lcm.exceptions.InvalidParamsError, ValueError),
        match=r"period|level|duplicate|missing",
    ):
        _model().solve(params={"wage": series}, log_level="off")


def test_temporal_decorator_survives_outer_beartype() -> None:
    # Import-hook instrumentation adds this outer decorator in downstream packages.
    consumer = beartype(lcm.time_varying_params("wage")(_flow))
    assert tuple(inspect.signature(consumer).parameters) == ("wage", "period")
    result = _model(flow=consumer).solve(
        params={"wage": lcm.TimeVarying(values=jnp.array([1.0, 2.0]), periods=(0, 1))},
        log_level="off",
    )
    np.testing.assert_allclose(result.values[0]["work"], 4.0)


def test_template_identifies_temporal_slot() -> None:
    template = _model().get_params_template()
    assert "TimeVarying" in template["work"]["utility"]["wage"]


def _manual_computed_index(*, wage: FloatND, period: Period) -> FloatND:
    return wage[period // 2]


def test_raw_computed_time_index_is_also_rejected() -> None:
    with pytest.raises(lcm.exceptions.InvalidParamsError, match=r"label|TimeVarying"):
        _model(flow=_manual_computed_index).solve(
            params={"wage": jnp.array([1.0, 2.0])}, log_level="off"
        )


def test_discarded_values_do_not_change_solution_compatibility() -> None:
    model = _model()
    initial = {"period": jnp.array([0]), "regime_id": jnp.array([RegimeId.work])}
    first = lcm.TimeVarying(values=jnp.array([1.0, 2.0]), periods=(0, 1))
    wider = lcm.TimeVarying(values=jnp.array([80.0, 2.0, 1.0]), periods=(9, 1, 0))
    solution = model.solve(params={"wage": first}, log_level="off")
    result = model.simulate(
        params={"wage": wider},
        initial_conditions=initial,
        solution=solution,
        seed=7,
        log_level="off",
    )
    np.testing.assert_allclose(
        result.to_dataframe(additional_targets=["utility"])["utility"], [1, 2, 8]
    )


def test_phased_temporal_consumers_keep_both_parameter_sets() -> None:
    def simulated(*, bonus: FloatND) -> FloatND:
        return bonus

    model = _model(
        flow=lcm.Phased(
            solve=lcm.time_varying_params("wage")(_flow),
            simulate=lcm.time_varying_params("bonus")(simulated),
        )
    )
    params = {
        "wage": lcm.TimeVarying(values=jnp.array([1.0, 2.0]), periods=(0, 1)),
        "bonus": lcm.TimeVarying(values=jnp.array([3.0, 7.0]), periods=(0, 1)),
    }
    result = model.simulate(
        params=params,
        initial_conditions={
            "period": jnp.array([0]),
            "regime_id": jnp.array([RegimeId.work]),
        },
        seed=3,
        log_level="off",
    )
    np.testing.assert_allclose(
        result.to_dataframe(additional_targets=["utility"])["utility"], [3, 7, 8]
    )


def test_shared_slot_cannot_mix_managed_and_manual_time_axes() -> None:
    with pytest.raises(  # noqa: PT012 - declaration or parameter preflight may reject
        (lcm.exceptions.ModelInitializationError, lcm.exceptions.InvalidParamsError),
        match=r"wage.*(temporal|managed)|(?:temporal|managed).*wage",
    ):
        model = _model(
            flow=lcm.Phased(
                solve=lcm.time_varying_params("wage")(_flow),
                simulate=_manual,
            )
        )
        model.solve(
            params={
                "wage": lcm.TimeVarying(values=jnp.array([1.0, 2.0]), periods=(0, 1))
            },
            log_level="off",
        )


@pytest.mark.parametrize("missing", [None, "wage", "bonus"])
@pytest.mark.parametrize("fixed", [False, True])
def test_temporal_coverage_follows_the_phase_that_reads_a_parameter(
    *, missing: str | None, fixed: bool
) -> None:
    def simulated(*, bonus: FloatND) -> FloatND:
        return bonus

    params = {
        "wage": lcm.TimeVarying(
            values=jnp.array([1.0] if missing == "wage" else [1.0, 2.0]),
            periods=(0,) if missing == "wage" else (0, 1),
        ),
        "bonus": lcm.TimeVarying(
            values=jnp.array([] if missing == "bonus" else [3.0]),
            periods=() if missing == "bonus" else (0,),
        ),
    }

    def simulate() -> lcm.SimulationResult:
        model = lcm.Model(
            n_periods=3,
            regimes={
                "work": lcm.Regime(
                    functions={
                        "utility": lcm.Phased(
                            solve=lcm.time_varying_params("wage")(_flow),
                            simulate=lcm.time_varying_params("bonus")(simulated),
                        )
                    }
                ),
                "done": lcm.Regime(functions={"utility": _terminal}),
            },
            regime_id_class=RegimeId,
            initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
            edges=lcm.Phased(
                solve={
                    "work": {
                        "work": lcm.Periods(values=(0,)),
                        "done": lcm.Periods(values=(1,)),
                    }
                },
                simulate={"work": {"done": lcm.Periods(values=(0,))}},
            ),
            fixed_params={"discount_factor": 0.5, **(params if fixed else {})},
        )
        return model.simulate(
            params={} if fixed else params,
            initial_conditions={
                "period": jnp.array([0]),
                "regime_id": jnp.array([RegimeId.work]),
            },
            seed=3,
            log_level="off",
        )

    if missing is not None:
        with pytest.raises(
            lcm.exceptions.InvalidParamsError, match=rf"{missing}.*missing"
        ):
            simulate()
    else:
        frame = simulate().to_dataframe(additional_targets=["utility"])
        np.testing.assert_array_equal(frame["period"], [0, 1])
        np.testing.assert_allclose(frame["utility"], [3, 8])


def test_deferred_temporal_target_uses_realized_period(tmp_path: Path) -> None:
    def report(*, bonus: FloatND) -> FloatND:
        return bonus

    model = lcm.Model(
        n_periods=3,
        regimes={
            "work": lcm.Regime(
                functions={
                    "utility": _zero,
                    "report": lcm.time_varying_params("bonus")(report),
                }
            ),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={
            "work": {"work": lcm.Periods(values=(0,)), "done": lcm.Periods(values=(1,))}
        },
        fixed_params={"discount_factor": 0.5},
    )
    result = model.simulate(
        params={
            "bonus": lcm.TimeVarying(
                values=jnp.array([7.0, 900.0, 3.0]), periods=(1, 9, 0)
            )
        },
        initial_conditions={
            "period": jnp.array([0]),
            "regime_id": jnp.array([RegimeId.work]),
        },
        seed=3,
        log_level="off",
    )
    path = tmp_path / "result.pkl"
    result.save(directory=path)
    restored = lcm.SimulationResult.load(directory=path)
    frame = restored.to_dataframe(additional_targets=["report"])
    np.testing.assert_allclose(frame.loc[frame["period"] < 2, "report"], [3, 7])


@lcm.categorical(ordered=False)
class Option:
    small: ScalarInt
    large: ScalarInt


def _tariff(*, tariff: FloatND, option: DiscreteAction) -> FloatND:
    return tariff[option]


@pytest.mark.parametrize("missing", [False, True])
def test_series_time_and_categorical_axes_are_validated_together(
    *, missing: bool
) -> None:
    model = lcm.Model(
        n_periods=2,
        regimes={
            "work": lcm.Regime(
                functions={"utility": lcm.time_varying_params("tariff")(_tariff)},
                actions={"option": lcm.DiscreteGrid(category_class=Option)},
            ),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={"work": {"done": lcm.Periods(values=(0,))}},
        fixed_params={"discount_factor": 0.5},
    )
    # Duplicate discarded full keys and unknown discarded categories are harmless.
    value = pd.Series(
        [5.0, 900.0, 901.0, 1.0],
        index=pd.MultiIndex.from_tuples(
            [(0, "large"), (8, "outside"), (8, "outside"), (0, "small")],
            names=["period", "option"],
        ),
    )
    if missing:
        with pytest.raises(lcm.exceptions.InvalidParamsError, match=r"tariff.*missing"):
            model.solve(params={"tariff": value.iloc[:-1]}, log_level="off")
    else:
        solution = model.solve(params={"tariff": value}, log_level="off")
        np.testing.assert_allclose(solution.values[0]["work"], 9.0)


def test_economic_profile_mapping_is_explicit_and_differentiable() -> None:
    def total(values: FloatND) -> FloatND:
        profile = lcm.TimeVarying.from_profile(
            values=values,
            labels=(4, 9, 3),
            period_to_label={0: 3, 1: 3, 2: 4, 3: 4},
        )
        assert profile.periods == (0, 1, 2, 3)
        return jnp.sum(profile.values)

    np.testing.assert_allclose(total(jnp.array([10.0, 999.0, 5.0])), 30.0)
    np.testing.assert_allclose(
        jax.grad(total)(jnp.array([10.0, 999.0, 5.0])), [2, 0, 2]
    )


@pytest.mark.parametrize("labels", [(3, 9), (3, 3)])
def test_economic_profile_mapping_rejects_missing_or_duplicate_selected_labels(
    labels: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match=r"missing|duplicate"):
        lcm.TimeVarying.from_profile(
            values=jnp.array([5.0, 10.0]),
            labels=labels,
            period_to_label={0: 3, 1: 4},
        )


def test_managed_consumer_cannot_index_its_time_axis_again() -> None:
    with pytest.raises(
        (lcm.exceptions.ModelInitializationError, lcm.exceptions.InvalidParamsError),
        match=r"wage.*(?:slice|index)|(?:slice|index).*wage",
    ):
        _model(flow=lcm.time_varying_params("wage")(_manual))


@pytest.mark.parametrize("kind", ["temporal", "scalar", "sequence", "empty"])
def test_managed_parameter_rejects_a_nested_container(kind: str) -> None:
    if kind == "temporal":
        value = UserMappingLeaf(
            {"inner": lcm.TimeVarying(values=jnp.array([1.0, 2.0]), periods=(0, 1))}
        )
    elif kind == "sequence":
        value = UserSequenceLeaf((1.0, 2.0))
    else:
        value = UserMappingLeaf({} if kind == "empty" else {"inner": 1.0})
    with pytest.raises(
        lcm.exceptions.InvalidParamsError, match=r"wage.*(?:container|TimeVarying)"
    ):
        _model().solve(params={"wage": value}, log_level="off")


def test_temporal_age_labels_reject_numpy_booleans() -> None:
    with pytest.raises(lcm.exceptions.InvalidParamsError, match="age coordinates"):
        align_time_varying(
            value=lcm.TimeVarying(values=jnp.array([1.0]), ages=(np.bool_(1),)),
            ages=lcm.AgeGrid(start=1, inclusive_stop=2, step="Y"),
            required_periods=(0,),
            name="wage",
        )


@pytest.mark.parametrize("admitted", [False, True])
def test_temporal_alignment_keeps_values_for_canonical_range_validation(
    *,
    admitted: bool,
) -> None:
    if jax.config.x64_enabled:
        pytest.skip("A float64-to-float32 range check requires float32 mode.")
    values = np.array([np.finfo(np.float64).max, 1.0])
    assert np.isfinite(values).all()
    with pytest.raises(OverflowError, match=r"wage.*overflow"):
        _model(budget=100_000_000 if admitted else None).solve(
            params={"wage": lcm.TimeVarying(values=values, periods=(0, 1))},
            log_level="off",
        )


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _aggregate(*, utility: FloatND, CE: FloatND, weight: FloatND) -> FloatND:
    return utility + weight * CE


def test_temporal_aggregator_uses_current_source_period() -> None:
    model = lcm.Model(
        n_periods=3,
        regimes={
            "work": lcm.Regime(
                functions={"utility": _terminal},
                koopmans_aggregator=lcm.time_varying_params("weight")(_aggregate),
            ),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={
            "work": {"work": lcm.Periods(values=(0,)), "done": lcm.Periods(values=(1,))}
        },
    )
    solution = model.solve(
        params={
            "weight": lcm.TimeVarying(values=jnp.array([0.5, 0.25]), periods=(1, 0))
        },
        log_level="off",
    )
    np.testing.assert_allclose(solution.values[0]["work"], 11.0)


def _option_utility(*, option: DiscreteAction) -> FloatND:
    return jnp.asarray(option, dtype=float)


def _ceiling(*, option: DiscreteAction, ceiling: FloatND) -> BoolND:
    return option <= ceiling


@pytest.mark.parametrize("n_subjects", [1, 3])
@pytest.mark.parametrize("budget", [None, 100_000_000])
def test_temporal_constraint_changes_the_feasible_action(
    *, n_subjects: int, budget: int | None
) -> None:
    model = lcm.Model(
        n_periods=3,
        regimes={
            "work": lcm.Regime(
                functions={"utility": _option_utility},
                actions={"option": lcm.DiscreteGrid(category_class=Option)},
                constraints={
                    "feasible_option": lcm.time_varying_params("ceiling")(_ceiling)
                },
            ),
            "done": lcm.Regime(functions={"utility": _terminal}),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={
            "work": {"work": lcm.Periods(values=(0,)), "done": lcm.Periods(values=(1,))}
        },
        fixed_params={"discount_factor": 0.5},
        execution_config=lcm.ExecutionConfig(
            device_memory_bytes=budget, axis_widths={"subject": 2}
        ),
    )
    result = model.simulate(
        params={
            "ceiling": lcm.TimeVarying(values=jnp.array([0.0, 1.0]), periods=(0, 1))
        },
        initial_conditions={
            "period": jnp.zeros(n_subjects, dtype=jnp.int32),
            "regime_id": jnp.full(n_subjects, RegimeId.work),
        },
        seed=5,
        log_level="off",
    )
    frame = result.to_dataframe(use_labels=False)
    for period, action, value in ((0, 0, 2.5), (1, 1, 5.0)):
        rows = frame.loc[frame["period"] == period]
        assert len(rows) == n_subjects
        np.testing.assert_array_equal(rows["option"], jnp.full(n_subjects, action))
        np.testing.assert_allclose(rows["value"], value)


def _joint_support(*, scale: FloatND) -> FloatND:
    return jnp.array([0.0, scale])


def _joint_probabilities(*, probability: FloatND) -> FloatND:
    return jnp.array([probability, 1.0 - probability])


def _next_clock_scalar(period: ScalarInt) -> DiscreteState:
    return period


def _next_clock_period(period: Period) -> DiscreteState:
    return period


def _clock_value(*, outcome: DiscreteState, clock: DiscreteState) -> FloatND:
    return 10.0 * outcome + clock


@pytest.mark.parametrize("next_clock", [_next_clock_scalar, _next_clock_period])
def test_managed_lottery_composes_with_explicit_period(
    next_clock: UserFunction,
) -> None:
    """Managed draws and explicit period laws share their source period."""
    model = lcm.Model(
        n_periods=3,
        regimes={
            "work": lcm.Regime(
                functions={"utility": _zero},
                state_transitions={
                    "outcome": {
                        "done": lcm.StochasticTransition(
                            func=lcm.time_varying_params("probability")(
                                _joint_probabilities
                            )
                        )
                    },
                    "clock": {"done": next_clock},
                },
            ),
            "done": lcm.Regime(
                functions={"utility": _clock_value},
                states={
                    "outcome": lcm.DiscreteGrid(category_class=Option),
                    "clock": lcm.DiscreteGrid(category_class=Option),
                },
            ),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={1: "work"}),
        edges={"work": {"done": lcm.Periods(values=(1,))}},
        fixed_params={"discount_factor": 0.5},
    )
    result = model.simulate(
        params={
            "probability": lcm.TimeVarying(values=jnp.array([1.0, 0.0]), periods=(0, 1))
        },
        initial_conditions={
            "period": jnp.array([1]),
            "regime_id": jnp.array([RegimeId.work]),
        },
        seed=3,
        log_level="off",
    )
    frame = result.to_dataframe(use_labels=False)
    np.testing.assert_array_equal(frame["period"], [1, 2])
    np.testing.assert_allclose(frame["value"], [5.5, 11.0])
    np.testing.assert_array_equal(frame.loc[frame["period"] == 2, "outcome"], [1])
    np.testing.assert_array_equal(frame.loc[frame["period"] == 2, "clock"], [1])


def _joint_output(*, match: FloatND, shift: FloatND) -> FloatND:
    return match + shift


def _wealth_value(*, wealth: FloatND) -> FloatND:
    return wealth


def _joint_model() -> lcm.Model:
    return lcm.Model(
        n_periods=2,
        regimes={
            "work": lcm.Regime(
                functions={"utility": _zero},
                joint_transitions={
                    "done": {
                        "match": lcm.JointTransition(
                            support_size=2,
                            support=lcm.time_varying_params("scale")(_joint_support),
                            probabilities=lcm.time_varying_params("probability")(
                                _joint_probabilities
                            ),
                            outputs={
                                "wealth": lcm.time_varying_params("shift")(
                                    _joint_output
                                )
                            },
                        )
                    }
                },
            ),
            "done": lcm.Regime(
                functions={"utility": _wealth_value},
                states={"wealth": lcm.LinSpacedGrid(start=0.0, stop=4.0, n_points=3)},
            ),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={"work": {"done": lcm.Periods(values=(0,))}},
        fixed_params={"discount_factor": 0.5},
    )


def test_joint_temporal_roles_read_the_source_period() -> None:
    model = _joint_model()
    solution = model.solve(
        params={
            name: lcm.TimeVarying(values=jnp.array([value]), periods=(0,))
            for name, value in {"scale": 2.0, "probability": 0.25, "shift": 0.5}.items()
        },
        log_level="off",
    )
    np.testing.assert_allclose(solution.values[0]["work"], 1.0)


def test_joint_template_identifies_temporal_roles() -> None:
    template = flatten_to_qnames(_joint_model().get_params_template())
    assert "TimeVarying" in template["work__done__match__support__scale"]
    assert "TimeVarying" in template["work__done__match__probabilities__probability"]
    assert "TimeVarying" in template["work__done__next_wealth__shift"]


@lcm.categorical(ordered=False)
class GateRegimeId:
    work: ScalarInt
    done: ScalarInt
    fallback: ScalarInt


def _one() -> FloatND:
    return jnp.asarray(1.0)


def _fallback() -> FloatND:
    return jnp.asarray(2.0)


def _gate(*, V_target: FloatND, threshold: FloatND) -> BoolND:
    return V_target >= threshold


@pytest.mark.parametrize(("threshold", "expected"), [(5.0, 4.0), (10.0, 1.0)])
def test_temporal_gate_requires_and_reads_target_period(
    *, threshold: float, expected: float
) -> None:
    model = lcm.Model(
        n_periods=2,
        regimes={
            "work": lcm.Regime(functions={"utility": _zero}),
            "done": lcm.Regime(functions={"utility": _terminal}),
            "fallback": lcm.Regime(functions={"utility": _fallback}),
        },
        regime_id_class=GateRegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={
            "work": lcm.Transition(
                law={"done": lcm.StochasticTransition(func=_one)},
                gates={
                    "done": lcm.Gate(
                        predicate=lcm.time_varying_params("threshold")(_gate),
                        routes={
                            "only": lcm.StakeholderRoute(
                                fallback=lcm.ProjectedRegimeValue(
                                    regime="fallback", projection={}
                                ),
                            )
                        },
                    )
                },
            )
        },
        fixed_params={"discount_factor": 0.5},
    )
    solution = model.solve(
        params={
            "threshold": lcm.TimeVarying(values=jnp.array([threshold]), periods=(1,))
        },
        log_level="off",
    )
    np.testing.assert_allclose(solution.values[0]["work"], expected)


@pytest.mark.parametrize("missing", [None, "wage", "bonus"])
@pytest.mark.parametrize("fixed", [False, True])
def test_edge_temporal_coverage_follows_the_declaring_phase(
    *, missing: str | None, fixed: bool
) -> None:
    def simulated(*, bonus: FloatND) -> FloatND:
        return bonus

    params = {
        "wage": lcm.TimeVarying(
            values=jnp.array([] if missing == "wage" else [1.0]),
            periods=() if missing == "wage" else (1,),
        ),
        "bonus": lcm.TimeVarying(
            values=jnp.array([] if missing == "bonus" else [1.0]),
            periods=() if missing == "bonus" else (0,),
        ),
    }

    def simulate() -> lcm.SimulationResult:
        model = lcm.Model(
            n_periods=3,
            regimes={
                name: lcm.Regime(functions={"utility": _terminal})
                for name in ("work", "done")
            },
            regime_id_class=RegimeId,
            initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
            edges=lcm.Phased(
                solve={
                    "work": lcm.Transition(
                        law=lcm.ByPeriod(
                            cases={
                                0: "work",
                                1: {
                                    "done": lcm.StochasticTransition(
                                        func=lcm.time_varying_params("wage")(_flow)
                                    )
                                },
                            }
                        )
                    )
                },
                simulate={
                    "work": lcm.Transition(
                        law={
                            "done": lcm.StochasticTransition(
                                func=lcm.time_varying_params("bonus")(simulated)
                            )
                        }
                    )
                },
            ),
            fixed_params={"discount_factor": 0.5, **(params if fixed else {})},
        )
        runtime_params = {} if fixed else params
        solution = model.solve(params=runtime_params, log_level="off")
        np.testing.assert_allclose(solution.values[0]["work"], 14.0)
        return model.simulate(
            params=runtime_params,
            initial_conditions={
                "period": jnp.array([0]),
                "regime_id": jnp.array([RegimeId.work]),
            },
            solution=solution,
            seed=3,
            log_level="off",
        )

    if missing is not None:
        with pytest.raises(
            lcm.exceptions.InvalidParamsError, match=rf"{missing}.*missing"
        ):
            simulate()
    else:
        frame = simulate().to_dataframe()
        np.testing.assert_array_equal(frame["period"], [0, 1])
        assert frame["regime_name"].tolist() == ["work", "done"]


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_temporal_gate_projection_reads_target_period(
    *, reference: bool, missing: bool
) -> None:
    def compare(*, V_target: FloatND, V_reference: FloatND) -> BoolND:
        return V_target >= V_reference

    projection = lcm.ProjectedRegimeValue(
        regime="fallback",
        projection={"wealth": lcm.time_varying_params("wage")(_flow)},
    )
    model = lcm.Model(
        n_periods=2,
        regimes={
            "work": lcm.Regime(functions={"utility": _zero}),
            "done": lcm.Regime(functions={"utility": _terminal}),
            "fallback": lcm.Regime(
                states={"wealth": lcm.LinSpacedGrid(start=0, stop=10, n_points=2)},
                functions={"utility": _wealth_value},
            ),
        },
        regime_id_class=GateRegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "work"}),
        edges={
            "work": lcm.Transition(
                law={"done": lcm.StochasticTransition(func=_one)},
                gates={
                    "done": lcm.Gate(
                        predicate=compare if reference else _gate,
                        references={"V_reference": projection} if reference else {},
                        routes={"only": lcm.StakeholderRoute(fallback=projection)},
                    )
                },
            )
        },
        fixed_params={
            "discount_factor": 0.5,
            **({} if reference else {"threshold": 10.0}),
        },
    )
    params = {
        "wage": lcm.TimeVarying(
            values=jnp.array([-123.0, 9.0] if missing else [-123.0, 9.0, 3.0]),
            periods=(4, 0) if missing else (4, 0, 1),
        )
    }
    if missing:
        with pytest.raises(lcm.exceptions.InvalidParamsError, match=r"wage.*missing"):
            model.solve(params=params, log_level="off")
    else:
        solution = model.solve(params=params, log_level="off")
        np.testing.assert_allclose(
            solution.values[0]["work"], 4.0 if reference else 1.5
        )


def _weighted_log(*, consumption: ContinuousAction, weight: FloatND) -> FloatND:
    return weight * jnp.log(consumption)


def _log_bequest(*, liquid: ContinuousState) -> FloatND:
    return jnp.log(liquid)


def _log_bequest_with_kind(*, liquid: ContinuousState, kind: DiscreteState) -> FloatND:
    return jnp.log(liquid) + 0.1 * kind


def _wealth_resources(*, liquid: ContinuousState) -> FloatND:
    return liquid


def _wealth_savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _return_savings(*, savings: FloatND, gross_return: FloatND) -> FloatND:
    return gross_return * savings


def _egm_time_model(
    *,
    solver_name: str,
    temporal_role: str,
    source_periods: tuple[int, ...] = (0,),
    resources: UserFunction | lcm.PeriodSpecializedFunction = _wealth_resources,
    ride_along: bool = False,
) -> tuple[lcm.Model, dict[str, Any], lcm.LinSpacedGrid]:
    """Build a two-period weighted-log model with one temporal consumer."""
    savings_grid = lcm.LinSpacedGrid(start=0.0, stop=20.0, n_points=400)
    solver = (
        EGM(savings_grid=savings_grid)
        if solver_name == "egm"
        else NBEGM(savings_grid=savings_grid)
    )
    wealth_grid = lcm.LinSpacedGrid(start=5.0, stop=20.0, n_points=4)
    extra_states = (
        {"kind": lcm.DiscreteGrid(category_class=Option)} if ride_along else {}
    )
    utility = (
        lcm.time_varying_params("weight")(_weighted_log)
        if temporal_role == "utility"
        else _weighted_log
    )
    law = (
        lcm.time_varying_params("gross_return")(_return_savings)
        if temporal_role == "law"
        else _return_savings
    )
    model = lcm.Model(
        n_periods=max(source_periods) + 2,
        regimes={
            "work": ConsumptionSavingsRegime(
                states={"liquid": wealth_grid, **extra_states},
                actions={
                    "consumption": lcm.LinSpacedGrid(start=0.01, stop=20.0, n_points=10)
                },
                functions={
                    "utility": utility,
                    "resources": resources,
                    "savings": _wealth_savings,
                },
                state_transitions={
                    "liquid": law,
                    **({"kind": lcm.fixed_transition("kind")} if ride_along else {}),
                },
                solver=solver,
                liquid=LiquidMargin(
                    state="liquid",
                    action="consumption",
                    resources="resources",
                    post_decision_state="savings",
                ),
            ),
            "done": lcm.Regime(
                states={
                    "liquid": lcm.LogSpacedGrid(start=0.01, stop=100.0, n_points=400),
                    **extra_states,
                },
                functions={
                    "utility": _log_bequest_with_kind if ride_along else _log_bequest
                },
            ),
        },
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period=dict.fromkeys(source_periods, "work")),
        edges={"work": {"done": lcm.Periods(values=source_periods)}},
        fixed_params={"discount_factor": 1.0},
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )
    params = {
        "weight": lcm.TimeVarying(
            values=jnp.array([9.0, *[2.0 for _ in source_periods]]),
            periods=(max(source_periods) + 1, *reversed(source_periods)),
        )
        if temporal_role == "utility"
        else 2.0,
        "gross_return": lcm.TimeVarying(
            values=jnp.array([4.0, *[2.0 for _ in source_periods]]),
            periods=(max(source_periods) + 1, *reversed(source_periods)),
        )
        if temporal_role == "law"
        else 2.0,
    }
    return model, params, wealth_grid


@pytest.mark.parametrize("solver_name", ["egm", "nbegm"])
@pytest.mark.parametrize("temporal_role", ["utility", "law"])
@pytest.mark.parametrize("source_period", [0, 2])
def test_egm_temporal_consumers_use_the_source_period(
    *, solver_name: str, temporal_role: str, source_period: int
) -> None:
    """Two-period weighted log saving reads current preference and return values."""
    model, params, wealth_grid = _egm_time_model(
        solver_name=solver_name,
        temporal_role=temporal_role,
        source_periods=(source_period,),
    )
    solution = model.solve(params=params, log_level="debug")
    wealth = np.asarray(wealth_grid.to_jax())
    consumption = 2.0 / 3.0 * wealth
    expected = 2.0 * np.log(consumption) + np.log(2.0 * (wealth - consumption))
    # The terminal log grid and savings interpolation approximate the closed form.
    np.testing.assert_allclose(
        solution.value(period=source_period, regime="work"),
        expected,
        atol=1e-3,
        rtol=0,
    )


@lcm.time_varying_params("curvature")
def _time_curved_resources(*, liquid: ContinuousState, curvature: FloatND) -> FloatND:
    return liquid + curvature * liquid**2


def test_nbegm_temporal_budget_probe_checks_source_period_zero() -> None:
    """A nonlinear budget at period zero cannot hide behind later affine entries."""
    model, params, _ = _egm_time_model(
        solver_name="nbegm", temporal_role="constant", resources=_time_curved_resources
    )
    params["curvature"] = lcm.TimeVarying(values=jnp.array([0.1, 0.0]), periods=(0, 1))
    with pytest.raises(lcm.exceptions.RegimeInitializationError, match="affine"):
        check_solver_params(
            regimes=model._regimes, flat_params=model._process_params(params)
        )


def test_nbegm_temporal_budget_probe_ignores_inactive_periods() -> None:
    """An unused nonlinear row does not invalidate the source period's budget."""
    model, params, _ = _egm_time_model(
        solver_name="nbegm", temporal_role="constant", resources=_time_curved_resources
    )
    params["curvature"] = lcm.TimeVarying(values=jnp.array([0.0, 0.1]), periods=(0, 1))
    check_solver_params(
        regimes=model._regimes, flat_params=model._process_params(params)
    )


@pytest.mark.parametrize("solver_name", ["egm", "nbegm"])
def test_egm_temporal_periods_share_a_core(solver_name: str) -> None:
    """Different current weights change values without splitting the compiled core."""
    model, params, wealth_grid = _egm_time_model(
        solver_name=solver_name, temporal_role="utility", source_periods=(0, 2)
    )
    params["weight"] = lcm.TimeVarying(values=jnp.array([3.0, 2.0]), periods=(2, 0))
    kernels = model._regimes["work"].solution.period_kernels
    first = core_program_graph(kernel=kernels[0])["main"].function
    second = core_program_graph(kernel=kernels[2])["main"].function
    assert isinstance(first, partial)
    assert isinstance(second, partial)
    assert first.func is second.func
    solution = model.solve(params=params, log_level="debug")
    wealth = np.asarray(wealth_grid.to_jax())
    for period, weight in ((0, 2.0), (2, 3.0)):
        consumption = weight / (weight + 1.0) * wealth
        expected = weight * np.log(consumption) + np.log(2.0 * (wealth - consumption))
        np.testing.assert_allclose(
            solution.value(period=period, regime="work"), expected, atol=1e-3, rtol=0
        )


def _period_local_resources(source_period: int) -> UserFunction:
    """The resolved budget is affine exactly where this callback is installed."""

    def resources(
        *, liquid: ContinuousState, curvature: FloatND, period: Period
    ) -> FloatND:
        return liquid + (period != source_period) * curvature * liquid**2

    return resources


def _period_identity(period: int) -> int:
    return period


def _period_local_ride_resources(source_period: int) -> UserFunction:
    """Keep a discrete co-state while restricting the callback to its own period."""

    def resources(
        *,
        liquid: ContinuousState,
        kind: DiscreteState,
        curvature: FloatND,
        period: Period,
    ) -> FloatND:
        return liquid + 0.1 * kind + (period != source_period) * curvature * liquid**2

    return resources


@pytest.mark.parametrize("ride_along", [False, True])
def test_nbegm_temporal_probe_respects_specialized_period_groups(
    *,
    ride_along: bool,
) -> None:
    """A representative callback must not be tested in another callback's period."""
    model, params, _ = _egm_time_model(
        solver_name="nbegm",
        temporal_role="constant",
        source_periods=(0, 2),
        ride_along=ride_along,
        resources=lcm.PeriodSpecializedFunction(
            build=_period_local_ride_resources
            if ride_along
            else _period_local_resources,
            signature=_period_identity,
        ),
    )
    params["curvature"] = 0.1
    check_solver_params(
        regimes=model._regimes, flat_params=model._process_params(params)
    )
