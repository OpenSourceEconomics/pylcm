"""Public tagged current innovations preserve original rows and producing laws."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from _lcm.simulation.entry_inputs import capture_simulation_entry_inputs
from lcm import (
    ArrivalDecision,
    ArrivalRead,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Periods,
    ProjectedRegimeValue,
    Regime,
    SimulationResult,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidSimulationInputError,
    ShockDeclarationError,
)
from lcm.shocks import (
    BaseInnovations,
    Discrete,
    GaussHermite,
    MonteCarlo,
    Normal,
    PhysicalValues,
    Product,
    ShockGroup,
    Uniform,
)
from lcm.typing import BoolND, FloatND, IntND, ScalarInt
from tests.conftest import DECIMAL_PRECISION
from tests.test_shock_overrides import _keys, _plan


@categorical(ordered=False)
class _Regimes:
    first: ScalarInt
    middle: ScalarInt
    last: ScalarInt


@categorical(ordered=False)
class _Actions:
    off: ScalarInt
    on: ScalarInt


def _utility(*, eps: FloatND, action: IntND) -> FloatND:
    return eps * (2 * action - 1)


def _model(*, budgeted: bool = False, group: ShockGroup | None = None) -> Model:
    return Model(
        regimes={
            name: Regime(
                functions={"utility": _utility},
                actions={"action": DiscreteGrid(category_class=_Actions)},
                shocks=("risk",),
            )
            for name in ("first", "middle", "last")
        },
        shocks={
            "risk": group
            or ShockGroup(
                shocks={"eps": Normal(mu=1, sigma=None)},
                nodes=MonteCarlo(2, seed=7),
            )
        },
        n_periods=3,
        initial_nodes=InitialNodes(by_period={0: "first", 1: "middle", 2: "last"}),
        edges={
            "first": {"middle": Periods(values=(0,))},
            "middle": {"last": Periods(values=(1,))},
        },
        regime_id_class=_Regimes,
        execution_config=ExecutionConfig(
            axis_widths={"subject": 2},
            device_memory_bytes=2**27 if budgeted else None,
        ),
    )


@pytest.mark.parametrize("base", [False, True])
@pytest.mark.parametrize("budgeted", [False, True])
@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_public_path_tag_binds_the_current_law(*, base: bool, budgeted: bool) -> None:
    """A base path changes with parameters; a physical path retains its bits."""
    model = _model(budgeted=budgeted)
    values = np.asarray([[2, 3, 4], [-2, -3, -4]], dtype=jnp.asarray(0.0).dtype)
    for scale in (1.0, 3.0):
        frame = model.simulate(
            params={"discount_factor": 0.5, "risk": {"eps__sigma": scale}},
            initial_conditions={
                "period": np.zeros(2, dtype=np.int32),
                "regime_id": np.zeros(2, dtype=np.int32),
            },
            shock_paths={
                "eps": BaseInnovations(values=values)
                if base
                else PhysicalValues(values=values)
            },
            seed=91,
            log_level="off",
            runtime_checks=False,
        ).to_dataframe(use_categorical_codes=True)
        expected = 1 + scale * values if base else values
        for period in range(3):
            np.testing.assert_array_equal(
                frame.query("period == @period")["eps"], expected[:, period]
            )
            np.testing.assert_array_equal(
                frame.query("period == @period")["action"], expected[:, period] > 0
            )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_entry_is_dated_and_path_tag_takes_precedence() -> None:
    """Only actual entry cells consume entry inputs, in original caller order."""
    model = _model()
    values = np.asarray([5, 7, 9], dtype=jnp.asarray(0.0).dtype)
    initial = {
        "period": np.asarray([2, 0, 1], dtype=np.int32),
        "regime_id": np.asarray([2, 0, 1], dtype=np.int32),
        "eps": PhysicalValues(values=values),
    }
    frame = model.simulate(
        params={"discount_factor": 0.5, "risk": {"eps__sigma": 2.0}},
        initial_conditions=initial,
        seed=91,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe()
    for row, period in enumerate(initial["period"]):
        # Result rows preserve caller subject identity despite late entry/chunks.
        selected = frame.loc[(frame["subject_id"] == row) & (frame["period"] == period)]
        np.testing.assert_array_equal(selected["eps"], values[row])
    baseline = model.simulate(
        params={"discount_factor": 0.5, "risk": {"eps__sigma": 2.0}},
        initial_conditions={
            name: value for name, value in initial.items() if name != "eps"
        },
        seed=91,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe()
    for row, entry in enumerate(initial["period"]):
        np.testing.assert_array_equal(
            frame.loc[(frame["subject_id"] == row) & (frame["period"] > entry), "eps"],
            baseline.loc[
                (baseline["subject_id"] == row) & (baseline["period"] > entry), "eps"
            ],
        )
    overridden = model.simulate(
        params={"discount_factor": 0.5, "risk": {"eps__sigma": 2.0}},
        initial_conditions=initial,
        shock_paths={"eps": BaseInnovations(values=np.ones(3, dtype=values.dtype))},
        seed=91,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe()
    np.testing.assert_array_equal(overridden["eps"], 3.0)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_untagged_current_iid_path_has_a_migration_error() -> None:
    with pytest.raises(
        InvalidSimulationInputError, match=r"BaseInnovations|PhysicalValues"
    ):
        _model().simulate(
            params={"discount_factor": 0.5, "risk": {"eps__sigma": 1.0}},
            initial_conditions={
                "period": np.zeros(1, dtype=np.int32),
                "regime_id": np.zeros(1, dtype=np.int32),
            },
            shock_paths={"eps": np.ones(3, dtype=jnp.asarray(0.0).dtype)},
            log_level="off",
            runtime_checks=False,
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_wrapped_original_buffer_is_charged_before_conversion() -> None:
    """A wrapper cannot hide its existing device allocation from entry admission."""
    original = jnp.ones(8)
    with pytest.raises(ExecutionPlanningError):
        capture_simulation_entry_inputs(
            execution=replace(_model()._execution, device_memory_bytes=1),
            params={},
            initial_conditions={"eps": BaseInnovations(values=original)},
            solution=None,
        )


@pytest.mark.parametrize("tiny", [False, True])
@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_inactive_observation_preserves_joint_draw_and_derivative(
    *, tiny: bool
) -> None:
    """Unread partial correlated observations cannot condition a joint frame."""
    group = ShockGroup(
        shocks={"x": Normal(mu=0, sigma=None), "y": Normal(mu=0, sigma=1)},
        nodes=MonteCarlo(2, seed=1),
        correlation=((1.0, 0.5), (0.5, 1.0)),
    )
    plan, keys = _plan(group=group), _keys(group=group)

    def draw(*, scale: FloatND, supplied: bool) -> FloatND:
        frame = plan.realise(
            keys=keys,
            group_params={"risk": {"x__sigma": scale}},
            arguments={"period": jnp.int32(0), "age": jnp.int32(0)},
            overrides={"x": PhysicalValues(values=jnp.asarray(jnp.nan))}
            if supplied
            else {},
            override_active={"x": jnp.asarray(a=False)} if supplied else {},
        )
        return jnp.where(
            frame.valid, frame.physical["x"] + frame.physical["y"], jnp.nan
        )

    dtype = np.dtype(jnp.asarray(0.0).dtype)
    scale_value = 0.5 * np.sqrt(np.finfo(dtype).tiny) if tiny else 2.0
    expected = jax.jit(
        jax.value_and_grad(lambda scale: draw(scale=scale, supplied=False))
    )(jnp.asarray(scale_value))
    actual = jax.jit(
        jax.value_and_grad(lambda scale: draw(scale=scale, supplied=True))
    )(jnp.asarray(scale_value))
    assert np.all(np.isfinite(np.asarray(expected))), (
        "ordinary forward value/gradient must be finite"
    )
    np.testing.assert_array_equal(actual, expected)
    invalid = plan.realise(
        keys=keys,
        group_params={"risk": {"x__sigma": 2.0}},
        arguments={"period": jnp.int32(0), "age": jnp.int32(0)},
        overrides={"x": PhysicalValues(values=jnp.asarray(1.0))},
        override_active={"x": jnp.asarray(a=True)},
    )
    assert not bool(invalid.valid)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_subject_vmap_keeps_inactive_inverse_arithmetic_and_gradients_safe(
    *, uniform: bool, mixed: bool
) -> None:
    """A subject mask protects unread tiny scales even after vmap batching."""
    group = ShockGroup(
        shocks={
            "x": Uniform(start=0, stop=None) if uniform else Normal(mu=0, sigma=None)
        },
        nodes=MonteCarlo(2, seed=1),
    )
    plan, keys = _plan(group=group), _keys(group=group)
    parameter = "x__stop" if uniform else "x__sigma"
    dtype = np.dtype(jnp.asarray(0.0).dtype)
    tiny = 0.5 * np.sqrt(np.finfo(dtype).tiny)
    active = jnp.asarray([False, mixed])
    scales = jnp.where(active, 2.0, tiny)
    supplied = jnp.where(active, 0.0, jnp.nan)

    # keyword-only-exempt: library-callback=jax.value_and_grad
    def draw(scale: FloatND, enabled: BoolND, value: FloatND) -> FloatND:
        frame = plan.realise(
            keys=keys,
            group_params={"risk": {parameter: scale}},
            arguments={"period": jnp.int32(0), "age": jnp.int32(0)},
            overrides={"x": PhysicalValues(values=value)},
            override_active={"x": enabled},
        )
        return jnp.where(frame.valid, frame.physical["x"], jnp.nan)

    def ordinary(scale: FloatND) -> FloatND:
        frame = plan.realise(
            keys=keys,
            group_params={"risk": {parameter: scale}},
            arguments={"period": jnp.int32(0), "age": jnp.int32(0)},
        )
        return jnp.where(frame.valid, frame.physical["x"], jnp.nan)

    expected = jax.jit(jax.vmap(jax.value_and_grad(ordinary)))(scales)
    expected = tuple(jnp.where(active, 0.0, leaf) for leaf in expected)
    actual = jax.jit(jax.vmap(jax.value_and_grad(draw)))(scales, active, supplied)
    assert np.all(np.isfinite(np.asarray(expected)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_superseded_physical_cell_does_not_undo_an_active_correlated_base() -> None:
    group = ShockGroup(
        shocks={"x": Normal(mu=0, sigma=1), "y": Normal(mu=0, sigma=1)},
        nodes=MonteCarlo(2, seed=1),
        correlation=((1.0, 0.5), (0.5, 1.0)),
    )
    plan = _plan(group=group)
    keys = _keys(group=group)
    base = BaseInnovations(values=jnp.asarray(2.0))
    expected = plan.realise(
        keys=keys,
        group_params={},
        arguments={"period": jnp.int32(0), "age": jnp.int32(0)},
        overrides={"x": base},
    )
    actual = plan.realise(
        keys=keys,
        group_params={},
        arguments={"period": jnp.int32(0), "age": jnp.int32(0)},
        overrides={"x": base, "y": PhysicalValues(values=jnp.asarray(jnp.nan))},
        override_active={"x": jnp.asarray(a=True), "y": jnp.asarray(a=False)},
    )
    assert bool(actual.valid)
    np.testing.assert_array_equal(actual.physical["y"], expected.physical["y"])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_tagged_inputs_refuse_narrowing_before_public_consumption() -> None:
    with pytest.raises(ShockDeclarationError, match="narrow"):
        PhysicalValues(values=np.ones(3, dtype=np.int64))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("masked", [False, True])
def test_physical_dtype_precedes_inactive_sanitization(*, masked: bool) -> None:
    group = ShockGroup(shocks={"x": Normal(mu=0, sigma=1)}, nodes=MonteCarlo(2, seed=7))
    dtype = np.float32 if jax.config.jax_enable_x64 else np.float16
    with pytest.raises(ShockDeclarationError, match="producing law's dtype"):
        _plan(group=group).realise(
            keys=_keys(group=group),
            group_params={},
            arguments={"period": jnp.int32(0), "age": jnp.int32(0)},
            overrides={"x": PhysicalValues(values=np.asarray(0.5, dtype=dtype))},
            override_active={"x": jnp.asarray(a=False)} if masked else {},
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_partial_correlated_frame_has_an_explicit_refusal() -> None:
    group = ShockGroup(
        shocks={"eps": Normal(mu=0, sigma=1), "other": Normal(mu=0, sigma=1)},
        correlation=((1.0, 0.5), (0.5, 1.0)),
        nodes=MonteCarlo(2, seed=7),
    )
    with pytest.raises(ShockDeclarationError, match="complete physical frame"):
        _model(group=group).simulate(
            params={"discount_factor": 0.5},
            initial_conditions={
                "period": np.zeros(1, dtype=np.int32),
                "regime_id": np.zeros(1, dtype=np.int32),
            },
            shock_paths={
                "eps": PhysicalValues(values=np.ones(3, dtype=jnp.asarray(0.0).dtype))
            },
            log_level="off",
            runtime_checks=False,
        )


def _simulate_correlated_paths(
    *, paths: dict[str, BaseInnovations | PhysicalValues], runtime_checks: bool
) -> pd.DataFrame:
    group = ShockGroup(
        shocks={"eps": Normal(mu=0, sigma=1), "other": Normal(mu=0, sigma=1)},
        correlation=((1.0, 0.5), (0.5, 1.0)),
        nodes=MonteCarlo(2, seed=7),
    )
    return (
        _model(group=group)
        .simulate(
            params={"discount_factor": 0.5},
            initial_conditions={
                "period": np.zeros(2, dtype=np.int32),
                "regime_id": np.zeros(2, dtype=np.int32),
            },
            shock_paths=paths,
            seed=3,
            log_level="off",
            runtime_checks=runtime_checks,
        )
        .to_dataframe(use_categorical_codes=True)
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_public_mixed_correlated_frame_has_an_explicit_refusal(
    *, runtime_checks: bool
) -> None:
    """A physical cell beside a base cell of one correlated group is refused."""
    dtype = jnp.asarray(0.0).dtype
    with pytest.raises(ShockDeclarationError, match="complete physical frame"):
        _simulate_correlated_paths(
            paths={
                "eps": PhysicalValues(values=np.ones(3, dtype=dtype)),
                "other": BaseInnovations(values=np.zeros(3, dtype=dtype)),
            },
            runtime_checks=runtime_checks,
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_public_complete_correlated_frame_keeps_supplied_bits(
    *, runtime_checks: bool
) -> None:
    """A complete correlated physical frame is admitted and replayed exactly."""
    eps = np.asarray([[0.5, -1.0, 2.0], [-0.25, 1.5, -2.0]])
    eps = eps.astype(jnp.asarray(0.0).dtype)
    frame = _simulate_correlated_paths(
        paths={
            "eps": PhysicalValues(values=eps),
            "other": PhysicalValues(values=-eps),
        },
        runtime_checks=runtime_checks,
    )
    for period in range(3):
        rows = frame.query("period == @period").sort_values("subject_id")
        np.testing.assert_array_equal(rows["eps"], eps[:, period])
        np.testing.assert_array_equal(rows["action"], eps[:, period] > 0)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("shape", [(), (4, 3), (2,)])
def test_public_path_shape_and_coverage_are_explicit(shape: tuple[int, ...]) -> None:
    with pytest.raises(InvalidSimulationInputError, match=r"shape|cover"):
        _model().simulate(
            params={"discount_factor": 0.5, "risk": {"eps__sigma": 1.0}},
            initial_conditions={
                "period": np.zeros(1, dtype=np.int32),
                "regime_id": np.zeros(1, dtype=np.int32),
            },
            shock_paths={
                "eps": BaseInnovations(
                    values=np.ones(shape, dtype=jnp.asarray(0.0).dtype)
                )
            },
            log_level="off",
            runtime_checks=False,
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_unknown_tagged_name_is_not_ignored() -> None:
    with pytest.raises(InvalidSimulationInputError, match="not a named current-IID"):
        _model().simulate(
            params={"discount_factor": 0.5, "risk": {"eps__sigma": 1.0}},
            initial_conditions={
                "period": np.zeros(1, dtype=np.int32),
                "regime_id": np.zeros(1, dtype=np.int32),
            },
            shock_paths={
                "unknown": BaseInnovations(
                    values=np.ones(3, dtype=jnp.asarray(0.0).dtype)
                )
            },
            log_level="off",
            runtime_checks=False,
        )


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_budgeted_call_cache_retains_tag_schema_without_freezing_values() -> None:
    model = _model(budgeted=True)
    dtype = jnp.asarray(0.0).dtype
    for record, supplied, expected in (
        (PhysicalValues, 1.0, 1.0),
        (BaseInnovations, 1.0, 3.0),
        (BaseInnovations, 2.0, 5.0),
        (PhysicalValues, 2.0, 2.0),
    ):
        frame = model.simulate(
            params={"discount_factor": 0.5, "risk": {"eps__sigma": 2.0}},
            initial_conditions={
                "period": np.zeros(1, dtype=np.int32),
                "regime_id": np.zeros(1, dtype=np.int32),
            },
            shock_paths={"eps": record(values=np.full(3, supplied, dtype=dtype))},
            seed=91,
            log_level="off",
            runtime_checks=False,
        ).to_dataframe()
        np.testing.assert_array_equal(frame["eps"], expected)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_unread_path_prefix_does_not_consume_invalid_values() -> None:
    dtype = jnp.asarray(0.0).dtype
    paths = np.asarray([[np.nan, np.nan, 5], [1, 2, 3], [np.nan, 4, 6]], dtype=dtype)
    frame = (
        _model()
        .simulate(
            params={"discount_factor": 0.5, "risk": {"eps__sigma": 2.0}},
            initial_conditions={
                "period": np.asarray([2, 0, 1], dtype=np.int32),
                "regime_id": np.asarray([2, 0, 1], dtype=np.int32),
            },
            shock_paths={"eps": PhysicalValues(values=paths)},
            seed=91,
            log_level="off",
            runtime_checks=False,
        )
        .to_dataframe()
    )
    np.testing.assert_array_equal(
        frame["eps"],
        paths[
            frame["subject_id"].to_numpy(dtype=int), frame["period"].to_numpy(dtype=int)
        ],
    )


@categorical(ordered=True)
class _OrderedOffer:
    unavailable: ScalarInt
    offered: ScalarInt


@categorical(ordered=False)
class _UnorderedOffer:
    unavailable: ScalarInt
    offered: ScalarInt


@categorical(ordered=False)
class _OfferRegime:
    alive: ScalarInt
    done: ScalarInt


def _offer_utility(*, eps: IntND, action: IntND) -> FloatND:
    return jnp.where(action == eps, 2.0, -1.0)


def _recorded_offer(*, eps: IntND) -> IntND:
    return eps


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_discrete_override_labels_codes_and_saved_raw_values(
    *, tmp_path: Path, physical: bool, ordered: bool
) -> None:
    """Tagged categorical draws retain labels, order and raw codes after loading."""
    model = Model(
        regimes={
            name: Regime(
                functions={
                    "utility": _offer_utility,
                    "recorded_offer": _recorded_offer,
                },
                output_only_functions=("recorded_offer",),
                actions={"action": DiscreteGrid(category_class=_Actions)},
                shocks=("offer",),
            )
            for name in ("alive", "done")
        },
        shocks={
            "offer": ShockGroup(
                shocks={
                    "eps": Discrete(
                        categories=_OrderedOffer if ordered else _UnorderedOffer,
                        probs=(0.5, 0.5),
                    )
                }
            )
        },
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "alive", 1: "done"}),
        edges={"alive": {"done": Periods(values=(0,))}},
        regime_id_class=_OfferRegime,
        execution_config=ExecutionConfig(
            axis_widths={"subject": 2}, device_memory_bytes=2**27
        ),
    )
    assert model.reachability.visited_nodes == frozenset({(0, "alive"), (1, "done")})
    starts = np.asarray([0, 1, 0], dtype=np.int32)
    expected = np.asarray([[0, 1], [1, 0], [1, 1]], dtype=np.int32)
    payload = (
        PhysicalValues(values=expected)
        if physical
        else BaseInnovations(
            values=np.where(expected == 0, 0.25, 0.75).astype(jnp.asarray(0.0).dtype)
        )
    )
    result = model.simulate(
        params={"discount_factor": 0.5},
        initial_conditions={
            "period": starts,
            "regime_id": starts,
        },
        shock_paths={"eps": payload},
        seed=91,
        log_level="off",
        runtime_checks=False,
    )
    labels = result.to_dataframe(additional_targets=["recorded_offer"])
    codes = result.to_dataframe(
        additional_targets=["recorded_offer"], use_categorical_codes=True
    )
    assert isinstance(labels["eps"].dtype, pd.CategoricalDtype)
    assert tuple(labels["eps"].cat.categories) == ("unavailable", "offered")
    assert labels["eps"].cat.ordered is ordered
    rows = codes["subject_id"].to_numpy()
    periods = codes["period"].to_numpy()
    np.testing.assert_array_equal(codes["eps"], expected[rows, periods])
    np.testing.assert_array_equal(codes["action"], codes["eps"])
    np.testing.assert_array_equal(codes["recorded_offer"], codes["eps"])
    np.testing.assert_array_equal(labels["eps"].cat.codes, codes["eps"])
    raw = {
        (regime_name, period): np.array(data.shocks["eps"], copy=True)
        for regime_name, periods in result.raw_results.items()
        for period, data in periods.items()
    }
    randomness = result.randomness
    directory = result.save(
        directory=tmp_path / "result", df_additional_targets=["recorded_offer"]
    )
    loaded = SimulationResult.load(directory=directory)
    assert_frame_equal(
        loaded.to_dataframe(additional_targets=["recorded_offer"]), labels
    )
    assert_frame_equal(
        loaded.to_dataframe(
            additional_targets=["recorded_offer"], use_categorical_codes=True
        ),
        codes,
    )
    assert loaded.randomness == randomness
    for (regime_name, period), values in raw.items():
        np.testing.assert_array_equal(
            loaded.raw_results[regime_name][period].shocks["eps"], values
        )


@categorical(ordered=False)
class _ArrivalRegimes:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


@categorical(ordered=True)
class _ArrivalBinary:
    low: ScalarInt
    high: ScalarInt


def _arrival_source(*, x: FloatND, source_signal: FloatND) -> FloatND:
    return x + 0.0 * source_signal


def _arrival_target(*, x: FloatND, offer: IntND, choice: IntND) -> FloatND:
    return x + offer + choice * (2.0 * offer - 1.0)


def _arrival_fallback(*, x: FloatND) -> FloatND:
    return x - 2.0


def _arrival_double(*, x: FloatND) -> FloatND:
    return 2.0 * x


def _arrival_identity(*, x: FloatND) -> FloatND:
    return x


def _arrival_accept(*, x: FloatND, V_target: FloatND) -> BoolND:
    return V_target > x + 0.5


def _arrival_model(*, observed: bool) -> Model:
    return Model(
        regimes={
            "source": Regime(
                states={"x": LinSpacedGrid(start=1.0, stop=3.0, n_points=3)},
                state_transitions={"x": _arrival_double},
                functions={"utility": _arrival_source},
                shocks=("source_noise",),
            ),
            "target": Regime(
                states={"x": LinSpacedGrid(start=2.0, stop=6.0, n_points=3)},
                functions={"utility": _arrival_target},
                actions={"choice": DiscreteGrid(category_class=_ArrivalBinary)},
                shocks=("target_offer",),
            ),
            "fallback": Regime(
                states={"x": LinSpacedGrid(start=2.0, stop=6.0, n_points=3)},
                functions={"utility": _arrival_fallback},
            ),
        },
        shocks={
            "source_noise": ShockGroup(
                shocks={"source_signal": Normal(mu=0, sigma=1)},
                nodes=Product(source_signal=GaussHermite(3)),
            ),
            "target_offer": ShockGroup(
                shocks={"offer": Discrete(categories=_ArrivalBinary, probs=(0.5, 0.5))}
            ),
        },
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        regime_id_class=_ArrivalRegimes,
        edges={
            "source": Transition(
                targets={
                    "target": Periods(values=(0,)),
                    "fallback": Periods(values=(0,)),
                },
                law="target",
                arrivals={
                    "target": ArrivalDecision(
                        predicate=_arrival_accept,
                        observes=("offer",) if observed else (),
                        reads={
                            "V_target": ArrivalRead(
                                regime="target",
                                kind="Raw",
                                projection={"x": _arrival_identity},
                            )
                        },
                        routes={
                            "self": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"x": _arrival_identity},
                                )
                            )
                        },
                    )
                },
            )
        },
    )


def _arrival_initial() -> dict[str, jax.Array]:
    return {
        "period": jnp.zeros(3, dtype=jnp.int32),
        "regime_id": jnp.zeros(3, dtype=jnp.int32),
        "x": jnp.asarray([3.0, 1.0, 2.0]),
    }


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_incoming_arrival_history_refuses_a_missing_consumed_period() -> None:
    """The incoming-only target consumes its offer at the landing date."""
    model = _arrival_model(observed=True)
    with pytest.raises(InvalidSimulationInputError, match=r"offer.*cover period 1"):
        model.simulate(
            params={"discount_factor": 0.5},
            initial_conditions=_arrival_initial(),
            shock_paths={"offer": BaseInnovations(values=jnp.asarray([0.25]))},
            seed=17,
            log_level="off",
            runtime_checks=False,
        )


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_incoming_arrival_history_keeps_rows_and_declared_information(
    *, observed: bool
) -> None:
    """Low supplied offers affect observed consent, but never hidden pricing."""
    frame = (
        _arrival_model(observed=observed)
        .simulate(
            params={"discount_factor": 0.5},
            initial_conditions=_arrival_initial(),
            shock_paths={
                "offer": PhysicalValues(
                    values=jnp.asarray([[1, 0], [0, 1], [1, 0]], dtype=jnp.int32)
                )
            },
            seed=17,
            log_level="off",
            runtime_checks=False,
        )
        .to_dataframe(use_categorical_codes=True)
    )
    source = frame.loc[frame["period"] == 0].sort_values("subject_id")
    np.testing.assert_array_almost_equal(
        source["value"],
        [6.0, 2.0, 4.0] if observed else [6.5, 2.5, 4.5],
        decimal=DECIMAL_PRECISION,
    )
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], [0, 1, 2])
    np.testing.assert_array_equal(landed["x"], [6.0, 2.0, 4.0])
    expected = ["fallback", "target", "fallback"] if observed else ["target"] * 3
    np.testing.assert_array_equal(landed["regime_name"], expected)
    kept = landed.loc[landed["regime_name"] == "target"]
    offers = np.asarray([0, 1, 0])[kept["subject_id"].to_numpy(dtype=int)]
    np.testing.assert_array_equal(kept["offer"], offers)
    np.testing.assert_array_equal(kept["choice"], offers)
    np.testing.assert_array_almost_equal(
        landed["value"],
        [4.0, 4.0, 2.0] if observed else [6.0, 4.0, 4.0],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_unread_arrival_history_prefix_preserves_fresh_target_draws() -> None:
    """A source-only input must not replace later target named draws."""
    model = _arrival_model(observed=True)
    frames = [
        model.simulate(
            params={"discount_factor": 0.5},
            initial_conditions=_arrival_initial(),
            shock_paths=paths,
            seed=17,
            log_level="off",
            runtime_checks=False,
        ).to_dataframe(use_categorical_codes=True)
        for paths in (
            None,
            {"source_signal": BaseInnovations(values=jnp.asarray([123.0]))},
        )
    ]
    landed = frames[0].loc[frames[0]["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], [0, 1, 2])
    np.testing.assert_array_equal(landed["x"], [6.0, 2.0, 4.0])
    assert_frame_equal(
        frames[0].loc[frames[0]["period"] == 1],
        frames[1].loc[frames[1]["period"] == 1],
    )
