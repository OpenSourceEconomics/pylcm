"""Public tagged current innovations preserve original rows and producing laws."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.entry_inputs import capture_simulation_entry_inputs
from lcm import (
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    Model,
    Periods,
    Regime,
    categorical,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidSimulationInputError,
    ShockDeclarationError,
)
from lcm.shocks import (
    BaseInnovations,
    MonteCarlo,
    Normal,
    PhysicalValues,
    ShockGroup,
    Uniform,
)
from lcm.typing import BoolND, FloatND, IntND, ScalarInt
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
