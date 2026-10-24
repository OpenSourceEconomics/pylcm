"""Current exits choose with caller information and act once after revelation."""

from functools import partial
from types import MappingProxyType
from typing import Unpack, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal

from _lcm.params.edges import regime_kernel_params
from _lcm.regime_building import processing
from _lcm.simulation.chunk_profiles import profile_simulation_chunk
from _lcm.simulation.exits import ExitRoute, route_original_subjects
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.random_address import (
    RandomAddress,
    draw_window_keys,
    encode_subject_row,
)
from _lcm.simulation.residency import measure_buffer_footprint, resident_bytes_by_device
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.solution.contract import SolverBuildContext
from _lcm.typing import PytreeValue
from lcm import (
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Periods,
    Phased,
    Regime,
    RuleResult,
    categorical,
)
from lcm.exceptions import (
    InvalidSimulationInputError,
    ShockDeclarationError,
)
from lcm.exits import ExitRule
from lcm.shocks import (
    BaseInnovations,
    Discrete,
    GaussHermite,
    Normal,
    PhysicalValues,
    Product,
    ShockGroup,
)
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import BoolND, FloatND, IntND, RegimeName, ScalarInt
from tests.conftest import DECIMAL_PRECISION
from tests.simulation._callback_types import HostDispatch, RuntimeDispatch
from tests.simulation.test_aot_collective_and_gated import _capture_compiled_dispatches
from tests.solution.test_public_exit_queries import _household, _rule_model, _StrictRule


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt


@categorical(ordered=True)
class _Binary:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _OnlySource:
    source: ScalarInt


@categorical(ordered=False)
class _ThreeRegimes:
    source: ScalarInt
    target: ScalarInt
    unread: ScalarInt


@categorical(ordered=False)
class _FormationRegimes:
    source: ScalarInt
    counterparty: ScalarInt
    target: ScalarInt


@categorical(ordered=False)
class _OpposingRegimes:
    source: ScalarInt
    target: ScalarInt
    source_done: ScalarInt
    target_done: ScalarInt


def _addressed_taste_actions(*, regime: RegimeName, values: FloatND) -> np.ndarray:
    """Literal choice from the named original-row taste address, seed17."""
    keys = draw_window_keys(
        key=jax.random.key(17),
        address_words=np.asarray(
            RandomAddress(kind="taste", name=regime).words(), dtype=np.uint32
        ),
        subject_start=encode_subject_row(row=0),
        last_real_subject=encode_subject_row(row=11),
        subject_count=12,
        period=0,
    )
    noise = jax.vmap(
        partial(jax.random.gumbel, shape=values.shape, dtype=values.dtype)
    )(keys)
    return np.asarray(jnp.argmax(values + noise, axis=-1))


@pytest.mark.parametrize("reverse", [False, True])
def test_opposing_routes_read_original_membership_and_coordinates(
    *, reverse: bool
) -> None:
    """An entrant cannot take the destination's original occupants' exit."""
    routes = (
        ExitRoute(
            source="source",
            target="target",
            identifier=1,
            source_role=None,
            target_role=-1,
            projection="outward",
        ),
        ExitRoute(
            source="target",
            target="source",
            identifier=1,
            source_role=None,
            target_role=-1,
            projection="inward",
        ),
    )
    ids, _, states, moved, paid = route_original_subjects(
        original_regime_ids=jnp.asarray([0, 1, 0], dtype=jnp.int32),
        original_roles=jnp.full(3, -1, dtype=jnp.int32),
        original_states={
            "source": {"x": jnp.asarray([10.0, 11.0, 12.0])},
            "target": {"y": jnp.asarray([20.0, 21.0, 22.0])},
        },
        decisions={
            "source": jnp.asarray([1, 0, 0], dtype=jnp.int32),
            "target": jnp.asarray([0, 1, 0], dtype=jnp.int32),
        },
        projections={
            "outward": {"y": jnp.asarray([110.0, 111.0, 112.0])},
            "inward": {"x": jnp.asarray([220.0, 221.0, 222.0])},
        },
        adjustments={
            "outward": jnp.full(3, -0.25),
            "inward": jnp.full(3, -0.5),
        },
        routes=routes[::-1] if reverse else routes,
        regime_ids={"source": 0, "target": 1},
    )
    np.testing.assert_array_equal(ids, [1, 0, 0])
    np.testing.assert_array_equal(states["source"]["x"], [10, 221, 12])
    np.testing.assert_array_equal(states["target"]["y"], [110, 21, 22])
    np.testing.assert_array_equal(moved, [True, True, False])
    np.testing.assert_array_equal(paid, [-0.25, -0.5, 0])


def _stay() -> float:
    return 0.5


def _target(*, offer: ScalarInt, choice: ScalarInt) -> FloatND:
    return offer + choice * (2.0 * offer - 1.0)


def _adjustment() -> float:
    return -0.25


def _unused_probabilities() -> FloatND:
    raise AssertionError("An unread permitted shock law must not execute.")


def _hidden_offer_model(
    *,
    observed: bool,
    enable_jit: bool,
    permit_unused: bool = False,
    execution_config: ExecutionConfig | None = None,
) -> Model:
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": _stay},
                shocks=("unused",) if permit_unused else (),
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target",
                                projection={},
                                utility_adjustment=_adjustment,
                            )
                        },
                        rule="unilateral",
                        observes=("offer",) if observed else (),
                    )
                },
            ),
            "target": Regime(
                functions={"utility": _target},
                actions={"choice": DiscreteGrid(category_class=_Binary)},
                shocks=("innovation",),
            ),
        },
        shocks={
            "innovation": ShockGroup(
                shocks={"offer": Discrete(categories=_Binary, probs=(0.5, 0.5))},
                nodes=Product(),
            ),
            **(
                {
                    "unused": ShockGroup(
                        shocks={
                            "unused_signal": Discrete(
                                categories=_Binary, probs=_unused_probabilities
                            )
                        },
                        nodes=Product(),
                    )
                }
                if permit_unused
                else {}
            ),
        },
        n_periods=1,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={},
        enable_jit=enable_jit,
        # Compiler-derived admission requires compiled execution.
        execution_config=execution_config
        if execution_config is not None
        else ExecutionConfig()
        if enable_jit
        else ExecutionConfig(device_memory_bytes=None),
    )


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("enable_jit", [False, True])
def test_hidden_target_reveals_action_and_value_after_irrevocable_exit(
    *,
    enable_jit: bool,
) -> None:
    """A mover keeps its prior choice even when the revealed payoff is lower."""
    model = _hidden_offer_model(observed=False, enable_jit=enable_jit)
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    # The source prices the target policy at E[max u] minus the one-off fee.
    assert_array_almost_equal(
        solution.values[0]["source"], 0.75, decimal=DECIMAL_PRECISION
    )
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(12, dtype=jnp.int32),
            "regime_id": jnp.zeros(12, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame["regime_name"], "target")
    assert set(frame["offer"]) == {0, 1}
    np.testing.assert_array_equal(frame["choice"], frame["offer"])
    assert_array_almost_equal(
        frame["value"], 2.0 * frame["offer"] - 0.25, decimal=DECIMAL_PRECISION
    )


def test_original_exit_frame_evaluates_only_demanded_permitted_groups() -> None:
    """Permission alone does not execute a law in the occupant's source frame."""
    model = _hidden_offer_model(observed=False, enable_jit=True, permit_unused=True)
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(1, dtype=jnp.int32),
            "regime_id": jnp.zeros(1, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame["regime_name"], "target")


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("entry", [False, True])
def test_tagged_exit_innovations_preserve_hidden_pricing(*, entry: bool) -> None:
    """A supplied low target draw changes the final action, not hidden pricing."""
    model = _hidden_offer_model(observed=False, enable_jit=True)
    initial = {
        "period": jnp.zeros(1, dtype=jnp.int32),
        "regime_id": jnp.zeros(1, dtype=jnp.int32),
    }
    tagged = BaseInnovations(values=jnp.asarray([0.25]))
    frame = model.simulate(
        params={},
        initial_conditions={**initial, **({"offer": tagged} if entry else {})},
        shock_paths=None if entry else {"offer": tagged},
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame["regime_name"], "target")
    np.testing.assert_array_equal(frame["offer"], 0)
    np.testing.assert_array_equal(frame["choice"], 0)
    assert_array_almost_equal(frame["value"], -0.25, decimal=DECIMAL_PRECISION)


@categorical(ordered=False)
class _DatedOverrideRegimes:
    start: ScalarInt
    source: ScalarInt
    target: ScalarInt
    source_done: ScalarInt
    target_done: ScalarInt


def _start_flow(source_signal: FloatND) -> FloatND:
    return 0.0 * source_signal


def _dated_override_model() -> Model:
    return Model(
        regimes={
            "start": Regime(
                functions={"utility": _start_flow}, shocks=("source_innovation",)
            ),
            "source": Regime(
                functions={"utility": _stay},
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target",
                                projection={},
                                utility_adjustment=_adjustment,
                            )
                        },
                        rule="unilateral",
                    )
                },
            ),
            "target": Regime(
                functions={"utility": _target},
                actions={"choice": DiscreteGrid(category_class=_Binary)},
                shocks=("innovation",),
            ),
            "source_done": Regime(functions={"utility": _zero}),
            "target_done": Regime(functions={"utility": _zero}),
        },
        shocks={
            "source_innovation": ShockGroup(
                shocks={"source_signal": Normal(mu=0, sigma=1)},
                nodes=Product(source_signal=GaussHermite(3)),
            ),
            "innovation": ShockGroup(
                shocks={"offer": Discrete(categories=_Binary, probs=(0.5, 0.5))}
            ),
        },
        n_periods=3,
        regime_id_class=_DatedOverrideRegimes,
        initial_nodes=InitialNodes(by_period={0: "start", 1: "source"}),
        edges={
            "start": {"source": Periods(values=(0,))},
            "source": {"source_done": Periods(values=(1,))},
            "target": {"target_done": Periods(values=(1,))},
        },
    )


def _dated_override_initial() -> dict[str, jax.Array]:
    return {
        "period": jnp.asarray([1, 0, 1], dtype=jnp.int32),
        "regime_id": jnp.asarray([1, 0, 1], dtype=jnp.int32),
    }


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_target_only_history_covers_incoming_participating_period() -> None:
    """An incoming-only final actor at period one needs that dated input cell."""
    model = _dated_override_model()
    with pytest.raises(InvalidSimulationInputError, match=r"offer.*cover period 1"):
        model.simulate(
            params={"discount_factor": 0.5},
            initial_conditions=_dated_override_initial(),
            shock_paths={"offer": BaseInnovations(values=jnp.asarray([0.25]))},
            seed=17,
            log_level="off",
            runtime_checks=False,
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("physical", [False, True])
def test_target_only_complete_history_keeps_original_rows(*, physical: bool) -> None:
    """Staggered original rows reach the incoming Part actor and its own edge."""
    model = _dated_override_model()
    values = (
        PhysicalValues(values=jnp.asarray([[0, 0], [0, 1], [1, 0]], dtype=jnp.int32))
        if physical
        else BaseInnovations(values=jnp.asarray([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1]]))
    )
    frame = model.simulate(
        params={"discount_factor": 0.5},
        initial_conditions=_dated_override_initial(),
        shock_paths={"offer": values},
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    acting = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(acting["regime_name"], "target")
    np.testing.assert_array_equal(acting["offer"], [0, 1, 0])
    np.testing.assert_array_equal(acting["choice"], [0, 1, 0])
    assert_array_almost_equal(
        acting["value"], [-0.25, 1.75, -0.25], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(
        frame.loc[frame["period"] == 2, "regime_name"], "target_done"
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_unread_target_history_does_not_replace_fresh_target_draws() -> None:
    """A source-only prefix leaves the later target's named random address intact."""
    model = _dated_override_model()
    frames = [
        model.simulate(
            params={"discount_factor": 0.5},
            initial_conditions=_dated_override_initial(),
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
    assert_frame_equal(
        frames[0].loc[frames[0]["period"] == 1],
        frames[1].loc[frames[1]["period"] == 1],
    )


def _correlated_override_utility(
    *, first_signal: FloatND, second_signal: FloatND, choice: ScalarInt
) -> FloatND:
    return first_signal + second_signal + choice * (first_signal - 3)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("complete", [False, True])
def test_target_only_correlated_physical_frame_uses_mapped_producer(
    *, complete: bool
) -> None:
    """A complete low physical frame reveals after exit; partial frames refuse."""
    model = Model(
        regimes={
            "source": Regime(
                functions={"utility": _stay},
                exits={
                    "leave": Exit(
                        targets={"self": ExitTarget(regime="target", projection={})},
                        rule="unilateral",
                    )
                },
            ),
            "target": Regime(
                functions={"utility": _correlated_override_utility},
                actions={"choice": DiscreteGrid(category_class=_Binary)},
                shocks=("signals",),
            ),
        },
        shocks={
            "signals": ShockGroup(
                shocks={
                    "first_signal": Normal(mu=2, sigma=2),
                    "second_signal": Normal(mu=4, sigma=2),
                },
                correlation=((1.0, 0.5), (0.5, 1.0)),
                nodes=Product(
                    first_signal=GaussHermite(3), second_signal=GaussHermite(3)
                ),
            )
        },
        n_periods=1,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={},
    )
    paths = {
        "first_signal": PhysicalValues(values=jnp.asarray([-4.0])),
        **(
            {"second_signal": PhysicalValues(values=jnp.asarray([-4.0]))}
            if complete
            else {}
        ),
    }
    if not complete:
        with pytest.raises(
            ShockDeclarationError, match=r"complete physical|invalid or unresolved"
        ):
            model.simulate(
                params={},
                initial_conditions={
                    "period": jnp.asarray([0]),
                    "regime_id": jnp.asarray([0]),
                },
                shock_paths=paths,
                seed=17,
                log_level="off",
                runtime_checks=False,
            )
        return
    frame = model.simulate(
        params={},
        initial_conditions={"period": jnp.asarray([0]), "regime_id": jnp.asarray([0])},
        shock_paths=paths,
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame["regime_name"], "target")
    np.testing.assert_array_equal(frame["first_signal"], -4.0)
    np.testing.assert_array_equal(frame["second_signal"], -4.0)
    np.testing.assert_array_equal(frame["choice"], 0)
    assert_array_almost_equal(frame["value"], -8.0, decimal=DECIMAL_PRECISION)


def _shared_source_flow(shared_signal: FloatND) -> FloatND:
    return 0.5 + 0.0 * shared_signal


def _shared_target_flow(shared_signal: FloatND) -> FloatND:
    return shared_signal


def _mapped_location(location: FloatND) -> FloatND:
    return location + 10


def _location_mean(location: FloatND) -> FloatND:
    return location


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_shared_physical_override_transports_recovered_base_to_changed_law() -> None:
    """Source physical two at mean one becomes target physical twelve at mean eleven."""
    model = Model(
        regimes={
            "source": Regime(
                states={"location": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _shared_source_flow},
                shocks=("shared",),
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target",
                                projection={"location": _mapped_location},
                            )
                        },
                        rule="unilateral",
                    )
                },
            ),
            "target": Regime(
                states={"location": LinSpacedGrid(start=10, stop=11, n_points=2)},
                functions={"utility": _shared_target_flow},
                shocks=("shared",),
            ),
        },
        shocks={
            "shared": ShockGroup(
                shocks={"shared_signal": Normal(mu=_location_mean, sigma=1)},
                nodes=Product(shared_signal=GaussHermite(3)),
            )
        },
        n_periods=1,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={},
    )
    frame = model.simulate(
        params={},
        initial_conditions={
            "period": jnp.asarray([0]),
            "regime_id": jnp.asarray([0]),
            "location": jnp.asarray([1.0]),
            "shared_signal": PhysicalValues(values=jnp.asarray([2.0])),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame["regime_name"], "target")
    np.testing.assert_array_equal(frame["location"], 11.0)
    assert_array_almost_equal(frame["shared_signal"], 12.0, decimal=DECIMAL_PRECISION)
    assert_array_almost_equal(frame["value"], 12.0, decimal=DECIMAL_PRECISION)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("enable_jit", [False, True])
def test_observed_target_choice_uses_same_original_innovation(
    *, enable_jit: bool
) -> None:
    """Observation changes the choice order, while original row addresses stay put."""
    model = _hidden_offer_model(observed=True, enable_jit=enable_jit)
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        solution.values[0]["source"], 1.125, decimal=DECIMAL_PRECISION
    )
    frame = (
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "period": jnp.zeros(12, dtype=jnp.int32),
                "regime_id": jnp.zeros(12, dtype=jnp.int32),
            },
            seed=17,
            log_level="off",
            runtime_checks=False,
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values("subject_id")
    )
    keys = draw_window_keys(
        key=jax.random.key(17),
        address_words=np.asarray(
            RandomAddress(kind="shock", name="offer").words(), dtype=np.uint32
        ),
        subject_start=encode_subject_row(row=0),
        last_real_subject=encode_subject_row(row=11),
        subject_count=12,
        period=0,
    )
    expected_offer = np.asarray(jax.vmap(jax.random.uniform)(keys)) >= 0.5
    np.testing.assert_array_equal(
        frame["regime_name"], np.where(expected_offer, "target", "source")
    )
    assert_array_almost_equal(
        frame["value"], np.where(expected_offer, 1.75, 0.5), decimal=DECIMAL_PRECISION
    )
    movers = frame["regime_name"] == "target"
    np.testing.assert_array_equal(frame.loc[movers, "offer"], 1)
    np.testing.assert_array_equal(frame.loc[movers, "choice"], 1)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_exit_replay_uses_compiled_programs_across_original_row_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual outer chunks keep the same original-row observations and actions."""
    compiled = _capture_compiled_dispatches(monkeypatch=monkeypatch)
    seen = set()
    native_dispatch = SimulationRuntime.dispatch

    def observe_native(
        self: SimulationRuntime, **call: Unpack[RuntimeDispatch]
    ) -> PytreeValue:
        if call["program"].name in {
            "simulate_exit_full",
            "simulate_exit_participating",
        }:
            seen.add((call["program"].name, call["n_subjects"]))
        return native_dispatch(self, **call)

    monkeypatch.setattr(SimulationRuntime, "dispatch", observe_native)
    frames = []
    for width in (3, 12):
        model = _hidden_offer_model(
            observed=True,
            enable_jit=True,
            execution_config=ExecutionConfig(
                device_memory_bytes=None, axis_widths={"subject": width}
            ),
        )
        solution = model.solve(params={}, log_level="off", runtime_checks=False)
        frames.append(
            model.simulate(
                params={},
                solution=solution,
                initial_conditions={
                    "period": jnp.zeros(12, dtype=jnp.int32),
                    "regime_id": jnp.zeros(12, dtype=jnp.int32),
                },
                seed=17,
                log_level="off",
                runtime_checks=False,
            )
            .to_dataframe(use_categorical_codes=True)
            .sort_values("subject_id")
            .reset_index(drop=True)
        )
    assert compiled
    assert {
        (name, width)
        for name in ("simulate_exit_full", "simulate_exit_participating")
        for width in (3, 12)
    } <= seen
    assert_frame_equal(frames[0], frames[1])


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_exit_chunk_profile_inventories_native_queries_and_retained_frames() -> None:
    """Outer admission includes the actual once-only route, not just final actions."""
    model = _hidden_offer_model(observed=True, enable_jit=True)
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    regimes = model._runtime_regimes_for_shape(compile_batch_size=3)
    runtime = regimes["source"].simulation.programs.executor
    assert isinstance(runtime, SimulationRuntime)
    flat_params = model._process_params({})
    spaces = MappingProxyType(
        {
            name: regime.solution.state_action_space(
                regime_params=regime_kernel_params(flat_params, regime_name=name)
            )
            for name, regime in regimes.items()
        }
    )
    profile = profile_simulation_chunk(
        runtime=runtime,
        regimes=regimes,
        flat_params=flat_params,
        base_spaces=spaces,
        values=solution.values,
        ages=model._time,
        initial_conditions={
            "age": jnp.zeros(3, dtype=jnp.int32),
            "regime_id": jnp.zeros(3, dtype=jnp.int32),
        },
        regime_names_to_ids=model.regime_names_to_ids,
        n_subjects=3,
        population=3,
        original_population=3,
        widths={"subject": 3},
        independent_taste=False,
        log_level="off",
    )
    assert {
        "core:exit_full",
        "core:exit_participating",
        "route_original_subjects",
        "_original_bases",
        "_answer_status",
        "_final_decision",
    } <= {stage.name for stage in profile.stages}


def test_original_innovation_keys_remain_charged_until_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native output must not discard the still-needed original key bank owner."""
    banks: list[PytreeValue] = []
    checkpoints: list[str] = []
    native_dispatch = SimulationRuntime.dispatch
    host_dispatch = ProfiledSimulationOperations.dispatch

    def observe_native(
        self: SimulationRuntime, **call: Unpack[RuntimeDispatch]
    ) -> PytreeValue:
        if call["program"].name == "simulate_exit_full":
            bank = call["arguments"]["keys"]
            assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(bank))
            banks.append(cast("PytreeValue", bank))
        return native_dispatch(self, **call)

    def observe_host(
        self: ProfiledSimulationOperations, **call: Unpack[HostDispatch]
    ) -> PytreeValue:
        if call["function"] is route_original_subjects:
            assert banks
            missing = resident_bytes_by_device(
                live=measure_buffer_footprint(tree=tuple(banks)),
                arguments=call["live_footprint"](),
                devices=call["budget_devices"],
            )
            assert all(size == 0 for size in missing.values()), (
                f"Original innovation key bank omitted at routing: {dict(missing)}"
            )
            checkpoints.append("route")
        return host_dispatch(self, **call)

    model = _hidden_offer_model(observed=False, enable_jit=True)
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    with monkeypatch.context() as capture:
        capture.setattr(SimulationRuntime, "dispatch", observe_native)
        capture.setattr(ProfiledSimulationOperations, "dispatch", observe_host)
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "period": jnp.zeros(3, dtype=jnp.int32),
                "regime_id": jnp.zeros(3, dtype=jnp.int32),
            },
            seed=17,
            log_level="off",
            runtime_checks=False,
        )
    assert checkpoints


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("newlywed", [False, True])
def test_simulated_formation_uses_mapped_lag_and_fresh_consent(
    *, newlywed: bool
) -> None:
    """The same mapped household policy prices formation and acts after routing."""
    model = _household(newlywed=newlywed, enable_jit=True, source_only=True)
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.zeros(2, dtype=jnp.int32),
            "wealth": jnp.asarray([0.25, 0.75]),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(
        frame["regime_name"], "couple" if newlywed else "single_f"
    )
    if newlywed:
        assert_array_almost_equal(
            frame[["value_f", "value_m"]], 6.0, decimal=DECIMAL_PRECISION
        )
        np.testing.assert_array_equal(
            frame["own_stakeholder"],
            model._regimes["couple"].stakeholder_names_to_ids["f"],
        )
        np.testing.assert_array_equal(frame["married_last_period"], 0)
        np.testing.assert_array_equal(frame["action"], 1)
    else:
        assert_array_almost_equal(frame["value"], 5.0, decimal=DECIMAL_PRECISION)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_simulated_formation_keeps_nested_mapped_lagged_outside_states() -> None:
    """The newborn household's schooling outside options block formation."""
    model = _household(
        newlywed=True, enable_jit=True, source_only=True, outside_lag=True
    )
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.zeros(2, dtype=jnp.int32),
            "wealth": jnp.asarray([0.25, 0.75]),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    # Nested lag0 maps both single benchmarks to schooling7; no common tuple
    # clears both thresholds, so the source keeps its actual current wealth.
    np.testing.assert_array_equal(frame["regime_name"], "single_f")
    assert_array_almost_equal(frame["value"], [5.5, 6.5], decimal=DECIMAL_PRECISION)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_hidden_destination_taste_acts_after_custom_exit_without_repricing() -> None:
    """Source prices log4, commits, then target draws its own unshocked action."""
    model = _rule_model(rule=_StrictRule(), target_taste=True)
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        solution.values[0]["single_f"], np.log(4.0), decimal=DECIMAL_PRECISION
    )
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(12, dtype=jnp.int32),
            "regime_id": jnp.zeros(12, dtype=jnp.int32),
            "wealth": jnp.zeros(12),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    expected = _addressed_taste_actions(
        regime="single_m", values=jnp.asarray([0.0, jnp.log(3.0)])
    )
    np.testing.assert_array_equal(frame["regime_name"], "single_m")
    np.testing.assert_array_equal(frame["action"], expected)
    assert_array_almost_equal(
        frame["value"], np.log(3.0) * expected, decimal=DECIMAL_PRECISION
    )


def _choice_value(*, choice: IntND) -> FloatND:
    return 1.0 * choice


def _double_choice(*, choice: IntND) -> FloatND:
    return 2.0 * choice


def _three() -> float:
    return 3.0


def _zero() -> float:
    return 0.0


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("fallback", [False, True])
def test_source_taste_keeps_dissolution_as_fallback_in_public_replay(
    *, fallback: bool
) -> None:
    """A surviving kept stay cannot acquire an optional noisy exit."""
    model = Model(
        regimes={
            "source": Regime(
                actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _choice_value},
                taste_shocks=ExtremeValueTasteShocks(),
                exits={
                    "leave": Exit(
                        targets={"self": ExitTarget(regime="target", projection={})},
                        rule="unilateral",
                    )
                },
            ),
            "target": Regime(functions={"utility": _two if fallback else _solve_flow}),
        },
        n_periods=1,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={},
        fixed_params={"source": {"taste_shocks": {"scale": 1.0}}},
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    expected = 2.0 if fallback else 1.0
    assert_array_almost_equal(
        solution.values[0]["source"], expected, decimal=DECIMAL_PRECISION
    )
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(12, dtype=jnp.int32),
            "regime_id": jnp.zeros(12, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(
        frame["regime_name"], "target" if fallback else "source"
    )
    assert_array_almost_equal(frame["value"], expected, decimal=DECIMAL_PRECISION)
    if not fallback:
        np.testing.assert_array_equal(frame["choice"], 1)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_source_taste_freezes_one_consent_choice_before_target_action() -> None:
    """Eligible formation joins one source draw; movers report target values."""
    model = Model(
        regimes={
            "source": Regime(
                actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _double_choice},
                taste_shocks=ExtremeValueTasteShocks(),
                exits={
                    "marry": Exit(
                        targets={
                            "self": ExitTarget(regime="target", role="f", projection={})
                        },
                        rule=Consent(
                            counterparty=Counterparty(
                                regime="counterparty", role="m", projection={}
                            )
                        ),
                    )
                },
            ),
            "counterparty": Regime(functions={"utility": _zero}),
            "target": Regime(
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _three, "m": _solve_flow}
                    )
                }
            ),
        },
        n_periods=1,
        regime_id_class=_FormationRegimes,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={},
        fixed_params={"source": {"taste_shocks": {"scale": 1.0}}},
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        solution.values[0]["source"],
        np.log(1.0 + np.exp(2.0) + np.exp(3.0)),
        decimal=DECIMAL_PRECISION,
    )
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(12, dtype=jnp.int32),
            "regime_id": jnp.zeros(12, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    selected = _addressed_taste_actions(
        regime="source", values=jnp.asarray([0.0, 2.0, 3.0])
    )
    moved = selected == 2
    np.testing.assert_array_equal(
        frame["regime_name"], np.where(moved, "target", "source")
    )
    np.testing.assert_array_equal(frame.loc[~moved, "choice"], selected[~moved])
    assert_array_almost_equal(
        frame.loc[~moved, "value"],
        np.asarray([0.0, 2.0])[selected[~moved]],
        decimal=DECIMAL_PRECISION,
    )
    assert_array_almost_equal(
        frame.loc[moved, "value_f"], 3.0, decimal=DECIMAL_PRECISION
    )
    assert_array_almost_equal(
        frame.loc[moved, "value_m"], 1.0, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(
        frame.loc[moved, "own_stakeholder"],
        model._regimes["target"].stakeholder_names_to_ids["f"],
    )


def _x_flow(x: FloatND) -> FloatND:
    return x


def _y_flow(y: FloatND) -> FloatND:
    return y


def _into_target(x: FloatND) -> FloatND:
    return 10.0 + 0.5 * x


def _into_source(y: FloatND) -> FloatND:
    return 20.0 + 0.5 * y


def _next_x(x: FloatND) -> FloatND:
    return jnp.minimum(x + 1.0, 100.0)


def _next_y(y: FloatND) -> FloatND:
    return jnp.minimum(y + 2.0, 100.0)


def _first_period(period: ScalarInt) -> BoolND:
    return period == 0


def _return_fee() -> float:
    return -0.5


class _RouteOnce:
    """Choose an available current move without recursive outside-policy pricing."""

    reads_outside_values = False

    def __call__(self, *, period: ScalarInt, alternative_ids: IntND) -> RuleResult:
        return RuleResult(
            action=jnp.where(period == 0, alternative_ids[-1], alternative_ids[0])
        )


def _opposing_model(*, rule: ExitRule, reverse: bool = False) -> Model:
    grid = LinSpacedGrid(start=0, stop=100, n_points=3)
    regimes = {
        "source": Regime(
            functions={"utility": _x_flow},
            states={"x": grid},
            state_transitions={"x": _next_x},
            exits={
                "outward": Exit(
                    targets={
                        "self": ExitTarget(
                            regime="target",
                            projection={"y": _into_target},
                            utility_adjustment=_adjustment,
                        )
                    },
                    rule=rule,
                    available=_first_period,
                )
            },
        ),
        "target": Regime(
            functions={"utility": _y_flow},
            states={"y": grid},
            state_transitions={"y": _next_y},
            exits={
                "inward": Exit(
                    targets={
                        "self": ExitTarget(
                            regime="source",
                            projection={"x": _into_source},
                            utility_adjustment=_return_fee,
                        )
                    },
                    rule=rule,
                    available=_first_period,
                )
            },
        ),
        "source_done": Regime(functions={"utility": _x_flow}, states={"x": grid}),
        "target_done": Regime(functions={"utility": _y_flow}, states={"y": grid}),
    }
    return Model(
        regimes=dict(reversed(tuple(regimes.items()))) if reverse else regimes,
        n_periods=2,
        regime_id_class=_OpposingRegimes,
        initial_nodes=InitialNodes(by_period={0: ("source", "target")}),
        edges={
            "source": {"source_done": Periods(values=(0,))},
            "target": {"target_done": Periods(values=(0,))},
        },
    )


def test_reciprocal_unilateral_exits_refuse_the_public_participating_cycle() -> None:
    """Reciprocal outside pricing cannot recursively price itself."""
    with pytest.raises(ValueError, match="Policy query cycle"):
        _opposing_model(rule="unilateral")


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("reverse", [False, True])
def test_opposing_public_exits_advance_only_the_final_actor(*, reverse: bool) -> None:
    """Original occupants move once, pay once, then use final scheduled laws."""
    model = _opposing_model(rule=_RouteOnce(), reverse=reverse)
    params = {"discount_factor": 0.0}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.asarray([0, 1], dtype=jnp.int32),
            "x": jnp.asarray([2.0, 0.0]),
            "y": jnp.asarray([0.0, 6.0]),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    entry = frame.query("period == 0").sort_values("subject_id")
    final = frame.query("period == 1").sort_values("subject_id")
    np.testing.assert_array_equal(entry["regime_name"], ["target", "source"])
    np.testing.assert_array_equal(final["regime_name"], ["target_done", "source_done"])
    assert_array_almost_equal(entry["y"].iloc[0], 11.0, decimal=DECIMAL_PRECISION)
    assert_array_almost_equal(entry["x"].iloc[1], 23.0, decimal=DECIMAL_PRECISION)
    assert_array_almost_equal(final["y"].iloc[0], 13.0, decimal=DECIMAL_PRECISION)
    assert_array_almost_equal(final["x"].iloc[1], 24.0, decimal=DECIMAL_PRECISION)
    assert_array_almost_equal(entry["value"], [10.75, 22.5], decimal=DECIMAL_PRECISION)
    assert_array_almost_equal(final["value"], [13.0, 24.0], decimal=DECIMAL_PRECISION)


class _StayWithoutOutside:
    """The target's custom Full rule does not constrain its Participating policy."""

    reads_outside_values = False

    def __call__(self, *, alternative_ids: jax.Array) -> RuleResult:
        return RuleResult(action=alternative_ids[0])


def _two() -> float:
    return 2.0


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_incoming_participating_actor_does_not_create_an_original_full_query() -> None:
    """A destination's unread custom outside option creates no extra query context."""
    model = Model(
        regimes={
            "source": Regime(
                functions={"utility": _solve_flow},
                exits={
                    "leave": Exit(
                        targets={"self": ExitTarget(regime="target", projection={})},
                        rule="unilateral",
                    )
                },
            ),
            "target": Regime(
                functions={"utility": _two},
                exits={
                    "unread": Exit(
                        targets={"self": ExitTarget(regime="unread", projection={})},
                        rule=_StayWithoutOutside(),
                    )
                },
            ),
            "unread": Regime(functions={"utility": _two}),
        },
        n_periods=1,
        regime_id_class=_ThreeRegimes,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={},
    )
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    assert set(solution.values[0]) == {"source"}
    assert not model._regimes["unread"].simulation.Q_and_F
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(1, dtype=jnp.int32),
            "regime_id": jnp.zeros(1, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame["regime_name"], "target")
    assert_array_almost_equal(frame["value"], 2.0, decimal=DECIMAL_PRECISION)


def _solve_flow() -> float:
    return 1.0


def _simulate_flow() -> float:
    return 2.0


def _helper_flow(*, benefit: float) -> float:
    return benefit


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_public_exit_replay_uses_simulation_phase_helper_closure() -> None:
    """A target's actual helper yields3 while its solve-period price was2."""
    model = Model(
        regimes={
            "source": Regime(
                functions={"utility": _solve_flow},
                exits={
                    "leave": Exit(
                        targets={"self": ExitTarget(regime="target", projection={})},
                        rule="unilateral",
                    )
                },
            ),
            "target": Regime(
                functions={
                    "utility": _helper_flow,
                    "benefit": Phased(solve=_two, simulate=_three),
                }
            ),
        },
        n_periods=1,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={},
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        solution.values[0]["source"], 2.0, decimal=DECIMAL_PRECISION
    )
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(1, dtype=jnp.int32),
            "regime_id": jnp.zeros(1, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame["regime_name"], "target")
    assert_array_almost_equal(frame["value"], 3.0, decimal=DECIMAL_PRECISION)


def test_forward_query_context_retains_actual_simulation_q_and_helpers() -> None:
    """Simulation query contexts carry their own canonical phase closures."""
    with patch.object(
        processing,
        "build_simulation_programs",
        wraps=processing.build_simulation_programs,
    ) as build:
        Model(
            regimes={
                "source": Regime(
                    functions={
                        "utility": Phased(solve=_solve_flow, simulate=_simulate_flow)
                    }
                )
            },
            n_periods=1,
            regime_id_class=_OnlySource,
            initial_nodes=InitialNodes(by_period={0: "source"}),
            edges={},
        )
    context = build.call_args.kwargs["context"]
    assert isinstance(context, SolverBuildContext)
    assert context.Q_and_F_functions is build.call_args.kwargs["Q_and_F_functions"]
    assert_array_almost_equal(
        context.functions["utility"](), 2.0, decimal=DECIMAL_PRECISION
    )
