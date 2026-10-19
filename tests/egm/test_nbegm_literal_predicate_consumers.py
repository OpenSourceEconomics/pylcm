"""Literal NB predicates govern current nodes and actual child choice reads."""

import importlib
import logging
from pathlib import Path
from typing import Literal, Protocol, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import continuation as continuation_module
from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import QueryNodeKind
from _lcm.solution.feasibility import (
    BoundedResult,
    FeasibilityKernelOutput,
    validate_query_input_status,
)
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from _lcm.solution.period_replay import _load_capture_payload
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    post_decision_lower_bound,
    ref,
)
from lcm.solver_api import KernelOutput
from lcm.solvers import NBEGM
from lcm.transition import fixed_transition
from lcm.typing import (
    Age,
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_passive_consumers import (
    Job,
    RegimeId,
    _AggregateInputs,
    _ChoiceInputs,
    _inverse,
    _parent_resources,
    _savings,
)
from tests.solution._nbegm_direct_oracle import OracleContext

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def test_literal_nb_liquid_predicate_beside_discrete_choice() -> None:
    """Admit a declarative liquid predicate beside an ordinary discrete envelope."""
    _assert_literal_control()
    model, params = _create_model(predicate="liquid")
    result = model.solve(params=params, log_level="off")
    np.testing.assert_array_almost_equal(
        result.values[1]["child"][0],
        np.full((2, 3), 10.0),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.values[0]["parent"][1],
        np.full((2, 3), 13.0),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("predicate", ["threshold", "island"])
def test_installed_literal_nb_predicate_precedes_choice_ranking(
    *,
    predicate: Literal["threshold", "island"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain conditional rows and rank their actual target feasibility."""
    _assert_literal_control()
    model, params = _create_model(predicate=predicate)
    kernel, context = _capture_nb_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    _assert_child_bank(model=model, context=context, predicate=predicate)
    assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
    output, conditional, aggregate = _observe_nb_consumer(
        kernel=kernel, context=context, monkeypatch=monkeypatch
    )
    selected = [
        item for item in conditional if np.array_equal(item[0][:3], [0.0, 0.5, 0.5])
    ]
    pairs = [
        pair for facts, pair in aggregate if np.array_equal(facts, [0.0, 0.5, 0.5])
    ]
    assert selected, "The installed NB reader did not reach the actual child target."
    assert pairs, "The installed parent did not aggregate the actual NB target."
    admitted = predicate == "island"
    for facts, values, marginals, savings in selected:
        assert facts.dtype == np.dtype(jnp.asarray(0.0).dtype)
        np.testing.assert_array_equal(
            facts,
            np.asarray(
                [0.0, 0.5, 0.5, 1.0, 41.0, 0.0, 1.0, 0.0 if admitted else 0.6, 4.0],
                dtype=facts.dtype,
            ),
        )
        np.testing.assert_array_equal(savings, [0.0, 0.0])
        np.testing.assert_array_almost_equal(
            values,
            [0.0, 10.0 if admitted else -np.inf],
            decimal=DECIMAL_PRECISION,
        )
        np.testing.assert_array_equal(marginals, [4.0, 4.0 if admitted else 0.0])
    for pair in pairs:
        np.testing.assert_array_almost_equal(
            pair, [10.0 if admitted else 0.0, 4.0], decimal=DECIMAL_PRECISION
        )
    np.testing.assert_array_almost_equal(
        output.value[1],
        np.full((2, 3), 10.0 if admitted else 3.0),
        decimal=DECIMAL_PRECISION,
    )
    input_authorities = model._regimes["parent"].solution.query_input_authorities
    assert input_authorities is not None
    assert output.query_status is not None
    assert any(slot[:2] == ("child", 1) for slot in input_authorities[0])
    for status in output.query_status:
        np.testing.assert_array_equal(jnp.stack(status), [False, False, False])
    validate_query_input_status(
        statuses=output.query_status, authorities=input_authorities[0]
    )
    for status in output.bound_status:
        np.testing.assert_array_equal(jnp.stack(status), [False, False])


def _assert_child_bank(
    *,
    model: Model,
    context: OracleContext,
    predicate: Literal["threshold", "island"],
) -> None:
    """Check the installed child's unmasked bank and masked published node values."""
    carry = context["next_regime_to_continuation"]["child"]
    assert isinstance(carry, EGMCarry)
    data = carry.query_data_for_transport()
    assert data is not None
    authority = model._regimes["child"].solution.query_authority
    assert authority is not None
    assert authority.state_names == ("z", "w")
    assert authority.action_names == ("job",)
    assert authority.target_periods == (1,)
    assert authority.coordinate_kind == "liquid"
    assert authority.corner_coordinate_kind == "compact_interval"
    assert data.node_kind.shape[:3] == (2, 3, 2)
    assert np.all(
        np.any(
            np.asarray(data.node_kind)[:, :, 1] == QueryNodeKind.FINITE,
            axis=-1,
        )
    )
    np.testing.assert_array_equal(data.floor_used, 0.0)
    assert data.corner_continuation is not None
    np.testing.assert_array_equal(data.corner_continuation, 0.0)
    expected_nodes = (
        np.zeros((2, 3))
        if predicate == "island"
        else np.broadcast_to(np.asarray([0.0, 10.0])[:, None], (2, 3))
    )
    np.testing.assert_array_almost_equal(
        context["next_regime_to_V_arr"]["child"][0],
        expected_nodes,
        decimal=DECIMAL_PRECISION,
    )
    if predicate == "threshold":
        cutoff = jnp.asarray(context["flat_params"]["child"]["eligibility__cutoff"])
        np.testing.assert_array_equal(cutoff, jnp.asarray(0.6, dtype=cutoff.dtype))


def _assert_literal_control() -> None:
    """Check ordinary public literal values before requesting predicate behavior."""
    model, params = _create_model(predicate="none")
    result = model.solve(params=params, log_level="off")
    np.testing.assert_array_almost_equal(
        result.values[1]["child"][0],
        np.full((2, 3), 10.0),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.values[0]["parent"][1],
        np.full((2, 3), 13.0),
        decimal=DECIMAL_PRECISION,
    )


def _create_model(
    *, predicate: Literal["none", "liquid", "threshold", "island"]
) -> tuple[Model, UserParams]:
    """Use the same literal physical grids and two-by-three passive stencil."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    passive = {
        "z": IrregSpacedGrid(points=np.array([0.0, 1.0])),
        "w": IrregSpacedGrid(points=np.array([0.0, 1.0, 2.0])),
    }
    child = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=np.array([0.0, 0.5, 1.5])), **passive},
        actions={
            "job": DiscreteGrid(category_class=Job),
            "consumption": IrregSpacedGrid(
                points=np.array([0.025, 0.125, 0.5, 1.0, 4.0])
            ),
        },
        functions={
            "utility": _child_utility,
            "resources": _resources,
            "savings": _savings,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=0.0),
            **(
                {"eligibility": ref("liquid") >= ref("cutoff")}
                if predicate == "liquid"
                else {}
            ),
            **({"eligibility": _threshold} if predicate == "threshold" else {}),
            **({"eligibility": _island} if predicate == "island" else {}),
        },
        state_transitions={
            "liquid": _next_liquid,
            "z": fixed_transition("z"),
            "w": fixed_transition("w"),
        },
        solver=NBEGM(savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 1.5]))),
    )
    parent = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=np.array([0.375, 0.5])), **passive},
        actions={
            "route": DiscreteGrid(category_class=Job),
            "consumption": IrregSpacedGrid(points=np.array([0.025, 0.125, 0.25])),
        },
        functions={
            "utility": _parent_utility,
            "resources": _parent_resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse,
        },
        constraints={"borrowing": post_decision_lower_bound(margin=margin, lower=0.25)},
        state_transitions={
            "liquid": _child_liquid,
            "z": _child_z,
            "w": _child_w,
        },
        solver=NBEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.25, 0.375, 0.5])),
        ),
    )
    model = Model(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={
                    "liquid": IrregSpacedGrid(points=np.array([0.0, 16.0])),
                    **passive,
                },
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=40, inclusive_stop=42, step="Y"),
        edges={"parent": {"child": 40}, "child": {"done": 41}},
        initial_nodes={40: "parent"},
        regime_id_class=RegimeId,
    )
    return model, {
        "parent": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "next_z": {"query_point": 0.5},
        },
        "child": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "resources": {"shift": 0.0},
            **(
                {"eligibility": {"cutoff": 0.0 if predicate == "liquid" else 0.6}}
                if predicate in ("liquid", "threshold")
                else {}
            ),
        },
        "done": {},
    }


def _child_utility(*, consumption: ContinuousAction, job: DiscreteAction) -> FloatND:
    return jnp.log(consumption / 2.0) + 10.0 * job


def _parent_utility(
    *, consumption: ContinuousAction, route: DiscreteAction, z: ContinuousState
) -> FloatND:
    return jnp.log(consumption / 0.25) + 3.0 * route + 0.0 * z


def _resources(
    *, liquid: ContinuousState, shift: float, period: ScalarInt, age: Age
) -> FloatND:
    return 2.0 * liquid + 2.0 + shift + (period - 1.0) + (age - 41.0)


def _threshold(*, job: DiscreteAction, z: ContinuousState, cutoff: float) -> BoolND:
    return (job == 0) | (z >= cutoff)


def _island(*, job: DiscreteAction, z: ContinuousState) -> BoolND:
    return (job == 0) | ((z > 0.4) & (z < 0.6))


def _next_liquid(savings: FloatND) -> ContinuousState:
    return savings


def _child_liquid(savings: FloatND) -> ContinuousState:
    return 4.0 * savings - 1.0


def _child_z(
    *, route: DiscreteAction, query_point: float | ScalarFloat
) -> ContinuousState:
    return jnp.where(route == 0, query_point, 0.0)


def _child_w(w: ContinuousState) -> ContinuousState:
    return 0.5 + 0.5 * w


def _terminal(
    *, liquid: ContinuousState, z: ContinuousState, w: ContinuousState
) -> FloatND:
    return liquid / 2.0 + 0.0 * z + 0.0 * w


def _capture_nb_with_checks_off(
    *,
    model: Model,
    params: UserParams,
    regime_name: RegimeName,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_RideAlongNBEGMPeriodKernel, OracleContext]:
    """Capture the real rolling NB inputs with optional checks disabled."""
    with monkeypatch.context() as patch:
        patch.setenv("LCM_CAPTURE_PERIOD", f"{regime_name}@0")
        patch.setenv("LCM_CAPTURE_DIR", str(directory))
        model.solve(params=params, log_level="off", runtime_checks=False)
    payload = _load_capture_payload(directory=directory / f"{regime_name}@0")
    inputs = payload["kernel_kwargs"]
    kernel = model._regimes[regime_name].solution.period_kernels[0]
    assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
    return kernel, {
        "state_action_space": inputs["state_action_space"],
        "next_regime_to_V_arr": inputs["next_regime_to_V_arr"],
        "next_regime_to_continuation": inputs["next_regime_to_continuation"],
        "flat_params": inputs["flat_params"],
        "period": inputs["period"],
        "ages": inputs["ages"],
    }


def _run_nb_installed(
    *, kernel: _RideAlongNBEGMPeriodKernel, context: OracleContext
) -> KernelOutput:
    """Run the installed NB main program on its real captured solve inputs."""
    return kernel(
        compiled_cores={"main": kernel.core_programs()["main"].function},
        **context,
        logger=logging.getLogger(__name__),
    )


class _QueryResult(Protocol):
    """Observe the installed reader's associated economic channels unchanged."""

    value: FloatND
    marginal: FloatND
    savings: FloatND


class _InstalledQuery(Protocol):
    """Type the installed reader's method without importing it before admission."""

    def read_choices(self, **kwargs: Unpack[_ChoiceInputs]) -> _QueryResult: ...


def _observe_nb_consumer(
    *,
    kernel: _RideAlongNBEGMPeriodKernel,
    context: OracleContext,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    FeasibilityKernelOutput,
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
]:
    """Observe real NB conditional tuples and the installed parent's aggregation."""
    module = importlib.import_module("_lcm.egm.nbegm_query")
    query_type = cast("type[_InstalledQuery]", module.NBEGMQuery)
    original_choices = query_type.read_choices
    original_aggregate = continuation_module._aggregate_child_choices
    conditional: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    aggregate: list[tuple[np.ndarray, np.ndarray]] = []

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record_choices(
        facts: np.ndarray,
        values: np.ndarray,
        marginals: np.ndarray,
        savings: np.ndarray,
    ) -> None:
        conditional.append((facts, values, marginals, savings))

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record_aggregate(facts: np.ndarray, pair: np.ndarray) -> None:
        aggregate.append((facts, pair))

    def observe_choices(
        self: _InstalledQuery, **kwargs: Unpack[_ChoiceInputs]
    ) -> _QueryResult:
        assert type(self) is query_type
        result = original_choices(self, **kwargs)
        assert kwargs["row_queries"].shape == (2, 3, 2)
        assert kwargs["row_gradients"].shape == (2, 3, 2)
        pool = kwargs["pool"]
        facts: list[ScalarFloat | ScalarInt | float | int] = []
        for name in (
            "liquid",
            "z",
            "w",
            "period",
            "age",
            "resources__shift",
            "koopmans_aggregator__discount_factor",
        ):
            numeric = pool[name]
            assert isinstance(numeric, jax.Array | float | int)
            facts.append(numeric)
        cutoff = pool.get("eligibility__cutoff", 0.0)
        assert isinstance(cutoff, jax.Array | float | int)
        facts.append(cutoff)
        facts.append(kwargs["euler_gradient"])
        jax.debug.callback(
            record_choices,
            jnp.stack(facts),
            result.value,
            result.marginal,
            result.savings,
        )
        return result

    def observe_aggregate(
        **kwargs: Unpack[_AggregateInputs],
    ) -> (
        tuple[ScalarFloat, ScalarFloat] | BoundedResult[tuple[ScalarFloat, ScalarFloat]]
    ):
        result = original_aggregate(**kwargs)
        if kwargs.get("query") is not None:
            assert type(kwargs["query"]) is query_type
            pool = kwargs.get("query_pool")
            assert pool is not None
            assert isinstance(result, BoundedResult)
            facts: list[ScalarFloat | ScalarInt | float | int] = []
            for name in ("liquid", "z", "w"):
                numeric = pool[name]
                assert isinstance(numeric, jax.Array | float | int)
                facts.append(numeric)
            jax.debug.callback(
                record_aggregate, jnp.stack(facts), jnp.stack(result.output)
            )
        return result

    monkeypatch.setattr(query_type, "read_choices", observe_choices)
    monkeypatch.setattr(
        continuation_module, "_aggregate_child_choices", observe_aggregate
    )
    output = _run_nb_installed(kernel=kernel, context=context)
    jax.block_until_ready(output.value)
    jax.effects_barrier()
    assert isinstance(output, FeasibilityKernelOutput)
    return output, conditional, aggregate
