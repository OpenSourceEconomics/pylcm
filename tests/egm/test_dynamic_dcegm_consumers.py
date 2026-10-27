"""Actual DC corner records and installed consumers preserve values and units."""

import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, NotRequired, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import ContinuationPayload
from _lcm.egm import continuation as continuation_module
from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import QueryNodeKind
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.solution.dcegm import _BoundDCEGM, _DCEGMPeriodKernel
from _lcm.solution.egm import _EGMPeriodKernel
from _lcm.solution.feasibility import (
    BoundedResult,
    FeasibilityKernelOutput,
    validate_query_input_status,
)
from _lcm.solution.period_replay import _load_capture_payload
from _lcm.typing import FlatRegimeParams
from lcm import (
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    post_decision_lower_bound,
)
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import EGM_CONTINUATION, KernelOutput
from lcm.solvers import MSSEnvelope
from lcm.typing import (
    Float1D,
    FloatND,
    IntND,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import (
    RegimeId,
    _lower,
    _period_lower,
    activation_params,
    build_activation_model,
)
from tests.egm.test_mixed_floor_baselines import RegimeId as IdentityRegimeId
from tests.egm.test_mixed_floor_baselines import _model as build_literal_mixed_model
from tests.solution._nbegm_direct_oracle import OracleContext

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _capture_with_checks_off(
    *,
    model: Model,
    params: UserParams,
    regime_name: RegimeName,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_DCEGMPeriodKernel | _EGMPeriodKernel, OracleContext]:
    """Capture the real rolling inputs from a solve with optional checks disabled."""
    with monkeypatch.context() as patch:
        patch.setenv("LCM_CAPTURE_PERIOD", f"{regime_name}@0")
        patch.setenv("LCM_CAPTURE_DIR", str(directory))
        model.solve(params=params, log_level="off", runtime_checks=False)
    payload = _load_capture_payload(directory=directory / f"{regime_name}@0")
    inputs = payload["kernel_kwargs"]
    kernel = model._regimes[regime_name].solution.period_kernels[0]
    assert isinstance(kernel, _DCEGMPeriodKernel | _EGMPeriodKernel)
    return kernel, {
        "state_action_space": inputs["state_action_space"],
        "next_regime_to_V_arr": inputs["next_regime_to_V_arr"],
        "next_regime_to_continuation": inputs["next_regime_to_continuation"],
        "flat_params": inputs["flat_params"],
        "period": inputs["period"],
        "ages": inputs["ages"],
    }


def _run_installed(
    *, kernel: _DCEGMPeriodKernel | _EGMPeriodKernel, context: OracleContext
) -> KernelOutput:
    return kernel(
        compiled_cores={"main": kernel.core_programs()["main"].function},
        **context,
        logger=logging.getLogger(__name__),
    )


@pytest.mark.parametrize(
    ("asset", "floor"),
    [(False, 0.5), (True, 0.0), (True, 0.5)],
    ids=("ordinary-half", "asset-zero", "asset-half"),
)
def test_installed_dc_producer_retains_exact_floor_continuation_and_direct_units(
    *,
    asset: bool,
    floor: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain computed W(b,x) and its native-resource direct derivative."""
    model = build_activation_model(floor=floor, literal=False, asset=asset)
    kernel, context = _capture_with_checks_off(
        model=model,
        params=activation_params(floor=floor, literal=False),
        regime_name="child",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    output = _run_installed(kernel=kernel, context=context)
    carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    data = carry.query_data_for_transport()
    assert data is not None
    authority = model._regimes["child"].solution.query_authority
    assert authority is not None
    assert authority.coordinate_kind == "resources"
    np.testing.assert_array_equal(data.floor_used, floor)
    if asset:
        assert authority.corner_coordinate_kind == "liquid_row"
        corner = data.liquid_corner_data
        assert corner is not None
        liquid_nodes = np.asarray([0.0, 0.125, 0.25, 0.5])
        np.testing.assert_array_almost_equal(
            corner.continuation[:, 0],
            np.log(3.0 * floor + 4.0 + liquid_nodes),
            decimal=DECIMAL_PRECISION,
        )
        np.testing.assert_array_almost_equal(
            corner.direct_marginal[:, 0],
            0.5 / (3.0 * floor + 4.0 + liquid_nodes),
            decimal=DECIMAL_PRECISION,
        )
        np.testing.assert_array_equal(
            corner.node_kind[:, 0], [QueryNodeKind.FINITE] * 4
        )
    else:
        assert authority.corner_coordinate_kind == "compact_interval"
        assert data.liquid_corner_data is None
        assert data.corner_continuation is not None
        np.testing.assert_array_almost_equal(
            data.corner_continuation, [np.log(5.5)], decimal=DECIMAL_PRECISION
        )
        np.testing.assert_array_equal(data.corner_kind, [QueryNodeKind.FINITE])


class _AggregateInputs(TypedDict):
    carry: EGMCarry
    prepared_search_grid: FloatND
    prepared_valid_length: IntND
    has_taste_shocks: bool
    child_index: tuple[ScalarInt, ...]
    child_passive_values: tuple[ScalarFloat, ...]
    child_passive_grids: tuple[Float1D, ...]
    row_queries: FloatND
    row_gradients: FloatND
    paired_marginal_read: NotRequired[bool]
    n_outer_candidates: NotRequired[int]


@pytest.mark.parametrize(
    ("asset", "floor", "expected_value", "expected_marginal"),
    [
        (False, 0.5, np.log(2.75), 20.0),
        (True, 0.0, np.log(4.0), 45.0 / 4.0),
        (True, 0.5, np.log(2.75), 230.0 / 11.0),
    ],
    ids=("ordinary-half", "asset-zero", "asset-half"),
)
@pytest.mark.parametrize("period_floor", [False, True])
def test_actual_dc_child_read_recovers_corner_and_composes_one_resource_chain(
    *,
    asset: bool,
    floor: float,
    expected_value: float,
    expected_marginal: float,
    period_floor: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover the corner's own pair at parent saving one fifth."""
    original = continuation_module._aggregate_child_choices
    observed: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record(
        query: np.ndarray,
        gradient: np.ndarray,
        value: np.ndarray,
        marginal: np.ndarray,
    ) -> None:
        observed.append((query, gradient, value, marginal))

    def observe(
        **kwargs: Unpack[_AggregateInputs],
    ) -> (
        tuple[ScalarFloat, ScalarFloat] | BoundedResult[tuple[ScalarFloat, ScalarFloat]]
    ):
        result = original(**kwargs)
        if kwargs["carry"].query_data_for_transport() is not None:
            pair = result.output if isinstance(result, BoundedResult) else result
            jax.debug.callback(
                record,
                kwargs["row_queries"],
                kwargs["row_gradients"],
                pair[0],
                pair[1],
            )
        return result

    monkeypatch.setattr(continuation_module, "_aggregate_child_choices", observe)
    model = build_activation_model(
        floor=floor,
        literal=False,
        asset=asset,
        parent=True,
        lower_function=_period_lower if period_floor else _lower,
    )
    authority = model._regimes["child"].solution.query_authority
    assert authority is not None
    assert "borrowing_lower__floor" in dict(authority.evaluator_inputs)["lower_bound"]
    params = activation_params(
        floor=floor - 0.5 if period_floor else floor, literal=False, parent=True
    )
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    jax.block_until_ready(solution.values[0]["parent"])
    jax.effects_barrier()
    assert observed, "The installed parent did not read its actual feature child."
    working_dtype = np.dtype(jnp.asarray(0.0).dtype)
    assert all(
        channel.shape == () and channel.dtype == working_dtype
        for item in observed
        for channel in item
    )
    query, gradient, value, marginal = min(observed, key=lambda item: item[0].item())
    np.testing.assert_array_equal(query, 1.0)
    np.testing.assert_array_equal(gradient, 10.0)
    np.testing.assert_array_almost_equal(
        value, expected_value, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        marginal, expected_marginal, decimal=DECIMAL_PRECISION
    )


def _identity_parent_model(*, literal: bool, dead_source: bool = False) -> Model:
    """Keep the admitted one-row mixed baseline and its child resources identity."""
    base = build_literal_mixed_model(plain_parent=True, floor=0.5)
    margin = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="wealth",
        post_decision_state="savings",
    )
    child = base.user_regimes["child"]
    child = replace(
        child,
        constraints={
            "borrowing": post_decision_lower_bound(
                margin=margin, lower=0.5 if literal else _lower
            )
        },
    )
    parent = base.user_regimes["parent"]
    if dead_source:
        parent = replace(
            parent, states={"wealth": IrregSpacedGrid(points=np.array([-0.125, 0.0]))}
        )
    return Model(
        regimes={"parent": parent, "child": child, "done": base.user_regimes["done"]},
        ages=base.ages,
        edges={"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "parent"},
        regime_id_class=IdentityRegimeId,
    )


@pytest.mark.parametrize("literal", [True, False])
def test_plain_parent_reads_identity_resource_dc_corner(*, literal: bool) -> None:
    """Consume a DC child at a represented corner node from a plain parent."""
    model = _identity_parent_model(literal=literal)
    params = activation_params(floor=0.5, literal=literal, parent=True)
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    np.testing.assert_array_almost_equal(
        solution.values[0]["parent"][0],
        np.log(0.125 * 2.75),
        decimal=DECIMAL_PRECISION,
    )


def _without_child_support(*, carry: EGMCarry) -> EGMCarry:
    data = carry.query_data_for_transport()
    assert data is not None
    corner = data.liquid_corner_data
    return replace(
        carry,
        query_data=replace(
            data,
            node_kind=jnp.full_like(data.node_kind, QueryNodeKind.PAD),
            link_valid=jnp.zeros_like(data.link_valid),
            corner_kind=None
            if data.corner_kind is None
            else jnp.full_like(data.corner_kind, QueryNodeKind.PAD),
            liquid_corner_data=None
            if corner is None
            else replace(
                corner,
                node_kind=jnp.full_like(corner.node_kind, QueryNodeKind.PAD),
                link_valid=jnp.zeros_like(corner.link_valid),
            ),
        ),
    )


def _with_invalid_child_floor(
    *, model: Model, context: OracleContext, failure: Literal["nonfinite", "collapse"]
) -> OracleContext:
    child_params = cast("FlatRegimeParams", context["flat_params"]["child"])
    assert "borrowing_lower__floor" in child_params
    child_solver = cast("_BoundDCEGM", model.user_regimes["child"].solver)
    nodes = child_solver.savings_grid.to_jax()
    offsets = nodes - nodes[0]
    dtype = offsets.dtype
    assert jnp.asarray(child_params["borrowing_lower__floor"]).dtype == dtype
    value = np.nan if failure == "nonfinite" else np.finfo(dtype).max / 2.0
    floor = jnp.asarray(value, dtype=dtype)
    assert floor.shape == ()
    assert floor.dtype == dtype
    if failure == "nonfinite":
        assert np.isnan(np.asarray(floor)).item()
    else:
        assert np.isfinite(np.asarray(floor)).item()
        assert (np.asarray(floor) > 0.0).item()
        shifted = np.asarray(floor + offsets)
        assert np.isfinite(shifted).all()
        np.testing.assert_array_equal(shifted[0], floor)
        np.testing.assert_array_equal(np.diff(shifted), np.zeros(offsets.size - 1))
    return {
        **context,
        "flat_params": MappingProxyType(
            {
                **context["flat_params"],
                "child": MappingProxyType(
                    {**child_params, "borrowing_lower__floor": floor}
                ),
            }
        ),
    }


@pytest.mark.parametrize("plain_parent", [False, True])
@pytest.mark.parametrize("dead_source", [False, True])
@pytest.mark.parametrize("failure", ["nonfinite", "collapse", "coverage"])
def test_actual_child_failure_survives_literal_source_with_optional_checks_off(
    *,
    plain_parent: bool,
    dead_source: bool,
    failure: Literal["nonfinite", "collapse", "coverage"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve mandatory target query integrity across current budget emptiness."""
    if plain_parent:
        model = _identity_parent_model(literal=False, dead_source=dead_source)
    else:
        base = build_activation_model(
            floor=0.5, literal=False, asset=False, parent=True
        )
        parent = base.user_regimes["parent"]
        if dead_source:
            parent = replace(
                parent,
                states={"liquid": IrregSpacedGrid(points=np.array([0.175, 0.2]))},
            )
        model = Model(
            regimes={**base.user_regimes, "parent": parent},
            ages=base.ages,
            edges={"parent": {"child": 0}, "child": {"done": 1}},
            initial_nodes={0: "parent"},
            regime_id_class=RegimeId,
        )
    kernel, context = _capture_with_checks_off(
        model=model,
        params=activation_params(floor=0.5, literal=False, parent=True),
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    authority = model._regimes["child"].solution.query_authority
    assert authority is not None
    registry = model._regimes["parent"].solution.query_input_authorities
    assert registry is not None, "The installed source lacks its child query slots."
    slots = registry[0]
    assert slots[0][0] == "child"
    assert slots[0][1] == 1
    assert isinstance(slots[0][2], PostDecisionBoundSpec)
    assert slots[0][2].lower_target == authority.bound.lower_target
    changed_context: OracleContext
    if failure == "coverage":
        carries = context["next_regime_to_continuation"]
        child_carry = carries["child"]
        assert isinstance(child_carry, EGMCarry)
        changed_context = {
            **context,
            "next_regime_to_continuation": cast(
                "Mapping[RegimeName, ContinuationPayload]",
                MappingProxyType(
                    {**carries, "child": _without_child_support(carry=child_carry)}
                ),
            ),
        }
        reason = "query_coverage_missing"
    else:
        changed_context = _with_invalid_child_floor(
            model=model, context=context, failure=failure
        )
        reason = (
            "nonfinite borrowing floor"
            if failure == "nonfinite"
            else "invalid shifted savings grid"
        )
    output = _run_installed(kernel=kernel, context=changed_context)
    assert isinstance(output, FeasibilityKernelOutput)
    assert output.query_status is not None
    expected = {
        "nonfinite": [True, True, False],
        "collapse": [False, True, False],
        "coverage": [False, False, True],
    }[failure]
    np.testing.assert_array_equal(jnp.stack(output.query_status[0])[:2], expected[:2])
    if failure == "coverage":
        np.testing.assert_array_equal(
            output.query_status[0].coverage_missing, jnp.asarray(a=True)
        )
    if dead_source:
        np.testing.assert_array_equal(output.feasibility_counts[0], 2)
    with pytest.raises(InvalidValueFunctionError, match=reason) as error:
        validate_query_input_status(statuses=output.query_status, authorities=slots)
    assert "child" in str(error.value)
    assert "period 1" in str(error.value)


def _mss_parent_model(*, literal_child: bool) -> Model:
    base = build_activation_model(
        floor=0.5, literal=literal_child, asset=False, parent=True
    )
    parent = base.user_regimes["parent"]
    parent = replace(
        parent,
        solver=replace(cast("_BoundDCEGM", parent.solver), envelope=MSSEnvelope()),
    )
    return Model(
        regimes={**base.user_regimes, "parent": parent},
        ages=base.ages,
        edges={"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
    )


def test_literal_mss_parent_admits_the_literal_dc_corner_control() -> None:
    """Reach the child's exact published corner node from a literal MSS source."""
    model = _mss_parent_model(literal_child=True)
    solution = model.solve(
        params=activation_params(floor=0.5, literal=True, parent=True),
        log_level="off",
        runtime_checks=False,
    )
    np.testing.assert_array_almost_equal(
        solution.values[1]["child"][0], np.log(2.75), decimal=DECIMAL_PRECISION
    )
    assert model._regimes["parent"].solution.query_authority is None


@pytest.mark.parametrize("invalid", [False, True])
def test_literal_mss_source_transports_child_status_without_source_query_payload(
    *, invalid: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve MSS source rows and absent query data when reading a feature child."""
    model = _mss_parent_model(literal_child=False)
    kernel, context = _capture_with_checks_off(
        model=model,
        params=activation_params(floor=0.5, literal=False, parent=True),
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    source = model._regimes["parent"].solution
    assert source.query_authority is None
    assert source.query_input_authorities is not None
    slots = source.query_input_authorities[0]
    assert len(slots) == 1
    assert slots[0][0] == "child"
    assert slots[0][1] == 1
    if invalid:
        context = _with_invalid_child_floor(
            model=model, context=context, failure="nonfinite"
        )
    output = _run_installed(kernel=kernel, context=context)
    assert isinstance(output, FeasibilityKernelOutput)
    carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    assert carry.query_data_for_transport() is None
    assert output.query_status is not None
    np.testing.assert_array_equal(jnp.stack(output.query_status[0])[:2], [invalid] * 2)
    if invalid:
        with pytest.raises(
            InvalidValueFunctionError,
            match=r"nonfinite borrowing floor.*child.*period 1",
        ):
            validate_query_input_status(statuses=output.query_status, authorities=slots)
    else:
        np.testing.assert_array_equal(
            jnp.stack(output.query_status[0]), [False, False, False]
        )
        validate_query_input_status(statuses=output.query_status, authorities=slots)
