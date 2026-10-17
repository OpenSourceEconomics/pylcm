"""Actual child factories retain boundary and represented-interior tuples."""

from pathlib import Path
from typing import Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.dcegm_query import DCEGMQuery, DCEGMQueryResult
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.feasibility import (
    FeasibilityKernelOutput,
    validate_query_input_status,
)
from _lcm.typing import FlatRegimeParams
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    post_decision_lower_bound,
)
from lcm.solvers import DCEGM, MSSEnvelope
from lcm.typing import ContinuousState, FloatND, ReferenceName, UserParams
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import (
    RegimeId,
    _asset_law,
    _inverse,
    _lower,
    _ordinary_law,
    _parent_resources,
    _resources,
    _savings,
    _terminal,
    _utility,
)
from tests.egm.test_dynamic_dcegm_consumers import (
    _capture_with_checks_off,
    _run_installed,
)
from tests.egm.test_dynamic_dcegm_passive_consumers import _ChoiceInputs
from tests.solution._nbegm_direct_oracle import OracleContext, interpolate_row

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _source_law(*, savings: FloatND, query_point: float) -> ContinuousState:
    return query_point + 5.0 * (savings - 0.25)


def _model(*, asset: bool) -> tuple[Model, UserParams]:
    switch = 0.8 if asset else 2.0 / 3.0
    liquid = (0.0, switch, 1.5, 2.5)
    saving = (0.5, 1.5, 2.5)
    terminal = tuple(
        sorted({3.0 * a + 4.0 + (x if asset else 0.0) for a in saving for x in liquid})
    )
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    functions = {
        "utility": _utility,
        "savings": _savings,
        "inverse_marginal_utility": _inverse,
    }
    actions = {"consumption": IrregSpacedGrid(points=(0.025, 0.125, 0.5, 1.0, 4.0))}
    model = Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={"liquid": IrregSpacedGrid(points=(0.5, 0.75))},
                actions=actions,
                functions={**functions, "resources": _parent_resources},
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=0.25)
                },
                state_transitions={"liquid": _source_law},
                solver=DCEGM(
                    savings_grid=IrregSpacedGrid(points=(0.25, 0.5, 0.75)),
                    n_constrained_points=4,
                    envelope=MSSEnvelope(),
                ),
            ),
            "child": ConsumptionSavingsRegime(
                liquid=margin,
                states={"liquid": IrregSpacedGrid(points=liquid)},
                actions=actions,
                functions={**functions, "resources": _resources},
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=_lower)
                },
                state_transitions={"liquid": _asset_law if asset else _ordinary_law},
                solver=DCEGM(
                    savings_grid=IrregSpacedGrid(points=saving), n_constrained_points=4
                ),
            ),
            "done": Regime(
                states={"liquid": IrregSpacedGrid(points=terminal)},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
    )
    return model, {
        "parent": {
            "koopmans_aggregator": {"discount_factor": 0.5},
            "next_liquid": {"query_point": 0.0},
        },
        "child": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "borrowing_lower": {"floor": 0.5},
        },
        "done": {},
    }


def _observe_choices(
    *,
    kernel: _DCEGMPeriodKernel,
    context: OracleContext,
    names: tuple[ReferenceName, ...],
    monkeypatch: pytest.MonkeyPatch,
    observations: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[FeasibilityKernelOutput, list[tuple[np.ndarray, np.ndarray]]]:
    """Observe unchanged conditional results from the actual installed factory."""
    observations = [] if observations is None else observations
    original = DCEGMQuery.read_choices

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record(facts: np.ndarray, values: np.ndarray) -> None:
        observations.append((np.array(facts, copy=True), np.array(values, copy=True)))

    def observe(self: DCEGMQuery, **kwargs: Unpack[_ChoiceInputs]) -> DCEGMQueryResult:
        result = original(self, **kwargs)
        assert isinstance(result, DCEGMQueryResult)
        facts = jnp.stack([cast("FloatND", kwargs["pool"][name]) for name in names])
        jax.debug.callback(
            record,
            facts,
            jnp.stack(
                (result.value, result.marginal, result.savings, result.coverage_missing)
            ),
        )
        return result

    with monkeypatch.context() as patch:
        patch.setattr(DCEGMQuery, "read_choices", observe)
        output = _run_installed(kernel=kernel, context=context)
        jax.block_until_ready(output.value)
        jax.effects_barrier()
    assert isinstance(output, FeasibilityKernelOutput)
    return output, observations


def _assert_child_only_status(*, model: Model, output: FeasibilityKernelOutput) -> None:
    """Accept every actual slot on a literal MSS source consuming one feature child."""
    slots = model._regimes["parent"].solution.query_input_authorities
    assert slots is not None
    assert tuple((name, period) for name, period, _ in slots[0]) == (("child", 1),)
    assert output.query_status is not None
    validate_query_input_status(statuses=output.query_status, authorities=slots[0])
    np.testing.assert_array_equal(
        jnp.stack(tuple(jnp.stack(flag) for flag in output.query_status)),
        np.zeros((1, 3), dtype=bool),
    )
    if output.bound_status:
        np.testing.assert_array_equal(
            jnp.stack(tuple(jnp.stack(flag) for flag in output.bound_status)),
            np.zeros((len(output.bound_status), 2), dtype=bool),
        )


def _owned_reference(
    *,
    grid: np.ndarray,
    values: np.ndarray,
    marginal: np.ndarray,
    savings: np.ndarray,
    data: EGMQueryData,
    query: float,
) -> tuple[float, float, float] | None:
    """Read a supported tuple with independent scalar interpolation."""
    count = int(np.count_nonzero(np.isfinite(grid)))
    assert count > 1
    insertion = int(np.searchsorted(grid[:count], query, side="right"))
    endpoint = min(max(insertion - 1, 0), count - 1)
    exact = query >= grid[count - 1] or query == grid[endpoint]
    upper = min(max(insertion, 1), count - 1)
    selected = [endpoint] if exact else [upper - 1, upper]
    kind = np.asarray(data.node_kind)[selected]
    if not np.all(kind == QueryNodeKind.FINITE):
        return None
    assert np.isfinite(
        np.stack((values[selected], marginal[selected], savings[selected]))
    ).all()
    if not exact and (
        not bool(np.asarray(data.link_valid)[upper - 1])
        or np.ptp(np.asarray(data.owner_id)[selected]) != 0
    ):
        return None
    return (
        interpolate_row(query=query, xp=grid, fp=values, slopes=marginal),
        interpolate_row(query=query, xp=grid, fp=marginal, slopes=None),
        interpolate_row(query=query, xp=grid, fp=savings, slopes=None),
    )


def _reference(
    *, child: EGMCarry, liquid: float, asset: bool
) -> tuple[float, float, float]:
    """Rank the represented ordinary tuple against the independently computed floor."""
    resources = 2.0 * liquid + 1.0
    if resources <= 0.5:
        return -np.inf, 0.0, 0.5
    data = child.query_data_for_transport()
    assert data is not None
    grid = np.asarray(child.endog_grid)
    if asset:
        corner = data.liquid_corner_data
        assert corner is not None
        continuation = interpolate_row(
            query=resources,
            xp=grid,
            fp=np.asarray(corner.continuation[..., 0]),
            slopes=np.asarray(corner.direct_marginal[..., 0]),
        )
        direct = interpolate_row(
            query=resources,
            xp=grid,
            fp=np.asarray(corner.direct_marginal[..., 0]),
            slopes=None,
        )
    else:
        assert data.corner_continuation is not None
        continuation = float(np.asarray(data.corner_continuation).item())
        direct = 0.0
    expected = (
        float(np.log(resources - 0.5) + continuation),
        10.0 * (1.0 / (resources - 0.5) + direct),
        0.5,
    )
    row = _owned_reference(
        grid=grid,
        values=np.asarray(child.value),
        marginal=np.asarray(child.marginal_utility),
        savings=np.asarray(data.savings),
        data=data,
        query=resources,
    )
    if (
        row is not None
        and 0.5 <= row[2] <= 2.5
        and resources > row[2]
        and row[0] >= expected[0]
    ):
        return row[0], 10.0 * row[1], row[2]
    return expected


@pytest.mark.parametrize("asset", [False, True])
def test_actual_factory_retains_binding_switch_interior_and_empty_queries(
    *, asset: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Select complete boundary and interior tuples while preserving empty budgets."""
    model, params = _model(asset=asset)
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    child = context["next_regime_to_continuation"]["child"]
    assert isinstance(child, EGMCarry)
    data = child.query_data_for_transport()
    assert data is not None
    np.testing.assert_array_equal(data.floor_used, 0.5)
    switch = 0.8 if asset else 2.0 / 3.0
    if asset:
        corner = data.liquid_corner_data
        assert corner is not None
        liquid = np.asarray(
            [0.0, switch, 1.5, 2.5], dtype=np.asarray(child.endog_grid).dtype
        )
        np.testing.assert_array_equal(child.endog_grid, 2.0 * liquid + 1.0)
        np.testing.assert_allclose(
            corner.continuation[..., 0],
            np.log(5.5 + liquid),
            rtol=10**-DECIMAL_PRECISION,
            atol=0,
        )
        np.testing.assert_allclose(
            corner.direct_marginal[..., 0],
            1.0 / (2.0 * (5.5 + liquid)),
            rtol=10**-DECIMAL_PRECISION,
            atol=0,
        )
    else:
        assert data.corner_continuation is not None
        np.testing.assert_allclose(
            data.corner_continuation, np.log(5.5), rtol=10**-DECIMAL_PRECISION, atol=0
        )
    parent = cast("FlatRegimeParams", context["flat_params"]["parent"])
    aliases = frozenset(name for name in parent if name.endswith("__query_point"))
    assert aliases == frozenset(
        {
            "child__next_liquid__query_point",
            "parent__next_liquid__query_point",
            "done__next_liquid__query_point",
        }
    )
    observed_frames: list[tuple[np.ndarray, np.ndarray]] = []
    for index, query in enumerate((0.0, switch, 1.5, -0.25)):
        working_query = jnp.asarray(query, dtype=jnp.asarray(0.0).dtype)
        changed: OracleContext = {
            **context,
            "flat_params": model._process_params(
                {
                    **params,
                    "parent": {
                        "koopmans_aggregator": {"discount_factor": 0.5},
                        "next_liquid": {"query_point": working_query},
                    },
                },
                runtime_checks=False,
            ),
        }
        changed_parent = cast("FlatRegimeParams", changed["flat_params"]["parent"])
        for alias in aliases:
            np.testing.assert_array_equal(changed_parent[alias], working_query)
        for name in parent.keys() - aliases:
            np.testing.assert_array_equal(changed_parent[name], parent[name])
        frame_start = len(observed_frames)
        output, _observed = _observe_choices(
            kernel=kernel,
            context=changed,
            names=("liquid", "period", "borrowing_lower__floor"),
            monkeypatch=monkeypatch,
            observations=observed_frames,
        )
        current_frames = observed_frames[frame_start:]
        selected = [
            value for facts, value in current_frames if facts[0] == float(working_query)
        ]
        assert selected, {
            "requested": np.asarray(working_query).tolist(),
            "requested_shape": working_query.shape,
            "requested_dtype": str(working_query.dtype),
            "frame_count": len(current_frames),
            "last_frames": [
                (facts.shape, str(facts.dtype), facts.tolist())
                for facts, _ in current_frames[-16:]
            ],
        }
        expected = _reference(child=child, liquid=float(working_query), asset=asset)
        for actual in selected:
            np.testing.assert_allclose(
                actual[:, 0], [*expected, 0.0], rtol=10**-DECIMAL_PRECISION, atol=0
            )
        if index < 2:
            np.testing.assert_allclose(
                expected[2], 0.5, rtol=10**-DECIMAL_PRECISION, atol=0
            )
        elif index == 2:
            assert expected[2] > 0.5
        _assert_child_only_status(model=model, output=output)
