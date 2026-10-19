"""Weight-dependent asset floors retain direct marginals on a passive stencil."""

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import QueryNodeKind
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from lcm import (
    AgeGrid,
    ByAge,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    post_decision_lower_bound,
)
from lcm.solvers import DCEGM, MSSEnvelope
from lcm.transition import fixed_transition
from lcm.typing import ContinuousState, FloatND, ScalarInt, UserParams
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import (
    _inverse,
    _lower,
    _parent_resources,
    _savings,
    _utility,
)
from tests.egm.test_dynamic_dcegm_boundary_factory import (
    _assert_child_only_status,
    _observe_choices,
)
from tests.egm.test_dynamic_dcegm_consumers import _capture_with_checks_off
from tests.egm.test_dynamic_dcegm_passive_consumers import (
    _child_liquid,
    _child_w,
    _child_z,
)
from tests.solution._nbegm_direct_oracle import OracleContext

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class RegimeId:
    parent: ScalarInt
    child: ScalarInt
    low: ScalarInt
    high: ScalarInt


def _resources(
    *, liquid: ContinuousState, z: ContinuousState, w: ContinuousState
) -> FloatND:
    return 2.0 * liquid + 2.0 + z / 4.0 + w / 8.0


def _high_probability(
    *, liquid: ContinuousState, z: ContinuousState, w: ContinuousState
) -> FloatND:
    return 0.25 + liquid / 8.0 + z / 16.0 + w / 32.0


def _low_probability(
    *, liquid: ContinuousState, z: ContinuousState, w: ContinuousState
) -> FloatND:
    return 0.75 - liquid / 8.0 - z / 16.0 - w / 32.0


def _to_terminal(savings: FloatND) -> ContinuousState:
    return savings


def _low_utility(liquid: ContinuousState) -> FloatND:
    return 2.0 * jnp.ones_like(liquid)


def _high_utility(liquid: ContinuousState) -> FloatND:
    return 6.0 * jnp.ones_like(liquid)


def _model() -> tuple[Model, UserParams]:
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
    actions = {
        "consumption": IrregSpacedGrid(points=np.array([0.025, 0.125, 0.5, 1.0, 4.0]))
    }
    functions = {
        "utility": _utility,
        "savings": _savings,
        "inverse_marginal_utility": _inverse,
    }
    model = Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={
                    "liquid": IrregSpacedGrid(points=np.array([0.5, 0.75])),
                    **passive,
                },
                actions=actions,
                functions={**functions, "resources": _parent_resources},
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=0.25)
                },
                state_transitions={
                    "liquid": _child_liquid,
                    "z": _child_z,
                    "w": _child_w,
                },
                solver=DCEGM(
                    savings_grid=IrregSpacedGrid(points=np.array([0.25, 0.5, 0.75])),
                    n_constrained_points=4,
                    envelope=MSSEnvelope(),
                ),
            ),
            "child": ConsumptionSavingsRegime(
                liquid=margin,
                states={
                    "liquid": IrregSpacedGrid(points=np.array([0.0, 0.5, 1.5])),
                    **passive,
                },
                actions=actions,
                functions={**functions, "resources": _resources},
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=_lower)
                },
                state_transitions={
                    "liquid": _to_terminal,
                    "z": fixed_transition("z"),
                    "w": fixed_transition("w"),
                },
                solver=DCEGM(
                    savings_grid=IrregSpacedGrid(points=np.array([0.5, 1.0, 1.5])),
                    n_constrained_points=4,
                ),
            ),
            "low": Regime(
                states={"liquid": IrregSpacedGrid(points=np.array([0.0, 32.0]))},
                functions={"utility": _low_utility},
            ),
            "high": Regime(
                states={"liquid": IrregSpacedGrid(points=np.array([0.0, 32.0]))},
                functions={"utility": _high_utility},
            ),
        },
        ages=AgeGrid(start=40, inclusive_stop=42, step="Y"),
        edges={
            "parent": {"child": 40},
            "child": Transition(
                targets={"low": 41, "high": 41},
                law=ByAge(
                    cases={
                        41: {
                            "low": StochasticTransition(func=_low_probability),
                            "high": StochasticTransition(func=_high_probability),
                        }
                    }
                ),
            ),
        },
        initial_nodes={40: "parent"},
        regime_id_class=RegimeId,
    )
    return model, {
        "parent": {"koopmans_aggregator": {"discount_factor": 0.5}},
        "child": {
            "koopmans_aggregator": {"discount_factor": 0.5},
            "borrowing_lower": {"floor": 0.5},
        },
        "low": {},
        "high": {},
    }


def test_actual_asset_factory_blends_weight_only_direct_channel_on_two_passives(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read affine floor continuation and direct marginals at nodes and interiors."""
    model, params = _model()
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
    corner = data.liquid_corner_data
    assert corner is not None
    assert data.corner_continuation is None
    assert child.endog_grid.shape == (2, 3, 3)
    assert corner.continuation.shape == (2, 3, 3, 1)
    z, w, liquid = np.meshgrid(
        [0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.5, 1.5], indexing="ij"
    )
    expected_bank = 3.0 + liquid / 2.0 + z / 4.0 + w / 8.0
    np.testing.assert_array_equal(data.floor_used, np.full((2, 3), 0.5))
    np.testing.assert_allclose(
        corner.continuation[..., 0], expected_bank, rtol=10**-DECIMAL_PRECISION, atol=0
    )
    np.testing.assert_allclose(
        corner.direct_marginal[..., 0],
        np.full((2, 3, 3), 0.25),
        rtol=10**-DECIMAL_PRECISION,
        atol=0,
    )
    np.testing.assert_array_equal(
        corner.node_kind, np.full((2, 3, 3, 1), QueryNodeKind.FINITE)
    )
    assert np.all(
        (0.25 + liquid / 8.0 + z / 16.0 + w / 32.0 > 0.0)
        & (0.25 + liquid / 8.0 + z / 16.0 + w / 32.0 < 1.0)
    )
    absent = replace(
        data,
        node_kind=jnp.full_like(data.node_kind, QueryNodeKind.PAD),
        link_valid=jnp.zeros_like(data.link_valid),
    )
    assert absent.liquid_corner_data is corner
    changed: OracleContext = {
        **context,
        "next_regime_to_continuation": {
            **context["next_regime_to_continuation"],
            "child": replace(child, query_data=absent),
        },
    }
    output, observations = _observe_choices(
        kernel=kernel,
        context=changed,
        names=("liquid", "z", "w", "period", "borrowing_lower__floor"),
        monkeypatch=monkeypatch,
    )
    for query in (0.0, 1.25):
        selected = [
            actual
            for facts, actual in observations
            if np.array_equal(facts, [query, 0.25, 0.5, 1.0, 0.5])
        ]
        assert selected
        resources = 2.0 * query + 2.0 + 0.25 / 4.0 + 0.5 / 8.0
        continuation = 3.0 + query / 2.0 + 0.25 / 4.0 + 0.5 / 8.0
        expected = [
            np.log(resources - 0.5) + 0.5 * continuation,
            5.0 * (2.0 / (resources - 0.5) + 0.25),
            0.5,
            0.0,
        ]
        for actual in selected:
            np.testing.assert_allclose(
                actual[:, 0],
                expected,
                rtol=10**-DECIMAL_PRECISION,
                atol=0,
            )
    _assert_child_only_status(model=model, output=output)
