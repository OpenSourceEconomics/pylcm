"""Scalar sampled NB rows retain continuous links without joining budget jumps."""

import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from _lcm.egm.carry import EGMCarry
from _lcm.solution.feasibility import BoundedResult
from _lcm.solution.nbegm import _NBEGMOwnedSingleAxisCore
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    post_decision_lower_bound,
)
from lcm.solvers import EGM, NBEGM
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, IntND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import RegimeId

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _resources(*, liquid: ContinuousState) -> FloatND:
    return liquid


def _budget(*, liquid: ContinuousState, tax: FloatND) -> FloatND:
    return liquid - tax


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _floor() -> FloatND:
    return jnp.asarray(0.5)


def _identity(*, savings: FloatND) -> ContinuousState:
    return savings


def _landing(*, savings: FloatND) -> ContinuousState:
    return 3.0 * savings + 4.0


def _terminal(*, liquid: ContinuousState) -> FloatND:
    return jnp.log(liquid)


def _tax_kink(*, liquid: ContinuousState, threshold: float, rate: float) -> FloatND:
    return rate * jnp.maximum(liquid - threshold, 0.0)


def _tax_jump(*, liquid: ContinuousState, threshold: float, rate: float) -> FloatND:
    return rate * jnp.where(liquid >= threshold, 1.0, 0.0)


def _model(*, kind: Literal["continuous_kink", "jump"], callable_floor: bool) -> Model:
    """Build genuine scalar source and child authority with a nonsampled budget cut."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    tax = lcm.piecewise_affine(
        output="tax",
        variable="liquid",
        breakpoints=(lcm.affine_breakpoint(threshold="threshold", kind=kind),),
    )(_tax_kink if kind == "continuous_kink" else _tax_jump)
    functions = {
        "utility": _utility,
        "inverse_marginal_utility": _inverse,
        "savings": _savings,
    }
    return Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={"liquid": IrregSpacedGrid(points=np.array([5.0, 6.0]))},
                actions={
                    "consumption": IrregSpacedGrid(
                        points=np.array([0.025, 0.5, 1.0, 3.0])
                    )
                },
                functions={**functions, "resources": _resources},
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=0.0)
                },
                state_transitions={"liquid": _identity},
                solver=EGM(
                    savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 2.0]))
                ),
            ),
            "child": ConsumptionSavingsRegime(
                liquid=margin,
                states={"liquid": IrregSpacedGrid(points=np.array([0.75, 1.5, 3.0]))},
                actions={
                    "consumption": IrregSpacedGrid(
                        points=np.array([0.025, 0.05, 0.25, 0.5, 1.0])
                    )
                },
                functions={**functions, "resources": _budget, "tax": tax},
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=margin, lower=_floor if callable_floor else 0.5
                    )
                },
                state_transitions={"liquid": _landing},
                solver=NBEGM(
                    savings_grid=IrregSpacedGrid(
                        points=np.array(
                            (0.0, 0.5, 1.0) if callable_floor else (0.5, 1.0, 1.5)
                        )
                    )
                ),
            ),
            "done": Regime(
                states={"liquid": IrregSpacedGrid(points=np.array([4.0, 5.5, 10.0]))},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
    )


@pytest.mark.parametrize("kind", ["continuous_kink", "jump"])
def test_scalar_sampled_run_preserves_continuous_links_and_jump_separation(
    *,
    kind: Literal["continuous_kink", "jump"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a declared continuous budget cut shares the sampled row's linked owner."""
    observed = []
    original = _NBEGMOwnedSingleAxisCore.__call__

    def observe[**P](
        function: Callable[
            P, BoundedResult[tuple[FloatND, EGMCarry, tuple[IntND, IntND], BoolND]]
        ],
    ) -> Callable[
        P, BoundedResult[tuple[FloatND, EGMCarry, tuple[IntND, IntND], BoolND]]
    ]:
        def capture(
            *args: P.args, **kwargs: P.kwargs
        ) -> BoundedResult[tuple[FloatND, EGMCarry, tuple[IntND, IntND], BoolND]]:
            result = function(*args, **kwargs)
            carry = result.output[1]
            data = carry.query_data_for_transport()
            if data is not None:
                jax.debug.callback(
                    lambda *leaves: observed.append(
                        tuple(np.asarray(leaf) for leaf in leaves)
                    ),
                    carry.endog_grid,
                    carry.value,
                    carry.marginal_utility,
                    data.savings,
                    data.node_kind,
                    data.link_valid,
                    data.owner_id,
                    data.side_id,
                    data.corner_interval_id,
                    data.corner_continuation,
                )
            return result

        return capture

    monkeypatch.setattr(_NBEGMOwnedSingleAxisCore, "__call__", observe(original))
    parameters = {"discount_factor": 1.0, "rate": 0.2, "threshold": 1.2}
    literal = _model(kind=kind, callable_floor=False).solve(
        params=parameters, log_level="off"
    )
    feature = _model(kind=kind, callable_floor=True).solve(
        params=parameters, log_level="off"
    )
    jax.effects_barrier()
    assert len(observed) == 1
    (
        grid,
        value,
        marginal,
        saving,
        node_kind,
        links,
        owners,
        sides,
        corner_ids,
        corner_values,
    ) = observed[0]
    np.testing.assert_array_equal(grid, [0.75, 1.5, 3.0])
    assert np.all(np.isfinite(value))
    assert np.all(np.isfinite(marginal))
    np.testing.assert_array_equal(node_kind, [1, 1, 1])
    assert np.all(saving >= 0.5)
    assert np.all(saving <= 1.5)
    np.testing.assert_array_equal(sides, [0, 0, 0])
    np.testing.assert_array_equal(corner_ids, [0, 1])
    np.testing.assert_array_almost_equal(
        corner_values, [math.log(5.5)] * 2, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        feature.values[1]["child"],
        literal.values[1]["child"],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(links, [kind == "continuous_kink", True])
    np.testing.assert_array_equal(
        owners, [0, 0, 0] if kind == "continuous_kink" else [0, 1, 1]
    )
