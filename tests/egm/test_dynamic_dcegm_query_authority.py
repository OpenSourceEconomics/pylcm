"""Canonical DC query predicates authenticate the producer's actual period pools."""

from dataclasses import replace
from types import MappingProxyType
from typing import Literal

import jax.numpy as jnp
import pytest

from _lcm.solution.model_authority import build_solution_authority
from lcm.solver_api import EGM_CONTINUATION, ArtifactRef
from lcm.typing import BoolND, FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import (
    activation_params,
    build_activation_model,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _foreign_floor(*, borrowing_lower__floor: FloatND) -> FloatND:
    return borrowing_lower__floor + 1.0


def _foreign_borrowing(*, savings: FloatND, borrowing_lower: FloatND) -> BoolND:
    return jnp.less(savings, borrowing_lower)


@pytest.mark.parametrize(
    "change",
    ["none", "missing_predicate", "predicate", "base_lower", "period_lower"],
)
def test_dc_query_authority_authenticates_full_constraints_and_actual_numeric_pools(
    *,
    change: Literal[
        "none", "missing_predicate", "predicate", "base_lower", "period_lower"
    ],
) -> None:
    """Reject changed source bindings while retaining every canonical DC predicate."""
    model = build_activation_model(floor=0.5, literal=False, asset=False)
    regime = model._regimes["child"]
    phase = regime.solution
    authority = phase.query_authority
    assert authority is not None
    assert ("predicate:borrowing", "borrowing") in authority.evaluator_targets
    query_constraints = phase.query_constraint_functions
    assert query_constraints is not None
    assert "borrowing" in query_constraints
    assert "borrowing" not in phase.constraints
    assert "borrowing" not in phase.continuation_functions
    lower_target = authority.bound.lower_target
    assert lower_target == "borrowing_lower"
    assert lower_target in phase.continuation_functions
    original_query_constraints = query_constraints
    if change == "missing_predicate":
        phase = replace(
            phase,
            query_constraint_functions=MappingProxyType(
                {
                    target: function
                    for target, function in query_constraints.items()
                    if target != "borrowing"
                }
            ),
        )
    elif change == "predicate":
        phase = replace(
            phase,
            query_constraint_functions=MappingProxyType(
                {**query_constraints, "borrowing": _foreign_borrowing}
            ),
        )
    elif change == "base_lower":
        phase = replace(
            phase,
            _continuation_functions=MappingProxyType(
                {**phase.continuation_functions, lower_target: _foreign_floor}
            ),
        )
        assert phase.query_constraint_functions is original_query_constraints
    elif change == "period_lower":
        functions, constraints = phase.feasibility_pools_by_period[0]
        phase = replace(
            phase,
            feasibility_pools_by_period=MappingProxyType(
                {
                    **phase.feasibility_pools_by_period,
                    0: (
                        MappingProxyType({**functions, lower_target: _foreign_floor}),
                        constraints,
                    ),
                }
            ),
        )
        assert phase.query_constraint_functions is original_query_constraints
    model_regimes = MappingProxyType(
        {**model._regimes, "child": replace(regime, solution=phase)}
    )
    assert model.ages is not None
    flat_params = model._process_params(activation_params(floor=0.5, literal=False))
    if change == "none":
        result = build_solution_authority(
            regimes=model_regimes, flat_params=flat_params, ages=model.ages
        )
        assert (
            ArtifactRef(period=0, regime="child", key=EGM_CONTINUATION)
            in result.artifact_descriptors
        )
    else:
        with pytest.raises(TypeError, match=r"Query|query"):
            build_solution_authority(
                regimes=model_regimes, flat_params=flat_params, ages=model.ages
            )


@pytest.mark.parametrize("pool", ["base", "period"])
def test_dc_query_authority_rejects_deleted_numeric_floor_despite_saved_fallback(
    *, pool: Literal["base", "period"]
) -> None:
    """Reject missing actual floor bindings despite an identical supplemental copy."""
    model = build_activation_model(floor=0.5, literal=False, asset=False)
    regime = model._regimes["child"]
    phase = regime.solution
    authority = phase.query_authority
    assert authority is not None
    lower_target = authority.bound.lower_target
    assert lower_target == "borrowing_lower"
    query_constraints = phase.query_constraint_functions
    assert query_constraints is not None
    assert lower_target not in query_constraints
    original_lower = phase.continuation_functions[lower_target]
    supplemental = MappingProxyType({**query_constraints, lower_target: original_lower})
    assert supplemental[lower_target] is original_lower
    if pool == "base":
        actual_functions = MappingProxyType(
            {
                target: function
                for target, function in phase.continuation_functions.items()
                if target != lower_target
            }
        )
        assert lower_target not in actual_functions
        phase = replace(
            phase,
            query_constraint_functions=supplemental,
            _continuation_functions=actual_functions,
        )
    else:
        functions, constraints = phase.feasibility_pools_by_period[0]
        assert functions[lower_target] is original_lower
        actual_functions = MappingProxyType(
            {
                target: function
                for target, function in functions.items()
                if target != lower_target
            }
        )
        assert lower_target not in actual_functions
        phase = replace(
            phase,
            query_constraint_functions=supplemental,
            feasibility_pools_by_period=MappingProxyType(
                {
                    **phase.feasibility_pools_by_period,
                    0: (actual_functions, constraints),
                }
            ),
        )
    model_regimes = MappingProxyType(
        {**model._regimes, "child": replace(regime, solution=phase)}
    )
    assert model.ages is not None
    flat_params = model._process_params(activation_params(floor=0.5, literal=False))
    with pytest.raises(TypeError, match=r"Query|query"):
        build_solution_authority(
            regimes=model_regimes, flat_params=flat_params, ages=model.ages
        )
