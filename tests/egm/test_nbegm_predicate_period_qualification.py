"""NB predicate authority requires every target period's actual supported topology."""

from collections.abc import Callable
from dataclasses import replace
from functools import partial, wraps
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import continuation as authority_module
from _lcm.continuation import EGMQuerySpec
from _lcm.egm.nbegm import collect_nbegm_metadata
from _lcm.egm.nbegm_query import build_nbegm_query
from _lcm.egm.nbegm_query_model import NBEGMQueryModel
from _lcm.regime_building import processing
from _lcm.regime_building.age_normalization import (
    resolve_periodized_node,
    resolve_periodized_nodes,
)
from _lcm.typing import EconFunction, EconFunctionsMapping, RegimeName
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    affine_breakpoint,
    piecewise_affine,
    post_decision_lower_bound,
)
from lcm.exceptions import ModelInitializationError
from lcm.solvers import NBEGM
from lcm.transition import AgeSpecializedFunction, fixed_transition
from lcm.typing import BoolND, ContinuousState, FloatND, UserFunction
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm import test_nbegm_literal_predicate_consumers as original
from tests.egm.test_dynamic_dcegm_activation import ChildRegimeId

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def test_uniform_period_topology_installs_actual_predicate_authority(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allowed topology in both periods retains its exact canonical target bindings."""
    pools, models, authorities = _observe_real_builders(monkeypatch=monkeypatch)
    _assert_ordinary_control(pools=pools)
    model = _model(predicate=True, later_jump=False)
    _assert_period_pools(
        pools=pools[-1], expected_kinds=("continuous_kink", "continuous_kink")
    )
    assert len(models) == 1
    query_model = models[0]
    assert query_model.evaluated_predicate_names == ("eligibility",)
    assert query_model.authority is model._regimes["child"].solution.query_authority
    assert query_model.authority.target_periods == (0, 1)
    assert any(item is query_model.authority for item in authorities)
    assert query_model.authority.bound.kind == "literal"
    assert dict(query_model.authority.evaluator_inputs)["predicate:eligibility"] == (
        "z",
    )
    for period in (0, 1):
        query = build_nbegm_query(
            model=query_model,
            functions=cast(
                "EconFunctionsMapping",
                resolve_periodized_nodes(mapping=query_model.functions, period=period),
            ),
            koopmans_aggregator=cast(
                "EconFunction",
                resolve_periodized_node(
                    node=query_model.koopmans_aggregator, period=period
                ),
            ),
            target_period=period,
        )
        assert query.model.authority is query_model.authority
        assert (
            query.model.intervals.source_identity
            == query_model.intervals_by_period[period].source_identity
        )
        assert query.lower is None


def test_later_excluded_topology_cannot_install_regime_wide_predicate_authority(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later jump prevents a literal predicate capability despite an allowed first
    period.
    """
    pools, models, authorities = _observe_real_builders(monkeypatch=monkeypatch)
    _assert_ordinary_control(pools=pools)
    # V1 may fail later at recipe identity; inspect actual activation first.
    with pytest.raises((ModelInitializationError, TypeError)) as error:
        _model(predicate=True, later_jump=True)
    assert len(pools) == 2, (
        "The feature model did not reach actual period-pool qualification."
    )
    _assert_period_pools(pools=pools[-1], expected_kinds=("continuous_kink", "jump"))
    assert authorities == [], (
        "A literal predicate authority was built before all-period qualification."
    )
    assert models == []
    assert isinstance(error.value, ModelInitializationError)
    assert "eligibility" in str(error.value)


def test_literal_query_keeps_original_nodes_and_reports_actual_grid_corruption(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A literal query preserves IEEE node bits and validates the original physical
    grid.
    """
    _pools, models, _authorities = _observe_real_builders(monkeypatch=monkeypatch)
    exponent = 25 if jnp.asarray(0.0).dtype == jnp.float32 else 54
    origin = -float(2**exponent)
    grid = IrregSpacedGrid(points=np.array([origin, 1.0, 5.0]))
    _model(predicate=True, later_jump=False, savings_grid=grid)
    assert len(models) == 1
    model = models[0]
    query = build_nbegm_query(
        model=model,
        functions=cast(
            "EconFunctionsMapping",
            resolve_periodized_nodes(mapping=model.functions, period=0),
        ),
        koopmans_aggregator=cast(
            "EconFunction",
            resolve_periodized_node(node=model.koopmans_aggregator, period=0),
        ),
        target_period=0,
    )
    assert query.lower is None
    floor, nodes, status = query.evaluate_floor(pool={})
    original_nodes = grid.to_jax()
    np.testing.assert_array_equal(nodes, original_nodes)
    assert np.asarray(nodes).tobytes() == np.asarray(original_nodes).tobytes()
    np.testing.assert_array_equal(floor, original_nodes[0])
    np.testing.assert_array_equal(jnp.stack(status), [False, False])
    rounded_reconstruction = original_nodes[0] + (original_nodes - original_nodes[0])
    np.testing.assert_array_equal(
        rounded_reconstruction, jnp.asarray([origin, 0.0, 4.0])
    )
    assert (
        np.asarray(rounded_reconstruction).tobytes()
        != np.asarray(original_nodes).tobytes()
    )
    # Corrupt numeric original-node input while retaining the genuine authority.
    bad_nodes = original_nodes.at[1].set(original_nodes[0])
    corrupted = replace(query, model=replace(query.model, literal_nodes=bad_nodes))
    bad_floor, returned_nodes, bad_status = corrupted.evaluate_floor(pool={})
    np.testing.assert_array_equal(bad_floor, floor)
    np.testing.assert_array_equal(returned_nodes, bad_nodes)
    np.testing.assert_array_equal(jnp.stack(bad_status), [False, True])


def _observe_real_builders(
    *, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    list[dict[int, EconFunctionsMapping]], list[NBEGMQueryModel], list[EGMQuerySpec]
]:
    pools: list[dict[int, EconFunctionsMapping]] = []
    models: list[NBEGMQueryModel] = []
    authorities: list[EGMQuerySpec] = []
    real_prepare = processing._prepare_nbegm_query_models

    def prepare(
        *, processor: processing._CanonicalRegimeBuilder
    ) -> MappingProxyType[RegimeName, NBEGMQueryModel] | None:
        periods = processor.regimes_to_active_periods["child"]
        pools.append(
            {
                period: cast(
                    "EconFunctionsMapping",
                    resolve_periodized_nodes(
                        mapping=processor.specs["child"].solution.functions,
                        period=period,
                    ),
                )
                for period in periods
            }
        )
        result = real_prepare(processor=processor)
        if result is not None:
            models.append(result["child"])
        return result

    monkeypatch.setattr(processing, "_prepare_nbegm_query_models", prepare)
    monkeypatch.setattr(
        authority_module,
        "build_query_model_authority",
        _record_authorities(
            function=authority_module.build_query_model_authority,
            authorities=authorities,
        ),
    )
    return pools, models, authorities


def _record_authorities[**P](
    *, function: Callable[P, EGMQuerySpec], authorities: list[EGMQuerySpec]
) -> Callable[P, EGMQuerySpec]:
    @wraps(function)
    def record(*args: P.args, **kwargs: P.kwargs) -> EGMQuerySpec:
        authority = function(*args, **kwargs)
        authorities.append(authority)
        return authority

    return record


def _assert_ordinary_control(*, pools: list[dict[int, EconFunctionsMapping]]) -> None:
    model = _model(predicate=False, later_jump=False)
    assert model._regimes["child"].active_periods == (0, 1)
    assert tuple(model._regimes["child"].solution.period_kernels) == (0, 1)
    assert model._regimes["child"].solution.query_authority is None
    _assert_period_pools(
        pools=pools[-1], expected_kinds=("continuous_kink", "continuous_kink")
    )


def _assert_period_pools(
    *, pools: dict[int, EconFunctionsMapping], expected_kinds: tuple[str, str]
) -> None:
    assert tuple(pools) == (0, 1)
    for period, expected_kind in enumerate(expected_kinds):
        registry = collect_nbegm_metadata(functions=pools[period])
        assert registry.piece_sets == ()
        assert tuple(
            point.kind
            for schedule in registry.piecewise_affine_schedules
            for point in schedule.breakpoints
        ) == (expected_kind,)
        resources = pools[period]["resources"]
        np.testing.assert_array_equal(
            resources(liquid=jnp.asarray(1.0), cutoff=jnp.asarray(0.5)),
            4.5 if expected_kind == "jump" else 4.0,
        )


def _model(
    *, predicate: bool, later_jump: bool, savings_grid: IrregSpacedGrid | None = None
) -> Model:
    liquid_margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    grid = (
        IrregSpacedGrid(points=np.array([0.0, 0.5, 1.5]))
        if savings_grid is None
        else savings_grid
    )
    assert grid.points is not None
    child = ConsumptionSavingsRegime(
        liquid=liquid_margin,
        states={
            "liquid": IrregSpacedGrid(points=np.array([0.0, 0.5, 1.5])),
            "z": IrregSpacedGrid(points=np.array([0.0, 1.0])),
            "w": IrregSpacedGrid(points=np.array([0.0, 1.0, 2.0])),
        },
        actions={
            "job": DiscreteGrid(category_class=original.Job),
            "consumption": IrregSpacedGrid(
                points=np.array([0.025, 0.125, 0.5, 1.0, 4.0])
            ),
        },
        functions={
            "utility": original._child_utility,
            "savings": original._savings,
            "resources": AgeSpecializedFunction(
                build=partial(_budget_at_age, later_jump=later_jump),
                signature=lambda age: (
                    "jump" if later_jump and age > 40.0 else "continuous_kink"
                ),
            ),
            "eligible_stock": _eligible_stock,
        },
        constraints={
            "borrowing": post_decision_lower_bound(
                margin=liquid_margin, lower=float(grid.points[0])
            ),
            **({"eligibility": _eligible} if predicate else {}),
        },
        state_transitions={
            "liquid": original._next_liquid,
            "z": fixed_transition("z"),
            "w": fixed_transition("w"),
        },
        solver=NBEGM(savings_grid=grid),
    )
    return Model(
        regimes={
            "child": child,
            "done": Regime(
                states={
                    "liquid": IrregSpacedGrid(points=np.array([0.0, 16.0])),
                    "z": IrregSpacedGrid(points=np.array([0.0, 1.0])),
                    "w": IrregSpacedGrid(points=np.array([0.0, 1.0, 2.0])),
                },
                functions={"utility": original._terminal},
            ),
        },
        ages=AgeGrid(start=40, inclusive_stop=42, step="Y"),
        edges={"child": {"child": 40, "done": 41}},
        initial_nodes={40: "child"},
        regime_id_class=ChildRegimeId,
        enable_jit=False,
    )


# keyword-only-exempt: library-callback=AgeSpecializedFunction.build
def _budget_at_age(age: float, *, later_jump: bool) -> UserFunction:
    return _jump_resources if later_jump and age > 40.0 else _kink_resources


@piecewise_affine(
    output="resources",
    variable="liquid",
    breakpoints=(affine_breakpoint(threshold="cutoff", kind="continuous_kink"),),
)
def _kink_resources(*, liquid: ContinuousState, cutoff: float) -> FloatND:
    return 2.0 * liquid + 2.0 + 0.0 * cutoff


@piecewise_affine(
    output="resources",
    variable="liquid",
    breakpoints=(affine_breakpoint(threshold="cutoff", kind="jump"),),
)
def _jump_resources(*, liquid: ContinuousState, cutoff: float) -> FloatND:
    return 2.0 * liquid + 2.0 + jnp.where(liquid >= cutoff, 0.5, 0.0)


def _eligible_stock(*, z: ContinuousState) -> FloatND:
    return z


def _eligible(*, eligible_stock: FloatND) -> BoolND:
    return eligible_stock >= 0.0
