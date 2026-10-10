"""Opt-in action partitions: the request, its refusals and the default route.

`ExecutionConfig(action_partitions={"working": 4})` shares the working regime's
action product over four devices. Without a request, or with a count of one,
the ordinary route runs unchanged, bit for bit. A request the route does not
serve is refused at model construction, before anything is lowered, naming
every failed condition. These checks need one device; the multi-device route
itself is exercised in `test_action_partitions_devices.py`.
"""

from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.execution.core_program import core_program_graph
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.placement import PlacementRequest, plan_submesh_placement
from _lcm.regime_building.action_partitioning import action_partition_width_ceilings
from _lcm.typing import DataclassInstance
from lcm import ExecutionConfig, Model
from lcm.exceptions import ExecutionPlanningError
from tests import collective_fixtures
from tests.test_models import dcegm_paper_twin, many_actions, taste_shocks_toy


def _values_bits(*, model: Model, params: dict) -> dict:
    values = model.solve(params=params, log_level="off").values
    return {
        (period, regime): (
            np.asarray(value).dtype,
            np.asarray(value).tobytes(),
            value.sharding,
        )
        for period, by_regime in values.items()
        for regime, value in by_regime.items()
    }


def test_execution_config_action_partitions_default_shares_nothing() -> None:
    """Without a request no regime's actions are shared."""
    assert dict(ExecutionConfig().action_partitions) == {}


def test_execution_config_action_partitions_are_frozen() -> None:
    """The request is held in an immutable mapping."""
    config = ExecutionConfig(action_partitions={"working": 2})

    assert isinstance(config.action_partitions, MappingProxyType)


@pytest.mark.parametrize(
    ("partitions", "error"),
    [
        ({"working": 0}, ValueError),
        ({"working": -2}, ValueError),
        ({"working": True}, TypeError),
        ({"working": 2.0}, BeartypeCallHintParamViolation),
        ({"": 2}, TypeError),
    ],
)
def test_execution_config_refuses_an_unusable_partition_count(
    *, partitions: dict, error: type[Exception]
) -> None:
    """A count is a positive exact integer keyed by a non-empty regime name."""
    with pytest.raises(error):
        ExecutionConfig(action_partitions=partitions)


@pytest.mark.parametrize(
    "config",
    [
        ExecutionConfig(action_partitions={"working": 1}),
        ExecutionConfig(action_partitions={"working": 1, "dead": 1}),
        ExecutionConfig(action_partitions={}),
    ],
    ids=["one_regime_one", "every_regime_one", "empty"],
)
def test_a_count_of_one_solves_bitwise_like_the_ordinary_route(
    config: ExecutionConfig,
) -> None:
    """Values, dtypes and stored shardings equal the default solve exactly."""
    params = many_actions.get_params()

    got = _values_bits(
        model=many_actions.get_model(execution_config=config), params=params
    )

    assert got == _values_bits(model=many_actions.get_model(), params=params)


def test_a_count_of_one_keeps_the_ordinary_kernel_and_placement() -> None:
    """No mesh axis, ceiling or partitioned kernel appears for a count of one."""
    model = many_actions.get_model(
        execution_config=ExecutionConfig(action_partitions={"working": 1})
    )
    reference = many_actions.get_model()
    solution = model._regimes["working"].solution

    assert (
        solution.action_partitions,
        solution.submesh_device_ids,
        dict(model._execution.axis_width_ceilings_by_regime),
        type(
            core_program_graph(kernel=solution.period_kernels[0])["main"].function
        ).__name__,
    ) == (
        1,
        reference._regimes["working"].solution.submesh_device_ids,
        {},
        type(
            core_program_graph(
                kernel=reference._regimes["working"].solution.period_kernels[0]
            )["main"].function
        ).__name__,
    )


def _refusal(*, build: object) -> str:
    with pytest.raises(ExecutionPlanningError) as caught:
        build()  # ty: ignore[call-non-callable]
    return str(caught.value)


def _many_actions(**config: Any) -> Model:
    return many_actions.get_model(execution_config=ExecutionConfig(**config))


def _rebuilt(
    *,
    model: Model,
    regime_id_class: type[DataclassInstance],
    initial_nodes: dict,
    **config: Any,
) -> Model:
    return Model(
        edges=model.edges,
        regimes=dict(model.user_regimes),
        ages=model.ages,
        regime_id_class=regime_id_class,
        initial_nodes=initial_nodes,
        execution_config=ExecutionConfig(**config),
    )


def _collective(**config: Any) -> Model:
    model, _ = collective_fixtures.make_two_stakeholder_model()
    return _rebuilt(
        model=model,
        regime_id_class=collective_fixtures.CoupleRegimeId,
        initial_nodes={0: "couple"},
        **config,
    )


def _folded(**config: Any) -> Model:
    model, _ = collective_fixtures.make_folding_singleton_model()
    return _rebuilt(
        model=model,
        regime_id_class=collective_fixtures.ShockRegimeId,
        initial_nodes={0: "shocked"},
        **config,
    )


def _dcegm(**config: Any) -> Model:
    return _rebuilt(
        model=dcegm_paper_twin.get_model("dcegm"),
        regime_id_class=dcegm_paper_twin.TwinRegimeId,
        initial_nodes={20: ("working_life", "retirement")},
        **config,
    )


_REFUSALS = {
    "unknown_regime": (
        lambda: _many_actions(action_partitions={"nowhere": 2}),
        "no regime is named 'nowhere'",
    ),
    "more_devices_than_visible": (
        lambda: _many_actions(action_partitions={"working": 2}),
        "regime 'working' (2 partitions) needs 2 devices, but the model may use 1",
    ),
    "terminal_regime": (
        lambda: _many_actions(action_partitions={"dead": 2}),
        "regime 'dead' (2 partitions) is terminal",
    ),
    "taste_shocks": (
        lambda: taste_shocks_toy.get_model(
            execution_config=ExecutionConfig(action_partitions={"alive": 2})
        ),
        "regime 'alive' (2 partitions) declares taste shocks",
    ),
    "another_solver": (
        lambda: _dcegm(action_partitions={"working_life": 2}),
        "regime 'working_life' (2 partitions) is solved by",
    ),
    "collective_regime": (
        lambda: _collective(action_partitions={"couple": 2}),
        "regime 'couple' (2 partitions) is a collective regime",
    ),
    "folded_process": (
        lambda: _folded(action_partitions={"shocked": 2}),
        "regime 'shocked' (2 partitions) folds the processes ['wage_shock']",
    ),
    "discrete_sharded_state": (
        lambda: many_actions.get_model(
            typed=True,
            type_at_model_level=True,
            execution_config=ExecutionConfig(
                sharded_states=("pref_type",), action_partitions={"working": 2}
            ),
        ),
        "carries the discrete sharded states ['pref_type']",
    ),
    "fewer_actions_than_devices": (
        lambda: _many_actions(action_partitions={"working": 43}),
        f"has {many_actions.N_ACTIONS} actions, fewer than its 43 devices",
    ),
    "fixed_width_leaves_a_device_without_a_block": (
        lambda: _many_actions(
            action_partitions={"working": 2},
            axis_widths={"action_product": many_actions.N_ACTIONS},
        ),
        (
            f"fixes 'action_product' at {many_actions.N_ACTIONS}, which cuts its "
            f"{many_actions.N_ACTIONS} actions into fewer blocks than its 2 devices"
        ),
    ),
}


@pytest.mark.parametrize("case", list(_REFUSALS))
def test_an_unserved_request_is_refused_before_anything_is_built(case: str) -> None:
    build, expected = _REFUSALS[case]

    assert expected in _refusal(build=build)


def test_a_refusal_names_every_failed_condition_and_the_remedy() -> None:
    message = _refusal(
        build=lambda: _many_actions(action_partitions={"dead": 3, "nowhere": 2})
    )

    assert [
        "regime 'dead' (3 partitions) needs 3 devices" in message,
        "regime 'dead' (3 partitions) is terminal" in message,
        "no regime is named 'nowhere'" in message,
        "Remove the regime from ExecutionConfig.action_partitions" in message,
    ] == [True] * 4


@pytest.mark.parametrize(
    ("count", "fixed", "expected"),
    [
        (2, {}, {"working": {"action_product": many_actions.N_ACTIONS // 2}}),
        (5, {}, {"working": {"action_product": many_actions.N_ACTIONS // 5}}),
        (1, {}, {}),
        (4, {"working": {"action_product": 6}}, {}),
    ],
)
def test_partitioned_regimes_cap_the_action_width_so_every_device_owns_a_block(
    *, count: int, fixed: dict, expected: dict
) -> None:
    model = many_actions.get_model()

    ceilings = action_partition_width_ceilings(
        user_regimes=model.user_regimes,
        action_partitions={"working": count},
        fixed_widths_by_regime=fixed,
    )

    assert {name: dict(axes) for name, axes in ceilings.items()} == expected


def test_a_regime_ceiling_tightens_but_never_loosens_the_model_wide_one() -> None:
    execution = ResolvedExecution(
        device_ids=(0,),
        sharded_states=frozenset(),
        axis_widths=MappingProxyType({}),
        axis_width_ceilings=MappingProxyType({"action_product": 8, "cell": 4}),
        axis_width_ceilings_by_regime=MappingProxyType(
            {
                "tight": MappingProxyType({"action_product": 3}),
                "loose": MappingProxyType({"action_product": 30}),
            }
        ),
        device_memory_bytes=None,
    )

    assert [
        dict(execution.ceilings_for(regime_name=name))
        for name in ("tight", "loose", "other")
    ] == [
        {"action_product": 3, "cell": 4},
        {"action_product": 8, "cell": 4},
        {"action_product": 8, "cell": 4},
    ]


def _request(
    *, name: str, extents: tuple[int, ...] = (), partitions: int = 1
) -> PlacementRequest:
    return PlacementRequest(
        regime_name=name,
        distributed_extents=extents,
        active_periods=(0, 1),
        template_bytes=64,
        action_partitions=partitions,
    )


@pytest.mark.parametrize(
    ("requests", "n_devices", "expected"),
    [
        (
            (_request(name="working", partitions=3), _request(name="dead")),
            4,
            {"working": (0, 1, 2), "dead": (3,)},
        ),
        (
            (
                _request(name="working", extents=(8,), partitions=2),
                _request(name="dead"),
            ),
            8,
            {"working": tuple(range(8)), "dead": (0,)},
        ),
        (
            (
                _request(name="working", extents=(8,), partitions=4),
                _request(name="dead", extents=(8,)),
            ),
            8,
            {"working": tuple(range(8)), "dead": tuple(range(8))},
        ),
        (
            (_request(name="working", extents=(6,), partitions=2),),
            8,
            {"working": tuple(range(6))},
        ),
    ],
    ids=["action_only", "state_by_action", "beside_a_state_mesh", "state_shrinks"],
)
def test_a_partitioned_regime_spans_its_state_mesh_times_its_partitions(
    *, requests: tuple, n_devices: int, expected: dict
) -> None:
    placement = plan_submesh_placement(requests=requests, n_devices=n_devices)

    assert dict(placement.device_ids_by_regime) == expected


def test_placement_never_reduces_an_explicit_partition_count() -> None:
    with pytest.raises(ExecutionPlanningError, match="shares its actions over 5"):
        plan_submesh_placement(
            requests=(_request(name="working", partitions=5),), n_devices=4
        )
