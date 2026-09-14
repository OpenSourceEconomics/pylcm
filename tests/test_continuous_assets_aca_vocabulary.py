"""Continuous sharding with unsharded interpolation and carried process states.

The synthetic model isolates phase and coordinate contracts; it is not an ACA
production solve. Run topology witnesses in a fresh eight-device process.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.max_Q_over_a import (
    get_max_Q_over_a,
    get_streaming_max_Q_over_a,
)
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    GridBreakpoint,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Phased,
    PiecewiseLinSpacedGrid,
    Regime,
    RouwenhorstAR1Process,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import FloatND, ScalarInt


@categorical(ordered=False)
class _Preference:
    first: ScalarInt
    second: ScalarInt


@categorical(ordered=False)
class _Decision:
    first: ScalarInt
    second: ScalarInt
    third: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _utility(
    *,
    assets: FloatND,
    aime: FloatND,
    decision: FloatND,
    pref_type: FloatND,
    shock: FloatND,
    persistent: FloatND,
    pension: FloatND,
) -> FloatND:
    return (
        -100
        - (assets - 5) ** 2 / 32
        - aime**2 / 8
        - 20 * (decision - pref_type) ** 2
        + shock / 8
        + persistent / 16
        + pension / 4
    )


def _terminal(*, assets: FloatND, pref_type: FloatND) -> FloatND:
    return -40 - (assets - 5) ** 2 / 8 + pref_type


def _impute(aime: FloatND) -> FloatND:
    return aime / 2


def _carry(pension: FloatND) -> FloatND:
    return pension + 2


def _next_assets(*, assets: FloatND, decision: FloatND) -> FloatND:
    return 15 - assets + (decision - 1) / 4


def _next_aime(aime: FloatND) -> FloatND:
    return 3 - aime / 2


def _next_regime(age: FloatND) -> FloatND:
    return jnp.where(age < 1, _RegimeId.working, _RegimeId.dead)


def _model(
    *,
    sharded: bool = True,
    fold: bool = False,
    widths: tuple[int, int] = (1, 1),
    state_overrides: dict[str, Any] | None = None,
    extra_shard: bool = False,
    budget: int = 2**30,
) -> Model:
    devices = tuple(device.id for device in jax.devices()[:8]) if sharded else (0,)
    return Model(
        regimes={
            "working": Regime(
                active=lambda age: age < 2,
                transition=_next_regime,
                actions={"decision": DiscreteGrid(_Decision)},
                functions={"utility": _utility},
                state_transitions={
                    "assets": _next_assets,
                    "aime": _next_aime,
                    "pref_type": fixed_transition("pref_type"),
                    "pension": _carry,
                },
            ),
            "dead": Regime(
                active=lambda age: age == 2,
                transition=None,
                functions={"utility": _terminal},
                states={"pension": None},
            ),
        },
        states={
            "assets": LinSpacedGrid(start=-4, stop=19, n_points=24),
            "aime": PiecewiseLinSpacedGrid(
                start=0,
                stop=4,
                breakpoints=(GridBreakpoint(value=1),),
                points_per_segment=(2, 2),
            ),
            "pref_type": DiscreteGrid(_Preference),
            "shock": NormalIIDProcess(
                n_points=3,
                gauss_hermite=True,
                mu=0.0,
                sigma=1.0,
                fold=fold,
            ),
            "persistent": RouwenhorstAR1Process(
                n_points=3,
                rho=0.5,
                sigma=1.0,
                mu=0.0,
            ),
            "pension": Phased(
                solve=_impute,
                simulate=LinSpacedGrid(start=0, stop=20, n_points=2),
            ),
            **(state_overrides or {}),
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(
            devices=devices,
            sharded_states=(("assets", "pref_type") if extra_shard else ("assets",))
            if sharded
            else (),
            axis_widths={"action_product": widths[0], "cell": widths[1], "subject": 32},
            device_memory_bytes=budget,
            simulation_chunk_policy="legacy",
        ),
    )


def test_assets_sharding_keeps_effective_phase_axes() -> None:
    """Piecewise AIME and unfolded nodes are solve axes; carried pension is not."""
    model = _model()
    assert model._regimes["working"].solution.state_names == (
        "pref_type",
        "shock",
        "persistent",
        "assets",
        "aime",
    )
    assert model._regimes["dead"].solution.state_names == ("pref_type", "assets")
    aime_grid = model.user_regimes["working"].states["aime"]
    assert isinstance(aime_grid, PiecewiseLinSpacedGrid)
    np.testing.assert_array_equal(
        aime_grid.to_jax(),
        np.array(
            [
                0.0,
                np.nextafter(
                    np.float64(1) if jax.config.jax_enable_x64 else np.float32(1),
                    np.float64(0) if jax.config.jax_enable_x64 else np.float32(0),
                ),
                1.0,
                4.0,
            ]
        ),
    )
    shock_grid = model.user_regimes["working"].states["shock"]
    assert isinstance(shock_grid, NormalIIDProcess)
    assert shock_grid.fold is False


@pytest.mark.parametrize(
    "unsupported", ["fold", "runtime_process", "extra_linear", "second_shard"]
)
def test_unsupported_compositions_refuse_before_solve(*, unsupported: str) -> None:
    """Unvalidated folded, runtime and multiple-shard compositions stay refused."""

    overrides = {}
    if unsupported == "runtime_process":
        overrides["shock"] = NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0)
    if unsupported == "extra_linear":
        overrides["aime"] = LinSpacedGrid(start=0, stop=4, n_points=4)
    with pytest.raises(ExecutionPlanningError, match="Continuous sharding"):
        _model(
            fold=unsupported == "fold",
            state_overrides=overrides,
            extra_shard=unsupported == "second_shard",
        )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("width", [1, 5, 24])
def test_middle_assets_axis_retains_named_coordinates(
    *, streamed: bool, width: int
) -> None:
    """Dense and streamed tiling restore a sharded axis between named coordinates."""

    def q(
        *,
        next_regime_to_V_arr: Any,
        pref_type: Any,
        assets: Any,
        aime: Any,
        decision: Any,
    ) -> tuple:
        del next_regime_to_V_arr
        return -4000.0 + 100 * pref_type + 10 * assets + aime - (
            decision - 1
        ) ** 2, jnp.asarray(pref_type >= 0)

    builder = get_streaming_max_Q_over_a if streamed else get_max_Q_over_a
    function = builder(
        Q_and_F=q,
        batch_sizes=dict.fromkeys(("pref_type", "assets", "aime"), 0),
        action_names=("decision",),
        state_names=("pref_type", "assets", "aime"),
        cell_width_keyword="cell_width",
        untiled_state_names=("assets",),
        **({"action_width_keyword": "action_width"} if streamed else {}),
    )
    got = jax.jit(
        function,
        static_argnames=("cell_width", "action_width") if streamed else ("cell_width",),
    )(
        next_regime_to_V_arr={},
        pref_type=jnp.arange(2),
        assets=jnp.arange(24),
        aime=jnp.array([0, 1, 4]),
        decision=jnp.arange(3),
        cell_width=width,
        **({"action_width": 2} if streamed else {}),
    )
    expected = (
        -4000
        + 100 * np.arange(2)[:, None, None]
        + 10 * np.arange(24)[None, :, None]
        + np.array([0, 1, 4])[None, None, :]
    )
    np.testing.assert_array_equal(got, expected)
