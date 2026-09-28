"""A binding cell ceiling on a collective model narrows the tile and nothing else.

The model is the two-stakeholder couple of `tests.collective_fixtures` over a
wage grid wide enough that the planner's own choice exceeds the ceiling, so
the ceiling binds rather than sitting inactive above the selected width. Each
solve's dispatched widths and compiler reservations are read from the planner
through `CensusRecorder`; the numerical arms compare every published array
byte for byte.
"""

from typing import Any

import numpy as np
import pytest

from lcm import ExecutionConfig, LinSpacedGrid, Model
from lcm.exceptions import ExecutionPlanningError
from tests.collective_fixtures import AGES, CoupleRegimeId, make_two_stakeholder_model
from tests.solution._candidate_census import Census, CensusRecorder, Triple

_N_WAGE = 64
_CELL_CEILING = 8
_BUDGET = 10**8
_COUPLE: Triple = ("couple", 0, "main")
_TERMINAL: Triple = ("couple_terminal", 1, "main")


def _model(*, n_wage: int = _N_WAGE, **execution: Any) -> tuple[Model, Any]:
    """Build the two-stakeholder couple over an `n_wage`-point wage grid."""
    base, params = make_two_stakeholder_model()
    grid = LinSpacedGrid(start=8.0, stop=40.0, n_points=n_wage)
    regimes = {
        name: regime.replace(states={"wage": grid})
        for name, regime in base.user_regimes.items()
    }
    model = Model(
        regimes=regimes,
        ages=AGES,
        regime_id_class=CoupleRegimeId,
        execution_config=ExecutionConfig(**execution),
    )
    return model, params


def _solve(
    *,
    monkeypatch: pytest.MonkeyPatch,
    n_wage: int = _N_WAGE,
    budget: int | None = _BUDGET,
    **execution: Any,
):
    """Solve under one execution config; return the result and the planner census."""
    recorder = CensusRecorder()
    recorder.install(monkeypatch=monkeypatch)
    model, params = _model(n_wage=n_wage, device_memory_bytes=budget, **execution)
    result = model.solve(params=params, log_level="off")
    monkeypatch.undo()
    return result, recorder.census()


def _dispatched_cell_widths(*, census: Census) -> dict[Triple, int]:
    """Read the `cell` width of every dispatched core from its planned executable."""
    return {
        triple: dict(core.tile_widths)["cell"]
        for triple, core in census.selected_cores.items()
    }


def _admitted_reservation(*, census: Census, triple: Triple) -> int:
    """Return the compiler reservation of the candidate the solve dispatched."""
    (row,) = [row for row in census.rows if row.triple == triple and row.admitted]
    return row.reservation


def _published_arrays(result) -> dict[object, np.ndarray]:
    """Collect every array the solve publishes, keyed by where it is published."""
    arrays: dict[object, np.ndarray] = {}
    for period, by_regime in result.values.items():
        for regime, value in by_regime.items():
            arrays[("value", period, regime)] = np.asarray(value)
    for ref, artifact in result.replay_artifacts.items():
        arrays[("replay", ref)] = np.asarray(artifact)
    for ref, artifact in result.auxiliary_artifacts.items():
        arrays[("auxiliary", ref)] = np.asarray(artifact)
    return arrays


def test_unbounded_collective_solve_tiles_cells_wider_than_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a ceiling the budget admits the full cell extent on every core."""
    _, census = _solve(monkeypatch=monkeypatch)

    assert _dispatched_cell_widths(census=census) == dict.fromkeys(
        census.selected_cores, _N_WAGE
    )


def test_ceiling_binds_the_collective_cell_width_on_every_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a ceiling every core dispatches the widest legal width at or below it."""
    _, census = _solve(
        monkeypatch=monkeypatch, axis_width_ceilings={"cell": _CELL_CEILING}
    )

    assert _dispatched_cell_widths(census=census) == dict.fromkeys(
        census.selected_cores, _CELL_CEILING
    )


def _bytes(result) -> dict[object, tuple[tuple[int, ...], np.dtype, bytes]]:
    """Shape, dtype and raw bytes of every published array, for bitwise comparison."""
    return {
        key: (array.shape, array.dtype, array.tobytes())
        for key, array in _published_arrays(result).items()
    }


def test_ceiling_leaves_every_published_collective_array_bitwise_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Values and dissolution flags agree byte for byte with the unbudgeted solve.

    Without a budget the planner tiles cells at the bootstrap width, twice the
    ceiling here, so the two arms differ only in tile width.
    """
    unbounded, _ = _solve(monkeypatch=monkeypatch, budget=None)
    bounded, _ = _solve(
        monkeypatch=monkeypatch,
        budget=None,
        axis_width_ceilings={"cell": _CELL_CEILING},
    )

    assert _bytes(bounded) == _bytes(unbounded)


def _non_float_bytes_and_float_masks(result) -> dict[object, tuple]:
    """Shape, dtype and finite mask of float arrays; raw bytes of every other array."""
    out: dict[object, tuple] = {}
    for key, array in _published_arrays(result).items():
        if np.issubdtype(array.dtype, np.floating):
            out[key] = (array.shape, array.dtype, np.isfinite(array).tobytes())
        else:
            out[key] = (array.shape, array.dtype, array.tobytes())
    return out


def _max_ulp_over_finite_floats(*, left, right) -> int:
    """Largest ULP distance between two results' finite float entries."""
    worst = 0
    right_arrays = _published_arrays(right)
    for key, a in _published_arrays(left).items():
        if not np.issubdtype(a.dtype, np.floating):
            continue
        b = right_arrays[key]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.any():
            worst = max(
                worst,
                int(
                    np.max(
                        np.testing.assert_array_max_ulp(
                            a[mask], b[mask], maxulp=np.iinfo(np.int32).max
                        )
                    )
                ),
            )
    return worst


def test_budgeted_full_extent_arm_matches_the_ceiling_arm_outside_float_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both arms publish the same shapes, dtypes, finite masks and non-float arrays."""
    unbounded, _ = _solve(monkeypatch=monkeypatch)
    bounded, _ = _solve(
        monkeypatch=monkeypatch, axis_width_ceilings={"cell": _CELL_CEILING}
    )

    assert _non_float_bytes_and_float_masks(
        bounded
    ) == _non_float_bytes_and_float_masks(unbounded)


def test_budgeted_full_extent_arm_stays_within_fma_rounding_of_the_ceiling_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every finite published value differs from the ceiling arm by at most 2 ulp.

    One fused versus unfused multiply-add moves a value by one ulp; the bound
    leaves one more for that difference carried into the next period's
    continuation value, and rejects anything larger.
    """
    unbounded, _ = _solve(monkeypatch=monkeypatch)
    bounded, _ = _solve(
        monkeypatch=monkeypatch, axis_width_ceilings={"cell": _CELL_CEILING}
    )

    assert _max_ulp_over_finite_floats(left=bounded, right=unbounded) <= 2


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The full-extent cell tile the budget admits lowers `u + beta * V` with a "
        "fused multiply-add on the CPU backend, while the ceiling-bound tile "
        "rounds the product first, so the continuation period differs by one ulp "
        "on some cells; both arms compute the same equation."
    ),
)
def test_budgeted_full_extent_arm_agrees_bitwise_with_the_ceiling_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget-admitted full-extent tile publishes the same bytes as the ceiling."""
    unbounded, _ = _solve(monkeypatch=monkeypatch)
    bounded, _ = _solve(
        monkeypatch=monkeypatch, axis_width_ceilings={"cell": _CELL_CEILING}
    )

    assert _bytes(bounded) == _bytes(unbounded)


def test_ceiling_publishes_a_stakeholder_value_for_every_wage_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiling narrower keeps the (wage, stakeholder) shape of the stored value."""
    bounded, _ = _solve(
        monkeypatch=monkeypatch, axis_width_ceilings={"cell": _CELL_CEILING}
    )

    assert np.asarray(bounded.values[0]["couple"]).shape == (_N_WAGE, 2)


@pytest.mark.parametrize("triple", [_COUPLE, _TERMINAL])
def test_ceiling_lowers_the_admitted_compiler_reservation(
    *, monkeypatch: pytest.MonkeyPatch, triple: Triple
) -> None:
    """The narrower tile the ceiling selects reserves fewer compiler bytes."""
    _, unbounded = _solve(monkeypatch=monkeypatch)
    _, bounded = _solve(
        monkeypatch=monkeypatch, axis_width_ceilings={"cell": _CELL_CEILING}
    )

    assert _admitted_reservation(census=bounded, triple=triple) < _admitted_reservation(
        census=unbounded, triple=triple
    )


def test_ceiling_narrows_a_fixed_width_that_exceeds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed width above the ceiling is dispatched at the ceiling."""
    _, census = _solve(
        monkeypatch=monkeypatch,
        axis_widths={"cell": 2 * _CELL_CEILING},
        axis_width_ceilings={"cell": _CELL_CEILING},
    )

    assert _dispatched_cell_widths(census=census)[_COUPLE] == _CELL_CEILING


def test_ceiling_binds_per_regime_fixed_widths_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regime fixed above the ceiling is narrowed; one fixed below keeps its width."""
    _, census = _solve(
        monkeypatch=monkeypatch,
        axis_widths={"cell": {"couple": 2 * _CELL_CEILING, "couple_terminal": 4}},
        axis_width_ceilings={"cell": _CELL_CEILING},
    )

    assert _dispatched_cell_widths(census=census) == {
        _COUPLE: _CELL_CEILING,
        _TERMINAL: 4,
        ("couple_terminal", 2, "main"): 4,
    }


def test_non_power_of_two_ceiling_is_dispatched_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ceiling off the power-of-two ladder is itself a legal width and is used."""
    _, census = _solve(
        monkeypatch=monkeypatch, n_wage=48, axis_width_ceilings={"cell": 12}
    )

    assert _dispatched_cell_widths(census=census)[_COUPLE] == 12


def test_ceiling_above_the_extent_leaves_the_full_extent_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ceiling wider than the axis is inert: the full extent is still selected."""
    _, census = _solve(
        monkeypatch=monkeypatch, axis_width_ceilings={"cell": 2 * _N_WAGE}
    )

    assert _dispatched_cell_widths(census=census)[_COUPLE] == _N_WAGE


def test_ceiling_naming_an_axis_no_program_declares_is_refused_at_build() -> None:
    """The refusal names the config field and the unknown axis."""
    with pytest.raises(
        ExecutionPlanningError, match=r"axis_width_ceilings.*'nonesuch'"
    ):
        _model(axis_width_ceilings={"nonesuch": 2})
