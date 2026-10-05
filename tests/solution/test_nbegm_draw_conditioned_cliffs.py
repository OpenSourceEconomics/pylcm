"""NB-EGM save-to-cliff targets use each realised child node's own cliffs.

The source saves `s`; next period's `kind` is drawn; child kind `j` loses a
subsidy at its own liquid cutoff `b_j`. The save-to-cliff targets pull each child
cliff back through the liquid law, so every node of the drawn `kind` contributes
the preimage of *that* child row's cutoff, never the source cell's. This holds
whether the liquid law reads the draw (`slope * s + offset[next_kind]`) or not
(`slope * s + offset[0]`), since in both cases the child's cliff moves with the
drawn row.

The exact preimages come from `_cliff_pullback_reference`, a rational-arithmetic
reference that shares no code with the solver. The two-period value comes from
the same module's closed form. The whole period kernel is also checked against
the scalar direct oracle, which reads each node's cliffs off the solved child
carry rather than re-evaluating the threshold declarations.
"""

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from _lcm.execution.core_program import (
    CoreBuildContext,
    core_program_graph,
    materialize_core_program,
)
from _lcm.solution.nbegm import _cliff_savings_targets
from lcm import DiscreteGrid, LinSpacedGrid, Model
from lcm.typing import ContinuousState, DiscreteState, FloatND
from tests.solution._cliff_pullback_reference import (
    child_cliff_preimages,
    two_period_log_value,
)
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.solution.test_nbegm_direct_oracle import (
    _assert_kernel_agrees_with_oracle as assert_kernel_agrees_with_oracle,
)
from tests.test_models.nbegm_common import (
    make_alive_dead_model,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_indexed_threshold_toy import (
    ConsumerKind,
    gross_income,
    resources,
)
from tests.test_models.nbegm_indexed_threshold_toy import (
    subsidy as kind_indexed_subsidy,
)

BASE_INCOME = 2.0
SUBSIDY = 5.0
DISCOUNT_FACTOR = 0.95


def next_liquid_reading_draw(
    *,
    savings: FloatND,
    next_kind: DiscreteState,
    law_slope: float,
    law_offset: FloatND,
) -> ContinuousState:
    """Liquid wealth next period: scaled savings plus the drawn kind's offset."""
    return law_slope * savings + law_offset[next_kind]


def next_liquid_draw_free(
    *, savings: FloatND, law_slope: float, law_offset: FloatND
) -> ContinuousState:
    """Liquid wealth next period: scaled savings plus the first offset."""
    return law_slope * savings + law_offset[0]


def kind_probabilities(*, kind: DiscreteState) -> FloatND:
    """Next-period kind is a fair coin, whatever the current kind."""
    return jnp.asarray(((0.5, 0.5), (0.5, 0.5)))[kind]


def zero_bequest(*, liquid: ContinuousState) -> FloatND:
    """The terminal regime values remaining wealth at zero."""
    return jnp.zeros_like(liquid)


@lcm.piecewise_affine(
    output="subsidy",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint(threshold="fpl_cliff", kind="jump"),),
)
def kind_invariant_subsidy(
    *,
    gross_income: FloatND,
    subsidy_low: float,
    subsidy_high: float,
    fpl_cliff: float,
) -> FloatND:
    """Lump-sum subsidy with one income cliff shared by both kinds."""
    return jnp.where(gross_income < fpl_cliff, subsidy_high, subsidy_low)


def _build_model(
    *, liquid_law: Callable[..., object], subsidy: Callable[..., object]
) -> Model:
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=31,
        liquid_max=30.0,
        n_consumption=31,
        liquid_grid=LinSpacedGrid(start=0.0, stop=30.0, n_points=31),
        alive_functions={
            "utility": utility,
            "gross_income": gross_income,
            "subsidy": subsidy,
            "resources": resources,
            "savings": savings,
        },
        liquid_law=liquid_law,
        alive_solver=resolve_solver(
            variant="nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=28.0, n_points=16),
        ),
        constraints={},
        extra_states={"kind": DiscreteGrid(category_class=ConsumerKind)},
        extra_state_transitions={
            "kind": lcm.StochasticTransition(func=kind_probabilities)
        },
        dead_functions={"utility": zero_bequest},
    )


def _params(
    *,
    fpl_cliff: float | FloatND,
    law_slope: float,
    law_offset: tuple[float, float],
) -> dict[str, Any]:
    return {
        "alive": {
            "utility": {"crra": 1.0},
            "koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR},
            "gross_income": {"base_income": BASE_INCOME},
            "subsidy": {
                "subsidy_low": 0.0,
                "subsidy_high": SUBSIDY,
                "fpl_cliff": fpl_cliff,
            },
            "next_liquid": {
                "law_slope": law_slope,
                "law_offset": jnp.asarray(law_offset),
            },
            "final_age_alive": 2.0,
        },
    }


def _indexed_params(*, cutoffs: tuple[float, float]) -> dict[str, Any]:
    return _params(
        fpl_cliff=jnp.asarray(cutoffs) + BASE_INCOME,
        law_slope=1.0,
        law_offset=(0.0, 0.1),
    )


def _solved_seam(*, model: Model, params: dict[str, Any]) -> dict[str, Any]:
    """Solve the model and return its period-0 kernel with the bound arguments."""
    kernel, context = ride_along_kernel(
        model=model, params=params, regime_name="alive", period=0
    )
    replay = core_program_graph(kernel=kernel)["replay"]
    materialized = materialize_core_program(
        program=replay, context=CoreBuildContext(**context)
    )
    kwargs = dict(materialized.arguments)
    kwargs.update(getattr(replay.function, "keywords", None) or {})
    assert kernel.cliff_candidates
    assert tuple(kernel.statics.ride_names) == ("kind",)
    return {"kernel": kernel, "kwargs": kwargs}


def _targets(
    *,
    seam: dict[str, Any],
    source_kind: int,
    overrides: dict[str, Any] | None = None,
    jit: bool = False,
) -> np.ndarray:
    """Return the save-to-cliff targets of one source cell, one row per node."""
    kernel = seam["kernel"]
    statics = kernel.statics
    kwargs = {**seam["kwargs"], **(overrides or {})}
    liquid_grid = jnp.asarray(kwargs[statics.liquid_name])
    cell = {"kind": jnp.asarray(source_kind, dtype=jnp.int32)}
    param_pool = {
        key: value
        for key, value in kwargs.items()
        if key not in statics.state_names and key != "next_regime_to_continuation"
    }

    def evaluate() -> FloatND:
        return _cliff_savings_targets(
            continuation_plan=kernel.continuation_plan,
            regime_name="alive",
            statics=statics,
            kwargs=kwargs,
            cell=cell,
            combo_pool={**param_pool, **cell},
            liquid_grid=liquid_grid,
            savings_grid=jnp.asarray(kernel.savings_grid),
            dtype=liquid_grid.dtype,
        )

    raw = jax.jit(evaluate)() if jit else evaluate()
    return np.asarray(raw).reshape(-1, 2 * statics.n_jumps)


@pytest.fixture(scope="module")
def reading_draw_seam() -> dict[str, Any]:
    model = _build_model(
        liquid_law=next_liquid_reading_draw, subsidy=kind_indexed_subsidy
    )
    return _solved_seam(model=model, params=_indexed_params(cutoffs=(9.0, 6.0)))


@pytest.fixture(scope="module")
def draw_free_seam() -> dict[str, Any]:
    model = _build_model(liquid_law=next_liquid_draw_free, subsidy=kind_indexed_subsidy)
    return _solved_seam(model=model, params=_indexed_params(cutoffs=(9.0, 6.0)))


def _overrides(*, seam: dict[str, Any], **values: FloatND) -> dict[str, FloatND]:
    """Replace every flat param ending in `__<name>` (all its qualified aliases)."""
    return {
        key: value
        for name, value in values.items()
        for key in seam["kwargs"]
        if key == name or key.endswith(f"__{name}")
    }


def _exact(value: float | np.floating) -> Fraction:
    return Fraction(float(value))


def _brackets_each_child_preimage(
    *,
    rows: np.ndarray,
    cutoffs: tuple[np.floating, np.floating],
    slope: np.floating,
    offsets: tuple[np.floating, np.floating],
) -> bool:
    """Whether row `j` tightly straddles child kind `j`'s exact preimage.

    The preimage is computed exactly from the stored (rounded) inputs, so the
    check is a structural predicate at either precision: the lower target lies
    strictly below it, the upper one strictly above, and the pair is within a
    few dozen units of roundoff of each other.
    """
    preimages = child_cliff_preimages(
        cutoffs=tuple(_exact(c) for c in cutoffs),
        slope=_exact(slope),
        offsets=tuple(_exact(o) for o in offsets),
    )
    eps = Fraction(float(np.finfo(rows.dtype).eps))
    for row, preimage in zip(rows, preimages, strict=True):
        below, above = _exact(row[0]), _exact(row[1])
        tight = above - below <= 64 * eps * max(Fraction(1), abs(preimage))
        if not (below < preimage < above and tight):
            return False
    return True


@pytest.mark.parametrize("source_kind", [0, 1])
@pytest.mark.parametrize("jit", [False, True])
def test_draw_reading_law_targets_each_child_rows_cliff(
    *, reading_draw_seam: dict[str, Any], source_kind: int, jit: bool
) -> None:
    """With cutoffs (9, 6) and offsets (0, 0.1), the targets straddle 9 and 5.9."""
    rows = _targets(seam=reading_draw_seam, source_kind=source_kind, jit=jit)
    dtype = rows.dtype
    assert _brackets_each_child_preimage(
        rows=rows,
        cutoffs=(dtype.type(9.0), dtype.type(6.0)),
        slope=dtype.type(1.0),
        offsets=(dtype.type(0.0), dtype.type(0.1)),
    ), rows


@pytest.mark.parametrize("source_kind", [0, 1])
def test_draw_free_law_targets_each_child_rows_cliff(
    *, draw_free_seam: dict[str, Any], source_kind: int
) -> None:
    """A law not reading the draw still targets cutoff 9 and cutoff 6."""
    rows = _targets(seam=draw_free_seam, source_kind=source_kind)
    dtype = rows.dtype
    assert _brackets_each_child_preimage(
        rows=rows,
        cutoffs=(dtype.type(9.0), dtype.type(6.0)),
        slope=dtype.type(1.0),
        offsets=(dtype.type(0.0), dtype.type(0.0)),
    ), rows


_MUTATIONS = [
    pytest.param(
        cuts, slope, shift, source_kind, id=f"{cuts}-{slope}-{shift}-k{source_kind}"
    )
    for cuts in ((9.0, 6.0), (6.0, 9.0), (9.0, 9.0))
    for slope in (0.5, 1.0, 2.0)
    for shift in (0.0, 2.0)
    for source_kind in (0, 1)
]


@pytest.mark.parametrize(("cuts", "slope", "shift", "source_kind"), _MUTATIONS)
def test_targets_follow_child_rows_across_scales_and_translations(
    *,
    reading_draw_seam: dict[str, Any],
    cuts: tuple[float, float],
    slope: float,
    shift: float,
    source_kind: int,
) -> None:
    """Every slope, translation and cutoff order lands each node on its own cutoff."""
    dtype = np.asarray(reading_draw_seam["kwargs"]["subsidy__fpl_cliff"]).dtype
    fpl_cliff = np.asarray([c + shift + BASE_INCOME for c in cuts], dtype=dtype)
    offsets = np.asarray([shift, shift + 0.1], dtype=dtype)
    rows = _targets(
        seam=reading_draw_seam,
        source_kind=source_kind,
        overrides=_overrides(
            seam=reading_draw_seam,
            fpl_cliff=jnp.asarray(fpl_cliff),
            law_slope=jnp.asarray(slope, dtype=dtype),
            law_offset=jnp.asarray(offsets),
        ),
    )
    # The child's liquid cutoff is where `liquid + base_income` meets `fpl_cliff`,
    # so it is exactly `fpl_cliff - base_income` in the stored precision's terms.
    base = dtype.type(BASE_INCOME)
    assert _brackets_each_child_preimage(
        rows=rows,
        cutoffs=(fpl_cliff[0] - base, fpl_cliff[1] - base),
        slope=dtype.type(slope),
        offsets=(offsets[0], offsets[1]),
    ), rows


@pytest.mark.parametrize("source_kind", [0, 1])
def test_relabelling_kinds_permutes_the_node_targets(
    *, reading_draw_seam: dict[str, Any], source_kind: int
) -> None:
    """Swapping both kinds' cutoffs and offsets swaps the per-node target rows."""
    kwargs = reading_draw_seam["kwargs"]
    original = _targets(seam=reading_draw_seam, source_kind=source_kind)
    relabelled = _targets(
        seam=reading_draw_seam,
        source_kind=1 - source_kind,
        overrides=_overrides(
            seam=reading_draw_seam,
            fpl_cliff=jnp.asarray(kwargs["subsidy__fpl_cliff"])[::-1],
            law_offset=jnp.asarray(kwargs["alive__next_liquid__law_offset"])[::-1],
        ),
    )
    np.testing.assert_array_equal(relabelled[::-1], original)


def test_kind_invariant_cliff_targets_agree_in_every_source_cell() -> None:
    """With one cliff shared by both kinds, both nodes target the shared cutoff."""
    model = _build_model(
        liquid_law=next_liquid_reading_draw, subsidy=kind_invariant_subsidy
    )
    seam = _solved_seam(
        model=model,
        params=_params(
            fpl_cliff=9.0 + BASE_INCOME, law_slope=1.0, law_offset=(0.0, 0.1)
        ),
    )
    rows = _targets(seam=seam, source_kind=0)
    np.testing.assert_array_equal(rows, _targets(seam=seam, source_kind=1))
    dtype = rows.dtype
    assert _brackets_each_child_preimage(
        rows=rows,
        cutoffs=(dtype.type(9.0), dtype.type(9.0)),
        slope=dtype.type(1.0),
        offsets=(dtype.type(0.0), dtype.type(0.1)),
    ), rows


@pytest.mark.parametrize("kind", [0, 1])
def test_solved_value_reaches_the_child_cliff_supremum(kind: int) -> None:
    """At resources 22 the period-0 value is the closed form's left limit at 5.9.

    The objective `log(22 - s) + 0.475 log(s + 7 [s < 9]) + 0.475 log(s + 7.1
    [s < 5.9])` (subsidy 5 below each child cutoff, base income 2) peaks as `s`
    approaches 5.9 from below. The best savings node, 5.6, is worse by about
    4e-3, more than the child-value interpolation error the tolerance allows.
    """
    model = _build_model(
        liquid_law=next_liquid_reading_draw, subsidy=kind_indexed_subsidy
    )
    values = model.solve(
        params=_indexed_params(cutoffs=(9.0, 6.0)), log_level="off"
    ).values
    expected = two_period_log_value(
        current_resources=22.0,
        cutoffs=(9.0, 6.0),
        offsets=(0.0, 0.1),
        base_income=BASE_INCOME,
        subsidy=SUBSIDY,
        discount_factor=DISCOUNT_FACTOR,
    )
    # Liquid 20 lies above both cliffs, so cash-on-hand there is 20 + 2 = 22.
    value = np.asarray(values[0]["alive"])[kind, 20]
    np.testing.assert_allclose(value, expected, atol=1e-3)


@pytest.mark.parametrize(
    "liquid_law",
    [next_liquid_reading_draw, next_liquid_draw_free],
    ids=["reading_draw", "draw_free"],
)
def test_period_kernel_agrees_with_the_child_carry_oracle(
    liquid_law: Callable[..., object],
) -> None:
    """Value, carry and consumption match the oracle reading the child carry rows.

    The scalar oracle takes each node's cliffs from the solved child carry's
    published breakpoint row rather than from the threshold declarations.
    """
    kernel, context = ride_along_kernel(
        model=_build_model(liquid_law=liquid_law, subsidy=kind_indexed_subsidy),
        params=_indexed_params(cutoffs=(9.0, 6.0)),
        regime_name="alive",
        period=0,
    )
    assert_kernel_agrees_with_oracle(kernel=kernel, context=context)
