"""F3 interim guard: `fold=True` on a regime a gate reads nodewise.

`fold=True` integrates a shock's node axis out of a regime's stored V — and,
for a collective regime, collapses its dissolution flag `D` by `jnp.any` —
immediately after that regime's OWN period solve
(`_wrap_with_fold_reduction` in `regime_building/max_Q_over_a.py`). E3'
gated-edge routing and E2 same-period reads both require gate-THEN-integrate:
each realized shock node must be routed through its own gate / consent
decision before any node is averaged away. If the folding regime is itself
read nodewise by ANOTHER regime's gate (as a gated-edge target) or same-period
reference (`same_period_refs` / a gate's `gate_refs`), that ordering is
violated: the reader only ever sees the already-averaged V and the
already-`jnp.any`-reduced D.

`_validate_fold_declarations` (regime-local) already rejects a regime's own
same-period gate / value-constraint reading a fold name IT declares; this
module pins the cross-regime gap the audit found — a DIFFERENT regime's gate
reading a folded TARGET or REFERENCE — which is caught by
`_fail_if_folded_regime_is_same_period_endpoint` in
`regime_building/processing.py`.

A follow-up audit found this collective-only guard's enumerated prohibition
was incomplete (it omitted gated-edge leg fallbacks, and skipped singleton
folding regimes entirely) plus a false positive in a different, regime-local
fold guard; see `test_fold_guard_complete.py` for the completed guard's
coverage (F1/F2/F4) and the singleton-vs-collective negative controls this
module's cases don't exercise.
"""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from lcm import DiscreteGrid, NormalIIDProcess, categorical
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, DiscreteAction, FloatND, ScalarInt


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


def _shock(*, fold: bool, n_points: int = 3, sigma: float = 1.0) -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=n_points, gauss_hermite=True, mu=0.0, sigma=sigma, fold=fold
    )


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _u_f(wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    return work * (5.0 + wage_shock)


def _u_m(wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    return work * (3.0 + wage_shock)


def _u_work(work: DiscreteAction) -> FloatND:
    return 0.0 * work


def _u_zero() -> FloatND:
    return jnp.asarray(0.0)


def _no_dissolution_gate(D_target: BoolND) -> BoolND:
    return ~D_target


def _solve_kwargs(regimes: dict[str, Regime], *, ages: AgeGrid) -> dict:
    names = list(regimes)
    return {
        "user_regimes": finalize_regimes(user_regimes=regimes, derived_categoricals={}),
        "ages": ages,
        "regime_names_to_ids": MappingProxyType(
            {name: jnp.int32(i) for i, name in enumerate(names)}
        ),
        "enable_jit": False,
    }


# --------------------------------------------------------------------------------------
# Case 1: the folding regime is the TARGET of another regime's gated_edges.
# --------------------------------------------------------------------------------------

_AGES_2P = AgeGrid(start=0, stop=2, step="Y")


def _make_gated_target_regimes(*, fold: bool) -> dict[str, Regime]:
    """`source` --gated_edges--> `target` (collective, folds `wage_shock`)."""
    source = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_work},
        gated_edges={
            "target": GatedEdge(
                gate=_no_dissolution_gate,
                legs={
                    "only": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(regime="source_terminal", projection={}),
                    )
                },
            )
        },
    )
    source_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": _u_zero},
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage_shock": _shock(fold=fold)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_f, "utility_m": _u_m},
    )
    return {"source": source, "source_terminal": source_terminal, "target": target}


def test_folded_collective_gated_edge_target_is_rejected():
    """A collective, folded gated-edge TARGET is rejected at model processing."""
    with pytest.raises(ModelInitializationError, match="gated_edges"):
        process_regimes(
            **_solve_kwargs(_make_gated_target_regimes(fold=True), ages=_AGES_2P)
        )


def test_unfolded_collective_gated_edge_target_still_constructs():
    """Pin: the SAME topology with `fold=False` is untouched (byte-identical path)."""
    process_regimes(
        **_solve_kwargs(_make_gated_target_regimes(fold=False), ages=_AGES_2P)
    )


# --------------------------------------------------------------------------------------
# Case 2a: the folding regime is a `same_period_refs` REFERENCE.
# --------------------------------------------------------------------------------------


def _dummy_constraint(Q_f: FloatND, V_ref: FloatND) -> BoolND:
    return Q_f >= V_ref - 100.0


def _make_same_period_ref_regimes(*, fold: bool) -> dict[str, Regime]:
    """`reader` (collective) --same_period_refs--> `ref_target` (collective, folded)."""
    ref_target = Regime(
        transition=None,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage_shock": _shock(fold=fold)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_f, "utility_m": _u_m},
    )
    reader = Regime(
        transition={"reader_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_work, "utility_m": _u_work},
        value_constraints={"dummy": _dummy_constraint},
        same_period_refs={
            "V_ref": SamePeriodRef(
                regime="ref_target",
                stakeholder="f",
                projection={"wage_shock": lambda: 0.0},
            )
        },
    )
    reader_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_work, "utility_m": _u_work},
    )
    return {
        "ref_target": ref_target,
        "reader": reader,
        "reader_terminal": reader_terminal,
    }


def test_folded_collective_same_period_reference_is_rejected():
    """A collective, folded `same_period_refs` REFERENCE is rejected at model
    processing."""
    with pytest.raises(ModelInitializationError, match="same_period_refs"):
        process_regimes(
            **_solve_kwargs(_make_same_period_ref_regimes(fold=True), ages=_AGES_2P)
        )


def test_unfolded_collective_same_period_reference_still_constructs():
    """Pin: the SAME topology with `fold=False` is untouched."""
    process_regimes(
        **_solve_kwargs(_make_same_period_ref_regimes(fold=False), ages=_AGES_2P)
    )


# --------------------------------------------------------------------------------------
# Case 2b: the folding regime is referenced via a gate's `gate_refs`.
# --------------------------------------------------------------------------------------


def _make_gate_refs_regimes(*, fold: bool) -> dict[str, Regime]:
    """`source`'s gate reads `gate_refs['V_ref']` -> `ref_target` (collective, folded).

    `target` (the gated-edge TARGET) is a plain, unfolded collective regime —
    isolates case 2b (the `gate_refs` reference) from case 1 (the edge target).
    """
    source = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_work},
        gated_edges={
            "target": GatedEdge(
                gate=lambda V_ref: V_ref > 0.0,
                legs={
                    "only": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(regime="source_terminal", projection={}),
                    )
                },
                gate_refs={
                    "V_ref": SamePeriodRef(
                        regime="ref_target",
                        stakeholder="f",
                        projection={"wage_shock": lambda: 0.0},
                    )
                },
            )
        },
    )
    source_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": _u_zero},
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_work, "utility_m": _u_work},
    )
    ref_target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage_shock": _shock(fold=fold)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_f, "utility_m": _u_m},
    )
    return {
        "source": source,
        "source_terminal": source_terminal,
        "target": target,
        "ref_target": ref_target,
    }


def test_folded_collective_gate_ref_reference_is_rejected():
    """A collective, folded `gate_refs` REFERENCE is rejected at model processing."""
    with pytest.raises(ModelInitializationError, match=r"gate_refs|same_period_refs"):
        process_regimes(
            **_solve_kwargs(_make_gate_refs_regimes(fold=True), ages=_AGES_2P)
        )


def test_unfolded_collective_gate_ref_reference_still_constructs():
    """Pin: the SAME topology with `fold=False` is untouched."""
    process_regimes(**_solve_kwargs(_make_gate_refs_regimes(fold=False), ages=_AGES_2P))


# --------------------------------------------------------------------------------------
# Negative control: a folded collective regime that is NEITHER a gated-edge
# target NOR a same-period reference stays allowed.
# --------------------------------------------------------------------------------------

_AGES_1P = AgeGrid(start=0, stop=1, step="Y")


def test_untargeted_unreferenced_folded_collective_regime_still_constructs():
    """A collective regime may still fold a shock when nothing gates into it."""
    couple = Regime(
        transition=None,
        stakeholders=("f", "m"),
        states={"wage_shock": _shock(fold=True)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_f, "utility_m": _u_m},
    )
    process_regimes(**_solve_kwargs({"couple": couple}, ages=_AGES_1P))
