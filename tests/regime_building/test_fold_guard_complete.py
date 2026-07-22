"""Complete the fold-before-gate guard: F1/F2/F3/F4 (fold-review round).

`test_fold_gate_guard.py` pins the INTERIM guard
(`_fail_if_folded_collective_regime_is_gate_target_or_ref`) that rejects a
COLLECTIVE regime folding a shock while being another regime's gated-edge
TARGET or same-period REFERENCE. An outside audit proved that guard's
enumerated prohibition was incomplete in three ways, plus a false positive in
a different (regime-local) fold guard:

- F1 (serious): the guard's reference set was built from `same_period_refs` +
  gated-edge `gate_refs` only — it omitted every `leg.fallback.regime` of a
  `gated_edges` declaration. `get_edge_fold` reads each leg's fallback
  nodewise (`jnp.where(gate, V_target, V_fallback)`) BEFORE integration, so a
  folded fallback regime violates gate-then-integrate exactly like a folded
  target or `gate_refs` reference, but passed construction.
- F2 (serious): the guard skipped every regime with `stakeholders is None`
  (singletons) — but gate-then-integrate does not depend on stakeholder
  count; a SINGLETON folded target/reference is just as unsafe.
- F3 (serious, moderate-confidence): `_wrap_with_fold_reduction` always
  averages arithmetically (`zero_safe_average`), exact only for the LINEAR
  expectation. A non-terminal singleton `GridSearch` regime may declare a
  nonlinear `certainty_equivalent`; `_fold_scope_errors` had no guard for
  this combination (collective regimes already reject nonlinear CE
  unconditionally, so the gap was singleton-only).
- F4 (moderate): `_fold_same_period_roots` walked a SOURCE regime's OWN
  outbound `gated_edges[...].gate` / `gate_refs` as same-period read roots —
  but those functions are compiled and evaluated on the TARGET regime's grid
  (`_attach_gated_edge_folds`), not the source's. If source and target both
  happen to declare a state of the same name, folding the SOURCE's copy was
  falsely rejected merely because the TARGET's gate has an argument of that
  name.

The now-COMPLETE cross-regime endpoint guard
(`_fail_if_folded_regime_is_same_period_endpoint`) covers F1/F2; F4's fix
removes the false-positive source-local check and relies on the endpoint
guard for cross-regime safety instead.
"""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from lcm import DiscreteGrid, NormalIIDProcess, categorical
from lcm.ages import AgeGrid
from lcm.certainty_equivalent import PowerMean
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
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


def _true_gate() -> BoolND:
    return jnp.asarray(1.0) > 0.0


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


_AGES_2P = AgeGrid(start=0, stop=2, step="Y")


# --------------------------------------------------------------------------------------
# F2: a SINGLETON folded regime used as a gated-edge TARGET.
# --------------------------------------------------------------------------------------


def _make_singleton_gated_target_regimes(*, fold: bool) -> dict[str, Regime]:
    """`source` --gated_edges--> `target` (SINGLETON, folds `wage_shock`)."""
    source = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_work},
        gated_edges={
            "target": GatedEdge(
                gate=_true_gate,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(regime="source_terminal", projection={})
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
        states={"wage_shock": _shock(fold=fold)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_work},
    )
    return {"source": source, "source_terminal": source_terminal, "target": target}


def test_folded_singleton_gated_edge_target_is_rejected():
    """F2: a SINGLETON, folded gated-edge TARGET is rejected at model
    processing — gate-then-integrate does not depend on stakeholder count."""
    with pytest.raises(ModelInitializationError, match="gated_edges"):
        process_regimes(
            **_solve_kwargs(
                _make_singleton_gated_target_regimes(fold=True), ages=_AGES_2P
            )
        )


def test_unfolded_singleton_gated_edge_target_still_constructs():
    """Pin: the SAME topology with `fold=False` is untouched."""
    process_regimes(
        **_solve_kwargs(_make_singleton_gated_target_regimes(fold=False), ages=_AGES_2P)
    )


# --------------------------------------------------------------------------------------
# F2: a SINGLETON folded regime used as a `same_period_refs` REFERENCE.
# --------------------------------------------------------------------------------------


def _dummy_constraint(Q_f: FloatND, V_ref: FloatND) -> BoolND:
    return Q_f >= V_ref - 100.0


def _make_singleton_same_period_ref_regimes(*, fold: bool) -> dict[str, Regime]:
    """`reader` (collective) --same_period_refs--> `ref_target` (SINGLETON, folded)."""
    ref_target = Regime(
        transition=None,
        active=lambda age: age < 1,
        states={"wage_shock": _shock(fold=fold)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_work},
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


def test_folded_singleton_same_period_reference_is_rejected():
    """F2: a SINGLETON, folded `same_period_refs` REFERENCE is rejected."""
    with pytest.raises(ModelInitializationError, match="same_period_refs"):
        process_regimes(
            **_solve_kwargs(
                _make_singleton_same_period_ref_regimes(fold=True), ages=_AGES_2P
            )
        )


def test_unfolded_singleton_same_period_reference_still_constructs():
    """Pin: the SAME topology with `fold=False` is untouched."""
    process_regimes(
        **_solve_kwargs(
            _make_singleton_same_period_ref_regimes(fold=False), ages=_AGES_2P
        )
    )


# --------------------------------------------------------------------------------------
# F1: a COLLECTIVE folded regime used as a gated-edge leg FALLBACK — neither
# the edge's target nor a `gate_refs`/`same_period_refs` name.
# --------------------------------------------------------------------------------------


def _make_edge_fallback_regimes(*, fold: bool) -> dict[str, Regime]:
    """`source` --gated_edges--> `target` (plain, unfolded, collective).

    The edge's leg `fallback` names `fallback_regime` (collective, folds
    `wage_shock`) — a role the OLD guard's reference set never enumerated.
    """
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
                        fallback=SamePeriodRef(
                            regime="fallback_regime",
                            stakeholder="f",
                            projection={"wage_shock": lambda: 0.0},
                        ),
                    )
                },
            )
        },
    )
    fallback_regime = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage_shock": _shock(fold=fold)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_f, "utility_m": _u_m},
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_work, "utility_m": _u_work},
    )
    return {"source": source, "fallback_regime": fallback_regime, "target": target}


def test_folded_collective_gated_edge_fallback_is_rejected():
    """F1: a collective, folded gated-edge leg FALLBACK is rejected — the
    edge routing reads it nodewise (`jnp.where(gate, V_target, V_fallback)`)
    before any integration, exactly like a folded target or `gate_refs`
    reference."""
    with pytest.raises(ModelInitializationError, match="gated_edges"):
        process_regimes(
            **_solve_kwargs(_make_edge_fallback_regimes(fold=True), ages=_AGES_2P)
        )


def test_unfolded_collective_gated_edge_fallback_still_constructs():
    """Pin: the SAME topology with `fold=False` is untouched."""
    process_regimes(
        **_solve_kwargs(_make_edge_fallback_regimes(fold=False), ages=_AGES_2P)
    )


# --------------------------------------------------------------------------------------
# F3: `fold=True` combined with a nonlinear `certainty_equivalent` on a
# non-terminal, singleton `GridSearch` regime.
# --------------------------------------------------------------------------------------


def test_fold_with_nonlinear_certainty_equivalent_is_rejected():
    """F3: the fold reduction is the arithmetic `zero_safe_average`, exact
    only for the LINEAR expectation `E[V']`. A nonlinear certainty
    equivalent needs the shock's node axis intact to apply its own
    aggregator, so combining it with `fold=True` must be rejected.

    Collective regimes already reject ANY nonlinear certainty equivalent
    unconditionally (`_fail_if_collective_scope_out_of_bounds`), fold or
    not — this gap was singleton-only.
    """
    with pytest.raises(RegimeInitializationError, match="certainty_equivalent"):
        Regime(
            transition={"terminal": MarkovTransition(_prob_one)},
            active=lambda age: age < 1,
            states={"wage_shock": _shock(fold=True)},
            actions={"work": DiscreteGrid(Work)},
            functions={"utility": _u_work},
            certainty_equivalent=PowerMean(),
        )


def test_fold_without_certainty_equivalent_still_constructs():
    """Pin: the SAME topology with no `certainty_equivalent` is untouched."""
    Regime(
        transition={"terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wage_shock": _shock(fold=True)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_work},
    )


# --------------------------------------------------------------------------------------
# F4: a source regime folding a state whose NAME the target's own gate
# happens to reuse must NOT be falsely rejected — negative control.
# --------------------------------------------------------------------------------------


def test_fold_source_state_name_reused_by_target_gate_is_not_rejected():
    """F4: `source` folds `wage_shock`; `source`'s OUTBOUND gated edge has a
    gate that reads an argument also named `wage_shock` — but that gate is
    compiled and evaluated on the TARGET's own grid
    (`_attach_gated_edge_folds`), not the source's. The two `wage_shock`s are
    unrelated states of different regimes; the source's fold must not be
    rejected merely because the names collide.

    The now-complete cross-regime endpoint guard (F1+F2) still catches a
    genuinely unsafe fold (this scenario is a negative control for exactly
    that reason: `source` — the regime that folds — is not itself a gated-edge
    target or same-period reference here, so no rule should fire).
    """
    source = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wage_shock": _shock(fold=True)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_f},
        gated_edges={
            "target": GatedEdge(
                gate=lambda wage_shock: wage_shock > 0.0,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(regime="source_terminal", projection={})
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
        states={"wage_shock": _shock(fold=False)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_work},
    )
    process_regimes(
        **_solve_kwargs(
            {"source": source, "source_terminal": source_terminal, "target": target},
            ages=_AGES_2P,
        )
    )
