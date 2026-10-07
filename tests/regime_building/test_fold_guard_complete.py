"""The fold-before-gate guard in the roles `test_fold_gate_guard.py` leaves out.

`test_fold_gate_guard.py` pins the cross-regime endpoint guard on a regime
that folds a shock while another regime's gated-edge TARGET or same-period
REFERENCE reads it nodewise. This module covers the remaining roles, plus the
negative controls that bound the prohibition:

- A gated-edge leg FALLBACK. `get_edge_fold` reads each leg's fallback
  nodewise (`jnp.where(gate, V_target, V_fallback)`) BEFORE integration, so a
  folded fallback regime violates gate-then-integrate exactly like a folded
  target or `gate_refs` reference, and is rejected the same way.
- A SINGLETON endpoint. Gate-then-integrate does not depend on stakeholder
  count, so a folded singleton target/reference is as unsafe as a collective
  one and is rejected too.
- `fold=True` alongside a nonlinear `certainty_equivalent` on a non-terminal
  singleton `GridSearch` regime. `_wrap_with_fold_reduction` averages
  arithmetically (`zero_safe_average`), which is exact only for the LINEAR
  expectation, so the combination is rejected. Collective regimes reject a
  nonlinear CE unconditionally, so only singletons can reach this rule.
- The negative control on name reuse: a SOURCE regime folding a state whose
  name the TARGET's gate happens to reuse stays legal. A source's outbound
  `gated_edges[...].gate` / `gate_refs` are compiled and evaluated on the
  TARGET regime's grid (`_attach_gated_edge_folds`), not the source's, so a
  shared argument name says nothing about the safety of the source's own
  fold; cross-regime safety rests on the endpoint guard
  (`_fail_if_folded_regime_is_same_period_endpoint`) instead.
"""

from collections.abc import Mapping
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.regime_law import bind_regime_law
from _lcm.user_regime_validation import validate_regime
from lcm import (
    AgeRange,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    NormalIIDProcess,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentConstraint,
    ValueDependentTransition,
    categorical,
)
from lcm.ages import AgeGrid
from lcm.certainty_equivalent import PowerMean
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.koopmans_aggregation import LinearAggregator
from lcm.transition import StochasticTransition
from lcm.typing import BoolND, DiscreteAction, FloatND, ScalarInt
from tests.conftest import bind_laws, build_prepared_structure


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


def _u_f(*, wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    return work * (5.0 + wage_shock)


def _u_m(*, wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    return work * (3.0 + wage_shock)


def _u_work(work: DiscreteAction) -> FloatND:
    return 0.0 * work


def _u_zero() -> FloatND:
    return jnp.asarray(0.0)


def _no_dissolution_gate(D_target: BoolND) -> BoolND:
    return ~D_target


def _true_gate() -> BoolND:
    return jnp.asarray(1.0) > 0.0


def _solve_kwargs(
    *, regimes: Mapping[str, tuple[Regime, object]], ages: AgeGrid
) -> dict:
    """Process keywords for regimes paired with their laws between regimes."""
    names = list(regimes)
    laws = bind_laws({name: law for name, (_, law) in regimes.items()})
    finalized = finalize_regimes(
        user_regimes={name: regime for name, (regime, _) in regimes.items()},
        laws=laws,
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )
    return {
        "user_regimes": finalized,
        "ages": ages,
        "regime_names_to_ids": MappingProxyType(
            {name: jnp.int32(i) for i, name in enumerate(names)}
        ),
        "enable_jit": False,
        "prepared_structure": build_prepared_structure(
            user_regimes=finalized, laws=laws, ages=ages
        ),
    }


_AGES_2P = AgeGrid(start=0, inclusive_stop=2, step="Y")


def _make_singleton_gated_target_regimes(
    *, fold: bool
) -> dict[str, tuple[Regime, object]]:
    """`source` --gated_edges--> `target` (SINGLETON, folds `wage_shock`)."""
    source = (
        Regime(
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_work},
        ),
        ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "target": ValueDependentTransition(
                        probability=StochasticTransition(func=_prob_one),
                        gate=_true_gate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="source_terminal", projection={}
                                )
                            )
                        },
                    )
                }
            }
        ),
    )
    source_terminal = (
        Regime(
            functions={"utility": _u_zero},
        ),
        None,
    )
    target = (
        Regime(
            states={"wage_shock": _shock(fold=fold)},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_work},
        ),
        None,
    )
    return {"source": source, "source_terminal": source_terminal, "target": target}


def test_folded_singleton_gated_edge_target_is_rejected():
    """A SINGLETON, folded gated-edge TARGET is rejected at model
    processing — gate-then-integrate does not depend on stakeholder count."""
    with pytest.raises(ModelInitializationError, match="gated edge"):
        process_regimes(
            **_solve_kwargs(
                regimes=_make_singleton_gated_target_regimes(fold=True), ages=_AGES_2P
            )
        )


def test_unfolded_singleton_gated_edge_target_still_constructs():
    """Pin: the SAME topology with `fold=False` still constructs."""
    process_regimes(
        **_solve_kwargs(
            regimes=_make_singleton_gated_target_regimes(fold=False), ages=_AGES_2P
        )
    )


def _dummy_constraint(*, Q_f: FloatND, V_ref: FloatND) -> BoolND:
    return Q_f >= V_ref - 100.0


def _make_singleton_same_period_ref_regimes(
    *, fold: bool
) -> dict[str, tuple[Regime, object]]:
    """`reader` (collective) --same_period_refs--> `ref_target` (SINGLETON, folded)."""
    ref_target = (
        Regime(
            states={"wage_shock": _shock(fold=fold)},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_work},
        ),
        None,
    )
    reader = (
        Regime(
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={
                "utility": CollectiveUtility(utilities={"f": _u_work, "m": _u_work})
            },
            constraints={
                "dummy": ValueDependentConstraint(
                    predicate=_dummy_constraint,
                    references={
                        "V_ref": ProjectedRegimeValue(
                            regime="ref_target",
                            projection={"wage_shock": lambda: 0.0},
                        )
                    },
                )
            },
        ),
        ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "reader_terminal": StochasticTransition(func=_prob_one)
                }
            }
        ),
    )
    reader_terminal = (
        Regime(
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={
                "utility": CollectiveUtility(utilities={"f": _u_work, "m": _u_work})
            },
        ),
        None,
    )
    return {
        "ref_target": ref_target,
        "reader": reader,
        "reader_terminal": reader_terminal,
    }


def test_folded_singleton_same_period_reference_is_rejected():
    """A SINGLETON, folded `same_period_refs` REFERENCE is rejected."""
    with pytest.raises(ModelInitializationError, match="same_period_refs"):
        process_regimes(
            **_solve_kwargs(
                regimes=_make_singleton_same_period_ref_regimes(fold=True),
                ages=_AGES_2P,
            )
        )


def test_unfolded_singleton_same_period_reference_still_constructs():
    """Pin: the SAME topology with `fold=False` still constructs."""
    process_regimes(
        **_solve_kwargs(
            regimes=_make_singleton_same_period_ref_regimes(fold=False), ages=_AGES_2P
        )
    )


def _make_edge_fallback_regimes(*, fold: bool) -> dict[str, tuple[Regime, object]]:
    """`source` --gated_edges--> `target` (plain, unfolded, collective).

    The edge's leg `fallback` names `fallback_regime` (singleton, folds
    `wage_shock`), which is a distinct role from the edge's target and from
    either kind of reference name.
    """
    source = (
        Regime(
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_work},
        ),
        ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "target": ValueDependentTransition(
                        probability=StochasticTransition(func=_prob_one),
                        gate=_no_dissolution_gate,
                        routes={
                            "only": StakeholderRoute(
                                target_stakeholder="f",
                                fallback=ProjectedRegimeValue(
                                    regime="fallback_regime",
                                    projection={"wage_shock": lambda: 0.0},
                                ),
                            )
                        },
                    )
                }
            }
        ),
    )
    fallback_regime = (
        Regime(
            states={"wage_shock": _shock(fold=fold)},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_work},
        ),
        None,
    )
    target = (
        Regime(
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={
                "utility": CollectiveUtility(utilities={"f": _u_work, "m": _u_work})
            },
        ),
        None,
    )
    return {"source": source, "fallback_regime": fallback_regime, "target": target}


def test_folded_gated_edge_leg_fallback_is_rejected():
    """A folded gated-edge leg FALLBACK is rejected.

    The edge routing reads the fallback nodewise
    (`jnp.where(gate, V_target, V_fallback)`) before any integration, exactly
    like a folded target or `gate_refs` reference, so a fallback whose node
    axis is already averaged away cannot supply what the route needs.
    """
    with pytest.raises(ModelInitializationError, match="gated_edges"):
        process_regimes(
            **_solve_kwargs(
                regimes=_make_edge_fallback_regimes(fold=True), ages=_AGES_2P
            )
        )


def test_unfolded_gated_edge_leg_fallback_still_constructs():
    """Pin: the SAME topology with `fold=False` still constructs."""
    process_regimes(
        **_solve_kwargs(regimes=_make_edge_fallback_regimes(fold=False), ages=_AGES_2P)
    )


def test_fold_with_nonlinear_certainty_equivalent_is_rejected():
    """`fold=True` beside a nonlinear `certainty_equivalent` is rejected.

    The fold reduction is the arithmetic `zero_safe_average`, exact only for
    the LINEAR expectation `E[V']`. A nonlinear certainty equivalent needs the
    shock's node axis intact to apply its own aggregator.

    Only a singleton regime can reach this rule: a collective regime rejects
    ANY nonlinear certainty equivalent unconditionally
    (`_fail_if_collective_scope_out_of_bounds`), fold or not.
    """
    with pytest.raises(RegimeInitializationError, match="certainty_equivalent"):
        validate_regime(
            Regime(
                states={"wage_shock": _shock(fold=True)},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={"utility": _u_work},
                certainty_equivalent=PowerMean(),
            ),
            law=bind_regime_law(
                ByAge(
                    cases={
                        AgeRange(exclusive_stop=1): {
                            "terminal": StochasticTransition(func=_prob_one)
                        }
                    }
                )
            ),
        )


def test_fold_without_certainty_equivalent_still_constructs():
    """Pin: the SAME topology with no `certainty_equivalent` still constructs."""
    validate_regime(
        Regime(
            states={"wage_shock": _shock(fold=True)},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_work},
        ),
        law=bind_regime_law(
            ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "terminal": StochasticTransition(func=_prob_one)
                    }
                }
            )
        ),
    )


def test_fold_source_state_name_reused_by_target_gate_is_not_rejected():
    """A fold survives the target's gate reusing the folded state's NAME.

    `source` folds `wage_shock`; `source`'s OUTBOUND gated edge has a gate
    that reads an argument also named `wage_shock` — but that gate is compiled
    and evaluated on the TARGET's own grid (`_attach_gated_edge_folds`), not
    the source's. The two `wage_shock`s are unrelated states of different
    regimes, so the source's fold stays legal.

    The negative control for the endpoint guard: `source` — the regime that
    folds — is not itself a gated-edge target or same-period reference here,
    so no rule may fire.
    """
    source = (
        Regime(
            states={"wage_shock": _shock(fold=True)},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_f},
        ),
        ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "target": ValueDependentTransition(
                        probability=StochasticTransition(func=_prob_one),
                        gate=lambda wage_shock: wage_shock > 0.0,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="source_terminal", projection={}
                                )
                            )
                        },
                    )
                }
            }
        ),
    )
    source_terminal = (
        Regime(
            functions={"utility": _u_zero},
        ),
        None,
    )
    target = (
        Regime(
            states={"wage_shock": _shock(fold=False)},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_work},
        ),
        None,
    )
    process_regimes(
        **_solve_kwargs(
            regimes={
                "source": source,
                "source_terminal": source_terminal,
                "target": target,
            },
            ages=_AGES_2P,
        )
    )
