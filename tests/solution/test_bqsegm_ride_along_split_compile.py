"""The ride-along BQSEGM solve splits into two independently-compiled cores.

The continuation fan-out (regime transition, child reads, stochastic integration,
interpolation) and the EGM/envelope math compile as two separate XLA programs, so
neither core ever sees the other's instruction graph. Splitting them must not move
the solved value function: both the non-interval ride-along regime and the
next-asset-cliff interval regime keep their brute agreement to tolerance.
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax
import numpy as np

from _lcm.solution.backward_induction import _build_continuation_templates
from lcm import Model
from tests.test_models import bqsegm_next_asset_cliff_toy as cliff_toy
from tests.test_models import bqsegm_ride_along_toy as ride_toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)
_MEDICAID_LIMIT = 12.0
_AWAY_FROM_CLIFF = _INTERIOR & (np.abs(_LIQUID - _MEDICAID_LIMIT) > 0.4)


def _build_ride_model(variant: str, *, n_consumption: int = 120) -> Model:
    return ride_toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )


def _build_cliff_model(variant: str, *, n_consumption: int = 120) -> Model:
    return cliff_toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )


def _ride_along_kernel(
    model: Model, *, params: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Return a representative ride-along period kernel and its lowering context."""
    flat_params = model._process_params(params)
    next_regime_to_V_arr, next_regime_to_egm_carry = _build_continuation_templates(
        regimes=model._regimes, flat_params=flat_params
    )
    regime = model._regimes["alive"]
    period = regime.active_periods[len(regime.active_periods) // 2]
    kernel = regime.solution.period_kernels[period]
    state_action_space = regime.solution.state_action_space(
        regime_params=flat_params["alive"],
    )
    return kernel, {
        "state_action_space": state_action_space,
        "next_regime_to_V_arr": next_regime_to_V_arr,
        "next_regime_to_egm_carry": next_regime_to_egm_carry,
        "flat_params": flat_params,
        "period": period,
        "ages": model.ages,
    }


def _hlo_instruction_count(core: Callable, lower_args: Mapping[str, object]) -> int:
    """Lower a core and count HLO instruction lines in its main computation."""
    hlo_module = jax.jit(core).lower(**dict(lower_args)).compiler_ir(dialect="hlo")
    assert hlo_module is not None
    hlo_text = hlo_module.as_hlo_text()
    return sum(1 for line in hlo_text.splitlines() if " = " in line)


def test_ride_along_kernel_exposes_continuation_and_envelope_cores():
    """The ride-along kernel splits its solve into a `continuation` and an
    `envelope` core, each a distinct compiled program."""
    model = _build_ride_model("bqsegm")
    kernel, _ = _ride_along_kernel(model, params=ride_toy.build_params())
    assert set(kernel.cores()) == {"continuation", "envelope"}


def test_split_partitions_the_solve_into_two_asymmetric_cores():
    """The continuation and envelope cores partition the solve asymmetrically: the
    lightweight half lowers to a small fraction of the heavy half's HLO, so the
    expensive half compiles as its own XLA program without the other's instructions.

    Which half dominates is model-dependent — on this small toy the EGM/envelope
    upper-envelope math is the heavy half (the continuation reads a terminal carry
    with no stochastic fan-out); at production scale the continuation fan-out (regime
    transition, stochastic integration, child interpolation) dominates instead. The
    boundary property the split guarantees is the asymmetry, in either direction.
    """
    model = _build_ride_model("bqsegm")
    kernel, ctx = _ride_along_kernel(model, params=ride_toy.build_params())
    cores = kernel.cores()
    cont_args = kernel.build_lower_args(core_key="continuation", **ctx)
    env_args = kernel.build_lower_args(core_key="envelope", **ctx)
    cont_ops = _hlo_instruction_count(cores["continuation"], cont_args)
    env_ops = _hlo_instruction_count(cores["envelope"], env_args)
    assert min(cont_ops, env_ops) < 0.25 * max(cont_ops, env_ops), (
        cont_ops,
        env_ops,
    )


def test_split_preserves_ride_along_value_against_brute():
    """The non-interval ride-along solve keeps its brute agreement after the
    continuation/envelope split, in every `kind` slice at every working age."""
    bqsegm = _build_ride_model("bqsegm").solve(
        params=ride_toy.build_params(), log_level="off"
    )
    brute = _build_ride_model("brute", n_consumption=1500).solve(
        params=ride_toy.build_params(), log_level="off"
    )
    for period in brute:
        if "alive" not in brute[period] or "alive" not in bqsegm[period]:
            continue
        brute_v = np.asarray(brute[period]["alive"])
        bqsegm_v = np.asarray(bqsegm[period]["alive"])
        for kind in range(brute_v.shape[0]):
            np.testing.assert_allclose(
                bqsegm_v[kind, _INTERIOR],
                brute_v[kind, _INTERIOR],
                atol=2e-2,
                rtol=5e-3,
                err_msg=f"period={period} kind={kind}",
            )


def test_split_preserves_next_asset_cliff_value_against_brute():
    """The interval (next-asset-cliff) solve keeps its brute agreement after the
    continuation/envelope split, away from the one cell straddling the cliff."""
    bqsegm = _build_cliff_model("bqsegm").solve(
        params=cliff_toy.build_params(), log_level="off"
    )
    brute = _build_cliff_model("brute", n_consumption=1500).solve(
        params=cliff_toy.build_params(), log_level="off"
    )
    terminal_adjacent = max(p for p in brute if "alive" in brute[p])
    brute_v = np.asarray(brute[terminal_adjacent]["alive"])
    bqsegm_v = np.asarray(bqsegm[terminal_adjacent]["alive"])
    for kind in range(brute_v.shape[0]):
        np.testing.assert_allclose(
            bqsegm_v[kind, _AWAY_FROM_CLIFF],
            brute_v[kind, _AWAY_FROM_CLIFF],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"kind={kind}",
        )
