"""Executable spec for the collective-regimes extension (DRAFT / WIP).

This file is the *executable specification* of the "collective regimes"
extension proposed in the design doc `pylcm-extension-collective-regimes.md`
(v2.1) and the tracking issue `pylcm-issue-collective-regimes.md`. It is NOT a
working feature: the numerics (E1-E4) are not implemented. The tests split into
two groups:

* Pinning tests (PASS today) — they nail down the current honest state: the
  `Regime.stakeholders` API surface exists, declaring it raises
  `NotImplementedError`, and the default singleton (`stakeholders=None`) path is
  untouched.

* Target-behavior tests (`xfail`, `strict=False`) — they encode what a
  collective regime must do once E1-E4 land. Written against the real
  construction API, they fail today (construction rejects a stakeholder-valued
  regime) and will start passing, one by one, as the follow-up PRs implement the
  numerics. `strict=False` so an implemented-but-not-yet-un-xfailed test does not
  turn the suite red.

Design-doc section references are on each target-behavior test.
"""

import jax.numpy as jnp
import pytest

from lcm import DiscreteGrid, LinSpacedGrid, categorical
from lcm.regime import Regime
from lcm.typing import (
    ContinuousAction,
    DiscreteAction,
    FloatND,
    ScalarInt,
)

# ----------------------------------------------------------------------------------
# Shared building blocks (a stripped-down couples problem)
# ----------------------------------------------------------------------------------


@categorical(ordered=True)
class LaborSupply:
    do_not_work: ScalarInt
    work: ScalarInt


def _utility_f(
    consumption: ContinuousAction,
    labor_supply_f: DiscreteAction,
    match_quality: float,
) -> FloatND:
    """Wife's per-stakeholder felicity (illustrative)."""
    return (
        jnp.log(consumption)
        - 0.3 * (labor_supply_f == LaborSupply.work)
        + match_quality
    )


def _utility_m(
    consumption: ContinuousAction,
    labor_supply_m: DiscreteAction,
    match_quality: float,
) -> FloatND:
    """Husband's per-stakeholder felicity (illustrative)."""
    return (
        jnp.log(consumption)
        - 0.5 * (labor_supply_m == LaborSupply.work)
        + match_quality
    )


_WEALTH = LinSpacedGrid(start=1, stop=10, n_points=5)
_CONSUMPTION = LinSpacedGrid(start=1, stop=5, n_points=5)


def _build_married_regime() -> Regime:
    """Construct a two-stakeholder `married` regime via the real API surface.

    The concrete declaration of the household scalarization `O({Q^s})` and the
    value-aware mask is still an open design question (design doc §7, issue
    open-questions 2-3); here the two per-stakeholder utilities are supplied as
    named functions and `stakeholders` names the value axis. Today this raises
    `NotImplementedError` at construction — which is exactly why the tests that
    call it are marked `xfail`.
    """
    return Regime(
        transition=None,  # terminal, keeps the spec minimal
        stakeholders=("f", "m"),
        states={"wealth": _WEALTH},
        actions={
            "labor_supply_f": DiscreteGrid(LaborSupply),
            "labor_supply_m": DiscreteGrid(LaborSupply),
            "consumption": _CONSUMPTION,
        },
        functions={
            # Per-stakeholder utilities (E1). One `utility` today; a stakeholder
            # axis tomorrow.
            "utility_f": _utility_f,
            "utility_m": _utility_m,
        },
    )


# ----------------------------------------------------------------------------------
# Pinning tests — PASS today (current honest state)
# ----------------------------------------------------------------------------------


def test_declaring_stakeholders_raises_not_implemented():
    """Constructing a stakeholder-valued regime raises `NotImplementedError`.

    Pins the current honest state: the API surface is real, the numerics are
    not. The error message must point at the design doc so a user is not left
    guessing.
    """
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        Regime(
            transition=None,
            stakeholders=("f", "m"),
            states={"wealth": _WEALTH},
            actions={"consumption": _CONSUMPTION},
            functions={"utility": _utility_f},
        )


def test_singleton_default_is_untouched():
    """The default `stakeholders=None` path constructs exactly as before.

    A regime that does not declare stakeholders must be byte-for-byte the
    current behavior: `stakeholders is None`, and construction succeeds without
    entering the not-yet-implemented branch.
    """

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    regime = Regime(
        transition=None,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        functions={"utility": utility},
    )
    assert regime.stakeholders is None


def test_stakeholders_field_default_is_none():
    """`stakeholders` defaults to `None` (the singleton) when omitted."""
    field = Regime.__dataclass_fields__["stakeholders"]
    assert field.default is None


# ----------------------------------------------------------------------------------
# Target-behavior tests — xfail until E1-E4 land
# ----------------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="collective regimes not yet implemented; see design doc (E1)",
    strict=False,
)
def test_two_stakeholder_values_differ():
    """E1: a two-stakeholder regime yields two distinct value arrays.

    At the common household argmax, the wife's and husband's per-stakeholder
    values are read off separately (`V^s = Q^s(x, a*)`). Because their felicities
    differ (different disutility of work), the two value arrays must not be
    identical. See design doc §2 (E1).
    """
    regime = _build_married_regime()
    # Target API: the solved regime exposes one value array per stakeholder.
    values = regime.solve_period_values()  # ty: ignore[unresolved-attribute]
    assert not jnp.allclose(values["f"], values["m"])


@pytest.mark.xfail(
    reason="collective regimes not yet implemented; see design doc (E2)",
    strict=False,
)
def test_value_aware_feasibility_reads_reference_value():
    """E2: the action mask compares Q^s against a same-period reference value.

    The married participation set is `Q^j(x, a) >= V^j(outside_j) - Delta_j`, so
    the mask must read a *same-period* single-regime reference value at the
    matched shock realization — it can no longer be computed before Q. The solve
    must also expose an explicit divorce flag `D = 1[mask empty]`, distinct from
    a numeric -inf value. See design doc §2 (E2).
    """
    regime = _build_married_regime()
    result = regime.solve_period_values()  # ty: ignore[unresolved-attribute]
    # Target API: a boolean divorce flag alongside the per-stakeholder values,
    # never inferred from V == -inf.
    assert result.divorce_flag.dtype == jnp.bool_


@pytest.mark.xfail(
    reason="collective regimes not yet implemented; see design doc (E3')",
    strict=False,
)
def test_mutual_consent_gate():
    """E3': the singles->married edge forms a marriage only by mutual consent.

    The gated edge object folds `E_eps[ kappa*V_married + (1-kappa)*V_single ]`
    where the consent gate `kappa` is `1` iff `V^{jM}_{t+1} > V^j_{t+1}` for BOTH
    stakeholders (strict, no slack). A candidate marriage that clears only one
    partner's outside option must NOT form. See design doc §2 (E3').
    """
    regime = _build_married_regime()
    # Target API: the edge gate is a callable reading both stakeholders' values
    # and returning a per-node acceptance indicator.
    gate = regime.consent_gate  # ty: ignore[unresolved-attribute]
    v_married = {"f": jnp.array([1.0, 0.0]), "m": jnp.array([1.0, 1.0])}
    v_single = {"f": jnp.array([0.5, 0.5]), "m": jnp.array([0.5, 0.5])}
    accepted = gate(v_married=v_married, v_single=v_single)
    # Node 0: both clear -> accept. Node 1: wife does not clear -> reject.
    assert accepted[0]
    assert not accepted[1]


@pytest.mark.xfail(
    reason="collective regimes not yet implemented; see design doc (E4)",
    strict=False,
)
def test_simulate_value_router_routes_on_realized_values():
    """E4: the simulator routes regimes by recomputed values, not by Phi(x,a).

    At simulation, the router draws candidate realizations, recomputes the
    candidate regimes' per-stakeholder values at the realized point, evaluates
    the same gates as E3', then routes and discards the losing candidate. See
    design doc §2 (E4).
    """
    regime = _build_married_regime()
    router = regime.simulate_value_router  # ty: ignore[unresolved-attribute]
    assert callable(router)
