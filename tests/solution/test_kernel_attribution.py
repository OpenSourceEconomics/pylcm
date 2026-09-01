"""`LCM_LOG_KERNEL_ATTRIBUTION` names the regime-period behind every executed kernel.

A compiled XLA module cannot be attributed to a regime-period from the compile log
alone: identical cores are deduplicated before lowering, so one module serves many
triples and the compile label names only the representative. When a kernel's
allocation is refused, the question "which age, which regime, how many discrete
action branches" is the first one asked, and neither the solve log nor an XLA dump
answers it.

The attribution lines close that gap from the execution side: one line per
regime-period the backward-induction loop actually runs, carrying the discrete
action cardinalities whose product is the branch width.
"""

import logging
import re

import pytest

from tests.test_models.deterministic.discrete import (
    get_model,
    get_params,
)

_N_PERIODS = 3

_ATTR = re.compile(
    r"\[attr\] (?P<regime>\S+) age (?P<age>[-\d.]+) period (?P<period>\d+): "
    r"branches=(?P<branches>\d+) actions=\((?P<actions>[^)]*)\)"
)


def _attribution_lines(caplog) -> list[re.Match[str]]:
    """Return one parsed `[attr]` record per executed regime-period."""
    return [
        match
        for record in caplog.records
        if (match := _ATTR.search(record.getMessage())) is not None
    ]


def _declared_discrete_action_cardinalities(
    *, model, regime_name: str
) -> dict[str, int]:
    """Return `{action name: cardinality}` as the user declared them."""
    regime = model.user_regimes[regime_name]
    return {
        name: len(grid.to_jax())
        for name, grid in regime.actions.items()
        if hasattr(grid, "categories")
    }


def _solve_with_attribution(*, monkeypatch, caplog, enabled: bool):
    """Solve the discrete toy once, returning the model and captured records."""
    if enabled:
        monkeypatch.setenv("LCM_LOG_KERNEL_ATTRIBUTION", "1")
    else:
        monkeypatch.delenv("LCM_LOG_KERNEL_ATTRIBUTION", raising=False)
    model = get_model(n_periods=_N_PERIODS)
    params = get_params(n_periods=_N_PERIODS)
    with caplog.at_level(logging.NOTSET, logger="lcm"):
        model.solve(params=params, log_level="off")
    return model


def test_attribution_is_silent_without_the_env_var(*, monkeypatch, caplog):
    """The instrument costs nothing until it is asked for."""
    _solve_with_attribution(monkeypatch=monkeypatch, caplog=caplog, enabled=False)
    assert _attribution_lines(caplog) == []


def test_every_executed_regime_period_is_attributed(*, monkeypatch, caplog):
    """One line per regime-period the loop runs, and none for the rest.

    `working_life` is declared inactive at the final age, so the count of
    attributed `working_life` periods must fall short of the age grid.
    """
    _solve_with_attribution(monkeypatch=monkeypatch, caplog=caplog, enabled=True)
    attributed = {
        (match["regime"], int(match["period"])) for match in _attribution_lines(caplog)
    }
    working = {period for regime, period in attributed if regime == "working_life"}
    assert working == {0, 1}


def test_the_branch_width_is_the_product_of_declared_discrete_actions(
    *, monkeypatch, caplog
):
    """`branches=` equals the product of the regime's discrete action cardinalities.

    This is the number the memory arithmetic is written against, so it is
    reported rather than inferred from a regime name.
    """
    model = _solve_with_attribution(
        monkeypatch=monkeypatch, caplog=caplog, enabled=True
    )
    declared = _declared_discrete_action_cardinalities(
        model=model, regime_name="working_life"
    )
    expected = 1
    for cardinality in declared.values():
        expected *= cardinality

    widths = {
        int(match["branches"])
        for match in _attribution_lines(caplog)
        if match["regime"] == "working_life"
    }
    assert widths == {expected}


def test_each_discrete_action_is_named_with_its_cardinality(*, monkeypatch, caplog):
    """`actions=` spells out every discrete action, so a change in one is visible."""
    model = _solve_with_attribution(
        monkeypatch=monkeypatch, caplog=caplog, enabled=True
    )
    declared = _declared_discrete_action_cardinalities(
        model=model, regime_name="working_life"
    )

    reported = {
        match["actions"]
        for match in _attribution_lines(caplog)
        if match["regime"] == "working_life"
    }
    expected = ", ".join(f"{name}={n}" for name, n in sorted(declared.items()))
    assert reported == {expected}


def test_the_age_is_reported_alongside_the_period(*, monkeypatch, caplog):
    """Failures are discussed by age, so the line carries it, not just the index."""
    model = _solve_with_attribution(
        monkeypatch=monkeypatch, caplog=caplog, enabled=True
    )
    ages = {
        int(match["period"]): float(match["age"])
        for match in _attribution_lines(caplog)
        if match["regime"] == "working_life"
    }
    expected = {period: float(model.ages.values[period]) for period in ages}
    assert ages == expected


def test_a_deduplicated_module_reports_how_many_triples_it_serves(
    *, monkeypatch, caplog
):
    """The compile log says how many regime-period-core triples share one module.

    Identical cores are lowered once, so a module named in an XLA dump may belong
    to any of them. A count of 1 is a safe attribution; anything higher is not,
    and the log has to say which it is.
    """
    monkeypatch.setenv("LCM_LOG_KERNEL_ATTRIBUTION", "1")
    model = get_model(n_periods=_N_PERIODS)
    params = get_params(n_periods=_N_PERIODS)
    with caplog.at_level(logging.NOTSET, logger="lcm"):
        model.solve(params=params, log_level="off")

    serves = [
        record.getMessage()
        for record in caplog.records
        if "[attr] serves" in record.getMessage()
    ]
    assert serves, "expected a per-module fan-out line for each lowered core"
    assert all(re.search(r"\[attr\] serves \d+ triple", line) for line in serves)


@pytest.mark.parametrize("value", ["0", ""])
def test_falsy_env_values_keep_the_instrument_off(*, monkeypatch, caplog, value):
    """Only an explicit opt-in turns it on."""
    monkeypatch.setenv("LCM_LOG_KERNEL_ATTRIBUTION", value)
    model = get_model(n_periods=_N_PERIODS)
    params = get_params(n_periods=_N_PERIODS)
    with caplog.at_level(logging.NOTSET, logger="lcm"):
        model.solve(params=params, log_level="off")
    assert _attribution_lines(caplog) == []
