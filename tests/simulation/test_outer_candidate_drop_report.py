"""Dropped outer candidates are announced once per regime-period.

N-NB-EGM replay recovers the outer action from the target the solve retained. At
a realized state that action can reach a stock outside the outer state's declared
domain, where no value function exists; the candidate is then dropped from the
affected subjects' choice sets rather than published.

Dropping silently would hide a shrunken choice set, so the drop is reported --
aggregated to one message carrying the counts, never one per candidate. The
public log level decides what the report does, and `"off"` still drops.
"""

import logging

import jax.numpy as jnp
import pytest

from _lcm.simulation.simulate import (
    _announce_dropped_outer_candidates,
    map_coordinates,
)
from _lcm.utils.logging import get_logger
from lcm.exceptions import UnrepresentableOuterCandidateError


def _masks():
    """Three live candidates for two subjects, two of the six dropped."""
    live = jnp.ones((3, 2), dtype=bool)
    dropped = jnp.asarray([[True, False], [False, True], [False, False]])
    return live, dropped


def test_debug_raises_and_names_the_counts() -> None:
    """At `log_level="debug"` a drop stops the run and reports how many."""
    live, dropped = _masks()

    with pytest.raises(UnrepresentableOuterCandidateError) as report:
        _announce_dropped_outer_candidates(
            logger=get_logger(log_level="debug"),
            dropped=dropped,
            n_live=live,
            regime_name="working",
            period=3,
        )

    message = str(report.value)
    assert "2 of 6" in message
    assert "working" in message
    assert "period 3" in message


@pytest.mark.parametrize("log_level", ["warning", "progress"])
def test_warning_levels_report_the_drop_and_continue(log_level, caplog) -> None:
    """At `"warning"` and `"progress"` the drop is logged and the run goes on."""
    live, dropped = _masks()

    with caplog.at_level(logging.WARNING):
        _announce_dropped_outer_candidates(
            logger=get_logger(log_level=log_level),
            dropped=dropped,
            n_live=live,
            regime_name="working",
            period=3,
        )

    assert "2 of 6" in caplog.text


def test_off_is_silent_while_the_candidates_are_still_dropped(caplog) -> None:
    """At `log_level="off"` nothing is reported; dropping is not a log setting.

    The drop itself happens in the replay, independently of this call. What the
    level governs is only whether it is announced.
    """
    live, dropped = _masks()

    with caplog.at_level(logging.DEBUG):
        _announce_dropped_outer_candidates(
            logger=get_logger(log_level="off"),
            dropped=dropped,
            n_live=live,
            regime_name="working",
            period=3,
        )

    assert caplog.text == ""


def test_a_run_dropping_nothing_reports_nothing(caplog) -> None:
    """No message and no raise when every live candidate was reconstructed."""
    live = jnp.ones((3, 2), dtype=bool)

    with caplog.at_level(logging.DEBUG):
        _announce_dropped_outer_candidates(
            logger=get_logger(log_level="debug"),
            dropped=jnp.zeros((3, 2), dtype=bool),
            n_live=live,
            regime_name="working",
            period=3,
        )

    assert caplog.text == ""


def test_interpolating_a_constant_target_surface_does_not_reproduce_its_node() -> None:
    """A constant surface read off the candidate bank misses its own node.

    This is why an adjuster candidate's target comes from the declared outer
    search nodes rather than from the interpolated bank: the bank's adjuster
    rows are constant along the state axes, so reading them "should" return the
    node, yet linear interpolation forms `(1 - w) * c + w * c`, which rounds. A
    target a rounding away from a domain endpoint is not that endpoint, and the
    candidate would then be admitted on containment and reach a stock outside
    the domain.
    """
    node = jnp.float32(20.0)
    surface = jnp.full((10,), node, dtype=jnp.float32)
    coordinates = [jnp.linspace(0.0, 9.0, 100_001, dtype=jnp.float32)]

    read = map_coordinates(input=surface, coordinates=coordinates)

    assert read.dtype == jnp.float32
    assert bool(jnp.isfinite(read).all())
    assert int(jnp.sum(read != node)) > 0
