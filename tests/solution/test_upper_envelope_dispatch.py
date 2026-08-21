"""Every typed upper-envelope backend is dispatched and documented."""

import pytest

from _lcm.egm import upper_envelope
from _lcm.egm.upper_envelope import UpperEnvelopeBackend, get_upper_envelope
from lcm import LinSpacedGrid
from lcm.solvers import (
    DCEGM,
    EnvelopeConfig,
    ExactEnvelope,
    FUESEnvelope,
    LTMEnvelope,
    MSSEnvelope,
    RFCEnvelope,
)

_BACKEND_TYPES = (
    ExactEnvelope,
    FUESEnvelope,
    RFCEnvelope,
    LTMEnvelope,
    MSSEnvelope,
)


def _solver(*, envelope: EnvelopeConfig) -> DCEGM:
    """Return a minimal DC-EGM solver using one typed backend configuration."""
    return DCEGM(
        savings_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=4),
        envelope=envelope,
    )


@pytest.mark.parametrize("backend_type", _BACKEND_TYPES)
def test_every_selectable_backend_is_dispatched(backend_type) -> None:
    """Every public backend configuration builds an upper-envelope callable."""
    backend = get_upper_envelope(
        solver=_solver(envelope=backend_type()),
        n_refined=8,
    )
    assert isinstance(backend, UpperEnvelopeBackend)


def test_the_package_docstring_names_every_selectable_backend() -> None:
    """Every public configuration class appears in the package documentation."""
    docstring = upper_envelope.__doc__ or ""
    undocumented = [
        backend_type.__name__
        for backend_type in _BACKEND_TYPES
        if backend_type.__name__ not in docstring
    ]
    assert undocumented == []
