"""Every selectable upper-envelope backend is dispatched and documented.

`DCEGM.envelope` is the single source of truth for which backends exist. A name
that reaches the dispatch table without reaching the package docstring is a
backend users can select and cannot read about — the default is the one that
matters most, since it is what a model gets without saying anything.
"""

import typing

import pytest

from _lcm.egm import upper_envelope
from _lcm.egm.upper_envelope import UpperEnvelopeBackend, get_upper_envelope
from lcm import LinSpacedGrid
from lcm.solvers import DCEGM


def _selectable_backends() -> tuple[str, ...]:
    """Tuple of backend names a user may write in `DCEGM(envelope=...)`."""
    return typing.get_args(typing.get_type_hints(DCEGM)["envelope"])


def _solver(*, envelope: str) -> DCEGM:
    """A minimal consumption-savings DC-EGM solver using `envelope`."""
    return DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=4),
        envelope=envelope,  # ty: ignore[invalid-argument-type]
    )


@pytest.mark.parametrize("envelope", _selectable_backends())
def test_every_selectable_backend_is_dispatched(envelope: str) -> None:
    """Each name `DCEGM.envelope` accepts builds a backend."""
    backend = get_upper_envelope(solver=_solver(envelope=envelope), n_refined=8)
    assert isinstance(backend, UpperEnvelopeBackend)


def test_the_package_docstring_names_every_selectable_backend() -> None:
    """Each name `DCEGM.envelope` accepts appears in the package docstring."""
    docstring = upper_envelope.__doc__ or ""
    undocumented = [
        name for name in _selectable_backends() if f'`"{name}"`' not in docstring
    ]
    assert undocumented == []
