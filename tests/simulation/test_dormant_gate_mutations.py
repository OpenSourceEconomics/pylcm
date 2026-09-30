"""An undemanded gated declaration never changes the budgeted main -> end panel.

Controls remove the gate or explicitly add `latent` to the initial regimes.
Registry order, subject width and compilation workers vary without changing the
simulated population or its exact values.
"""

import pytest

from tests.simulation.test_dormant_gate_admission import exercise


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("promote", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("workers", [1, 2])
def test_dormant_gate_admission_class(
    *, gated: bool, promote: bool, reverse: bool, width: int, workers: int
) -> None:
    """Budgeted and unbudgeted panels agree exactly on main -> end."""
    exercise(
        gated=gated, promote=promote, reverse=reverse, width=width, workers=workers
    )
