from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any


def _reverse_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Reverse the order of keys in a dictionary."""
    return {k: d[k] for k in reversed(list(d))}


@pytest.mark.skip(
    reason="We should use a proper example here where with 2 states and 2 actions."
)
def test_order_of_states_and_actions_does_not_matter():
    pass
