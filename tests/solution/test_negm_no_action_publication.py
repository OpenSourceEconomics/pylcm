"""NEGM publishes no finite value when no positive consumption can satisfy b."""

import numpy as np
import pytest

from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution.test_nested_dynamic_borrowing import (
    _assert_published_bank,
    _branches,
    _model,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def test_finite_negm_nonpositive_consumption_has_no_action(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original target2 candidate has value -inf at R<0 and R=b=0."""
    published, outputs = _branches(
        model=_model(family="negm", lower=0.0), monkeypatch=monkeypatch
    )
    _assert_published_bank(published=published, outputs=outputs, family="negm")
    # Same original liquid nodes0 and.75, fixed stock0 and original target2.
    # Their own resource values are-.75 and0, with literal saving floor0.
    np.testing.assert_array_equal(
        np.asarray(outputs[3].value)[[0, 3], 0], (-np.inf, -np.inf)
    )
