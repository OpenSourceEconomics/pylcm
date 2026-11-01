"""Entry-owned original U constructor schemas, copies and retained roots."""

import numpy as np
import pytest

from lcm._solver_api.authority import _snapshot_artifact_template_once
from lcm.exceptions import ExecutionPlanningError
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_generation_materialization import _owner

pytestmark = [
    pytest.mark.coverage(backends=("cpu",), precisions="both"),
    pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON),
]


def test_native_template_host_upload_refuses_before_copy() -> None:
    """The optional copier grants no implicit host-array producer admission."""
    owner = _owner()
    try:
        with pytest.raises(ExecutionPlanningError, match="host uploads"):
            _snapshot_artifact_template_once(
                template=(np.array([1.0]),),
                payload_runtime_type=tuple,
                array_copier=owner.copy_solution_leaf,
            )
        assert not owner._foreign_copies
    finally:
        owner.close()
