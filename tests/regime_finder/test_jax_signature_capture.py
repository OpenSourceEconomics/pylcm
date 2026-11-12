"""Real native-wrapper identity and incomplete-capture reporting regressions."""

import pytest
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from _lcm_regime_finder.callable_identity import _callable_stamp, capture_session
from _lcm_regime_finder.model_snapshot import source_function
from _lcm_regime_finder.pytask_api import analyse_snapshot


class SignatureWrapper:
    def __init__(self):
        self.default: int | list[int] = 1
        self.annotation = "Array"

    @property
    def __signature__(self):
        return inspect.Signature(
            [
                inspect.Parameter(
                    "x", inspect.Parameter.POSITIONAL_ONLY, default=self.default
                )
            ],
            return_annotation=self.annotation,
        )

    def __call__(self, x=1):
        return x


@pytest.mark.skipif(
    not (importlib.util.find_spec("jax")),
    reason="real production JAX environment required",
)
def test_real_jax_abs_reallocated_signature_is_not_callable_mutation() -> None:
    import jax.numpy as jnp

    signatures = [getattr(jnp.abs, "__signature__"), getattr(jnp.abs, "__signature__")]
    assert (signatures[0]) is not (signatures[1])
    assert (signatures[0]) == (signatures[1])
    captured = source_function(jnp.abs)
    assert (captured["capture_report"]["status"]) == ("COMPLETE")
    assert not (any("mutation" in item.lower() for item in captured["obligations"]))
    assert ("Native or unavailable implementation source") in (captured["obligations"])
    assert ("opaque_boundary") in (captured)


def test_equivalent_new_advertised_signatures_keep_the_same_stamp() -> None:
    wrapper = SignatureWrapper()
    before = _callable_stamp(wrapper)
    held = [wrapper.__signature__ for _ in range(4)]
    assert all(signature == held[0] for signature in held)
    assert (before) == (_callable_stamp(wrapper))


def test_changed_advertised_default_is_still_detected() -> None:
    wrapper = SignatureWrapper()
    before = _callable_stamp(wrapper)
    wrapper.default = 2
    assert (before) != (_callable_stamp(wrapper))


def test_changed_advertised_annotation_is_still_detected() -> None:
    wrapper = SignatureWrapper()
    before = _callable_stamp(wrapper)
    wrapper.annotation = "OtherArray"
    assert (before) != (_callable_stamp(wrapper))


def test_mutated_signature_default_binding_still_invalidates_capture() -> None:
    wrapper = SignatureWrapper()
    wrapper.default = [1]
    with capture_session() as session:
        captured = source_function(wrapper)
        wrapper.default.append(2)
    assert (session.report()["status"]) == ("INCOMPLETE")
    assert (captured["capture_status"]) == ("UNRESOLVED_MUTATION")


def test_incomplete_capture_is_not_reported_as_resource_exhaustion() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        snapshot = root / "original.json"
        snapshot.write_text("{}")
        capture = {
            "schema": "regime-metadata-capture-receipt/v1",
            "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "status": "INCOMPLETE",
            "elapsed_seconds": 0.25,
            "capture_report": {
                "status": "INCOMPLETE",
                "exhaustion_reason": None,
                "obligations": ["Callable/binding mutation"],
            },
        }
        snapshot.with_name("original.metadata-capture.json").write_text(
            json.dumps(capture)
        )
        with patch("_lcm_regime_finder.resources.run_bounded") as runner:
            with pytest.raises(RuntimeError, match="capture INCOMPLETE"):
                analyse_snapshot(snapshot, root / "analysis", seconds=900)
            runner.assert_not_called()
        summary = json.loads((root / "analysis/summary.json").read_text())
        assert (summary["status"]) == ("INCOMPLETE")
        assert (summary["phase"]) == ("capture")
        assert (summary["capture_exhaustion_reason"]) == (None)
        assert ("Callable/binding mutation") in (summary["capture_obligations"])
        assert summary["metadata_budget"]["within_time_budget"]
        assert not (summary["analysis_started"])
        assert not (summary["transformation_authorized"])
        assert (summary["recommendation"]) is None
        assert ("resource boundary") not in (summary["reason"])
        assert (root / "analysis/next-actions.html").is_file()
