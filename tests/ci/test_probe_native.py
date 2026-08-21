"""Contract for the CI native-payload setup probe."""

from tests.ci import probe_native


def test_absent_or_stale_payload_requests_one_reinstall(monkeypatch):
    """Only verified absence/staleness enters the conditional reinstall path."""
    monkeypatch.setattr(probe_native, "payload_present", lambda: False)

    assert probe_native.probe().exit_code == probe_native.REINSTALL_REQUIRED


def test_present_but_unloadable_payload_fails_without_reinstall(monkeypatch):
    """A broken installed library is a failed build, not a cache miss."""
    monkeypatch.setattr(probe_native, "payload_present", lambda: True)
    monkeypatch.setattr(probe_native, "manifest_matches", lambda _root: True)
    monkeypatch.setattr(probe_native, "kernel_available", lambda: False)

    assert probe_native.probe().exit_code == probe_native.BROKEN_PAYLOAD


def test_current_loadable_payload_needs_no_second_build(monkeypatch):
    """A cache hit with the installed payload goes directly to tests."""
    monkeypatch.setattr(probe_native, "payload_present", lambda: True)
    monkeypatch.setattr(probe_native, "manifest_matches", lambda _root: True)
    monkeypatch.setattr(probe_native, "kernel_available", lambda: True)

    assert probe_native.probe().exit_code == 0
