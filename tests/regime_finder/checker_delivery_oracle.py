"""Literal completed-invocation delivery oracle, independent of project code."""
import hashlib
import json


def file_identity(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_delivery(exit_code, result_path, protected, *, success):
    """Check the specified public destination, authority and input identities.

    No production path resolver, cleanup inventory, status mapper or resource
    implementation is imported. The caller supplies the contract's destination.
    """
    for path, digest in protected.items():
        assert path.is_file(), f'Original input disappeared: {path}'
        assert file_identity(path) == digest, f'Original input changed: {path}'
    assert not result_path.with_name(result_path.name + '.writing').exists()
    if success:
        assert exit_code == 0, f'Expected completed generated check, got {exit_code}'
        assert json.loads(result_path.read_text())['status'] == 'PASS'
        return
    assert exit_code not in (0, 3), f'Failed check returned a success code: {exit_code}'
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text())
    assert result.get('status') in {'FAIL', 'INVALID', 'INCOMPLETE', 'FAILED'}, result
    assert not result.get('transformation_authorized'), result
    assert not result.get('recommendation'), result
    assert not result.get('decisions'), result
    assert not result.get('proposal_decisions'), result
