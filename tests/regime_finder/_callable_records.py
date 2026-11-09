"""Check that serialized callable records are internally consistent.

Capture writes each record's digests (`callable_record_digest`); this re-derives
them for a whole record set, so the capture tests can check what was written.
"""

from _lcm_regime_finder.callable_identity import _digest, callable_record_digest


def validate_callable_records(records):
    """Validate serialization identity only, never callback semantics."""
    if not isinstance(records, dict):
        return False
    local = {}
    for ref, record in records.items():
        if not isinstance(record, dict) or record.get('capture_ref') != ref:
            return False
        local[ref] = callable_record_digest(record)
        if local[ref] != record.get('record_sha256'):
            return False
    graph = _digest(local)
    for ref, record in records.items():
        expected = _digest({'capture_ref': ref, 'record_sha256': local[ref], 'capture_graph_sha256': graph})
        if record.get('capture_graph_sha256') != graph or record.get('identity_sha256') != expected:
            return False
        pending = list(record.values())
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if value.get('kind') == 'callable-reference':
                    target = records.get(value.get('capture_ref'), {})
                    if (target.get('identity_sha256') != value.get('identity_sha256') or
                            value.get('capture_graph_sha256') != graph):
                        return False
                else:
                    pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
    return True
