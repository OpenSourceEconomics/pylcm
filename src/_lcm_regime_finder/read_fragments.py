"""Reassemble losslessly fragmented review JSON without model/solver imports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys


def read(index: Path, *, reads=None):
    data = reads.document(index) if reads is not None else json.loads(index.read_text())
    assert data['format'] == 'lossless-json-fragments-1'
    base = index.parent.resolve()
    manifest = {entry['path']: entry['sha256'] for entry in data['parts']}
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {data['marker']}:
                name = value[data['marker']]
                target = (base / name).resolve()
                assert target.parent == base and name in manifest
                if reads is not None:
                    target = reads.child(index, name, name, base='supplied_parent', confined=True)
                raw = reads.read_bytes(target) if reads is not None else target.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == manifest[name]
                return json.loads(raw)
            return {key: decode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [decode(child) for child in value]
        return value
    result = decode(data['value'])
    identity = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert identity == data['canonical_value_sha256']
    return result


if __name__ == '__main__':
    for name in sys.argv[1:]:
        result = read(Path(name))
        print(json.dumps({'input': name, 'status': 'PASS', 'reconstructed_type': type(result).__name__}))
