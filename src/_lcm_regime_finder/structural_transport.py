"""Inert typed object transport, with packed row occurrences and exact bytes."""
from array import array
import hashlib
import json
import struct
import sys


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode('ascii')


def atom(value):
    kind = type(value)
    if value is None: return ['none']
    if kind is bool: return ['bool', value]
    if kind is int: return ['int', str(value)]
    if kind is float: return ['binary64', struct.pack('>d', value).hex()]
    if kind is str: return ['str', value]
    if kind is bytes: return ['bytes', value.hex()]
    return None


def order_key(value):
    scalar = atom(value)
    if scalar is not None: return canonical(scalar)
    if type(value) is tuple: return canonical(['tuple', [order_key(v).hex() for v in value]])
    if type(value) is frozenset: return canonical(['frozenset', sorted(order_key(v).hex() for v in value)])
    raise ValueError('Unsupported unordered structural element')


def ordered_members(value):
    if value and all(type(row) is tuple and all(type(x) is int for x in row) for row in value):
        return sorted(value)
    return sorted(value, key=order_key)


class Writer:
    def __init__(self, *, check=None, logical_array_type=None):
        self.check = check or (lambda: None)
        self.array_type = logical_array_type
        self.records, self.rows, self.memo = [], {}, {}
        self.payloads, self.blobs, self.content = {}, {}, {}
        self.watchers, self.active = [], set()

    def put(self, data):
        digest = hashlib.sha256(data).hexdigest()
        # Compare actual bytes as well as their hash before sharing storage.
        for key in self.content.get(digest, []):
            if self.payloads[key] == data: return key
        key = 'p' + str(len(self.payloads))
        self.payloads[key] = data
        self.blobs[key] = {'bytes': len(data), 'sha256': digest}
        self.content.setdefault(digest, []).append(key)
        return key

    def vector(self, values):
        result = array('Q', values)
        if sys.byteorder != 'little': result.byteswap()
        return self.put(result.tobytes())

    def token(self, value):
        scalar = atom(value)
        return scalar if scalar is not None else ['object', id(value)]

    def stamp(self, value):
        kind = type(value)
        if kind is dict: return [(self.token(k), self.token(v)) for k, v in value.items()]
        if kind in (list, tuple): return [self.token(v) for v in value]
        if kind in (set, frozenset): return [self.token(v) for v in ordered_members(value)]
        if kind is bytearray: return hashlib.sha256(value).digest()
        if kind is self.array_type:
            return self.token(value.shape), value.dtype, value.order, self.token(value.data)
        raise ValueError('Unsupported structural source type: ' + kind.__qualname__)

    def visit(self, value):
        self.check()
        scalar = atom(value)
        if scalar is not None: return scalar
        key = id(value)
        if key in self.active: raise ValueError('Cyclic structural source requires an explicit supported view')
        if key in self.memo: return self.memo[key][1]
        kind = type(value)
        if kind is tuple and all(type(v) is int for v in value):
            width = str(len(value))
            table = self.rows.setdefault(width, {'values': [], 'intern': {}, 'handles': array('Q')})
            row_id = table['intern'].get(value)
            if row_id is None:
                row_id = len(table['values'])
                table['intern'][value] = row_id
                table['values'].append(value)
            ref = ['row', width, len(table['handles'])]
            table['handles'].append(row_id)
            self.memo[key] = (value, ref)
            return ref
        ref = ['ref', len(self.records)]
        self.memo[key] = (value, ref)
        self.records.append(None)
        self.active.add(key)
        before = self.stamp(value)
        self.watchers.append((value, before))
        if kind is dict:
            body = ['dict', [[self.visit(k), self.visit(v)] for k, v in value.items()]]
        elif kind in (list, tuple, set, frozenset):
            values = ordered_members(value) if kind in (set, frozenset) else value
            if values and all(type(v) is tuple and len(v) == len(values[0]) and all(type(x) is int for x in v) for v in values):
                row_refs = [self.visit(v) for v in values]
                body = ['rows_' + kind.__name__, row_refs[0][1], self.vector(v[2] for v in row_refs), len(row_refs)]
            else:
                tokens = [self.visit(v) for v in values]
                body = (['items_' + kind.__name__, self.put(canonical(tokens)), len(tokens)]
                        if len(tokens) > 8192 else [kind.__name__, tokens])
        elif kind is bytearray:
            body = ['bytearray', self.put(bytes(value))]
        elif kind is self.array_type:
            body = ['array', self.visit(value.shape), value.dtype, value.order, self.visit(value.data)]
        else:
            raise ValueError('Unsupported structural source type: ' + kind.__qualname__)
        self.records[ref[1]] = body
        self.active.remove(key)
        return ref

    def finish(self, source, anchor, before_final_validation=None):
        root = self.visit(source)
        if before_final_validation is not None: before_final_validation()
        for value, expected in self.watchers:
            self.check()
            if self.stamp(value) != expected: raise ValueError('Structural source changed during capture')
        tables = {}
        for width, table in self.rows.items():
            self.check()
            tables[width] = {'values': self.put(canonical([[str(x) for x in row] for row in table['values']])),
                             'handles': self.vector(table['handles']),
                             'value_count': len(table['values']), 'object_count': len(table['handles'])}
        manifest = {'schema': 'regime-structural-store/v1', 'generation_id': 'content-bound',
                    'status': 'COMPLETE', 'roots': root, 'source_anchor': anchor,
                    'source_consistency': 'Endpoint observations only; an external quiescence contract is required for a coherent production snapshot',
                    'records': self.records, 'row_tables': tables, 'views': {},
                    'blobs': self.blobs, 'counts': {'objects': len(self.records),
                    'row_values': sum(t['value_count'] for t in tables.values()),
                    'row_objects': sum(t['object_count'] for t in tables.values()),
                    'payload_bytes': sum(v['bytes'] for v in self.blobs.values())}, 'missing': []}
        manifest['root_digest'] = hashlib.sha256(canonical(manifest)).hexdigest()
        return {'manifest': manifest, 'payloads': self.payloads}


def capture_structure(*, source, source_anchor, before_final_validation=None,
                      logical_array_type=None, check=None):
    return Writer(check=check, logical_array_type=logical_array_type).finish(
        source, source_anchor, before_final_validation)


def reseal(*, bundle):
    for key, data in bundle['payloads'].items():
        bundle['manifest']['blobs'][key]['sha256'] = hashlib.sha256(data).hexdigest()
    manifest = bundle['manifest']
    manifest['root_digest'] = hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != 'root_digest'})).hexdigest()
    return bundle
