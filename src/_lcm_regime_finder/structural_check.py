"""Independent inert structural decoder and source-anchor check.

No writer intern table, normalizer, digest function or exclusion list is used.
"""
from array import array
from dataclasses import dataclass
import hashlib
import json
import struct
import sys
from types import MappingProxyType


@dataclass
class LogicalArray:
    shape: tuple
    dtype: str
    order: str
    data: bytearray


@dataclass(frozen=True)
class BufferView:
    chunks: tuple
    size: int
    payloads: object

    def iter_bytes(self):
        for key in self.chunks: yield self.payloads[key]


@dataclass(frozen=True)
class NativeArrayView:
    flavor: str
    shape: tuple
    dtype: str
    strides: tuple
    offset: int
    buffer: BufferView
    readonly: bool


@dataclass(frozen=True)
class DataclassView:
    authority: tuple
    fields: object


def encoded(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('ascii')


def scalar(value):
    kind = type(value)
    if value is None: return ['none']
    if kind is bool: return ['bool', value]
    if kind is int: return ['int', str(value)]
    if kind is float: return ['binary64', struct.pack('>d', value).hex()]
    if kind is str: return ['str', value]
    if kind is bytes: return ['bytes', value.hex()]
    return None


def sort_key(value):
    item = scalar(value)
    if item is not None: return encoded(item)
    if type(value) is tuple: return encoded(['tuple', [sort_key(v).hex() for v in value]])
    if type(value) is frozenset: return encoded(['frozenset', sorted(sort_key(v).hex() for v in value)])
    raise ValueError('Unsupported unordered value')


def logical_observation(source, *, array_type=LogicalArray):
    seen, objects = {}, []
    def walk(value):
        item = scalar(value)
        if item is not None: return item
        key = id(value)
        if key in seen: return ['ref', seen[key]]
        index = len(objects)
        seen[key] = index
        objects.append(None)
        kind = type(value)
        if kind in (dict, MappingProxyType): item = ['dict' if kind is dict else 'mappingproxy', [[walk(k), walk(v)] for k, v in value.items()]]
        elif kind in (tuple, list): item = [kind.__name__, [walk(v) for v in value]]
        elif kind in (set, frozenset): item = [kind.__name__, [walk(v) for v in sorted(value, key=sort_key)]]
        elif kind is bytearray: item = ['bytearray', bytes(value).hex()]
        elif kind is array_type: item = ['array', walk(value.shape), value.dtype, value.order, walk(value.data)]
        elif kind is BufferView:
            digest = hashlib.sha256()
            for part in value.iter_bytes(): digest.update(part)
            item = ['native-buffer', value.size, digest.hexdigest()]
        elif kind is NativeArrayView:
            item = ['native-array', value.flavor, list(value.shape), value.dtype, list(value.strides),
                    value.offset, walk(value.buffer), value.readonly]
        else: raise ValueError('Unsupported source comparison type')
        objects[index] = item
        return ['ref', index]
    root = walk(source)
    return {'schema': 'typed-logical-graph/v1', 'root': root, 'objects': objects}


class Reader:
    def __init__(self, bundle, *, array_factory=None):
        self.doc, self.payloads = bundle['manifest'], bundle['payloads']
        doc = self.doc
        if doc['schema'] != 'regime-structural-store/v1' or doc['status'] != 'COMPLETE' or doc['missing']:
            raise ValueError('Incomplete or unsupported structural store')
        digest = hashlib.sha256(encoded({k: v for k, v in doc.items() if k != 'root_digest'})).hexdigest()
        if digest != doc['root_digest'] or set(self.payloads) != set(doc['blobs']):
            raise ValueError('Structural root or payload inventory changed')
        for key, data in self.payloads.items():
            if type(data) is not bytes or len(data) != doc['blobs'][key]['bytes'] or hashlib.sha256(data).hexdigest() != doc['blobs'][key]['sha256']:
                raise ValueError('Structural chunk changed')
        self.cache, self.rows, self.handles, self.row_cache = {}, {}, {}, {}
        self.used, self.row_used, self.blob_used = set(), set(), set()
        self.active = set()
        self.factory = array_factory or LogicalArray
        for width, table in doc['row_tables'].items():
            if str(int(width)) != width or int(width) < 0: raise ValueError('Invalid row width')
            rows = json.loads(self.blob(table['values']))
            if type(table['value_count']) is not int or len(rows) != table['value_count']: raise ValueError('Row value count changed')
            values = []
            for row in rows:
                if len(row) != int(width) or any(type(x) is not str or str(int(x)) != x for x in row):
                    raise ValueError('Invalid integer row')
                values.append(tuple(int(x) for x in row))
            if len(set(values)) != len(values): raise ValueError('Duplicate interned row')
            handles = self.vector(table['handles'])
            if type(table['object_count']) is not int or len(handles) != table['object_count'] or any(x >= len(values) for x in handles):
                raise ValueError('Invalid row object mapping')
            if set(handles) != set(range(len(values))):
                raise ValueError('Unreachable interned row value')
            self.rows[width], self.handles[width] = values, handles
        counts = {'objects': len(doc['records']), 'row_values': sum(len(v) for v in self.rows.values()),
                  'row_objects': sum(len(v) for v in self.handles.values()),
                  'payload_bytes': sum(len(v) for v in self.payloads.values())}
        if encoded(doc['counts']) != encoded(counts): raise ValueError('Incorrect structural counts')

    def blob(self, key):
        self.blob_used.add(key)
        return self.payloads[key]

    def vector(self, key):
        data = self.blob(key)
        if len(data) % 8: raise ValueError('Truncated index vector')
        values = array('Q')
        values.frombytes(data)
        if sys.byteorder != 'little': values.byteswap()
        return values

    def read(self, ref):
        if type(ref) is not list or not ref: raise ValueError('Malformed structural reference')
        tag = ref[0]
        if tag == 'none' and len(ref) == 1: return None
        if tag == 'bool' and len(ref) == 2 and type(ref[1]) is bool: return ref[1]
        if tag == 'str' and len(ref) == 2 and type(ref[1]) is str: return ref[1]
        if tag == 'int' and len(ref) == 2 and type(ref[1]) is str and str(int(ref[1])) == ref[1]: return int(ref[1])
        if tag == 'binary64' and len(ref) == 2: return struct.unpack('>d', bytes.fromhex(ref[1]))[0]
        if tag == 'bytes' and len(ref) == 2: return bytes.fromhex(ref[1])
        if tag == 'row' and len(ref) == 3:
            width, index = ref[1:]
            if type(index) is not int or index < 0 or index >= len(self.handles[width]): raise ValueError('Invalid row reference')
            key = (width, index)
            self.row_used.add(key)
            if key not in self.row_cache:
                self.row_cache[key] = tuple(x for x in self.rows[width][self.handles[width][index]])
            return self.row_cache[key]
        if tag != 'ref' or len(ref) != 2 or type(ref[1]) is not int or ref[1] < 0 or ref[1] >= len(self.doc['records']):
            raise ValueError('Unknown or out-of-range structural reference')
        index = ref[1]
        self.used.add(index)
        if index in self.active: raise ValueError('Unsupported cyclic structural graph')
        if index in self.cache: return self.cache[index]
        self.active.add(index)
        record = self.doc['records'][index]
        if type(record) is not list or not record or type(record[0]) is not str:
            raise ValueError('Malformed structural record')
        kind = record[0]
        if kind.startswith('rows_'):
            width, blob, count = record[1:]
            ids = self.vector(blob)
            if type(count) is not int or len(ids) != count: raise ValueError('View count changed')
            values = [self.read(['row', width, int(i)]) for i in ids]
            kind = kind[5:]
        elif kind.startswith('items_'):
            tokens = json.loads(self.blob(record[1]))
            if type(tokens) is not list or type(record[2]) is not int or len(tokens) != record[2]:
                raise ValueError('Packed structural sequence count changed')
            values = [self.read(v) for v in tokens]
            kind = kind[6:]
        elif kind in ('tuple', 'list', 'set', 'frozenset'):
            values = [self.read(v) for v in record[1]]
        if kind in ('list', 'tuple', 'set', 'frozenset'):
            if kind in ('set', 'frozenset') and len(set(values)) != len(values): raise ValueError('Duplicate logical set member')
            value = {'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset}[kind](values)
        elif kind in ('dict', 'mappingproxy'):
            pairs = [(self.read(k), self.read(v)) for k, v in record[1]]
            value = dict(pairs)
            if len(value) != len(pairs): raise ValueError('Duplicate mapping key')
            if kind == 'mappingproxy': value = MappingProxyType(value)
        elif kind == 'native-dataclass':
            info, payload = record[1:]
            if type(info) is not list or len(info) != 3 or type(info[0]) is not str or type(info[1]) is not str or len(info[1]) != 64:
                raise ValueError('Invalid dataclass source authority')
            names = [row[0] for row in info[2]]
            if len(set(names)) != len(names) or [row[0] for row in payload] != names:
                raise ValueError('Dataclass field inventory changed')
            value = DataclassView(tuple(info), MappingProxyType({name: self.read(ref) for name, ref in payload}))
        elif kind == 'bytearray': value = bytearray(self.blob(record[1]))
        elif kind == 'array': value = self.factory(self.read(record[1]), record[2], record[3], self.read(record[4]))
        elif kind == 'native-buffer':
            chunks, size = record[1:]
            if type(size) is not int or size < 0 or sum(len(self.blob(key)) for key in chunks) != size:
                raise ValueError('Native buffer byte count changed')
            value = BufferView(tuple(chunks), size, self.payloads)
        elif kind == 'native-array':
            flavor, shape, dtype, strides, offset, buffer, readonly = record[1:]
            if flavor not in ('numpy', 'jax-value') or type(readonly) is not bool:
                raise ValueError('Invalid native array contract')
            if type(shape) is not list or type(strides) is not list or len(shape) != len(strides) or any(type(n) is not int or n < 0 for n in shape) or any(type(n) is not int for n in strides) or type(offset) is not int:
                raise ValueError('Invalid native array geometry')
            # Primitive dtype grammar; no NumPy import or object dtype execution.
            import re
            match = re.fullmatch(r'[<>=|]([biufc])(\d+)', dtype)
            if not match or int(match[2]) not in (1, 2, 4, 8, 16, 32):
                raise ValueError('Unsupported primitive dtype descriptor')
            storage = self.read(buffer)
            if type(storage) is not BufferView: raise ValueError('Wrong native buffer reference type')
            low = offset + sum(min(0, (n - 1) * s) for n, s in zip(shape, strides))
            high = offset + sum(max(0, (n - 1) * s) for n, s in zip(shape, strides)) + int(match[2])
            if 0 not in shape and (low < 0 or high > storage.size): raise ValueError('Native view escapes source buffer')
            if 0 in shape and not 0 <= offset <= storage.size: raise ValueError('Invalid empty view offset')
            if flavor == 'jax-value' and not readonly: raise ValueError('JAX value view cannot be mutable')
            value = NativeArrayView(flavor, tuple(shape), dtype, tuple(strides), offset, storage, readonly)
        else: raise ValueError('Unknown structural record kind')
        self.active.remove(index)
        self.cache[index] = value
        return value

    def validate_graph(self):
        """Check all references and incidences without constructing repeated rows."""
        objects, active = set(), set()
        rows = {width: bytearray(len(handles)) for width, handles in self.handles.items()}
        def visit(token):
            if type(token) is not list or not token: raise ValueError('Malformed structural token')
            if token[0] == 'row':
                if len(token) != 3 or token[1] not in rows or type(token[2]) is not int or not 0 <= token[2] < len(rows[token[1]]):
                    raise ValueError('Invalid structural row reference')
                rows[token[1]][token[2]] = 1
                return
            if token[0] != 'ref':
                self.read(token)
                return
            if len(token) != 2 or type(token[1]) is not int or not 0 <= token[1] < len(self.doc['records']):
                raise ValueError('Invalid structural object reference')
            index = token[1]
            if index in active: raise ValueError('Cyclic structural graph')
            if index in objects: return
            active.add(index); objects.add(index)
            record = self.doc['records'][index]
            if type(record) is not list or not record or type(record[0]) is not str:
                raise ValueError('Malformed structural record')
            kind = record[0]
            if kind.startswith('rows_'):
                if len(record) != 4 or kind[5:] not in ('list', 'tuple', 'set', 'frozenset'):
                    raise ValueError('Invalid packed row view')
                width, blob, count = record[1:]
                if width not in rows or type(count) is not int: raise ValueError('Invalid row view descriptor')
                ids = self.vector(blob)
                if len(ids) != count: raise ValueError('Row view length changed')
                values_seen = set() if kind[5:] in ('set', 'frozenset') else None
                for identity in ids:
                    visit(['row', width, identity])
                    if values_seen is not None:
                        value = self.handles[width][identity]
                        if value in values_seen: raise ValueError('Duplicate row in structural set')
                        values_seen.add(value)
            elif kind in ('dict', 'mappingproxy', 'list', 'tuple', 'set', 'frozenset') or kind.startswith('items_'):
                if kind.startswith('items_'):
                    if kind[6:] not in ('list', 'tuple', 'set', 'frozenset') or len(record) != 3:
                        raise ValueError('Invalid packed item view')
                    tokens = json.loads(self.blob(record[1]))
                    if type(tokens) is not list or type(record[2]) is not int or len(tokens) != record[2]:
                        raise ValueError('Packed item count changed')
                else:
                    if len(record) != 2 or type(record[1]) is not list: raise ValueError('Invalid structural container')
                    tokens = record[1]
                if kind in ('dict', 'mappingproxy'):
                    keys = set()
                    for pair in tokens:
                        if type(pair) is not list or len(pair) != 2: raise ValueError('Invalid mapping pair')
                        visit(pair[0]); visit(pair[1])
                        key = self.read(pair[0])
                        if key in keys: raise ValueError('Duplicate structural mapping key')
                        keys.add(key)
                else:
                    for child in tokens: visit(child)
                    if kind in ('set', 'frozenset', 'items_set', 'items_frozenset'):
                        values = [self.read(child) for child in tokens]
                        if len(set(values)) != len(values): raise ValueError('Duplicate structural set value')
            elif kind == 'native-dataclass':
                if len(record) != 3: raise ValueError('Invalid dataclass record')
                info, payload = record[1:]
                if type(info) is not list or len(info) != 3 or type(info[0]) is not str or type(info[1]) is not str or len(info[1]) != 64:
                    raise ValueError('Invalid dataclass source identity')
                names = [row[0] for row in info[2]]
                if len(set(names)) != len(names) or [row[0] for row in payload] != names:
                    raise ValueError('Dataclass field inventory changed')
                for name, token in payload: visit(token)
            elif kind == 'bytearray':
                if len(record) != 2: raise ValueError('Invalid bytearray')
                self.blob(record[1])
            elif kind == 'array':
                if len(record) != 5: raise ValueError('Invalid logical array')
                visit(record[1]); visit(record[4])
                self.read(token)
            elif kind == 'native-buffer':
                if len(record) != 3: raise ValueError('Invalid native buffer')
                self.read(token)
            elif kind == 'native-array':
                if len(record) != 8: raise ValueError('Invalid native array')
                visit(record[6]); self.read(token)
            else:
                raise ValueError('Unknown structural record type')
            active.remove(index)
        visit(self.doc['roots'])
        if len(objects) != len(self.doc['records']) or any(flags.count(1) != len(flags) for flags in rows.values()) or self.blob_used != set(self.payloads):
            raise ValueError('Unreachable structural inventory')
        return {'objects': len(objects), 'row_objects': sum(len(v) for v in rows.values())}

    def decode(self):
        value = self.read(self.doc['roots'])
        if self.used != set(range(len(self.doc['records']))) or len(self.row_used) != sum(len(v) for v in self.handles.values()) or self.blob_used != set(self.payloads):
            raise ValueError('Unreachable structural inventory')
        return value


def decode(*, bundle, array_factory=None):
    try: return Reader(bundle, array_factory=array_factory).decode()
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError, struct.error) as error:
        raise ValueError('Malformed structural store') from error


def validate(*, bundle, source_anchor):
    value = decode(bundle=bundle)
    if bundle['manifest']['source_anchor'] != source_anchor:
        raise ValueError('External source authority differs')
    actual = hashlib.sha256(encoded(logical_observation(value))).hexdigest()
    if actual != source_anchor.get('logical_sha256'):
        raise ValueError('Decoded logical contents or alias graph differ from source')
