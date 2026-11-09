"""Independent streaming comparison of original roots with checked wire views.

The native event format is separate from the small generated-fixture oracle.
Rows are atomic typed tuples; object aliases still have distinct event handles.
"""
from array import array
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import inspect
import json
from pathlib import Path
import struct
import sys
from types import MappingProxyType

from .structural_check import scalar, sort_key, encoded


def row_bytes(row):
    try:
        values = array('q', row)
        if sys.byteorder != 'little': values.byteswap()
        return b'i64' + struct.pack('<Q', len(row)) + values.tobytes()
    except OverflowError:
        return b'bigint' + encoded([str(v) for v in row])


@dataclass(frozen=True)
class _SourceBuffer:
    owner: object
    flavor: str


class SourceEvents:
    def identity(self, value):
        if scalar(value) is not None: return None
        if type(value) is _SourceBuffer: return ('buffer', value.flavor, id(value.owner))
        return ('object', id(value))

    def describe(self, value):
        item = scalar(value)
        if item is not None: return None, b'atom' + encoded(item), ()
        kind = type(value)
        if kind is _SourceBuffer:
            import numpy as np
            data = np.asarray(value.owner) if value.flavor == 'jax-value' else value.owner
            if value.flavor == 'jax-value' and not data.flags.c_contiguous:
                data = np.ascontiguousarray(data)
            raw = memoryview(data.ravel(order='K')).cast('B')
            digest = hashlib.sha256()
            for start in range(0, len(raw), 1024**2): digest.update(raw[start:start + 1024**2])
            return ('buffer', value.flavor, id(value.owner)), b'native-buffer' + encoded([len(raw), digest.hexdigest()]), ()
        if kind is tuple and all(type(v) is int for v in value):
            return ('object', id(value)), b'tuple-int' + row_bytes(value), ()
        np, jax, impl = (sys.modules.get(name) for name in ('numpy', 'jax', 'jaxlib._jax'))
        numpy_array = np is not None and kind is np.ndarray
        jax_array = jax is not None and impl is not None and kind is getattr(impl, 'ArrayImpl', None) and isinstance(value, jax.Array)
        if numpy_array or jax_array:
            import numpy as np
            dtype = np.dtype(value.dtype)
            if dtype.fields is not None or dtype.hasobject or dtype.kind not in 'biufc':
                raise ValueError('Unsupported source array dtype')
            shape = tuple(int(v) for v in value.shape)
            if numpy_array:
                base = value
                while type(base.base) is np.ndarray: base = base.base
                if base.base is not None:
                    raise ValueError('Unsupported source buffer ownership')
                if not (base.flags.c_contiguous or base.flags.f_contiguous):
                    width = base.dtype.itemsize
                    for axis in sorted(range(base.ndim), key=base.strides.__getitem__):
                        if base.shape[axis] <= 1:
                            continue
                        if base.strides[axis] != width:
                            raise ValueError('Unsupported source buffer ownership')
                        width *= base.shape[axis]
                    if not base.flags.owndata or width != base.nbytes:
                        raise ValueError('Unsupported source buffer ownership')
                flavor = 'numpy'
                strides = tuple(int(v) for v in value.strides)
                offset = int(value.ctypes.data) - int(base.ctypes.data)
                readonly = not value.flags.writeable
            else:
                base, flavor, offset, readonly = value, 'jax-value', 0, True
                stride, strides = dtype.itemsize, []
                for dimension in reversed(shape): strides.insert(0, stride); stride *= dimension
                strides = tuple(strides)
            metadata = [flavor, list(shape), dtype.str, list(strides), offset, readonly]
            return ('object', id(value)), b'native-array' + encoded(metadata), (_SourceBuffer(base, flavor),)
        if kind is bytearray:
            return ('object', id(value)), b'bytearray' + encoded([len(value), hashlib.sha256(value).hexdigest()]), ()
        if kind in (dict, MappingProxyType):
            label = b'dict' if kind is dict else b'mappingproxy'
            return ('object', id(value)), label + encoded(len(value)), (v for pair in value.items() for v in pair)
        if is_dataclass(value) and not isinstance(value, type):
            cls = type(value)
            source = inspect.getsourcefile(cls)
            if source is None: raise ValueError('Dataclass source authority unavailable')
            info = [cls.__module__ + '.' + cls.__qualname__, hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                    [[field.name, str(field.type)] for field in fields(cls)]]
            return ('object', id(value)), b'native-dataclass' + encoded(info), (getattr(value, field.name) for field in fields(cls))
        if kind in (list, tuple):
            return ('object', id(value)), kind.__name__.encode() + encoded(len(value)), iter(value)
        if kind in (set, frozenset):
            if value and all(type(v) is tuple and all(type(x) is int for x in v) for v in value):
                values = sorted(value)
            else: values = sorted(value, key=sort_key)
            return ('object', id(value)), kind.__name__.encode() + encoded(len(value)), iter(values)
        raise ValueError('Unsupported original structural type: ' + kind.__qualname__)


class WireEvents:
    def __init__(self, reader): self.reader = reader

    def identity(self, token):
        if token[0] == 'row': return ('row', token[1], token[2])
        if token[0] == 'ref': return ('object', token[1])
        return None

    def describe(self, token):
        reader = self.reader
        if token[0] == 'row':
            width, identity = token[1:]
            row = reader.rows[width][reader.handles[width][identity]]
            return ('row', width, identity), b'tuple-int' + row_bytes(row), ()
        if token[0] != 'ref':
            # Use checked scalar decoding instead of trusting a printed tag.
            return None, b'atom' + encoded(scalar(reader.read(token))), ()
        identity = token[1]
        record = reader.doc['records'][identity]
        kind = record[0]
        if kind == 'native-buffer':
            digest = hashlib.sha256()
            for key in record[1]: digest.update(reader.payloads[key])
            return ('object', identity), b'native-buffer' + encoded([record[2], digest.hexdigest()]), ()
        if kind == 'native-array':
            metadata = [record[1], record[2], record[3], record[4], record[5], record[7]]
            return ('object', identity), b'native-array' + encoded(metadata), (record[6],)
        if kind == 'bytearray':
            data = reader.blob(record[1])
            return ('object', identity), b'bytearray' + encoded([len(data), hashlib.sha256(data).hexdigest()]), ()
        if kind in ('dict', 'mappingproxy'):
            return ('object', identity), kind.encode() + encoded(len(record[1])), (v for pair in record[1] for v in pair)
        if kind == 'native-dataclass':
            return ('object', identity), b'native-dataclass' + encoded(record[1]), (ref for name, ref in record[2])
        if kind.startswith('rows_'):
            width = record[1]
            ids = reader.vector(record[2])
            container = kind[5:]
            if container in ('set', 'frozenset'):
                ids = sorted(ids, key=lambda i: reader.rows[width][reader.handles[width][i]])
            values = (['row', width, i] for i in ids)
            return ('object', identity), container.encode() + encoded(record[3]), values
        if kind.startswith('items_'):
            values = json.loads(reader.blob(record[1])); container = kind[6:]
        elif kind in ('list', 'tuple', 'set', 'frozenset'):
            values, container = record[1], kind
        else: raise ValueError('Unsupported structural event record')
        if container in ('set', 'frozenset'):
            values = sorted(values, key=lambda v: sort_key(reader.read(v)))
        return ('object', identity), container.encode() + encoded(len(values)), iter(values)


def event_digest(provider, root, *, check=None):
    """Stream typed events, retaining alias IDs but no repeated logical tree."""
    digest = hashlib.sha256(b'typed-structural-events/v1')
    seen, active = {}, set()
    check = check or (lambda: None)
    def emit(data):
        digest.update(struct.pack('<Q', len(data))); digest.update(data)
    def walk(value):
        check()
        identity = provider.identity(value)
        if identity in active: raise ValueError('Cyclic structural source')
        if identity is not None and identity in seen:
            emit(b'ref' + struct.pack('<Q', seen[identity])); return
        observed, metadata, children = provider.describe(value)
        if observed != identity: raise ValueError('Structural event identity changed')
        if identity is None:
            emit(metadata); return
        index = len(seen); seen[identity] = index; active.add(identity)
        emit(b'object' + struct.pack('<Q', index)); emit(metadata)
        for child in children: walk(child)
        emit(b'end'); active.remove(identity)
    walk(root)
    return digest.hexdigest()


def source_anchor(source, *, source_context, transaction, check=None):
    return {'schema': 'typed-structural-events/v1',
            'logical_sha256': event_digest(SourceEvents(), source, check=check),
            'source_context': source_context, 'transaction': transaction,
            'scope': 'Exact declared structural roots, supported NumPy layout and immutable JAX logical values; physical JAX buffer observations are not proved'}


def compare_source(reader, anchor, *, check=None):
    reader.validate_graph()
    if anchor.get('schema') != 'typed-structural-events/v1' or reader.doc['source_anchor'] != anchor:
        raise ValueError('Independent structural source authority differs')
    actual = event_digest(WireEvents(reader), reader.doc['roots'], check=check)
    if actual != anchor['logical_sha256']:
        raise ValueError('Structural values, aliases, ownership or layout differ from original roots')
    return actual
