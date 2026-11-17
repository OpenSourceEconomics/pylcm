"""Native primitive array values and supported NumPy buffer/view aliases.

JAX arrays retain immutable logical values and object aliases. Physical device
buffer observations remain an explicit semantic premise, never an inferred law.
"""
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import inspect
from pathlib import Path
import sys
from types import MappingProxyType

from .structural_transport import Writer, reseal


@dataclass(frozen=True)
class BufferSource:
    owner: object
    flavor: str


@dataclass(frozen=True)
class TypeSource:
    cls: type


def type_information(cls):
    path = inspect.getsourcefile(cls)
    if path is None: raise ValueError('Structural dataclass has no source authority')
    return [cls.__module__ + '.' + cls.__qualname__,
            hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            [[field.name, str(field.type)] for field in fields(cls)]]


class NativeWriter(Writer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buffers = {}
        self.storage_obligations = []

    @staticmethod
    def flavor(value):
        np = sys.modules.get('numpy')
        if np is not None and type(value) is np.ndarray: return 'numpy'
        jax = sys.modules.get('jax')
        impl = sys.modules.get('jaxlib._jax')
        if jax is not None and impl is not None and type(value) is getattr(impl, 'ArrayImpl', None) and isinstance(value, jax.Array): return 'jax-value'
        return None

    def geometry(self, value, flavor):
        import numpy as np
        dtype = np.dtype(value.dtype)
        if dtype.hasobject or dtype.fields is not None or dtype.kind not in 'biufc':
            raise ValueError('Unsupported primitive array dtype')
        shape = tuple(int(v) for v in value.shape)
        if flavor == 'jax-value':
            return shape, dtype.str, None, 0, id(value), True
        owner = value
        while type(owner.base) is np.ndarray: owner = owner.base
        packed = owner.flags.c_contiguous or owner.flags.f_contiguous
        if not packed and owner.flags.owndata:
            width, packed = owner.dtype.itemsize, True
            # Owning axis permutations can be densely packed without C/F order.
            for stride, dimension in sorted((s, n) for n, s in zip(owner.shape, owner.strides) if n > 1):
                if stride != width:
                    packed = False
                    break
                width *= dimension
            packed = packed and width == owner.nbytes
        if owner.base is not None or not packed:
            raise ValueError('External or unsupported native array storage')
        offset = int(value.ctypes.data) - int(owner.ctypes.data)
        return shape, dtype.str, tuple(int(v) for v in value.strides), offset, id(owner), not value.flags.writeable

    def raw(self, owner, flavor):
        import numpy as np
        array = np.asarray(owner) if flavor == 'jax-value' else owner
        if flavor == 'jax-value' and not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return memoryview(array.ravel(order='K')).cast('B')

    def stamp(self, value):
        if type(value) is TypeSource: return type_information(value.cls)
        if type(value) is MappingProxyType:
            return [(self.token(k), self.token(v)) for k, v in value.items()]
        if is_dataclass(value) and not isinstance(value, type) and type(value) is not BufferSource:
            return [(field.name, self.token(getattr(value, field.name))) for field in fields(value)]
        if type(value) is BufferSource:
            digest = hashlib.sha256()
            data = self.raw(value.owner, value.flavor)
            for start in range(0, len(data), 1024**2):
                self.check(); digest.update(data[start:start + 1024**2])
            return len(data), digest.digest()
        flavor = self.flavor(value)
        if flavor: return self.geometry(value, flavor)
        return super().stamp(value)

    def visit(self, value):
        if type(value) is MappingProxyType or (is_dataclass(value) and not isinstance(value, type)):
            self.check()
            if id(value) in self.active: raise ValueError('Cyclic structural data')
            if id(value) in self.memo: return self.memo[id(value)][1]
            record_index = len(self.records)
            ref = ['ref', record_index]
            self.memo[id(value)] = (value, ref)
            self.records.append(None); self.active.add(id(value))
            self.watchers.append((value, self.stamp(value)))
            if type(value) is MappingProxyType:
                body = ['mappingproxy', [[self.visit(k), self.visit(v)] for k, v in value.items()]]
            else:
                authority = TypeSource(type(value))
                info = type_information(type(value))
                self.watchers.append((authority, info))
                body = ['native-dataclass', info, [[field.name, self.visit(getattr(value, field.name))] for field in fields(value)]]
            self.records[record_index] = body
            self.active.remove(id(value))
            return ref
        flavor = self.flavor(value)
        if flavor is None: return super().visit(value)
        self.check()
        if id(value) in self.memo: return self.memo[id(value)][1]
        shape, dtype, strides, offset, owner_id, readonly = self.geometry(value, flavor)
        record_index = len(self.records)
        ref = ['ref', record_index]
        self.memo[id(value)] = (value, ref)
        self.records.append(None)
        self.watchers.append((value, self.geometry(value, flavor)))
        if owner_id not in self.buffers:
            owner = value
            if flavor == 'numpy':
                import numpy as np
                while type(owner.base) is np.ndarray: owner = owner.base
            observed = BufferSource(owner, flavor)
            data = self.raw(owner, flavor)
            chunks, digest = [], hashlib.sha256()
            for start in range(0, len(data), 1024**2):
                self.check()
                part = bytes(data[start:start + 1024**2])
                digest.update(part); chunks.append(self.put(part))
            buffer_ref = ['ref', len(self.records)]
            self.records.append(['native-buffer', chunks, len(data)])
            self.buffers[owner_id] = buffer_ref
            self.watchers.append((observed, (len(data), digest.digest())))
        buffer_ref = self.buffers[owner_id]
        if flavor == 'jax-value':
            item = __import__('numpy').dtype(dtype).itemsize
            packed = []
            for dimension in reversed(shape):
                packed.append(item); item *= dimension
            strides = tuple(reversed(packed))
            self.storage_obligations.append('JAX physical device-buffer layout is not a model-value contract')
        self.records[record_index] = ['native-array', flavor, list(shape), dtype, list(strides), offset, buffer_ref, bool(readonly)]
        return ref

    def finish(self, source, anchor, before_final_validation=None):
        bundle = super().finish(source, anchor, before_final_validation)
        bundle['manifest']['storage_obligations'] = sorted(set(self.storage_obligations))
        return reseal(bundle=bundle)
