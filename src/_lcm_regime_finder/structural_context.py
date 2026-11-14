"""Read-only access to independently checked structural generations."""
import weakref

from .structural_check import encoded


_STORES = weakref.WeakValueDictionary()


def register(reader):
    _STORES[reader.doc['root_digest']] = reader


def checked_reader(root):
    reader = _STORES.get(root)
    if reader is None: raise ValueError('Unverified structural store; load the bound capture first')
    return reader


def named_roots(reader):
    token = reader.doc['roots']
    if token[0] != 'ref': raise ValueError('Structural named roots must be a mapping')
    record = reader.doc['records'][token[1]]
    if record[0] != 'dict': raise ValueError('Structural named root inventory missing')
    result = {}
    for key, value in record[1]:
        if key[0] != 'str' or key[1] in result: raise ValueError('Invalid named structural root')
        result[key[1]] = value
    return result


def descriptor(reader, token):
    if token[0] == 'row': return {'type': 'tuple', 'length': int(token[1])}
    if token[0] != 'ref': raise ValueError('A structural root must retain object identity')
    record = reader.doc['records'][token[1]]
    kind = record[0]
    if kind.startswith('rows_'): return {'type': kind[5:], 'length': record[3]}
    if kind.startswith('items_'): return {'type': kind[6:], 'length': record[2]}
    if kind in ('list', 'tuple', 'set', 'frozenset', 'dict', 'mappingproxy'): return {'type': kind, 'length': len(record[1])}
    if kind == 'native-dataclass': return {'type': 'dataclass', 'native_type': record[1][0], 'fields': [row[0] for row in record[1][2]]}
    if kind == 'native-array': return {'type': 'array', 'shape': record[2], 'dtype': record[3]}
    if kind == 'bytearray': return {'type': 'bytearray', 'length': len(reader.blob(record[1]))}
    raise ValueError('Unsupported structural root descriptor')


def resolve_reference(value):
    if not isinstance(value, dict) or value.get('kind') != 'structural-reference': return None
    reader = _STORES.get(value.get('store_root'))
    if reader is None: raise ValueError('Unverified structural reference; load its complete capture/store first')
    token = named_roots(reader).get(value.get('root'))
    if token is None: raise ValueError('Unknown structural root')
    info = descriptor(reader, token)
    actual = {key: value[key] for key in info if key in value}
    if encoded(actual) != encoded(info): raise ValueError('Structural reference descriptor differs from checked source')
    return reader, token


def reference_info(value):
    resolved = resolve_reference(value)
    return descriptor(*resolved) if resolved is not None else None


def validate_references(snapshot):
    pending, roots, references = [snapshot], set(), []
    while pending:
        value = pending.pop()
        if isinstance(value, list): pending.extend(value)
        elif isinstance(value, dict):
            header = value.get('structural_store')
            if isinstance(header, dict) and isinstance(header.get('manifest'), dict):
                roots.add(header['manifest'].get('root_digest'))
            if value.get('kind') == 'structural-reference':
                references.append(value)
            else: pending.extend(value.values())
    for ref in references:
        if ref.get('store_root') not in roots: raise ValueError('Structural reference has no bound capture inventory')
        reference_info(ref)
