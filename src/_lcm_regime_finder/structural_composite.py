"""Check stored composite joins, without solving or reconstructing reachability."""
import json
from array import array
import math
import re
import sys

from .structural_context import resolve_reference


def tokens(reader, token):
    record = reader.doc['records'][token[1]]
    kind = record[0]
    if kind.startswith('rows_'):
        return (['row', record[1], int(i)] for i in reader.vector(record[2]))
    if kind.startswith('items_'):
        return iter(json.loads(reader.blob(record[1])))
    if kind in ('tuple', 'list', 'set', 'frozenset'):
        return iter(record[1])
    raise ValueError('Expected stored row sequence')


def row_value(reader, token):
    if token[0] == 'row':
        return reader.rows[token[1]][reader.handles[token[1]][token[2]]]
    value = reader.read(token)
    if type(value) is not tuple or any(type(x) is not int for x in value):
        raise ValueError('Composite support row must contain integer codes')
    return value


def array_value(reference):
    reference = reference.get('structural_ref', reference)
    reader, token = resolve_reference(reference)
    view = reader.read(token)
    match = re.fullmatch(r'([<>=|])i([1248])', view.dtype)
    if match is None: raise NotImplementedError('Composite arrays require signed integer storage')
    endian, size = match.group(1), int(match.group(2))
    stride, strides = size, []
    for dimension in reversed(view.shape):
        strides.insert(0, stride); stride *= dimension
    count = math.prod(view.shape)
    if count and tuple(strides) != view.strides:
        raise NotImplementedError('Composite join checker requires contiguous C-order views')
    # Read only original stored bytes; never enumerate a new Cartesian grid.
    data = b''.join(view.buffer.iter_bytes())
    words = array({1: 'b', 2: 'h', 4: 'i', 8: 'q'}[size])
    if words.itemsize != size: raise NotImplementedError('Unsupported host integer layout')
    words.frombytes(memoryview(data)[view.offset:view.offset+count*size])
    if size > 1 and ((endian == '>' and sys.byteorder == 'little') or
                     (endian == '<' and sys.byteorder == 'big')): words.byteswap()
    if len(words) != count: raise ValueError('Truncated composite array')
    return words, view.shape


def check_composite(metadata, *, missing_code, check=lambda: None):
    """Validate inventory/table/encoder consistency; economics stays unresolved."""
    values = metadata.get('fields', {})
    reachable = values.get('reachable', {})
    resolved = resolve_reference(reachable.get('structural_backing'))
    if resolved is None:
        return {'status': 'UNRESOLVED', 'reason': 'Complete typed reachable-state backing unavailable'}
    reader, token = resolved
    record = reader.doc['records'][token[1]]
    if record[0] != 'native-dataclass': raise ValueError('Reachability backing is not a typed dataclass')
    fields = dict(record[2])
    members = reader.read(fields['state_names'])
    carriers = reader.read(fields['carriers'])
    if members != tuple(values['members']) or carriers != frozenset(values['carriers']):
        raise ValueError('Composite member order/carrier join changed')
    union = [row_value(reader, ref) for ref in tokens(reader, fields['union'])]
    if any(len(row) != len(members) for row in union) or union != sorted(set(union)):
        raise ValueError('Composite union is not a sorted distinct member-width domain')
    support = set()
    nodes = reader.doc['records'][fields['by_node'][1]]
    if nodes[0] != 'mappingproxy': raise ValueError('Typed node inventory missing')
    for owner, rows in nodes[1]:
        check()
        period, regime = reader.read(owner)
        if type(period) is not int or period < 0 or regime not in carriers:
            raise ValueError('Composite support has an invalid period/carrier owner')
        for ref in tokens(reader, rows):
            row = row_value(reader, ref)
            if len(row) != len(members): raise ValueError('Node support row width changed')
            support.add(row)
    if support != set(union): raise ValueError('Composite union differs from original per-node supports')
    try:
        table, shape = array_value(values['table'])
        encoding, encoding_shape = array_value(values['encoding'])
    except NotImplementedError as error:
        return {'status': 'UNRESOLVED', 'reason': str(error)}
    if shape != (len(union), len(members)):
        raise ValueError('Composite code table geometry changed')
    width = len(members)
    for code, row in enumerate(union):
        if code % 8192 == 0: check()
        if tuple(table[code*width:(code+1)*width]) != row:
            raise ValueError('Composite table no longer matches union code order')
    if len(encoding_shape) != width:
        raise ValueError('Unsupported composite encoding geometry')
    if any(any(x < 0 or x >= n for x, n in zip(row, encoding_shape)) for row in union):
        raise ValueError('Member support falls outside original encoding shape')
    valid_count = 0
    for location, code in enumerate(encoding):
        if location % 262144 == 0: check()
        if code == missing_code: continue
        valid_count += 1
        if not 0 <= code < len(union):
            raise ValueError('Encoder has an out-of-range composite code')
        expected = 0
        for member, dimension in zip(union[code], encoding_shape):
            expected = expected*dimension + member
        if location != expected: raise ValueError('Encoder member/code correspondence changed')
    if valid_count != len(union): raise ValueError('Encoder omitted or duplicated a composite code')
    return {'status': 'PASS', 'scope': 'Stored node/union/member/table/encoder consistency only',
            'nodes': len(nodes[1]), 'rows': len(union), 'members': len(members),
            'economic_reachability': 'UNRESOLVED', 'decoder_mask_semantics': 'UNRESOLVED'}
