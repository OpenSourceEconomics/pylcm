"""One file-backed structural generation per original capture transaction."""
from copy import deepcopy
from dataclasses import is_dataclass, fields
import hashlib
import json
import errno
import os
from pathlib import Path
import shutil
import uuid

from .structural_arrays import NativeWriter
from .structural_check import Reader
from .structural_events import source_anchor, compare_source
from .structural_files import FilePayloads, load_store, verify_store_again
from .structural_transport import canonical
from .structural_context import register, reference_info, validate_references, checked_reader




class LoadedCapture(dict):
    """JSON mapping retaining its independently checked, immutable store context."""
    structure: Reader
    structures: tuple[Reader, ...]
    def __deepcopy__(self, memo):
        result = type(self)()
        memo[id(self)] = result
        result.update(deepcopy(dict(self), memo))
        result.structure = self.structure
        result.structures = getattr(self, 'structures', (self.structure,))
        return result


class CaptureStore:
    def __init__(self, destination, *, check, source_context, transaction=None):
        self.destination = Path(destination)
        self.parent = self.destination.with_name(self.destination.stem + '.structure')
        self.stage = self.parent / ('.writing-' + uuid.uuid4().hex)
        self.writer = NativeWriter(check=check)
        self.writer.payloads = FilePayloads(self.stage, writable=True)
        self.context, self.transaction, self.check = source_context, transaction, check
        self.roots, self.names, self.refs = {}, {}, []
        self.header, self.failure = None, None

    def reference(self, value):
        if id(value) not in self.names:
            name = 'S' + str(len(self.roots))
            self.writer.visit(value)
            self.roots[name] = value
            self.names[id(value)] = name
        name = self.names[id(value)]
        kind = 'array' if self.writer.flavor(value) else type(value).__name__
        if is_dataclass(value) and not isinstance(value, type): kind = 'dataclass'
        ref = {'kind': 'structural-reference', 'root': name, 'store_root': None, 'type': kind}
        if kind == 'array': ref.update(shape=[int(n) for n in value.shape], dtype=str(__import__('numpy').dtype(value.dtype).str))
        elif kind == 'dataclass': ref.update(native_type=type(value).__module__ + '.' + type(value).__qualname__, fields=[field.name for field in fields(value)])
        else: ref['length'] = len(value)
        self.refs.append(ref)
        return ref

    def finish(self):
        if self.header is not None: return self.header
        if not self.roots: return None
        bundle = self.writer.finish(self.roots, {})
        manifest = bundle['manifest']
        # Release producer lookup/observation work before the independent pass.
        self.writer.memo.clear(); self.writer.watchers.clear(); self.writer.rows.clear()
        self.writer.buffers.clear(); self.writer.content.clear()
        transaction = self.transaction or {'kind': 'UNPROVED', 'detail': 'Caller has not established quiescence'}
        anchor = source_anchor(self.roots, source_context=self.context, transaction=transaction, check=self.check)
        manifest['source_anchor'] = anchor
        manifest['source_consistency'] = ('CALLER_QUIESCENCE_CONTRACT' if transaction.get('kind') == 'exclusive-owner' and transaction.get('authority')
                                          else 'UNRESOLVED_QUIESCENCE')
        manifest['root_digest'] = hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != 'root_digest'})).hexdigest()
        self.stage.mkdir(parents=True, exist_ok=True)
        reader = load_store(manifest, self.stage)
        compare_source(reader, anchor, check=self.check)
        self.check()
        manifest_path = self.stage / 'manifest.json'
        temporary = manifest_path.with_suffix('.writing')
        temporary.write_bytes(canonical(manifest))
        temporary.replace(manifest_path)
        target = self.parent / manifest['root_digest']
        if target.exists():
            existing = json.loads((target / 'manifest.json').read_text())
            if existing != manifest: raise ValueError('Existing structural generation differs')
            load_store(existing, target)
            # Both complete generations have been checked against the same bytes.
            reader.payloads.directory = target
            shutil.rmtree(self.stage)
            final_path = target
        else:
            self.stage.rename(target); final_path = target
        # Existing inert BufferViews retain this payload object across sealing.
        reader.payloads.directory = final_path
        root = manifest['root_digest']
        register(reader)
        for ref in self.refs: ref['store_root'] = root
        self.header = {'schema': 'regime-structural-capture/v1', 'manifest': manifest,
                       'path': str(final_path.relative_to(self.destination.parent)),
                       'source_comparison': 'PASS', 'complete_roots': list(self.roots)}
        self.reader = reader
        self.roots.clear(); self.names.clear()
        return self.header


def capture_headers(value):
    pending, seen, headers = [value], set(), []
    while pending:
        item = pending.pop()
        if not isinstance(item, (dict, list)) or id(item) in seen: continue
        seen.add(id(item))
        if isinstance(item, list): pending.extend(item)
        else:
            header = item.get('structural_store')
            if isinstance(header, dict): headers.append(header)
            pending.extend(v for k, v in item.items() if k != 'structural_store')
    return headers


def capture_inputs(snapshot, snapshot_path, *, reads, fresh=False):
    """Resolve and register the exact manifest/chunks used by the loader."""
    from .structural_files import FilePayloads
    stores = []
    for index, header in enumerate(capture_headers(snapshot)):
        if header.get('schema') != 'regime-structural-capture/v1' or 'manifest' not in header:
            raise ValueError('Unsupported or incomplete structural capture format')
        # Structural captures deliberately anchor companions at the resolved file.
        context = reads.context(snapshot_path)
        store = context.child(header['path'], base='resolved_parent', confined=True)
        manifest_path = reads.child(snapshot_path, f'store.{index}.manifest',
            str(Path(header['path'])/'manifest.json'), base='resolved_parent', confined=True)
        manifest = reads.document(manifest_path, fresh=fresh)
        payloads = FilePayloads(store, header['manifest']['blobs'], reads=reads)
        for chunk in payloads:
            reads.child(snapshot_path, f'store.{index}.{chunk}',
                str(Path(header['path'])/(chunk+'.bin')), base='resolved_parent', confined=True)
            reads.bind(payloads.path(chunk))
        stores.append((header, manifest, payloads))
    return stores


def load_capture_structure(snapshot, snapshot_path, *, reads=None):
    if not capture_headers(snapshot): return snapshot
    if reads is None:
        from .input_reads import discover_inputs
        reads = discover_inputs([snapshot_path])
    readers = []
    for header, manifest, payloads in capture_inputs(snapshot, snapshot_path, reads=reads):
        if header.get('schema') != 'regime-structural-capture/v1' or 'manifest' not in header:
            raise ValueError('Unsupported or incomplete structural capture format')
        if canonical(manifest) != canonical(header['manifest']): raise ValueError('Structural manifest differs from bound capture')
        reader = load_store(manifest, payloads.directory, reads=reads)
        compare_source(reader, manifest['source_anchor'])
        register(reader); readers.append(reader)
    result = LoadedCapture(snapshot)
    result.structure, result.structures = readers[0], tuple(readers)
    validate_references(result)
    return result


def relocate_captures(value, output_directory):
    """Mirror checked stores without rewriting any already-committed metadata.

    Structural header locators participate in source/analysis/decision hashes.
    Publication must preserve those strings, not relocate them after checking.
    Each target is confined to the output directory and independently checked.
    """
    headers = [header for header in capture_headers(value) if 'manifest' in header]
    if not headers: return value
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    for header in headers:
        root = header['manifest']['root_digest']
        reader = checked_reader(root)
        if canonical(header['manifest']) != canonical(reader.doc):
            raise ValueError('Unverified structural header')
        relative = Path(header['path'])
        target = (output / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(output) or target == output:
            raise ValueError('Structural output locator escapes its capture directory')
        key = (root, target)
        if key in seen: continue
        seen.add(key)
        source = reader.payloads.directory
        if source == target:
            continue
        artifacts = getattr(reader, 'artifacts', {})
        reader.artifacts = artifacts
        # One source may occur at several committed relative locators.
        if target in artifacts:
            continue
        if target.exists():
            if canonical(json.loads((target / 'manifest.json').read_text())) != canonical(reader.doc):
                raise ValueError('Existing output store differs from original input')
            load_store(reader.doc, target)
        else:
            stage = target.with_name('.writing-' + uuid.uuid4().hex)
            stage.mkdir(parents=True)
            for chunk in reader.payloads:
                source_path = reader.payloads.path(chunk)
                destination = stage / source_path.name
                try: os.link(source_path, destination, follow_symlinks=False)
                except OSError as error:
                    if error.errno != errno.EXDEV: raise
                    shutil.copyfile(source_path, destination, follow_symlinks=False)
            (stage / 'manifest.json').write_bytes(canonical(reader.doc))
            load_store(reader.doc, stage)
            stage.rename(target)
        artifacts[target] = FilePayloads(target, reader.doc['blobs'])
    return value


def final_input_check(snapshot):
    if isinstance(snapshot, LoadedCapture):
        for reader in getattr(snapshot, 'structures', (snapshot.structure,)):
            verify_store_again(reader)
            for payloads in getattr(reader, 'artifacts', {}).values():
                for key, descriptor in reader.doc['blobs'].items():
                    data = payloads[key]
                    if len(data) != descriptor['bytes'] or hashlib.sha256(data).hexdigest() != descriptor['sha256']:
                        raise ValueError('Structural output artifact changed during analysis')


def check_capture_chunks(snapshot_path, *, reads=None):
    """Final parent-side integrity check without repeating source interpretation."""
    from types import SimpleNamespace
    if reads is None:
        from .input_reads import discover_inputs
        reads = discover_inputs([snapshot_path])
    for header, manifest, payloads in capture_inputs(reads.document(snapshot_path, fresh=True), snapshot_path, reads=reads, fresh=True):
        if canonical(manifest) != canonical(header['manifest']):
            raise ValueError('Structural input manifest changed')
        verify_store_again(SimpleNamespace(doc=manifest, payloads=payloads))
