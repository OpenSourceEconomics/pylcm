"""One contextual read closure shared by input loaders and write admission.

Discovery reads declarations, manifests and source text, never native chunks or
callbacks. After freezing, loaders can open only registered input identities.
The single-owner contract excludes concurrent namespace mutation.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType


def absolute_name(path):
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


@dataclass(frozen=True)
class InputIdentity:
    supplied: Path
    resolved: Path
    inode: tuple[int, int]


@dataclass(frozen=True)
class DocumentContext:
    supplied: Path
    resolved: Path

    def child(self, locator, *, base, confined=False):
        if base not in {'supplied_parent', 'resolved_parent'}:
            raise ValueError(f'Unknown input resolution context: {base}')
        parent = self.supplied.parent if base == 'supplied_parent' else self.resolved.parent
        relative = Path(locator)
        path = relative if relative.is_absolute() else parent / relative
        if confined and (relative.is_absolute() or not path.resolve().is_relative_to(parent.resolve())):
            raise ValueError('Input locator escapes its reader context')
        return path


@dataclass(frozen=True)
class InputReads:
    identities: tuple[InputIdentity, ...]
    paths: object
    documents: object
    texts: object
    edges: tuple

    def bind(self, path):
        name = absolute_name(path)
        try:
            return self.paths[name]
        except KeyError:
            raise ValueError(f'Unregistered input read: {name}') from None

    def context(self, path):
        return DocumentContext(absolute_name(path), self.bind(path))

    def read_text(self, path):
        resolved = self.bind(path)
        return self.texts[resolved] if resolved in self.texts else resolved.read_text()

    def read_bytes(self, path):
        return self.bind(path).read_bytes()

    def document(self, path, *, fresh=False):
        resolved = self.bind(path)
        if fresh: return json.loads(resolved.read_text())
        return self.documents[resolved] if resolved in self.documents else json.loads(self.read_text(path))

    def child(self, document, label, locator, *, base, confined=False):
        path = self.context(document).child(locator, base=base, confined=confined)
        self.bind(path)
        return path


class ReadDiscovery:
    def __init__(self):
        self.paths, self.documents, self.texts, self.identities = {}, {}, {}, {}
        self.edges = []

    def bind(self, path):
        name = absolute_name(path)
        if name not in self.paths:
            resolved = name.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f'Original input is not a regular file: {name}')
            stat = resolved.stat()
            self.paths[name] = resolved
            self.identities[name] = InputIdentity(name, resolved, (stat.st_dev, stat.st_ino))
        return self.paths[name]

    context = InputReads.context

    def read_text(self, path):
        resolved = self.bind(path)
        if resolved not in self.texts:
            self.texts[resolved] = resolved.read_text()
        return self.texts[resolved]

    def read_bytes(self, path):
        return self.bind(path).read_bytes()

    def document(self, path, *, fresh=False):
        resolved = self.bind(path)
        if fresh: return json.loads(resolved.read_text())
        if resolved not in self.documents:
            self.documents[resolved] = json.loads(self.read_text(path))
        return self.documents[resolved]

    def child(self, document, label, locator, *, base, confined=False):
        path = self.context(document).child(locator, base=base, confined=confined)
        self.bind(path)
        edge = (absolute_name(document), label, absolute_name(path), base)
        if edge not in self.edges:
            self.edges.append(edge)
        return path

    def freeze(self):
        return InputReads(tuple(self.identities.values()),
                          MappingProxyType(dict(self.paths)),
                          MappingProxyType(dict(self.documents)),
                          MappingProxyType(dict(self.texts)), tuple(self.edges))


def discover_inputs(paths):
    """Use the same format adapters and context functions as actual loaders."""
    from .structural_store import capture_inputs
    reads, visited = ReadDiscovery(), set()
    def visit(path):
        context = reads.context(path)
        # Wrapper links use the supplied parent; structural links resolve first.
        key = (context.resolved, context.supplied.parent)
        if key in visited:
            return
        visited.add(key)
        try:
            data = reads.document(path)
        except (ValueError, UnicodeError):
            return  # The actual decoder will reject malformed input, before work.
        if not isinstance(data, dict):
            return
        if data.get('schema') == 'interned-json-cost-instance/v1':
            from .symbolic_frontier import load_packet
            load_packet(path=path, reads=reads)
        if data.get('format') == 'lossless-json-fragments-1':
            from read_fragments import read
            data = read(Path(path), reads=reads)
        if data.get('adapter') in {'pylcm_model_graph', 'pylcm_declarations'}:
            for label in ('declaration_spec', 'snapshot'):
                if label in data:
                    visit(reads.child(path, label, data[label], base='supplied_parent'))
        elif data.get('adapter') == 'pylcm_ast':
            from .source_adapter import manifest_source, register_source_inputs
            source, _ = manifest_source(path, reads=reads)
            register_source_inputs(source, reads)
        capture_inputs(data, path, reads=reads)
    for path in paths:
        visit(path)
    return reads.freeze()
