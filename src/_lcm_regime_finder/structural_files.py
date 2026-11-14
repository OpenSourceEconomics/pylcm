"""Confined immutable structural chunks; JSON headers never contain byte trees."""
from collections.abc import MutableMapping
import hashlib
import os
from pathlib import Path
import re


class FilePayloads(MutableMapping):
    def __init__(self, directory, keys=(), *, writable=False, reads=None):
        self.directory = Path(directory).resolve()
        self.keys_set = set(keys)
        self.writable = writable
        self.reads = reads

    def path(self, key):
        if type(key) is not str or re.fullmatch(r'p\d+', key) is None:
            raise ValueError('Invalid structural chunk name')
        path = self.directory / (key + '.bin')
        if path.is_symlink() or path.resolve().parent != self.directory:
            raise ValueError('Structural chunk escapes its store')
        return path

    def __getitem__(self, key):
        if key not in self.keys_set: raise KeyError(key)
        path = self.path(key)
        if self.reads is not None: path = self.reads.bind(path)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
            return stream.read()

    def __setitem__(self, key, data):
        if not self.writable or key in self.keys_set: raise ValueError('Structural chunks are write-once')
        if type(data) is not bytes: raise TypeError('Structural payload must be bytes')
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(key)
        temporary = path.with_suffix('.writing')
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
        self.keys_set.add(key)

    def __delitem__(self, key):
        raise ValueError('Structural chunks are immutable')

    def __iter__(self): return iter(sorted(self.keys_set))
    def __len__(self): return len(self.keys_set)


def load_store(manifest, directory, *, reads=None):
    from .structural_check import Reader
    chunks = FilePayloads(directory, manifest['blobs'], reads=reads)
    expected = {key + '.bin' for key in chunks}
    actual = {p.name for p in chunks.directory.iterdir() if p.name.endswith('.bin')}
    if expected != actual: raise ValueError('Structural chunk file inventory differs')
    bundle = {'manifest': manifest, 'payloads': chunks}
    reader = Reader(bundle)
    reader.validate_graph()
    return reader


def verify_store_again(reader):
    """One final input-integrity pass, not one pass per callback or scenario."""
    for key, descriptor in reader.doc['blobs'].items():
        data = reader.payloads[key]
        if len(data) != descriptor['bytes'] or hashlib.sha256(data).hexdigest() != descriptor['sha256']:
            raise ValueError('Structural input changed during analysis')
