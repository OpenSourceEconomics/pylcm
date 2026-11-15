"""Bounded, source-bound capture of the actual callable graph.

One record represents one callable and its binding environment in a capture
transaction. Edges are references, never duplicated identity subtrees. Capture
is inspection, not semantic certification: unsupported library targets and
mutation remain explicit obligations. Neither model callbacks nor wrappers run.
"""
from __future__ import annotations

import ast
import builtins
from collections import deque
from collections.abc import Hashable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, asdict, is_dataclass, fields as dataclass_fields
from fractions import Fraction
import functools
import hashlib
import inspect
import json
import marshal
import os
from pathlib import Path
import sys
import sysconfig
import textwrap
import time
import types
from typing import Protocol

from .records import JSONValue
try:
    import resource
except ImportError:
    resource = None


class _ValueNormalizer(Protocol):
    """Normalize captured metadata using the existing depth and identity boundary."""

    def __call__(
        self,
        value: object,  # noqa: PAN001 - metadata capture accepts unrelated user objects
        *,
        depth: int = 0,
        seen: frozenset[int] = frozenset(),
    ) -> JSONValue: ...


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class CaptureLimits:
    seconds: float = 900
    host_gib: float = 8
    max_records: int = 32768
    max_record_bytes: int = 64 * 1024**2
    max_metadata_items: int = 2_000_000

    def __post_init__(self):
        for field in dataclass_fields(self):
            if getattr(self, field.name) is None:
                object.__setattr__(self, field.name, float('inf'))
        if self.seconds < 0 or self.host_gib <= 0 or min(self.max_records, self.max_record_bytes, self.max_metadata_items) < 1:
            raise ValueError('Capture limits require nonnegative seconds and positive size/memory bounds')


class CaptureIncomplete(RuntimeError):
    """Capture stopped without asserting a complete original declaration."""


_CAPTURE_SESSION: ContextVar[CaptureSession | None] = ContextVar('regime_finder_capture_session', default=None)


def current_capture_session():
    return _CAPTURE_SESSION.get()


def callable_record_payload(record):
    """The exact finite local-hash contract (reference links are not unfolded)."""
    omitted = {'record_sha256', 'identity_sha256', 'capture_graph_sha256',
               'callable_records', 'capture_report'}
    def strip(value):
        if isinstance(value, dict):
            if value.get('kind') == 'callable-reference':
                return {'kind': 'callable-reference', 'capture_ref': value.get('capture_ref')}
            return {key: strip(item) for key, item in value.items()}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value
    return {key: strip(value) for key, value in record.items() if key not in omitted}


def callable_record_digest(record):
    return _digest(callable_record_payload(record))


def resolve_callable(value, records=None):
    """Resolve a finite reference; integrity failure is an explicit unknown.

    The independent checker implements its own table integrity checks. This
    utility is the producer-side accessor, not a semantic proof routine.
    """
    if not isinstance(value, dict) or value.get('kind') != 'callable-reference':
        return value
    records = records or {}
    record = records.get(value.get('capture_ref'))
    if (not isinstance(record, dict) or
            record.get('capture_ref') != value.get('capture_ref') or
            not value.get('identity_sha256') or
            record.get('identity_sha256') != value.get('identity_sha256') or
            record.get('capture_graph_sha256') != value.get('capture_graph_sha256') or
            callable_record_digest(record) != record.get('record_sha256')):
        return {'kind': 'unresolved', 'reason': 'Missing or inconsistent shared callable record'}
    return record


def resolve_binding(value, records=None, *, memo=None, active=None):
    """Decode finite captured binding references, refusing cycles/unknown identity.

    A memo may be reused for one immutable table. This does not assign semantics
    to a callable reference: only a typed binding-record is decoded as data.
    """
    records = records or {}
    memo = {} if memo is None else memo
    active = set() if active is None else active
    if isinstance(value, dict) and value.get('kind') == 'callable-reference':
        ref = value.get('capture_ref')
        record = resolve_callable(value, records)
        if record.get('kind') != 'binding-record':
            return value
        if record.get('capture_status', 'COMPLETE') != 'COMPLETE':
            raise ValueError('Captured binding changed or is incomplete: ' + str(ref))
        if ref in active:
            raise ValueError('Cyclic captured binding requires an explicit semantic contract')
        if ref not in memo:
            active.add(ref)
            try:
                decoded = resolve_binding(record['value'], records, memo=memo, active=active)
                if record.get('binding_type') == 'builtins.tuple':
                    decoded = tuple(decoded)
                elif record.get('binding_type') in {'builtins.set', 'builtins.frozenset'}:
                    decoded = frozenset(decoded) if record['binding_type'] == 'builtins.frozenset' else set(decoded)
                memo[ref] = decoded
            finally:
                active.remove(ref)
        return memo[ref]
    if isinstance(value, list):
        return [resolve_binding(item, records, memo=memo, active=active) for item in value]
    if isinstance(value, dict):
        return {key: resolve_binding(item, records, memo=memo, active=active) for key, item in value.items()}
    return value


def validate_callable_records(records):
    """Validate serialization identity only, never callback semantics."""
    if not isinstance(records, dict):
        return False
    local = {}
    for ref, record in records.items():
        if not isinstance(record, dict) or record.get('capture_ref') != ref:
            return False
        local[ref] = callable_record_digest(record)
        if local[ref] != record.get('record_sha256'):
            return False
    graph = _digest(local)
    for ref, record in records.items():
        expected = _digest({'capture_ref': ref, 'record_sha256': local[ref], 'capture_graph_sha256': graph})
        if record.get('capture_graph_sha256') != graph or record.get('identity_sha256') != expected:
            return False
        pending = list(record.values())
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if value.get('kind') == 'callable-reference':
                    target = records.get(value.get('capture_ref'), {})
                    if (target.get('identity_sha256') != value.get('identity_sha256') or
                            value.get('capture_graph_sha256') != graph):
                        return False
                else:
                    pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
    return True


def _file_stamp(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _code_digest(value):
    code = getattr(value, '__code__', None)
    return hashlib.sha256(marshal.dumps(code)).hexdigest() if code is not None else None


def _live_token(value):
    """Shallow transaction observation; child objects get their own records.

    IDs are only in-process mutation tokens. They never stand in for serialized
    semantic identities. In particular, this function does not recursively copy
    a large Tables object once per callback that closes over it.
    """
    if value is None or type(value) in (str, bool, int, float, bytes):
        return (type(value).__name__, value)
    if inspect.ismethod(value):
        return ('method', id(value.__func__), _code_digest(value.__func__), id(value.__self__))
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        return ('callable', id(value), _code_digest(value))
    return ('object', id(value), type(value).__module__, type(value).__qualname__)


def _binding_stamp(value):
    from .current_api_contract import inspect_native_floatnd_alias
    annotation = inspect_native_floatnd_alias(value)
    configuration = _beartype_input_config(value)
    if annotation is not None or configuration is not None:
        return ("checked-input-type", _digest(annotation), _digest(configuration))
    if isinstance(value, Mapping):
        return ('mapping', _observation_digest((_live_token(key), _live_token(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, _observation_digest(_live_token(item) for item in value))
    if isinstance(value, (set, frozenset)):
        # Iteration is stable for an unmodified in-process set. A reordered
        # mutation may conservatively invalidate capture; this is not a
        # serialized identity or a claim about set ordering semantics.
        return ('set', _observation_digest(_live_token(item) for item in value))
    module = type(value).__module__
    if module.startswith('numpy') and hasattr(value, 'shape') and hasattr(value, 'dtype'):
        if int(value.size) > 8192:
            return ('unresolved-large-array', id(value), tuple(value.shape), str(value.dtype))
        return ('array', tuple(value.shape), str(value.dtype), hashlib.sha256(value.tobytes(order='C')).hexdigest())
    if module.startswith(('jax', 'jaxlib')) and hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return ('immutable-device-array', id(value), tuple(value.shape), str(value.dtype))
    if is_dataclass(value):
        return ('dataclass', tuple((field.name, _live_token(getattr(value, field.name))) for field in dataclass_fields(value)))
    if hasattr(value, '__dict__'):
        return ('object', _binding_stamp(vars(value)))
    return _live_token(value)


def _observation_digest(tokens):
    """Observe large containers without constructing a second live data tree."""
    digest = hashlib.sha256()
    session = current_capture_session()
    count = 0
    for count, token in enumerate(tokens, 1):
        if session is not None and count % 256 == 1:
            session.check()
            if count > session.limits.max_metadata_items:
                raise CaptureIncomplete('Binding observation exceeds metadata item bound')
        encoded = json.dumps(token, sort_keys=True, separators=(',', ':'), default=str).encode()
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
    return count, digest.hexdigest()


def _advertised_signature_stamp(value):
    """Observe immutable signature contents, not a descriptor's temporary object.

    Defaults/annotations retain their transaction binding tokens. Arbitrary
    advertised objects remain opaque identities; this does not reconcile an
    advertised signature with the actual implementation or certify a native law.
    """
    if type(value) is inspect.Signature:
        return ('signature', tuple(
            (name, parameter.kind.name, _live_token(parameter.default),
             _live_token(parameter.annotation))
            for name, parameter in value.parameters.items()),
            _live_token(value.return_annotation))
    return _live_token(value)


def _callable_stamp(value):
    partial = value if isinstance(value, functools.partial) else None
    raw = partial.func if partial is not None else value
    receiver = raw.__self__ if inspect.ismethod(raw) else None
    raw = raw.__func__ if inspect.ismethod(raw) else raw
    if not inspect.isfunction(raw) and not inspect.isbuiltin(raw) and not isinstance(raw, type):
        receiver = raw
        raw = inspect.getattr_static(type(raw), '__call__', raw)
    bindings = {}
    boundary = _library_boundary(raw)
    if inspect.isfunction(raw) and not boundary:
        closure = inspect.getclosurevars(raw)
        bindings = {**closure.builtins, **closure.globals, **closure.nonlocals}
    receiver_token = ('opaque-receiver', id(receiver)) if boundary else (_binding_stamp(receiver) if receiver is not None else None)
    return (_code_digest(raw), _binding_stamp(getattr(raw, '__defaults__', None)),
            _binding_stamp(getattr(raw, '__kwdefaults__', None)),
            _binding_stamp(bindings), receiver_token,
            _binding_stamp(partial.args if partial else None),
            _binding_stamp(partial.keywords if partial else None),
            id(getattr(value, '__wrapped__', None)),
            _advertised_signature_stamp(getattr(value, '__signature__', None)))


class CaptureSession:
    """Linear graph capture with bounded metadata and transaction validation."""
    def __init__(self, limits=None):
        self.structural_store = None
        self.declaration_interfaces = None
        self.limits = limits if isinstance(limits, CaptureLimits) else CaptureLimits(**(limits or {}))
        self.started = time.perf_counter()
        self.records = {}
        self.memo = {}
        self.binding_memo = {}
        self.binding_expansion = False
        self.queue = deque()
        self.references = []
        self.files = {}
        self.watchers = []
        self.processing = False
        self.finished = False
        self.record_bytes = 0
        self.metadata_items = 0
        self.metadata_bytes = 0
        self.source_bytes = 0
        self.peak_rss = 0
        self.peak_self_rss = 0
        self.tree_rss_samples = 0
        self.tree_rss_unavailable_samples = 0
        self.last_memory_sample = -1.0
        self.obligations = []
        self.exhaustion_reason = None
        self.suspend_callbacks = False

    def check(self, *, force=False):
        elapsed = time.perf_counter() - self.started
        reason = None
        if elapsed >= self.limits.seconds:
            reason = 'elapsed_time'
        elif self.metadata_items > self.limits.max_metadata_items:
            reason = 'metadata_items'
        elif self.record_bytes > self.limits.max_record_bytes or self.metadata_bytes > self.limits.max_record_bytes:
            reason = 'record_bytes'
        if force or elapsed - self.last_memory_sample >= 0.05:
            from .resources import _tree_rss
            try:
                rss, _ = _tree_rss(os.getpid())
            except OSError:
                rss = 0
            if rss > 0:
                self.tree_rss_samples += 1
            else:
                self.tree_rss_unavailable_samples += 1
            # A restricted executor may expose /proc but hide every PID. Zero
            # is then missing telemetry, not evidence of zero resident memory.
            own_peak = 0
            if resource is not None:
                own_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform != 'darwin':
                    own_peak *= 1024
            self.peak_self_rss = max(self.peak_self_rss, own_peak)
            observed = rss if rss > 0 else own_peak
            self.peak_rss = max(self.peak_rss, observed)
            # A slow observation must not immediately schedule another one.
            self.last_memory_sample = time.perf_counter() - self.started
            if self.last_memory_sample >= self.limits.seconds and reason is None:
                reason = 'elapsed_time'
            if observed > self.limits.host_gib * 1024**3:
                reason = 'process_tree_memory'
        if reason:
            self.exhaustion_reason = self.exhaustion_reason or reason
            if reason not in self.obligations:
                self.obligations.append(reason)
            raise CaptureIncomplete(reason)

    def file_digest(self, path):
        if not path:
            return None
        try:
            key = str(Path(path).resolve())
            stamp = _file_stamp(key)
            if key in self.files:
                before, digest = self.files[key]
                if before != stamp:
                    raise CaptureIncomplete('Source file changed during capture: ' + key)
                return digest
            self.check()
            if stamp[2] > self.limits.max_record_bytes:
                raise CaptureIncomplete('Source file exceeds capture byte bound: ' + key)
            payload = Path(key).read_bytes()
            if _file_stamp(key) != stamp:
                raise CaptureIncomplete('Source file changed while reading: ' + key)
            digest = hashlib.sha256(payload).hexdigest()
            self.files[key] = (stamp, digest)
            self.source_bytes += len(payload)
            return digest
        except (OSError, TypeError):
            return None

    def intern(self, value, normalize):
        self.check()
        # Attribute lookup creates fresh bound-method objects. The actual
        # function+receiver pair is their callable identity, not the wrapper ID.
        key = ((id(value.__func__), id(value.__self__), id(normalize)) if inspect.ismethod(value)
               else (id(value), id(normalize)))
        if key in self.memo:
            return self.records[self.memo[key][1]]
        if len(self.records) >= self.limits.max_records:
            raise CaptureIncomplete('max_records')
        ref = 'C' + str(len(self.records))
        record = {'kind': 'callable', 'capture_ref': ref, 'capture_status': 'INCOMPLETE',
                  'arguments': [], 'variadic': True, 'obligations': ['Capture record pending']}
        self.memo[key] = (value, ref, normalize)  # retain objects against ID reuse
        self.records[ref] = record
        self.queue.append(('callable', value, normalize, record))
        return record

    def reference(self, value, normalize):
        record = self.intern(value, normalize)
        ref = {'kind': 'callable-reference', 'capture_ref': record['capture_ref']}
        self.references.append(ref)
        return ref

    def binding_value(self, value, normalize):
        if value is (0).__class__:
            return self.reference(value, normalize)
        if callable(value) and not isinstance(value, type):
            return self.reference(value, normalize)
        if not self.structured_binding(value):
            return normalize(value, depth=0, seen=frozenset())
        self.check()
        key = (id(value), id(normalize))
        if key not in self.binding_memo:
            if len(self.records) >= self.limits.max_records:
                raise CaptureIncomplete('max_records')
            ref = 'B' + str(len(self.records))
            record = {'kind': 'binding-record', 'capture_ref': ref, 'capture_status': 'INCOMPLETE',
                      'obligations': ['Binding record pending']}
            self.records[ref] = record
            self.binding_memo[key] = (value, ref, normalize)
            self.queue.append(('binding', value, normalize, record))
        ref = {'kind': 'callable-reference', 'capture_ref': self.binding_memo[key][1]}
        self.references.append(ref)
        return ref

    @staticmethod
    def structured_binding(value):
        return not (value is None or isinstance(value, (str, bytes, bool, int, float, Fraction, range, types.ModuleType, type)))

    def watch(self, getter, before, record, description):
        self.watchers.append((getter, before, record, description))

    def drain(self):
        if self.processing:
            return
        self.processing = True
        try:
            while self.queue:
                self.check()
                kind, value, normalize, record = self.queue.popleft()
                observer = _callable_stamp if kind == 'callable' else _binding_stamp
                before = observer(value)
                self.watch(lambda value=value, observer=observer: observer(value), before, record, 'Callable/binding mutation')
                first_watcher = len(self.watchers)
                if kind == 'binding':
                    self.binding_expansion = True
                    try:
                        captured = {'kind': 'binding-record', 'binding_type': type(value).__module__ + '.' + type(value).__qualname__,
                                    'value': normalize(value, depth=0, seen=frozenset()), 'obligations': []}
                    finally:
                        self.binding_expansion = False
                    from .current_api_contract import inspect_native_floatnd_alias
                    annotation = inspect_native_floatnd_alias(value)
                    configuration = _beartype_input_config(value)
                    if annotation is not None:
                        captured["native_array_annotation"] = annotation
                    if configuration is not None:
                        captured["beartype_input_configuration"] = configuration
                else:
                    captured = _capture_record(value, normalize, self)
                for index in range(first_watcher, len(self.watchers)):
                    getter, observed, watched, description = self.watchers[index]
                    if watched is captured:
                        self.watchers[index] = (getter, observed, record, description)
                ref = record['capture_ref']
                record.clear()
                record.update(captured, capture_ref=ref, capture_status='COMPLETE')
                self.record_bytes += len(json.dumps(callable_record_payload(record), sort_keys=True, separators=(',', ':'), default=str).encode())
        finally:
            self.processing = False

    def finish(self):
        if self.finished:
            return
        self.finished = True
        try:
            self.check(force=True)
            if self.structural_store is not None:
                self.structural_store.finish()
            for getter, before, record, description in self.watchers:
                self.check()
                if getter() != before:
                    record['capture_status'] = 'UNRESOLVED_MUTATION'
                    record.setdefault('obligations', []).append(description + ' during capture; retry on an immutable source')
                    self.obligations.append(description)
            for path, (stamp, _) in self.files.items():
                if _file_stamp(path) != stamp:
                    raise CaptureIncomplete('Source file changed during capture: ' + path)
        except (CaptureIncomplete, OSError, ValueError, RuntimeError, RecursionError, LookupError, AttributeError, TypeError) as exc:
            self.obligations.append(str(exc))
            for record in self.records.values():
                record['capture_status'] = 'INCOMPLETE'
                record.setdefault('obligations', []).append('Capture transaction could not be validated: ' + str(exc))
        if self.queue or self.exhaustion_reason:
            self.obligations.append('Capture incomplete; unexpanded records do not establish dependency absence')
        reverse = {ref: set() for ref in self.records}
        for owner, record in self.records.items():
            pending = list(record.values())
            while pending:
                value = pending.pop()
                if isinstance(value, list):
                    pending.extend(value)
                elif isinstance(value, dict):
                    if value.get('kind') == 'callable-reference':
                        reverse.setdefault(value.get('capture_ref'), set()).add(owner)
                    else:
                        pending.extend(value.values())
        invalid = deque(ref for ref, record in self.records.items() if record.get('capture_status') != 'COMPLETE')
        notified = set(invalid)
        while invalid:
            child = invalid.popleft()
            for owner in reverse.get(child, ()):
                if owner in notified:
                    continue
                notified.add(owner)
                record = self.records[owner]
                record['capture_status'] = self.records[child].get('capture_status', 'INCOMPLETE')
                record.setdefault('obligations', []).append('Shared binding/callable mutation or incomplete capture at ' + child)
                invalid.append(owner)
        # The complete graph hash binds cycles without recursively hashing them.
        local = {}
        for ref, record in self.records.items():
            record['record_sha256'] = callable_record_digest(record)
            local[ref] = record['record_sha256']
        graph = _digest(local)
        for ref, record in self.records.items():
            record['capture_graph_sha256'] = graph
            record['identity_sha256'] = _digest({'capture_ref': ref, 'record_sha256': local[ref], 'capture_graph_sha256': graph})
        for ref in self.references:
            record = self.records[ref['capture_ref']]
            ref['identity_sha256'] = record['identity_sha256']
            ref['capture_graph_sha256'] = graph

    def report(self):
        tree_available = self.tree_rss_samples > 0 and self.tree_rss_unavailable_samples == 0
        return {'status': 'INCOMPLETE' if self.obligations or self.queue else 'COMPLETE',
                'elapsed_seconds': time.perf_counter() - self.started,
                'peak_process_tree_gib': self.peak_rss / 1024**3 if tree_available else None,
                'peak_observed_host_gib': self.peak_rss / 1024**3 if self.peak_rss else None,
                'peak_self_process_lifetime_gib': self.peak_self_rss / 1024**3 if self.peak_self_rss else None,
                'process_tree_rss_available': tree_available,
                'process_tree_rss_samples': self.tree_rss_samples,
                'process_tree_rss_unavailable_samples': self.tree_rss_unavailable_samples,
                'resource_status': 'SAMPLED_PROCESS_TREE' if tree_available else 'UNVERIFIED_PROCESS_TREE',
                'measurement': 'Cooperative 50ms process-tree RSS cadence measured after sample completion; forced checks may sample sooner. Transient unsampled peaks may be higher. Missing process-tree telemetry uses process-lifetime self ru_maxrss for the host guard, which may include native construction and does not certify descendant memory. Includes the already constructed Model resident in this process.',
                'record_count': len(self.records),
                'callable_record_count': sum(record.get('kind') != 'binding-record' for record in self.records.values()),
                'binding_record_count': sum(record.get('kind') == 'binding-record' for record in self.records.values()),
                'source_file_count': len(self.files),
                'source_bytes': self.source_bytes, 'serialized_record_bytes': self.record_bytes,
                'estimated_metadata_payload_bytes': self.metadata_bytes,
                'limits': {key: None if value == float('inf') else value for key, value in asdict(self.limits).items()}, 'exhaustion_reason': self.exhaustion_reason,
                'obligations': sorted(set(self.obligations))}


@contextmanager
def capture_session(*, limits=None):
    current = _CAPTURE_SESSION.get()
    if current is not None:
        yield current
        return
    session = CaptureSession(limits)
    token = _CAPTURE_SESSION.set(session)
    try:
        yield session
    finally:
        session.finish()
        _CAPTURE_SESSION.reset(token)


def _file_digest(path):
    session = _CAPTURE_SESSION.get()
    if session is not None:
        return session.file_digest(path)
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    except (OSError, TypeError):
        return None


def _closed_numeric_defaults(
    function: object,  # noqa: PAN001 - reject arbitrary captured callable implementations
) -> bool:
    """No opaque/default callback can hide behind a public numerical name."""
    return (isinstance(function, types.FunctionType) and not function.__closure__
            and all(any(type(item) is allowed for allowed in (type(None), bool, int, float, str, bytes))
                    for item in (*(function.__defaults__ or ()), *(function.__kwdefaults__ or {}).values())))


def _numeric_implementation_sealed(
    value: object,  # noqa: PAN001 - authenticate arbitrary captured numerical call targets
    module_name: str,
) -> bool:
    """Require genuine public objects and unchanged executable implementation.

    The native identity module anchors actual imported numeric objects/code;
    copying a name, signature or class spelling cannot enter those sets.
    """
    try:
        from _lcm.solution import fingerprint as native
        if module_name == 'numpy':
            if any(value is item for item in native._NUMPY_UFUNCS):
                return True
            if native._is_captured_numpy_array_function(value):
                return _closed_numeric_defaults(vars(value).get('_implementation'))
            return inspect.isbuiltin(value) and getattr(value, '__module__', '') == 'numpy'
        if native._is_captured_jax_python_function(value):
            return _closed_numeric_defaults(value)
        if native._is_captured_jax_pjit(value):
            return _closed_numeric_defaults(vars(value).get('_fun'))
        if not any(value is item for item in native._JAX_NUMPY_UFUNCS):
            return False
        from .current_api_contract import inspect_source_bound_function
        raw = inspect.getattr_static(type(value), '__call__', None)
        if not inspect.isfunction(raw):
            return False
        module = sys.modules.get(raw.__module__)
        filename = vars(module).get('__file__') if isinstance(module, types.ModuleType) else None
        if (not filename or inspect_source_bound_function(raw, module_name=raw.__module__,
                qualname=raw.__qualname__, file_sha256=_file_digest(filename)) is None):
            return False
        properties = vars(value).get('_ufunc__static_props')
        if type(properties) is not dict:
            return False
        for item in properties.values():
            if inspect.isfunction(item):
                if (not native._is_captured_jax_python_function(item)
                        or not _closed_numeric_defaults(item)):
                    return False
            elif callable(item):
                if (not native._is_captured_jax_pjit(item)
                        or not _closed_numeric_defaults(vars(item).get('_fun'))):
                    return False
            elif not any(type(item) is allowed for allowed in (type(None), bool, int, float, str)):
                return False
        return True
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _dtype_primitive(
    value: object,  # noqa: PAN001 - authenticate arbitrary captured dtype bindings
) -> Mapping[str, JSONValue] | None:
    """A finite genuine numeric dtype token, never a callable value contract."""
    try:
        from _lcm.solution.fingerprint import _is_native_numeric_type
    except ImportError:
        return None
    if not isinstance(value, type) or not _is_native_numeric_type(value):
        return None
    for module_name in ('numpy', 'jax.numpy'):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name in ('bool_', 'int32', 'int64', 'float32', 'float64'):
            if value is vars(module).get(name):
                return {'kind': 'primitive', 'operation': 'array_dtype',
                        'canonical': module_name + '.' + name, 'resolution': 'actual_object_identity',
                        'implementation_seal': 'captured-native-numeric-object-code/v1',
                        'library_module_sha256': _file_digest(getattr(module, '__file__', None))}
    return None


def _primitive(value):
    if value is (0).__class__:
        return {'kind': 'primitive', 'operation': 'builtin_int_type',
                'canonical': 'builtins.int', 'resolution': 'actual_object_identity'}
    if (value is builtins.isinstance and inspect.isbuiltin(value)
            and value.__module__ == 'builtins' and value.__name__ == 'isinstance'):
        return {'kind': 'primitive', 'operation': 'builtin_isinstance',
                'canonical': 'builtins.isinstance', 'resolution': 'actual_object_identity'}
    dtype = _dtype_primitive(value)
    if dtype is not None:
        return dtype
    for module_name in ('numpy', 'jax.numpy'):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attribute, operation in (('where', 'select'), ('logical_and', 'logical_and'), ('logical_or', 'logical_or'),
                ('asarray', 'asarray'), ('maximum', 'maximum'), ('minimum', 'minimum'), ('exp', 'exp'), ('clip', 'clip'),
                ('stack', 'stack'), ('sum', 'sum'), ('log', 'log'), ('sqrt', 'sqrt')):
            canonical = vars(module).get(attribute)
            defining_module = getattr(value, '__module__', '')
            prefix = 'numpy' if module_name == 'numpy' else 'jax'
            physical = value.__func__ if inspect.ismethod(value) else value
            user_wrapper = inspect.isfunction(physical) and _library_boundary(physical) is None
            # Do not let a callable object with a forged module/name become an
            # array constructor when that module binding has been overwritten.
            if (value is canonical and getattr(value, '__name__', None) == attribute and not user_wrapper and
                    (defining_module == prefix or defining_module.startswith(prefix + '.'))):
                sealed = _numeric_implementation_sealed(value, module_name)
                if operation not in {'select', 'logical_and', 'logical_or'} and not sealed:
                    continue
                return {'kind': 'primitive', 'operation': operation, 'canonical': module_name + '.' + attribute,
                        'resolution': 'actual_object_identity',
                        **({'implementation_seal': 'captured-native-numeric-object-code/v1'} if sealed else {}),
                        'library_module_sha256': _file_digest(getattr(module, '__file__', None))}
    return None


def _resolve(node, bindings):
    """Read existing namespaces; never execute properties or descriptors."""
    if isinstance(node, ast.Name):
        if node.id not in bindings:
            raise LookupError('Callee is local, dynamic, or absent from captured bindings')
        return bindings[node.id]
    if isinstance(node, ast.Attribute):
        parent = _resolve(node.value, bindings)
        if isinstance(parent, types.ModuleType):
            if node.attr not in vars(parent):
                raise LookupError('Module attribute is absent')
            return vars(parent)[node.attr]
        value = inspect.getattr_static(parent, node.attr)
        if isinstance(value, property):
            raise LookupError('A property cannot be resolved without execution')
        if isinstance(value, staticmethod):
            return value.__func__
        if inspect.isfunction(value) and not isinstance(parent, type):
            return types.MethodType(value, parent)
        if hasattr(value, '__get__') and not callable(value):
            raise LookupError('Descriptor binding requires execution')
        return value
    raise LookupError('Dynamic callee expression requires an explicit semantic obligation')


_LIBRARY_ROOTS = tuple({Path(root).resolve() for key in ('stdlib', 'platstdlib', 'purelib', 'platlib')
                        if (root := sysconfig.get_path(key))})


def _library_boundary(raw):
    """A pessimistic source boundary, never a module-name semantic whitelist."""
    try:
        source = inspect.getsourcefile(raw)
    except TypeError:
        source = None
    if not source:
        return 'Native or unavailable implementation source'
    path = Path(source).resolve()
    for root in _LIBRARY_ROOTS:
        if path.is_relative_to(root):
            return 'Installed-library implementation; supply an independent primitive contract'
    return None


def capture_callable(value, normalize, *, depth=0, seen=frozenset()):
    """Capture the actual object, with path/depth-independent shared records."""
    own = _CAPTURE_SESSION.get() is None
    with capture_session() as session:
        if session.suspend_callbacks:
            return {'kind': 'callable', 'arguments': [], 'variadic': True,
                    'capture_status': 'INCOMPLETE', 'source_unavailable': True,
                    'obligations': ['Original graph checkpoint omits optional callback expansion']}
        if session.processing:
            return session.reference(value, normalize)
        record = session.intern(value, normalize)
        session.drain()
    if own:
        # Copy only the root envelope; adding a table to its own table record
        # would create a Python/JSON cycle. Every table record stays shallow.
        return {**record, 'callable_records': session.records, 'capture_report': session.report()}
    return record


def _beartype_default_renames(
    actual: types.FunctionType,
    regenerated: types.FunctionType,
    public_names: frozenset[str],
) -> Mapping[str, str] | None:
    """Reconcile generated tuple names only after checking their bound type predicates."""
    left, right = actual.__code__, regenerated.__code__
    defaults, expected = actual.__kwdefaults__ or {}, regenerated.__kwdefaults__ or {}
    if (type(defaults) is not dict or type(expected) is not dict
            or len(defaults) != len(expected)
            or len(left.co_varnames) != len(right.co_varnames)
            or len(set(left.co_varnames)) != len(left.co_varnames)
            or len(set(right.co_varnames)) != len(right.co_varnames)):
        return None
    renames: dict[str, str] = {}
    for index, (name, other_name) in enumerate(zip(left.co_varnames, right.co_varnames, strict=True)):
        if name == other_name:
            continue
        value, other = defaults.get(name), expected.get(other_name)
        if (not left.co_argcount <= index < left.co_argcount + left.co_kwonlyargcount
                or not right.co_argcount <= index < right.co_argcount + right.co_kwonlyargcount
                or name in public_names or other_name in public_names
                or type(value) is not tuple or type(other) is not tuple
                or name != '__beartype_object_' + str(id(value))
                or other_name != '__beartype_object_' + str(id(other))
                or len(value) != len(other)
                or any(not isinstance(item, type) for item in value)
                or any(item is not other_item for item, other_item in zip(value, other, strict=True))):
            return None
        renames[name] = other_name
    return types.MappingProxyType(renames)


def _same_annotation_identity(left: object, right: object) -> bool:
    """Match regenerated builtin aliases by origin and identity-only argument leaves."""
    pending = [(left, right, 0, False)]
    active: set[tuple[int, int]] = set()
    visited = 0
    while pending:
        a, b, depth, leaving = pending.pop()
        pair = (id(a), id(b))
        if leaving:
            active.remove(pair)
            continue
        visited += 1
        if visited > 128:
            return False
        if a is b:
            continue
        if (depth >= 16 or type(a) is not types.GenericAlias or type(b) is not types.GenericAlias
                or a.__unpacked__ is not False or b.__unpacked__ is not False
                or a.__origin__ is not b.__origin__ or pair in active):
            return False
        args, other_args = a.__args__, b.__args__
        if (type(args) is not tuple or type(other_args) is not tuple
                or len(args) != len(other_args) or len(args) > 128):
            return False
        active.add(pair)
        pending.append((a, b, depth, True))
        pending.extend((x, y, depth + 1, False) for x, y in zip(args, other_args, strict=True))
    return True



def _beartype_input_config(
    value: object,  # noqa: PAN001 - reject arbitrary captured wrapper configuration values
) -> Mapping[str, JSONValue] | None:
    """Observe finite input-check fields and the exact builtin numeric tower."""
    from beartype import BeartypeConf, BeartypeStrategy
    from beartype._util.kind.maplike.utilmapfrozen import FrozenDict
    from beartype.roar import BeartypeCallHintParamViolation
    if type(value) is not BeartypeConf:
        return None
    overrides, facts = value.hint_overrides, None
    tower = value.is_pep484_tower
    tower_module = sys.modules.get('beartype._conf._confoverrides')
    tower_file = vars(tower_module).get('__file__') if type(tower_module) is types.ModuleType else None
    tower_source = (_file_digest(tower_file) if tower is True and isinstance(tower_file, str)
                    and not any(part.casefold() in {"econ_data", "hrs"} for part in Path(tower_file).parts)
                    and not Path(tower_file).is_symlink() else None)
    if type(overrides) is FrozenDict:
        if dict.__len__(overrides) == 0:
            facts = []
        elif tower is True and tower_source == 'f203b27517f63c7b3185d4a721cf94635cb163d823675afb7c96acf644063520':
            entries = tuple(dict.items(overrides))
            expected = ((float, (float, int)), (complex, (complex, float, int)))
            if len(entries) == 2 and all(
                    key is source and type(target) is types.UnionType
                    and type(target.__args__) is tuple and len(target.__args__) == len(arguments)
                    and all(item is required for item, required in zip(target.__args__, arguments, strict=True))
                    for (key, target), (source, arguments) in zip(entries, expected, strict=True)):
                facts = [['builtins.float', ['builtins.float', 'builtins.int']],
                         ['builtins.complex', ['builtins.complex', 'builtins.float', 'builtins.int']]]
    return {'strategy': ('O1' if value.strategy is BeartypeStrategy.O1 else 'On' if value.strategy is BeartypeStrategy.On else 'unsupported'),
            'hint_overrides': facts, 'numeric_tower': tower, 'tower_source_sha256': tower_source,
            'violation_param_type': ('beartype.roar.BeartypeCallHintParamViolation'
                if value.violation_param_type is BeartypeCallHintParamViolation else None),
            'is_parameter_warning': value._is_violation_param_warn}


def _floatnd_signature_refs(
    target: object,  # noqa: PAN001 - authenticate an arbitrary original callable target
    signature: inspect.Signature | None,
    session: CaptureSession,
    normalize: _ValueNormalizer,
) -> tuple[Mapping[str, JSONValue], Mapping[str, JSONValue]]:
    """Observe effective annotations and their actual source bindings separately."""
    from .current_api_contract import inspect_native_floatnd_alias, inspect_source_bound_function
    if (type(target) is not types.FunctionType or signature is None
            or '__signature__' in vars(target) or '__wrapped__' in vars(target)):
        return {}, {}
    native = sys.modules.get("lcm.typing")
    alias = vars(native).get("FloatND") if native is not None else None
    if alias is None or not any(parameter.annotation is alias for parameter in signature.parameters.values()):
        return {}, {}
    filename = inspect.getsourcefile(target)
    source_hash = _file_digest(filename) if filename is not None else None
    if (source_hash is None or inspect_source_bound_function(target, module_name=target.__module__,
            qualname=target.__qualname__, file_sha256=source_hash) is not target):
        return {}, {}
    try:
        fn = next(node for node in ast.parse(textwrap.dedent(inspect.getsource(target))).body
                  if isinstance(node, ast.FunctionDef))
    except (OSError, TypeError, SyntaxError, StopIteration):
        return {}, {}
    parameters, bindings = {}, {}
    for argument in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
        parameter = signature.parameters.get(argument.arg)
        if parameter is None or not isinstance(argument.annotation, ast.Name):
            continue
        bound = target.__globals__.get(argument.annotation.id)
        authority = inspect_native_floatnd_alias(parameter.annotation)
        if authority is None or bound is not parameter.annotation:
            continue
        parameters[argument.arg] = session.binding_value(parameter.annotation, normalize)
        bindings[argument.annotation.id] = session.binding_value(bound, normalize)
    return parameters, bindings
def _beartype_forwarding(value: object) -> tuple[types.FunctionType, inspect.Signature] | None:
    """Authenticate generated forwarding without granting a finite input bound."""
    if inspect.ismethod(value):
        receiver, value = value.__self__, value.__func__
    elif not inspect.isfunction(value):
        receiver, value = value, inspect.getattr_static(type(value), '__call__', None)
    else:
        receiver = None
    if not inspect.isfunction(value) or not vars(value).get('__beartype_wrapper'):
        return None
    target = vars(value).get('__wrapped__')
    if (not inspect.isfunction(target) or '__wrapped__' in vars(target)
            or '__signature__' in vars(target) or '__signature__' in vars(value)):
        return None
    defaults = value.__kwdefaults__ or {}
    if defaults.get('__beartype_func') is not target:
        return None
    try:
        from beartype import BeartypeConf
        from beartype._decor.decorcore import beartype_object
        conf = defaults.get('__beartype_conf')
        if type(conf) is not BeartypeConf:
            return None
        # Regeneration may annotate its input; clone it to keep capture read-only.
        clone = types.FunctionType(target.__code__, target.__globals__, target.__name__,
                                   target.__defaults__, target.__closure__)
        clone.__qualname__, clone.__module__ = target.__qualname__, target.__module__
        clone.__kwdefaults__ = dict(target.__kwdefaults__ or {})
        clone.__annotations__ = dict(target.__annotations__)
        native_return = None
        if target.__module__ == '_lcm.dtypes' and target.__qualname__ == '_first_array':
            from .current_api_contract import inspect_native_first_array_return
            native_return = inspect_native_first_array_return(value, target)
            if native_return is None:
                return None
            clone.__annotations__['return'] = native_return[1]
        if any(isinstance(hint, str) for hint in clone.__annotations__.values()):
            return None
        current = defaults.get('__beartype_call_curr')
        reference = beartype_object(clone, conf=conf, cls_stack=getattr(current, 'cls_stack', None))
        if reference is clone or value.__closure__ or reference.__closure__:
            return None
        signature = inspect.signature(target, follow_wrapped=False)
        renames = _beartype_default_renames(value, reference, frozenset(signature.parameters))
        if renames is None:
            return None
        # Validated injected tuple slots differ only in their generated object IDs.
        code = value.__code__.replace(
            co_filename='<beartype>',
            co_varnames=tuple(renames.get(name, name) for name in value.__code__.co_varnames),
        )
        if code != reference.__code__.replace(co_filename='<beartype>'):
            return None
        expected = reference.__kwdefaults__ or {}
        defaults = {renames.get(name, name): actual for name, actual in defaults.items()}
        if defaults.keys() != expected.keys() or value.__defaults__ != reference.__defaults__:
            return None
        for name, actual in defaults.items():
            other = expected[name]
            if name == '__beartype_func':
                continue
            if name == '__beartype_call_curr':
                if type(actual) is not type(other):
                    return None
                for field in ('conf', 'cls_stack', 'decoratee', 'decoratee_annotations', 'decoratee_scope_forward'):
                    left, right = getattr(actual, field), getattr(other, field)
                    if field == 'decoratee':
                        if left is not target or right is not clone:
                            return None
                    elif field == 'decoratee_annotations':
                        if left.keys() != right.keys():
                            return None
                        for key in left:
                            if _same_annotation_identity(left[key], right[key]):
                                continue
                            if (native_return is None or key != 'return' or left[key] is not native_return[0]
                                    or type(right[key]) is not types.UnionType
                                    or type(native_return[1]) is not types.UnionType
                                    or len(right[key].__args__) != len(native_return[1].__args__)
                                    or any(a is not b for a, b in zip(right[key].__args__, native_return[1].__args__, strict=True))):
                                return None
                    elif left is not right:
                        return None
            elif name == '__beartype_args_name_keywordable':
                # Regeneration rebuilds this exact set of keywordable names.
                keywordable = {parameter.name for parameter in signature.parameters.values()
                               if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)}
                if (not any(parameter.kind is parameter.VAR_KEYWORD for parameter in signature.parameters.values())
                        or type(actual) is not set or type(other) is not set
                        or any(type(item) is not str for item in actual)
                        or any(type(item) is not str for item in other)
                        or actual != other or actual != keywordable):
                    return None
            elif name in renames.values():
                continue
            elif actual is not other:
                return None
        import builtins
        import dis
        for instruction in dis.get_instructions(value):
            if instruction.opname == 'LOAD_GLOBAL':
                name = instruction.argval
                if value.__globals__.get(name, getattr(builtins, name, None)) is not reference.__globals__.get(name, getattr(builtins, name, None)):
                    return None
        if receiver is not None:
            parameters = list(signature.parameters.values())
            if not parameters or parameters[0].kind not in (parameters[0].POSITIONAL_ONLY, parameters[0].POSITIONAL_OR_KEYWORD):
                return None
            signature = signature.replace(parameters=parameters[1:])
        return target, signature
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _beartype_interface(value: object) -> tuple[types.FunctionType, inspect.Signature] | None:
    """Grant only the existing finite interface after full forwarding checks."""
    interface = _beartype_forwarding(value)
    if interface is None:
        return None
    signature = interface[1]
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()):
        return None
    return interface


def _capture_record(value, normalize, session):
    result = {'kind': 'callable', 'module': getattr(value, '__module__', type(value).__module__),
              'qualname': getattr(value, '__qualname__', type(value).__qualname__),
              'capture_contract': 'inspection-only-frozen-bindings-2', 'obligations': []}
    primitive = _primitive(value)
    if primitive is not None:
        result.update(primitive)
        result.update(arguments=[], raw_arguments=[], variadic=True, defaults={}, bindings={},
                      referenced_functions={}, call_targets={}, source_unavailable=True,
                      primitive_capture_contract='canonical-object-callsite-1', binding_sha256=_digest({}))
        return result
    try:
        # __wrapped__ advertises another callable's interface; it does not
        # change the wrapper's actual defaults or call behavior.
        signature = inspect.signature(value, follow_wrapped=False)
        result['arguments'] = [name for name, parameter in signature.parameters.items()
                               if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)]
        result['parameter_kinds'] = {name: parameter.kind.name for name, parameter in signature.parameters.items()}
        result['variadic'] = any(parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
                                 for parameter in signature.parameters.values())
    except (TypeError, ValueError):
        signature = None
        result.update(arguments=[], variadic=True)
    norm = lambda item: session.binding_value(item, normalize)
    result['defaults'] = ({name: norm(parameter.default) for name, parameter in signature.parameters.items()
                           if parameter.default is not parameter.empty} if signature is not None else {})
    partial = value if isinstance(value, functools.partial) else None
    raw = partial.func if partial else value
    receiver = raw.__self__ if inspect.ismethod(raw) else None
    raw = raw.__func__ if inspect.ismethod(raw) else raw
    if not inspect.isfunction(raw) and not inspect.isbuiltin(raw) and not isinstance(raw, type):
        receiver = raw
        raw = inspect.getattr_static(type(raw), '__call__', raw)
    from .current_api_contract import inspect_native_input_authority, inspect_temporal_input_authority
    native_input_source = (inspect_native_input_authority(receiver, raw)
                           if receiver is not None else None)
    if native_input_source is not None:
        result['native_input_source'] = native_input_source
    temporal_interface = inspect_temporal_input_authority(value) if receiver is None else None
    if temporal_interface is not None:
        temporal_source, temporal_target, _ = temporal_interface
        result['native_temporal_source'] = dict(temporal_source)
        result['temporal_target'] = session.reference(temporal_target, normalize)
    result['code_sha256'] = _code_digest(raw)
    try:
        raw_signature = inspect.signature(raw, follow_wrapped=False)
        result['raw_arguments'] = list(raw_signature.parameters)
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in raw_signature.parameters.values()):
            result['variadic'] = True
    except (TypeError, ValueError):
        raw_signature = None
        result['raw_arguments'] = list(result['arguments'])
    if inspect.getattr_static(value, '__signature__', None) is not None:
        result['variadic'] = True
        result['obligations'].append('Explicit advertised signature requires reconciliation with the actual callable code')
    declared = (session.declaration_interfaces or {}).get(id(value))
    # The declaration owns the inputs; forwarding needs separate authentication.
    interface = _beartype_interface(value)
    if declared is not None and declared.function is value and declared.signature is not None:
        result['physical_parameter_kinds'] = dict(result.get('parameter_kinds', {}))
        result['physical_variadic'] = result['variadic']
        result['arguments'] = [name for name, p in declared.signature.parameters.items()
                               if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        result['parameter_kinds'] = {name: p.kind.name for name, p in declared.signature.parameters.items()}
        result['variadic'] = any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                                 for p in declared.signature.parameters.values())
        result['advertised_signature_origin'] = 'pylcm-model-declarations/v1'
    elif interface is not None:
        _, effective_signature = interface
        result['physical_parameter_kinds'] = dict(result.get('parameter_kinds', {}))
        result['arguments'] = list(effective_signature.parameters)
        result['parameter_kinds'] = {name: p.kind.name for name, p in effective_signature.parameters.items()}
        result['advertised_signature_origin'] = 'regenerated-beartype-wrapper/v1'
    elif temporal_interface is not None:
        effective_signature = temporal_interface[2]
        result['physical_parameter_kinds'] = dict(result.get('parameter_kinds', {}))
        result['physical_variadic'] = result['variadic']
        result['arguments'] = list(effective_signature.parameters)
        result['parameter_kinds'] = {name: p.kind.name for name, p in effective_signature.parameters.items()}
        result['variadic'] = False
        result['advertised_signature_origin'] = 'source-bound-native-temporal/v1'
    elif native_input_source is not None:
        try:
            effective_signature = inspect.signature(value)
        except (TypeError, ValueError):
            effective_signature = None
        if (effective_signature is not None and all(
                p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                for p in effective_signature.parameters.values())):
            result['physical_parameter_kinds'] = dict(result.get('parameter_kinds', {}))
            result['physical_variadic'] = result['variadic']
            result['arguments'] = list(effective_signature.parameters)
            result['parameter_kinds'] = {name: p.kind.name for name, p in effective_signature.parameters.items()}
            result['variadic'] = False
            result['advertised_signature_origin'] = 'source-bound-native-wrapper/v1'
    annotation_refs, annotation_bindings = _floatnd_signature_refs(
        raw, raw_signature if inspect.isfunction(raw) else None, session, normalize)
    if annotation_refs:
        result["parameter_annotation_refs"] = annotation_refs
        result["annotation_bindings"] = annotation_bindings
    if interface is not None:
        result['wrapped_target'] = session.reference(interface[0], normalize)
        result['wrapped_target_contract'] = 'regenerated-beartype-wrapper/v1'
        wrapped_annotations, _ = _floatnd_signature_refs(interface[0], interface[1], session, normalize)
        if wrapped_annotations:
            result["wrapped_parameter_annotation_refs"] = wrapped_annotations
            result["wrapped_input_config"] = result["defaults"].get("__beartype_conf")
    if partial is not None:
        result['partial_args'] = [norm(item) for item in partial.args]
        result['partial_keywords'] = {name: norm(item) for name, item in (partial.keywords or {}).items()}
        result['partial_target'] = session.reference(partial.func, normalize)
        try:
            result['partial_bindings'] = {name: norm(item) for name, item in inspect.signature(partial.func, follow_wrapped=False).bind_partial(
                *partial.args, **(partial.keywords or {})).arguments.items()}
        except (TypeError, ValueError):
            result['obligations'].append('Partial argument binding could not be reconstructed')
    result.update(bindings={}, referenced_functions={}, call_targets={})
    boundary = _library_boundary(raw)
    if isinstance(value, type):
        boundary = 'Class constructor target; numerical semantics not interpreted'
    if boundary:
        result.update(source_unavailable=True, opaque_boundary=boundary)
        try:
            path = inspect.getsourcefile(raw)
        except TypeError:
            path = None
        result.update(source_file=path, source_sha256=_file_digest(path))
        result['obligations'].append(boundary)
        if receiver is not None:
            result['bound_self'] = {'type': type(receiver).__module__ + '.' + type(receiver).__qualname__,
                                    'identity_status': 'UNRESOLVED_OPAQUE_RECEIVER',
                                    'obligation': 'Receiver storage requires a source-bound declaration adapter'}
        wrapped = inspect.getattr_static(value, '__wrapped__', None)
        if callable(wrapped) and wrapped is not value:
            result['wrapped_target'] = session.reference(wrapped, normalize)
        result['binding_sha256'] = _digest({key: result.get(key) for key in
                                          ('defaults', 'partial_bindings', 'bound_self')})
        return result
    if receiver is not None:
        if not hasattr(receiver, '__dict__'):
            result['obligations'].append('Bound object has no inspectable dictionary; slot/descriptor state unresolved')
        result['bound_self'] = {'type': type(receiver).__module__ + '.' + type(receiver).__qualname__,
                                'fields': {name: norm(item) for name, item in (vars(receiver) if hasattr(receiver, '__dict__') else {}).items()}}
    bindings = {}
    try:
        path = inspect.getsourcefile(raw)
        result['source_sha256'] = _file_digest(path)
        lines, first = inspect.findsource(raw)
        result.update(source=textwrap.dedent(''.join(inspect.getblock(lines[first:]))), source_file=path, source_line=first + 1)
        closure = inspect.getclosurevars(raw)
        bindings = {**closure.builtins, **closure.globals, **closure.nonlocals}
        if receiver is not None and result['raw_arguments']:
            bindings[result['raw_arguments'][0]] = receiver
    except (OSError, TypeError, ValueError):
        result['source_unavailable'] = True
        result['obligations'].append('Callable source unavailable; only captured signature is available')
    for name, item in sorted(bindings.items()):
        session.check()
        if isinstance(item, types.ModuleType):
            result['bindings'][name] = {'kind': 'module', 'name': item.__name__,
                                        'file_sha256': _file_digest(getattr(item, '__file__', None))}
        elif callable(item) and not isinstance(item, type):
            reference = session.reference(item, normalize)
            result['bindings'][name] = reference
            result['referenced_functions'][name] = reference
        elif receiver is not None and item is receiver:
            result['bindings'][name] = result['bound_self']
        else:
            result['bindings'][name] = norm(item)
    pending = list(result['bindings'].values())
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if str(item.get('identity_status', '')).startswith('UNRESOLVED'):
                result['obligations'].append('Captured binding requires immutable input identity or a source-bound adapter')
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    if result.get('source'):
        try:
            tree = ast.parse(result['source'])
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                spelling = ast.unparse(call.func)
                if spelling in result['call_targets']:
                    continue
                try:
                    target = _resolve(call.func, bindings)
                    if not callable(target):
                        raise LookupError('Resolved callee is not callable')
                    result['call_targets'][spelling] = session.reference(target, normalize)
                    def current_target(raw=raw, receiver=receiver, node=call.func):
                        closure = inspect.getclosurevars(raw)
                        current = {**closure.builtins, **closure.globals, **closure.nonlocals}
                        if receiver is not None and result['raw_arguments']:
                            current[result['raw_arguments'][0]] = receiver
                        return _live_token(_resolve(node, current))
                    session.watch(current_target, _live_token(target), result, 'Call-target rebinding: ' + spelling)
                except (LookupError, AttributeError, TypeError) as exc:
                    result['call_targets'][spelling] = {'kind': 'unresolved', 'reason': str(exc)}
                    result['obligations'].append('Unresolved call target: ' + spelling)
        except SyntaxError:
            result['obligations'].append('Captured source is outside the parser Python version')
        # A dtype is data, not a callee. Record the actual class only where a
        # cast names one, without resolving a dynamic receiver or executing it.
        try:
            dtype_nodes = []
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                if isinstance(call.func, ast.Attribute) and call.func.attr == 'astype' and call.args:
                    dtype_nodes.append(call.args[0])
                dtype_nodes.extend(kw.value for kw in call.keywords if kw.arg == 'dtype')
            for node in dtype_nodes:
                spelling = ast.unparse(node)
                try:
                    target = _resolve(node, bindings)
                    if _dtype_primitive(target) is None:
                        continue
                    result.setdefault('dtype_references', {})[spelling] = session.reference(target, normalize)
                    def current_dtype(
                        raw: types.FunctionType = raw,
                        receiver: object = receiver,  # noqa: PAN001 - callable receivers have unrelated native types
                        node: ast.expr = node,
                    ) -> Hashable:
                        closure = inspect.getclosurevars(raw)
                        current = {**closure.builtins, **closure.globals, **closure.nonlocals}
                        if receiver is not None and result['raw_arguments']:
                            current[result['raw_arguments'][0]] = receiver
                        return _live_token(_resolve(node, current))
                    session.watch(current_dtype, _live_token(target), result, 'Dtype-target rebinding: ' + spelling)
                except (LookupError, AttributeError, TypeError):
                    pass
        except (SyntaxError, UnboundLocalError):
            pass
    result['binding_sha256'] = _digest({key: callable_record_payload({'value': result.get(key)})['value']
                                       for key in ('bindings', 'defaults', 'bound_self', 'partial_bindings')})
    return result


def primitive_operation(metadata, callee, callable_records=None):
    """Use actual identity evidence; unsupported or mutated targets stay opaque."""
    target = metadata.get('call_targets', {}).get(ast.unparse(callee), {})
    target = resolve_callable(target, callable_records or metadata.get('callable_records'))
    if (target.get('kind') != 'primitive' or target.get('resolution') != 'actual_object_identity' or
            target.get('capture_status', 'COMPLETE') != 'COMPLETE'):
        return None
    expected = {'numpy.where': 'select', 'jax.numpy.where': 'select',
                'numpy.logical_and': 'logical_and', 'jax.numpy.logical_and': 'logical_and',
                'numpy.logical_or': 'logical_or', 'jax.numpy.logical_or': 'logical_or'}
    operation = expected.get(target.get('canonical'))
    return operation if operation == target.get('operation') and target.get('identity_sha256') else None
