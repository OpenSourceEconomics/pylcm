"""Read a constructed pylcm Model through graph/user-regime declarations.

The normal task supplies an existing Model or original declaration mapping.
No constructor, solver, simulation method, or numerical callback is executed or
replaced. Output is metadata with explicit capture and semantic obligations.
"""
from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from fractions import Fraction
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import types
from typing import Any
from .records import JSONValue


def digest_file(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def source_function(value: Any, *, depth: int = 0, seen: frozenset[int] = frozenset()) -> dict[str, Any]:
    """Capture real callable/library identities and frozen bindings; never call a law."""
    from .callable_identity import capture_callable
    return capture_callable(value, normalize, depth=depth, seen=seen)


def normalize(value: Any, *, depth: int = 0, seen: frozenset[int] = frozenset()) -> Any:
    """Capture a frozen value graph, sharing repeated callback records locally."""
    from .callable_identity import capture_session, current_capture_session
    own = current_capture_session() is None
    with capture_session() as session:
        result = _normalize(value, depth=depth, seen=seen)
    if own and session.records and isinstance(result, dict):
        # The table records never point back to this enclosing envelope.
        return {**result, "callable_records": session.records,
                "capture_report": session.report()}
    return result


def _normalize(value: Any, *, depth: int = 0, seen: frozenset[int] = frozenset()) -> Any:
    from .callable_identity import current_capture_session
    session = current_capture_session()
    if session is not None:
        session.metadata_items += 1
        session.metadata_bytes += len(value) if isinstance(value, (str, bytes)) else 16
        session.check()
        store = session.structural_store
        pd = sys.modules.get('pandas')
        if pd is not None and type(value) is pd.Series:
            def plain_label(item):
                return (item is None or type(item) in (str, bool, int, float)
                        or (type(item) is tuple and all(plain_label(part) for part in item)))
            labels = value.index.tolist()
            if not all(plain_label(item) for item in [value.name, *value.index.names, *labels]):
                from .callable_identity import CaptureIncomplete
                raise CaptureIncomplete('Unsupported labelled Series index or name')
            before = value.copy(deep=True)
            result = {'kind': 'object', 'type': 'pandas.Series',
                      'capture_contract': 'labelled-series-logical-values/v1',
                      'fields': {'name': normalize(value.name),
                                 'dtype': str(value.dtype),
                                 'index_type': type(value.index).__module__ + '.' + type(value.index).__qualname__,
                                 'index_names': normalize(list(value.index.names)),
                                 'index': normalize(labels),
                                 'values': normalize(value.to_numpy(copy=True))}}
            session.watch(lambda: (value.equals(before)
                                   and type(value.index) is type(before.index)
                                   and value.index.names == before.index.names
                                   and value.name == before.name),
                          True, result, 'Labelled Series changed during capture')
            return result
        if store is not None:
            try:
                if store.writer.flavor(value):
                    shape = [int(n) for n in value.shape]
                    result = {'kind': 'array', 'shape': shape, 'dtype': str(value.dtype),
                              'structural_ref': store.reference(value)}
                    if value.size <= 8192:
                        import numpy as np
                        data = np.asarray(value)
                        result['sha256'] = hashlib.sha256(data.tobytes(order='C')).hexdigest()
                        result['values'] = normalize(data.tolist(), depth=depth + 1, seen=seen)
                    return result
                if isinstance(value, (Mapping, list, tuple, set, frozenset)) and len(value) > 8192:
                    return store.reference(value)
                np = sys.modules.get('numpy')
                scalar_array_value = np is not None and isinstance(value, np.generic)
                if not isinstance(value, type) and not scalar_array_value and hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    raise ValueError('Unsupported native array class or storage contract: '
                                     + type(value).__module__ + '.' + type(value).__qualname__)
            except ValueError as error:
                from .callable_identity import CaptureIncomplete
                raise CaptureIncomplete('Unsupported structural source: ' + str(error)) from error
        if session.binding_expansion and depth > 0 and session.structured_binding(value):
            return session.binding_value(value, normalize)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else {"nonfinite": str(value)}
    if isinstance(value, Fraction):
        return {"fraction": str(value)}
    if isinstance(value, range):
        return {"kind": "object", "type": "builtins.range",
                "fields": {"start": value.start, "stop": value.stop, "step": value.step}}
    if id(value) in seen:
        if callable(value):
            return source_function(value, depth=depth, seen=seen)
        return {"kind": "recursive-reference", "type": type(value).__qualname__,
                "identity_status": "UNRESOLVED_RECURSIVE_BINDING",
                "obligation": "Cyclic data requires an explicit immutable binding contract"}
    visited = seen | {id(value)}
    if isinstance(value, Mapping):
        if all(isinstance(k, str) for k in value):
            return {k: normalize(v, depth=depth + 1, seen=visited) for k, v in value.items()}
        return {"kind": "mapping", "items": [[normalize(k, depth=depth + 1, seen=visited),
                                                 normalize(v, depth=depth + 1, seen=visited)] for k, v in value.items()]}
    if isinstance(value, (tuple, list, set, frozenset)):
        result = [normalize(v, depth=depth + 1, seen=visited) for v in value]
        if isinstance(value, (set, frozenset)):
            # repr may contain process addresses or user-defined formatting.
            # Canonical captured values preserve unordered semantics without
            # making identities depend on those incidental spellings.
            result.sort(key=lambda item: json.dumps(item, sort_keys=True,
                                                    separators=(",", ":"), allow_nan=False))
        return result
    # Array classes expose shape/dtype descriptors, not array metadata. Retain
    # their identity without accessing descriptors or inventing an instance.
    if isinstance(value, type) and not dataclasses.is_dataclass(value):
        return {"kind": "type", "type": value.__module__ + "." + value.__qualname__}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = [int(x) for x in value.shape]
        size = 1
        for dimension in shape:
            size *= dimension
        result = {"kind": "array", "shape": shape, "dtype": str(value.dtype)}
        if size <= 8192:
            import numpy as np
            array = np.asarray(value)
            result["sha256"] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
            result["values"] = normalize(array.tolist(), depth=depth + 1, seen=visited)
        else:
            result["identity_status"] = "UNRESOLVED_LARGE_BINDING"
            result["obligation"] = "Supply an immutable input-data digest; dense parameter contents were not materialized or hashed"
        return result
    if dataclasses.is_dataclass(value) and isinstance(value, type):
        return {"kind": "categorical-class", "type": value.__module__ + "." + value.__qualname__,
                "members": {f.name: normalize(getattr(value, f.name, None), depth=depth + 1, seen=visited)
                            for f in dataclasses.fields(value)}}
    if dataclasses.is_dataclass(value):
        result = {"kind": "dataclass", "type": type(value).__module__ + "." + type(value).__qualname__,
                "fields": {f.name: normalize(getattr(value, f.name), depth=depth + 1, seen=visited)
                           for f in dataclasses.fields(value) if not f.name.startswith("_")}}
        native_reachability = sys.modules.get('_lcm.regime_building.reachable_states')
        if session is not None and session.structural_store is not None and native_reachability is not None and type(value) is getattr(native_reachability, 'ReachableStates', None):
            result['structural_backing'] = session.structural_store.reference(value)
        from .current_api_contract import inspect_authority
        authority = inspect_authority(type(value))
        if authority is not None:
            result['canonical_type_source'] = authority
        if callable(value):
            result['callable'] = source_function(value,depth=depth,seen=seen)
        return result
    if callable(value):
        return source_function(value, depth=depth, seen=seen)
    if type(value).__module__+'.'+type(value).__qualname__=='lcm.decoupled.Decoupled':
        return {'kind':'object','type':'lcm.decoupled.Decoupled',
                'fields':{name:normalize(getattr(value,name),depth=depth+1,seen=visited)
                          for name in ('objective','value')}}
    if type(value).__name__ in {"ByAge", "ByPeriod"}:
        payload = vars(value)
        return {"kind": "object", "type": type(value).__module__ + "." + type(value).__qualname__,
                "fields": {"cases": normalize(payload.get("_cases", ()), depth=depth + 1, seen=visited),
                           "default": normalize(payload.get("_default"), depth=depth + 1, seen=visited),
                           "until": normalize(payload.get("_until"), depth=depth + 1, seen=visited)},
                "capture_contract": "pinned-date-schedule-storage-1"}
    fields = {}
    for name in ("n_points", "points", "start", "stop", "inclusive_stop", "step", "exact_values", "values", "codes", "ordered", "tolerance", "value_to_code", "categories", "solve", "simulate"):
        try:
            item = getattr(value, name)
            if not callable(item) or name in {"solve", "simulate"}:
                fields[name] = normalize(item, depth=depth + 1, seen=visited)
        except (AttributeError, ValueError, TypeError):
            pass
    return {"kind": "object", "type": type(value).__module__ + "." + type(value).__qualname__, "fields": fields}


def _capture_cached_specializations(model, captured, *, declarations=None):
    """Read the actual dated construction products, never execute a factory."""
    if declarations is not None:
        periods = declarations.regimes_by_period
    else:
        preparation = getattr(model, '_declaration_preparation', None)
        structure = getattr(preparation, 'prepared_structure', None)
        schedule = getattr(structure, 'period_regimes', None)
        periods = getattr(schedule, 'by_period', {})
    if not periods:
        return

    def has_marker(value):
        return (type(value).__name__ in {'AgeSpecializedFunction', 'PeriodSpecializedFunction'}
                or (type(value).__name__ == 'Phased'
                    and any(has_marker(getattr(value, phase)) for phase in ('solve', 'simulate'))))

    def attach(original, record, values):
        if not has_marker(original):
            return
        tag = type(original).__name__
        if tag == 'Phased':
            for phase in ('solve', 'simulate'):
                attach(getattr(original, phase), record['fields'][phase],
                       [(period, getattr(value, phase) if type(value).__name__ == 'Phased' else value)
                        for period, value in values])
        elif tag in {'AgeSpecializedFunction', 'PeriodSpecializedFunction'}:
            record['resolved_by_period'] = [
                {'period': period, 'value': normalize(value)}
                for period, value in values
                if callable(value) and type(value).__name__ not in {'AgeSpecializedFunction', 'PeriodSpecializedFunction'}]
            record['resolution_capture_contract'] = 'constructed-period-regimes/v1'

    for name, regime in model.user_regimes.items():
        record = captured[name]['fields']
        for slot in ('functions', 'constraints'):
            for key, original in getattr(regime, slot, {}).items():
                if not has_marker(original):
                    continue
                values = [(period, getattr(regimes[name], slot)[key])
                          for period, regimes in sorted(periods.items())
                          if name in regimes and key in getattr(regimes[name], slot, {})]
                attach(original, record[slot][key], values)


def read_model(model: Any, *, checkpoint=None, capture_limits=None, structural_store=None) -> dict[str, Any]:
    """Inspect an already constructed Model, emitting topology before callbacks.

    ``checkpoint`` receives JSON-safe original-graph/progress records. The
    capture limit includes every normalization field and transaction validation.
    A stopped capture preserves its partial metadata and cannot claim an optimum.
    Native task-owned construction is outside this timer.
    """
    import importlib.metadata
    from .callable_identity import capture_session, CaptureIncomplete

    graph = getattr(model, "graph", None)
    reachability = getattr(model, "reachability", None)
    ages = getattr(model, "ages", None)
    try:
        version = importlib.metadata.version("pylcm")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable; existing object inspection only"
    result: dict[str, JSONValue] = {"schema": "constructed-pylcm-model-graph-1", "capture_format_revision": 4,
              "pylcm_version": version, "description": model.description,
              "n_periods": model.n_periods, "regime_names": sorted(model.user_regimes),
              "scope": "Actual constructed graph and declarations; no solve or simulation executed."}
    with capture_session(limits=capture_limits) as session:
        if structural_store is not None:
            session.structural_store = structural_store
            structural_store.check = session.check
            structural_store.writer.check = session.check
        try:
            declarations = None
            native = sys.modules.get('lcm.model')
            getter = getattr(getattr(native, 'Model', None), 'get_declarations', None)
            if native is not None and getter is not None and isinstance(model, native.Model):
                declarations = getter(model)
                if declarations.schema_version != 1:
                    raise CaptureIncomplete('Unsupported pylcm declaration API version')
                session.declaration_interfaces = {id(row.function): row for row in declarations.functions}
                graph = declarations.graph
                import inspect
                filename = inspect.getsourcefile(type(declarations))
                api_declaration: dict[str, JSONValue] = {'schema_version': 1, 'source_file': filename,
                    'source_sha256': session.file_digest(filename),
                    'scope': 'Original Model declared DAG inputs; callback implementation semantics remain separate'}
                result['declaration_api'] = api_declaration
                names = ('original_periods_by_regime', 'value_periods_by_regime',
                         'query_periods_by_phase', 'exit_function_demand')
                result['construction_authorities'] = {
                    'values': {name: normalize(getattr(declarations, name)) for name in names},
                    'source': 'pylcm.Model.get_declarations/v1'}
            session.suspend_callbacks = True
            result.update(ages=normalize(ages), graph=normalize(graph),
                          reachability=normalize(reachability), initial_nodes=normalize(model.initial_nodes),
                          regime_names_to_ids=normalize(model.regime_names_to_ids))
            if checkpoint is not None:
                checkpoint({"stage": "original_graph", "topology_complete": True,
                            "snapshot": dict(result), "capture_report": session.report()})
            session.suspend_callbacks = False
            for name, value in (("graph", graph), ("reachability", reachability),
                                ("user_regimes", model.user_regimes),
                                ("fixed_params", model.fixed_params),
                                ("shocks", getattr(model, "shocks", {})),
                                ("composite_states", getattr(model, "composite_states", {})),
                                ("composite_metadata", getattr(model, "_composite_states", {})),
                                ("pruned_variables", getattr(model, "pruned_variables", {})),
                                ("allocation_contract", getattr(model, "allocation_contract", {"status": "unknown"})),
                                ("semantic_contract", getattr(model, "semantic_contract", {}))):
                result[name] = normalize(value)
                session.check()
            _capture_cached_specializations(model, result['user_regimes'], declarations=declarations)
            if declarations is not None:
                api_declaration['function_interfaces'] = [
                    {'function': normalize(row.function), 'arg_names': row.arg_names}
                    for row in declarations.functions]
            discrete = sys.modules.get('_lcm.grids.discrete')
            if result.get('composite_metadata') and discrete is not None:
                result['composite_transport_contract'] = {
                    'missing_code': int(discrete.MISSING_CAT_CODE),
                    'module': discrete.__name__,
                    'source_sha256': session.file_digest(discrete.__file__)}
                session.watch(lambda: int(discrete.MISSING_CAT_CODE),
                              int(discrete.MISSING_CAT_CODE), result,
                              'Composite missing-code constant changed')
        except (CaptureIncomplete, RecursionError) as exc:
            session.exhaustion_reason = session.exhaustion_reason or str(exc)
            session.obligations.append(str(exc))
            if "graph" not in result and checkpoint is not None:
                checkpoint({"stage": "original_graph", "topology_complete": False,
                            "snapshot": dict(result), "capture_report": session.report()})
        finally:
            session.suspend_callbacks = False
    result["callable_records"] = session.records
    capture_report = session.report()
    result["capture_report"] = capture_report
    if structural_store is not None and structural_store.header is not None:
        from .structural_store import LoadedCapture
        result['structural_store'] = structural_store.header
        result['capture_format_revision'] = 5
        result = LoadedCapture(result)
        result.structure = structural_store.reader
    elif structural_store is not None and structural_store.roots:
        result['structural_store'] = {'schema': 'regime-structural-capture/v1', 'status': 'INCOMPLETE',
                                      'path': str(structural_store.stage.relative_to(structural_store.destination.parent)),
                                      'completed_chunks': list(structural_store.writer.payloads),
                                      'scope': 'Partial chunks only; no complete structural or source authority'}
    if graph is None:
        result["graph_api"] = "reachability-only"
    result["specialization_capture_revision"] = 3
    result["specialization_execution"] = "NOT_RUN; cached native period functions inspected; absent construction products remain unresolved"
    result["test_defined_callbacks"] = [
        {"file": record["source_file"], "qualname": record.get("qualname"), "sha256": record.get("source_sha256")}
        for record in session.records.values() if "/tests/" in (record.get("source_file") or "")]
    if checkpoint is not None:
        checkpoint({"stage": "capture_complete" if capture_report["status"] == "COMPLETE" else "capture_incomplete",
                    "capture_report": result["capture_report"],
                    "original_graph_preserved": "graph" in result})
    return result


def read_declarations(spec: Any, *, production_metadata: dict | None = None,
                      semantic_contract: dict | None = None,
                      specialization_metadata: dict | None = None) -> dict[str, Any]:
    """Public pre-program metadata contract, independent of builder syntax.

    Pass actual declaration objects/mappings returned by any factory. This
    function never constructs Model, calls a specialization, solves, simulates,
    or enumerates state/action products. Construction-only metadata must be
    supplied by the source provider; absence remains explicit.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("A pre-program declaration mapping is required")
    from .callable_identity import capture_session
    with capture_session() as session:
        result = normalize(spec)
        result["production_metadata"] = normalize(production_metadata or {"status": "unverified"})
        result["semantic_contract"] = normalize(semantic_contract or {})
        result["specialization_metadata"] = normalize(specialization_metadata or {})
    result["callable_records"] = session.records
    capture_report = session.report()
    result["capture_report"] = capture_report
    result["capture_format_revision"] = 4
    result["construction_capture_phase"] = "original declaration mapping before Model/program construction"
    result["execution_contract"] = {"numerical_model_rows": 0, "solver_compilation": False,
                                     "specialization_factories_executed": 0}
    return result


def pytest_configure(config: Any) -> None:
    raise RuntimeError(
        'Constructor interception is disabled: Model declaration preparation can '
        'lower composite states or evaluate construction callbacks. Export the '
        'original pre-program specification with read_declarations before Model '
        'construction. Historical captured JSON remains supported as input.'
    )
