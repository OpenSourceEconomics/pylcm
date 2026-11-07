"""Metadata-only entry points for task-owned production setup and analysis.

The task constructs its actual production Model with its ordinary inputs. This
module inspects that existing object; it never selects a recipe or calls solve,
simulate or a grid factory. Separate analysis consumes the resulting immutable
JSON and runs the public CLI inside an outer process-tree budget.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time


def _write_json(path, value):
    """Publish complete records atomically without a second serialized copy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from .structural_store import relocate_captures
    value = relocate_captures(value, path.parent)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _capture_receipt_path(snapshot):
    snapshot = Path(snapshot)
    return snapshot.with_name(snapshot.stem + '.metadata-capture.json')


def export_model(model, destination, *, production_metadata, model_id=None,
                 seconds=900, host_gib=8, source_transaction=None):
    """Capture an existing Model, charging capture/serialization to metadata time.

    The normal task has already constructed the Model. No constructor or
    numerical method is replaced or executed here. A topology checkpoint is
    published before callable expansion; the adjacent receipt is bound to the
    exact snapshot bytes and is charged by ``analyse_snapshot``.
    """
    from .model_snapshot import read_model, normalize
    if (seconds is not None and seconds < 0) or (host_gib is not None and host_gib <= 0):
        raise ValueError('Resource bounds require seconds>=0 and host_gib>0')
    started = time.perf_counter()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    topology_path = destination.with_name(destination.stem + '.original-graph.json')
    progress_path = destination.with_name(destination.stem + '.capture-progress.json')

    def checkpoint(record):
        if record.get('stage') == 'original_graph':
            _write_json(topology_path, record)
        _write_json(progress_path, record)

    from .structural_store import CaptureStore
    structure = CaptureStore(destination, check=lambda: None,
                             source_context={'production_metadata': production_metadata,
                                             'capture_route': 'original constructed Model'},
                             transaction=source_transaction)
    from .callable_identity import CaptureLimits
    capture_limits = ({name: None for name in CaptureLimits.__dataclass_fields__}
                      if seconds is None and host_gib is None else {'seconds': seconds, 'host_gib': host_gib})
    snapshot = read_model(model, checkpoint=checkpoint,
                          capture_limits=capture_limits,
                          structural_store=structure)
    if model_id is not None:
        snapshot['model_id'] = str(model_id)
    if snapshot.get('capture_report', {}).get('status') == 'COMPLETE':
        _enrich_native_grid_metadata(model, snapshot)
    if 'construction_authorities' not in snapshot:
        prepared = (getattr(getattr(model, '_declaration_preparation', None), 'prepared_structure', None)
                    if snapshot.get('capture_report', {}).get('status') == 'COMPLETE' else None)
        names = ('original_periods_by_regime', 'value_periods_by_regime',
                 'query_periods_by_phase', 'exit_function_demand')
        snapshot['construction_authorities'] = {
            'values': {name: normalize(getattr(prepared, name)) for name in names if hasattr(prepared, name)},
            'source': 'existing Model._declaration_preparation.prepared_structure',
            'scope': 'captured construction metadata; no query evaluator or allocation semantics asserted',
        }
    snapshot['production_metadata'] = {
        **production_metadata,
        'export_route': 'existing production Model supplied by its normal setup task',
        'constructor_replaced': False, 'model_methods_replaced': False,
        'numerical_solve': 'NOT_RUN_BY_EXPORTER', 'simulation': 'NOT_RUN_BY_EXPORTER',
        'specialization_factories_executed_by_exporter': False,
    }
    _write_json(destination, snapshot)
    identity = _sha256(destination)
    elapsed = time.perf_counter() - started
    capture = snapshot.get('capture_report', {})
    complete = capture.get('status') == 'COMPLETE' and (seconds is None or elapsed <= seconds)
    receipt = {
        'schema': 'regime-metadata-capture-receipt/v1',
        'snapshot_sha256': identity,
        'status': 'COMPLETE' if complete else 'INCOMPLETE',
        'elapsed_seconds': elapsed,
        'peak_process_tree_gib': capture.get('peak_process_tree_gib'),
        'peak_observed_host_gib': capture.get('peak_observed_host_gib'),
        'process_tree_rss_available': capture.get('process_tree_rss_available'),
        'resource_status': capture.get('resource_status', 'UNVERIFIED_PROCESS_TREE'),
        'budget': {'seconds': seconds, 'host_gib': host_gib},
        'original_graph_checkpoint': topology_path.name,
        'progress': progress_path.name,
        'capture_report': capture,
        'scope': 'Capture, native grid metadata reads and serialization after the ordinary Model constructor returned; native construction/input preparation excluded.',
        'measurement': 'perf_counter through snapshot publication and hash; cooperative process-tree RSS sampling in capture. An outer task supervisor is still required for a hard bound on an individual allocation.',
    }
    _write_json(_capture_receipt_path(destination), receipt)
    return receipt


def analyse_snapshot(snapshot, output, *, seconds=900, host_gib=8, command_prefix=None,
                     allow_final_only_split=False, search_mode='heuristic',
                     heuristic_rounds=2, heuristic_neighbors=2):
    """One failure boundary for receipt, worker and final delivery validation."""
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if snapshot.is_relative_to(output):
        raise ValueError('Analysis output must not contain the original input; use a separate analysis subdirectory')
    try:
        return _analyse_snapshot(snapshot, output, seconds=seconds, host_gib=host_gib,
                                 command_prefix=command_prefix,
                                 allow_final_only_split=allow_final_only_split,
                                 search_mode=search_mode, heuristic_rounds=heuristic_rounds,
                                 heuristic_neighbors=heuristic_neighbors)
    except Exception as error:
        from .delivery import invalidate_delivery
        try:
            previous = json.loads((output/'report.json').read_text())
            if not isinstance(previous, dict): previous = {}
        except (OSError, ValueError):
            previous = {}
        status = previous.get('status')
        if status not in {'INVALID', 'INCOMPLETE', 'FAILED'}: status = 'INVALID'
        invalidate_delivery(output, status=status,
                            reason=type(error).__name__ + ': ' + str(error), details=previous)
        raise


def _analyse_snapshot(snapshot, output, *, seconds, host_gib, command_prefix,
                      allow_final_only_split, search_mode, heuristic_rounds, heuristic_neighbors):
    from .resources import run_bounded
    if search_mode not in {'heuristic', 'exact'}:
        raise ValueError('search_mode must be heuristic or exact')
    if type(allow_final_only_split) is not bool:
        raise ValueError('allow_final_only_split must be a boolean')
    if seconds < 0 or host_gib <= 0:
        raise ValueError('Resource bounds require seconds>=0 and host_gib>0')
    started = time.perf_counter()
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if snapshot.is_relative_to(output):
        raise ValueError('Analysis output must not contain the original input; use a separate analysis subdirectory')
    # A task launched by Pixi already runs in its pinned interpreter. The
    # standalone dependency-free API must also work in a supported interpreter
    # without requiring a second environment manager or installing packages.
    prefix = list(command_prefix or [sys.executable])
    output.mkdir(parents=True, exist_ok=True)
    for name in ('summary.json', 'report.json', 'next-actions.md', 'next-actions.html',
                 'report.md', 'report.html', 'partial-frontier.json',
                 'original.json', 'normalized.json', 'analysis.json',
                 'metadata-progress.json', 'semantic-check.json',
                 'typed-graph.json', 'candidate-boundaries.json',
                 'recommended-boundaries.json', 'proposal-decisions.json',
                 'report-data.json', 'report-files.json', 'report-manifest.json', 'family-boundary.json',
                 'delivery-timings.json',
                 'completed-before-budget-check.json'):
        path = output / name
        if path.is_file():
            path.unlink()
        incomplete_path = path.with_name(path.name + '.writing')
        if incomplete_path.is_file():
            incomplete_path.unlink()
    capture = None
    capture_path = _capture_receipt_path(snapshot)
    if capture_path.is_file():
        capture = json.loads(capture_path.read_text())
        if capture.get('schema') != 'regime-metadata-capture-receipt/v1' or capture.get('snapshot_sha256') != _sha256(snapshot):
            raise ValueError('Capture receipt does not identify the original snapshot bytes')
    capture_seconds = (capture or {}).get('elapsed_seconds', 0)
    if isinstance(capture_seconds, bool) or not isinstance(capture_seconds, (float, int)) or not 0 <= capture_seconds < float('inf'):
        raise ValueError('Capture elapsed time must be finite and nonnegative')
    remaining = max(0, seconds - capture_seconds - (time.perf_counter() - started))
    if capture is not None and capture.get('status') != 'COMPLETE':
        # An incomplete identity capture cannot be analysed. It is not evidence
        # that the available metadata time or memory budget was exhausted.
        report = capture.get('capture_report', {})
        obligations = report.get('obligations', [])
        exhaustion = report.get('exhaustion_reason')
        detail = '; '.join(str(item) for item in obligations)
        reason = 'Source capture is incomplete' + (': ' + detail if detail else '.')
        elapsed = capture_seconds + time.perf_counter() - started
        record = {'schema': 'regime-production-result/v1', 'status': 'INCOMPLETE',
                  'phase': 'capture', 'reason': reason, 'analysis_started': False,
                  'capture_receipt': str(capture_path),
                  'capture_obligations': obligations,
                  'capture_exhaustion_reason': exhaustion,
                  'recommendation': None, 'scenarios': [],
                  'transformation_authorized': False,
                  'metadata_budget': {
                      'total_seconds': seconds, 'capture_seconds_charged': capture_seconds,
                      'remaining_worker_seconds': remaining, 'total_elapsed_seconds': elapsed,
                      'within_time_budget': elapsed <= seconds}}
        _write_json(output/'summary.json', record)
        _write_json(output/'report.json', record)
        message = ('# Source capture incomplete\n\n' + reason +
                   '\n\nAnalysis was not started. No optimum or transformation is authorized. '
                   'The original snapshot and its capture receipt are unchanged.\n')
        import html
        for name in ('next-actions', 'report'):
            (output/(name+'.md')).write_text(message)
            (output/(name+'.html')).write_text('<!doctype html><meta charset="utf-8"><title>Source capture incomplete</title><pre>' + html.escape(message) + '</pre>')
        from .delivery import invalidate_delivery
        invalidate_delivery(output, status='INCOMPLETE', reason=reason, details=record)
        raise RuntimeError(f'Metadata capture INCOMPLETE; inspect {capture_path}')
    command = [*prefix, '-m', f'{__package__}.pipeline', str(snapshot), '--output', str(output),
               '--seconds', str(remaining), '--host-gib', str(host_gib), '--worker',
               '--search-mode', search_mode, '--heuristic-rounds', str(heuristic_rounds),
               '--heuristic-neighbors', str(heuristic_neighbors)]
    if allow_final_only_split:
        command.append('--allow-final-only-split')
    from .owned_mutations import resource_plan
    worker_plan = resource_plan(output/'outer-resource', inputs=[snapshot])
    try:
        code = run_bounded(command, output=output/'outer-resource',
                           seconds=remaining, host_gib=host_gib,
                           checkpoint_path=output/'partial-frontier.json', mutation_plan=worker_plan)
    except Exception as error:
        from .delivery import invalidate_delivery
        invalidate_delivery(output, status='FAILED',
                            reason=type(error).__name__ + ': ' + str(error))
        raise RuntimeError('Metadata worker delivery failed') from error
    from .structural_store import check_capture_chunks
    input_error = None
    try: check_capture_chunks(snapshot, reads=worker_plan.reads)
    except (ValueError, OSError, KeyError, TypeError) as error:
        input_error = str(error)
        code = 1
    summary = output/'summary.json'
    finished = output/'completed-before-budget-check.json'
    if code == 2 and (output/'report.json').is_file():
        (output/'report.json').replace(finished)
    if input_error is not None:
        if (output/'report.json').is_file():
            _write_json(output/'completed-before-source-check.json', json.loads((output/'report.json').read_text()))
        record = {'schema': 'regime-production-result/v1', 'status': 'INVALID',
                  'phase': 'structural_input_integrity', 'reason': input_error,
                  'recommendation': None, 'scenarios': [], 'transformation_authorized': False}
        _write_json(output/'report.json', record)
        import html
        message = 'Structural input changed or is incomplete. No transformation is authorized. ' + input_error
        for name in ('report', 'next-actions'):
            (output/(name+'.html')).write_text('<!doctype html><meta charset="utf-8"><title>Structural input invalid</title><pre>' + html.escape(message) + '</pre>')
            (output/(name+'.md')).write_text(message + '\n')
    elif code == 2 or not summary.exists():
        record = {'status':'INCOMPLETE' if code==2 else 'FAILED',
                  'operation':'metadata analysis only',
                  'reason':'Outer process did not complete its bounded readable summary',
                  'output':str(output), 'exit_code':code}
        outer_report = output/'outer-resource/report.json'
        incomplete = json.loads(outer_report.read_text()) if outer_report.is_file() else {
            'schema': 'regime-production-result/v1', 'status': record['status'],
            'recommendation': None, 'scenarios': [], 'error': record['reason']}
        _write_json(output/'report.json', incomplete)
        # A killed worker cannot finish the rich report. Publish a small readable
        # result instead of leaving a stale successful report from an older run.
        message = ('# Metadata analysis incomplete\n\n' + record['reason'] +
                   '. No optimum or transformation is authorized. The original snapshot is unchanged.\n\n'
                   'See outer-resource/resource.json and report.json for retained checkpoint references.\n')
        (output/'next-actions.md').write_text(message)
        import html
        (output/'next-actions.html').write_text('<!doctype html><meta charset="utf-8"><title>Metadata analysis incomplete</title><pre>' + html.escape(message) + '</pre>')
    else:
        record = json.loads(summary.read_text())
    elapsed = capture_seconds + time.perf_counter() - started
    record['metadata_budget'] = {
        'total_seconds': seconds, 'capture_seconds_charged': capture_seconds,
        'capture_receipt': str(capture_path) if capture is not None else None,
        'capture_accounting': 'source-bound receipt (may originate in an earlier task)' if capture is not None else 'pre-existing snapshot replay; original capture not measured in this run',
        'remaining_worker_seconds': remaining, 'total_elapsed_seconds': elapsed,
        'within_time_budget': code != 2 and elapsed <= seconds,
    }
    if code == 2 or (elapsed > seconds and code in (0, 3)):
        record['status'] = 'INCOMPLETE'
        record['reason'] = ('The outer metadata resource boundary was reached; completed worker candidates remain unverified for delivery'
                            if code == 2 else
                            'Capture, checking and reporting together exceeded the metadata time budget')
        # Preserve the completed worker evidence by rename, without decoding a
        # potentially large report outside its budget. The public entry report
        # must not keep claiming complete delivery after this final check.
        if code != 2 and (output/'report.json').exists():
            (output/'report.json').replace(finished)
        _write_json(output/'report.json', {
            'schema': 'regime-production-result/v1', 'status': 'INCOMPLETE',
            'reason': record['reason'], 'recommendation': None,
            'completed_worker_evidence': finished.name if finished.exists() else None,
            'resource_report': 'outer-resource/report.json' if (output/'outer-resource/report.json').is_file() else None,
            'scope': 'Worker evidence is retained; complete metadata delivery exceeded its budget. No automatic graph change is authorized.'})
        record.pop('decisions', None)
        message = ('# Metadata analysis incomplete\n\n' + record['reason'] +
                   '. No optimum or transformation is authorized. The original snapshot is unchanged.\n')
        for name in ('next-actions.md', 'report.md'):
            (output/name).write_text(message)
        import html
        for name in ('next-actions.html', 'report.html'):
            (output/name).write_text('<!doctype html><meta charset="utf-8"><title>Metadata analysis incomplete</title><pre>' + html.escape(message) + '</pre>')
        code = 2
    if code not in (0,3):
        from .delivery import invalidate_delivery
        status = record.get('status')
        if status not in {'INVALID', 'INCOMPLETE', 'FAILED'}:
            status = 'INCOMPLETE' if code == 2 else 'FAILED'
        if finished.exists():
            record['completed_worker_evidence'] = finished.name
        if (output/'completed-before-source-check.json').exists():
            record['unverified_source_check_evidence'] = 'completed-before-source-check.json'
        if (output/'outer-resource/report.json').exists():
            record['resource_report'] = 'outer-resource/report.json'
        record = invalidate_delivery(output, status=status,
                                     reason=record.get('reason', 'Worker delivery failed'),
                                     details=record)
    else:
        _write_json(summary, record)
    if code not in (0,3):
        raise RuntimeError(f'Metadata analysis {record["status"]}; inspect {output / "outer-resource"}')
    return record


def _enrich_native_grid_metadata(model, snapshot):
    """Read grids already resolved by native construction; execute no factories."""
    if snapshot.get('declaration_api', {}).get('schema_version') == 1:
        return _enrich_public_grid_metadata(model.get_declarations(), snapshot)
    prepared = getattr(model, "_declaration_preparation", None)
    structure = getattr(prepared, "prepared_structure", None)
    schedule = getattr(structure, "grid_schedule", None)
    groups = {}
    for period, regimes in getattr(schedule, "by_period", {}).items():
        for name, states in regimes.items():
            for state, resolved in states.items():
                shape = tuple(resolved.nodes.shape)
                if len(shape) != 1:
                    raise ValueError(f"Native specialized grid is not one-dimensional: {name}/{state}")
                groups.setdefault((str(name), str(state)), []).append({
                    "period": int(period), "size": int(shape[0]), "dtype": str(resolved.nodes.dtype),
                    "grid_type": type(resolved.grid).__module__ + "." + type(resolved.grid).__qualname__,
                })
    observed = []
    for (name, state), periods in sorted(groups.items()):
        regime = snapshot.get("user_regimes", {}).get(name, {})
        fields = regime.get("fields", regime)
        original = fields.get("states", {}).get(state)
        if not isinstance(original, dict) or not original.get("type", "").endswith("SpecializedGrid"):
            continue
        sizes = {row["size"] for row in periods}
        record = {"regime": name, "state": state, "periods": sorted(periods, key=lambda item: item["period"]),
                  "source": "Model._declaration_preparation.prepared_structure.grid_schedule.by_period",
                  "factories_executed_by_exporter": False}
        if len(sizes) == 1:
            original["resolved_cardinality"] = next(iter(sizes))
            original["resolved_cardinality_source"] = record["source"]
        else:
            record["status"] = "varying native cardinalities; no common size asserted"
        observed.append(record)
    # Binding fixed parameters can prepare a fresh canonical builder; the
    # declaration-preparation object then predates its resolved grid schedule.
    # Canonical phase grids still retain the concrete representative grid.
    # Native age normalization has already checked shape invariance across all
    # active periods, so read its actual cardinality, never the marker factory.
    recorded = {(row["regime"], row["state"]) for row in observed}
    for name, declaration in snapshot.get("user_regimes", {}).items():
        fields = declaration.get("fields", declaration)
        internal = getattr(model, "_regimes", {}).get(name)
        if internal is None:
            continue
        for state, original in fields.get("states", {}).items():
            if (name, state) in recorded or not isinstance(original, dict) or not original.get("type", "").endswith("SpecializedGrid"):
                continue
            counts = []
            for phase_name in ("solution", "simulation"):
                phase = getattr(internal, phase_name, None)
                grid = getattr(phase, "grids", {}).get(state)
                count = getattr(grid, "n_points", None)
                if count is None:
                    space = getattr(phase, "_base_state_action_space", None)
                    nodes = getattr(space, "states", {}).get(state)
                    shape = getattr(nodes, "shape", ())
                    count = shape[0] if len(shape) == 1 else None
                if count is not None and int(count) > 0:
                    counts.append({"phase": phase_name, "size": int(count),
                                   "grid_type": type(grid).__module__ + "." + type(grid).__qualname__})
            sizes = {row["size"] for row in counts}
            if len(sizes) == 1:
                original["resolved_cardinality"] = next(iter(sizes))
                original["resolved_cardinality_source"] = "native canonical phase grid; native specialization shape-invariance contract"
                observed.append({"regime": name, "state": state, "phases": counts,
                                 "source": "Model._regimes[regime].solution/simulation.grids[state]",
                                 "factories_executed_by_exporter": False})
    snapshot["native_grid_metadata"] = observed
    snapshot["specialization_execution"] = "Native Model constructor completed; exporter reads its stored metadata only"
    return observed


def _enrich_public_grid_metadata(declarations, snapshot):
    groups = {}
    for (period, name, phase), spec in declarations.phases.items():
        for state, grid in spec.grid_states.items():
            count = getattr(grid, 'n_points', None)
            if count is not None:
                groups.setdefault((name, state), []).append({
                    'period': period, 'phase': phase, 'size': int(count),
                    'grid_type': type(grid).__module__ + '.' + type(grid).__qualname__})
    observed = []
    for (name, state), rows in sorted(groups.items()):
        fields = snapshot.get('user_regimes', {}).get(name, {}).get('fields', {})
        original = fields.get('states', {}).get(state)
        if not isinstance(original, dict) or not original.get('type', '').endswith('SpecializedGrid'):
            continue
        sizes = {row['size'] for row in rows}
        if len(sizes) == 1:
            original['resolved_cardinality'] = next(iter(sizes))
            original['resolved_cardinality_source'] = 'pylcm.Model.get_declarations/v1'
        observed.append({'regime': name, 'state': state, 'periods': rows,
                         'source': 'pylcm.Model.get_declarations/v1',
                         'factories_executed_by_exporter': False})
    snapshot['native_grid_metadata'] = observed
    snapshot['specialization_execution'] = 'Public cached declaration view; factories not executed by exporter'
    return observed
