"""CPU-bounded declaration analysis; explicit scalar regressions stay separate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from . import find_regimes, verify_certificate, verify_production_result, is_production_input
from .expressions import jsonable


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    from .structural_store import relocate_captures
    value = relocate_captures(value, path.parent)
    temporary = path.with_name(path.name + '.writing')
    def default(item):
        converted = jsonable(item)
        if converted is item:
            raise TypeError(f'Unsupported JSON value: {type(item).__name__}')
        return converted
    # Standard JSON containers are streamed directly. Only non-JSON leaves
    # (e.g. exact Fractions) need conversion; do not duplicate a whole snapshot.
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, default=default, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def input_artifact_identity(path, *, reads=None):
    """Record exact input bytes separately from canonical object certificates."""
    path = Path(path)
    if reads is not None: path = reads.bind(path)
    with path.open('rb') as stream:
        identity = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': identity, 'filename': path.name,
            'meaning': 'exact CLI input bytes; loaded source objects have separate canonical JSON anchors'}


def candidate_boundary_maps(result):
    """Export cost preferences with their independently computed disposition."""
    from .graph_view import boundary_maps
    decisions = {(row.get('scenario_index'), row.get('graph_id')): row
                 for row in result.get('proposal_decisions', [])}
    selections = {row.get('scenario_index'): row for row in result.get('recommendations', [])}
    maps = []
    for scenario_index, scenario in enumerate(result.get('scenarios', [])):
        recommendation = selections.get(scenario_index, {}) if result.get('search_mode') == 'heuristic' else scenario.get('recommendation') or {}
        selected = recommendation.get('graph_id')
        for plan in scenario.get('candidates', scenario.get('frontier', [])):
            if plan['id'] != selected:
                continue
            decision = decisions.get((scenario_index, selected), {
                'disposition': 'investigate',
                'reasons': ['No independent local decision record was supplied.']})
            maps.append({'scenario_index': scenario_index,
                         'allocation_contract': scenario['allocation_contract'],
                         'graph_id': selected, 'cost_preferred': True,
                         'decision': decision,
                         'transformation_authorized': decision.get('disposition') == 'adopt',
                         'boundary_maps': boundary_maps(result['analysis'], plan)})
    return maps


def load_problem(path, *, reads=None):
    """Read original data. A constructed-graph manifest never enters legacy IR."""
    path = Path(path)
    if reads is None:
        from .input_reads import discover_inputs
        reads = discover_inputs([path])
    data = reads.document(path)
    if data.get('schema') == 'interned-json-cost-instance/v1':
        from .symbolic_frontier import load_packet
        return load_packet(path=path, reads=reads)
    if data.get('format') == 'lossless-json-fragments-1':
        from .read_fragments import read
        return read(path, reads=reads)
    if data.get('adapter') == 'pylcm_ast':
        from .source_adapter import adapt_manifest
        return adapt_manifest(path, reads=reads)
    if data.get('adapter') in ('pylcm_model_graph', 'pylcm_declarations'):
        def load(name):
            target = reads.child(path, name, data[name], base='supplied_parent')
            return load_problem(target, reads=reads)
        if 'declaration_spec' in data and 'snapshot' in data:
            return {'declaration_spec': load('declaration_spec'), 'graph_snapshot': load('snapshot')}
        return load('snapshot' if 'snapshot' in data else 'declaration_spec')
    from .structural_store import load_capture_structure
    return load_capture_structure(data, path, reads=reads)


def exit_status(status):
    return {'HEURISTIC': 0, 'COMPLETE': 0, 'COMPLETE_COST_FAMILY': 0, 'OPTIMAL': 0, 'PASS': 0, 'INCOMPLETE': 2,
            'UNRESOLVED': 3, 'MISSING_PRODUCTION_INPUT': 3}.get(status, 1)


def _analysis_options(parser):
    parser.add_argument('problem'); parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=900)
    parser.add_argument('--search-mode', choices=('heuristic', 'exact'), default='heuristic',
                        help='Default: checked heuristic candidates. Exact explicitly requests complete-family search.')
    parser.add_argument('--heuristic-rounds', type=int, default=2,
                        help='Optional local grouping sweep effort; no Model-size rejection threshold')
    parser.add_argument('--heuristic-neighbors', type=int, default=2,
                        help='Sparse merge-neighbor effort per group')
    parser.add_argument('--minimum-improvement', default='0',
                        help='Required relative fixed-reference scalar cost gain, in [0, 1)')
    parser.add_argument('--host-gib', type=float, default=8)
    parser.add_argument('--cost-limits', help='JSON object of exact cost-search bookkeeping limits; not Model parameters')
    parser.add_argument('--max-segments', type=int, default=2_000_000)
    parser.add_argument('--max-groupings', type=int, default=100_000)
    parser.add_argument('--max-frontier', type=int, default=5000)
    parser.add_argument('--max-projection-policies', type=int, default=4096)
    parser.add_argument('--contract', help='Actual source-bound production roles/allocation metadata JSON')
    parser.add_argument('--weights', nargs=3, metavar=('WORK', 'STORAGE', 'CLARITY'))
    parser.add_argument('--reference', nargs=3, type=int, metavar=('WORK', 'STORAGE', 'CLARITY'))
    parser.add_argument('--allow-final-only-split', action='store_true',
                        help='Allow the cost recommendation to select an extra final-date regime used only to omit inactive coordinates; the frontier is unchanged')
    parser.add_argument('--strict', action='store_true', help='Exact represented probability normalization in explicit finite scalar regressions')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)


def _run_parent(args, argv, delivery):
    from .resources import run_bounded
    check = delivery.checker
    output, destination = delivery.run_output, delivery.destination
    command_args = delivery.worker_arguments(argv)
    delivery.prepare()
    command = [sys.executable, '-m', __package__, *command_args]
    code = run_bounded(command, output=output, seconds=args.seconds, host_gib=args.host_gib,
                       mutation_plan=delivery.plan)
    if destination.exists():
        result = json.loads(destination.read_text())
    elif check and (output / 'report.json').exists():
        result = json.loads((output / 'report.json').read_text())
        delivery.publish(result) if check else save_json(destination, result)
    else:
        result = {'schema': 'regime-production-result/v1', 'status': 'INVALID',
                  'error': 'Worker produced no result; see analysis-execution.log',
                  'scenarios': [], 'layers': {'native_numerical_execution': 'NOT_RUN'}}
        delivery.publish(result) if check else save_json(destination, result)
    if check and code not in (0, 3):
        result = delivery.fail(status='INCOMPLETE' if code == 2 else 'INVALID',
                               reason='Checker worker failed its delivery boundary')
    if not check and code not in (0, 3):
        from .delivery import invalidate_delivery
        status = result.get('status')
        if status not in {'INVALID', 'INCOMPLETE', 'FAILED'}:
            status = 'INCOMPLETE' if code == 2 else 'FAILED'
        result = invalidate_delivery(output, status=status,
                                     reason=result.get('reason', result.get('error', 'Worker delivery failed')),
                                     details=result)
    if check:
        print(json.dumps(result, sort_keys=True))
    else:
        semantic = result.get('semantic_check', {})
        scenarios = result.get('scenarios', [])
        if not isinstance(semantic, dict) or not isinstance(scenarios, list):
            raise ValueError('Malformed analysis result')
        print(json.dumps({'status': result.get('status'), 'output': str(destination),
                          'resources': str(output / 'resource.json'),
                          'source_semantics': semantic.get('status'),
                          'frontier_points': [len(s.get('frontier', [])) for s in scenarios]}, sort_keys=True))
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description='Analyze original production declarations without solver execution or dense model rows.')
    sub = parser.add_subparsers(dest='command', required=True)
    _analysis_options(sub.add_parser('find-regimes'))
    _analysis_options(sub.add_parser('analyze-model'))
    for name in ('check-result', 'check-model-result'):
        check = sub.add_parser(name); check.add_argument('problem'); check.add_argument('report')
        check.add_argument('--output'); check.add_argument('--seconds', type=float, default=900)
        check.add_argument('--host-gib', type=float, default=8)
        check.add_argument('--cost-limits', help='JSON object of independent cost-check bookkeeping limits')
        check.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    adapt = sub.add_parser('adapt-source'); adapt.add_argument('manifest'); adapt.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    delivery = None
    try:
        if args.command == 'adapt-source':
            from .source_adapter import adapt_manifest
            save_json(args.output, adapt_manifest(Path(args.manifest)))
            print(json.dumps({'status': 'SUPPORTED_STATIC', 'output': args.output})); return 0
        from .delivery import resolve_result_delivery
        # An internal checker worker without --output remains stdout-only.
        # Public parents always forward their canonical destination explicitly.
        stdout_only = (args.worker and args.command in ('check-result', 'check-model-result')
                       and args.output is None)
        if not stdout_only:
            delivery = resolve_result_delivery(args.command, output=args.output,
                                               problem=args.problem,
                                               report=getattr(args, 'report', None))
        if not args.worker:
            return _run_parent(args, argv, delivery)
        if delivery is not None:
            reads = delivery.plan.reads
        else:
            from .input_reads import discover_inputs
            reads = discover_inputs([args.problem, args.report])
        problem = load_problem(args.problem, reads=reads)
        if args.cost_limits is not None and problem.get('schema') != 'exact-graph-cost-instance/v1':
            raise ValueError('--cost-limits applies only to an exact cost-family input')
        if args.command in ('check-result', 'check-model-result'):
            report = reads.document(args.report)
            recorded_input = report.get('input_artifact')
            if recorded_input and recorded_input.get('sha256') != input_artifact_identity(args.problem, reads=reads)['sha256']:
                result = {'status': 'FAIL', 'error': 'Exact input artifact differs from the analyzed source file'}
            elif report.get('schema') == 'regime-production-result/v1':
                result = verify_production_result(problem, report, limits={'seconds': args.seconds})
            elif report.get('schema') == 'regime-cost-family/v1':
                from . import check_cost_frontier
                result = check_cost_frontier(problem, report, seconds=args.seconds,
                    limits=json.loads(args.cost_limits) if args.cost_limits is not None else None)
            else:
                result = verify_certificate(problem, report)
                if result['status'] == 'PASS' and report.get('schema') != 'regime-cost-family/v1':
                    from .transport_bridge import validate_generated_transport
                    result['generated_transport'] = validate_generated_transport(problem, report)
                    if result['generated_transport']['status'] == 'FAIL': result['status'] = 'FAIL'
            from .structural_store import final_input_check
            final_input_check(problem)
            if delivery is not None:
                delivery.publish(result)
            print(json.dumps(result, indent=2, sort_keys=True)); return exit_status(result['status'])
        if delivery is None:
            raise ValueError('Analysis requires an output delivery plan')
        output = delivery.run_output; output.mkdir(parents=True, exist_ok=True)
        if problem.get('schema') == 'exact-graph-cost-instance/v1':
            from . import find_cost_frontier, check_cost_frontier
            def cost_checkpoint(record):
                if record.get('schema') == 'regime-forest-table-checkpoint/v1':
                    folder = output / 'cost-checkpoints' / record['instance_sha256']
                    folder.mkdir(parents=True, exist_ok=True)
                    table = folder / (str(record['table_index']) + '.json')
                    save_json(table, record)
                    save_json(folder / 'ROOT.json', {'table': table.name,
                        'sha256': hashlib.sha256(table.read_bytes()).hexdigest(),
                        'complete_cost_family': False,
                        'scope': 'Atomic completed table only; independent replay required; resume not claimed'})
                else:
                    save_json(output / 'cost-partial.json', record)
            started = time.perf_counter()
            cost_limits = json.loads(args.cost_limits) if args.cost_limits is not None else None
            result = find_cost_frontier(problem, seconds=args.seconds, limits=cost_limits, checkpoint=cost_checkpoint)
            checked = check_cost_frontier(problem, result,
                seconds=max(0, args.seconds - (time.perf_counter() - started)), limits=cost_limits)
            result['independent_cost_check'] = checked
            result['input_artifact'] = input_artifact_identity(args.problem, reads=reads)
            if result['status'] == 'COMPLETE_COST_FAMILY' and not checked.get('cost_family_complete_verified'):
                result['status'] = 'INCOMPLETE'
                result['reason'] = 'Independent full cost-family coverage did not finish'
            save_json(output / 'report.json', result)
            save_json(output / 'summary.json', {key: result.get(key) for key in
                ('schema', 'status', 'backend', 'scope', 'reason', 'independent_cost_check')})
        elif is_production_input(problem):
            from .production import analyze_production
            from .production_render import write_production_report
            checkpoint_started = time.perf_counter()
            phase_timings = {}
            delivery_timings = {}
            phase_started = time.perf_counter()
            save_json(output / 'original.json', problem)
            delivery_timings['original_snapshot_write_seconds'] = time.perf_counter() - phase_started
            checkpoint_seconds = 0.0
            def checkpoint(record):
                nonlocal checkpoint_seconds
                checkpoint_entry = time.perf_counter()
                stage = record['stage']
                phase_timings.update(record.get('phase_timings', {}))
                save_json(output / 'metadata-progress.json', {
                    'stage': stage, 'elapsed_seconds': time.perf_counter() - checkpoint_started,
                    'phase_timings': phase_timings,
                    'scope': 'Progress since entry to the production metadata branch; the supervisor measures the complete run. No complete or adoption claim.'})
                if stage == 'normalized':
                    save_json(output / 'normalized.json', record['declarations'])
                elif stage == 'metadata_analysis':
                    save_json(output / 'analysis.json', record['analysis'])
                elif stage == 'analysis':
                    if not (output / 'analysis.json').exists():
                        save_json(output / 'analysis.json', record['analysis'])
                    if not (output / 'normalized.json').exists():
                        save_json(output / 'normalized.json', record['analysis']['declarations'])
                    save_json(output / 'semantic-check.json', record['semantic_check'])
                else:
                    save_json(output / 'partial-frontier.json', record)
                checkpoint_seconds += time.perf_counter() - checkpoint_entry
            limits = {'seconds': args.seconds, 'segments': args.max_segments,
                      'groupings': args.max_groupings, 'frontier_points': args.max_frontier,
                      'projection_policies': args.max_projection_policies}
            recommendation = {}
            metrics = ('work', 'storage', 'clarity')
            if args.weights: recommendation['weights'] = dict(zip(metrics, args.weights))
            if args.reference: recommendation['reference'] = dict(zip(metrics, args.reference))
            recommendation['allow_final_only_split'] = args.allow_final_only_split
            recommendation['minimum_improvement'] = args.minimum_improvement
            contract = json.loads(Path(args.contract).read_text()) if args.contract else None
            phase_started = time.perf_counter()
            result = analyze_production(problem, limits=limits, recommendation=recommendation,
                                        production_contract=contract, checkpoint=checkpoint,
                                        search_mode=args.search_mode,
                                        heuristic_effort={'rounds': args.heuristic_rounds,
                                                          'neighbor_degree': args.heuristic_neighbors})
            delivery_timings['analysis_call_including_checkpoints_seconds'] = time.perf_counter() - phase_started
            delivery_timings['checkpoint_write_seconds'] = checkpoint_seconds
            phase_started = time.perf_counter()
            result['input_artifact'] = input_artifact_identity(args.problem, reads=reads)
            delivery_timings['input_identity_seconds'] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            save_json(output / 'report.json', result)
            delivery_timings['canonical_report_write_seconds'] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            from .graph_view import typed_incidence_graph
            if 'cells' in result.get('analysis', {}).get('declarations', {}):
                save_json(output / 'typed-graph.json', typed_incidence_graph(result['analysis']['declarations']))
            maps = candidate_boundary_maps(result)
            save_json(output / 'candidate-boundaries.json', maps)
            save_json(output / 'recommended-boundaries.json',
                      [row for row in maps if row['transformation_authorized']])
            delivery_timings['graph_and_boundary_artifacts_seconds'] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            manifest = write_production_report(result, output)
            save_json(output / 'report-files.json', manifest)
            delivery_timings['readable_report_seconds'] = time.perf_counter() - phase_started
            save_json(output / 'delivery-timings.json', {
                'schema': 'regime-metadata-delivery-timings/v1',
                'phase_timings': result.get('phase_timings', {}),
                'delivery_timings': delivery_timings,
                'scope': 'perf_counter subphases within the outer worker budget; analysis_call includes checkpoint_write, and frontier_search includes frontier checkpoint callbacks. Do not add overlapping fields. The outer receipt also includes input JSON parsing, summary delivery and process overhead.'})
        else:
            if args.command == 'analyze-model':
                raise ValueError('analyze-model requires original declaration metadata; use find-regimes for explicit finite scalar regressions')
            if args.strict: problem.setdefault('numerical', {})['probability_mass_absolute'] = '0'
            result = find_regimes(problem, {'candidate_segments': args.max_segments, 'seconds': args.seconds})
            if 'certificate' in result:
                result['independent_certificate_check'] = verify_certificate(problem, result)
                from .transport_bridge import validate_generated_transport
                result['generated_transport'] = validate_generated_transport(problem, result)
                if result['independent_certificate_check']['status'] == 'FAIL' or result['generated_transport']['status'] == 'FAIL':
                    result['status'] = 'INVALID'
            from .render import write_result
            write_result(output, problem, result)
        from .structural_store import final_input_check
        final_input_check(problem)
        if exit_status(result['status']) not in (0, 3):
            from .delivery import invalidate_delivery
            status = result['status'] if result['status'] in {'INVALID', 'INCOMPLETE', 'FAILED'} else 'INVALID'
            result = invalidate_delivery(output, status=status,
                                         reason=result.get('reason', result.get('error', 'Analysis did not complete its delivery boundary')),
                                         details=result)
        print(json.dumps({'status': result['status'], 'output': str(output)}, sort_keys=True))
        return exit_status(result['status'])
    except Exception as exc:
        result = {'schema': 'regime-production-result/v1', 'status': 'INVALID',
                  'error': type(exc).__name__ + ': ' + str(exc), 'scenarios': []}
        if delivery is not None:
            result = delivery.fail(status='INVALID', reason=result['error'], details=result)
        print(json.dumps(result), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
