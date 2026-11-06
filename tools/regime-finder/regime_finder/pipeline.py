"""Public metadata task runner with readable, source-bound next actions."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def readable_summary(report, *, snapshot, output):
    decisions = report.get('proposal_decisions',[])
    family = report.get('analysis', {}).get('family_boundary', {})
    rows=[]
    for decision in decisions:
        counts=Counter(item.get('kind','unspecified') for item in decision.get('residual_obligations',[]))
        examples=[]
        for item in decision.get('residual_obligations',[]):
            detail={k:item[k] for k in ('kind','cell','callback','phase','source_path','detail','reasons') if k in item}
            if detail not in examples:examples.append(detail)
            if len(examples)>=8:break
        rows.append({'operation':decision.get('operation'),'scenario':decision.get('scenario_index'),
                     'graph':decision.get('graph_id'),'decision':decision.get('disposition'),
                     'changed_interfaces':decision.get('changed_interfaces',[]),
                     'remaining_kinds':dict(counts),'examples':examples,
                     'reasons':decision.get('reasons',[])})
    summary={'status':report.get('status'),'source_snapshot_sha256':_digest(snapshot),
             'checker_module_sha256':{p.name:_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
             'semantic_status':report.get('semantic_check',{}).get('status'),
             'semantic_errors':report.get('semantic_check',{}).get('errors',[]),
             'search_mode':report.get('search_mode', 'exact'),
             'recommendations':report.get('recommendations', []),
             'decisions':rows,'execution':report.get('analysis',{}).get('production_execution'),
             'family_boundary':family,
             'numerical_scope':'Metadata only; no solve, simulation or scientific validation',
             'paper_action_policy':'Do not implement recommendations without separate authorization'}
    selected = {row.get('scenario_index'): row for row in report.get('recommendations', [])}
    summary['recommendation_preferences'] = [
        {'scenario': index, 'recommended_graph': (selected.get(index, {}) if report.get('search_mode') == 'heuristic' else scenario.get('recommendation') or {}).get('graph_id'),
         **((selected.get(index, {}) if report.get('search_mode') == 'heuristic' else scenario.get('recommendation') or {}).get('endpoint_preference') or {})}
        for index, scenario in enumerate(report.get('scenarios', []))]
    lines=['# Regime analysis: what can be concluded','',
           'This analysis uses production declarations, not a smaller numerical model. It does not certify paper economics or measured speed.','',
           '## Read the decisions','',
           '- **Retain original / adopt:** keeping the exact supplied graph is safe by identity. This does not establish that it is optimal.',
           '- **Representation / adopt:** the proposed specification passes its source-bound local checks. It is not permission to edit paper code.',
           '- **Investigate:** the named premise or resource check is unfinished; this is not an instruction to rewrite working paper code.',
           '- **Reject:** an independently checked contradiction was found.','']
    if report.get('search_mode') == 'heuristic':
        lines += ['## Heuristic search scope', '',
                  'Use each independent candidate-cost and local source-semantic check record. '
                  'The search can omit layouts and has an unknown optimality gap. '
                  'Each selected graph below is a freshly checked adoptable specification; otherwise retain the original. '
                  'Request --search-mode exact only for an explicit complete-family oracle run.', '']
    for preference in summary['recommendation_preferences']:
        if 'allow_final_only_split' in preference:
            lines += ['## Endpoint recommendation preference', '', preference['reason'], '',
                      'Explicit override: ' + str(preference['allow_final_only_split']) + '. '
                      'Endpoint-only alternatives: ' + (', '.join(preference['deferred_graph_ids']) or 'none') + '. '
                      'Recommended graph: ' + str(preference['recommended_graph']) + '.', '',
                      'The alternatives remain visible with their separate semantic checks. '
                      'Use --allow-final-only-split, or allow_final_only_split=True in analyse_snapshot, to permit them in the recommendation.', '']
    if family:
        lines += ['## Search scope and unresolved merge candidates', '',
                  family.get('complete_claim_scope', 'No complete-family scope recorded') + '.', '',
                  'This is not an unrestricted optimum over all possible model representations. '
                  'Final-date simplification is a recommendation preference, not an admissibility exclusion.', '']
        candidates = family.get('conditional_merge_candidates', [])
        for candidate in candidates[:8]:
            lines.append('- **' + ' / '.join(candidate.get('regimes', [])) + '**: ' +
                         candidate.get('status', 'unresolved') + '. ' +
                         ' '.join(item.get('required_fact', '') for item in candidate.get('required_obligations', [])))
        unexamined = family.get('conditional_merge_pairs_unexamined', 0)
        omitted = family.get('conditional_merge_candidates_omitted_by_display_budget', 0)
        lines += ['', f'{len(candidates)} conditional candidates are recorded; {unexamined} original-regime pairs were not examined under the diagnostic budget; {omitted} further candidates were omitted from the diagnostic display.',
                  'The exact source paths, matching protected facts and component-specific obligations are in family-boundary.json. These records do not authorize a graph change.', '']
        (output/'family-boundary.json').write_text(json.dumps(family, indent=2)+'\n')
    for row in rows:
        title='Retain supplied original' if row['operation']=='retain_original' else f'Scenario {row["scenario"]}: {row["graph"]}'
        lines.extend(['## '+title,'','Decision: **'+str(row['decision'])+'**.',''])
        if row['changed_interfaces']:
            for change in row['changed_interfaces'][:12]:
                removed=set(change.get('original_states',{}))-set(change.get('states',{}))
                removed_actions=set(change.get('original_actions',{}))-set(change.get('actions',{}))
                lines.append(f'- {change.get("original_regime")} at period {change.get("period")}, age {change.get("age")}: omitted states {", ".join(sorted(removed)) or "none"}; omitted action axes {", ".join(sorted(removed_actions)) or "none"}.')
        if row['remaining_kinds']:
            lines.extend(['','### What is still needed',''])
            for kind in row['remaining_kinds']:
                explanation={
                    'allocation_contract':'Backend allocation authority is missing. Structural storage estimates are conditional; do not claim measured memory or speed gains.',
                    'role_timing_contract':'This proposal still needs a justified relationship between original actor/information/timing contexts. Same-context identity is checked separately; cross-context equivalence is not assumed.',
                    'capture_identity':'Recapture the original callable and immutable bindings with the current reader; the retained capture does not establish its actual target identity.',
                    'callback_interpretation':'Provide a checked interpretation for the named operation if this proposal changes its application. Unchanged opaque applications are checked by congruence.',
                    'opaque consumer factorization':'Prove the named changed callback can use the projected inputs without changing its result; do not drop its unknown inputs.',
                    'variadic_dependency':'Establish the actual wrapper forwarding/binding contract. A variadic signature alone cannot prove its input dependencies.',
                    'original policy query factorization unproved':'Connect the original owned query to its actual requested value/policy evaluator, including phase, information and projection.',
                    'policy_query_evaluation':'The query owner exists, but its evaluation/factorization still needs an independently checked bridge.',
                    'proposal checking budget exhausted':'Repeat or improve the bounded metadata check; the incomplete check authorizes no new transformation.',
                }.get(kind,'Resolve the exact source-bound premise shown in the detailed report.')
                lines.append('- **'+kind+'**: '+explanation)
            lines+=['','The first exact locations/reasons are below; the complete list remains in the report.','', '```json',json.dumps(row['examples'],indent=2),'```','']
        else:lines+=['The local decision has no remaining recorded premise. Its scope remains the representation specification above.','']
    lines+=['## Evidence and use','',
            '[Detailed readable report](report.html) · [Exact proposal decisions](proposal-decisions.json) · [Bounded whole-run receipt](outer-resource/resource.json)','',
            '**No recommendation is applied automatically.**','']
    text='\n'.join(lines)
    (output/'next-actions.md').write_text(text)
    # Existing renderer produces HTML without a third-party Markdown runtime.
    chunks=[];code=False;buffer=[]
    for line in lines:
        if line.startswith('```'):
            if code:chunks.append('<details><summary>Exact diagnostic locations</summary><pre>'+html.escape('\n'.join(buffer))+'</pre></details>');buffer=[]
            code=not code;continue
        if code:buffer.append(line);continue
        if line.startswith('# '):chunks.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):chunks.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('### '):chunks.append('<h3>'+html.escape(line[4:])+'</h3>')
        elif line:chunks.append('<p>'+html.escape(line.replace('**',''))+'</p>')
    chunks.append('<p><a href="report.html">Detailed report</a> · <a href="proposal-decisions.json">Exact decisions</a> · <a href="outer-resource/resource.json">Whole-run resource receipt</a></p>')
    (output/'next-actions.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Regime analysis: next actions</title><style>body{background:#101820;color:#d9e4ed;max-width:1100px;margin:2rem auto;padding:1rem;font:16px/1.6 system-ui;overflow-wrap:anywhere}a{color:#79d8e8}pre{white-space:pre-wrap;background:#182732;padding:1rem}</style></head><body>'+''.join(chunks)+'</body></html>')
    from .pytask_api import _write_json
    _write_json(output/'summary.json', summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=900);parser.add_argument('--host-gib',type=float,default=8)
    parser.add_argument('--search-mode', choices=('heuristic', 'exact'), default='heuristic')
    parser.add_argument('--heuristic-rounds', type=int, default=2)
    parser.add_argument('--heuristic-neighbors', type=int, default=2)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--allow-final-only-split',action='store_true',help='Allow an endpoint-only extra regime in the recommendation; preserve the same frontier')
    args=parser.parse_args()
    # Every failure handler below may write the complete output namespace.
    # Reject overlap before source hashing, checker calls or invalidation.
    if args.snapshot.resolve().is_relative_to(args.output.resolve()):
        raise ValueError('Analysis output must not contain the original input')
    if not args.worker:
        from .pytask_api import analyse_snapshot
        result=analyse_snapshot(args.snapshot,args.output,seconds=args.seconds,host_gib=args.host_gib,
                                allow_final_only_split=args.allow_final_only_split,
                                search_mode=args.search_mode,
                                heuristic_rounds=args.heuristic_rounds,
                                heuristic_neighbors=args.heuristic_neighbors)
        print(json.dumps({'status':result['status'],'readable_report':str(args.output/'next-actions.html')}))
        return 0
    from .cli import main as cli_main
    try:
        identities_before={p.name:_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        arguments=['analyze-model',str(args.snapshot),'--output',str(args.output),
                   '--seconds',str(args.seconds),'--host-gib',str(args.host_gib),'--worker',
                   '--search-mode', args.search_mode, '--heuristic-rounds', str(args.heuristic_rounds),
                   '--heuristic-neighbors', str(args.heuristic_neighbors)]
        if args.allow_final_only_split:
            arguments.append('--allow-final-only-split')
        code=cli_main(arguments)
        report=json.loads((args.output/'report.json').read_text())
        if code not in (0, 3):
            from .delivery import invalidate_delivery
            status = report.get('status')
            if status not in {'INVALID', 'INCOMPLETE', 'FAILED'}:
                status = 'INCOMPLETE' if code == 2 else 'INVALID'
            invalidate_delivery(args.output, status=status,
                                reason=report.get('reason', report.get('error', 'Checker failed')),
                                details=report)
            return code
        summary=readable_summary(report,snapshot=args.snapshot,output=args.output)
        summary['checker_modules_at_start']=identities_before
        summary['checker_modules_unchanged']=identities_before==summary['checker_module_sha256']
        from .pytask_api import _write_json
        _write_json(args.output/'summary.json', summary)
        if not summary['checker_modules_unchanged']:
            from .delivery import invalidate_delivery
            invalidate_delivery(args.output, status='INVALID',
                                reason='Checker implementation changed during analysis',
                                details={'checker_modules_unchanged': False})
            return 1
        return code
    except Exception as error:
        from .delivery import invalidate_delivery
        invalidate_delivery(args.output, status='INVALID',
                            reason=type(error).__name__ + ': ' + str(error))
        return 1

if __name__=='__main__':raise SystemExit(main())
