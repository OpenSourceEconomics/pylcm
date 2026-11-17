"""Public metadata task runner with readable, source-bound next actions."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def readable_summary(report, *, snapshot, output):
    layouts = report.get('layouts') or {}
    summary={'status':report.get('status'),'source_snapshot_sha256':_digest(snapshot),
             'checker_module_sha256':{p.name:_digest(p) for p in sorted(Path(__file__).parent.iterdir())
                                      if p.suffix in ('.py', '.json')},
             'semantic_status':report.get('semantic_check',{}).get('status'),
             'semantic_errors':report.get('semantic_check',{}).get('errors',[]),
             'layout_status':layouts.get('status'),
             'layout_check':report.get('layout_check',{}).get('status'),
             'original_cost':report.get('original_cost'),
             'curve':[{k: p[k] for k in ('max_blocks','blocks','evaluations','storage')}
                      for p in layouts.get('curve', [])],
             'recommendation':report.get('recommendation'),
             'execution':report.get('analysis',{}).get('production_execution'),
             'numerical_scope':'Metadata only; no solve, simulation or scientific validation',
             'paper_action_policy':'Do not implement recommendations without separate authorization'}
    lines=['# Regime analysis: what can be concluded','',
           'This analysis uses production declarations, not a smaller numerical model. It does not certify paper economics or measured speed.','',
           'Status: **'+str(report.get('status'))+'**; source check: '+str(summary['semantic_status'])+
           '; layout check: '+str(summary['layout_check'])+'.','']
    if layouts.get('status') == 'RESTRICTED':
        lines += ['The statuses '+', '.join(layouts.get('restricted_statuses', []))+' were kept as ordinary coordinates, '
                  'so every count below is an upper bound.','']
    lines += ['## Layout curve','','| at most k regimes | regimes | evaluations | stored values |','|---|---|---|---|']
    lines += [f"| {p['max_blocks']} | {p['blocks']} | {p['evaluations']:,} | {p['storage']:,} |" for p in summary['curve']]
    chosen = report.get('recommendation') or {}
    lines += ['','## Recommendation','',
              (f"The layout with at most {chosen['max_blocks']} regimes." if chosen.get('operation') == 'layout'
               else 'Retain the original model: '+str(chosen.get('reason', 'no checked layout curve'))+'.'),'',
              '## Evidence and use','',
              '[Detailed readable report](report.html) · [Bounded whole-run receipt](outer-resource/resource.json)','',
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
    chunks.append('<p><a href="report.html">Detailed report</a> · <a href="outer-resource/resource.json">Whole-run resource receipt</a></p>')
    (output/'next-actions.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Regime analysis: next actions</title><style>body{background:#101820;color:#d9e4ed;max-width:1100px;margin:2rem auto;padding:1rem;font:16px/1.6 system-ui;overflow-wrap:anywhere}a{color:#79d8e8}pre{white-space:pre-wrap;background:#182732;padding:1rem}</style></head><body>'+''.join(chunks)+'</body></html>')
    from .pytask_api import _write_json
    _write_json(output/'summary.json', summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=900);parser.add_argument('--host-gib',type=float,default=8)
    parser.add_argument('--facts-backend', choices=('metadata', 'dense', 'implicit'), required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    # Every failure handler below may write the complete output namespace.
    # Reject overlap before source hashing, checker calls or invalidation.
    if args.snapshot.resolve().is_relative_to(args.output.resolve()):
        raise ValueError('Analysis output must not contain the original input')
    if not args.worker:
        from .pytask_api import analyse_snapshot
        result=analyse_snapshot(args.snapshot,args.output,seconds=args.seconds,host_gib=args.host_gib,
                                facts_backend=args.facts_backend)
        print(json.dumps({'status':result['status'],'readable_report':str(args.output/'next-actions.html')}))
        return 0
    from .cli import main as cli_main
    try:
        identities_before={p.name:_digest(p) for p in sorted(Path(__file__).parent.iterdir())
                           if p.suffix in ('.py', '.json')}
        arguments=['analyze-model',str(args.snapshot),'--output',str(args.output),
                   '--seconds',str(args.seconds),'--host-gib',str(args.host_gib),'--worker',
                   '--facts-backend', args.facts_backend]
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
