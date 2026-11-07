"""Run supplied original captures and account for every missing corpus input."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

from .cli import save_json


def run_corpus(manifest_path, output, *, seconds=900, host_gib=8, reuse_verified=False):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    records = []
    for ordinal, row in enumerate(manifest.get('models', manifest.get('rows', []))):
        key = row.get('key', row.get('id', str(ordinal)))
        # A generated safe directory name is independent of user labels.
        folder = output / ('model-' + str(ordinal).zfill(3))
        declared_input = row.get('input') or row.get('declaration_input')
        record = {'key': key, 'source_metadata': row,
                  'native_execution': 'NOT_RUN', 'status': 'MISSING_PRODUCTION_INPUT',
                  'missing_original_inputs': list(row.get('missing_original_inputs', [])),
                  'analysis_report': None}
        if declared_input:
            source = Path(declared_input)
            source = source if source.is_absolute() else manifest_path.parent / source
            if not source.is_file():
                record['missing_original_inputs'].append('Missing declared capture: ' + str(source))
            else:
                raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                command = [sys.executable, '-m', __package__, 'analyze-model', str(source),
                           '--output', str(folder), '--seconds', str(seconds), '--host-gib', str(host_gib)]
                # Reuse is only for assembling a final index after this exact
                # CLI already ran. Input and producer/checker identity must match.
                identity = _implementation_identity()
                existing = folder / 'corpus-run.json'
                reusable = False
                if reuse_verified and existing.exists():
                    old = json.loads(existing.read_text())
                    reusable = (old.get('input_sha256') == raw_hash and old.get('implementation_sha256') == identity
                                and (folder / 'report.json').is_file() and (folder / 'resource.json').is_file())
                if reusable:
                    run = old
                else:
                    process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
                    run = {'command': command, 'exit_code': process.returncode,
                           'stdout': process.stdout, 'stderr': process.stderr,
                           'input_sha256': raw_hash, 'implementation_sha256': identity}
                    save_json(existing, run)
                record['execution'] = run
                report_path = folder / 'report.json'
                if report_path.exists():
                    result = json.loads(report_path.read_text())
                    record.update(status=result.get('status', 'INVALID'),
                                  analysis_report=str(report_path.relative_to(output)),
                                  source_semantics=result.get('semantic_check', {}).get('status'),
                                  layers=result.get('layers', {}),
                                  frontier_points=[len(s.get('frontier', [])) for s in result.get('scenarios', [])],
                                  production_metadata=result.get('analysis', {}).get('declarations', {}).get('production_metadata'))
                    resource = folder / 'resource.json'
                    if resource.exists(): record['resources'] = json.loads(resource.read_text())
                else:
                    record.update(status='INVALID', error='The bounded worker produced no report')
        if record['analysis_report'] is None:
            folder.mkdir(parents=True, exist_ok=True)
            save_json(folder / 'missing-input.json', record)
        records.append(record)
    result = {'schema': 'regime-corpus-result/v1', 'models': records,
              'counts': {'requested': len(records), 'captures_analyzed': sum(r['analysis_report'] is not None for r in records),
                         'missing_capture': sum(r['analysis_report'] is None for r in records),
                         'complete_verified': sum(r['status'] == 'COMPLETE' for r in records),
                         'production_complete_verified': sum(r['status'] == 'COMPLETE' and
                            (r.get('production_metadata') or {}).get('status') in ('production', 'verified') for r in records)},
              'budget_per_model': {'seconds': seconds, 'host_gib': host_gib},
              'native_execution': 'NOT_RUN',
              'scope': 'Every manifest entry is accounted for. Fixture captures demonstrate declaration handling only; they do not substitute for missing production inputs.'}
    save_json(output / 'coverage.json', result)
    _render(result, output)
    return result


def _implementation_identity():
    base = Path(__file__).resolve().parent
    return hashlib.sha256(b''.join(path.name.encode() + b'\0' + path.read_bytes()
                                  for path in sorted(base.glob('*.py')))).hexdigest()


def _render(result, output):
    intro = ('# Corpus declaration coverage\n\nEvery requested entry appears below. A missing original capture is an input obligation; '
             'it is not a recommendation to retain or merge a model. Captured fixture grids and zero-valued test schedules '
             'are never relabeled as production evidence.\n\n')
    lines = [intro, '| Model | Analysis | Capture scope | Source semantics | Report |', '| --- | --- | --- | --- | --- |']
    body = []
    for record in result['models']:
        scope = (record.get('production_metadata') or {}).get('status', record['source_metadata'].get('capture_scope', 'source metadata only'))
        scope = str(scope)
        report = record.get('analysis_report')
        link = str(Path(report).with_name('report.html')) if report else None
        safe = lambda x: str(x).replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {safe(record['key'])} | {record['status']} | {safe(scope)} | {record.get('source_semantics', 'NOT_RUN')} | " + (f'[Details]({link})' if link else 'Exact missing inputs in coverage.json') + ' |')
        body.append('<tr>' + ''.join('<td>' + html.escape(str(value)) + '</td>' for value in
                    (record['key'], record['status'], scope, record.get('source_semantics', 'NOT_RUN'))) +
                    ('<td><a href="' + html.escape(link, quote=True) + '">Details</a></td>' if link else '<td>See coverage.json</td>') + '</tr>')
    (output / 'coverage.md').write_text('\n'.join(lines) + '\n')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Corpus declaration coverage</title>'
                '<style>body{font:16px/1.55 system-ui;max-width:1300px;margin:3rem auto;padding:0 2rem;background:#101827;color:#e5edf5}'
                'a{color:#8dd9ff}table{border-collapse:collapse;width:100%}th,td{padding:.6rem;border-bottom:1px solid #34445c;text-align:left}'
                'th{position:sticky;top:0;background:#152239}h1{font-size:2.2rem}</style><h1>Corpus declaration coverage</h1>'
                '<p>Every requested entry is accounted for. Missing original captures remain input obligations. Fixture captures '
                'show declaration handling; they do not establish production model performance or recommended production splits.</p>'
                '<p><a href="coverage.json">Exact evidence and missing inputs</a> · <a href="coverage.md">Markdown report</a></p>'
                '<table><thead><tr><th>Model</th><th>Analysis</th><th>Capture scope</th><th>Source semantics</th><th>Report</th></tr></thead>'
                '<tbody>' + ''.join(body) + '</tbody></table></html>')
    (output / 'coverage.html').write_text(document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=900); parser.add_argument('--host-gib', type=float, default=8)
    parser.add_argument('--reuse-verified', action='store_true')
    args = parser.parse_args(argv)
    result = run_corpus(args.manifest, args.output, seconds=args.seconds, host_gib=args.host_gib, reuse_verified=args.reuse_verified)
    print(json.dumps(result['counts'], sort_keys=True))
    return 0 if result['counts']['complete_verified'] == result['counts']['requested'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
