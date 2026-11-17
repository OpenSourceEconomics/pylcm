"""JSON, Markdown, and Mermaid output; no browser or image renderer required."""
import json
from fractions import Fraction
from pathlib import Path


def _label(text):
    return str(text).replace('"', "'").replace('\n',' ').replace('`','').replace('<','(').replace('>',')')


def graph(report, problem=None):
    problem = problem or {}
    aliases = problem.get('source_adapter', {}).get('coordinate_names', {})
    def coordinate(name):
        return aliases.get(name, name)
    def age_label(tick):
        label = problem.get('age_labels', {}).get(str(tick), str(tick))
        try:
            value = Fraction(label)
            return str(value.numerator) if value.denominator == 1 else f'{float(value):.3f}'.rstrip('0').rstrip('.')
        except (ValueError, TypeError, ZeroDivisionError):
            return label
    lines=['flowchart TD']
    regimes=report.get('recommended',{}).get('regimes',[])
    for r in regimes:
        ages=r['ages'];age_text=age_label(ages[0]) if len(ages)==1 else f'{age_label(ages[0])}..{age_label(ages[-1])}'
        global_steps = {b-a for a,b in zip(problem.get('ages', []), problem.get('ages', [])[1:])}
        local_steps = {b-a for a,b in zip(ages,ages[1:])}
        if len(ages)>1 and local_steps and local_steps != global_steps:
            age_text += f', step {next(iter(local_steps))}' if len(local_steps)==1 else ', explicit date mask'
        statuses=r.get('status_admission',[])
        status_text='; '.join(', '.join(f'{k}={v}' for k,v in s.items()) for s in statuses if s)
        live=sorted(set(coordinate(x) for s in r['semantic_interfaces'].values() for x in s['states']))
        act=sorted(set(coordinate(x) for s in r['semantic_interfaces'].values() for x in s['actions']))
        original=' / '.join(r.get('original_labels', []))
        time_word = 'period' if problem.get('source_adapter', {}).get('time_coordinate_kind') == 'period' else 'age'
        label=f"{r['id']} {original} | {time_word} {age_text}"
        if status_text and 'graph_lane' not in status_text:label+=' | '+status_text
        label+=' | states: '+(', '.join(live) or 'none')+' | actions: '+(', '.join(act) or 'none')
        lines.append(f'  {r["id"]}["{_label(label)}"]')
    edges={}
    for edge in report.get('recommended',{}).get('transitions',[]):
        if edge['source']==edge['target']:continue
        key=(edge['source'],edge['target'])
        edges.setdefault(key,set()).add(edge['kind']+':'+edge['phase'])
    for (source,target),kinds in sorted(edges.items()):
        lines.append(f'  {source} -->|"{_label(", ".join(sorted(kinds)))}"| {target}')
    if not regimes:lines.append('  unresolved["No admitted recommendation"]')
    return '\n'.join(lines)+'\n'


def markdown(problem,report):
    lines=[f"# Regime finder: {problem.get('name','original problem')}",'',f"**Result: {report['status']}**",'']
    if report.get('error'):lines += [report['error'],'']
    if 'recommended'not in report:return '\n'.join(lines)+'\n'
    lines += ['The recommendation is conditional on the supplied original primitive and source-adapter contracts. Semantic interfaces, numerical observations, implementation restrictions, and cost optimality are reported separately.','',
              '## Cost and search','',
              '| Representation | Enumeration | Boundary work | Total work | Blocks | Allocated value entries |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label,key in [('Original','original'),('Recommended','recommended')]:
        c=report[key]['cost'];lines.append(f"| {label} | {c['enumeration']} | {c['boundary']} | {c['work']} | {c['blocks']} | {c['storage']} |")
    search=report['search']
    lines += ['',f"Search complete: **{search['complete']}**. Candidate intervals evaluated: **{search['segments_evaluated']}**. The objective orders declared enumeration plus source-row boundary work, then block count. Ties use earliest cuts and favor the specialized branch when branch scores are equal.",'',
              search['optimality_scope'],'',
              'Storage is the sum of declared value-table entries, not peak live memory. Work is an integer proxy, not a hardware-runtime claim. '+
              (report['search']['counterfactual_storage'] if 'counterfactual_storage' in report['search'] else 'The counterfactual segment table and every rejected interval are in report.json.'),'',
              '## Recommended regimes','',
              '| Regime | Age ticks | Observed-status admission | Semantic states | Free actions | Allocated solve axes |',
              '| --- | --- | --- | --- | --- | --- |']
    for r in report['recommended']['regimes']:
        live=sorted(set(x for s in r['semantic_interfaces'].values() for x in s['states']))
        act=sorted(set(x for s in r['semantic_interfaces'].values() for x in s['actions']))
        age=str(r['ages'][0]) if len(r['ages'])==1 else f"{r['ages'][0]}–{r['ages'][-1]}"
        lines.append('| '+' | '.join([r['id'],age,json.dumps(r.get('status_admission',[]),sort_keys=True),', '.join(live)or'none',', '.join(act)or'none',', '.join(r['allocated_state_axes'])or'none'])+' |')
    if 'age_scale'in problem:lines += ['',f"One age tick represents {problem['age_scale']} source-age units; age_labels preserves the source labels."]
    lines += ['', 'Each regime also records per-atom admission and semantic interfaces, exact forced-action reconstruction constants, and boundary projections. Union axes describe dense evaluation padding over fixed semantic states. They do not introduce economically meaningful latent histories.','',
              '## Diagnostics','']
    for d in report['diagnostics']:
        if d['kind']=='RETAIN' and d.get('proposed_change')=='specialize function signatures and project boundary state':continue
        lines.append(f"- **{d['kind']}** at tick {d.get('at_age','n/a')}: {d.get('reason','')}. Proposed change: {d.get('proposed_change','')}.")
    lines += ['', 'Full per-atom liveness, retained-history witnesses, and specialized argument lists are in report.json.','', '## Claims and limits','']
    for name,layer in report['layers'].items():
        lines.append(f"- **{name}: {layer['status']}** — {' '.join(layer.get('notes',[]))}")
    transport=report.get('generated_transport')
    if transport:lines += ['',f"Generated finite transport verification: **{transport['status']}**. See transport-check.json for the exact materialization scope."]
    lines += ['', 'The generated search certificate is checked by enumerating the complete admitted segment family and verifying lower-bound inequalities and path attainment. Arbitrary state sufficiency, cross-status coalescence, continuous domains, native pylcm code generation, and runtime speed are outside this optimum claim.','']
    return '\n'.join(lines)


def write_result(output,problem,report):
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('problem.json',problem),('report.json',report),('certificate.json',report.get('certificate',{})),
                       ('certificate-check.json',report.get('independent_certificate_check',{})),
                       ('transport-check.json',report.get('generated_transport',{}))]:
        (output/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    (output/'report.md').write_text(markdown(problem,report))
    (output/'recommended.mmd').write_text(graph(report,problem))
