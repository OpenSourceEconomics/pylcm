"""Independent optimum-certificate check, without invoking the search recurrence.

The semantic interpreter and segment contract are shared definitions. This
checker regenerates EVERY possible segment, checks every lower-bound inequality,
and checks attainment by the supplied path. It does not call find_regimes.
"""
from __future__ import annotations

from copy import deepcopy
from .core import prepare, segment, original_partition, original_admissibility, total_cost, digest, _public_model, reported_transitions


def verify_certificate(problem, report):
    if not isinstance(problem, dict):
        from .model_adapter import adapt_model
        problem = adapt_model(problem)
    if report.get('certificate', {}).get('version') == 2:
        from .compact import verify_compact
        return verify_compact(problem, report)
    errors=[]
    def require(condition,message):
        if not condition: errors.append(message)
    try:
        cert=report['certificate']
        require(cert.get('problem_sha256')==digest(problem),'original problem identity mismatch')
        require(report.get('problem_sha256')==digest(problem),'report problem identity mismatch')
        if errors:return {'status':'FAIL','errors':errors}
        model=prepare(problem)
        require(cert.get('analysis_sha256')==digest(_public_model(model)),'derived semantic interface digest mismatch')
        require(report.get('analysis')==_public_model(model),'reported semantic interfaces or support edges were altered')
        original=original_partition(problem,model)
        original_ok=original_admissibility(problem,model,original)
        oc=total_cost(original)
        require(report['original']['cost']==oc,'original allocation cost mismatch')
        require(report['original'].get('admissible')==original_ok['admissible'],'original admissibility mismatch')
        if not cert.get('complete'):
            require(report.get('status')=='INCOMPLETE' and report['search'].get('complete') is False,'incomplete search mislabeled as optimal')
            require(cert.get('branch')=='original','incomplete certificate may only claim its verified original incumbent')
            require(report['recommended']['cost']==oc,'incumbent cost mismatch')
            for i,s in enumerate(original):s['id']=f'R{i:03d}'
            require(report['recommended']['regimes']==original,'incumbent representation mismatch')
            return {'status':'FAIL' if errors else 'INCOMPLETE','errors':errors,'optimum_verified':False,
                    'incumbent_verified':not errors and original_ok['admissible'],'lower_bound':[0,0]}
        require(report.get('status')=='OPTIMAL' and report['search'].get('complete') is True,'complete result has inconsistent status')
        rebuilt={}; chosen=[]; refined_score=[0,0]; all_reachable=True
        inequality_count=0; segment_count=0
        for key,lane in model['lanes'].items():
            n=len(lane)
            pot=cert['potentials'].get(key)
            require(isinstance(pot,list) and len(pot)==n+1,f'{key}: invalid potential table length')
            if not isinstance(pot,list) or len(pot)!=n+1:continue
            require(pot[-1]==[0,0],f'{key}: sink potential must be zero')
            for v in pot:
                require(v is None or (isinstance(v,list) and len(v)==2 and all(type(x)is int and x>=0 for x in v)),f'{key}: invalid potential score')
            if errors:return {'status':'FAIL','errors':errors}
            edges=[]; records=[]
            for i in range(n):
                for j in range(i+1,n+1):
                    segment_count+=1
                    s,reason=segment(problem,model,lane,i,j)
                    if s is None:continue
                    edges.append((i,j,s))
                    records.append({'start':i,'stop':j,'cost':s['cost']})
                    if pot[j] is not None:
                        require(pot[i] is not None,f'{key}: reachable suffix falsely labeled unreachable')
                        if pot[i] is not None:
                            rhs=(s['cost']['work']+pot[j][0],1+pot[j][1])
                            require(tuple(pot[i])<=rhs,f'{key}: lower-bound inequality violated on {i}:{j}')
                            inequality_count+=1
            rebuilt[key]=records
            # Pure reachability, not an optimization recurrence.
            reachable={n}
            for i in range(n-1,-1,-1):
                if any(a==i and b in reachable for a,b,_ in edges):reachable.add(i)
                require((pot[i] is not None)==(i in reachable),f'{key}: false feasibility in potential table')
            if 0 not in reachable:
                all_reachable=False;continue
            refined_score[0]+=pot[0][0];refined_score[1]+=pot[0][1]
            # Reconstruct the unique earliest attaining path directly from inequalities.
            cursor=0; path=[]
            while cursor<n:
                attaining=[(j,s) for i,j,s in edges if i==cursor and pot[j] is not None
                           and pot[cursor]==[s['cost']['work']+pot[j][0],1+pot[j][1]]]
                require(bool(attaining),f'{key}: lower bound is not attained by any admissible path')
                if not attaining:break
                j,s=min(attaining,key=lambda x:x[0]);path.append([cursor,j]);chosen.append(s);cursor=j
            if cert['branch']=='refined':
                require(cert.get('chosen_segments',{}).get(key)==path,f'{key}: selected path or deterministic tie rule mismatch')
        require(cert.get('family_sha256')==digest(rebuilt),'complete admissible family digest mismatch')
        expected_branch='refined' if all_reachable and (not original_ok['admissible'] or tuple(refined_score)<=tuple([oc['work'],oc['blocks']])) else 'original'
        require(expected_branch=='refined' or original_ok['admissible'],'no admissible incumbent in claimed complete family')
        require(cert.get('branch')==expected_branch,'branch is not globally optimal over original and refined families')
        selected=chosen if expected_branch=='refined' else original
        for i,s in enumerate(selected):s['id']=f'R{i:03d}'
        expected_cost=total_cost(selected)
        require(report['recommended']['regimes']==selected,'recommended layout/domain/boundary map differs from verified selected segments')
        require(report['recommended']['cost']==expected_cost,'recommended cost mismatch')
        require(cert.get('score')==[expected_cost['work'],expected_cost['blocks']],'claimed optimum score mismatch')
        require(report['search'].get('lower_bound')==cert.get('score') and report['search'].get('gap')==0,'optimality gap/lower bound mismatch')
        expected_transitions=reported_transitions(model,selected)
        require(report['recommended']['transitions']==expected_transitions,'generated transition projections mismatch')
        require(report['recommended'].get('external_report_tags',[])==problem.get('external_report_tags',[]),'external subject/report association dropped')
        return {'status':'FAIL' if errors else 'PASS','errors':errors,'optimum_verified':not errors,
                'segments_reconstructed':segment_count,'bellman_inequalities_checked':inequality_count,
                'score':[expected_cost['work'],expected_cost['blocks']],
                'scope':'Exact integer cost optimum within the regenerated fixed-interface interval/status family; not a source-to-kernel proof or hardware-runtime optimum.'}
    except Exception as exc:
        return {'status':'FAIL','errors':errors+[type(exc).__name__+': '+str(exc)],'optimum_verified':False}
