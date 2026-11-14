"""Fresh CPU-process time and resident-memory boundary for metadata analysis."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def _tree_rss(pid):
    """Sample descendants in the caller's PID namespace, using read-only /proc.

    A container may mount /proc from an ancestor namespace. Its directory and
    PPid numbers then differ from subprocess.pid/os.getpid(). NSpid plus the
    namespace identity maps the requested root without confusing another
    namespace's process with the same local PID. (0, 0) means unavailable.
    """
    records = {}
    root = None
    try:
        namespace = os.readlink('/proc/self/ns/pid')
        entries = list(Path('/proc').iterdir())
    except OSError:
        return 0, 0
    for p in entries:
        if not p.name.isdigit():
            continue
        try:
            stat = (p/'stat').read_text().rsplit(')', 1)[1].split()
            ppid = int(stat[1])
            rss = 0
            namespace_pids = None
            for line in (p/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith('NSpid:'):
                    namespace_pids = [int(value) for value in line.split()[1:]]
            records[int(p.name)] = (ppid, rss)
            local_pid = namespace_pids[-1] if namespace_pids else int(p.name)
            if local_pid == pid and os.readlink(p/'ns/pid') == namespace:
                root = int(p.name)
        except (OSError, ValueError, IndexError):
            continue
    if root is None:
        return 0, 0
    seen = {root};changed = True
    while changed:
        more = {p for p, (parent, _) in records.items() if parent in seen} - seen
        changed = bool(more);seen.update(more)
    return sum(records[p][1] for p in seen if p in records), len(seen)


def run_bounded(command, *, output, cwd=None, seconds=900, host_gib=8,
                checkpoint_path=None, mutation_plan=None):
    if (seconds is not None and seconds < 0) or (host_gib is not None and host_gib <= 0):
        raise ValueError('Resource bounds require seconds>=0 and host_gib>0, or None for monitoring only')
    from .owned_mutations import resource_plan
    plan = mutation_plan if mutation_plan is not None else resource_plan(output)
    output = Path(output)
    plan.path_for('resource.mkdir', 'mkdir').mkdir(parents=True, exist_ok=True)
    for role in ('resource.remove_json', 'resource.remove_md'):
        stale = plan.path_for(role, 'unlink')
        if stale.is_file(): stale.unlink()
    started = time.perf_counter();maximum = 0;maximum_processes = 0;reason = None
    tree_samples = 0;unavailable_samples = 0
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    with plan.path_for('resource.log', 'open_wb').open('wb') as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        usage = None
        while True:
            found, status, current_usage = os.wait4(child.pid, os.WNOHANG)
            if found:
                usage = current_usage;child.returncode = os.waitstatus_to_exitcode(status)
                break
            rss, count = _tree_rss(child.pid)
            if count:
                tree_samples += 1
            else:
                unavailable_samples += 1
            maximum = max(maximum, rss);maximum_processes = max(maximum_processes, count)
            if seconds is not None and time.perf_counter() - started >= seconds:
                reason = 'elapsed_time'
            elif host_gib is not None and rss > host_gib * 1024**3:
                reason = 'process_tree_memory'
            if reason:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(0.05)
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
                break
            time.sleep(0.02)
    elapsed = time.perf_counter() - started
    leader_peak = usage.ru_maxrss * 1024 if usage is not None else 0
    maximum = max(maximum, leader_peak)
    if reason is None and seconds is not None and elapsed > seconds:
        reason = 'elapsed_time'
    if reason is None and host_gib is not None and maximum > host_gib * 1024**3:
        reason = 'process_tree_memory'
    record = {'command': command, 'exit_code': child.returncode,
              'elapsed_seconds': elapsed, 'peak_host_gib': maximum / 1024**3,
              'maximum_observed_processes': maximum_processes,
              'process_tree_rss_available': tree_samples > 0 and unavailable_samples == 0,
              'process_tree_rss_samples': tree_samples,
              'process_tree_rss_unavailable_samples': unavailable_samples,
              'resource_status': 'SAMPLED_PROCESS_TREE' if tree_samples > 0 and unavailable_samples == 0 else 'UNVERIFIED_PROCESS_TREE',
              'leader_peak_host_gib': leader_peak / 1024**3,
              'measurement': '20ms sampled sum of process-tree RSS, with PID-namespace translation, supplemented by wait4 leader maximum RSS. Transient unsampled descendant peaks are not an exact tree-wide maximum. If process_tree_rss_available is false, peak_host_gib is only an observed lower bound, not a verified tree-memory bound. The worker performs metadata analysis without model child jobs.',
              'budget': {'seconds': seconds, 'host_gib': host_gib}, 'exhaustion_reason': reason,
              'within_declared_budget': reason is None and (seconds is None or elapsed <= seconds) and (host_gib is None or maximum <= host_gib * 1024**3) and tree_samples > 0 and unavailable_samples == 0,
              'external_data_provisioning_seconds': None}
    if reason:
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else output/'partial-frontier.json'
        # Checkpoints are producer output, and can be much larger than the final
        # receipt. Never decode/copy one outside the bounded worker after a kill.
        # Its candidates require source-bound semantic and cost checks before use.
        reference = ({'path': os.path.relpath(checkpoint, output), 'size_bytes': checkpoint.stat().st_size,
                      'candidate_safety': 'unverified',
                      'semantic_check': 'NOT_RUN_ON_CHECKPOINT',
                      'optimization_check': 'NOT_RUN_ON_CHECKPOINT'}
                     if checkpoint.is_file() else None)
        result = {'schema': 'regime-production-result/v1', 'status': 'INCOMPLETE',
                  'exhaustion_reason': reason, 'partial_frontier': None,
                  'partial_frontier_reference': reference, 'recommendation': None,
                  'established_lower_bound': dict.fromkeys(('work', 'storage', 'clarity'), 0),
                  'lower_bound_basis': 'All admitted structural work, storage and clarity costs are nonnegative; this is only the trivial lower bound.',
                  'scope': 'Resource boundary stopped the unchanged original-input analysis. A retained checkpoint is unverified candidate evidence, not an adoption certificate. No global optimum or complete discovery is claimed.'}
        plan.path_for('resource.incomplete_json', 'write_text').write_text(json.dumps(result, indent=2) + '\n')
        plan.path_for('resource.incomplete_md', 'write_text').write_text('# Analysis incomplete\n\nThe CPU resource limit was reached. '
                                      'The original input was not reduced. Any retained checkpoint is referenced by report.json '
                                      'and must pass separate source-bound semantic and cost checks before adoption.\n')
    # Include the constant-size exhaustion report in the supervisor timing. The
    # receipt itself is small; sampled RSS limitations remain explicit above.
    record['elapsed_seconds'] = time.perf_counter() - started
    record['within_declared_budget'] = (reason is None and
        (seconds is None or record['elapsed_seconds'] <= seconds) and (host_gib is None or maximum <= host_gib * 1024**3) and
        tree_samples > 0 and unavailable_samples == 0)
    record['terminal_reporting'] = 'constant-size metadata only; retained checkpoint bytes are not decoded by the supervisor'
    plan.path_for('resource.receipt', 'write_text').write_text(json.dumps(record, indent=2) + '\n')
    return 2 if reason else child.returncode
