"""Revoke every public decision export when a delivery boundary fails.

Original inputs and diagnostic checkpoints remain available. Derived exports
are valid only after all source, checker and resource boundaries have passed;
rewriting report.json alone cannot revoke a standalone adoption record.
"""
import html
import json
from dataclasses import dataclass
from pathlib import Path

from .owned_mutations import Mutation, OwnedMutationPlan, resource_mutations


DECISION_EXPORTS = (
    'candidate-boundaries.json', 'recommended-boundaries.json',
    'proposal-decisions.json', 'report-data.json', 'report-files.json', 'report-manifest.json',
    'family-boundary.json', 'certificate.json', 'certificate-check.json',
    'transport-check.json', 'recommended.mmd',
)


@dataclass(frozen=True)
class ResultDelivery:
    """The validated result destination for one complete CLI invocation.

    Resolve once before work begins. The parent, worker and exception boundary
    then refer to the same output, including the checker's implicit destination.
    Original inputs and the temporary result path are checked at resolution.
    """
    destination: Path
    run_output: Path
    checker: bool
    plan: OwnedMutationPlan
    startup_roles: tuple[str, ...]

    def worker_arguments(self, argv):
        """Forward the canonical output, independent of how it was specified."""
        arguments = []
        skip_value = False
        # Forward options before argparse's end-of-options marker.
        boundary = argv.index('--') if '--' in argv else len(argv)
        for argument in argv[:boundary]:
            if skip_value:
                skip_value = False
            elif argument == '--output':
                skip_value = True
            elif not argument.startswith('--output='):
                arguments.append(argument)
        target = self.destination if self.checker else self.run_output
        return [*arguments, '--output', str(target), '--worker', *argv[boundary:]]

    def fail(self, *, status, reason, details=None):
        """Revoke this invocation's result after a completed failed boundary."""
        if not self.checker:
            return invalidate_delivery(self.run_output, status=status,
                                       reason=reason, details=details)
        if status not in {'INVALID', 'INCOMPLETE', 'FAILED'}:
            raise ValueError('Delivery invalidation requires a failure status')
        # A failure record cannot carry a previous independent PASS or authority.
        record = {'status': status, 'error': str(reason),
                  'transformation_authorized': False}
        self.publish(record)
        return record

    def publish(self, record):
        """Publish only through the same admitted final/temporary roles."""
        from .expressions import jsonable
        self.plan.path_for('result.mkdir', 'mkdir').mkdir(parents=True, exist_ok=True)
        with self.plan.path_for('result.open', 'open_w').open('w', encoding='utf-8') as stream:
            json.dump(record, stream, default=jsonable, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        self.plan.path_for('result.replace_source', 'replace').replace(
            self.plan.path_for('result.replace_target', 'replace'))

    def prepare(self):
        self.plan.path_for('parent.mkdir', 'mkdir').mkdir(parents=True, exist_ok=True)
        for role in self.startup_roles:
            path = self.plan.path_for(role, 'unlink')
            if path.is_file(): path.unlink()



def resolve_result_delivery(command, *, output, problem, report=None):
    """Resolve and validate an analysis/check result before any output write."""
    checker = command in {'check-result', 'check-model-result'}
    protected = {Path(problem).resolve()}
    if checker:
        protected.add(Path(report).resolve())
        destination = (Path(output) if output else
                       Path(report).parent / 'verification' / 'check.json').resolve()
        run_output = (destination.parent / 'verification-run').resolve()
    else:
        run_output = Path(output).resolve()
        destination = (run_output / 'report.json').resolve()
    temporary = destination.with_name(destination.name + '.writing').resolve()
    # Retain the parent's original protections for aliases of reserved exports,
    # in addition to the complete run directory and final temporary result.
    reserved = {destination, temporary, (run_output / 'partial-frontier.json').resolve()}
    if not checker:
        reserved.update((run_output / name).resolve() for name in (
            'original.json', 'analysis.json', 'normalized.json', 'semantic-check.json',
            'typed-graph.json', 'candidate-boundaries.json',
            'recommended-boundaries.json', 'proposal-decisions.json', 'report-data.json'))
    if reserved & protected or any(
            path.is_relative_to(run_output) for path in protected):
        raise ValueError('The output would overwrite an original input or the report being checked; choose a separate output path')
    startup = {run_output/'partial-frontier.json', run_output/'report.json', destination}
    if not checker:
        startup.update(reserved)
        startup.update(run_output/name for name in (
            'metadata-progress.json', 'family-boundary.json', 'report-files.json', 'report-manifest.json',
            'delivery-timings.json', 'report.md', 'report.html', 'summary.json',
            'next-actions.md', 'next-actions.html'))
    startup |= {path.with_name(path.name+'.writing') for path in startup}
    removals = tuple(Mutation('parent.remove.'+str(index), 'unlink', path)
                     for index,path in enumerate(sorted(startup)))
    mutations = (*removals, Mutation('parent.mkdir','mkdir',run_output),
                 Mutation('result.mkdir','mkdir',destination.parent),
                 Mutation('result.open','open_w',temporary),
                 Mutation('result.replace_source','replace',temporary),
                 Mutation('result.replace_target','replace',destination),
                 *resource_mutations(run_output))
    # Preserve both caller-supplied names and target/inode identities.
    plan = OwnedMutationPlan.admit(inputs=[problem, *([report] if checker else [])],
                                   owned_roots=[run_output], mutations=mutations)
    return ResultDelivery(destination, run_output, checker, plan,
                          tuple(mutation.role for mutation in removals))



def invalidate_delivery(output, *, status, reason, details=None):
    """Publish one fail-closed result for all normal result-reading routes.

    The caller has already validated the output namespace. This runs only after a
    failed boundary, never on a normal complete or conditional analysis. Files
    explicitly named completed-before-* are diagnostic worker evidence, not
    public authorization exports.
    """
    if status not in {'INVALID', 'INCOMPLETE', 'FAILED'}:
        raise ValueError('Delivery invalidation requires a failure status')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in DECISION_EXPORTS:
        (output / name).unlink(missing_ok=True)
        (output / (name + '.writing')).unlink(missing_ok=True)
    record = dict(details or {})
    # No result copied from a failed worker can smuggle an old authorization.
    for key in ('recommended', 'certificate', 'decisions', 'proposal_decisions',
                'analysis', 'independent_certificate_check', 'generated_transport'):
        record.pop(key, None)
    record.update(schema='regime-production-result/v1', status=status,
                  reason=str(reason), recommendation=None, scenarios=[],
                  transformation_authorized=False)
    message = ('# Metadata analysis ' + status.lower() + '\n\n' + str(reason) +
               '\n\nNo optimum or transformation is authorized by this delivery. '
               'Original inputs and diagnostic checkpoints are retained.\n')
    rendered = ('<!doctype html><meta charset="utf-8"><title>Metadata analysis ' +
                html.escape(status.lower()) + '</title><pre>' +
                html.escape(message) + '</pre>')
    files = {name: json.dumps(record, indent=2, sort_keys=True) + '\n'
             for name in ('report.json', 'summary.json')}
    files.update({name + '.md': message for name in ('report', 'next-actions')})
    files.update({name + '.html': rendered for name in ('report', 'next-actions')})
    for name, contents in files.items():
        path = output / name
        temporary = path.with_name(name + '.writing')
        temporary.write_text(contents, encoding='utf-8')
        temporary.replace(path)
    return record
