"""Lossless typed incidence view; NetworkX is an optional consumer."""
from __future__ import annotations
from copy import deepcopy


def boundary_maps(analysis, plan):
    """Instantiate explicit source/action and target/state maps for one plan.

    The full original law is transported jointly. The original source callback
    is an identity anchor; this object never asserts that a coarse coordinate
    can reconstruct a unique original state.
    """
    chosen = {cell: interface for block in plan['blocks'] for cell, interface in block['option_by_cell'].items()}
    options = {cell: next(o for o in value['options'] if o['id'] == chosen[cell])
               for cell, value in analysis['interfaces'].items()}
    result = []
    for relation in analysis['declarations']['relations']:
        source = options[relation['source']]
        result.append({'original_relation': relation['id'], 'kind': relation['kind'],
                       'phase': relation.get('phase', 'both'), 'ownership': deepcopy(relation.get('ownership')),
                       'source_cell': relation['source'], 'source_interface': source['id'],
                       'source_action_reconstruction': deepcopy(source['action_reconstruction']),
                       'original_callbacks': deepcopy(relation.get('callbacks', [])),
                       'original_joint_outputs': deepcopy(relation.get('joint_outputs', {})),
                       'original_state_updates': deepcopy(relation.get('state_updates', {})),
                       'query_target': relation.get('query_target'),
                       'targets': [{'cell': target, 'interface': options[target]['id'],
                                    'state_projection': deepcopy(options[target]['state_projection']),
                                    'bound_constants': deepcopy(options[target]['constants'])}
                                   for target in relation.get('targets', [])],
                       'transport': 'push forward the full original joint law through these target maps; factor source consumers through the checked reduced DAG',
                       'native_execution': 'NOT_RUN'})
    return result


def typed_incidence_graph(declarations):
    """Keep each full relation as a node, including joint laws and owned routes.

    A relation with several targets is not replaced by independent marginal
    probabilities. Incidence edges only encode membership and orientation.
    """
    nodes = [{'id': 'cell:' + c['id'], 'node_kind': 'decision', 'declaration': deepcopy(c)}
             for c in declarations['cells']]
    nodes += [{'id': 'query:' + q['id'], 'node_kind': 'policy_query_owner', 'declaration': deepcopy(q)}
              for q in declarations.get('query_owners', {}).values()]
    edges = []
    for relation in declarations['relations']:
        rid = 'relation:' + relation['id']
        nodes.append({'id': rid, 'node_kind': relation['kind'], 'relation': deepcopy(relation)})
        edges.append({'source': 'cell:' + relation['source'], 'target': rid,
                      'key': relation['id'] + ':source', 'incidence': 'source'})
        if relation.get('query_target') is not None:
            edges.append({'source': rid, 'target': 'query:' + relation['query_target'],
                          'key': relation['id'] + ':query', 'incidence': 'policy_query'})
        for ordinal, target in enumerate(relation.get('targets', [])):
            edges.append({'source': rid, 'target': 'cell:' + target,
                          'key': relation['id'] + ':target:' + str(ordinal),
                          'incidence': 'target', 'target_ordinal': ordinal})
    # Nested demand is carried in owner records rather than physical relation
    # records. Export it as typed incidence too, so graph traversal sees the
    # same closure as a reader of the payload. Keep parallel owned reads and
    # unknown targets distinct; no edge here asserts a physical transition.
    for owner in declarations.get('query_owners', {}).values():
        for ordinal, dependency in enumerate(owner.get('dependencies', [])):
            rid = 'query-dependency:' + owner['id'] + ':' + str(ordinal)
            nodes.append({'id': rid, 'node_kind': 'policy_query_dependency',
                          'query_owner': owner['id'], 'phase': owner['phase'],
                          'period': owner['period'], 'dependency': deepcopy(dependency)})
            edges.append({'source': 'query:' + owner['id'], 'target': rid,
                          'key': rid + ':source', 'incidence': 'query_source'})
            if 'query_owner' in dependency:
                target = 'query:' + dependency['query_owner']
            elif 'cell' in dependency:
                target = 'cell:' + dependency['cell']
            else:
                target = 'unavailable-query:' + owner['id'] + ':' + str(ordinal)
                nodes.append({'id': target, 'node_kind': 'unavailable_policy_query_owner',
                              'regime': dependency.get('unavailable_owner'),
                              'period': owner['period'], 'phase': owner['phase']})
            edges.append({'source': rid, 'target': target,
                          'key': rid + ':read', 'incidence': 'query_read'})
    return {'schema': 'regime-typed-incidence-graph/v1', 'directed': True, 'multigraph': True,
            'source_identity': deepcopy(declarations.get('source_identity')),
            'nodes': nodes, 'edges': edges,
            'domains': deepcopy(declarations['domains']),
            'callbacks': deepcopy(declarations['callbacks']),
            'scope': 'lossless view of the typed declaration records; incidence edges are not transition probability marginals'}


def to_networkx(declarations):
    """Return MultiDiGraph when NetworkX is installed by the caller.

    NetworkX is not needed by source analysis, cost search or independent checks.
    No package installation is attempted by this function.
    """
    import networkx as nx
    view = typed_incidence_graph(declarations)
    graph = nx.MultiDiGraph(schema=view['schema'], source_identity=view['source_identity'],
                            domains=view['domains'], callbacks=view['callbacks'])
    for node in view['nodes']:
        graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
    for edge in view['edges']:
        graph.add_edge(edge['source'], edge['target'], key=edge['key'],
                       **{k: v for k, v in edge.items() if k not in ('source', 'target', 'key')})
    return graph
