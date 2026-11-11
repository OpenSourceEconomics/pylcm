"""Independent tiny partition, signature-assignment and binary-model oracles."""
from copy import deepcopy
from itertools import product
import unittest
from _lcm_regime_finder.signature_proof import (
    binary_model, literal_partition_cost, pareto, union_closure, signature,
)
from tests.regime_finder._signature_oracle import (
    partition_frontier, signature_assignment_frontier, verify_binary_assignment,
)



def fixture(layouts=(('x',), ('y',), ('x',)), pairs=()):
    return {'schema': 'regime-singleton-layout-instance/v1', 'allocation': 'per_block',
            'cells': [{'id': str(i), 'label': f'label-{i}', 'context': 'same-protected-context',
                       'period': i, 'states': {axis: axis for axis in axes},
                       'actions': {'a': 'a'}, 'characteristics': []} for i, axes in enumerate(layouts)],
            'domains': {'x': {'size': 2}, 'y': {'size': 3}, 'z': {'size': 5}, 'a': {'size': 2}},
            'pairs': [list(pair) for pair in pairs], 'boundary_unit': 1,
            'clarity_weights': {'block': 4, 'dispatch': 1, 'characteristic': 2}}


def witness(instance, model, bins_by_cell):
    owner = dict(zip(model['cell_order'], bins_by_cell))
    pairs = sorted(set(map(tuple, instance['pairs'])))
    result = {}
    for variable in model['variables']:
        parts = variable.split('_')
        if parts[0] == 'x':
            result[variable] = int(bins_by_cell[int(parts[1])] == int(parts[2]))
        elif parts[0] == 'y':
            result[variable] = int(int(parts[1]) in bins_by_cell)
        elif parts[0] == 'z':
            source, target = pairs[int(parts[1])]
            k = int(parts[2])
            result[variable] = int(owner[source] == k and owner.get(target) != k)
    return result



class ExactSignatureReferenceTests(unittest.TestCase):
    def test_all_three_vertex_directed_graphs(self):
        arcs = list(product(('0', '1', '2'), repeat=2))
        for mask in range(1 << len(arcs)):
            pairs = [arc for bit, arc in enumerate(arcs) if mask & (1 << bit)]
            instance = fixture(pairs=pairs)
            self.assertEqual(partition_frontier(instance), signature_assignment_frontier(instance), mask)

    def test_edges_outside_component_duplicate_edges_self_edges(self):
        instance = fixture(pairs=(('0', 'outside'), ('0', 'outside'), ('1', '1'), ('2', '0')))
        self.assertEqual(partition_frontier(instance), signature_assignment_frontier(instance))
        deduplicated = deepcopy(instance)
        deduplicated['pairs'] = [list(p) for p in set(map(tuple, instance['pairs']))]
        self.assertEqual(partition_frontier(instance), partition_frontier(deduplicated))

    def test_axis_characteristic_domain_and_scale_mutations(self):
        for boundary, domain_size, char_case, reversed_cells in product((0, 1, 3), (1, 2, 11), (False, True), (False, True)):
            instance = fixture(pairs=(('0', '1'), ('1', '2'), ('2', 'outside')))
            instance['boundary_unit'] = boundary
            instance['domains']['x']['size'] = domain_size
            if char_case:
                instance['cells'][0]['characteristics'] = ['h0']
                instance['cells'][1]['characteristics'] = ['h1']
                instance['cells'][2]['characteristics'] = ['h0']
            if reversed_cells:
                instance['cells'].reverse()
            self.assertEqual(partition_frontier(instance), signature_assignment_frontier(instance))
        collision = fixture()
        collision['domains']['other-x'] = {'size': 2}
        collision['cells'][2]['states']['x'] = 'other-x'
        self.assertEqual(partition_frontier(collision), signature_assignment_frontier(collision))

    def test_binary_costs_equal_literal_partition_costs(self):
        instance = fixture(pairs=(('0', '1'), ('1', '2'), ('2', 'outside')))
        model = binary_model(instance)
        accepted = []
        for labels in product(range(len(model['bins'])), repeat=3):
            assignment = witness(instance, model, labels)
            try:
                vector = verify_binary_assignment(model, assignment)
            except ValueError:
                continue
            blocks = [[cell for cell, label in zip(model['cell_order'], labels) if label == k]
                      for k in sorted(set(labels))]
            self.assertEqual(vector, literal_partition_cost(instance, blocks))
            accepted.append(vector)
        self.assertEqual(pareto(accepted), partition_frontier(instance))

    def test_invalid_assignment_activation_and_crossing_bits(self):
        for boundary in (0, 1):
            instance = fixture(layouts=(('x',), ('x',)), pairs=(('0', '1'),))
            instance['boundary_unit'] = boundary
            model = binary_model(instance)
            good = witness(instance, model, (0, 0))
            verify_binary_assignment(model, good)
            for variable in ('x_0_0', 'y_0', 'z_0_0'):
                bad = dict(good)
                bad[variable] = 1 - bad[variable]
                with self.assertRaises(ValueError):
                    verify_binary_assignment(model, bad)
        # An active empty bin is forbidden independently of objective weights.
        instance = fixture(layouts=(('x',), ('y',)))
        model = binary_model(instance)
        chosen = tuple(model['bins'].index(signature(c)) for c in instance['cells'])
        good = witness(instance, model, chosen)
        verify_binary_assignment(model, good)
        absent = next(k for k in range(len(model['bins'])) if k not in chosen)
        bad = dict(good)
        bad[f'y_{absent}'] = 1
        with self.assertRaises(ValueError):
            verify_binary_assignment(model, bad)

    def test_phantom_union_tokens_fail_even_when_cost_hides_them(self):
        instance = fixture(layouts=(('x',), ('y',)))
        instance['domains']['y']['solve_axis'] = False
        instance['clarity_weights']['characteristic'] = 0
        instance['cells'][1]['characteristics'] = ['h-y']
        model = binary_model(instance)
        # Put x alone into an xy/h-y bin while y/h-y is in a separate bin.
        # y has no solve-row cost and h-y has no clarity cost, but both tokens
        # must still have a provider inside the actual union block.
        whole = next(k for k, sig in enumerate(model['bins'])
                     if len(sig[0]) == 2 and sig[2] == ('h-y',))
        other = model['bins'].index(signature(instance['cells'][1]))
        bad = witness(instance, model, (whole, other))
        with self.assertRaises(ValueError):
            verify_binary_assignment(model, bad)

    def test_external_pairs_are_charged_to_their_original_source(self):
        instance = fixture(layouts=(('x',), ('y',)))
        instance['boundary_unit'] = 2
        separate = [['0'], ['1']]
        base = literal_partition_cost(instance, separate)
        instance['pairs'] = [['0', 'outside']]
        source_x = literal_partition_cost(instance, separate)
        self.assertEqual(source_x[0] - base[0], 2 * 2 * 2)
        instance['pairs'] = [['1', 'outside']]
        source_y = literal_partition_cost(instance, separate)
        self.assertEqual(source_y[0] - base[0], 2 * 3 * 2)
        self.assertEqual(partition_frontier(instance), signature_assignment_frontier(instance))
        instance['pairs'].append(['outside', '0'])
        with self.assertRaisesRegex(ValueError, 'source-owned'):
            binary_model(instance)

    def test_action_domain_identity_and_context_mutations(self):
        instance = fixture(pairs=(('0', 'outside'), ('1', '2')))
        instance['domains']['other-a'] = {'size': 2}
        instance['cells'][1]['actions']['a'] = 'other-a'
        self.assertEqual(partition_frontier(instance), signature_assignment_frontier(instance))
        instance['cells'][0]['context'] = 'different-owner'
        with self.assertRaisesRegex(ValueError, 'Protected contexts'):
            binary_model(instance)

    def test_guard_prevents_silent_family_narrowing(self):
        for mutation in ('repeated_label', 'different_context', 'dispatch', 'allocation'):
            instance = fixture()
            if mutation == 'repeated_label':
                instance['cells'][1]['label'] = instance['cells'][0]['label']
            elif mutation == 'different_context':
                instance['cells'][1]['context'] = 'other'
            elif mutation == 'dispatch':
                instance['clarity_weights']['dispatch'] = 4
            else:
                instance['allocation'] = 'global_union'
            with self.assertRaises(ValueError):
                binary_model(instance)

    def test_signature_budget_is_incomplete(self):
        with self.assertRaisesRegex(RuntimeError, 'INCOMPLETE'):
            union_closure(fixture(), max_signatures=2)



if __name__ == '__main__':
    unittest.main()
