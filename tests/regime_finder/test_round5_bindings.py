"""Source-backed binding regressions and independent Python call oracles.

No Model, solve, simulation, or Bellman evaluation occurs. The independent
oracle invokes only the explicit tiny scalar fixtures below. Native numerical
integration still requires the pinned production environment.
"""
from dataclasses import dataclass
import functools
import inspect
import unittest

from _lcm_regime_finder.declarations import DeclarationError, ScalarSource, normalize_declarations
from _lcm_regime_finder.expressions import partial as evaluate_expression
from _lcm_regime_finder.model_snapshot import read_declarations, source_function
from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder.semantic_check import SourceProof
from tests.regime_finder.test_model_adapter import snapshot


_OPTIONAL_BINDING = None


def optional_global(history):
    return history if _OPTIONAL_BINDING is None else 0


def optional_factory(marker):
    def callback(history):
        return history if marker is None else marker
    return callback


def identity(history):
    return history


def difference(left, right):
    return left - right


def triple(first, second, third=3):
    return first + second - third


def default_helper(value=9):
    return value


def call_default(history):
    del history
    return default_helper()


def positional_callback(helper):
    def callback(history):
        return helper(history)
    return callback


def named_callback(helper):
    def callback(history):
        return helper(history=history)
    return callback


def keyword_override(helper, replacement):
    def callback(history):
        return helper(left=history, right=replacement)
    return callback


def keyword_default(helper):
    def callback(history):
        return helper(left=history)
    return callback


def keyword_only(*, history=7):
    return history


def variadic(**kwargs):
    return kwargs['history']


@dataclass
class Receiver:
    optional: int | None

    def callback(self, history):
        return history if self.optional is None else 0


def direct_oracle(callback, **arguments):
    """Python's actual invocation semantics; no production AST is reused."""
    bound = inspect.signature(callback).bind(**arguments)
    return callback(*bound.args, **bound.kwargs)


class NoneBindingCaptureTests(unittest.TestCase):
    def test_none_global_is_a_value_not_an_absent_bound_receiver(self):
        captured = read_declarations({'callback': optional_global})['callback']
        self.assertIsNone(captured['bindings']['_OPTIONAL_BINDING'])
        self.assertNotIn('bound_self', captured)
        self.assertFalse(captured['source_unavailable'] if 'source_unavailable' in captured else False)

    def test_none_closure_and_other_scalar_bindings(self):
        identities = set()
        for marker in (None, False, 0, 1, -1, 'marker'):
            with self.subTest(marker=marker):
                captured = source_function(optional_factory(marker))
                self.assertEqual(captured['bindings']['marker'], marker)
                self.assertNotIn('bound_self', captured)
                identities.add(captured['binding_sha256'])
        self.assertEqual(len(identities), 6)

    def test_bound_receiver_containing_none_remains_a_receiver(self):
        captured = source_function(Receiver(None).callback)
        self.assertIsNone(captured['bound_self']['fields']['optional'])
        self.assertEqual(captured['bindings']['self'], captured['bound_self'])


class PartialApplicationTests(unittest.TestCase):
    def check_scalar(self, callback, history):
        captured = source_function(callback)
        expected = direct_oracle(callback, history=history)
        expression = ScalarSource(captured).expression()
        translated = evaluate_expression(expression, {'history': history})
        independently_checked = SourceProof({}, static={'history': history}).callback(captured)
        self.assertTrue(translated.known)
        self.assertTrue(independently_checked.proved)
        self.assertEqual(translated.value, expected)
        self.assertEqual(independently_checked.value, expected)
        return captured

    def test_exposed_partial_keyword_is_not_frozen(self):
        for frozen in (-3, 0, 2):
            callback = functools.partial(identity, history=frozen)
            captured = source_function(callback)
            self.assertEqual(captured['parameter_kinds']['history'], 'KEYWORD_ONLY')
            self.assertEqual(ScalarSource(captured).expression(), 'history')
            self.assertEqual(SourceProof({}).callback(captured).reads, frozenset({'history'}))
            for history in (-4, -1, 0, 1, 5):
                with self.subTest(frozen=frozen, history=history):
                    self.check_scalar(callback, history)

    def test_public_normalization_retains_overridable_current_input(self):
        original = snapshot()
        original['user_regimes']['original']['fields']['functions']['utility'] = source_function(
            functools.partial(identity, history=0))
        normalized = normalize_declarations(original)
        for cell in normalized['cells']:
            callback = normalized['callbacks'][cell['functions']['utility']]
            self.assertEqual(callback['reads'], ['history'])
            self.assertEqual(callback['expression'], 'history')

    def test_public_production_cannot_adopt_deletion_of_overridable_input(self):
        original = snapshot()
        original['user_regimes']['original']['fields']['functions']['utility'] = source_function(
            functools.partial(identity, history=0))
        original['semantic_contract'] = {
            'roles': {'original': {'actor': 'one', 'information': ['history', 'x'], 'commitments': []}},
            'stages': {'original': 'decision'},
            'allocation': {'kind': 'per_block', 'status': 'known', 'identity': 'toy-declared-allocation'},
            'production_metadata': {'status': 'complete', 'scope': 'explicit tiny declaration witness'},
        }
        result = analyze_production(original, limits={'seconds': 10}, search_mode="exact")
        self.assertEqual(result['semantic_check']['status'], 'PASS')
        self.assertTrue(all('history' in axes for axes in result['analysis']['dependency_closure'].values()))

    def test_positional_partial_helper_uses_effective_signature(self):
        for fixed in (-3, 0, 10):
            callback = positional_callback(functools.partial(difference, fixed))
            for history in (-2, 0, 7):
                with self.subTest(fixed=fixed, history=history):
                    self.check_scalar(callback, history)

    def test_partial_keyword_default_and_explicit_override(self):
        for default in (-2, 0, 4):
            helper = functools.partial(difference, right=default)
            callbacks = [keyword_default(helper)]
            callbacks += [keyword_override(helper, override) for override in (-1, 0, 5)]
            for callback in callbacks:
                for history in (-3, 0, 6):
                    with self.subTest(default=default, history=history, callback=callback):
                        self.check_scalar(callback, history)

    def test_mixed_nested_partials_keep_positional_and_default_roles(self):
        for first in (-2, 0, 5):
            for third in (-3, 2):
                helper = functools.partial(functools.partial(triple, first), third=third)
                callback = positional_callback(helper)
                for history in (-1, 0, 4):
                    with self.subTest(first=first, third=third, history=history):
                        self.check_scalar(callback, history)

    def test_empty_helper_call_uses_the_captured_effective_default(self):
        self.check_scalar(call_default, history=2)

    def test_keyword_only_binding_works_by_name(self):
        self.check_scalar(named_callback(keyword_only), history=11)

    def test_wrong_positional_binding_is_not_certified(self):
        captured = source_function(positional_callback(keyword_only))
        with self.assertRaises(DeclarationError):
            ScalarSource(captured).expression()
        checked = SourceProof({}, static={'history': 1}).callback(captured)
        self.assertFalse(checked.proved)

    def test_variadic_input_is_still_an_obligation(self):
        captured = source_function(variadic)
        self.assertTrue(captured['variadic'])
        checked = SourceProof({}, static={'history': 1}).callback(captured)
        self.assertFalse(checked.proved)


if __name__ == '__main__':
    unittest.main()
