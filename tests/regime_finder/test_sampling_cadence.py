"""Slow cooperative RSS observations do not consume every metadata check."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from _lcm_regime_finder.callable_identity import CaptureIncomplete, CaptureSession


class SamplingCadenceTests(unittest.TestCase):
    def test_slow_sample_waits_for_work_after_completion(self) -> None:
        clock = [0.0]

        def observe(_pid: int) -> tuple[int, int]:
            clock[0] += 0.06
            return 1024, 1

        with patch('_lcm_regime_finder.callable_identity.time.perf_counter', side_effect=lambda: clock[0]), \
                patch('_lcm_regime_finder.resources._tree_rss', side_effect=observe) as sampler:
            session = CaptureSession({'seconds': 10, 'host_gib': 8})
            session.check()
            for _ in range(10):
                session.check()
            self.assertEqual(sampler.call_count, 1)
            clock[0] += 0.051
            session.check()
            self.assertEqual(sampler.call_count, 2)
            self.assertEqual(session.tree_rss_samples, 2)

    def test_time_spent_observing_is_charged_to_the_same_deadline(self) -> None:
        clock = [0.0]

        def observe(_pid: int) -> tuple[int, int]:
            clock[0] += 0.06
            return 1024, 1

        with patch('_lcm_regime_finder.callable_identity.time.perf_counter', side_effect=lambda: clock[0]), \
                patch('_lcm_regime_finder.resources._tree_rss', side_effect=observe):
            session = CaptureSession({'seconds': 0.05, 'host_gib': 8})
            with self.assertRaisesRegex(CaptureIncomplete, 'elapsed_time'):
                session.check()
            self.assertEqual(session.exhaustion_reason, 'elapsed_time')

    def test_elapsed_deadline_fires_even_when_another_sample_is_not_due(self) -> None:
        clock = [0.0]
        with patch('_lcm_regime_finder.callable_identity.time.perf_counter', side_effect=lambda: clock[0]), \
                patch('_lcm_regime_finder.resources._tree_rss', return_value=(1024, 1)) as sampler:
            session = CaptureSession({'seconds': 0.02, 'host_gib': 8})
            session.check()
            clock[0] = 0.03
            with self.assertRaisesRegex(CaptureIncomplete, 'elapsed_time'):
                session.check()
            self.assertEqual(sampler.call_count, 1)

    def test_forced_observation_still_checks_memory_immediately(self) -> None:
        with patch('_lcm_regime_finder.callable_identity.time.perf_counter', return_value=0), \
                patch('_lcm_regime_finder.resources._tree_rss', side_effect=[(1024, 1), (9 * 1024**3, 1)]) as sampler:
            session = CaptureSession({'seconds': 10, 'host_gib': 8})
            session.check()
            with self.assertRaisesRegex(CaptureIncomplete, 'process_tree_memory'):
                session.check(force=True)
            self.assertEqual(sampler.call_count, 2)
            self.assertEqual(session.exhaustion_reason, 'process_tree_memory')

    def test_unavailable_slow_sample_uses_cadence_and_self_memory_guard(self) -> None:
        clock = [0.0]

        def observe(_pid: int) -> tuple[int, int]:
            clock[0] += 0.06
            return 0, 1

        with patch('_lcm_regime_finder.callable_identity.time.perf_counter', side_effect=lambda: clock[0]), \
                patch('_lcm_regime_finder.resources._tree_rss', side_effect=observe) as sampler, \
                patch('_lcm_regime_finder.callable_identity.resource.getrusage', return_value=SimpleNamespace(ru_maxrss=1024)):
            session = CaptureSession({'seconds': 10, 'host_gib': 8})
            session.check()
            session.check()
            self.assertEqual(sampler.call_count, 1)
            self.assertEqual(session.tree_rss_unavailable_samples, 1)
            with patch('_lcm_regime_finder.callable_identity.resource.getrusage', return_value=SimpleNamespace(ru_maxrss=9 * 1024**2)):
                with self.assertRaisesRegex(CaptureIncomplete, 'process_tree_memory'):
                    session.check(force=True)


if __name__ == '__main__':
    unittest.main()
