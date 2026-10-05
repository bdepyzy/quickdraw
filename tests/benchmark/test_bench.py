"""Check timing boundaries and exact processed-token counts without a GPU."""

import unittest

from benchmark.offline.bench import profile_workload, run_once
from quickdraw import Engine
from tests.engine.check import FakeForward, tiny_checkpoint


class BenchmarkTimingTests(unittest.TestCase):
    def test_prefill_and_decode_boundaries(self):
        ckpt = tiny_checkpoint()
        fake = FakeForward(ckpt, 8)
        engine = Engine(ckpt, max_context=8, device="cpu", step=fake)
        engine.eos_token_ids = {7}
        ticks = iter([10., 10.25, 10.75])
        timing = run_once(engine, [11, 22, 33], 3,
                          synchronize=lambda: None, clock=lambda: next(ticks))
        self.assertEqual(fake.calls, [(11, 0), (22, 1), (33, 2), (7, 3), (9, 4)])
        self.assertEqual(engine.state.position, 5)
        self.assertEqual(engine.pending_token.item(), 4)
        self.assertEqual((timing.prefill_ms, timing.decode_step_ms, timing.total_ms), (250., 250., 750.))

    def test_one_output_needs_no_decode_and_repeat_starts_fresh(self):
        ckpt = tiny_checkpoint()
        fake = FakeForward(ckpt, 3)
        engine = Engine(ckpt, max_context=3, device="cpu", step=fake)
        for _ in range(2):
            fake.calls.clear()
            timing = run_once(engine, [11, 22, 33], 1, synchronize=lambda: None)
            self.assertIsNone(timing.decode_step_ms)
            self.assertEqual(fake.calls, [(11, 0), (22, 1), (33, 2)])
            self.assertEqual(engine.state.position, 3)

    def test_capacity_validation_precedes_state_changes(self):
        engine = Engine(tiny_checkpoint(), max_context=3, device="cpu")
        for prompt, count in (([], 1), ([11], 0), ([11, 22, 33], 2)):
            with self.assertRaises(AssertionError):
                run_once(engine, prompt, count, synchronize=lambda: None)
        self.assertEqual(engine.state.position, 0)

    def test_profile_generation_includes_prompt_and_all_outputs(self):
        ckpt = tiny_checkpoint()
        fake = FakeForward(ckpt, 8)
        engine = Engine(ckpt, max_context=8, device="cpu", step=fake)
        engine.prefill([11, 22])
        fake.calls.clear()
        profile_workload(engine, [11, 22, 33], 3, "generation", synchronize=lambda: None)
        self.assertEqual(fake.calls, [(11, 0), (22, 1), (33, 2), (7, 3), (9, 4)])
        self.assertEqual(engine.pending_token.item(), 4)

    def test_profile_prefill_does_not_process_output_token(self):
        ckpt = tiny_checkpoint()
        fake = FakeForward(ckpt, 8)
        engine = Engine(ckpt, max_context=8, device="cpu", step=fake)
        profile_workload(engine, [11, 22, 33], 3, "prefill", synchronize=lambda: None)
        self.assertEqual(fake.calls, [(11, 0), (22, 1), (33, 2)])
        self.assertEqual(engine.pending_token.item(), 7)

    def test_profile_decode_only_does_not_repeat_prefill(self):
        ckpt = tiny_checkpoint()
        fake = FakeForward(ckpt, 8)
        engine = Engine(ckpt, max_context=8, device="cpu", step=fake)
        engine.pending_token = engine.prefill([11, 22, 33]).argmax()
        fake.calls.clear()
        profile_workload(engine, [11, 22, 33], 3, "decode", synchronize=lambda: None)
        self.assertEqual(fake.calls, [(7, 3)])
        self.assertEqual(engine.pending_token.item(), 9)


if __name__ == "__main__":
    unittest.main()
