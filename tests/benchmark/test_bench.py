"""Check timing boundaries and exact processed-token counts without a GPU."""

import json
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

from benchmark.offline.bench import Timing, parse_args, profile_workload, run_once, speedup_report
from benchmark.offline.compare import DEFAULT_WORKLOAD, QUICK_WORKLOAD
from quickdraw import Engine
from tests.engine.check import FakeForward, tiny_checkpoint


class BenchmarkTimingTests(unittest.TestCase):
    def test_default_workload_uses_mini_sglang_token_limits_and_profiles_both_phases(self):
        for argv in ([], ["--profile"], ["--profile", "nsys"], ["--profile", "ncu"]):
            args = parse_args(argv)
            self.assertEqual({name: getattr(args, name) for name in DEFAULT_WORKLOAD}, DEFAULT_WORKLOAD)
            self.assertEqual(args.profile_phase, "both")
            self.assertEqual((args.input_lens, args.output_len, args.runs), ([1024], 1024, 5))
        self.assertIsNone(parse_args([]).profile)
        self.assertEqual(parse_args(["--profile"]).profile, "torch")
        args = parse_args(["--output-len", "32", "--runs", "2", "--profile-phase", "decode"])
        self.assertEqual((args.output_len, args.runs, args.profile_phase), (32, 2, "decode"))

    def test_quick_mode_preserves_saved_workload_and_explicit_options_override_it(self):
        for argv in (["--quick"], ["--quick", "--profile"]):
            args = parse_args(argv)
            self.assertEqual({name: getattr(args, name) for name in QUICK_WORKLOAD}, QUICK_WORKLOAD)
        args = parse_args(["--quick", "--input-lens", "512", "--output-len", "128"])
        self.assertEqual((args.input_lens, args.output_len), ([512], 128))

    def test_cute_decode_preserves_workload_and_requires_graph_backend(self):
        for argv in (["--cute-decode"], ["--cute-decode", "--profile", "ncu"]):
            args = parse_args(argv)
            self.assertTrue(args.cute_decode)
            self.assertEqual({name: getattr(args, name) for name in DEFAULT_WORKLOAD}, DEFAULT_WORKLOAD)
        for argv in (["--cute-decode", "--eager"], ["--cute-decode", "--backend", "reference"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                parse_args(argv)
            self.assertEqual(stopped.exception.code, 2)

    def test_short_help_and_full_help(self):
        for flag, advanced in (("--help", False), ("--help-all", True)):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                parse_args([flag])
            self.assertEqual(stopped.exception.code, 0)
            self.assertIn("--profile", output.getvalue())
            self.assertEqual("--profile-phase" in output.getvalue(), advanced)

    def test_speedup_uses_throughput_ratio_and_handles_missing_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = dict(conditions=dict(output_len=16), peak_bandwidth_gbs=1,
                            traffic=dict(active_weight_bytes=1000000, recurrent_conv_rw_bytes=0,
                                         kv_bytes_per_position=0))
            (folder / "manifest.json").write_text(json.dumps(manifest))
            payload = dict(workloads=[dict(input_len=128, median=dict(
                ttft_ms=100, decode_ms_per_token=2, total_ms=150))])
            for name in ("vllm", "sglang"):
                (folder / f"{name}.json").write_text(json.dumps(payload))
            slower = speedup_report({128: Timing(200, 4, 300)}, folder)
            faster = speedup_report({128: Timing(50, 1, 75)}, folder)
            self.assertEqual(slower.count("0.50x"), 6)
            self.assertEqual(faster.count("2.00x"), 6)
            self.assertIn("prefill -100.000 ms | decode -2.000 ms/token | total -150.000 ms", slower)
            self.assertIn("prefill +50.000 ms | decode +1.000 ms/token | total +75.000 ms", faster)
            equal = speedup_report({128: Timing(100, 2, 150)}, folder)
            self.assertEqual(equal.count("1.00x"), 7)
            self.assertIn("0.250", slower)
            self.assertIn("0.500", equal)
            self.assertNotIn("%", slower)
            self.assertIn("vllm", slower)
            self.assertIn("sglang", slower)
            single = speedup_report({128: Timing(100, None, 100)}, folder)
            self.assertEqual(single.count("n/a"), 7)
            custom = speedup_report({128: Timing(100, 4, 150)}, None, manifest=manifest)
            self.assertIn("quickdraw", custom)
            self.assertIn("0.250", custom)
            self.assertIn("unavailable for these settings", custom)
        self.assertIn("unavailable", speedup_report({}, None))

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
