"""Check the comparison's denominators and traffic estimate without loading a model."""

import json
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from benchmark.offline.compare import bandwidth_estimate, gaps, main, measure, saved_baselines, summarize, traffic_model


class ComparisonTests(unittest.TestCase):
    def test_baseline_selection_skips_mismatch_and_partial_samples(self):
        manifest = dict(checkpoint={'hash': 'same'}, gpu={'uuid': 'same'}, prompts=[[1, 2]],
                        conditions=dict(input_lens=[2], runs=5, quickdraw_backend='triton'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, runs, checkpoint in (('compare-1', 5, 'same'), ('compare-2', 4, 'same'), ('compare-3', 5, 'different')):
                folder = root / name
                folder.mkdir()
                saved = dict(manifest, checkpoint={'hash': checkpoint},
                             conditions=dict(manifest['conditions'], quickdraw_backend='reference'))
                (folder / 'manifest.json').write_text(json.dumps(saved))
                for engine in ('vllm', 'sglang'):
                    (folder / f'{engine}.json').write_text(json.dumps(dict(workloads=[dict(input_len=2, samples=[{}]*runs)])))
            self.assertEqual(saved_baselines(manifest, runs_dir=root), root / 'compare-1')
            with self.assertRaisesRegex(AssertionError, 'No complete saved baselines'):
                saved_baselines(manifest, root / 'compare-2')
            with self.assertRaisesRegex(AssertionError, 'No complete saved baselines'):
                saved_baselines(dict(manifest, gpu={'uuid': 'changed'}), runs_dir=root)

    def test_baseline_engine_launch_requires_explicit_refresh(self):
        with patch('sys.argv', ['compare.py', '--engines', 'vllm']), patch('benchmark.offline.compare.subprocess.run') as launch:
            with self.assertRaisesRegex(AssertionError, '--refresh-baselines'):
                main()
            launch.assert_not_called()

    def test_default_launches_only_quickdraw_and_copies_saved_medians(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text(json.dumps(dict(text_config=dict(vocab_size=40))))
            for name in ('hf_quant_config.json', 'model.safetensors.index.json'):
                (model / name).write_text('{}')
            python = root / '.venv/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            target = root / 'output'
            baseline = root / 'saved'
            payload = dict(workloads=[dict(input_len=128, samples=[dict(output_ids=[7]*16)]*5,
                                           median=dict(ttft_ms=10, decode_ms_per_token=2, decode_tps=500, total_ms=40))])

            def select(manifest, source):
                baseline.mkdir()
                (baseline / 'manifest.json').write_text(json.dumps(manifest))
                for engine in ('vllm', 'sglang'):
                    (baseline / f'{engine}.json').write_text(json.dumps(dict(payload, engine=engine)))
                return baseline

            def launch(command, **kwargs):
                self.assertEqual(command[command.index('--worker') + 1], 'quickdraw')
                self.assertNotIn('--refresh-baselines', command)
                (target / 'quickdraw.json').write_text(json.dumps(dict(payload, engine='quickdraw')))
                return SimpleNamespace(returncode=0)

            with patch('sys.argv', ['compare.py', '--model', str(model), '--save-dir', str(target)]), \
                 patch('benchmark.offline.compare.ROOT', root), \
                 patch('benchmark.offline.compare.saved_baselines', side_effect=select), \
                 patch('benchmark.offline.compare.traffic_model', return_value=dict(active_weight_bytes=1000, recurrent_conv_rw_bytes=200, kv_bytes_per_position=10)), \
                 patch('benchmark.offline.compare.subprocess.check_output', return_value='RTX 5090, UUID, DRIVER, 32000, 575, 14001\n'), \
                 patch('benchmark.offline.compare.subprocess.run', side_effect=launch) as run, \
                 patch('sys.stdout', new_callable=io.StringIO):
                self.assertFalse(main())
                self.assertEqual(run.call_count, 1)
            for engine in ('vllm', 'sglang'):
                self.assertEqual(json.loads((target / f'{engine}.json').read_text())['workloads'][0]['median'], payload['workloads'][0]['median'])

    def test_delivery_timing_excludes_first_token_from_decode_and_ignores_duplicate_finish(self):
        ticks = iter([10., 10.25, 10.75])
        result = measure(lambda prompt: iter([[], [7], [7, 9], [7, 9, 4], [7, 9, 4]]),
                         [11, 22, 33], 3, clock=lambda: next(ticks))
        self.assertEqual(result, dict(ttft_ms=250, decode_ms_per_token=250, decode_tps=4,
                                      total_ms=750, output_ids=[7, 9, 4]))

    def test_missing_or_coalesced_first_output_cannot_claim_ttft(self):
        for chunks in ([], [[7]], [[7, 9, 4]]):
            with self.assertRaises(AssertionError):
                measure(lambda prompt: iter(chunks), [11], 3)

    def test_medians_resist_one_slow_run(self):
        samples = [dict(ttft_ms=x, decode_ms_per_token=2*x, decode_tps=500/x, total_ms=5*x)
                   for x in (1, 2, 100, 3, 4)]
        result = summarize(samples)
        self.assertEqual(result, dict(ttft_ms=3, decode_ms_per_token=6, decode_tps=500/3, total_ms=15))

    def test_latency_and_throughput_gaps_have_different_denominators(self):
        baseline = dict(ttft_ms=10, decode_ms_per_token=2, total_ms=50, decode_tps=500)
        actual = dict(ttft_ms=20, decode_ms_per_token=8, total_ms=25, decode_tps=125)
        self.assertEqual(gaps(actual, baseline), dict(ttft_extra_pct=100, decode_extra_pct=300,
                                                    total_extra_pct=-50, decode_tps_deficit_pct=75))
        self.assertEqual(gaps(baseline, baseline), dict(ttft_extra_pct=0, decode_extra_pct=0,
                                                       total_extra_pct=0, decode_tps_deficit_pct=0))

    def test_bandwidth_uses_mean_history_and_decimal_gigabytes(self):
        # P=3, N=3: the two decode steps read 4 and 5 KV positions and write one.
        traffic = dict(active_weight_bytes=1000, recurrent_conv_rw_bytes=200, kv_bytes_per_position=10)
        result = bandwidth_estimate(traffic, 3, 3, decode_ms=1, peak_gbs=1)
        self.assertEqual(result['modeled_bytes_per_decode'], 1255)
        self.assertAlmostEqual(result['ideal_bandwidth_ms'], .001255)
        self.assertAlmostEqual(result['bandwidth_sol_estimate_pct'], .1255)

    def test_traffic_counts_selected_experts_and_skips_vision_and_embedding_table(self):
        config = dict(hidden_size=4, num_experts=4, num_experts_per_tok=2,
                      layer_types=['full_attention', 'linear_attention'],
                      linear_num_value_heads=2, linear_key_head_dim=3, linear_value_head_dim=4,
                      linear_num_key_heads=1, linear_conv_kernel_dim=3, num_key_value_heads=1, head_dim=2)
        sizes = {'model.language_model.layers.0.self_attn.q_proj.weight': 100,
                 'model.language_model.layers.0.mlp.experts.0.gate_proj.weight': 40,
                 'model.language_model.layers.0.mlp.experts.1.gate_proj.weight': 40,
                 'model.language_model.layers.0.mlp.experts.2.gate_proj.weight': 40,
                 'model.language_model.layers.0.mlp.experts.3.gate_proj.weight': 40,
                 'lm_head.weight': 100,
                 'model.language_model.embed_tokens.weight': 10000,
                 'model.visual.weight': 20000,
                 'model.language_model.layers.0.mlp.experts.0.gate_proj.input_scale': 4}
        header = {name: dict(data_offsets=[0, size]) for name, size in sizes.items()}
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / 'config.json').write_text(json.dumps(dict(text_config=config)))
            encoded = json.dumps(header).encode()
            (model / 'model-00001.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded)
            result = traffic_model(model)
        self.assertEqual(result['active_weight_bytes'], 288)
        self.assertEqual(result['kv_bytes_per_position'], 8)
        self.assertEqual(result['recurrent_conv_rw_bytes'], 2 * (2*3*4*4 + (2*1*3 + 2*4)*2*2))


if __name__ == '__main__':
    unittest.main()
