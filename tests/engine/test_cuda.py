"""GPU checks for graph replay and chunked prompt history."""

import unittest

import torch

from quickdraw import Engine
from tests.models.mathcheck import pack, unpack
from tests.models.referencegen import random_checkpoint, small_config


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_num_threads(1)

    def history(self, actual, expected, *, exact=False):
        self.assertEqual(actual.state.position, expected.state.position)
        for name in ("kv_k", "kv_v", "rec_state", "conv_state"):
            a, b = getattr(actual.state, name), getattr(expected.state, name)
            if name.startswith("kv"):
                a, b = a[:, :actual.state.position], b[:, :expected.state.position]
            torch.testing.assert_close(a, b, rtol=0 if exact else .025, atol=0 if exact else .003)

    @torch.inference_mode()
    def test_cute_decode_retains_prefill_and_matches_eager_history(self):
        from functools import partial
        from benchmark.offline.bench import enable_cute_decode
        from quickdraw.kernel.cute.matvec import nvfp4_gemv
        from quickdraw.kernel.cute.moe import moe_block

        torch.manual_seed(950)
        ckpt = unpack(pack(random_checkpoint(small_config())), torch.device("cuda"))
        original = Engine(ckpt, max_context=16, cuda_graph=False)
        ported = Engine(ckpt, max_context=16)
        original_prefill = original.step
        eager_decode = partial(original_prefill.func, **(original_prefill.keywords | {
            "experts": moe_block, "project_fp4": nvfp4_gemv}))
        prefill_step = ported.step
        enable_cute_decode(ported)
        self.assertIs(ported.step, prefill_step)
        for _ in range(2):
            original.step = original_prefill
            a, b = original.prefill([3, 7, 5]), ported.prefill([3, 7, 5])
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            # Validate graph replay against the same CuTe math in eager mode;
            # checkpoint parity against other reductions is a separate check.
            original.step = eager_decode
            for token in (4, 6, 2):
                original.pending_token = ported.pending_token = torch.tensor(token, device="cuda")
                original.decode_step()
                ported.decode_step()
                self.history(ported, original)

    @torch.inference_mode()
    def test_kernel_override_retains_graphs_and_chunked_prefill(self):
        from quickdraw.kernel.triton.norm import rms_norm

        torch.manual_seed(940)
        ckpt = unpack(pack(random_checkpoint(small_config())), torch.device("cuda"))
        calls = []

        def replacement(x, weight, eps):
            calls.append(x.shape)
            return rms_norm(x, weight, eps)

        original = Engine(ckpt, max_context=16)
        ported = Engine(ckpt, max_context=16, backend="triton", kernels={"rms_norm": replacement})
        self.assertIsNotNone(ported.graph)
        self.assertTrue(ported.sequence_prefill)
        a, b = original.prefill([3, 7, 5]), ported.prefill([3, 7, 5])
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertIn(torch.Size([3, 32]), calls)
        original.pending_token = ported.pending_token = a.argmax()
        count = len(calls)
        for _ in range(3):
            torch.testing.assert_close(ported.decode_step(), original.decode_step(), rtol=0, atol=0)
            self.history(ported, original, exact=True)
        self.assertEqual(len(calls), count, "graph replay reentered Python kernel dispatch")

    @torch.inference_mode()
    def test_batched_projection_scales_and_tail_tiles(self):
        from quickdraw.kernel.triton.matvec import matmul
        from quickdraw.models import qwen
        from tests.models.referencegen import random_fp4, random_fp8

        torch.manual_seed(930)
        x = torch.randn(19, 80, device="cuda", dtype=torch.bfloat16)
        for proj, reference in ((random_fp8(37, 80), qwen.fp8_gemv),
                                (random_fp4(37, 80), qwen.nvfp4_gemv)):
            proj = unpack(pack(proj), torch.device("cuda"))
            proj.input_scale.fill_(.137)
            if hasattr(proj, "weight_scale_2"):
                proj.weight_scale_2.fill_(.173)
            else:
                proj.weight_scale.fill_(.173)
            expected = torch.stack([reference(row, proj) for row in x])
            torch.testing.assert_close(matmul(x, proj), expected, rtol=.016, atol=.002)

    @torch.inference_mode()
    def test_graph_replay_positions_resets_and_capacity(self):
        for variant in (0, 1):
            torch.manual_seed(910 + variant)
            ckpt = unpack(pack(random_checkpoint(small_config(variant))), torch.device("cuda"))
            ckpt.config.rope_theta = int(ckpt.config.rope_theta)
            eager = Engine(ckpt, max_context=140, cuda_graph=False, sequence_prefill=False)
            graph = Engine(ckpt, max_context=140, sequence_prefill=False)
            addresses = [getattr(graph.state, name).data_ptr() for name in ("kv_k", "rec_state", "conv_state")]
            for prompt in ([3, 7, 5], [2] * 126, [1] * 137):
                torch.testing.assert_close(graph.prefill(prompt), eager.prefill(prompt), rtol=0, atol=0)
                graph.pending_token = eager.pending_token = torch.tensor(4, device="cuda")
                saved = None
                for _ in range(3):
                    expected, actual = eager.decode_step(), graph.decode_step()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.history(graph, eager, exact=True)
                    if saved is not None:
                        torch.testing.assert_close(saved[0], saved[1], rtol=0, atol=0)
                    saved = (actual, actual.clone())
            with self.assertRaises(AssertionError):
                graph.decode_step()
            self.assertEqual(addresses, [getattr(graph.state, name).data_ptr()
                                        for name in ("kv_k", "rec_state", "conv_state")])
            graph.reset()
            self.assertEqual(graph.state.position, 0)
            self.assertIsNone(graph.pending_token)
            self.assertFalse(graph.state.rec_state.count_nonzero().item())
            self.assertFalse(graph.state.conv_state.count_nonzero().item())

    @torch.inference_mode()
    def test_chunked_prefill_and_subsequent_decode(self):
        for variant in (0, 1):
            torch.manual_seed(920 + variant)
            ckpt = unpack(pack(random_checkpoint(small_config(variant))), torch.device("cuda"))
            eager = Engine(ckpt, max_context=48, cuda_graph=False, sequence_prefill=False)
            for chunk in (1, 2, 7, 16, 128):
                batched = Engine(ckpt, max_context=48, cuda_graph=False, prefill_chunk_size=chunk)
                for prompt in ([3], [3, 7, 5, 2, 8], [2, 7, 3] * 11):
                    expected, actual = eager.prefill(prompt), batched.prefill(prompt)
                    torch.testing.assert_close(actual, expected, rtol=.025, atol=.003)
                    self.history(batched, eager)
                    for token in (4, 6, 2):
                        eager.pending_token = batched.pending_token = torch.tensor(token, device="cuda")
                        torch.testing.assert_close(batched.decode_step(), eager.decode_step(), rtol=0, atol=0)
                        self.history(batched, eager)


if __name__ == "__main__":
    unittest.main()
