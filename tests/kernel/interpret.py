"""Execute Triton kernels on CPU for indexing, arithmetic, and state checks."""

import os
os.environ["TRITON_INTERPRET"] = "1"

import unittest
from pathlib import Path

import torch
import triton.language as tl
from triton.runtime import interpreter

from quickdraw.kernel.triton.delta import _conv_kernel, _delta_kernel
from quickdraw.kernel.triton.matvec import expert_gemv, fp8_gemv, nvfp4_gemv
from quickdraw.kernel.triton.moe import _route_kernel, moe_block
from quickdraw.models import qwen
from quickdraw.models.weights import FP8Linear, NVFP4Linear


def ieee_cast(values, source, destination, rounding):
    """Use Torch conversions: this interpreter truncates BF16 and mishandles FP8 ties."""
    dtypes = {tl.float32: torch.float32, tl.bfloat16: torch.bfloat16,
              tl.float8e4nv: torch.float8_e4m3fn}
    tensor = torch.from_numpy(values.copy()).view(dtypes[source]).to(dtypes[destination])
    storage = {8: torch.uint8, 16: torch.uint16, 32: torch.uint32}[destination.primitive_bitwidth]
    return tensor.view(storage).numpy()


interpreter._convert_float = ieee_cast


class InterpreterChecks(unittest.TestCase):
    def test_saved_blocks_and_complete_steps(self):
        from tests.kernel.check import check_saved_math
        check_saved_math(device="cpu")

    def test_generation_and_reset_with_triton_bindings(self):
        from quickdraw import Engine
        from quickdraw.kernel.triton import kernels
        from tests.models.mathcheck import unpack

        directory = Path(__file__).resolve().parents[2] / "references/model"
        for variant in (0, 1):
            fixture = unpack(torch.load(directory / f"forward-v{variant}-p0.pt", weights_only=True), "cpu")
            ckpt = fixture["kwargs"]["ckpt"]
            reference = Engine(ckpt, device="cpu", max_context=8)
            engine = Engine(ckpt, device="cpu", max_context=8, kernels=kernels())
            expected = reference.generate([3, 7, 5], 5)
            for _ in range(2):
                self.assertEqual(engine.generate([3, 7, 5], 5), expected)
                self.assertEqual(engine.state.position, 7)
                self.assertEqual(engine.pending_token.item(), expected[-1])
                torch.testing.assert_close(engine.state.rec_state, reference.state.rec_state, rtol=.02, atol=.002)

    def test_convolution_history_and_width_one(self):
        torch.manual_seed(12)
        for width in (1, 2, 4):
            state = torch.randn(7, width - 1).bfloat16()
            expected_state = state.clone()
            weight = torch.randn(7, width).bfloat16()
            for _ in range(5):
                x = torch.randn(7).bfloat16()
                history = torch.cat((expected_state, x[:, None]), -1)
                expected = torch.nn.functional.silu((history * weight).float().sum(-1)).bfloat16()
                expected_state.copy_(history[:, 1:])
                output = torch.empty_like(x)
                _conv_kernel[(1,)](x, weight, state, output, 7, width, 16, enable_fp_fusion=False)
                torch.testing.assert_close(output, expected, rtol=.008, atol=1e-6)
                torch.testing.assert_close(state, expected_state, rtol=0, atol=0)

    def test_recurrence_non_square_state_and_grouped_heads(self):
        torch.manual_seed(13)
        state = torch.randn(4, 3, 5) * .1
        expected_state = state.clone()
        alog = torch.randn(4).bfloat16()
        bias = torch.randn(4).bfloat16()
        for _ in range(3):
            packed = torch.randn(2 * 2 * 3 + 4 * 5).bfloat16()
            a, b = torch.randn(4).bfloat16(), torch.randn(4).bfloat16()
            q, k, v = packed[:6].reshape(2, 3).float(), packed[6:12].reshape(2, 3).float(), packed[12:].reshape(4, 5).float()
            q = (q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / 3 ** .5).repeat_interleave(2, 0)
            k = (k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(2, 0)
            expected_core = []
            decay = (-alog.float().exp() * torch.nn.functional.softplus(a.float() + bias.float())).exp()
            for head in range(4):
                prior = expected_state[head] * decay[head]
                correction = (v[head] - k[head] @ prior) * b[head].float().sigmoid()
                expected_state[head] = prior + torch.outer(k[head], correction)
                expected_core.append(q[head] @ expected_state[head])
            output = torch.empty(4, 5, dtype=torch.bfloat16)
            _delta_kernel[(4, 2)](packed, a, b, alog, bias, state, output, 2, 4, 3, 5, 4, 4,
                                   enable_fp_fusion=False)
            torch.testing.assert_close(state, expected_state, rtol=2e-5, atol=1e-6)
            torch.testing.assert_close(output, torch.stack(expected_core).bfloat16(), rtol=.008, atol=1e-6)

    def test_routing_non_power_of_two_and_ties(self):
        for scores in (torch.tensor([-4., 2., .5, -1., 3., 1.]), torch.ones(6)):
            ids = torch.empty(3, dtype=torch.int32)
            weights = torch.empty(3)
            _route_kernel[(1,)](scores.bfloat16(), ids, weights, 6, 3, 8, enable_fp_fusion=False)
            probabilities = scores.softmax(-1)
            expected = probabilities[ids.long()] / probabilities[ids.long()].sum()
            torch.testing.assert_close(weights, expected)
            self.assertEqual(ids.tolist(), [0, 1, 2] if scores.min() == scores.max() else [4, 1, 5])

    def test_expert_bank_offsets_and_per_slot_inputs(self):
        torch.manual_seed(14)
        proj = NVFP4Linear(torch.randint(0, 256, (4, 13, 8), dtype=torch.uint8),
                           (torch.randint(1, 5, (4, 13, 1)).float() / 4).to(torch.float8_e4m3fn),
                           torch.tensor([.137, .173, .213, .119]))
        ids = torch.tensor([3, 0, 2], dtype=torch.int32)
        for per_slot in (False, True):
            x = torch.randn((3, 16) if per_slot else (16,)).bfloat16()
            for weights in (None, torch.tensor([.13, .57, .30])):
                output = expert_gemv(x, proj, ids, weights)
                expected = []
                for slot, expert in enumerate(ids.tolist()):
                    view = qwen._expert_projection(proj, expert)
                    expected.append(qwen.nvfp4_gemv(x[slot] if per_slot else x, view,
                        routing_weight=None if weights is None else weights[slot]))
                torch.testing.assert_close(output, torch.stack(expected), rtol=.02, atol=.002)

    def test_joint_routing_scale_rounding(self):
        proj = NVFP4Linear(torch.full((1, 8), 0x77, dtype=torch.uint8),
                           torch.tensor([[.5]]).to(torch.float8_e4m3fn), torch.tensor(.137))
        actual = nvfp4_gemv(torch.arange(1, 17).bfloat16(), proj, routing_weight=torch.tensor(.1))
        self.assertEqual(actual.item(), 5.5625)

    def test_fp8_rounding_ties(self):
        projection = FP8Linear(torch.eye(3).to(torch.float8_e4m3fn), torch.tensor(1.), torch.tensor(1.))
        result = fp8_gemv(torch.tensor([1.0625, 1.1875, -1.0625]), projection)
        torch.testing.assert_close(result, torch.tensor([1., 1.25, -1.]), rtol=0, atol=0)

    def test_moe_ties_match_reference(self):
        from tests.models.mathcheck import unpack
        path = Path(__file__).resolve().parents[2] / "references/model/moe-v0-top2.pt"
        kwargs = unpack(torch.load(path, weights_only=True), "cpu")["kwargs"]
        kwargs["w"].router.zero_()
        torch.testing.assert_close(moe_block(**kwargs), qwen.moe_block(**kwargs), rtol=.02, atol=.002)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
