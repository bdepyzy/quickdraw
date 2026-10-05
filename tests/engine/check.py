"""CPU request-control checks using synthetic logits and history."""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import traceback
from unittest.mock import patch

import torch

from quickdraw.models import weights as checkpoint, qwen as model
from quickdraw.engine import engine

CHECKS = []
GROUPS = ("init", "from_pretrained", "prefill", "decode_step", "generate", "reset")
EOS = 39
PROMPT = [11, 22, 33]
STATE_TENSORS = ("kv_k", "kv_v", "rec_state", "conv_state")


def check(group, description):
    def register(fn):
        CHECKS.append((group, description, fn))
        return fn
    return register


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def tiny_checkpoint(config=None):
    """Real checkpoint dataclasses with small shapes; all weights are synthetic."""
    c = config or checkpoint.ModelConfig(
        hidden_size=16, num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention"] * 2,
        full_attention_interval=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, linear_num_key_heads=1, linear_key_head_dim=4,
        linear_num_value_heads=2, linear_value_head_dim=4, linear_conv_kernel_dim=4,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, rms_norm_eps=1e-6,
        partial_rotary_factor=0.5, rope_theta=10000., vocab_size=40)

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.bfloat16)

    def fp8(out, inp):
        return checkpoint.FP8Linear(torch.zeros(out, inp, dtype=torch.float8_e4m3fn),
                                    torch.tensor(1.), torch.tensor(1.))

    def fp4(out, inp, experts=None):
        prefix = () if experts is None else (experts,)
        return checkpoint.NVFP4Linear(
            torch.zeros((*prefix, out, inp // 2), dtype=torch.uint8),
            torch.ones((*prefix, out, inp // 16)).to(torch.float8_e4m3fn),
            torch.ones(()) if experts is None else torch.ones(experts),
            torch.ones(()) if experts is None else torch.ones(experts))

    layers = []
    h, mi, si = c.hidden_size, c.moe_intermediate_size, c.shared_expert_intermediate_size
    qk = c.linear_num_key_heads * c.linear_key_head_dim
    value = c.linear_num_value_heads * c.linear_value_head_dim
    channels = 2 * qk + value
    for idx, kind in enumerate(c.layer_types):
        moe = checkpoint.MoEWeights(
            zeros(c.num_experts, h), zeros(1, h),
            fp4(mi, h, c.num_experts), fp4(mi, h, c.num_experts),
            fp4(h, mi, c.num_experts), fp4(si, h), fp4(si, h), fp4(h, si))
        if kind == "linear_attention":
            attn = checkpoint.GDNWeights(
                fp8(channels, h), fp8(value, h),
                zeros(c.linear_num_value_heads, h), zeros(c.linear_num_value_heads, h),
                zeros(channels, 1, c.linear_conv_kernel_dim),
                zeros(c.linear_num_value_heads), zeros(c.linear_num_value_heads),
                torch.ones(c.linear_value_head_dim, dtype=torch.bfloat16), fp8(h, value))
        else:
            q = c.num_attention_heads * c.head_dim
            kv = c.num_key_value_heads * c.head_dim
            attn = checkpoint.FullAttnWeights(
                fp8(2 * q, h), fp8(kv, h), fp8(kv, h), fp8(h, q),
                torch.ones(c.head_dim, dtype=torch.bfloat16),
                torch.ones(c.head_dim, dtype=torch.bfloat16))
        layers.append(checkpoint.LayerWeights(
            idx, kind, torch.ones(h, dtype=torch.bfloat16),
            torch.ones(h, dtype=torch.bfloat16), attn, moe))
    return checkpoint.Checkpoint(c, zeros(c.vocab_size, h), torch.ones(h, dtype=torch.bfloat16),
                                 fp4(c.vocab_size, h), layers)


def scalar_token(token, name="token"):
    require(isinstance(token, torch.Tensor), f"{name} must be a torch.Tensor")
    require(token.shape == torch.Size([]), f"{name} must be scalar; got {token.shape}")
    require(token.dtype == torch.long, f"{name} must have torch.long dtype")
    require(token.device.type == "cpu", f"{name} must be on CPU in this checker")
    return token.item()


class FakeForward:
    next_token = {11: 22, 22: 33, 33: 7, 7: 9, 9: 4, 4: EOS, EOS: 2}

    def __init__(self, ckpt, capacity):
        self.ckpt = ckpt
        self.capacity = capacity
        self.calls = []

    def __call__(self, ckpt, token_id, position, state):
        require(ckpt is self.ckpt, "forward_step received a different checkpoint")
        token = scalar_token(token_id, "forward_step token_id")
        pos = int(position)
        require(0 <= pos < self.capacity, f"model processed out-of-range position {pos}")
        require(int(state.position) == pos,
                f"state.position={state.position}, but forward_step position={pos}")
        require(token in self.next_token, f"unexpected model input token {token}")
        self.calls.append((token, pos))
        state.kv_k[:, pos].fill_(token + 0.5)
        state.kv_v[:, pos].fill_(token + 0.25)
        state.rec_state.mul_(2).add_(token)
        state.conv_state.copy_(state.conv_state.roll(-1, dims=-1))
        state.conv_state[..., -1].fill_(token)
        logits = torch.full((ckpt.config.vocab_size,), -20., dtype=torch.bfloat16)
        logits[self.next_token[token]] = 10.
        return logits


@contextmanager
def fake_model(ckpt, capacity):
    fake = FakeForward(ckpt, capacity)
    with ExitStack() as stack:
        stack.enter_context(patch.object(model, "forward_step", fake))
        stack.enter_context(patch.object(engine, "load_checkpoint",
                                         side_effect=AssertionError("real checkpoint load forbidden")))
        stack.enter_context(patch.object(checkpoint, "load_checkpoint",
                                         side_effect=AssertionError("real checkpoint load forbidden")))
        yield fake


def new_engine(ckpt=None, capacity=16):
    ckpt = ckpt or tiny_checkpoint()
    e = engine.Engine(ckpt, max_context=capacity, device="cpu")
    require(hasattr(e, "state"), "Engine must expose its DecodeState as self.state")
    require(isinstance(e.state, engine.DecodeState), "self.state must be DecodeState")
    require(hasattr(e, "pending_token"), "Engine must expose self.pending_token (initially None)")
    require(hasattr(e, "eos_token_ids") and isinstance(e.eos_token_ids, set),
            "Engine must initialize self.eos_token_ids as a set (empty disables EOS)")
    e.eos_token_ids = {EOS}
    return e


def expect_calls(fake, tokens):
    expected = list(zip(tokens, range(len(tokens))))
    require(fake.calls == expected,
            f"model calls (token, position): got {fake.calls}; expected {expected}")


def expect_state(e, tokens):
    require(int(e.state.position) == len(tokens),
            f"position: got {e.state.position}; expected {len(tokens)} processed tokens")
    value = 0
    for token in tokens:
        value = 2 * value + token
    torch.testing.assert_close(e.state.rec_state,
                               torch.full_like(e.state.rec_state, value), rtol=0, atol=0)
    width = e.state.conv_state.shape[-1]
    history = ([0] * width + tokens)[-width:]
    expected_conv = torch.tensor(history, dtype=e.state.conv_state.dtype).expand_as(e.state.conv_state)
    torch.testing.assert_close(e.state.conv_state, expected_conv, rtol=0, atol=0)
    for pos, token in enumerate(tokens):
        for name, offset in (("kv_k", 0.5), ("kv_v", 0.25)):
            actual = getattr(e.state, name)[:, pos]
            torch.testing.assert_close(actual, torch.full_like(actual, token + offset), rtol=0, atol=0)


def expect_logits(logits, token, vocab=40):
    expected = torch.full((vocab,), -20., dtype=torch.bfloat16)
    expected[token] = 10.
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def expect_output(output, expected):
    require(isinstance(output, list), f"generate must return list[int]; got {type(output).__name__}")
    require(all(type(x) is int for x in output), "generate output must contain Python int IDs")
    require(output == expected, f"output: got {output}; expected {expected}")


@check("init", "state shapes/dtypes follow config; recurrent/conv state starts at zero")
def init_shapes():
    base = tiny_checkpoint().config
    variants = [base, replace(base, num_hidden_layers=3,
                              layer_types=["full_attention", "linear_attention", "full_attention"],
                              linear_num_value_heads=3, linear_conv_kernel_dim=2)]
    for config in variants:
        e = new_engine(tiny_checkpoint(config), capacity=7)
        c = e.ckpt.config
        n_gdn = c.layer_types.count("linear_attention")
        n_full = c.layer_types.count("full_attention")
        channels = (2 * c.linear_num_key_heads * c.linear_key_head_dim
                    + c.linear_num_value_heads * c.linear_value_head_dim)
        expected = {
            "kv_k": ((n_full, 7, c.num_key_value_heads, c.head_dim), torch.bfloat16),
            "kv_v": ((n_full, 7, c.num_key_value_heads, c.head_dim), torch.bfloat16),
            "rec_state": ((n_gdn, c.linear_num_value_heads, c.linear_key_head_dim,
                           c.linear_value_head_dim), torch.float32),
            "conv_state": ((n_gdn, channels, c.linear_conv_kernel_dim - 1), torch.bfloat16),
        }
        pointers = []
        for name, (shape, dtype) in expected.items():
            tensor = getattr(e.state, name)
            require(isinstance(tensor, torch.Tensor), f"{name} must be a tensor")
            require(tuple(tensor.shape) == shape, f"{name} shape: got {tuple(tensor.shape)}, expected {shape}")
            require(tensor.dtype == dtype, f"{name} dtype: got {tensor.dtype}, expected {dtype}")
            require(tensor.device.type == "cpu", f"{name} must honor device='cpu'")
            pointers.append(tensor.data_ptr())
        require(len(set(pointers)) == len(pointers), "state buffers must not start at the same address")
        require(e.pending_token is None, "pending_token must initially be None")
        expect_state(e, [])


@check("from_pretrained", "explicit CPU device reaches both loader and constructor")
def factory_device():
    ckpt = tiny_checkpoint()
    seen = {}

    def init(self, received, max_context=32768, device="cuda"):
        seen.update(ckpt=received, max_context=max_context, device=device)

    with patch.object(engine, "load_checkpoint", return_value=ckpt) as loader:
        with patch.object(engine.Engine, "__init__", init):
            engine.Engine.from_pretrained("synthetic/no-files", device="cpu", max_context=7)
    loader.assert_called_once_with("synthetic/no-files", device="cpu")
    require(seen["ckpt"] is ckpt, "constructor must receive the loaded checkpoint")
    require(str(seen["device"]) == "cpu", f"constructor device: got {seen['device']}; expected cpu")
    require(seen["max_context"] == 7, "max_context must reach constructor")


@check("init", "custom step handles prompt/decode; reset retains its state buffers")
def custom_step():
    ckpt = tiny_checkpoint()
    fake = FakeForward(ckpt, 16)

    step = fake
    with patch.object(model, "forward_step", side_effect=AssertionError("default execution forbidden")):
        e = engine.Engine(ckpt, max_context=16, device="cpu", step=step)
        pointers = {name: getattr(e.state, name).data_ptr() for name in STATE_TENSORS}
        for _ in range(2):
            fake.calls.clear()
            expect_output(e.generate(PROMPT.copy(), max_tokens=3), [7, 9, 4])
            expect_calls(fake, PROMPT + [7, 9])
            expect_state(e, PROMPT + [7, 9])
            e.reset()
            require(e.step is step, "reset replaced step")
            for name, pointer in pointers.items():
                require(getattr(e.state, name).data_ptr() == pointer, "reset replaced captured history")


@check("prefill", "process prompt exactly once; return first-output logits")
def prefill_order():
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 16) as fake:
        e = new_engine(ckpt)
        prompt = PROMPT.copy()
        logits = e.prefill(prompt)
        require(prompt == PROMPT, "prefill changed the caller's prompt list")
        expect_logits(logits, 7)
        expect_calls(fake, PROMPT)
        expect_state(e, PROMPT)


@check("prefill", "second prefill starts fresh; no state leaks from previous prompt")
def prefill_fresh():
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 16) as fake:
        e = new_engine(ckpt)
        e.prefill(PROMPT.copy())
        fake.calls.clear()
        logits = e.prefill([33])
        expect_logits(logits, 7)
        expect_calls(fake, [33])
        expect_state(e, [33])


@check("decode_step", "consume pending token once; advance position; select successor")
def decode_pending():
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 16) as fake:
        e = new_engine(ckpt)
        e.prefill(PROMPT.copy())
        e.pending_token = torch.tensor(7, dtype=torch.long)
        require(scalar_token(e.decode_step()) == 9, "decode_step should select token 9")
        require(scalar_token(e.pending_token) == 9, "pending_token should become token 9")
        require(scalar_token(e.decode_step()) == 4, "second decode_step should select token 4")
        require(scalar_token(e.pending_token) == 4, "pending_token should become token 4")
        expect_calls(fake, PROMPT + [7, 9])
        expect_state(e, PROMPT + [7, 9])


@check("generate", "three outputs: [7, 9, 4]; five processed tokens; no extra step")
def generate_chain():
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 16) as fake:
        e = new_engine(ckpt)
        prompt = PROMPT.copy()
        expect_output(e.generate(prompt, max_tokens=3), [7, 9, 4])
        require(prompt == PROMPT, "generate changed the caller's prompt list")
        expect_calls(fake, PROMPT + [7, 9])
        expect_state(e, PROMPT + [7, 9])
        require(scalar_token(e.pending_token) == 4, "final selected token must remain pending")


@check("generate", "one output uses prefill logits without a decode call")
def generate_one():
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 16) as fake:
        e = new_engine(ckpt)
        expect_output(e.generate(PROMPT.copy(), max_tokens=1), [7])
        expect_calls(fake, PROMPT)
        expect_state(e, PROMPT)


@check("generate", "zero outputs perform no model work and do not change existing state")
def generate_zero():
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 16) as fake:
        e = new_engine(ckpt)
        e.prefill(PROMPT.copy())
        e.pending_token = torch.tensor(7, dtype=torch.long)
        before = {name: getattr(e.state, name).clone() for name in STATE_TENSORS}
        pos = int(e.state.position)
        fake.calls.clear()
        expect_output(e.generate([33], max_tokens=0), [])
        expect_calls(fake, [])
        require(int(e.state.position) == pos, "zero-output generate changed position")
        require(scalar_token(e.pending_token) == 7, "zero-output generate changed pending token")
        for name, expected in before.items():
            torch.testing.assert_close(getattr(e.state, name), expected, rtol=0, atol=0,
                                       equal_nan=True)  # unused KV entries may be uninitialized


@check("generate", "stop on EOS; include EOS in returned IDs; never process it")
def generate_eos():
    for prompt, expected, consumed in ((PROMPT, [7, 9, 4, EOS], PROMPT + [7, 9, 4]),
                                       ([4], [EOS], [4])):
        ckpt = tiny_checkpoint()
        with fake_model(ckpt, 16) as fake:
            e = new_engine(ckpt)
            expect_output(e.generate(prompt.copy(), max_tokens=8), expected)
            expect_calls(fake, consumed)
            expect_state(e, consumed)


@check("generate", "empty prompt and negative output limit fail before model work")
def generate_invalid():
    for prompt, count in (([], 1), (PROMPT, -1)):
        ckpt = tiny_checkpoint()
        with fake_model(ckpt, 16) as fake:
            e = new_engine(ckpt)
            try:
                e.generate(prompt.copy(), max_tokens=count)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"expected AssertionError for prompt={prompt}, max_tokens={count}")
            expect_calls(fake, [])
            expect_state(e, [])


@check("generate", "full-context prompt permits one output without an out-of-range step")
def generate_exact_capacity():
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 3) as fake:
        e = new_engine(ckpt, capacity=3)
        expect_output(e.generate(PROMPT.copy(), max_tokens=1), [7])
        expect_calls(fake, PROMPT)
        expect_state(e, PROMPT)


@check("generate", "reject oversized prompt; reject or stop outputs at context capacity")
def generate_capacity():
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 2) as fake:
        e = new_engine(ckpt, capacity=2)
        try:
            e.generate(PROMPT.copy(), max_tokens=1)
        except AssertionError:
            pass
        else:
            raise AssertionError("oversized prompt must raise AssertionError")
        expect_calls(fake, [])
        expect_state(e, [])
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 4) as fake:
        e = new_engine(ckpt, capacity=4)
        try:
            output = e.generate(PROMPT.copy(), max_tokens=3)
        except AssertionError:
            expect_calls(fake, [])
            expect_state(e, [])
        else:
            expect_output(output, [7, 9])
            expect_calls(fake, PROMPT + [7])
            expect_state(e, PROMPT + [7])


@check("reset", "clear recurrent/conv state, position and pending token; retain buffers")
def reset_state():
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 16):
        e = new_engine(ckpt)
        e.prefill(PROMPT.copy())
        e.pending_token = torch.tensor(7, dtype=torch.long)
        pointers = {name: getattr(e.state, name).data_ptr() for name in STATE_TENSORS}
        for _ in range(2):
            e.reset()
            expect_state(e, [])
            require(e.pending_token is None, "reset must clear pending_token")
            for name, pointer in pointers.items():
                require(getattr(e.state, name).data_ptr() == pointer,
                        f"reset replaced {name}; reuse allocated buffers")


@check("reset", "same request after reset matches a fresh engine's output and active state")
def reset_repeat():
    ckpt = tiny_checkpoint()
    with fake_model(ckpt, 16) as fake:
        e = new_engine(ckpt)
        expect_output(e.generate(PROMPT.copy(), max_tokens=3), [7, 9, 4])
        e.reset()
        fake.calls.clear()
        repeated = e.generate([33], max_tokens=3)
        expect_calls(fake, [33, 7, 9])
        expect_state(e, [33, 7, 9])
        fresh = new_engine(ckpt)
        fake.calls.clear()
        expected = fresh.generate([33], max_tokens=3)
        expect_output(repeated, [7, 9, 4])
        expect_output(expected, repeated)
        expect_state(fresh, [33, 7, 9])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=GROUPS)
    parser.add_argument("--strict", action="store_true", help="fail on skipped checks")
    parser.add_argument("--verbose", action="store_true", help="show failure tracebacks")
    args = parser.parse_args(argv)
    passed = failed = skipped = 0
    print("Engine checks on CPU; fake model tests control flow, not Qwen math or GPU speed.\n")
    with torch.inference_mode():
        for group, description, fn in CHECKS:
            if args.only is not None and group != args.only:
                continue
            try:
                fn()
            except NotImplementedError as exc:
                skipped += 1
                print(f"SKIP {group}: {description}\n     NotImplementedError: {exc}")
            except Exception as exc:
                failed += 1
                print(f"FAIL {group}: {description}\n     {type(exc).__name__}: {exc}")
                if args.verbose:
                    traceback.print_exc()
            else:
                passed += 1
                print(f"PASS {group}: {description}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped.")
    if skipped:
        print("Skipped checks fail with --strict.")
    return int(failed > 0 or (args.strict and skipped > 0))


if __name__ == "__main__":
    raise SystemExit(main())
