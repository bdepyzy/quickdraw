"""Formula, dtype, and mutable-state checks against independent CPU references.
RMSNorm accepts gamma directly. Fixture provenance is in references/model/README.md."""

import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path
import traceback

import torch

from quickdraw.kernel import bind_kernels
from quickdraw.models import weights as checkpoint, qwen as model
from quickdraw.kvcache.state import DecodeState


CHECKS = []
FUNCTIONS = (
    "rms_norm", "quantize_fp8", "fp8_gemv", "full_attention_block",
    "gated_delta_net_block", "moe_block", "forward_step", "sample",
)
STATE_ARGUMENTS = {
    "full_attention_block": {"kv_cache"},
    "gated_delta_net_block": {"rec_state", "conv_state"},
    "forward_step": {"state"},
}


def check(group, name):
    """Register a formula check."""
    def register(fn):
        CHECKS.append((group, name, fn))
        return fn
    return register


def close(actual, expected, *, rtol=1e-5, atol=1e-6):
    assert isinstance(actual, torch.Tensor), "expected a torch.Tensor result"
    assert actual.shape == expected.shape, f"shape: got {actual.shape}, expected {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype: got {actual.dtype}, expected {expected.dtype}"
    assert actual.device == expected.device, f"device: got {actual.device}, expected {expected.device}"
    # CPU FP8 comparisons require conversion.
    if actual.is_floating_point():
        torch.testing.assert_close(actual.double(), expected.double(), rtol=rtol, atol=atol)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def reference_rms(x, weight, eps):
    rows = x.detach().cpu().double().reshape(-1, x.shape[-1]).tolist()
    gamma = weight.detach().cpu().double().tolist()
    result = []
    for row in rows:
        denominator = (sum(value * value for value in row) / len(row) + eps) ** 0.5
        result.append([value / denominator * scale for value, scale in zip(row, gamma)])
    return torch.tensor(result, device=x.device, dtype=x.dtype).reshape(x.shape)


def rms_case(x, weight, eps):
    saved_x, saved_weight = x.clone(), weight.clone()
    expected = reference_rms(x, weight, eps)
    actual = model.rms_norm(x, weight, eps)
    close(actual, expected, rtol=0.016 if x.dtype == torch.bfloat16 else 2e-5,
          atol=1e-7)
    close(x, saved_x, rtol=0, atol=0)
    close(weight, saved_weight, rtol=0, atol=0)


@check("rms_norm", "hand calculation: [3, 4], gamma=[1, 2]")
def rms_hand(device):
    x = torch.tensor([3., 4.], device=device)
    weight = torch.tensor([1., 2.], device=device)
    close(model.rms_norm(x, weight, 0.),
          torch.tensor([3 / (12.5 ** 0.5), 8 / (12.5 ** 0.5)], device=device))


@check("rms_norm", "last dimension, 1D/2D/3D and noncontiguous inputs")
def rms_axes(device):
    for shape in [(4,), (3, 4), (2, 3, 4)]:
        x = torch.arange(1, 1 + 2 * torch.Size(shape).numel(), device=device).float()
        x = x.reshape(*shape[:-1], 8)[..., ::2]  # noncontiguous view
        rms_case(x, torch.tensor([0., -2., 0.5, 3.], device=device), 1e-6)


@check("rms_norm", "epsilon, zero input and constant input (no mean subtraction)")
def rms_epsilon(device):
    for values in [[0., 0., 0., 0.], [2., 2., 2., 2.], [1e-5, -1e-5, 2e-5, 0.]]:
        for eps in [1e-6, 0.5]:
            rms_case(torch.tensor(values, device=device), torch.ones(4, device=device), eps)


@check("rms_norm", "BF16 output and FP32 accumulation")
def rms_bf16(device):
    # Finite BF16 inputs whose squares overflow FP16 arithmetic.
    x = torch.tensor([[300., -600., 900., -1200.], [0.01, 0.02, -0.03, 0.04]],
                     device=device, dtype=torch.bfloat16)
    weight = torch.tensor([1., -0.5, 2., 0.], device=device, dtype=x.dtype)
    rms_case(x, weight, 1e-6)


@check("rms_norm", "model-sized hidden vectors (2048 features)")
def rms_hidden(device):
    x = torch.randn(2, 2048, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2048, device=device, dtype=x.dtype)
    rms_case(x, weight, 1e-6)


@check("quantize_fp8", "scale direction, rounding, saturation, shape and dtype")
def fp8_quant(device):
    x = torch.tensor([[-1000., -224., -0.55, 0.], [0.55, 0.6, 224., 1000.]], device=device)
    scale = torch.tensor(0.5, device=device)
    saved = x.clone()
    q, returned_scale = model.quantize_fp8(x, scale)
    expected = torch.tensor([[-448., -448., -1.125, 0.], [1.125, 1.25, 448., 448.]],
                            device=device).to(torch.float8_e4m3fn)
    close(q, expected, rtol=0, atol=0)
    assert isinstance(returned_scale, torch.Tensor) and returned_scale.numel() == 1
    close(returned_scale.reshape(()), scale, rtol=0, atol=0)
    close(x, saved, rtol=0, atol=0)


@check("quantize_fp8", "BF16 and noncontiguous input with non-unit scale")
def fp8_quant_bf16(device):
    x = torch.tensor([[1., 99., -2., 99.], [0., 99., 4., 99.]],
                     dtype=torch.bfloat16, device=device)[:, ::2]
    scale = torch.tensor(2., device=device)
    q, returned_scale = model.quantize_fp8(x, scale)
    close(q, torch.tensor([[0.5, -1.], [0., 2.]], device=device).to(torch.float8_e4m3fn),
          rtol=0, atol=0)
    close(returned_scale.reshape(()), scale, rtol=0, atol=0)


@check("fp8_gemv", "matrix orientation and BOTH activation/weight scales")
def fp8_matvec(device):
    # x/input_scale rounds to [1.125, -2.25, 0.5, 0]; multiplying
    # unquantized BF16 activations would give a different answer.
    x = torch.tensor([0.55, -1.1, 0.25, 0.], device=device, dtype=torch.bfloat16)
    w = torch.tensor([[1., 2., 0., -1.], [-2., 0., 4., 1.], [1., -1., 2., 0.]], device=device)
    proj = checkpoint.FP8Linear(w.to(torch.float8_e4m3fn),
                                torch.tensor(0.25, device=device),
                                torch.tensor(0.5, device=device))
    expected = torch.tensor([-0.421875, -0.03125, 0.546875], device=device, dtype=x.dtype)
    close(model.fp8_gemv(x, proj), expected, rtol=0.008, atol=1e-6)


@check("sample", "greedy selection, negative scores and scalar token ID")
def sample_greedy(device):
    for scores, token in [([-5., -2., -9.], 1), ([4., 1., 0.], 0), ([0., 1., 4.], 2)]:
        x = torch.tensor(scores, device=device)
        saved = x.clone()
        close(model.sample(x, temperature=0.), torch.tensor(token, device=device))
        close(x, saved, rtol=0, atol=0)


@check("sample", "positive temperature draws from the expected distribution")
def sample_distribution(device):
    logits = torch.tensor([0., 2.], device=device)
    counts = [0, 0]
    for _ in range(2048):
        token = model.sample(logits, temperature=2.)
        assert isinstance(token, torch.Tensor) and token.shape == torch.Size([])
        assert token.dtype == torch.long and token.device == logits.device
        value = token.item()
        assert value in (0, 1), f"invalid token: {value}"
        counts[value] += 1
    # softmax([0, 1])[1] ~= 0.731; broad band avoids flaky exact-seed tests.
    assert 0.68 < counts[1] / 2048 < 0.78, f"unexpected sampling counts: {counts}"


# Reconstruct known dataclasses from tensor-only fixtures.
TYPES = {cls.__name__: cls for cls in (
    checkpoint.FP8Linear, checkpoint.NVFP4Linear, checkpoint.GDNWeights,
    checkpoint.FullAttnWeights, checkpoint.MoEWeights, checkpoint.LayerWeights,
    checkpoint.ModelConfig, checkpoint.Checkpoint, DecodeState,
)}


def pack(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        assert type(value).__name__ in TYPES, f"unsupported dataclass: {type(value)}"
        return {"__type__": type(value).__name__,
                "fields": {f.name: pack(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(pack(v) for v in value)
    return value


def unpack(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device).clone()
    if isinstance(value, dict):
        if "__type__" in value:
            return TYPES[value["__type__"]](**unpack(value["fields"], device))
        return {k: unpack(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(unpack(v, device) for v in value)
    return value


def save_reference(path, *, function, kwargs, expected, source, after=None,
                   rtol=0.02, atol=0.002):
    """Save pre-call inputs, independent output, post-call state, and provenance.
    Arguments use quickdraw layouts; source identifies the implementation and version."""
    if function not in FUNCTIONS or not source.strip():
        raise ValueError("a known function and nonempty reference source are required")
    if not STATE_ARGUMENTS.get(function, set()) <= set(after or {}):
        raise ValueError(f"after must include {STATE_ARGUMENTS[function]}")
    torch.save(pack(dict(function=function, kwargs=kwargs, expected=expected,
                         after=after or {}, source=source, rtol=rtol, atol=atol)), path)


def compare_tree(actual, expected, rtol, atol, path="result"):
    try:
        if isinstance(expected, torch.Tensor):
            close(actual, expected, rtol=rtol, atol=atol)
        elif is_dataclass(expected):
            assert type(actual) is type(expected)
            for f in fields(expected):
                compare_tree(getattr(actual, f.name), getattr(expected, f.name),
                             rtol, atol, f"{path}.{f.name}")
        elif isinstance(expected, dict):
            assert isinstance(actual, dict) and actual.keys() == expected.keys()
            for k in expected:
                compare_tree(actual[k], expected[k], rtol, atol, f"{path}.{k}")
        elif isinstance(expected, (list, tuple)):
            assert type(actual) is type(expected) and len(actual) == len(expected)
            for i, (a, e) in enumerate(zip(actual, expected)):
                compare_tree(a, e, rtol, atol, f"{path}[{i}]")
        else:
            assert actual == expected
    except AssertionError as exc:
        raise AssertionError(f"{path}: {exc}") from exc


def run_reference(data, device, *, kernels=None):
    """Check a saved output and state, optionally through a supplied kernel set."""
    data = unpack(data, device)
    function, kwargs = data["function"], data["kwargs"]
    if not STATE_ARGUMENTS.get(function, set()) <= set(data["after"]):
        raise ValueError("reference is missing expected state updates")
    if kernels is None:
        actual = getattr(model, function)(**kwargs)
    else:
        actual = bind_kernels(**kernels)[function](**kwargs)
    compare_tree(actual, data["expected"], data["rtol"], data["atol"])
    for name, expected in data["after"].items():
        compare_tree(kwargs[name], expected, data["rtol"], data["atol"], f"after.{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=FUNCTIONS, help="check just one function")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--reference", type=Path, action="append", default=[],
                        help="saved independent reference; repeat for multiple cases")
    parser.add_argument("--reference-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "references" / "model",
                        help="directory of saved references (default: references/model)")
    parser.add_argument("--strict", action="store_true", help="also fail on skips or missing coverage")
    parser.add_argument("--verbose", action="store_true", help="show tracebacks on failures")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    cases = list(CHECKS)
    paths = sorted(args.reference_dir.glob("*.pt")) + args.reference
    for path in dict.fromkeys(paths):
        if not path.is_file():
            parser.error(f"reference file does not exist: {path}; see references/model/README.md")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["function"] not in FUNCTIONS or not data["source"].strip():
            parser.error(f"invalid reference metadata: {path}")
        cases.append((data["function"], f"reference {path.name}",
                      lambda device, data=data: run_reference(data, device)))
    selected = {args.only} if args.only else set(FUNCTIONS)
    passed = failed = skipped = missing = 0
    print(f"Math checks on {args.device}; built-ins check formulas, not engine parity.\n")
    with torch.inference_mode():
        for group, name, fn in cases:
            if group not in selected:
                continue
            torch.manual_seed(0)
            try:
                fn(torch.device(args.device))
            except NotImplementedError:
                skipped += 1
                print(f"SKIP {group}: {name} (NotImplementedError)")
            except Exception as exc:
                failed += 1
                print(f"FAIL {group}: {name}\n     {type(exc).__name__}: {exc}")
                if args.verbose:
                    traceback.print_exc()
            else:
                passed += 1
                print(f"PASS {group}: {name}")
    covered = {group for group, _, _ in cases}
    for group in sorted(selected - covered):
        missing += 1
        print(f"UNVERIFIED {group}: supply --reference with independent output/state tensors")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped; {missing} functions unverified.")
    return int(failed > 0 or (args.strict and (skipped > 0 or missing > 0)))


if __name__ == "__main__":
    raise SystemExit(main())
