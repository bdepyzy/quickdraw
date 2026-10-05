"""Compare raw logits and greedy choices on identical forced token histories.
Capture uses eager vLLM, BF16 KV, FP32 recurrence, and no prefix/speculation."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch

DEFAULT_MODEL = Path.home() / "data/models/vllm/Qwen3.6-35B-A3B-NVFP4"
DEFAULT_CAPTURE = Path("references/engine/vllm.pt")


class CaptureWorker:
    """Named vLLM worker RPCs for independent logits; no pickle RPC or server."""

    def qd_install(self, destination):
        module = self.get_model()
        module._qd_logits = []
        original = module.compute_logits

        def capture(hidden, *args, **kwargs):
            logits = original(hidden, *args, **kwargs)
            module._qd_logits.append(logits.detach().float().cpu())
            return logits

        module.compute_logits = capture
        scales = {}
        for name, block in module.named_modules():
            if hasattr(block, "input_scale") and hasattr(block, "weight_scale"):
                scales[name] = {key: getattr(block, key).detach().float().cpu().tolist()
                                for key in ("input_scale", "weight_scale")}
        torch.save({"type": type(module).__name__, "projection_scales": scales}, destination)

    def qd_clear(self):
        self.get_model()._qd_logits.clear()

    def qd_take(self, destination):
        module = self.get_model()
        torch.save(torch.cat(module._qd_logits, dim=0), destination)
        module._qd_logits.clear()


def validate_capture(logits, output_ids, vocab_size):
    """Keep output rows only; reject missing rows or a shifted prediction history."""
    assert output_ids and logits.ndim == 2 and logits.shape[1] == vocab_size
    assert logits.shape[0] >= len(output_ids), "missing reference logits"
    logits = logits[:len(output_ids)].contiguous()
    assert torch.isfinite(logits).all(), "nonfinite reference logits"
    assert logits.argmax(-1).tolist() == output_ids, "logits do not align with returned tokens"
    return logits


def logit_metrics(actual, expected, token, *, max_rmse, max_tv):
    """Greedy agreement and distribution error are separate acceptance conditions."""
    assert actual.shape == expected.shape and actual.ndim == 1
    actual, expected = actual.float().cpu(), expected.float().cpu()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    assert expected.argmax().item() == token
    delta = actual - expected
    greedy = actual.argmax().item()
    top = expected.topk(2).values
    rmse = delta.square().mean().sqrt().item()
    tv = (actual.softmax(-1) - expected.softmax(-1)).abs().sum().item() / 2
    return dict(reference_id=token, quickdraw_id=greedy, top1_match=greedy == token,
                max_abs_error=delta.abs().max().item(), mean_abs_error=delta.abs().mean().item(),
                rms_error=rmse, total_variation=tv,
                reference_top1_margin=(top[0] - top[1]).item(),
                passed=greedy == token and rmse <= max_rmse and tv <= max_tv)


def checkpoint_signature(model):
    """Hash metadata and record shard sizes/mtimes, without another full weights read."""
    files = ("config.json", "hf_quant_config.json", "model.safetensors.index.json")
    hashes = {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in files}
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = {name: {"size": (model / name).stat().st_size,
                     "mtime_ns": (model / name).stat().st_mtime_ns}
              for name in sorted(set(index["weight_map"].values()))}
    return {"metadata_sha256": hashes, "shards": shards}


def capture_vllm(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, __version__
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [
        ("arithmetic", "What is 2 + 2? Answer briefly."),
        ("counting", "Write the numbers from 1 to 8, separated by commas. No explanation."),
        ("recall", "Remember the code 1739. What is the code? Answer with the code only."),
    ]
    prompts.insert(1, ("arithmetic_repeat", prompts[0][1]))
    filler = "The sky is blue. Water is wet. "
    long_text = "Remember the code 1739.\n"
    while len(tokenizer.encode(long_text, add_special_tokens=False)) < args.long_prompt_tokens:
        long_text += filler
    prompts += [("long_recall", long_text + "\nWhat is the code? Answer with the code only."),
                ("arithmetic_after_reset", prompts[0][1])]
    cases = [dict(name=name, prompt=text, prompt_ids=tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False)) for name, text in prompts]
    capacity = max(512, max(len(case["prompt_ids"]) for case in cases) + args.tokens - 1)
    settings = dict(
        model=str(args.model), dtype="bfloat16", quantization="modelopt_fp4",
        safetensors_load_strategy="lazy", language_model_only=True,
        mm_processor_cache_gb=0, max_model_len=capacity, max_num_seqs=1,
        max_num_batched_tokens=capacity, gpu_memory_utilization=.85,
        enable_prefix_caching=False, enforce_eager=True, async_scheduling=False, seed=0,
        kv_cache_dtype="auto", mamba_ssm_cache_dtype="float32",
        attention_backend="FLASHINFER",
        worker_extension_cls="benchmark.offline.parity.CaptureWorker",
        kernel_config={"enable_jit_warmup": False, "enable_cutedsl_warmup": False},
    )
    generation = json.loads((args.model / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos_ids = eos if isinstance(eos, list) else [eos]
    signature = checkpoint_signature(args.model)
    llm = LLM(**settings)
    params = SamplingParams(temperature=0, max_tokens=args.tokens, ignore_eos=True)
    for case in cases:
        llm.generate([TokensPrompt(prompt_token_ids=case["prompt_ids"])], params, use_tqdm=False)
    print("All capture workloads warmed once; starting measured references.", flush=True)
    vocab_size = json.loads((args.model / "config.json").read_text())["text_config"]["vocab_size"]
    with TemporaryDirectory(prefix="quickdraw-parity-") as directory:
        inspection_path = str(Path(directory) / "inspection.pt")
        llm.collective_rpc("qd_install", args=(inspection_path,))
        inspection = torch.load(inspection_path, weights_only=True)
        for case in cases:
            llm.collective_rpc("qd_clear")
            result = llm.generate([TokensPrompt(prompt_token_ids=case["prompt_ids"])],
                                  params, use_tqdm=False)[0].outputs[0]
            case_path = str(Path(directory) / "logits.pt")
            llm.collective_rpc("qd_take", args=(case_path,))
            raw = torch.load(case_path, map_location="cpu", weights_only=True)
            output_ids = list(result.token_ids)
            assert len(output_ids) == args.tokens
            logits = validate_capture(raw, output_ids, vocab_size)
            case.update(output_ids=output_ids, text=result.text, logits=logits,
                        discarded_rows=raw.shape[0] - logits.shape[0])
            print(f"Captured {case['name']}: prompt={len(case['prompt_ids'])}, "
                  f"outputs={len(output_ids)}, discarded={case['discarded_rows']}; "
                  f"IDs={output_ids}", flush=True)
    reset_delta = (cases[0]["logits"] - cases[-1]["logits"]).abs().max().item()
    data = dict(engine="vllm", version=str(__version__), torch_version=str(torch.__version__),
                settings=settings, max_context=capacity, model=str(args.model.resolve()),
                gpu=torch.cuda.get_device_name(), checkpoint_signature=signature,
                inspection=inspection, cases=cases, warmup=1, eos_token_ids=eos_ids,
                reference_reset_max_abs=reset_delta,
                reference_repeat_max_abs=(cases[0]["logits"] - cases[1]["logits"]).abs().max().item())
    args.capture.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.capture)
    print(f"Saved raw logits: {args.capture}; reference reset max error={reset_delta:.6f}", flush=True)
    return 0


def compare_quickdraw(args):
    from quickdraw import Engine

    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    assert str(args.model.resolve()) == data["model"]
    assert checkpoint_signature(args.model) == data["checkpoint_signature"], "checkpoint changed"
    engine = Engine.from_pretrained(str(args.model), device="cuda",
                                    max_context=data["max_context"], backend=args.backend)
    report = dict(reference_engine=data["engine"], reference_version=data["version"],
                  model=data["model"], gpu=torch.cuda.get_device_name(), backend=args.backend,
                  torch_version=str(torch.__version__),
                  history="reference token IDs forced; fixed output count, EOS included as input",
                  capture=str(args.capture.resolve()),
                  limits={"max_rmse": args.max_rmse, "max_total_variation": args.max_tv},
                  reference_reset_max_abs=data["reference_reset_max_abs"],
                  reference_repeat_max_abs=data["reference_repeat_max_abs"],
                  reference_repetitions=[], cases=[])
    for repeated in (data["cases"][1], data["cases"][-1]):
        # Compare logits only after identical token histories.
        repeated_rows = []
        for actual, expected, original_id, repeat_id in zip(
                repeated["logits"], data["cases"][0]["logits"],
                data["cases"][0]["output_ids"], repeated["output_ids"]):
            repeated_rows.append(logit_metrics(actual, expected, original_id,
                                               max_rmse=args.max_rmse, max_tv=args.max_tv))
            if original_id != repeat_id:
                break
        report["reference_repetitions"].append(dict(name=repeated["name"], steps=repeated_rows))
    report["projection_layout_differences"] = []
    scales = data["inspection"]["projection_scales"]
    for layer in engine.ckpt.layers:
        if layer.layer_type == "full_attention":
            merged = f"language_model.model.layers.{layer.idx}.self_attn.qkv_proj"
            names = ("q_proj", "k_proj", "v_proj")
        else:
            merged = f"language_model.model.layers.{layer.idx}.linear_attn.in_proj_qkvz"
            names = ("in_proj_qkv", "in_proj_z")
        for name in names:
            projection = getattr(layer.attn, name)
            original_input = projection.input_scale.item()
            original_weight = projection.weight_scale.item()
            if (original_input != scales[merged]["input_scale"]
                    or original_weight != scales[merged]["weight_scale"]):
                report["projection_layout_differences"].append(dict(
                    layer=layer.idx, projection=name,
                    checkpoint_input_scale=original_input,
                    checkpoint_weight_scale=original_weight,
                    vllm_merged_input_scale=scales[merged]["input_scale"],
                    vllm_merged_weight_scale=scales[merged]["weight_scale"]))
    print(f"Separate versus merged FP8 scale differences: "
          f"{len(report['projection_layout_differences'])} projections", flush=True)
    pointers = {name: getattr(engine.state, name).data_ptr()
                for name in ("kv_k", "kv_v", "rec_state", "conv_state")}
    first_logits = None
    with torch.inference_mode():
        for case in data["cases"]:
            started = time.perf_counter()
            logits = engine.prefill(case["prompt_ids"])
            rows, actual_logits = [], []
            after_eos = False
            for step, token in enumerate(case["output_ids"]):
                actual = logits.float().cpu()
                actual_logits.append(actual)
                rows.append(dict(step=step, position=engine.state.position - 1, after_eos=after_eos,
                    **logit_metrics(actual, case["logits"][step], token,
                                    max_rmse=args.max_rmse, max_tv=args.max_tv)))
                print(f"  {case['name']} {step}: target={token} quickdraw={rows[-1]['quickdraw_id']} "
                      f"RMSE={rows[-1]['rms_error']:.4f} TV={rows[-1]['total_variation']:.4f}", flush=True)
                after_eos |= token in data["eos_token_ids"]
                if step + 1 < len(case["output_ids"]):
                    forced = torch.tensor(token, dtype=torch.long, device="cuda")
                    logits = engine.step(engine.ckpt, forced, engine.state.position, engine.state)
                    engine.state.position += 1
            assert engine.state.position == len(case["prompt_ids"]) + len(rows) - 1
            assert all(getattr(engine.state, name).data_ptr() == pointer
                       for name, pointer in pointers.items()), "history buffer replaced"
            actual_logits = torch.stack(actual_logits)
            if first_logits is None:
                first_logits = actual_logits
            if case["name"] == "arithmetic_after_reset":
                assert case["prompt_ids"] == data["cases"][0]["prompt_ids"]
                shared = 1
                for first_id, repeated_id in zip(data["cases"][0]["output_ids"][:-1],
                                                 case["output_ids"][:-1]):
                    if first_id != repeated_id:
                        break
                    shared += 1
                report["quickdraw_reset_compared_positions"] = shared
                report["quickdraw_reset_max_abs"] = (
                    first_logits[:shared] - actual_logits[:shared]).abs().max().item()
            entry = dict(name=case["name"], input_tokens=len(case["prompt_ids"]),
                         reference_text=case["text"], steps=rows, position=engine.state.position,
                         seconds=time.perf_counter() - started)
            report["cases"].append(entry)
            matches = sum(row["top1_match"] for row in rows)
            print(f"{case['name']}: {matches}/{len(rows)} top-1; "
                  f"max RMSE={max(row['rms_error'] for row in rows):.4f}; "
                  f"max TV={max(row['total_variation'] for row in rows):.4f}", flush=True)
    report["positions_checked"] = sum(len(case["steps"]) for case in report["cases"])
    report["positions_through_eos"] = sum(not row["after_eos"]
                                           for case in report["cases"] for row in case["steps"])
    report["all_top1_match"] = all(row["top1_match"] for case in report["cases"] for row in case["steps"])
    report["reference_repeat_within_limits"] = all(
        row["passed"] for repetition in report["reference_repetitions"] for row in repetition["steps"])
    report["numerical_limits_met"] = all(
        row["passed"] for case in report["cases"] for row in case["steps"])
    report["passed"] = (report["numerical_limits_met"]
                        and report["quickdraw_reset_max_abs"] == 0
                        and report["reference_repeat_within_limits"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Greedy: {sum(row['top1_match'] for case in report['cases'] for row in case['steps'])}"
          f"/{report['positions_checked']} positions; "
          f"{report['positions_through_eos']} through EOS", flush=True)
    print(f"vLLM repeat max errors: immediate={report['reference_repeat_max_abs']:.6f}, "
          f"after long prompt={report['reference_reset_max_abs']:.6f}", flush=True)
    print(f"{'PASS' if report['passed'] else 'FAIL'} parity; "
          f"quickdraw reset error={report['quickdraw_reset_max_abs']:.6f}; "
          f"report={args.report}", flush=True)
    return int(not report["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["capture", "compare"])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--report", type=Path, default=Path("benchmark/parity/vllm.json"))
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--long-prompt-tokens", type=int, default=256)
    parser.add_argument("--backend", choices=["reference", "triton"], default="triton")
    parser.add_argument("--max-rmse", type=float, default=.25)
    parser.add_argument("--max-tv", type=float, default=.05)
    args = parser.parse_args()
    assert args.tokens > 0 and args.long_prompt_tokens > 0
    assert args.max_rmse >= 0 and 0 <= args.max_tv <= 1
    assert torch.cuda.is_available(), "CUDA is required"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return capture_vllm(args) if args.stage == "capture" else compare_quickdraw(args)


if __name__ == "__main__":
    raise SystemExit(main())
