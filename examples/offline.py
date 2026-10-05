"""Generate text from a local checkpoint with Quickdraw."""

import argparse
import json
from pathlib import Path
import time

import torch

from quickdraw import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path.home() / "data/models/vllm/Qwen3.6-35B-A3B-NVFP4")
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--backend", choices=["reference", "triton"], default="reference")
    parser.add_argument("--max-context", type=int,
                        help="default: enough space for this prompt and output")
    parser.add_argument("--raw", action="store_true", help="omit the tokenizer's chat template")
    parser.add_argument("--report", type=Path, help="write output IDs and measured timings as JSON")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.max_context is not None and args.max_context <= 0:
        parser.error("--max-context must be positive")
    if not (args.model / "config.json").is_file():
        parser.error(f"local model config does not exist: {args.model / 'config.json'}")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Check nvidia-smi and the GPU driver before running inference.")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        parser.error("tokenization requires Transformers; run with .venv-vllm/bin/python")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.raw:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    else:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if not prompt_ids:
        parser.error("the encoded prompt is empty")
    needed = len(prompt_ids) + args.max_tokens - 1
    capacity = args.max_context if args.max_context is not None else needed
    if capacity < needed:
        parser.error(f"this prompt and output require --max-context >= {needed}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
    print("Loading packed checkpoint; reference weights are decoded on demand.", flush=True)
    load_started = time.perf_counter()
    engine = Engine.from_pretrained(str(args.model), device="cuda", max_context=capacity,
                                    backend=args.backend)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    print(f"Loaded in {load_seconds:.2f}s; prefill {len(prompt_ids)} prompt tokens.", flush=True)
    generation = json.loads((args.model / "generation_config.json").read_text())
    eos = generation.get("eos_token_id", tokenizer.eos_token_id)
    engine.eos_token_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))

    with torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits = engine.prefill(prompt_ids)
        if logits.shape != (engine.ckpt.config.vocab_size,) or not torch.isfinite(logits).all().item():
            raise RuntimeError("prefill produced invalid logits")
        engine.pending_token = torch.argmax(logits)
        output = [int(engine.pending_token.item())]
        first_output = time.perf_counter()
        while len(output) < args.max_tokens and output[-1] not in engine.eos_token_ids:
            output.append(int(engine.decode_step().item()))
        torch.cuda.synchronize()
        finished = time.perf_counter()

    text = tokenizer.decode(output, skip_special_tokens=True)
    decode_seconds = finished - first_output
    report = dict(
        model=str(args.model.resolve()), implementation=f"quickdraw {args.backend} W4A16/static FP8",
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        input_tokens=len(prompt_ids), output_tokens=len(output), output_ids=output,
        text=text, processed_tokens=engine.state.position,
        pending_token=int(engine.pending_token.item()), load_seconds=load_seconds,
        ttft_ms=(first_output - started) * 1000, generation_seconds=finished - started,
        decode_tokens_per_second=((len(output) - 1) / decode_seconds if len(output) > 1 else None),
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_vram_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    print(text, flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
