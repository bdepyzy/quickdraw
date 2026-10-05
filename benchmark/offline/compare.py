"""Compare offline BS=1 generation and save samples, medians, and bandwidth estimates."""

import argparse
from datetime import datetime, timezone
import errno
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from random import Random
import re
from statistics import median
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PEAK_SOURCE = "https://images.nvidia.cn/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf"


def summarize(samples):
    """Median each measured quantity; first output belongs to TTFT, not decode."""
    keys = ("ttft_ms", "decode_ms_per_token", "decode_tps", "total_ms")
    return {key: median(sample[key] for sample in samples) for key in keys}


def gaps(actual, baseline):
    """Positive latency overhead and TPS deficit mean Quickdraw is behind."""
    return {"ttft_extra_pct": 100 * (actual["ttft_ms"] / baseline["ttft_ms"] - 1),
            "decode_extra_pct": 100 * (actual["decode_ms_per_token"] / baseline["decode_ms_per_token"] - 1),
            "total_extra_pct": 100 * (actual["total_ms"] / baseline["total_ms"] - 1),
            "decode_tps_deficit_pct": 100 * (1 - actual["decode_tps"] / baseline["decode_tps"])}


def measure(outputs, prompt, count, *, clock=time.perf_counter):
    """Time cumulative IDs at CPU delivery, ignoring duplicate finish notifications."""
    begin = clock()
    first = last = None
    ids = []
    for ids in outputs(prompt):
        if ids and first is None:
            first = clock()
            assert len(ids) == 1, "first stream update must deliver one token for TTFT"
        if len(ids) == count and last is None:
            last = clock()
    assert first is not None and len(ids) == count, "engine did not return the requested output length"
    decode_ms = 1000 * (last - first) / (count - 1)
    return dict(ttft_ms=1000 * (first - begin), decode_ms_per_token=decode_ms,
                decode_tps=1000 / decode_ms, total_ms=1000 * (last - begin), output_ids=list(ids))


def traffic_model(model):
    """Ideal decode traffic: active weights, BF16 KV, FP32 recurrence, BF16 conv.
    Excludes L2 reuse, intermediates, and repacking; not measured DRAM traffic."""
    config = json.loads((model / "config.json").read_text())["text_config"]
    weights = 0
    for shard in sorted(model.glob("model-*.safetensors")):
        with shard.open("rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        for name, tensor in header.items():
            if not (name.startswith("model.language_model.layers.")
                    or name.startswith("model.language_model.norm.")
                    or name.startswith("lm_head.")):
                continue
            if name.endswith("input_scale"):
                continue
            size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
            if re.search(r"\.mlp\.experts\.\d+\.", name):
                size *= config["num_experts_per_tok"] / config["num_experts"]
            weights += size
    weights += config["hidden_size"] * 2  # BF16 embedding row.
    linear = config["layer_types"].count("linear_attention")
    full = config["layer_types"].count("full_attention")
    recurrent = linear * config["linear_num_value_heads"] * config["linear_key_head_dim"] * config["linear_value_head_dim"] * 4
    channels = (2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
                + config["linear_num_value_heads"] * config["linear_value_head_dim"])
    conv = linear * channels * (config["linear_conv_kernel_dim"] - 1) * 2
    kv = full * 2 * config["num_key_value_heads"] * config["head_dim"] * 2
    return dict(active_weight_bytes=weights, recurrent_conv_rw_bytes=2 * (recurrent + conv),
                kv_bytes_per_position=kv,
                assumptions="Stored active weights once, BF16 KV read/write, FP32 recurrent and BF16 conv read/write; no L2 reuse, intermediate traffic, or backend padding/repacking.")


def bandwidth_estimate(traffic, input_len, output_len, decode_ms, peak_gbs):
    # Mean history length across N-1 decode steps, plus one KV write.
    positions = input_len + output_len / 2
    byte_count = (traffic["active_weight_bytes"] + traffic["recurrent_conv_rw_bytes"]
                  + traffic["kv_bytes_per_position"] * (positions + 1))
    ideal_ms = byte_count / (peak_gbs * 1e9) * 1000
    return dict(modeled_bytes_per_decode=byte_count, ideal_bandwidth_ms=ideal_ms,
                bandwidth_sol_estimate_pct=100 * ideal_ms / decode_ms)


def prepare_memory():
    """Reclaim this job's checkpoint file cache outside timing, retaining RAM limits."""
    path = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split("::")[1].lstrip("/")
    assert path.name.startswith("quickdraw-direct-"), "run workers through scripts/direct_run.py"
    current = int((path / "memory.current").read_text())
    # Drop the loader's file cache before timing.
    if current > 4 * 2**30:
        try:
            (path / "memory.reclaim").write_text(f"{current - 4 * 2**30} swappiness=0")
        except OSError as error:
            assert error.errno == errno.EAGAIN, error  # Partial reclaim.
    return dict(bytes=int((path / "memory.current").read_text()),
                high_events=int(dict(line.split() for line in (path / "memory.events").read_text().splitlines())["high"]))


def worker(args, manifest, destination):
    import torch

    name = args.worker
    conditions = manifest["conditions"]
    capacity = conditions["max_context"]
    count = conditions["output_len"]
    started = time.perf_counter()
    if name == "quickdraw":
        from quickdraw import Engine
        settings = dict(max_context=capacity, backend=conditions["quickdraw_backend"])
        engine = Engine.from_pretrained(manifest["model"], **settings)
        engine.eos_token_ids.clear()

        def outputs(prompt):
            engine.pending_token = engine.prefill(prompt).argmax()
            ids = [engine.pending_token.item()]
            yield ids
            for _ in range(count - 1):
                ids.append(engine.decode_step().item())
                yield ids

    elif name == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        settings = dict(
            dtype="bfloat16", quantization="modelopt_fp4", skip_tokenizer_init=True,
            safetensors_load_strategy="lazy", language_model_only=True, mm_processor_cache_gb=0,
            max_model_len=capacity, max_num_seqs=1, max_num_batched_tokens=capacity,
            gpu_memory_utilization=.85, enable_prefix_caching=False, enforce_eager=False,
            async_scheduling=True, seed=conditions["seed"], kv_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32", attention_backend="FLASHINFER",
            compilation_config={"cudagraph_capture_sizes": [1]},
            kernel_config={"enable_jit_warmup": False, "enable_cutedsl_warmup": False})
        engine = LLM(model=manifest["model"], **settings)
        params = SamplingParams(temperature=0, max_tokens=count, ignore_eos=True, detokenize=False)
        request = 0

        def outputs(prompt):
            nonlocal request
            request += 1
            core = engine.llm_engine
            core.add_request(str(request), TokensPrompt(prompt_token_ids=prompt), params)
            while core.has_unfinished_requests():
                for result in core.step():
                    if result.outputs:
                        yield list(result.outputs[0].token_ids)

    else:
        import sglang
        settings = dict(
            dtype="bfloat16", quantization="modelopt_mixed", skip_tokenizer_init=True,
            context_length=capacity, max_running_requests=1, mem_fraction_static=.85,
            disable_radix_cache=True, cuda_graph_max_bs_decode=1, chunked_prefill_size=4096,
            kv_cache_dtype="bf16", mamba_ssm_dtype="float32", random_seed=conditions["seed"],
            moe_runner_backend="marlin", fp4_gemm_runner_backend="marlin",
            stream_interval=1, batch_notify_size=1, incremental_streaming_output=True,
            model_loader_extra_config='{"enable_multithread_load":false,"num_threads":1}',
            json_model_override_args='{"language_model_only":true}')
        engine = sglang.Engine(model_path=manifest["model"], **settings)
        params = dict(temperature=0, max_new_tokens=count, ignore_eos=True)

        def outputs(prompt):
            ids = []
            for chunk in engine.generate(input_ids=prompt, sampling_params=params, stream=True):
                ids.extend(chunk.get("output_ids", []))
                yield ids

    result = dict(engine=name, settings=settings, load_s=time.perf_counter() - started,
                  versions={package: version(package) for package in ("torch", "quickdraw" if name == "quickdraw" else name)},
                  cuda=torch.version.cuda, workloads=[])
    try:
        with torch.inference_mode():
            for prompt in manifest["prompts"]:
                samples = []
                torch.cuda.synchronize()
                prepare_memory()
                before = None
                for run in range(conditions["warmup"] + conditions["runs"]):
                    torch.cuda.synchronize()
                    if run == conditions["warmup"]:
                        before = prepare_memory()
                    sample = measure(outputs, prompt, count)
                    torch.cuda.synchronize()
                    if run < conditions["warmup"]:
                        continue
                    samples.append(sample)
                    print(f"{name}: input={len(prompt)}, run={run - conditions['warmup'] + 1}/{conditions['runs']}, total={samples[-1]['total_ms']:.3f} ms", flush=True)
                after = prepare_memory()
                assert after["high_events"] == before["high_events"], "RAM soft-limit throttling invalidates these samples"
                result["workloads"].append(dict(input_len=len(prompt), samples=samples, median=summarize(samples),
                                               memory_before=before, memory_after=after))
                destination.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if name == "sglang":
            engine.shutdown()


def report(manifest, results):
    conditions = manifest["conditions"]
    lines = [f"BS=1 | {conditions['output_len']} outputs | median of {conditions['runs']} runs | warmup={conditions['warmup']}",
             "Greedy IDs; EOS ignored; no prefix reuse or speculation; BF16 KV; FP32 recurrent state.",
             "Offline first/last ID delivery includes engine scheduling/IPC. Load and warmup excluded.",
             "Columns are independent medians. All individual samples and IDs are saved.",
             f"GPU: {manifest['gpu']['name']} | peak bandwidth: {manifest['peak_bandwidth_gbs']:.0f} GB/s",
             "", f"{'Input':>7}  {'Engine':<12} {'TTFT ms':>11} {'Decode ms/tok':>14} {'Decode tok/s':>14} {'Total ms':>12} {'BW SoL est %':>13}"]
    comparisons = []
    for length in conditions["input_lens"]:
        medians = {}
        for name in ("vllm", "sglang", "quickdraw"):
            if name not in results:
                continue
            result = results[name]
            workload = next(case for case in result["workloads"] if case["input_len"] == length)
            values = workload["median"]
            estimate = bandwidth_estimate(manifest["traffic"], length, conditions["output_len"], values["decode_ms_per_token"], manifest["peak_bandwidth_gbs"])
            workload["bandwidth_estimate"] = estimate
            medians[name] = values
            lines.append(f"{length:7d}  {name:<12} {values['ttft_ms']:11.3f} {values['decode_ms_per_token']:14.3f} {values['decode_tps']:14.2f} {values['total_ms']:12.3f} {estimate['bandwidth_sol_estimate_pct']:13.2f}")
        if "quickdraw" not in medians:
            continue
        for name in ("vllm", "sglang"):
            if name not in medians:
                continue
            gap = gaps(medians["quickdraw"], medians[name])
            comparisons.append(dict(input_len=length, baseline=name, **gap))
            lines.append(f"  Quickdraw vs {name}: TTFT {gap['ttft_extra_pct']:+.1f}% time; decode {gap['decode_extra_pct']:+.1f}% time; total {gap['total_extra_pct']:+.1f}% time; decode TPS deficit {gap['decode_tps_deficit_pct']:+.2f}%")
    lines += ["", "Positive time percentages mean extra time relative to the baseline. Negative means faster.",
              "BW SoL est = modeled decode bytes / (peak bandwidth * measured decode time).",
              "It is an ideal traffic estimate, not measured utilization or NCU kernel SoL.",
              "L2 reuse, backend weight layouts and extra memory passes can change the estimate.",
              "CUDA graphs: enabled for baselines; Quickdraw uses its current backend without graphs.",
              "Returned IDs are saved. These timing runs do not establish numerical parity."]
    return "\n".join(lines) + "\n", comparisons


def saved_baselines(manifest, source=None, *, runs_dir=None):
    """Newest complete baseline matching the checkpoint, GPU, and workload."""
    candidates = [source] if source else sorted((runs_dir or ROOT / "benchmark/runs").glob("compare-*"), reverse=True)
    conditions = {k: v for k, v in manifest["conditions"].items() if k != "quickdraw_backend"}
    for folder in candidates:
        if not all((folder / name).is_file() for name in ("manifest.json", "vllm.json", "sglang.json")):
            continue
        saved = json.loads((folder / "manifest.json").read_text())
        if any(saved[key] != manifest[key] for key in ("checkpoint", "prompts", "gpu")):
            continue
        if {k: v for k, v in saved["conditions"].items() if k != "quickdraw_backend"} != conditions:
            continue
        complete = True
        for name in ("vllm", "sglang"):
            workloads = json.loads((folder / f"{name}.json").read_text())["workloads"]
            complete &= (sorted(case["input_len"] for case in workloads) == sorted(conditions["input_lens"])
                         and all(len(case["samples"]) == conditions["runs"] for case in workloads))
        if complete:
            return folder.resolve()
    raise AssertionError("No complete saved baselines match this checkpoint, GPU, and workload. "
                         "Use the saved workload settings or explicitly pass --refresh-baselines.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / "data/models/vllm/Qwen3.6-35B-A3B-NVFP4")
    parser.add_argument("--input-lens", type=int, nargs="+", default=[128])
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("reference", "triton"), default="triton")
    parser.add_argument("--engines", choices=("quickdraw", "vllm", "sglang"), nargs="+",
                        help="engines to run (default: Quickdraw; with --refresh-baselines: all three)")
    parser.add_argument("--baselines", type=Path, help="choose saved baselines (default: newest complete matching session)")
    parser.add_argument("--refresh-baselines", action="store_true",
                        help="explicitly allow starting vLLM/SGLang to measure new baselines")
    parser.add_argument("--peak-bandwidth-gbs", type=float, default=1792, help="RTX 5090 advertised peak; decimal GB/s")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="rerun selected engines in a matching session, retaining other results")
    parser.add_argument("--report", type=Path, help="print a saved comparison without running inference")
    parser.add_argument("--worker", choices=("quickdraw", "vllm", "sglang"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.report:
        saved = json.loads((args.report / "results.json").read_text())
        text, _ = report(saved["manifest"], saved["engines"])
        print(text, end="")
        return
    if args.worker:
        assert args.worker == "quickdraw" or args.refresh_baselines, "baseline workers require --refresh-baselines"
        manifest = json.loads((args.save_dir / "manifest.json").read_text())
        worker(args, manifest, args.save_dir / f"{args.worker}.json")
        return
    assert not (args.refresh_baselines and args.baselines), "choose saved baselines or a refresh"
    assert args.refresh_baselines or not args.engines or set(args.engines) <= {"quickdraw"}, "starting vLLM/SGLang requires --refresh-baselines"
    args.engines = args.engines or (["vllm", "sglang", "quickdraw"] if args.refresh_baselines else ["quickdraw"])
    assert min(args.input_lens) > 0 and len(set(args.input_lens)) == len(args.input_lens)
    assert args.output_len >= 2, "use at least two outputs to measure decode"
    assert args.runs > 0 and args.warmup >= 0 and args.peak_bandwidth_gbs > 0
    gpu_fields = ("name", "uuid", "driver_version", "memory.total", "power.limit", "clocks.max.memory")
    gpu_line = subprocess.check_output(["nvidia-smi", f"--query-gpu={','.join(gpu_fields)}", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    assert len(gpu_line) == 1, "this comparison expects one GPU"
    gpu = dict(zip(gpu_fields, (value.strip() for value in gpu_line[0].split(","))))
    assert "5090" in gpu["name"] or args.peak_bandwidth_gbs != 1792, "set this GPU's peak bandwidth"
    signature = {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
                 for name in ("config.json", "hf_quant_config.json", "model.safetensors.index.json")}
    shards = {path.name: [path.stat().st_size, path.stat().st_mtime_ns] for path in sorted(args.model.glob("model-*.safetensors"))}
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    rng = Random(args.seed)
    prompts = [[rng.randrange(config["vocab_size"]) for _ in range(length)] for length in args.input_lens]
    conditions = dict(input_lens=args.input_lens, output_len=args.output_len, runs=args.runs, warmup=args.warmup,
                      seed=args.seed, max_context=max(512, max(args.input_lens) + args.output_len - 1),
                      quickdraw_backend=args.backend, batch_size=1, temperature=0, ignore_eos=True,
                      prefix_cache=False, speculative=False, kv_dtype="bfloat16", recurrent_dtype="float32",
                      cpu_cores=2, compiler_workers=1, host_ram_cap_gib=12)
    conditions["reclaim_checkpoint_cache"] = True
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    assert not (args.resume and args.baselines), "choose resume or baseline reuse"
    folder = (args.resume or args.save_dir or ROOT / "benchmark/runs" / f"compare-{stamp}").resolve()
    manifest = dict(created_utc=stamp, model=str(args.model.resolve()), checkpoint=dict(metadata=signature, shards=shards),
                    gpu=gpu, conditions=conditions, prompts=prompts, traffic=traffic_model(args.model),
                    peak_bandwidth_gbs=args.peak_bandwidth_gbs, peak_source=PEAK_SOURCE)
    if not args.refresh_baselines:
        selected = saved_baselines(manifest, args.resume or args.baselines)
        if not args.resume:
            args.baselines = selected
    folder.mkdir(parents=True, exist_ok=bool(args.resume))
    results = {}
    if args.resume:
        previous = json.loads((folder / "manifest.json").read_text())
        for key in ("checkpoint", "conditions", "prompts", "gpu", "peak_bandwidth_gbs"):
            assert previous[key] == manifest[key], f"session {key} differs"
        manifest = previous
        for name in ("vllm", "sglang", "quickdraw"):
            if (folder / f"{name}.json").exists():
                results[name] = json.loads((folder / f"{name}.json").read_text())
        assert args.refresh_baselines or all(name in results for name in ("vllm", "sglang")), "resume needs saved baselines or --refresh-baselines"
    if args.baselines:
        manifest["baseline_session"] = str(args.baselines.resolve())
        for name in ("vllm", "sglang"):
            results[name] = json.loads((args.baselines / f"{name}.json").read_text())
            (folder / f"{name}.json").write_text(json.dumps(results[name], indent=2) + "\n")
        args.engines = ["quickdraw"]
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Session: {folder}", flush=True)
    if args.baselines:
        print(f"Saved baselines: {args.baselines} (vLLM/SGLang will not start)", flush=True)
    commands = json.loads((folder / "commands.json").read_text()) if args.resume else []
    failures = []
    for name in dict.fromkeys(args.engines):
        results.pop(name, None)
        if args.resume:
            for suffix in ("json", "log"):
                existing = folder / f"{name}.{suffix}"
                if existing.exists():
                    existing.rename(folder / f"{name}-{stamp}.{suffix}")
        python = ROOT / (".venv" if name == "quickdraw" else f".venv-{name}") / "bin/python"
        assert python.exists(), f"missing environment: {python}"
        command = [sys.executable, str(ROOT / "scripts/direct_run.py"), "--cache-tag", name,
                   "--", str(python), str(Path(__file__).resolve()), "--worker", name, "--save-dir", str(folder)]
        if args.refresh_baselines:
            command.append("--refresh-baselines")
        commands.append(command)
        (folder / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(f"Running {name}: {args.warmup} warmup, {args.runs} samples per input length; log: {folder / (name + '.log')}", flush=True)
        with (folder / f"{name}.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if completed.returncode:
            failures.append(dict(engine=name, exit_code=completed.returncode))
            print(f"{name} failed ({completed.returncode}); see its saved log", flush=True)
            continue
        results[name] = json.loads((folder / f"{name}.json").read_text())
    text, comparisons = report(manifest, results)
    (folder / "results.json").write_text(json.dumps(dict(manifest=manifest, engines=results, comparisons=comparisons, failures=failures), indent=2) + "\n")
    (folder / "timings.txt").write_text(text)
    print(text, end="")
    print(f"Saved: {folder}")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
