#!/usr/bin/env python3
"""Checkpoint loading and tensor-format checks."""

import argparse
from pathlib import Path

import torch

from quickdraw.models.weights import load_checkpoint, unpack_fp4, ModelConfig

DEFAULT_MODEL = Path.home() / "data/models/vllm/Qwen3.6-35B-A3B-NVFP4"


def stats(name, t):
    tf = t.flatten()[:32768].float()
    print(f"    {name:34s} {str(tuple(t.shape)):22s} {str(t.dtype):16s} "
          f"std {tf.std().item():.5f} mean {tf.mean().item():+.5f} finite {torch.isfinite(tf).all().item()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--full", action="store_true", help="load all 40 layers (needs ~22 GB)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    cfg = ModelConfig.from_json(a.model / "config.json")
    n_lin = sum(1 for t in cfg.layer_types if t == "linear_attention")
    print(f"config: {cfg.num_hidden_layers} layers ({n_lin} GDN + "
          f"{cfg.num_hidden_layers - n_lin} full attn), {cfg.num_experts} experts "
          f"top-{cfg.num_experts_per_tok}, hidden {cfg.hidden_size}, vocab {cfg.vocab_size}")

    layers = None if a.full else [0, 3]
    print(f"loading layers {layers or 'all'} on {a.device} ...")
    ckpt = load_checkpoint(a.model, device=a.device, layers=layers)
    print(f"loaded {len(ckpt.layers)} layers + embed/lm_head")

    gdn = ckpt.layers[0] if ckpt.layers[0].layer_type == "linear_attention" else None
    full = next((l for l in ckpt.layers if l.layer_type == "full_attention"), None)
    assert gdn and full, "expected one GDN and one full-attention layer"

    print("\n-- FP8 projection (GDN in_proj_qkv) --")
    stats("weight (f8e4m3)", gdn.attn.in_proj_qkv.weight)
    dq = gdn.attn.in_proj_qkv.dequantize()
    stats("dequantized", dq)
    assert 0.001 < dq.float().std().item() < 0.5, "FP8 dequant scale looks wrong"

    print("\n-- NVFP4 (layer 0, expert 0 gate_proj slice) --")
    eg = ckpt.layers[0].moe.experts_gate
    stats("stacked packed weight", eg.weight)
    assert eg.weight.shape[0] == cfg.num_experts
    one = type(eg)(
        weight=eg.weight[0], weight_scale=eg.weight_scale[0],
        weight_scale_2=eg.weight_scale_2[0], input_scale=eg.input_scale[0],
    )
    dq4 = one.dequantize()
    stats("expert0 gate dequantized", dq4)
    assert 0.001 < dq4.float().std().item() < 0.5, "NVFP4 dequant scale looks wrong"
    assert dq4.shape == (cfg.moe_intermediate_size, cfg.hidden_size)

    print("\n-- unpack_fp4 nibble order probe --")
    probe = unpack_fp4(torch.tensor([0x21], device=a.device, dtype=torch.uint8))
    print(f"    0x21 -> {probe.flatten().tolist()}  (expect [0.5, 1.0]: low nibble 1->0.5 is even)")
    assert probe.flatten().tolist() == [0.5, 1.0]

    print("\n-- lm_head --")
    assert ckpt.lm_head.out_features == cfg.vocab_size
    assert ckpt.lm_head.in_features == cfg.hidden_size
    head_slice = type(ckpt.lm_head)(ckpt.lm_head.weight[:64],
                                   ckpt.lm_head.weight_scale[:64],
                                   ckpt.lm_head.weight_scale_2,
                                   group_size=ckpt.lm_head.group_size)
    stats("lm_head first 64 rows", head_slice.dequantize())

    print("\n-- byte accounting (this load) --")
    nbytes = 0
    for l in ckpt.layers:
        for proj in (l.moe.experts_gate, l.moe.experts_up, l.moe.experts_down):
            nbytes += proj.weight.nbytes + proj.weight_scale.nbytes + proj.weight_scale_2.nbytes
        for proj in (l.moe.shared_gate, l.moe.shared_up, l.moe.shared_down):
            nbytes += proj.weight.nbytes + proj.weight_scale.nbytes
    nbytes += ckpt.lm_head.weight.nbytes + ckpt.lm_head.weight_scale.nbytes
    print(f"    NVFP4 bytes on GPU for {len(ckpt.layers)} layers + lm_head: {nbytes/2**30:.2f} GiB")

    print("\nOK: loader sane. Run mathcheck --strict for the reference math; "
          "full-checkpoint engine parity needs separate verification.")


if __name__ == "__main__":
    main()
