"""CPU loading and generation checks with synthetic weights and HF references."""

import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from quickdraw.models import weights as checkpoint, qwen as model
from tests.models import referencegen as ref
from quickdraw.engine import Engine
from tests.models.mathcheck import compare_tree


def verify_chunking():
    torch.manual_seed(501)
    proj = ref.random_fp4(19, 32)
    x = ref.random_bf16(32)
    expected = F.linear(x.unsqueeze(0), ref.decode_nvfp4(proj)).squeeze(0)
    for chunk in (1, 3, 8, 1024):
        compare_tree(model.nvfp4_gemv(x, proj, row_chunk=chunk), expected, 0, 0)
    stacked = ref.random_fp4(19, 32, experts=3)
    compare_tree(stacked.dequantize(), ref.decode_nvfp4(stacked), 0, 0)
    for expert in range(3):
        compare_tree(model._expert_projection(stacked, expert).dequantize(),
                     ref.decode_nvfp4(stacked)[expert], 0, 0)
    print('PASS chunked NVFP4 projection: four chunk sizes and per-expert scales')


def tensor_map(ckpt):
    p='model.language_model.'
    ts={p+'embed_tokens.weight':ckpt.embed_tokens,p+'norm.weight':ckpt.final_norm}
    def proj(prefix, value):
        for field in ('weight','weight_scale','weight_scale_2','input_scale'):
            if hasattr(value,field) and getattr(value,field) is not None:
                ts[prefix+'.'+field] = getattr(value,field).contiguous().clone()
    proj('lm_head',ckpt.lm_head)
    for l in ckpt.layers:
        lp=p+f'layers.{l.idx}.'
        ts[lp+'input_layernorm.weight']=l.input_layernorm
        ts[lp+'post_attention_layernorm.weight']=l.post_attention_layernorm
        if l.layer_type=='full_attention':
            for name in ('q_proj','k_proj','v_proj','o_proj'): proj(lp+'self_attn.'+name,getattr(l.attn,name))
            for name in ('q_norm','k_norm'): ts[lp+'self_attn.'+name+'.weight']=getattr(l.attn,name)
        else:
            for name in ('in_proj_qkv','in_proj_z','out_proj'): proj(lp+'linear_attn.'+name,getattr(l.attn,name))
            for name in ('in_proj_a','in_proj_b','conv1d','norm'): ts[lp+'linear_attn.'+name+'.weight']=getattr(l.attn,name)
            for name in ('A_log','dt_bias'): ts[lp+'linear_attn.'+name]=getattr(l.attn,name)
        ts[lp+'mlp.gate.weight']=l.moe.router
        ts[lp+'mlp.shared_expert_gate.weight']=l.moe.shared_expert_gate
        for name in ('gate','up','down'):
            proj(lp+'mlp.shared_expert.'+name+'_proj',getattr(l.moe,'shared_'+name))
            packed=getattr(l.moe,'experts_'+name)
            for expert in range(ckpt.config.num_experts):
                one=model._expert_projection(packed,expert)
                one.input_scale=packed.input_scale[expert]
                proj(lp+f'mlp.experts.{expert}.{name}_proj',one)
    return ts


def verify_streamed_loader():
    torch.manual_seed(410)
    ckpt=ref.random_checkpoint(ref.small_config())
    ts=tensor_map(ckpt)
    ts['model.visual.ignored.weight']=torch.ones(3)
    ts['mtp.test.weight']=torch.ones(2,dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory() as root:
        root=Path(root); values=asdict(ckpt.config)
        values['rope_parameters']={'rope_theta':values.pop('rope_theta')}
        (root/'config.json').write_text(json.dumps({'text_config':values}))
        groups=[{},{}]; index={}
        for i,(name,tensor) in enumerate(ts.items()):
            shard=f'shard-{i%2}.safetensors'; groups[i%2][name]=tensor; index[name]=shard
        for i, group in enumerate(groups): save_file(group,root/f'shard-{i}.safetensors')
        (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':index}))
        loaded=checkpoint.load_checkpoint(root,device='cpu')
        compare_tree(loaded,ckpt,0,0)
        partial=checkpoint.load_checkpoint(root,device='cpu',layers=[1,2],include_mtp=True)
        compare_tree(partial.layers,ckpt.layers[1:3],0,0)
        compare_tree(partial.mtp,{'mtp.test.weight':torch.ones(2,dtype=torch.bfloat16)},0,0)
        try: model.forward_step(partial,torch.tensor(3),0,Engine(partial,max_context=8,device='cpu').state)
        except AssertionError: pass
        else: raise AssertionError('partial checkpoint must not generate full-model logits')
    print('PASS streamed two-shard loader: all tensors, expert scales, subset, MTP, partial-model rejection')


def verify_generation():
    _, hf=ref.backend()
    for variant in (0,1):
        torch.manual_seed(410+variant)
        ckpt=ref.random_checkpoint(ref.small_config(variant))
        expected_model=ref.text_model(ckpt)
        lm_head=ref.decode_nvfp4(ckpt.lm_head)
        cache=hf.DynamicCache(config=expected_model.config)
        prompt=[3,7,5]
        expected=[]
        pos=0
        for token in prompt:
            hidden=expected_model(input_ids=torch.tensor([[token]]), position_ids=torch.tensor([[pos]]),
                attention_mask={'full_attention':None,'linear_attention':None},
                past_key_values=cache,use_cache=True).last_hidden_state
            logits=F.linear(hidden,lm_head).reshape(-1); pos+=1
        expected.append(int(logits.argmax()))
        for _ in range(4):
            hidden=expected_model(input_ids=torch.tensor([[expected[-1]]]),position_ids=torch.tensor([[pos]]),
                attention_mask={'full_attention':None,'linear_attention':None},
                past_key_values=cache,use_cache=True).last_hidden_state
            expected.append(int(F.linear(hidden,lm_head).reshape(-1).argmax())); pos+=1
        engine=Engine(ckpt,max_context=8,device='cpu')
        actual=engine.generate(prompt,5)
        assert actual==expected,(variant,actual,expected)
        assert engine.state.position==7 and engine.pending_token.item()==actual[-1]
        assert engine.generate(prompt,5)==expected,'reset leaked previous request state'
        calls = {}

        def count(name, function):
            def wrapped(*args, **kwargs):
                calls[name] += 1
                return function(*args, **kwargs)
            calls[name] = 0
            return wrapped

        kernels = {name: count(name, getattr(model, name)) for name in (
            "rms_norm", "fp8_gemv", "nvfp4_gemv", "full_attention_block",
            "gated_delta_net_block", "moe_block")}
        for name in ("full_attention_block", "gated_delta_net_block"):
            kernels[name] = partial(kernels[name], project_fp8=kernels["fp8_gemv"])
        kernels["moe_block"] = partial(kernels["moe_block"], project_fp4=kernels["nvfp4_gemv"])
        custom = Engine(ckpt, max_context=8, device='cpu', kernels=kernels)
        assert custom.generate(prompt, 5) == expected, 'kernel injection changed generated IDs'
        processed = custom.state.position
        assert processed == engine.state.position
        for name in ('kv_k', 'kv_v'):
            compare_tree(getattr(custom.state, name)[:, :processed],
                         getattr(engine.state, name)[:, :processed], 0, 0)
        for name in ('rec_state', 'conv_state'):
            compare_tree(getattr(custom.state, name), getattr(engine.state, name), 0, 0)
        assert calls['full_attention_block'] == processed * ckpt.config.layer_types.count('full_attention')
        assert calls['gated_delta_net_block'] == processed * ckpt.config.layer_types.count('linear_attention')
        assert calls['moe_block'] == processed * ckpt.config.num_hidden_layers
        assert calls['rms_norm'] == processed * (2 * ckpt.config.num_hidden_layers + 1)
        assert calls['nvfp4_gemv'] > 0 and calls['fp8_gemv'] > 0
        engine.reset()
        try: engine.decode_step()
        except AssertionError: pass
        else: raise AssertionError('decode without a pending token must fail')
        print(f'PASS actual model generation, variant {variant}: HF IDs {expected}, position 7, repeat/reset')
        print(f'PASS supplied kernels, variant {variant}: all operation hooks used; identical IDs and history')


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with torch.inference_mode():
        verify_chunking()
        verify_streamed_loader()
        verify_generation()
    print('All integration checks passed on CPU; no full-checkpoint GPU parity claim.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
