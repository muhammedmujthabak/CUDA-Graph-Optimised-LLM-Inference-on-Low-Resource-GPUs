"""
Isolate graph capture diff — matches validate.py phase structure exactly:
  prefill -> warmup (eager, pos=prompt_len) -> capture (graph, pos=prompt_len+1)
No extra forward at capture position before graph capture.
"""
from __future__ import annotations

import functools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers import masking_utils

from test_graph_performance import build_chat_inputs

MODEL_ID = "Qwen/Qwen2.5-0.5B"
PROMPT = "Explain simulated PMPD decode with CUDA Graph in 3 lines."
KV_MAX = 256


def load_model(attn_implementation=None):
    kwargs = dict(dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return model, tokenizer


def make_mask(device, dtype):
    return torch.full((1, 1, 1, KV_MAX), torch.finfo(dtype).min, dtype=dtype, device=device)


def update_mask(m, pos):
    m.fill_(torch.finfo(m.dtype).min)
    m[..., : pos + 1] = 0.0


def fresh_cache(model, device):
    cache = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=KV_MAX,
        device=device, dtype=torch.float16,
    )
    for layer in cache.layers:
        layer.cumulative_length = layer.cumulative_length.to(device)
    return cache


def run_through_warmup(model, input_ids, device, dtype, *, mask_dict=False, static_position_ids=None):
    """Returns state after prefill + warmup; ready for capture at pos with token t1."""
    cache = fresh_cache(model, device)
    pre = model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
    t0 = torch.argmax(pre.logits[:, -1:, :], dim=-1)
    pos = int(input_ids.shape[1])

    static_token = torch.zeros((1, 1), dtype=torch.long, device=device)
    static_pos = torch.zeros((1,), dtype=torch.long, device=device)
    static_pos_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    static_mask = make_mask(device, dtype)

    static_token.copy_(t0)
    static_pos.fill_(pos)
    static_pos_ids.fill_(pos)
    update_mask(static_mask, pos)

    kw = dict(input_ids=static_token, past_key_values=cache, use_cache=True, return_dict=True)
    if mask_dict:
        kw["attention_mask"] = {"full_attention": static_mask}
    if static_position_ids is not None:
        kw["position_ids"] = static_pos_ids

    warmup_out = model(**kw)
    t1 = torch.argmax(warmup_out.logits[:, -1:, :], dim=-1)
    pos += 1

    return {
        "cache": cache,
        "t1": t1,
        "pos": pos,
        "static_token": static_token,
        "static_pos": static_pos,
        "static_pos_ids": static_pos_ids,
        "static_mask": static_mask,
    }


def eager_capture_forward(model, st):
    """Single eager forward at capture position (validate.py capture-equiv)."""
    pos = st["pos"]
    update_mask(st["static_mask"], pos)
    st["static_token"].copy_(st["t1"])
    st["static_pos"].fill_(pos)
    st["static_pos_ids"].fill_(pos)
    kw = dict(
        input_ids=st["static_token"],
        past_key_values=st["cache"],
        use_cache=True,
        return_dict=True,
        attention_mask={"full_attention": st["static_mask"]},
    )
    kw["position_ids"] = st["static_pos_ids"]
    return model(**kw).logits.detach()


def graph_capture_forward(model, st, *, patch_fn=None):
    """Graph capture at capture position — NO extra forward at capture pos before capture."""
    pos = st["pos"]
    st["static_token"].copy_(st["t1"])
    st["static_pos"].fill_(pos)
    st["static_pos_ids"].fill_(pos)
    update_mask(st["static_mask"], pos)

    kw = dict(
        input_ids=st["static_token"],
        past_key_values=st["cache"],
        cache_position=st["static_pos"],
        attention_mask={"full_attention": st["static_mask"]},
        position_ids=st["static_pos_ids"],
        use_cache=True,
        return_dict=True,
    )

    if patch_fn:
        patch_fn()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = model(**kw).logits
    torch.cuda.synchronize(st["static_token"].device)
    return logits.detach()


def measure_capture_diff(model, input_ids, device, dtype, label, **run_kw):
    st_e = run_through_warmup(model, input_ids, device, dtype, mask_dict=True, static_position_ids=True)
    st_g = run_through_warmup(model, input_ids, device, dtype, mask_dict=True, static_position_ids=True)
    el = eager_capture_forward(model, st_e)
    gl = graph_capture_forward(model, st_g, **run_kw)
    d = torch.max(torch.abs(el - gl)).item()
    print(f"  [{label}] capture max logit diff = {d:.5f}")
    return d


def main():
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    input_ids, _ = build_chat_inputs(tokenizer, PROMPT, device)

    print("=== create_causal_mask bypass verification ===")
    calls = []
    orig = masking_utils.create_causal_mask

    @functools.wraps(orig)
    def hook(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    masking_utils.create_causal_mask = hook
    st = run_through_warmup(model, input_ids, device, dtype, mask_dict=True)
    calls.clear()
    _ = eager_capture_forward(model, st)
    print(f"  eager capture forward: create_causal_mask calls = {len(calls)}")
    calls.clear()
    st2 = run_through_warmup(model, input_ids, device, dtype, mask_dict=True)
    _ = graph_capture_forward(model, st2)
    print(f"  graph capture forward: create_causal_mask calls = {len(calls)}")
    masking_utils.create_causal_mask = orig

    print("\n=== BASELINE (validate.py-matched flow, 4D dict mask) ===")
    baseline = measure_capture_diff(model, input_ids, device, dtype, "baseline")

    print("\n=== Per-op isolation at capture ===")

    # 1) position_ids from get_seq_length vs static
    d1 = measure_capture_diff(model, input_ids, device, dtype, "static position_ids (already used in baseline)")

    # 2) KV index_copy_: no-op update ONLY during graph capture stream
    from transformers.cache_utils import StaticLayer
    orig_update = StaticLayer.update
    in_graph = {"v": False}

    def noop_update(self, ks, vs, *a, **k):
        if in_graph["v"]:
            return self.keys, self.values
        return orig_update(self, ks, vs, *a, **k)

    StaticLayer.update = noop_update
    try:
        st_e = run_through_warmup(model, input_ids, device, dtype, mask_dict=True, static_position_ids=True)
        st_g = run_through_warmup(model, input_ids, device, dtype, mask_dict=True, static_position_ids=True)
        el = eager_capture_forward(model, st_e)
        pos = st_g["pos"]
        st_g["static_token"].copy_(st_g["t1"])
        st_g["static_pos"].fill_(pos)
        st_g["static_pos_ids"].fill_(pos)
        update_mask(st_g["static_mask"], pos)
        kw = dict(
            input_ids=st_g["static_token"], past_key_values=st_g["cache"],
            cache_position=st_g["static_pos"],
            attention_mask={"full_attention": st_g["static_mask"]},
            position_ids=st_g["static_pos_ids"],
            use_cache=True, return_dict=True,
        )
        graph = torch.cuda.CUDAGraph()
        in_graph["v"] = True
        with torch.cuda.graph(graph):
            gl = model(**kw).logits
        in_graph["v"] = False
        torch.cuda.synchronize(device)
        d2 = torch.max(torch.abs(el - gl.detach())).item()
        print(f"  [KV index_copy_ no-op in graph] capture diff = {d2:.5f}")
    finally:
        in_graph["v"] = False
        StaticLayer.update = orig_update

    # 3) Skip embed_tokens — use inputs_embeds
    print("\n  [inputs_embeds path]")
    st_e = run_through_warmup(model, input_ids, device, dtype, mask_dict=True, static_position_ids=True)
    st_g = run_through_warmup(model, input_ids, device, dtype, mask_dict=True, static_position_ids=True)
    pos = st_e["pos"]
    embeds = model.model.embed_tokens(st_e["t1"])
    static_embeds = embeds.clone()
    update_mask(st_e["static_mask"], pos)
    update_mask(st_g["static_mask"], pos)
    st_e["static_pos_ids"].fill_(pos)
    st_g["static_pos_ids"].fill_(pos)
    el = model(
        inputs_embeds=embeds, past_key_values=st_e["cache"],
        position_ids=st_e["static_pos_ids"],
        attention_mask={"full_attention": st_e["static_mask"]},
        use_cache=True, return_dict=True,
    ).logits.detach()
    kw = dict(
        inputs_embeds=static_embeds, past_key_values=st_g["cache"],
        position_ids=st_g["static_pos_ids"],
        attention_mask={"full_attention": st_g["static_mask"]},
        use_cache=True, return_dict=True,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gl = model(**kw).logits
    torch.cuda.synchronize(device)
    d3 = torch.max(torch.abs(el - gl)).item()
    print(f"  [inputs_embeds + static position_ids] capture diff = {d3:.5f}")

    # 4) cumulative_length .item() hook
    print("\n  [.item() on cumulative_length]")
    item_calls = []
    st = run_through_warmup(model, input_ids, device, dtype, mask_dict=True)
    cl = st["cache"].layers[0].cumulative_length
    orig_item = cl.item

    def item_hook(*a, **k):
        item_calls.append(1)
        return orig_item(*a, **k)

    cl.item = item_hook
    item_calls.clear()
    _ = graph_capture_forward(model, st)
    print(f"  cumulative_length.item() during graph capture: {len(item_calls)} calls")

    # 5) SDPA vs eager attention
    print("\n=== Attention implementation ===")
    for attn in ["sdpa", "eager"]:
        try:
            m2, _ = load_model(attn_implementation=attn)
            d = measure_capture_diff(m2, input_ids, device, dtype, f"attn={attn}")
            del m2
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [attn={attn}] ERROR: {e}")

    # 6) Layer 0 only vs full model — hook after layer 0
    print("\n=== Layer-0 hidden-state divergence ===")
    st_e = run_through_warmup(model, input_ids, device, dtype, mask_dict=True, static_position_ids=True)
    st_g = run_through_warmup(model, input_ids, device, dtype, mask_dict=True, static_position_ids=True)
    hidden_e = {}

    def hook_e(mod, inp, out):
        hidden_e["h"] = out[0].detach().clone()

    h = model.model.layers[0].register_forward_hook(hook_e)
    _ = eager_capture_forward(model, st_e)
    h.remove()
    hidden_g = {}

    def hook_g(mod, inp, out):
        hidden_g["h"] = out[0].detach().clone()

    h2 = model.model.layers[0].register_forward_hook(hook_g)
    _ = graph_capture_forward(model, st_g)
    h2.remove()
    d_h0 = torch.max(torch.abs(hidden_e["h"] - hidden_g["h"])).item()
    print(f"  layer-0 output max diff (eager vs graph capture): {d_h0:.5f}")

    print("\n=== SUMMARY ===")
    print(f"  baseline capture diff:     {baseline:.5f}")
    print(f"  KV no-op in graph:         {d2:.5f}")
    print(f"  inputs_embeds path:        {d3:.5f}")
    print(f"  layer-0 hidden diff:       {d_h0:.5f}")


if __name__ == "__main__":
    main()
