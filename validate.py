import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from test_graph_performance import build_chat_inputs


def cache_key_fingerprint(cache) -> float:
    """Sum of abs values of layer-0 key cache (diagnostic for KV writes)."""
    keys = cache.layers[0].keys
    if keys is None:
        return 0.0
    return keys.abs().double().sum().item()


def make_static_mask_buffer(max_cache_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Pre-allocated 4D additive mask: (1, 1, 1, max_cache_len)."""
    return torch.full(
        (1, 1, 1, max_cache_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )


def update_static_mask4d(static_mask: torch.Tensor, pos: int) -> None:
    """In-place update: attend positions 0..pos, mask positions > pos."""
    static_mask.fill_(torch.finfo(static_mask.dtype).min)
    static_mask[..., : pos + 1] = 0.0


def validate_graph_correctness(model, tokenizer, prompt, n_steps=20):
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    input_ids, attention_mask = build_chat_inputs(tokenizer, prompt, device)
    prompt_len = int(input_ids.shape[1])

    # ── Eager reference path ────────────────────────────────────────────
    print("Running Eager...")
    out_eager_logits = []

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)

    for _ in range(n_steps + 2):
        out = model(input_ids=next_token, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        out_eager_logits.append(out.logits.detach().clone())
        next_token = torch.argmax(out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)

    # ── Graph path with StaticCache + static 4D additive mask ───────────
    print("Running Graph... (static 4D additive mask dict)")
    out_graph_logits = []

    kv_cache_max = 256
    cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=kv_cache_max,
        device=device,
        dtype=torch.float16,
    )
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = layer.cumulative_length.to(device)

    prefill_out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
    next_token = torch.argmax(prefill_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)

    static_token = torch.zeros((1, 1), dtype=torch.long, device=device)
    static_pos = torch.zeros((1,), dtype=torch.long, device=device)
    static_mask = make_static_mask_buffer(kv_cache_max, model_dtype, device)

    def graph_kwargs(pos: int) -> dict:
        update_static_mask4d(static_mask, pos)
        return {
            "input_ids": static_token,
            "past_key_values": cache,
            "cache_position": static_pos,
            "attention_mask": {"full_attention": static_mask},
            "use_cache": True,
            "return_dict": True,
        }

    pos = prompt_len
    static_token.copy_(next_token)
    static_pos.fill_(pos)

    print(f"[fingerprint] before warmup: {cache_key_fingerprint(cache):.1f}")
    warmup_out = model(**graph_kwargs(pos))
    torch.cuda.synchronize(device)
    print(f"[fingerprint] after warmup:  {cache_key_fingerprint(cache):.1f}")

    next_token = torch.argmax(warmup_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
    pos += 1

    static_token.copy_(next_token)
    static_pos.fill_(pos)
    update_static_mask4d(static_mask, pos)

    capture_kwargs = {
        "input_ids": static_token,
        "past_key_values": cache,
        "cache_position": static_pos,
        "attention_mask": {"full_attention": static_mask},
        "use_cache": True,
        "return_dict": True,
    }
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = model(**capture_kwargs)
        static_logits = graph_out.logits

    next_token = torch.argmax(static_logits[:, -1:, :], dim=-1).to(dtype=torch.long)
    pos += 1

    for step_idx in range(n_steps):
        static_token.copy_(next_token)
        static_pos.fill_(pos)
        update_static_mask4d(static_mask, pos)
        if step_idx == 0:
            print(f"[fingerprint] before 1st replay: {cache_key_fingerprint(cache):.1f}")
        graph.replay()
        if step_idx == 0:
            print(f"[fingerprint] after 1st replay:  {cache_key_fingerprint(cache):.1f}")
        out_graph_logits.append(static_logits.detach().clone())
        next_token = torch.argmax(static_logits[:, -1:, :], dim=-1).to(dtype=torch.long)
        pos += 1

    skip = 2
    print("Step | Max Logit Diff")
    print("---------------------")
    for i in range(n_steps):
        eager_idx = skip + i
        if eager_idx >= len(out_eager_logits):
            break
        diff = torch.max(torch.abs(out_eager_logits[eager_idx] - out_graph_logits[i])).item()
        print(f"{i+1:4d} | {diff:15.5f}")


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-0.5B"
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    prompt = "Explain simulated PMPD decode with CUDA Graph in 3 lines."
    validate_graph_correctness(model, tokenizer, prompt, n_steps=20)
