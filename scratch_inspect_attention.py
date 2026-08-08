import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()

seq_len = 32
N_STEPS = 5
MAX_CACHE_LEN = 128

inputs = torch.randint(10, 1000, (1, seq_len), device="cuda")
cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN, device=model.device, dtype=torch.float16)

# We will inject a hook into the first layer's attention to print the shapes of query and key
def hook_fn(module, input, output):
    # input is usually (hidden_states, attention_mask, position_ids, past_key_value, ...)
    # Let's inspect the shapes
    pass

# We can intercept math via monkey patching LlamaAttention
original_forward = model.model.layers[0].self_attn.forward

def patched_forward(*args, **kwargs):
    hidden_states = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
    attention_mask = kwargs.get("attention_mask") if "attention_mask" in kwargs else (args[1] if len(args) > 1 else None)
    position_ids = kwargs.get("position_ids") if "position_ids" in kwargs else (args[2] if len(args) > 2 else None)
    
    print(f"--- Attn Forward ---")
    print(f"hidden_states shape (Q length): {hidden_states.shape}")
    if attention_mask is not None:
        print(f"attention_mask shape: {attention_mask.shape}")
    else:
        print("attention_mask is None")
    
    if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
        kv = kwargs["past_key_value"]
        if hasattr(kv, 'key_cache'):
            # StaticCache
            print(f"KV Cache Key shape: {kv.key_cache[0].shape}")
            print(f"passed cache_position: {kwargs.get('cache_position')}")
        elif isinstance(kv, tuple):
             print(f"KV Cache Key shape: {kv[0].shape}")

    res = original_forward(*args, **kwargs)
    return res

model.model.layers[0].self_attn.forward = patched_forward

print("Prefill:")
pos_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
cache_pos = torch.arange(0, seq_len, dtype=torch.long, device="cuda")
with torch.no_grad():
    out = model(input_ids=inputs, position_ids=pos_ids, cache_position=cache_pos, past_key_values=cache, use_cache=True, return_dict=True)

print("Decode 1:")
dec_in = torch.tensor([[5]], device="cuda")
dec_pos = torch.tensor([[seq_len]], dtype=torch.long, device="cuda")
dec_cache = torch.tensor([seq_len], dtype=torch.long, device="cuda")
with torch.no_grad():
    out = model(input_ids=dec_in, position_ids=dec_pos, cache_position=dec_cache, past_key_values=cache, use_cache=True, return_dict=True)

print("Decode 2:")
dec_in = torch.tensor([[5]], device="cuda")
dec_pos.add_(1)
dec_cache.add_(1)
with torch.no_grad():
    out = model(input_ids=dec_in, position_ids=dec_pos, cache_position=dec_cache, past_key_values=cache, use_cache=True, return_dict=True)

