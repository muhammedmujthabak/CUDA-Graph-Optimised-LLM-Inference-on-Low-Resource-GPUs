import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()

prompt = "<|system|>\nYou are a friendly assistant.</s>\n<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
seq_len = inputs.input_ids.shape[1]
MAX_CACHE_LEN = 128
N_STEPS = 10

def generate_eager():
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN, device=model.device, dtype=torch.float16)
    pos_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    cache_pos = torch.arange(0, seq_len, dtype=torch.long, device="cuda")
    
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, position_ids=pos_ids, cache_position=cache_pos, past_key_values=cache, use_cache=True, return_dict=True)
    
    dec_input = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
    dec_pos = torch.tensor([[seq_len]], dtype=torch.long, device="cuda")
    dec_cache = torch.tensor([seq_len], dtype=torch.long, device="cuda")
    
    generated = [dec_input.item()]
    with torch.no_grad():
        for _ in range(N_STEPS - 1):
            out = model(input_ids=dec_input, position_ids=dec_pos, cache_position=dec_cache, past_key_values=cache, use_cache=True, return_dict=True)
            dec_input = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
            dec_pos.add_(1)
            dec_cache.add_(1)
            generated.append(dec_input.item())
    return generated

def generate_graph():
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN, device=model.device, dtype=torch.float16)
    pos_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    cache_pos = torch.arange(0, seq_len, dtype=torch.long, device="cuda")
    
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, position_ids=pos_ids, cache_position=cache_pos, past_key_values=cache, use_cache=True, return_dict=True)
        
    dec_input = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
    dec_pos = torch.tensor([[seq_len]], dtype=torch.long, device="cuda")
    dec_cache = torch.tensor([seq_len], dtype=torch.long, device="cuda")
    
    g_in = dec_input.clone()
    g_pos = dec_pos.clone()
    g_cache = dec_cache.clone()
    
    warmup_stream = torch.cuda.Stream()
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            with torch.no_grad():
                model(input_ids=g_in, position_ids=g_pos, cache_position=g_cache, past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            static_out = model(input_ids=g_in, position_ids=g_pos, cache_position=g_cache, past_key_values=cache, use_cache=True, return_dict=True)
            
    generated = [dec_input.item()]
    for _ in range(N_STEPS - 1):
        g.replay()
        next_tok = torch.argmax(static_out.logits[:, -1, :], dim=-1).unsqueeze(0)
        g_in.copy_(next_tok)
        g_pos.add_(1)
        g_cache.add_(1)
        generated.append(next_tok.item())
    return generated

e_toks = generate_eager()
g_toks = generate_graph()

print("Eager generated:", tokenizer.decode(e_toks))
print("Graph generated:", tokenizer.decode(g_toks))
