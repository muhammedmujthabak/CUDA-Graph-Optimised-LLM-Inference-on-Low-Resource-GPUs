import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
model.eval()

# STEP 2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of india? Short answer only."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Prompt format:", repr(prompt))
print("Input IDs shape:", inputs.input_ids.shape)

temperature = 0.7

print("\n--- EAGER TRACE ---") # Eager baseline with multinomial
input_ids = inputs.input_ids
past_key_values = None
generated = []

torch.cuda.synchronize()
wall_start = time.perf_counter()

# Ensure same seed for deterministic tracing logic test
torch.manual_seed(42)

for step in range(20):
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        
        logits = out.logits[:, -1, :] # shape (batch, vocab)
        
        # apply temperature
        logits = logits / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated.append(next_token.item())
        print(f"Step {step}: token {next_token.item()} '{tokenizer.decode([next_token.item()])}'")
        input_ids = next_token
        
        if next_token.item() == tokenizer.eos_token_id:
            break

torch.cuda.synchronize()
wall_time = time.perf_counter() - wall_start

print("Eager generated:", tokenizer.decode(generated))
print(f"Time: {wall_time:.4f} s")

print("\n--- CUDA GRAPH TRACE ---")

# Setup cache
try:
    cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=128,
        device=model.device,
        dtype=model.dtype,
    )

    # Prefill
    print("Graph Prefill...")
    prefill_out = model(input_ids=inputs.input_ids, past_key_values=cache, use_cache=True)
    logits = prefill_out.logits[:, -1, :] / temperature
    next_token = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), num_samples=1)

    generated_g = [next_token.item()]
    
    pos = inputs.input_ids.shape[1]
    static_token = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    static_pos = torch.zeros((1,), dtype=torch.long, device="cuda")
    
    static_token.copy_(next_token)
    static_pos.fill_(pos)
    
    print("Graph Capture Warmup...")
    _ = model(input_ids=static_token, past_key_values=cache, cache_position=static_pos, use_cache=True)
    torch.cuda.synchronize()
    
    print("Capturing Graph...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = model(input_ids=static_token, past_key_values=cache, cache_position=static_pos, use_cache=True)
        static_logits = graph_out.logits
        
    torch.cuda.synchronize()
    
    print("Graph decode loop...")
    torch.manual_seed(42) # reset seed for comparison
    
    for step in range(19):
        # We need multinomial logic which doesn't sit well inside graph if not captured. 
        # But we only capture the forward pass!
        
        # update token & pos
        static_token.copy_(next_token)
        static_pos.fill_(pos)
        
        # replay
        graph.replay()
        pos += 1
        
        # read logits
        logits = static_logits[:, -1, :] / temperature
        next_token = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), num_samples=1)
        generated_g.append(next_token.item())
        
        if next_token.item() == tokenizer.eos_token_id:
            break

    print("Graph generated:", tokenizer.decode(generated_g))

except Exception as e:
    import traceback
    traceback.print_exc()

