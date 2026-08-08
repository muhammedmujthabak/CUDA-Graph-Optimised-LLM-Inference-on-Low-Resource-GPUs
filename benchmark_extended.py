import torch
import time
import gc
from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache

N_STEPS = 50
SEQ_LENGTHS = [32, 64, 128, 256]
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MAX_CACHE_LEN = 512

# -----------------------------------------------------------------------------
# Core Benchmark Functions
# -----------------------------------------------------------------------------
def benchmark_eager(model, initial_input_ids):
    slen = initial_input_ids.shape[1]
    
    static_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN, device=model.device, dtype=model.dtype)

    eager_position_ids = torch.arange(0, slen, dtype=torch.long, device="cuda").unsqueeze(0)
    eager_cache_pos = torch.arange(0, slen, dtype=torch.long, device="cuda")
    
    with torch.no_grad():
        out = model(
            input_ids=initial_input_ids, position_ids=eager_position_ids,
            cache_position=eager_cache_pos, past_key_values=static_cache, return_dict=True
        )
    
    decode_input_ids = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
    decode_pos_ids = torch.tensor([[slen]], dtype=torch.long, device="cuda")
    decode_cache_pos = torch.tensor([slen], dtype=torch.long, device="cuda")
    
    logits_history = []
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(N_STEPS):
            out = model(
                input_ids=decode_input_ids, position_ids=decode_pos_ids,
                cache_position=decode_cache_pos, past_key_values=static_cache, return_dict=True
            )
            logits_history.append(out.logits.clone())
            
            decode_input_ids = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
            decode_pos_ids.add_(1)
            decode_cache_pos.add_(1)
            
    torch.cuda.synchronize()
    latency = ((time.perf_counter() - start_time) / N_STEPS) * 1000
    
    return latency, logits_history

def benchmark_graph(model, initial_input_ids):
    slen = initial_input_ids.shape[1]
    graph_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN, device=model.device, dtype=model.dtype)

    graph_position_ids = torch.arange(0, slen, dtype=torch.long, device="cuda").unsqueeze(0)
    graph_cache_pos = torch.arange(0, slen, dtype=torch.long, device="cuda")

    with torch.no_grad():
        out = model(
            input_ids=initial_input_ids, position_ids=graph_position_ids,
            cache_position=graph_cache_pos, past_key_values=graph_cache, return_dict=True
        )

    next_val = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
    
    # Allocate pure static GPU bounds
    static_input_ids = torch.zeros((1, 1), device="cuda", dtype=torch.long)
    static_position_ids = torch.zeros((1, 1), device="cuda", dtype=torch.long)
    static_cache_position = torch.zeros((1,), device="cuda", dtype=torch.long)
    
    # Establish structural tracking (No python CPU scalars beyond this)
    static_input_ids.copy_(next_val)
    static_position_ids.fill_(slen)
    static_cache_position.fill_(slen)

    # Ensure all allocations and fills finish before warmup kernel reads them
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            with torch.no_grad():
                _ = model(
                    input_ids=static_input_ids, position_ids=static_position_ids,
                    cache_position=static_cache_position, past_key_values=graph_cache, return_dict=True
                )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    
    torch.cuda.empty_cache()
    
    mempool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=mempool):
        with torch.no_grad():
            static_out = model(
                input_ids=static_input_ids, position_ids=static_position_ids,
                cache_position=static_cache_position, past_key_values=graph_cache, return_dict=True
            )
            
    logits_history = []
    initial_pos_check = static_position_ids.clone()
            
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(N_STEPS):
        g.replay()
        logits_history.append(static_out.logits.clone())
        
        new_val = torch.argmax(static_out.logits[:, -1, :], dim=-1).unsqueeze(0)
        
        # IN-PLACE Updates exclusively
        static_input_ids.copy_(new_val)
        static_position_ids.add_(1)
        static_cache_position.add_(1)
        
    torch.cuda.synchronize()
    latency = ((time.perf_counter() - start_time) / N_STEPS) * 1000
    
    final_pos_check = static_position_ids.clone()
    
    return latency, logits_history, initial_pos_check, final_pos_check

# -----------------------------------------------------------------------------
# A) FP16 Mode (CUDA Graph Validation)
# -----------------------------------------------------------------------------
def run_fp16_mode():
    print("\n" + "="*60)
    print("=== A) FP16 Mode (CUDA Graph Validation) ===")
    print("="*60)
    
    print("Loading model WITHOUT bitsandbytes (Standard FP16)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    
    for slen in SEQ_LENGTHS:
        print(f"\n--- Sequence Length: {slen} ---")
        common_input = torch.randint(10, 1000, (1, slen), device="cuda")
        
        eager_lat, eager_logits = benchmark_eager(model, common_input)
        graph_lat, graph_logits, init_pos, final_pos = benchmark_graph(model, common_input)
        
        print(f"Debug [Initialization] - position_ids before replay: {init_pos.item()}")
        print(f"Debug [Progression]    - position_ids safely incremented locally to: {final_pos.item()}")
        
        max_diffs = [torch.max(torch.abs(el - gl)).item() for el, gl in zip(eager_logits, graph_logits)]
        overall_max_diff = max(max_diffs)
        
        print(f"FP16 Eager Latency : {eager_lat:.2f} ms/token")
        print(f"FP16 Graph Latency : {graph_lat:.2f} ms/token")
        print(f"Speedup            : {eager_lat / graph_lat:.2f}x")
        
        print(f"Max Logit Diff     : {overall_max_diff:.8f}")
        
        if overall_max_diff < 1e-2:
            print("Graph correctness  : PASS (Valid mathematical tracing)")
        else:
            print("WARNING: Graph still diverging due to dynamic masking limitations")
            print("Graph correctness  : FAIL")
            
    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)

if __name__ == "__main__":
    run_fp16_mode()

