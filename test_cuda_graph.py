import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    print("[START] Qwen2.5-0.5B CUDA Graph benchmark", flush=True)
    # Measures forward-pass latency, not full autoregressive decoding.
    print("NOTE: This benchmark uses fixed input (not real decoding)", flush=True)
    print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a CUDA-capable GPU.")

    device = torch.device("cuda")
    model_id = "Qwen/Qwen2.5-0.5B"
    n_steps = 20

    print("[MODEL] Loading fp16 model...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    print("[MODEL] Loaded and set to eval()", flush=True)

    prompt = tok("What is the capital of France?", return_tensors="pt").input_ids.to(device)

    print("[WARMUP] Running eager warmup...", flush=True)
    with torch.inference_mode():
        for _ in range(3):
            model(prompt, use_cache=False)

    print("[EAGER] Timing eager execution...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_steps):
            model(prompt, use_cache=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    eager_ms = (t1 - t0) * 1000.0 / n_steps
    print(f"eager latency: {eager_ms:.2f} ms/step", flush=True)

    print("[GRAPH] Preparing static input and warmup before capture...", flush=True)
    static_input = prompt.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()

    with torch.inference_mode():
        with torch.cuda.stream(stream):
            for _ in range(3):
                model(static_input, use_cache=False)
        torch.cuda.current_stream().wait_stream(stream)

        print("[GRAPH] Capturing CUDA graph...", flush=True)
        with torch.cuda.graph(graph):
            model(static_input, use_cache=False)

    print("[GRAPH] Timing graph replay...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_steps):
            graph.replay()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    graph_ms = (t1 - t0) * 1000.0 / n_steps
    print(f"graph latency: {graph_ms:.2f} ms/step", flush=True)

    print(f"speedup factor: {eager_ms / graph_ms:.2f}x", flush=True)


if __name__ == "__main__":
    main()
