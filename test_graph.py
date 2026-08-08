"""
test_graph.py — Simulated PMPD Section 5.2 benchmark: eager vs CUDA Graph decode.

Measures:
  ① Per-token wall-time for eager Simulated PMPD (pmpd_generate) vs graph Simulated PMPD (pmpd_generate_fast)
  ② CPU dispatch overhead vs GPU compute time using torch.cuda.Event timers
  ③ Sequence-length sweep at [32, 64, 128, 256]
  ④ Scheduler switch-point sweep varying high_bit_steps at [5, 10, 20, 40]

Four experimental configurations:
  fp16_eager         — full-precision fp16 baseline (no precision switching)
  uniform_fp16_eager — uniform fp16 (no precision switching)
  simulated_pmpd_eager     — Algorithm 1 with NaiveScheduler, eager dispatch
  simulated_pmpd_graph     — Algorithm 1 with NaiveScheduler, CUDA Graph replay

Hardware target: RTX 3050 Laptop 4 GB VRAM, Windows 11.
Model: Qwen/Qwen2.5-0.5B.
Single FP16 model, no real precision switching.
Scheduler is simulated; no real precision switching.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("FLASH_ATTENTION_SKIP_IMPORT", "1")
os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")

import torch
import torch.nn as nn

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# pmpd is a declared benchmarking dependency (see requirements.txt).
# Install via: pip install git+https://github.com/SamsungLabs/PMPD.git
from pmpd.modules.model_fp16 import PMPDForCausalLMFP16
from pmpd.modules.scheduler import Scheduler
from pmpd.utils.model_loader import load_fp16_model

# ───────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEFAULT_PRECISIONS = [4, 8]
DEFAULT_HIGH_BIT_STEPS = 20
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_SEQ_LENGTHS = [32, 64, 128, 256]
DEFAULT_SWITCH_POINTS = [5, 10, 20, 40]
PROMPT = "What is the future of artificial intelligence in healthcare?"


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────
def _cuda_event_pair():
    """Return (start, end) CUDA events with timing enabled."""
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _print_banner(text: str, char: str = "=") -> None:
    line = char * 70
    print(f"\n{line}\n  {text}\n{line}")


def _free_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2


# ───────────────────────────────────────────────────────────────────────────
# CUDA diagnostics
# ───────────────────────────────────────────────────────────────────────────
def print_cuda_diagnostics() -> bool:
    _print_banner("Simulated PMPD CUDA Graph Benchmark - Section 5.2")
    print(f"  torch.cuda.is_available() : {torch.cuda.is_available()}")
    print(f"  torch.version.cuda        : {torch.version.cuda!r}")
    if torch.cuda.is_available():
        print(f"  GPU                       : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  Total VRAM                : {total:.0f} MB")
    else:
        print("  ❌ No CUDA device — cannot run benchmark.")
        return False
    print(f"  Model                     : {MODEL_NAME}")
    print(f"  Backend                   : FP16 (No Quantisation)")
    return True


# ───────────────────────────────────────────────────────────────────────────
# Model loading
# ───────────────────────────────────────────────────────────────────────────
def load_pmpd_fp16(precisions, high_bit_steps, capture_graphs=True):
    """Load PMPDForCausalLMFP16 with simulated scheduler behavior."""
    sched = Scheduler.get_scheduler(
        "naive",
        precisions=precisions,
        high_bit_steps=high_bit_steps,
    )
    model = PMPDForCausalLMFP16(
        model_name_or_path=MODEL_NAME,
        precisions=precisions,
        scheduler=sched,
    )
    if capture_graphs:
        model.build_cuda_graphs(max_seq_len=256)
    return model


def load_fp16_baseline():
    """Load fp16 Qwen2.5-0.5B context for baseline comparison."""
    return load_fp16_model(MODEL_NAME, label="FP16")


# ───────────────────────────────────────────────────────────────────────────
# Core timing functions
# ───────────────────────────────────────────────────────────────────────────
def measure_generation(
    generate_fn,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    n_warmup: int = 2,
    n_runs: int = 3,
    label: str = "",
) -> dict:
    """Time a generate function over multiple runs.

    Returns dict with:
        wall_ms_per_token, gpu_ms_per_token, cpu_overhead_ms_per_token,
        tokens_per_sec, total_tokens, wall_samples_ms
    """
    # Warmup
    for _ in range(n_warmup):
        _free_vram()
        _ = generate_fn(input_ids, max_new_tokens=max_new_tokens)

    wall_samples = []
    gpu_samples = []
    token_counts = []

    for run_i in range(n_runs):
        _free_vram()

        start_event, end_event = _cuda_event_pair()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        start_event.record()

        result = generate_fn(input_ids, max_new_tokens=max_new_tokens)

        end_event.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        wall_ms = (t1 - t0) * 1000.0
        gpu_ms = start_event.elapsed_time(end_event)
        n_tok = result.new_token if hasattr(result, "new_token") else max_new_tokens

        wall_samples.append(wall_ms)
        gpu_samples.append(gpu_ms)
        token_counts.append(n_tok)

    avg_tokens = sum(token_counts) / len(token_counts)
    avg_wall   = sum(wall_samples) / len(wall_samples)
    avg_gpu    = sum(gpu_samples)  / len(gpu_samples)

    # spec: latency_ms = (total_time / tokens) * 1000
    #        tokens_per_sec = tokens / total_time
    #        gpu_ms = (gpu_time / tokens) * 1000
    #        cpu_overhead = max(latency_ms - gpu_ms, 0)   ← clamped
    wall_per_tok = avg_wall / max(avg_tokens, 1)
    gpu_per_tok  = avg_gpu  / max(avg_tokens, 1)
    cpu_overhead = max(wall_per_tok - gpu_per_tok, 0.0)   # spec: clamp ≥ 0
    tps = 1000.0 / wall_per_tok if wall_per_tok > 0 else 0.0

    return {
        "label": label,
        "wall_ms_per_token": wall_per_tok,
        "gpu_ms_per_token": gpu_per_tok,
        "cpu_overhead_ms_per_token": cpu_overhead,
        "tokens_per_sec": tps,
        "total_tokens": avg_tokens,
        "wall_samples_ms": wall_samples,
        "precision_log": getattr(result, "precision_log", {}),
    }


def measure_fp16(model, tokenizer, input_ids, max_new_tokens, n_runs=3, label="fp16_eager"):
    """Measure plain fp16 autoregressive generation (no Simulated PMPD).

    Uses multinomial sampling at temperature=0.7 to match the app decode loop.
    Spec:
        probs = torch.softmax(logits[:, -1, :] / 0.7, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    """
    from transformers.generation.utils import ModelOutput

    _TEMPERATURE = 0.7  # spec: temperature = 0.7

    @torch.inference_mode()
    def _gen(ids, max_new_tokens):
        ids  = ids.clone().to("cuda")
        out  = model(ids, use_cache=True)
        past = out.past_key_values
        n    = 0
        while n < max_new_tokens:
            # spec: sampling instead of argmax
            probs = torch.softmax(out.logits[:, -1, :] / _TEMPERATURE, dim=-1)
            nxt   = torch.multinomial(probs, num_samples=1)          # shape (1,1)
            out   = model(nxt, past_key_values=past, use_cache=True)
            past  = out.past_key_values
            ids   = torch.cat([ids, nxt], dim=-1)
            n    += 1
            # spec: stop on EOS token
            if tokenizer.eos_token_id is not None and nxt.item() == tokenizer.eos_token_id:
                break
        return ModelOutput(input_ids=ids, new_token=n)

    return measure_generation(_gen, input_ids, max_new_tokens, n_runs=n_runs, label=label)


# ───────────────────────────────────────────────────────────────────────────
# Result formatting
# ───────────────────────────────────────────────────────────────────────────
def print_result_table(results: list[dict], title: str = ""):
    if title:
        _print_banner(title, "-")
    hdr = f"{'Config':<26} {'Wall ms/tok':>12} {'GPU ms/tok':>11} {'CPU OH ms':>10} {'tok/s':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['label']:<26} "
            f"{r['wall_ms_per_token']:>12.3f} "
            f"{r['gpu_ms_per_token']:>11.3f} "
            f"{r['cpu_overhead_ms_per_token']:>10.3f} "
            f"{r['tokens_per_sec']:>8.2f}"
        )
    if len(results) >= 2:
        base = results[0]["tokens_per_sec"]
        for r in results[1:]:
            if base > 0:
                sp = r["tokens_per_sec"] / base
                oh_reduction = (
                    (results[0]["cpu_overhead_ms_per_token"] - r["cpu_overhead_ms_per_token"])
                    / max(results[0]["cpu_overhead_ms_per_token"], 1e-9)
                    * 100
                )
                print(
                    f"  -> {r['label']} vs {results[0]['label']}: "
                    f"{sp:.2f}x throughput, "
                    f"{oh_reduction:+.1f}% CPU overhead change"
                )


def print_thesis_summary(results: list[dict]):
    _print_banner("MSc THESIS SUMMARY TABLE")
    print(f"{'Implementation':<28} | {'Tokens/s':>10} | {'Wall ms/tok':>12} | {'CPU OH ms/tok':>13} | {'GPU ms/tok':>11}")
    print("-" * 88)
    for r in results:
        print(
            f"{r['label']:<28} | "
            f"{r['tokens_per_sec']:>10.2f} | "
            f"{r['wall_ms_per_token']:>12.3f} | "
            f"{r['cpu_overhead_ms_per_token']:>13.3f} | "
            f"{r['gpu_ms_per_token']:>11.3f}"
        )

    # Speedup summary
    if len(results) >= 2:
        fp16_tps = next((r["tokens_per_sec"] for r in results if "fp16" in r["label"]), None)
        eager_tps = next((r["tokens_per_sec"] for r in results if "pmpd" in r["label"] and "graph" not in r["label"]), None)
        graph_tps = next((r["tokens_per_sec"] for r in results if "graph" in r["label"]), None)

        print()
        if eager_tps and fp16_tps and fp16_tps > 0:
            print(f"  Simulated PMPD Eager vs FP16:  {eager_tps / fp16_tps:.2f}x throughput")
        if graph_tps and eager_tps and eager_tps > 0:
            print(f"  Simulated PMPD Graph vs Eager: {graph_tps / eager_tps:.2f}x throughput  ← CUDA Graph contribution")
        if graph_tps and fp16_tps and fp16_tps > 0:
            print(f"  Simulated PMPD Graph vs FP16:  {graph_tps / fp16_tps:.2f}x throughput")


# ───────────────────────────────────────────────────────────────────────────
# Experiment runners
# ───────────────────────────────────────────────────────────────────────────
def run_main_experiments(args) -> list[dict]:
    """Run the four core experiments for the thesis results table."""
    _print_banner("EXPERIMENT SET 1: Core Comparison (4 configs)")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    inputs = tok(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    print(f"  Prompt: \"{PROMPT}\"")
    print(f"  Input tokens: {input_ids.shape[1]}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Runs per config: {args.runs}")

    all_results = []

    # 1) FP16 eager baseline
    print("\n-- [1/4] fp16_eager --")
    try:
        fp16_model, fp16_tok = load_fp16_baseline()
        r = measure_fp16(fp16_model, fp16_tok, input_ids, args.max_new_tokens,
                         n_runs=args.runs, label="fp16_eager")
        all_results.append(r)
        del fp16_model
        _free_vram()
    except Exception as e:
        print(f"  ⚠ fp16 baseline failed: {e}")
        all_results.append({"label": "fp16_eager", "wall_ms_per_token": 0,
                            "gpu_ms_per_token": 0, "cpu_overhead_ms_per_token": 0,
                            "tokens_per_sec": 0, "total_tokens": 0,
                            "wall_samples_ms": [], "precision_log": {}})

    # 2) Uniform fp16 eager
    print("\n-- [2/4] uniform_fp16_eager --")
    try:
        fp16_sim = load_pmpd_fp16([4, 4], high_bit_steps=0, capture_graphs=False)
        r = measure_generation(
            fp16_sim.pmpd_generate, input_ids, args.max_new_tokens,
            n_runs=args.runs, label="uniform_fp16_eager",
        )
        all_results.append(r)
        del fp16_sim
        _free_vram()
    except Exception as e:
        print(f"  ⚠ uniform fp16 failed: {e}")
        all_results.append({"label": "uniform_fp16_eager", "wall_ms_per_token": 0,
                            "gpu_ms_per_token": 0, "cpu_overhead_ms_per_token": 0,
                            "tokens_per_sec": 0, "total_tokens": 0,
                            "wall_samples_ms": [], "precision_log": {}})

    # 3) Simulated PMPD eager (NaiveScheduler, [4,8], high_bit_steps=20)
    print("\n-- [3/4] simulated_pmpd_eager --")
    try:
        fp16_pmpd = load_pmpd_fp16(DEFAULT_PRECISIONS, args.high_bit_steps, capture_graphs=False)
        r = measure_generation(
            fp16_pmpd.pmpd_generate, input_ids, args.max_new_tokens,
            n_runs=args.runs, label="simulated_pmpd_eager",
        )
        all_results.append(r)

        # 4) Simulated PMPD graph (same model, capture graphs)
        print("\n-- [4/4] simulated_pmpd_graph --")
        fp16_pmpd.build_cuda_graphs(max_seq_len=256)
        r2 = measure_generation(
            fp16_pmpd.pmpd_generate_fast, input_ids, args.max_new_tokens,
            n_runs=args.runs, label="simulated_pmpd_graph",
        )
        all_results.append(r2)
        del fp16_pmpd
        _free_vram()
    except Exception as e:
        print(f"  ⚠ Simulated PMPD FP16 failed: {e}")
        import traceback; traceback.print_exc()
        for lbl in ["simulated_pmpd_eager", "simulated_pmpd_graph"]:
            if not any(r["label"] == lbl for r in all_results):
                all_results.append({"label": lbl, "wall_ms_per_token": 0,
                                    "gpu_ms_per_token": 0, "cpu_overhead_ms_per_token": 0,
                                    "tokens_per_sec": 0, "total_tokens": 0,
                                    "wall_samples_ms": [], "precision_log": {}})

    return all_results


def run_seq_length_sweep(args) -> list[dict]:
    """Experiment 2: vary decode length to measure throughput scaling."""
    _print_banner("EXPERIMENT SET 2: Sequence Length Sweep")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    input_ids = tok(PROMPT, return_tensors="pt")["input_ids"].to("cuda")

    results = []
    model = load_pmpd_fp16(DEFAULT_PRECISIONS, args.high_bit_steps)
    for sl in DEFAULT_SEQ_LENGTHS:
        print(f"\n  -> max_new_tokens = {sl}")
        r_eager = measure_generation(
            model.pmpd_generate, input_ids, sl, n_runs=args.runs,
            label=f"eager_len{sl}",
        )
        r_graph = measure_generation(
            model.pmpd_generate_fast, input_ids, sl, n_runs=args.runs,
            label=f"graph_len{sl}",
        )
        results.extend([r_eager, r_graph])
    del model
    _free_vram()
    return results


def run_switch_point_sweep(args) -> list[dict]:
    """Experiment 3: vary NaiveScheduler high_bit_steps."""
    _print_banner("EXPERIMENT SET 3: Scheduler Switch-Point Sweep")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    input_ids = tok(PROMPT, return_tensors="pt")["input_ids"].to("cuda")

    results = []
    for sp in DEFAULT_SWITCH_POINTS:
        print(f"\n  -> high_bit_steps = {sp}")
        model = load_pmpd_fp16(DEFAULT_PRECISIONS, high_bit_steps=sp)
        r_eager = measure_generation(
            model.pmpd_generate, input_ids, args.max_new_tokens,
            n_runs=args.runs, label=f"eager_sw{sp}",
        )
        r_graph = measure_generation(
            model.pmpd_generate_fast, input_ids, args.max_new_tokens,
            n_runs=args.runs, label=f"graph_sw{sp}",
        )
        results.extend([r_eager, r_graph])
        del model
        _free_vram()
    return results


# ───────────────────────────────────────────────────────────────────────────
# Demo mode (no model download — synthetic MLP, validates CUDA Graph path)
# ───────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def run_demo_benchmark(args) -> list[dict]:
    """Synthetic MLP CUDA Graph demo — same metrics without downloading any weights."""
    _print_banner("DEMO MODE: Synthetic MLP CUDA Graph Benchmark")
    print("  Illustrates Section 5.2 (CPU launch overhead gap) without model weights.")
    print("  Single FP16 model, no real precision switching.")

    hidden, layers = 2048, 24
    net_layers = []
    for _ in range(layers):
        net_layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
    net = nn.Sequential(*net_layers).cuda().half().eval()
    print(f"  Network: {layers}-layer MLP (hidden={hidden})")

    static_in = torch.randn(1, hidden, device="cuda", dtype=torch.float16)
    pattern = torch.randn_like(static_in)

    # Warmup + capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = net(static_in)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _ = net(static_in)

    n_iter = args.iters if hasattr(args, "iters") else 100

    # Eager timing
    def eager_step():
        static_in.copy_(pattern)
        _ = net(static_in)

    def graph_step():
        static_in.copy_(pattern)
        g.replay()

    def _bench(step_fn, label, n_warmup=5, n_iter=n_iter):
        for _ in range(n_warmup):
            torch.cuda.synchronize()
            step_fn()

        # spec: torch.cuda.synchronize() BEFORE timing start
        torch.cuda.synchronize()

        se, ee = _cuda_event_pair()
        t0 = time.perf_counter()   # wall-clock start
        se.record()                # GPU Event start

        for _ in range(n_iter):
            step_fn()

        ee.record()                # GPU Event end
        # spec: torch.cuda.synchronize() AFTER loop
        torch.cuda.synchronize()
        t1 = time.perf_counter()   # wall-clock end

        wall_ms = (t1 - t0) * 1000.0
        gpu_ms  = se.elapsed_time(ee)

        wall_per = wall_ms / n_iter
        gpu_per  = gpu_ms  / n_iter
        # spec: cpu_overhead = max(latency_ms - gpu_ms, 0)   ← clamped
        cpu_oh   = max(wall_per - gpu_per, 0.0)
        return {
            "label": label,
            "wall_ms_per_token": wall_per,
            "gpu_ms_per_token": gpu_per,
            "cpu_overhead_ms_per_token": cpu_oh,
            "tokens_per_sec": 1000.0 / wall_per if wall_per > 0 else 0,
            "total_tokens": n_iter,
            "wall_samples_ms": [wall_ms],
            "precision_log": {},
        }

    r_eager = _bench(eager_step, "demo_eager")
    r_graph = _bench(graph_step, "demo_graph_replay")
    return [r_eager, r_graph]


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Simulated PMPD Section 5.2 Benchmark - Eager vs CUDA Graph Decode"
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--high-bit-steps", type=int, default=DEFAULT_HIGH_BIT_STEPS)
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per config")
    parser.add_argument("--iters", type=int, default=100, help="Iterations for demo mode")
    parser.add_argument("--demo", action="store_true", help="Run synthetic MLP demo only")
    parser.add_argument(
        "--experiments",
        type=str,
        default="core",
        choices=["core", "seqlen", "switchpoint", "all", "demo"],
        help="Which experiment set to run",
    )
    args = parser.parse_args()

    if not print_cuda_diagnostics():
        return 1

    if args.demo or args.experiments == "demo":
        demo_results = run_demo_benchmark(args)
        print_result_table(demo_results, "Demo Results")
        print_thesis_summary(demo_results)
        return 0

    all_results = []

    if args.experiments in ("core", "all"):
        core = run_main_experiments(args)
        print_result_table(core, "Core Experiment Results")
        all_results.extend(core)

    if args.experiments in ("seqlen", "all"):
        sl = run_seq_length_sweep(args)
        print_result_table(sl, "Sequence Length Sweep Results")
        all_results.extend(sl)

    if args.experiments in ("switchpoint", "all"):
        sp = run_switch_point_sweep(args)
        print_result_table(sp, "Switch-Point Sweep Results")
        all_results.extend(sp)

    # Final thesis summary from core results
    core_results = [r for r in all_results if r["label"] in
                    ("fp16_eager", "uniform_fp16_eager", "simulated_pmpd_eager", "simulated_pmpd_graph")]
    if core_results:
        print_thesis_summary(core_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
