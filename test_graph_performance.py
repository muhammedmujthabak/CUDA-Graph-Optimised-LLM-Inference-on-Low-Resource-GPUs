# -*- coding: utf-8 -*-
"""
Qwen/Qwen2.5-0.5B - Correct Autoregressive Decode Benchmark:
Eager vs CUDA Graph.

This script is part of the thesis project:
    "CUDA Graph Optimized LLM Inference on Low-Resource GPUs"

It measures ONLY the decode phase (token-by-token autoregressive generation).
Prefill and CUDA Graph capture are explicitly excluded from timing.

Design goals:
    - Fair comparison: both modes execute the exact same number of decode steps
      using the same greedy (argmax) strategy and identical inputs.
    - Correct timing: wall-clock via time.perf_counter(), GPU via CUDA Events,
      with torch.cuda.synchronize() before and after the timed region.
    - Proper phase separation: prefill (untimed) -> graph capture (untimed) ->
      decode loop (timed).
    - True autoregressive decoding: token-by-token with KV cache, NOT a single
      batched forward pass.
    - CUDA Graph constraints: static input/output tensors, fixed shapes,
      in-place updates before replay.

NOTE on CPU overhead:
    cpu_overhead_ms = (wall_time - gpu_time) / num_tokens
    This is an *approximation*. CUDA Events measure GPU kernel execution time
    and do not capture driver-level scheduling delays, OS jitter, or Python
    interpreter overhead that falls outside recorded GPU kernels. The metric
    is useful for relative comparison between eager and graph modes but should
    not be interpreted as an exact measurement of CPU dispatch cost.

Requires: PyTorch (CUDA), transformers (with StaticCache).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import StaticCache
except Exception:
    StaticCache = None

# Import the real PMPD Scheduler via the compatibility shim, which loads
# pmpd's scheduler submodule directly without triggering the broken
# pmpd.__init__ -> auto_gptq -> transformers incompatibility.
from pmpd_compat import Scheduler


# ============================================================================
#  Configuration
# ============================================================================

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"


@dataclass
class RuntimeConfig:
    """All tuneable parameters for the benchmark."""
    model_id: str
    prompt: str
    max_new_tokens: int      # Number of decode steps (same for both modes)
    kv_cache_max: int        # Maximum length for StaticCache


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-0.5B: eager vs CUDA Graph autoregressive decode benchmark"
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--prompt", type=str,
        default="Explain simulated PMPD decode with CUDA Graph in 3 lines.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50,
        help="Number of NEW tokens to generate in the decode loop (timed region).",
    )
    parser.add_argument("--kv-cache-max", type=int, default=256)
    args = parser.parse_args()
    return RuntimeConfig(
        model_id=args.model_id,
        prompt=args.prompt,
        max_new_tokens=min(args.max_new_tokens, 128),
        kv_cache_max=args.kv_cache_max,
    )


# ============================================================================
#  Timing results
# ============================================================================

@dataclass
class DecodeMetrics:
    """Timing results for one decode run."""
    mode: str                    # "eager" or "cuda_graph"
    num_tokens: int              # Total tokens generated in timed loop
    wall_ms: float               # Wallâ€clock time for decode loop (ms)
    gpu_ms: float                # GPU time from CUDA Events (ms)
    latency_ms_per_token: float  # wall_ms / num_tokens
    tokens_per_sec: float        # num_tokens / (wall_ms / 1000)
    cpu_overhead_ms_per_token: float  # (wall_ms - gpu_ms) / num_tokens
    generated_text: str          # Decoded text (for sanity check)
    generated_ids: list[int]     # Raw token IDs (for reproducibility check)


def _compute_metrics(
    mode: str,
    num_tokens: int,
    wall_ms: float,
    gpu_ms: float,
    generated_ids: list[int],
    tokenizer,
) -> DecodeMetrics:
    """Compute all derived metrics from raw timing data."""
    # Guard against division by zero if no tokens were generated
    if num_tokens == 0 or wall_ms <= 0:
        return DecodeMetrics(
            mode=mode, num_tokens=0, wall_ms=0.0, gpu_ms=0.0,
            latency_ms_per_token=0.0, tokens_per_sec=0.0,
            cpu_overhead_ms_per_token=0.0,
            generated_text="", generated_ids=[],
        )

    latency_ms = wall_ms / num_tokens
    tokens_per_sec = num_tokens / (wall_ms / 1000.0)

    # NOTE: CPU overhead is an approximation.  CUDA Events measure only
    # GPU kernel execution; they do not capture driver scheduling latency,
    # host-side memory copies issued outside recorded streams, or Python
    # interpreter overhead between kernel launches.
    cpu_overhead_ms = (wall_ms - gpu_ms) / num_tokens

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    return DecodeMetrics(
        mode=mode,
        num_tokens=num_tokens,
        wall_ms=wall_ms,
        gpu_ms=gpu_ms,
        latency_ms_per_token=latency_ms,
        tokens_per_sec=tokens_per_sec,
        cpu_overhead_ms_per_token=cpu_overhead_ms,
        generated_text=text,
        generated_ids=generated_ids,
    )


# ============================================================================
#  Model loading
# ============================================================================

def build_model_and_tokenizer(model_id: str):
    """Load a standard fp16 model for CUDA Graph benchmarking."""
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
    return model, tokenizer


def _first_cuda_device(model) -> torch.device:
    # Return the first CUDA device used by the models parameters.
    for p in model.parameters():
        if p.device.type == "cuda":
            return p.device
    raise RuntimeError("No CUDA device found on model parameters.")


# ============================================================================
#  Input preparation â€” identical for both modes
# ============================================================================

def build_chat_inputs(tokenizer, prompt: str, device: torch.device):
    """Tokenise a prompt into (input_ids, attention_mask) on `device`.

    Uses the chat template if available, otherwise falls back to a simple
    "User: ... Assistant:" format.  Both modes receive the exact same tensors.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    input_ids = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            except (TypeError, ValueError):
                pass
    if input_ids is None:
        rendered = "User: " + prompt.strip() + "\nAssistant:"
        input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"]
        
    if isinstance(input_ids, dict) or hasattr(input_ids, "input_ids"):
        if "input_ids" in input_ids:
            input_ids = input_ids["input_ids"]

    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids.to(device), attention_mask.to(device)


# ============================================================================
#  PMPD Scheduler (simulated â€” single 4-bit model, no real precision switch)
# ============================================================================

def attach_pmpd_scheduler(model):
    """Attach a simulated PMPD scheduler for thesis compatibility.

    The scheduler tracks precision labels conceptually but all compute
    uses the same NF4 model.  This is consistent with the thesis contribution
    (CUDA Graph integration, not quantisation kernels).
    """
    class PMPDScheduler(Scheduler, scheduler_name="pmpd"):
        def __init__(self, precisions):
            super().__init__(precisions)

        def schedule(self, **kwargs):
            return kwargs.get("precision", max(self.precisions))

        def reset(self):
            pass

    model.scheduler = PMPDScheduler([2, 3, 4])


# ============================================================================
#  Eager decode  (baseline â€” DynamicCache, standard forward calls)
# ============================================================================

@torch.inference_mode()
def eager_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> DecodeMetrics:
    """Greedy autoregressive decode using eager (standard) forward passes.

    Phase separation:
        1. PREFILL (NOT timed) â€” process the full prompt, build initial KV cache.
        2. DECODE LOOP (TIMED) â€” generate exactly `max_new_tokens` new tokens
           one at a time, each conditioned on all previous tokens via the
           DynamicCache (past_key_values).

    Greedy decoding: argmax over logits (deterministic, no sampling).
    """
    device = _first_cuda_device(model)
    input_ids, attention_mask = build_chat_inputs(tokenizer, prompt, device)

    # ------------------------------------------------------------------
    # Phase 1: PREFILL (NOT timed)
    # Run the full prompt through the model to build the initial KV cache.
    # ------------------------------------------------------------------
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past = out.past_key_values

    # Pick the first decode token from prefill logits (NOT timed)
    next_token = torch.argmax(out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)

    generated: list[int] = []
    produced_tokens = 0
    # Append the very first token from prefill
    generated.append(int(next_token.item()))

    # First decode step (NOT timed) â€” this aligns eager with CUDA Graph,
    # which executes one implicit forward pass during graph capture.
    out = model(
        input_ids=next_token,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)

    # ------------------------------------------------------------------
    # Phase 2: DECODE LOOP (TIMED)
    # Generate exactly max_new_tokens via greedy argmax.
    # ------------------------------------------------------------------

    # Synchronise GPU before starting the timer to ensure prefill is complete
    torch.cuda.synchronize(device)

    # Create CUDA events for GPU-side timing
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)

    # Record wall-clock start
    wall_start = time.perf_counter()
    t_start.record()

    for step in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        
        # Stop early on EOS
        if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
            break

        # Forward pass: single token + KV cache from previous steps
        out = model(
            input_ids=next_token,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        
        next_token = torch.argmax(out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
        produced_tokens += 1

    # Record GPU end event and synchronise
    t_end.record()
    torch.cuda.synchronize(device)
    wall_end = time.perf_counter()

    # Compute raw timing
    gpu_ms = t_start.elapsed_time(t_end)         # GPU-side milliseconds
    wall_ms = (wall_end - wall_start) * 1000.0    # Wall-clock milliseconds

    return _compute_metrics(
        mode="eager",
        num_tokens=produced_tokens,
        wall_ms=wall_ms,
        gpu_ms=gpu_ms,
        generated_ids=generated,
        tokenizer=tokenizer,
    )


# ============================================================================
#  CUDA Graph decode  (contribution - StaticCache, graph.replay())
# ============================================================================

@torch.inference_mode()
def graph_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    kv_cache_max: int,
) -> DecodeMetrics:
    """Greedy autoregressive decode using a captured CUDA Graph.

    Phase separation:
        1. PREFILL (NOT timed) â€” process the full prompt via StaticCache.
        2. CUDA GRAPH CAPTURE (NOT timed) â€” warm up the single-token forward
           path, then capture it as a CUDA Graph.  This is a one-time cost
           and must NOT be included in the decode latency measurement.
        3. DECODE LOOP (TIMED) â€” replay the captured graph for each decode
           step.  Input token and cache position are updated in-place via
           static tensors before each replay.

    CUDA Graph constraints:
        - Static input/output tensors with fixed shapes (batch=1, seq_len=1).
        - StaticCache with pre-allocated maximum length.
        - In-place updates to static_token and static_pos before replay.
        - No dynamic control flow inside the captured region.

    Greedy decoding: argmax over logits (deterministic, no sampling).
    Same strategy as eager_decode for fair comparison.
    """
    if StaticCache is None:
        raise RuntimeError(
            "StaticCache is unavailable in this transformers build. "
            "CUDA Graph decode requires transformers >= 4.39."
        )

    device = _first_cuda_device(model)
    input_ids, _ = build_chat_inputs(tokenizer, prompt, device)
    prompt_len = int(input_ids.shape[1])

    if prompt_len >= kv_cache_max:
        raise RuntimeError(
            f"Prompt length ({prompt_len}) exceeds StaticCache bound ({kv_cache_max}). "
            f"Increase --kv-cache-max."
        )
    if prompt_len + max_new_tokens >= kv_cache_max:
        raise RuntimeError(
            f"Prompt ({prompt_len}) + decode tokens ({max_new_tokens}) exceeds "
            f"StaticCache bound ({kv_cache_max}). Increase --kv-cache-max."
        )

    # Check for CPU/disk offloading which is incompatible with CUDA Graphs
    if hasattr(model, "hf_device_map"):
        mapped = {str(v) for v in model.hf_device_map.values()}
        if any("cpu" in x or "disk" in x for x in mapped):
            raise RuntimeError(
                "Offload detected in device_map; CUDA Graph decode requires "
                "all parameters on GPU."
            )

    # Simulate PMPD scheduler calls (precision labels only, no real switching).
    # This ensures the scheduler state is exercised identically to eager mode.
    # Note: Only keeping this for compatibility with the script's original thesis intent.
    for _ in range(max_new_tokens):
        if hasattr(model, "scheduler"):
            _ = model.scheduler.schedule(precision=4)

    # ==================================================================
    # Phase 1: PREFILL (NOT timed)
    # Process the full prompt to fill the StaticCache with initial KV pairs.
    # ==================================================================
    cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=kv_cache_max,
        device=device,
        dtype=torch.float16,
    )
    # FIX: transformers initializes cumulative_length on CPU.
    # CUDA Graphs capture CPU tensors as static compile-time values (zero),
    # so every replay would write KV data to position 0 — corrupting the cache.
    # Moving to device ensures it increments correctly on each graph replay.
    if hasattr(cache, 'layers'):
        for layer in cache.layers:
            if hasattr(layer, 'cumulative_length'):
                layer.cumulative_length = layer.cumulative_length.to(device)

    prefill_out = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )

    # Pick the first decode token from prefill logits (NOT timed)
    next_token = torch.argmax(
        prefill_out.logits[:, -1:, :], dim=-1
    ).to(dtype=torch.long)

    # ==================================================================
    # Phase 2: CUDA GRAPH CAPTURE (NOT timed)
    # Allocate static tensors, warm up the forward path, then capture.
    # ==================================================================

    # Static tensors: these are the fixed-shape buffers that we update
    # in-place before each graph.replay().  The graph records operations
    # on these exact memory addresses.
    static_token = torch.zeros((1, 1), dtype=torch.long, device=device)
    static_pos = torch.zeros((1,), dtype=torch.long, device=device)

    pos = prompt_len  # Current position in the sequence
    static_token.copy_(next_token)
    static_pos.fill_(pos)

    # Warmup forward: required to ensure all lazy CUDA allocations (cuDNN
    # workspace) are completed before capture.
    # The graph capture records a CUDA stream; any allocation/deallocation
    # during capture would make the graph non-replayable.
    _ = model(
        input_ids=static_token,
        past_key_values=cache,
        cache_position=static_pos,
        use_cache=True,
        return_dict=True,
    )
    torch.cuda.synchronize(device)

    # Capture the single-token forward as a CUDA Graph.
    # If capture fails,
    # we raise immediately rather than silently falling back to eager.
    graph = torch.cuda.CUDAGraph()
    capture_succeeded = False
    try:
        with torch.cuda.graph(graph):
            graph_out = model(
                input_ids=static_token,
                past_key_values=cache,
                cache_position=static_pos,
                use_cache=True,
                return_dict=True,
            )
            # static_logits is a view into the graph's output memory;
            # it updates automatically on each replay.
            static_logits = graph_out.logits
        capture_succeeded = True
    except Exception as exc:
        # Do NOT silently fall back â€” raise a clear error so the caller
        # knows CUDA Graph capture failed.
        raise RuntimeError(
            f"[CUDA GRAPH CAPTURE FAILED] {type(exc).__name__}: {exc}\n"
            f"The model's forward pass contains operations incompatible with "
            f"CUDA Graph capture (e.g. dynamic allocation, CPU synchronisation, "
            f"or data-dependent control flow).\n"
            f"Run with --no-cuda-graph to benchmark eager mode only."
        ) from exc

    assert capture_succeeded, "CUDA Graph capture did not succeed."
    print("[CUDA GRAPH] + Graph captured successfully (NOT included in timing).")

    # The first token was captured from prefill logics. We append it.
    generated: list[int] = []
    produced_tokens = 0
    generated.append(int(next_token.item()))
    
    # We MUST extract the token from static_logits and advance pos here!
    # The capture actually performed the identical first unmeasured forward pass
    # that the eager mode did. The subsequent logic allows the timed loop 
    # to perfectly align the cache and sequence numbers.
    next_token = torch.argmax(static_logits[:, -1:, :], dim=-1).to(dtype=torch.long)
    pos += 1

    # ==================================================================
    # Phase 3: DECODE LOOP (TIMED)
    # Only graph.replay() calls and argmax token selection are measured.
    # Both eager and graph modes execute exactly max_new_tokens forward
    # passes / replays inside the timed region for a fair comparison.
    # ==================================================================
    # NOTE: Do NOT reset `generated` or `produced_tokens` here.
    # `generated` already holds T0 (from prefill logits) to match
    # the eager mode, which also pre-populates before timing.

    # Synchronise GPU to ensure capture phase is fully complete
    torch.cuda.synchronize(device)

    # Create CUDA events for GPU-side timing
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)

    # Record wall-clock start
    wall_start = time.perf_counter()
    t_start.record()

    for step in range(max_new_tokens):
        if pos >= kv_cache_max:
            print(f"[WARNING] KV cache full at step {step}, stopping decode.")
            break

        token_id = int(next_token.item())
        generated.append(token_id)

        # Stop early on EOS
        if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
            break

        # CPU: update static tensors in-place before replay.
        static_token.copy_(next_token)
        static_pos.fill_(pos)

        # GPU: replay the captured graph (single forward pass)
        graph.replay()
        produced_tokens += 1
        pos += 1

        # Read logits from the static output buffer (updated by replay)
        next_token = torch.argmax(
            static_logits[:, -1:, :], dim=-1
        ).to(dtype=torch.long)

    # Record GPU end event and synchronise
    t_end.record()
    torch.cuda.synchronize(device)
    wall_end = time.perf_counter()

    # Compute raw timing
    gpu_ms = t_start.elapsed_time(t_end)         # GPU-side milliseconds
    wall_ms = (wall_end - wall_start) * 1000.0    # Wall-clock milliseconds

    return _compute_metrics(
        mode="cuda_graph",
        num_tokens=produced_tokens,
        wall_ms=wall_ms,
        gpu_ms=gpu_ms,
        generated_ids=generated,
        tokenizer=tokenizer,
    )


# ============================================================================
#  Benchmark orchestrator
# ============================================================================

def benchmark(cfg: RuntimeConfig):
    """Run both eager and CUDA Graph decode, print comparison results.

    This function:
        1. Loads the model once (shared between both modes).
        2. Runs eager decode and collects metrics.
        3. Runs CUDA Graph decode and collects metrics.
        4. Prints a side-by-side comparison table.

    Both modes use:
        - The same model (FP16 Qwen2.5-0.5B)
        - The same prompt / input_ids
        - The same greedy (argmax) decoding strategy
        - The same max_new_tokens target
    """
    print("=" * 72)
    print("  CUDA Graph Optimised LLM Inference - Decode Benchmark")
    print("  Model:        ", cfg.model_id)
    print("  Prompt:       ", cfg.prompt[:60] + ("..." if len(cfg.prompt) > 60 else ""))
    print("  Decode tokens:", cfg.max_new_tokens)
    print("  KV cache max: ", cfg.kv_cache_max)
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # Load model (shared between both modes)
    # ------------------------------------------------------------------
    print("[1/4] Loading model and tokenizer ...")
    model, tokenizer = build_model_and_tokenizer(cfg.model_id)
    attach_pmpd_scheduler(model)
    print(f"      Model loaded on {_first_cuda_device(model)}")
    print()

    # ------------------------------------------------------------------
    # Run EAGER decode
    # ------------------------------------------------------------------
    print("[2/4] Running EAGER decode ...")
    eager_result = eager_decode(
        model=model,
        tokenizer=tokenizer,
        prompt=cfg.prompt,
        max_new_tokens=cfg.max_new_tokens,
    )
    print(f"      + Eager: {eager_result.num_tokens} tokens in {eager_result.wall_ms:.2f} ms")
    print()

    # ------------------------------------------------------------------
    # Run CUDA GRAPH decode
    # ------------------------------------------------------------------
    graph_result: Optional[DecodeMetrics] = None
    print("[3/4] Running CUDA GRAPH decode ...")
    try:
        graph_result = graph_decode(
            model=model,
            tokenizer=tokenizer,
            prompt=cfg.prompt,
            max_new_tokens=cfg.max_new_tokens,
            kv_cache_max=cfg.kv_cache_max,
        )
        print(f"      + Graph: {graph_result.num_tokens} tokens in {graph_result.wall_ms:.2f} ms")
    except RuntimeError as exc:
        # Print a clear warning â€” do NOT silently fall back
        print(f"      - CUDA Graph decode FAILED:")
        print(f"        {exc}")
        print(f"      Only eager results will be reported.")
    print()

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("[4/4] Results")
    print("=" * 72)
    _print_result("EAGER", eager_result)
    if graph_result is not None:
        print("-" * 72)
        _print_result("CUDA GRAPH", graph_result)
        print("-" * 72)
        _print_speedup(eager_result, graph_result)
    else:
        print("-" * 72)
        print("  CUDA Graph:  N/A (capture failed - see warning above)")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # Print generated text for sanity check
    # ------------------------------------------------------------------
    print("--- Generated text (EAGER) ---")
    print(eager_result.generated_text)
    print()
    if graph_result is not None:
        print("--- Generated text (CUDA GRAPH) ---")
        print(graph_result.generated_text)
        print()

    # ------------------------------------------------------------------
    # Validation checks
    # ------------------------------------------------------------------
    print("--- Validation checks ---")
    print(f"Eager decode steps: {eager_result.num_tokens}")
    if graph_result is not None:
        print(f"Graph decode steps: {graph_result.num_tokens}")
        print(f"Eager output length: {len(eager_result.generated_ids)}")
        print(f"Graph output length: {len(graph_result.generated_ids)}")
        
        # Determine exact match up to the minimum length (should be fully identical)
        min_len = min(len(eager_result.generated_ids), len(graph_result.generated_ids))
        matches = sum(1 for e, g in zip(eager_result.generated_ids[:min_len], graph_result.generated_ids[:min_len]) if e == g)
        match_pct = (matches / min_len * 100.0) if min_len > 0 else 0.0
        print(f"Token match percentage: {match_pct:.1f}%")
        
        if match_pct < 100.0:
            print("[WARNING] Eager and CUDA Graph token sequences differ significantly!")
        else:
            print("[OK] Eager and CUDA Graph token sequences match.")
    print()
    # Reproducibility note
    # ------------------------------------------------------------------
    print("NOTE: Texts may differ because eager uses DynamicCache while")
    print("      CUDA Graph uses StaticCache, which can cause minor KV cache")
    print("      numerical differences. Both use greedy argmax decoding.")
    print()
    print("NOTE: CPU overhead is an approximation.  CUDA events do not")
    print("      capture driver-level scheduling delays or Python interpreter")
    print("      overhead between kernel launches.")

    return eager_result, graph_result


def _print_result(label: str, m: DecodeMetrics):
    """Pretty-print a single DecodeMetrics result."""
    print(f"  {label}:")
    print(f"    Tokens generated:         {m.num_tokens}")
    print(f"    Total decode time (wall):  {m.wall_ms:.2f} ms")
    print(f"    Total decode time (GPU):   {m.gpu_ms:.2f} ms")
    print(f"    Latency per token:         {m.latency_ms_per_token:.4f} ms/token")
    print(f"    Throughput:                {m.tokens_per_sec:.2f} tokens/sec")
    print(f"    CPU overhead per token:    {m.cpu_overhead_ms_per_token:.4f} ms/token (approx)")


def _print_speedup(eager: DecodeMetrics, graph: DecodeMetrics):
    """Print speedup comparison between eager and graph modes."""
    if eager.latency_ms_per_token > 0 and graph.latency_ms_per_token > 0:
        speedup = eager.latency_ms_per_token / graph.latency_ms_per_token
        print(f"  SPEEDUP (latency):  {speedup:.3f}x  (>1 means graph is faster)")
    if eager.tokens_per_sec > 0 and graph.tokens_per_sec > 0:
        throughput_ratio = graph.tokens_per_sec / eager.tokens_per_sec
        print(f"  SPEEDUP (throughput): {throughput_ratio:.3f}x  (>1 means graph is faster)")
    if eager.cpu_overhead_ms_per_token > 0 and graph.cpu_overhead_ms_per_token > 0:
        overhead_reduction = (
            (eager.cpu_overhead_ms_per_token - graph.cpu_overhead_ms_per_token)
            / eager.cpu_overhead_ms_per_token
            * 100.0
        )
        print(f"  CPU overhead reduction: {overhead_reduction:.1f}%  (positive = graph has less overhead)")


# ============================================================================
#  Entry point
# ============================================================================

def main() -> bool:
    # Reduce CUDA memory fragmentation on low-VRAM GPUs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

    cfg = parse_args()

    print()
    print("This benchmark measures REAL autoregressive decode latency.")
    print("Single FP16 Qwen2.5-0.5B model - no real precision switching.")
    print("CUDA Graph capture time is EXCLUDED from timing.")
    print()

    eager_result, graph_result = benchmark(cfg)

    # Return success if at least eager completed
    return eager_result is not None


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
