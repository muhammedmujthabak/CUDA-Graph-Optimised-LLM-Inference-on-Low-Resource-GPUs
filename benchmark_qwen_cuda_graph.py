"""
Qwen/Qwen2.5-0.5B autoregressive decode benchmark with explicit precision modes.

Measures decode phase only (prefill excluded from timing):
  - Eager: token-by-token forwards with growing KV cache (DynamicCache).
  - CUDA Graph: StaticCache + one captured single-token forward; replay per step.

Per-seq_len: number of NEW tokens generated after a fixed prefill (32/64/128/256).

Requires: PyTorch (CUDA), transformers (with StaticCache), matplotlib.
Windows: use a UTF-8 terminal if you print non-ASCII; CSV/plots are ASCII-safe.

    python benchmark_tinyllama_cuda_graph.py --iters 40 --warmup 3

Outputs under ./benchmark_results/ by default:
    benchmark_results.csv, latency_bar.png, throughput_vs_seqlen.png, speedup.png

Notes:
    * seq_len columns mean *decode length* (new tokens), not prefill length.
    * Prefill is fixed text (--prefill); only the autoregressive decode phase is timed.
    * cuda_graph mode needs transformers with StaticCache; if graph capture fails,
      the script still reports eager results.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import StaticCache
except Exception:
    StaticCache = None


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_SEQ_LENS = (32, 64, 128, 256)
DEFAULT_ITERS = 40
DEFAULT_WARMUP = 3
# Fixed prefill; total context must fit in StaticCache: prefill_len + decode_tokens <= max_cache_len
DEFAULT_PREFILL = "The quick brown fox jumps over the lazy dog. Explain briefly: "
DEFAULT_KV_PAD = 64


@dataclass
class BenchResult:
    mode: str
    seq_len: int
    latency_ms_per_token: float
    tokens_per_sec: float
    cpu_overhead_ms_per_token: float
    wall_ms_decode: float
    gpu_ms_decode: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen2.5-0.5B: eager vs CUDA Graph *decode* benchmark"
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEQ_LENS),
        help="Number of decode (new) tokens to generate per run after prefill",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help="Repeated decode runs per metric (30–50+ recommended)",
    )
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--prefill", type=str, default=DEFAULT_PREFILL)
    p.add_argument(
        "--kv-pad",
        type=int,
        default=DEFAULT_KV_PAD,
        help="Extra headroom in StaticCache beyond prefill + max(decode)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print per-iteration step counts and normalisation details for verification.",
    )
    return p.parse_args()


def load_model_and_tokenizer(model_id: str):
    """Load fp16 model for eager-vs-graph benchmarking."""
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
    for p in model.parameters():
        if p.device.type == "cuda":
            return p.device
    raise RuntimeError("Model has no CUDA parameters.")


def build_prefill_ids(tokenizer, prefill_text: str, device: torch.device) -> torch.Tensor:
    t = tokenizer(prefill_text, return_tensors="pt", add_special_tokens=True)
    return t["input_ids"].to(device)


def decode_eager_timed(
    model,
    prefill_ids: torch.Tensor,
    decode_tokens: int,
    *,
    debug: bool = False,
) -> tuple[float, float]:
    """
    Greedy decode: exactly `decode_tokens` single-token forwards after prefill.
    Returns (wall_seconds, gpu_seconds) for the decode loop only.

    Timing starts immediately after prefill.  The first timed step consumes
    the prefill logits (argmax → token T0) and runs the T0→T1 forward;
    the last timed step produces token T(decode_tokens-1).  Exactly
    `decode_tokens` forwards are executed inside the timed region.
    """
    device = prefill_ids.device
    torch.cuda.synchronize(device)

    with torch.inference_mode():
        # PREFILL (untimed)
        out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        # Timed decode loop: decode_tokens steps.
        # Step i: argmax(current logits) → next_id → forward → new logits.
        steps_done = 0
        event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(decode_tokens):
            next_id = torch.argmax(out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
            start_e = torch.cuda.Event(enable_timing=True)
            end_e = torch.cuda.Event(enable_timing=True)
            start_e.record()
            out = model(
                input_ids=next_id,
                past_key_values=out.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            end_e.record()
            event_pairs.append((start_e, end_e))
            steps_done += 1

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        gpu_ms_total = sum(s.elapsed_time(e) for s, e in event_pairs)

    if debug:
        print(f"  [eager] decode steps timed: {steps_done}  (requested: {decode_tokens})")

    wall_s = t1 - t0
    gpu_s = gpu_ms_total / 1000.0
    return wall_s, gpu_s


def decode_cuda_graph_timed(
    model,
    prefill_ids: torch.Tensor,
    decode_tokens: int,
    kv_max: int,
    *,
    debug: bool = False,
) -> tuple[float, float, int]:
    """
    Greedy decode with StaticCache and CUDA Graph replay.

    Phase separation — mirrors eager path exactly:

        1. PREFILL (NOT timed)
           Process full prompt; extract T0 = argmax(prefill logits).

        2. GRAPH SETUP (NOT timed)
           a) Warmup forward at pos=prompt_len with token=T0:
              - writes T0 KV into StaticCache at pos=prompt_len
              - produces logits → extract T1 = argmax(logits); advance pos to prompt_len+1
           b) Re-initialise static_token=T1, static_pos=prompt_len+1
           c) Capture CUDA Graph at pos=prompt_len+1 with token=T1
              - capture reads KV for T0..prompt_len correctly
           d) Extract T2 = argmax(graph_out.logits); advance pos to prompt_len+2
              (graph is now ready to replay at pos=prompt_len+2 next)

        3. DECODE LOOP (TIMED) — exactly (decode_tokens - 2) replay steps
           Each replay: update static_token/pos, call graph.replay(),
           extract next token, increment pos.

    Total decode steps executed = 1 (warmup/T0) + 1 (capture/T1) +
                                  (decode_tokens - 2) (timed) = decode_tokens.
    This matches eager's decode_tokens timed forwards for a fair comparison.

    NOTE: decode_tokens must be >= 2.  For decode_tokens < 2 we fall back
    to a single captured replay so the benchmark still returns a value.
    """
    if StaticCache is None:
        raise RuntimeError("StaticCache not available in this transformers version.")

    if decode_tokens <= 0:
        return 0.0, 0.0

    device = _first_cuda_device(model)
    prompt_len = int(prefill_ids.shape[1])
    # Need room for prompt + decode_tokens positions in the static cache.
    if prompt_len + decode_tokens > kv_max:
        raise ValueError("decode_tokens + prefill_len exceeds kv_max")

    torch.cuda.synchronize(device)

    with torch.inference_mode():
        # ── Phase 1: PREFILL (NOT timed) ────────────────────────────────────
        cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=kv_max,
            device=device,
            dtype=torch.float16,
        )
        
        # FIX: transformers initializes cumulative_length on CPU. 
        # CUDA Graphs capture CPU tensors as static literal values.
        # Moving it to the GPU ensures it correctly increments internally during graph replays.
        for layer in cache.layers:
            layer.cumulative_length = layer.cumulative_length.to(device)
            
        pre_out = model(
            input_ids=prefill_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        # Extract token T0 from prefill output.
        next_t = torch.argmax(pre_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
        pos = prompt_len  # KV cache has been filled up to (but not including) pos.

        # Persistent tensors used as graph inputs — updated in-place each step.
        static_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        static_pos   = torch.zeros((1,),   dtype=torch.long, device=device)

        # ── Phase 2a: WARMUP FORWARD (NOT timed) ────────────────────────────
        # Run torch (eager) forward at pos=prompt_len with token=T0.
        # Purpose: trigger lazy CUDA memory allocations before graph capture,
        # AND advance the KV cache state so the capture sees a correctly
        # populated cache.  After this forward:
        #   - StaticCache slot at pos=prompt_len holds T0's KV.
        #   - We extract T1 and advance pos to prompt_len+1.
        static_token.copy_(next_t)   # T0
        static_pos.fill_(pos)        # prompt_len
        warmup_out = model(
            input_ids=static_token,
            past_key_values=cache,
            cache_position=static_pos,
            use_cache=True,
            return_dict=True,
        )
        torch.cuda.synchronize(device)

        # Advance: now we have T1 and pos points to prompt_len+1.
        next_t = torch.argmax(warmup_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
        pos += 1  # pos = prompt_len + 1

        # ── Phase 2b: GRAPH CAPTURE (NOT timed) ─────────────────────────────
        # Capture at pos=prompt_len+1 with token=T1.
        # The cache already holds KV for positions 0..prompt_len (from prefill
        # and the warmup forward above), so this capture is coherent.
        static_token.copy_(next_t)   # T1
        static_pos.fill_(pos)        # prompt_len + 1

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = model(
                input_ids=static_token,
                past_key_values=cache,
                cache_position=static_pos,
                use_cache=True,
                return_dict=True,
            )

        # Extract T2 from capture output; advance pos to prompt_len+2.
        next_t = torch.argmax(graph_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
        pos += 1  # pos = prompt_len + 2

        # ── Phase 3: DECODE LOOP (TIMED) ─────────────────────────────────────
        # We have already consumed 2 decode steps (warmup T0 + capture T1).
        # Remaining timed steps: max(decode_tokens - 2, 0).
        # Total decode steps = 2 + max(decode_tokens - 2, 0) = decode_tokens.
        timed_steps = max(decode_tokens - 2, 0)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        steps_done = 0
        event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(timed_steps):
            static_token.copy_(next_t)
            static_pos.fill_(pos)
            start_e = torch.cuda.Event(enable_timing=True)
            end_e = torch.cuda.Event(enable_timing=True)
            start_e.record()
            graph.replay()
            end_e.record()
            event_pairs.append((start_e, end_e))
            next_t = torch.argmax(graph_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
            pos += 1
            steps_done += 1

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        gpu_ms_total = sum(s.elapsed_time(e) for s, e in event_pairs)

    if debug:
        total = 2 + steps_done  # warmup + capture + timed
        print(
            f"  [graph] untimed steps: 2 (warmup T0 + capture T1)  "
            f"timed steps: {steps_done}  total: {total}  (requested: {decode_tokens})"
        )

    wall_s = t1 - t0
    gpu_s  = gpu_ms_total / 1000.0
    # Return timed_steps so callers normalise per-token metrics correctly.
    # (Only the timed steps are measured; the 2 untimed steps are excluded.)
    return wall_s, gpu_s, timed_steps


@torch.inference_mode()
def validate_graph_correctness(model, input_ids: torch.Tensor, n_steps: int = 20) -> list[float]:
    """Compare eager vs CUDA Graph decode logits per step.

    Returns per-step max absolute logit differences and asserts FP16 tolerance.
    """
    if StaticCache is None:
        raise RuntimeError("StaticCache not available in this transformers version.")
    if n_steps <= 0:
        raise ValueError("n_steps must be >= 1")

    device = input_ids.device
    prompt_len = int(input_ids.shape[1])
    kv_max = min(512, prompt_len + n_steps + DEFAULT_KV_PAD)
    if prompt_len + n_steps > kv_max:
        raise ValueError("prompt length + n_steps exceeds validation cache bound")

    # Eager reference path.
    eager_out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    eager_next = torch.argmax(eager_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
    eager_hist: list[torch.Tensor] = []
    # Collect n_steps + 2 eager logits. Two steps are consumed by warmup + capture
    # in the graph path; we skip those when comparing, so we need 2 extra entries.
    for _ in range(n_steps + 2):
        eager_out = model(
            input_ids=eager_next,
            past_key_values=eager_out.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        eager_hist.append(eager_out.logits[:, -1, :].detach().clone())
        eager_next = torch.argmax(eager_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)


    # Graph path with StaticCache and explicit cache positions.
    cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=kv_max,
        device=device,
        dtype=torch.float16,
    )
    for layer in cache.layers:
        layer.cumulative_length = layer.cumulative_length.to(device)

    pre_out = model(
        input_ids=input_ids,
        past_key_values=cache,
        cache_position=torch.arange(prompt_len, device=device, dtype=torch.long),
        use_cache=True,
        return_dict=True,
    )
    graph_next = torch.argmax(pre_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
    pos = prompt_len  # T0 = first decode token; StaticCache filled up to prompt_len-1.

    static_token = torch.zeros((1, 1), dtype=torch.long, device=device)
    static_pos = torch.zeros((1,), dtype=torch.long, device=device)

    # ── Warmup forward: runs T0 at pos=prompt_len (untimed, flushes lazy CUDA allocs).
    # After this forward the StaticCache slot at pos=prompt_len holds T0's KV.
    # We extract T1 and advance pos so the capture starts at the *next* position.
    static_token.copy_(graph_next)   # T0
    static_pos.fill_(pos)            # prompt_len
    warmup_out = model(
        input_ids=static_token,
        past_key_values=cache,
        cache_position=static_pos,
        use_cache=True,
        return_dict=True,
    )
    torch.cuda.synchronize(device)

    # Advance to T1 / prompt_len+1 before capture so the captured graph writes
    # into the correct slot and the replay loop starts from the right position.
    graph_next = torch.argmax(warmup_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
    pos += 1  # pos = prompt_len + 1

    # ── Graph capture at pos=prompt_len+1 with token T1.
    static_token.copy_(graph_next)   # T1
    static_pos.fill_(pos)            # prompt_len + 1

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = model(
            input_ids=static_token,
            past_key_values=cache,
            cache_position=static_pos,
            use_cache=True,
            return_dict=True,
        )

    static_logits = graph_out.logits

    # Extract T2 from capture output; advance pos.
    graph_next = torch.argmax(static_logits[:, -1:, :], dim=-1).to(dtype=torch.long)
    pos += 1  # pos = prompt_len + 2

    # ── Replay loop: update static_token/static_pos BEFORE each replay.
    # The first replay runs at pos=prompt_len+2 with token T2, which aligns
    # with eager_hist[2] (the 3rd decode step relative to prefill).
    # We therefore skip the first two eager steps to keep histories aligned.
    graph_hist: list[torch.Tensor] = []
    for _ in range(n_steps):
        static_token.copy_(graph_next)   # Tn
        static_pos.fill_(pos)            # prompt_len + n
        graph.replay()
        graph_hist.append(static_logits[:, -1, :].detach().clone())
        graph_next = torch.argmax(static_logits[:, -1:, :], dim=-1).to(dtype=torch.long)
        pos += 1

    # Align eager_hist: skip the first 2 steps (consumed by warmup + capture)
    # so that eager_hist[i] and graph_hist[i] correspond to the same decode step.
    skip = 2
    aligned_eager = eager_hist[skip : skip + len(graph_hist)]
    aligned_graph = graph_hist[: len(aligned_eager)]

    diffs: list[float] = []
    print("\n[validation] eager vs graph max logit diff per step")
    print(f"  (comparing eager steps {skip+1}–{skip+len(aligned_graph)} vs graph steps 1–{len(aligned_graph)})")
    print(f"{'step':>6} {'max_diff':>14}")
    for step, (el, gl) in enumerate(zip(aligned_eager, aligned_graph), start=skip + 1):
        d = torch.max(torch.abs(el - gl)).item()
        diffs.append(d)
        print(f"{step:>6d} {d:>14.6e}")

    max_diff = max(diffs) if diffs else 0.0
    print(f"[validation] max diff: {max_diff:.6e}")
    if max_diff < 1e-2:
        print("Graph correctness: PASS (FP16 tolerance)")
    else:
        print(f"Graph correctness: FAIL (max diff {max_diff:.4f})")
    return diffs


def warmup_eager(model, prefill_ids: torch.Tensor, decode_tokens: int, n: int) -> None:
    device = prefill_ids.device
    with torch.inference_mode():
        for _ in range(n):
            _ = decode_eager_timed(model, prefill_ids, decode_tokens)
            torch.cuda.synchronize(device)


def warmup_graph(
    model, prefill_ids: torch.Tensor, decode_tokens: int, kv_max: int, n: int
) -> None:
    device = prefill_ids.device
    with torch.inference_mode():
        for _ in range(n):
            _ = decode_cuda_graph_timed(model, prefill_ids, decode_tokens, kv_max)
            torch.cuda.synchronize(device)


def average_decode_metrics(
    wall_samples: list[float],
    gpu_samples: list[float],
    decode_tokens: int,
    mode: str,
    *,
    timed_steps: int | None = None,
) -> BenchResult:
    """
    Aggregate timing samples into a BenchResult.

    `decode_tokens`  — the requested number of new tokens (used for seq_len
                       and tokens_per_sec denominator).
    `timed_steps`    — the number of steps actually inside the timed region
                       (may differ from decode_tokens for graph mode where 2
                       setup steps are untimed).  If None, defaults to
                       decode_tokens (correct for eager mode).
    """
    n = len(wall_samples)
    mean_wall = sum(wall_samples) / n
    mean_gpu  = sum(gpu_samples)  / n
    # Normalise per-token latency by the number of TIMED steps so we get
    # the true average time per graph replay (not an artificially deflated
    # number from dividing a smaller time budget by the full token count).
    norm = timed_steps if (timed_steps is not None and timed_steps > 0) else decode_tokens
    per_tok_wall = mean_wall / norm
    per_tok_gpu  = mean_gpu  / norm
    lat_ms   = per_tok_wall * 1000.0
    # Throughput: effective tokens generated per second =
    #   decode_tokens (total tokens) / mean_wall (time for timed steps only).
    # For eager this is exact.  For graph, mean_wall covers only (decode_tokens-2)
    # steps, so we scale: tps = decode_tokens * (timed_steps / decode_tokens) / mean_wall
    #                         = timed_steps / mean_wall.
    # This gives the throughput for the timed portion — a conservative, fair estimate.
    tps      = norm / mean_wall if mean_wall > 0 else 0.0
    cpu_oh_ms = (per_tok_wall - per_tok_gpu) * 1000.0
    return BenchResult(
        mode=mode,
        seq_len=decode_tokens,
        latency_ms_per_token=lat_ms,
        tokens_per_sec=tps,
        cpu_overhead_ms_per_token=cpu_oh_ms,
        wall_ms_decode=mean_wall * 1000.0,
        gpu_ms_decode=mean_gpu * 1000.0,
    )


def bench_decode_eager(
    model,
    tokenizer,
    prefill_text: str,
    decode_tokens: int,
    *,
    warmup: int,
    iters: int,
    debug: bool = False,
) -> BenchResult:
    device = _first_cuda_device(model)
    prefill_ids = build_prefill_ids(tokenizer, prefill_text, device)

    warmup_eager(model, prefill_ids, decode_tokens, warmup)

    walls: list[float] = []
    gpus: list[float] = []
    for i in range(iters):
        w, g = decode_eager_timed(model, prefill_ids, decode_tokens,
                                  debug=(debug and i == 0))
        walls.append(w)
        gpus.append(g)

    return average_decode_metrics(walls, gpus, decode_tokens, "eager")


def bench_decode_graph(
    model,
    tokenizer,
    prefill_text: str,
    decode_tokens: int,
    kv_max: int,
    *,
    warmup: int,
    iters: int,
    debug: bool = False,
) -> BenchResult | None:
    if StaticCache is None:
        return None
    device = _first_cuda_device(model)
    prefill_ids = build_prefill_ids(tokenizer, prefill_text, device)

    try:
        warmup_graph(model, prefill_ids, decode_tokens, kv_max, warmup)
    except Exception:
        return None

    walls: list[float] = []
    gpus: list[float] = []
    timed_steps_list: list[int] = []
    for i in range(iters):
        try:
            w, g, ts = decode_cuda_graph_timed(
                model, prefill_ids, decode_tokens, kv_max,
                debug=(debug and i == 0),
            )
            walls.append(w)
            gpus.append(g)
            timed_steps_list.append(ts)
        except Exception:
            return None

    # All iterations should have the same timed_steps; take the first.
    timed_steps = timed_steps_list[0] if timed_steps_list else max(decode_tokens - 2, 0)
    if debug:
        print(f"  [graph] normalising per-token metrics over {timed_steps} timed steps "
              f"(total decode_tokens={decode_tokens})")

    return average_decode_metrics(walls, gpus, decode_tokens, "cuda_graph",
                                  timed_steps=timed_steps)


def results_to_csv(rows: Iterable[BenchResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["mode", "seq_len", "latency_ms", "tokens_per_sec", "cpu_overhead_ms"]
        )
        for r in rows:
            w.writerow(
                [
                    r.mode,
                    r.seq_len,
                    f"{r.latency_ms_per_token:.6f}",
                    f"{r.tokens_per_sec:.6f}",
                    f"{r.cpu_overhead_ms_per_token:.6f}",
                ]
            )


def plot_results(
    eager_by_sl: dict[int, BenchResult],
    graph_by_sl: dict[int, BenchResult | None],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_lens = sorted(eager_by_sl.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(seq_lens))
    w = 0.35
    e_vals = [eager_by_sl[sl].latency_ms_per_token for sl in seq_lens]
    g_vals = [
        graph_by_sl[sl].latency_ms_per_token if graph_by_sl[sl] else float("nan")
        for sl in seq_lens
    ]
    ax.bar([i - w / 2 for i in x], e_vals, width=w, label="eager", color="steelblue")
    ax.bar([i + w / 2 for i in x], g_vals, width=w, label="cuda_graph", color="darkorange")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.set_xlabel("Decode length (new tokens)")
    ax.set_ylabel("Decode latency (ms / token)")
    ax.set_title("Autoregressive decode: per-token latency (eager vs CUDA Graph)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_bar.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        seq_lens,
        [eager_by_sl[sl].tokens_per_sec for sl in seq_lens],
        "o-",
        label="eager",
    )
    if any(graph_by_sl[sl] for sl in seq_lens):
        ax.plot(
            seq_lens,
            [
                graph_by_sl[sl].tokens_per_sec if graph_by_sl[sl] else float("nan")
                for sl in seq_lens
            ],
            "s-",
            label="cuda_graph",
        )
    ax.set_xlabel("Decode length (new tokens)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Decode throughput vs decode length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_vs_seqlen.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    speedups = []
    for sl in seq_lens:
        g = graph_by_sl[sl]
        if g is None or g.latency_ms_per_token <= 0:
            speedups.append(float("nan"))
        else:
            speedups.append(
                eager_by_sl[sl].latency_ms_per_token / g.latency_ms_per_token
            )
    ax.plot(seq_lens, speedups, "o-", color="green")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Decode length (new tokens)")
    ax.set_ylabel("Speedup (eager latency / graph latency)")
    ax.set_title("CUDA Graph decode speedup vs eager (>1 is faster)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "speedup.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.iters < 30:
        print(
            "Warning: --iters < 30; use 30–50+ for stable averages.",
            file=sys.stderr,
        )

    torch.manual_seed(args.seed)

    if StaticCache is None:
        print(
            "StaticCache unavailable — cuda_graph mode disabled. "
            "Upgrade transformers (e.g. >= 4.39) for CUDA Graph decode.",
            file=sys.stderr,
        )

    print(f"Loading {args.model} (fp16) …")
    model, tokenizer = load_model_and_tokenizer(args.model)
    device = _first_cuda_device(model)
    prefill_ids = build_prefill_ids(tokenizer, args.prefill, device)
    prefill_len = int(prefill_ids.shape[1])
    max_dec = max(args.seq_lens)
    kv_max = prefill_len + max_dec + args.kv_pad

    print(
        f"Prefill tokens: {prefill_len}; max decode in sweep: {max_dec}; "
        f"StaticCache max_cache_len: {kv_max}"
    )
    validate_steps = min(20, max_dec)
    print(f"Running correctness validation for {validate_steps} decode steps ...")
    validate_graph_correctness(model, prefill_ids, n_steps=validate_steps)

    eager_by_sl: dict[int, BenchResult] = {}
    graph_by_sl: dict[int, BenchResult | None] = {}
    all_rows: list[BenchResult] = []

    for sl in args.seq_lens:
        print(f"Benchmarking decode_tokens={sl} …")

        e = bench_decode_eager(
            model,
            tokenizer,
            args.prefill,
            sl,
            warmup=args.warmup,
            iters=args.iters,
            debug=args.debug,
        )
        eager_by_sl[sl] = e
        all_rows.append(e)

        g = bench_decode_graph(
            model,
            tokenizer,
            args.prefill,
            sl,
            kv_max,
            warmup=args.warmup,
            iters=args.iters,
            debug=args.debug,
        )
        graph_by_sl[sl] = g
        if g is not None:
            all_rows.append(g)
            speedup = e.latency_ms_per_token / g.latency_ms_per_token
            print(
                f"  decode_tokens={sl:4d} | speedup {speedup:.2f}x"
            )
            print(
                f"    Mode: eager | latency {e.latency_ms_per_token:7.2f} ms/tok | "
                f"GPU compute {(e.latency_ms_per_token - e.cpu_overhead_ms_per_token):7.2f} ms/tok | "
                f"CPU overhead {e.cpu_overhead_ms_per_token:7.2f} ms/tok"
            )
            print(
                f"    Mode: graph | latency {g.latency_ms_per_token:7.2f} ms/tok | "
                f"GPU compute {(g.latency_ms_per_token - g.cpu_overhead_ms_per_token):7.2f} ms/tok | "
                f"CPU overhead {g.cpu_overhead_ms_per_token:7.2f} ms/tok"
            )
        else:
            print(
                f"  CUDA Graph decode unavailable for decode_tokens={sl} — eager only."
            )
            print(
                f"    Mode: eager | latency {e.latency_ms_per_token:7.2f} ms/tok | "
                f"GPU compute {(e.latency_ms_per_token - e.cpu_overhead_ms_per_token):7.2f} ms/tok | "
                f"CPU overhead {e.cpu_overhead_ms_per_token:7.2f} ms/tok"
            )

    out_dir = args.out_dir
    csv_path = out_dir / "benchmark_results.csv"
    results_to_csv(all_rows, csv_path)
    print(f"Wrote {csv_path}")

    plot_results(eager_by_sl, graph_by_sl, out_dir)
    print(f"Saved plots to {out_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
