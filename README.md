# CUDA Graph Optimised LLM Inference on Low-Resource GPUs

> **Dissertation project** — Migrating Qwen2.5-0.5B-Instruct autoregressive decode from `DynamicCache` to `StaticCache` with CUDA Graph capture, achieving up to **4.66× decode speedup** on an RTX 3050 Laptop GPU (4 GB VRAM).

---

## The Problem

Autoregressive LLM decoding is **memory-bandwidth-bound** at small batch sizes: each decode step issues a single-token forward pass that produces only a handful of FLOPs while reading the entire KV cache from HBM. On a 4 GB mobile GPU this means:

* **Python/CUDA launch overhead dominates** — the PyTorch runtime re-dispatches every operator on every step, contributing significant per-token CPU scheduling latency.
* **DynamicCache grows dynamically** — each decode step concatenates a new KV slice, which can trigger reallocations and pointer invalidations. Inside a captured CUDA Graph this is fatal.
* At sequence lengths 32–256, the GPU is idle for a large fraction of wall-clock time waiting for the CPU to launch the next kernel.

---

## My Approach

| Phase | What I did |
|---|---|
| **Cache migration** | Replaced `DynamicCache` with `StaticCache` — pre-allocates the full KV tensor slab once at graph capture time; decode writes into fixed slots via `cache_position`. |
| **CUDA Graph capture** | Captures the single-token decode forward pass (`model(input_ids, past_key_values=cache, cache_position=pos)`) as a `torch.cuda.CUDAGraph`. Replay skips all Python dispatch overhead. |
| **In-place tensor updates** | `static_token.copy_(next_token)` and `static_pos.fill_(pos)` update the captured graph's input buffers in-place before each `graph.replay()`. No allocations inside the captured region. |
| **Prefill separation** | Prefill runs eagerly (untimed) so that the KV cache is populated before capture. Only the decode loop is timed. |

---

## The Bug I Found and Fixed

When I first attempted CUDA Graph capture with `DynamicCache`, logit divergence grew monotonically:

```
step   1 | max_diff  1.2e-03
step   5 | max_diff  0.84
step  10 | max_diff  4.31
step  15 | max_diff 11.7
step  20 | max_diff 27.4   ← MAE blows up
```

**Root cause:** `DynamicCache` concatenates a new KV tensor slice on every step. Inside a captured CUDA Graph, the graph records the *memory addresses* of the old (pre-capture) KV tensors. After one decode step the cache reallocates to a larger buffer — the graph keeps writing into the now-stale addresses, producing garbage logits that diverge progressively.

**Fix:** `StaticCache` pre-allocates the full KV slab at construction time. The memory addresses are fixed for the entire decode sequence. The captured graph always writes into the correct cache slots via `cache_position`, and logit max-diff stays at `< 1e-2` across 20 validation steps.

---

## Results

Numbers pulled directly from `results/benchmark_results.csv` (40 decode iterations per cell, 3-step warmup, RTX 3050 Laptop 4 GB VRAM, FP16):

| Decode length (new tokens) | Eager latency (ms/tok) | CUDA Graph latency (ms/tok) | Throughput — Eager (tok/s) | Throughput — Graph (tok/s) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 32  | 73.60 | 8.49 | 13.59 | 117.78 | **8.67×** |
| 64  | 38.74 | 8.64 | 25.81 | 115.77 | **4.48×** |
| 128 | 35.87 | 8.41 | 27.88 | 118.96 | **4.27×** |
| 256 | 40.84 | 8.74 | 24.48 | 114.41 | **4.67×** |

> The outsized speedup at seq_len=32 reflects Python/CUDA launch overhead dominating eager at very short decode runs. The 4.27–4.67× range at 64–256 tokens is the steady-state contribution of graph replay.

---

## Comparison with PMPD

As part of this dissertation I benchmarked against the **Progressive Mixed-Precision Decoding (PMPD)** method, which improves LLM inference throughput via phase-aware precision allocation and progressive precision lowering during decode.

* **Official PMPD repository:** [https://github.com/SamsungLabs/PMPD](https://github.com/SamsungLabs/PMPD)
* **Paper:** *Progressive Mixed-Precision Decoding for Efficient LLM Inference* — arXiv [2410.13461](https://arxiv.org/abs/2410.13461), ICLR 2025

> **Note:** The PMPD codebase is licensed CC BY-NC-SA 4.0 (Samsung Labs). **None of their source code or assets are included here.** The scripts in this repo (`test_graph.py`, `test_graph_performance.py`, `validate.py`) use the `pmpd` package as a declared benchmarking dependency (see `requirements.txt`) — it is fetched from upstream at install time, not vendored.

---

## ⚠ `pmpd` Package — Benchmarking Dependency

`test_graph.py`, `test_graph_performance.py`, and `validate.py` import from the `pmpd` package at the top level. This is a **declared dependency** in `requirements.txt` installed via:

```bash
pip install git+https://github.com/SamsungLabs/PMPD.git
```

> **Note:** `pmpd` is not on PyPI — `pip install -r requirements.txt` fetches it directly from [SamsungLabs/PMPD](https://github.com/SamsungLabs/PMPD) on GitHub. It is a benchmarking reference dependency, **not vendored code** — no PMPD source is distributed in this repo. The CC BY-NC-SA 4.0 license applies to the PMPD package itself when installed.

**Scripts that do NOT require `pmpd`** (clean, self-contained):
- `app.py` — Streamlit demo app
- `benchmark_qwen_cuda_graph.py` — Primary benchmark script
- `benchmark_extended.py` — Extended validation
- `test_cuda_graph.py` — CUDA Graph unit test
- `scratch_text_gen.py`, `scratch_inspect_attention.py` — Dev scratch scripts
- `trace_qwen.py` — Tracing utility

**Scripts that require `pmpd`** (install via `requirements.txt`):
- `test_graph.py`
- `test_graph_performance.py`
- `validate.py`

---

## Hardware & Software

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) |
| OS | Windows 11 |
| Python | 3.10 |
| PyTorch | CUDA-enabled (tested with CUDA 11.8 / 12.x) |
| Transformers | ≥ 4.39 (required for `StaticCache`) |
| Model | `Qwen/Qwen2.5-0.5B` (FP16, HuggingFace Hub) |

---

## Setup

```bash
# Clone this repo
git clone https://github.com/muhammedmujthabak/<this-repo>.git
cd <this-repo>

# Install dependencies (minimal for the benchmark scripts)
pip install torch transformers accelerate matplotlib streamlit
```

> The full `requirements.txt` includes extra packages used during the PMPD comparison phase; most are not needed to run the core benchmark scripts.

---

## How to Run

### 1. Primary Benchmark (`benchmark_qwen_cuda_graph.py`)

Sweeps decode lengths [32, 64, 128, 256], outputs `results/benchmark_results.csv` and three plots.

```bash
python benchmark_qwen_cuda_graph.py --iters 40 --warmup 3
```

Optional flags:
```
--model          Qwen/Qwen2.5-0.5B (default)
--seq-lens       32 64 128 256
--iters          40   (number of timed iterations per cell)
--warmup         3
--out-dir        results/
--debug          Print per-iteration step counts
```

### 2. Streamlit Demo App (`app.py`)

Interactive chat interface with switchable eager / CUDA Graph mode and live performance metrics.

```bash
streamlit run app.py
```

### 3. Extended Benchmark (`benchmark_extended.py`)

Validates correctness (logit diff) across sequence lengths alongside latency.

```bash
python benchmark_extended.py
```

### 4. CUDA Graph Unit Test (`test_cuda_graph.py`)

Quick smoke-test for the CUDA Graph capture/replay pipeline.

```bash
python test_cuda_graph.py
```

### 5. Graph Correctness Validation (`test_graph_performance.py`)

> Requires the `pmpd` package (installed via `requirements.txt`) — used here as a benchmarking dependency, not vendored code.

Full side-by-side eager vs graph correctness and latency comparison.

```bash
python test_graph_performance.py --max-new-tokens 50
```

### 6. PMPD Comparison Benchmark (`test_graph.py`)

> Requires the `pmpd` package (installed via `requirements.txt`) — used here as a benchmarking dependency, not vendored code.

Simulated PMPD vs CUDA Graph decode benchmark (Section 5.2 of the dissertation).

```bash
python test_graph.py --experiments core
# or --demo for a synthetic MLP smoke-test with no model weights required
python test_graph.py --demo
```

### 7. Graph Correctness Validator (`validate.py`)

> Requires the `pmpd` package (installed via `requirements.txt`) — used here as a benchmarking dependency, not vendored code.

Step-by-step eager vs CUDA Graph logit comparison.

```bash
python validate.py
```

---

## Results Folder

```
results/
  benchmark_results.csv        # Primary latency/throughput sweep (40 iters)
  summary_metrics.csv          # Aggregated speedup summary
  latency_bar.png              # Per-token latency bar chart
  throughput_vs_seqlen.png     # Throughput vs decode length
  speedup.png                  # Speedup vs decode length
  latency_comparison.png       # Alternative latency visualisation
  throughput_comparison.png    # Alternative throughput visualisation
  speedup_trend.png            # Speedup trend line
  performance_evaluation_section.md  # Dissertation section notes
```

---

## License

This project is licensed under the MIT License — see [LICENSE.md](LICENSE.md).

The PMPD method and its official implementation are the work of Samsung Labs and are licensed separately under CC BY-NC-SA 4.0. No PMPD source code or assets are distributed in this repository.
