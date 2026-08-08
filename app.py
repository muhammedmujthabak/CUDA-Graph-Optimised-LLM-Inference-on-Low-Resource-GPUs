import time
from dataclasses import dataclass

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import StaticCache
except Exception:
    StaticCache = None


MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_MAX_NEW_TOKENS = 50        # spec: limit max_new_tokens = 50
MAX_CACHE_LEN = 512
TEMPERATURE = 0.7                  # spec: temperature = 0.7
TOP_P = 0.95
BENCHMARK_DECODE_STEPS  = 128     # tokens per timed repeat (safe for RTX 3050)
BENCHMARK_REPEATS       = 3       # timed repeats to average (reduces noise)
BENCHMARK_WARMUP_STEPS  = 5      # untimed warmup steps to stabilise GPU


@dataclass
class RunMetrics:
    tokens_per_sec: float
    latency_ms_per_token: float
    gpu_ms_per_token: float
    cpu_overhead_ms_per_token: float


def _sample_next_token(
    logits_last: torch.Tensor,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """Sample the next token using temperature scaling + multinomial.

    Spec:
        probs = torch.softmax(logits[:, -1, :] / 0.7, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    """
    # spec: temperature = 0.7 scaling then multinomial sampling
    probs = torch.softmax(logits_last / max(temperature, 1e-5), dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _safe_metrics(tokens: int, total_wall_ms: float, gpu_ms_total: float) -> RunMetrics:
    """Compute per-token metrics with spec-compliant formulas.

    Spec:
        latency_ms       = (total_time / tokens) * 1000
        tokens_per_sec   = tokens / total_time
        gpu_ms           = (gpu_time / tokens) * 1000
        cpu_overhead     = max(latency_ms - gpu_ms, 0)   ← clamped
    """
    tokens = max(int(tokens), 1)
    total_wall_s  = max(float(total_wall_ms) / 1000.0, 1e-9)   # seconds
    gpu_time_s    = max(float(gpu_ms_total)  / 1000.0, 0.0)    # seconds

    latency_ms    = (total_wall_s / tokens) * 1000.0
    tokens_per_s  = tokens / total_wall_s
    gpu_ms_pt     = (gpu_time_s  / tokens) * 1000.0
    cpu_oh_ms     = max(latency_ms - gpu_ms_pt, 0.0)           # spec: clamp ≥ 0

    return RunMetrics(
        tokens_per_sec=tokens_per_s,
        latency_ms_per_token=latency_ms,
        gpu_ms_per_token=gpu_ms_pt,
        cpu_overhead_ms_per_token=cpu_oh_ms,
    )


@st.cache_resource(show_spinner="Loading Qwen2.5-0.5B (FP16)...")
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _build_qwen_prompt(user_input: str) -> str:
    """Build the correct Qwen chat prompt format.

    Spec:
        prompt = f"You are a helpful assistant.\nUser: {user_input}\nAssistant:"
    """
    return f"You are a helpful assistant.\nUser: {user_input}\nAssistant:"


def _decode_eager(model, tokenizer, user_input: str, max_new_tokens: int):
    """Normal eager autoregressive generation (benchmark_mode OFF).

    Spec requirements:
      * use_cache=True, past_key_values updated every step
      * input_ids = next_token each step
      * multinomial sampling at temperature 0.7
      * stop on EOS
      * CUDA sync before start, after end
      * GPU Event timing per step
    """
    prompt = _build_qwen_prompt(user_input)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device) if "attention_mask" in enc else torch.ones_like(input_ids)

    with torch.inference_mode():
        # ── Prefill ────────────────────────────────────────────────────────
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        if out.logits.ndim != 3:
            raise RuntimeError(f"Unexpected logits rank: {out.logits.shape}")

        # spec: probs = softmax(logits[:, -1, :] / 0.7);  next_token = multinomial
        next_token = _sample_next_token(out.logits[:, -1, :])
        past = out.past_key_values

        generated_ids: list[int] = []
        event_pairs: list[tuple] = []

        # spec: torch.cuda.synchronize() BEFORE timing start
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        for _ in range(max_new_tokens):
            token_id = int(next_token[0, 0].item())
            generated_ids.append(token_id)

            # spec: stop on EOS token
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

            # GPU Event timing per step
            start_e = torch.cuda.Event(enable_timing=True)
            end_e   = torch.cuda.Event(enable_timing=True)
            start_e.record()

            # spec: input_ids = next_token each step; past_key_values updated
            out = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )

            end_e.record()
            event_pairs.append((start_e, end_e))

            past       = out.past_key_values
            next_token = _sample_next_token(out.logits[:, -1, :])

        # spec: torch.cuda.synchronize() AFTER loop
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

    # Collect GPU event times only after sync
    gpu_ms_total  = sum(s.elapsed_time(e) for s, e in event_pairs)
    total_wall_ms = (t1 - t0) * 1000.0
    tokens        = len(generated_ids)

    text    = tokenizer.decode(generated_ids, skip_special_tokens=True)
    metrics = _safe_metrics(tokens, total_wall_ms, gpu_ms_total)
    return text, metrics


def _benchmark_eager(
    model,
    tokenizer,
    user_input: str,
    steps: int    = BENCHMARK_DECODE_STEPS,
    repeats: int  = BENCHMARK_REPEATS,
    warmup: int   = BENCHMARK_WARMUP_STEPS,
) -> RunMetrics:
    """Benchmark-mode eager path — averaged over multiple repeats.

    Pipeline (benchmark_mode ON, CUDA Graph OFF):
      1. Prefill once  (not timed)
      2. Warmup  `warmup` steps  (not timed, GPU pipeline stabilisation)
      3. Repeat  `repeats` times:
           sync → wall-clock start → GPU Event start
           decode `steps` tokens
           GPU Event end → sync → wall-clock end
           accumulate wall_ms + gpu_ms
      4. Average accumulated totals → compute metrics
    Returns ONLY a RunMetrics object — no generated text.
    """
    prompt = _build_qwen_prompt(user_input)
    device = next(model.parameters()).device
    enc    = tokenizer(prompt, return_tensors="pt")
    input_ids      = enc.input_ids.to(device)
    attention_mask = (
        enc.attention_mask.to(device)
        if "attention_mask" in enc
        else torch.ones_like(input_ids)
    )

    with torch.inference_mode():
        # ── 1. Prefill (not timed) ─────────────────────────────────────────
        out  = model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=True, return_dict=True,
        )
        next_token = _sample_next_token(out.logits[:, -1, :])
        past       = out.past_key_values

        # ── 2. Warmup: stabilise GPU pipeline (not timed) ──────────────────
        torch.cuda.synchronize(device)
        for _ in range(warmup):
            out        = model(input_ids=next_token, past_key_values=past,
                               use_cache=True, return_dict=True)
            past       = out.past_key_values
            next_token = _sample_next_token(out.logits[:, -1, :])
        torch.cuda.synchronize(device)

        # ── 3. Timed repeats ───────────────────────────────────────────────
        total_wall_ms = 0.0
        total_gpu_ms  = 0.0

        for rep in range(repeats):
            event_pairs: list[tuple] = []

            # Strict sync BEFORE each repeat
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            rep_start = torch.cuda.Event(enable_timing=True)
            rep_end   = torch.cuda.Event(enable_timing=True)
            rep_start.record()

            for _ in range(steps):
                se = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                se.record()
                out = model(
                    input_ids=next_token, past_key_values=past,
                    use_cache=True, return_dict=True,
                )
                ee.record()
                event_pairs.append((se, ee))
                past       = out.past_key_values
                next_token = _sample_next_token(out.logits[:, -1, :])

            rep_end.record()
            # Strict sync AFTER each repeat
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            total_wall_ms += (t1 - t0) * 1000.0
            total_gpu_ms  += sum(s.elapsed_time(e) for s, e in event_pairs)

    # ── 4. Average over repeats → compute metrics ──────────────────────────
    avg_wall_ms = total_wall_ms / repeats
    avg_gpu_ms  = total_gpu_ms  / repeats
    return _safe_metrics(steps, avg_wall_ms, avg_gpu_ms)


def _benchmark_graph(
    model,
    tokenizer,
    user_input: str,
    steps: int    = BENCHMARK_DECODE_STEPS,
    repeats: int  = BENCHMARK_REPEATS,
    warmup: int   = BENCHMARK_WARMUP_STEPS,
) -> RunMetrics:
    """Benchmark-mode CUDA Graph path — averaged over multiple repeats.

    Pipeline (benchmark_mode ON, CUDA Graph ON):
      1. Prefill eagerly into StaticCache  (not timed)
      2. Capture CUDA Graph for the single-token decode step
      3. Warmup  `warmup` replay calls    (not timed)
      4. Repeat  `repeats` times:
           sync → wall-clock start → GPU Event start
           replay graph `steps` times  (static tensors updated each step)
           GPU Event end → sync → wall-clock end
           accumulate wall_ms + gpu_ms
      5. Average accumulated totals → compute metrics
    Returns ONLY a RunMetrics object — no generated text.
    CUDA constraints:
      * No .item() inside the captured region
      * All allocations are outside the capture
      * static_token / static_pos are updated via .copy_() / .fill_()
    """
    if StaticCache is None:
        raise RuntimeError("StaticCache not available; upgrade transformers.")

    prompt = _build_qwen_prompt(user_input)
    device = next(model.parameters()).device
    enc    = tokenizer(prompt, return_tensors="pt")
    input_ids      = enc.input_ids.to(device)
    attention_mask = (
        enc.attention_mask.to(device)
        if "attention_mask" in enc
        else torch.ones_like(input_ids)
    )
    prompt_len = int(input_ids.shape[1])

    # Safety: total positions needed = prompt + warmup + repeats*steps
    total_positions = prompt_len + warmup + repeats * steps
    if total_positions >= MAX_CACHE_LEN:
        raise RuntimeError(
            f"Prompt ({prompt_len}) + warmup ({warmup}) + "
            f"{repeats}×{steps} steps = {total_positions} >= MAX_CACHE_LEN ({MAX_CACHE_LEN}). "
            "Reduce BENCHMARK_DECODE_STEPS or MAX_CACHE_LEN."
        )

    with torch.inference_mode():
        # ── 1. Build StaticCache + Prefill (not timed) ─────────────────────
        cache = StaticCache(
            config=model.config, max_batch_size=1,
            max_cache_len=MAX_CACHE_LEN, device=device, dtype=torch.float16,
        )
        pre_out = model(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=torch.arange(prompt_len, device=device, dtype=torch.long),
            use_cache=True, return_dict=True,
        )
        next_token = _sample_next_token(pre_out.logits[:, -1, :])
        pos = prompt_len

        # Static tensors — NEVER reallocated; only .copy_() / .fill_() allowed
        static_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        static_pos   = torch.zeros((1,),   dtype=torch.long, device=device)

        # ── 2. Graph capture ───────────────────────────────────────────────
        # One warmup pass in the capture stream before recording
        static_token.copy_(next_token)
        static_pos.fill_(pos)
        _ = model(
            input_ids=static_token, past_key_values=cache,
            cache_position=static_pos, use_cache=True, return_dict=True,
        )
        # Advance state so capture starts at a clean decode position
        next_token = _sample_next_token(_.logits[:, -1, :])
        pos += 1
        static_token.copy_(next_token)
        static_pos.fill_(pos)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # No .item(), no dynamic alloc — only static tensor ops recorded
            graph_out = model(
                input_ids=static_token, past_key_values=cache,
                cache_position=static_pos, use_cache=True, return_dict=True,
            )
        # First replay to advance state past capture
        graph.replay()
        next_token = _sample_next_token(graph_out.logits[:, -1, :])
        pos += 1
        torch.cuda.synchronize(device)

        # ── 3. Warmup replays (not timed) ─────────────────────────────────
        for _ in range(warmup):
            static_token.copy_(next_token)
            static_pos.fill_(pos)
            graph.replay()
            next_token = _sample_next_token(graph_out.logits[:, -1, :])
            pos += 1
        torch.cuda.synchronize(device)

        # ── 4. Timed repeats ───────────────────────────────────────────────
        total_wall_ms = 0.0
        total_gpu_ms  = 0.0

        for rep in range(repeats):
            event_pairs: list[tuple] = []

            # Strict sync BEFORE each repeat
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            rep_start = torch.cuda.Event(enable_timing=True)
            rep_end   = torch.cuda.Event(enable_timing=True)
            rep_start.record()

            for _ in range(steps):
                static_token.copy_(next_token)
                static_pos.fill_(pos)

                se = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                se.record()
                graph.replay()
                ee.record()
                event_pairs.append((se, ee))

                # Sample OUTSIDE graph capture — no .item() inside replay
                next_token = _sample_next_token(graph_out.logits[:, -1, :])
                pos += 1

            rep_end.record()
            # Strict sync AFTER each repeat
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            total_wall_ms += (t1 - t0) * 1000.0
            total_gpu_ms  += sum(s.elapsed_time(e) for s, e in event_pairs)

    # ── 5. Average over repeats → compute metrics ──────────────────────────
    avg_wall_ms = total_wall_ms / repeats
    avg_gpu_ms  = total_gpu_ms  / repeats
    return _safe_metrics(steps, avg_wall_ms, avg_gpu_ms)


@torch.inference_mode()
def validate_graph_correctness(model, input_ids: torch.Tensor, steps: int = 10) -> tuple[bool, float]:
    """Compare per-step logits: eager vs CUDA Graph over `steps` decode steps.

    spec:
      * Run 10-step validation comparing eager vs graph logits.
      * IF max_diff > 1e-2: DISABLE CUDA Graph automatically.
    Uses argmax (greedy) for determinism during validation — sampling is
    NOT used here so both paths follow the same token trajectory.
    """
    if StaticCache is None:
        return False, float("inf")

    device = input_ids.device
    prompt_len = int(input_ids.shape[1])
    if prompt_len + steps >= MAX_CACHE_LEN:
        return False, float("inf")

    # ── Eager path (argmax for reproducibility) ────────────────────────
    eager_out  = model(input_ids=input_ids, use_cache=True, return_dict=True)
    eager_past = eager_out.past_key_values
    eager_next = torch.argmax(eager_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
    eager_hist = []
    for _ in range(steps):
        eager_out  = model(input_ids=eager_next, past_key_values=eager_past,
                           use_cache=True, return_dict=True)
        eager_past = eager_out.past_key_values
        eager_hist.append(eager_out.logits[:, -1, :].detach().clone())
        eager_next = torch.argmax(eager_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)

    # ── Graph path ─────────────────────────────────────────────────────
    cache = StaticCache(
        config=model.config, max_batch_size=1,
        max_cache_len=MAX_CACHE_LEN, device=device, dtype=torch.float16,
    )
    pre_out    = model(
        input_ids=input_ids, past_key_values=cache,
        cache_position=torch.arange(prompt_len, device=device, dtype=torch.long),
        use_cache=True, return_dict=True,
    )
    graph_next = torch.argmax(pre_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
    pos = prompt_len

    static_token = torch.zeros((1, 1), dtype=torch.long, device=device)
    static_pos   = torch.zeros((1,),   dtype=torch.long, device=device)
    static_token.copy_(graph_next)
    static_pos.fill_(pos)

    # Warmup pass before capture
    _ = model(input_ids=static_token, past_key_values=cache,
               cache_position=static_pos, use_cache=True, return_dict=True)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = model(
            input_ids=static_token, past_key_values=cache,
            cache_position=static_pos, use_cache=True, return_dict=True,
        )

    graph_hist = []
    for _ in range(steps):
        graph.replay()
        graph_hist.append(graph_out.logits[:, -1, :].detach().clone())
        graph_next = torch.argmax(graph_out.logits[:, -1:, :], dim=-1).to(dtype=torch.long)
        pos += 1
        static_token.copy_(graph_next)
        static_pos.fill_(pos)

    diffs = []
    print("[graph_val] step | max_diff")
    for idx, (el, gl) in enumerate(zip(eager_hist, graph_hist), start=1):
        d = torch.max(torch.abs(el - gl)).item()
        diffs.append(d)
        print(f"[graph_val] {idx:>4d} | {d:.6e}")
    max_diff = max(diffs) if diffs else 0.0
    # spec: IF max_diff > 1e-2 → DISABLE CUDA Graph automatically
    is_valid = max_diff <= 1e-2
    print(f"[graph_val] result: {'PASS' if is_valid else 'FAIL — CUDA Graph DISABLED'} "
          f"(max_diff={max_diff:.6e}, threshold=1e-2)")
    return is_valid, max_diff


st.set_page_config(page_title="PMPD Chatbot (Optimized Inference)", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #05070f 0%, #0b1d3a 45%, #07101f 100%);
        color: #e8eefc;
    }
    .panel-card {
        background: rgba(17, 24, 39, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
        min-height: 90px;
    }
    .panel-title {
        font-size: 0.88rem;
        color: #a7b7d8;
    }
    .panel-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e7f0ff;
    }
    .section-block {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.28);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PMPD Chatbot (Optimized Inference)")

if "generated_output" not in st.session_state:
    st.session_state.generated_output = ""
if "metrics" not in st.session_state:
    st.session_state.metrics = RunMetrics(0.0, 0.0, 0.0, 0.0)
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "graph_validated" not in st.session_state:
    st.session_state.graph_validated = False
if "graph_enabled" not in st.session_state:
    st.session_state.graph_enabled = False

# ── Input controls (always rendered first) ────────────────────────────────
st.markdown("<div class='section-block'>", unsafe_allow_html=True)
user_input = st.text_input("Your Input", value="", placeholder="Type your prompt...")

b1, b2, b3 = st.columns(3)
send_clicked  = b1.button("Send",  key="send_btn",  use_container_width=True, type="primary")
stop_clicked  = b2.button("Stop",  key="stop_btn",  use_container_width=True)
clear_clicked = b3.button("Clear", key="clear_btn", use_container_width=True)

s1, s2, s3 = st.columns([1, 1.2, 1.2])
max_new_tokens = s1.number_input(
    "Max New Tokens", min_value=1, max_value=256, value=DEFAULT_MAX_NEW_TOKENS,
    help="Used only in Normal mode. Benchmark mode always runs a fixed 32-step loop.",
)
execution_mode = s2.radio("Execution Mode", ["Eager", "CUDA Graph"], horizontal=True)
benchmark_mode = s3.toggle(
    "Benchmark Mode", value=False,
    help="ON → measure latency only (fixed 32 tokens, no text). OFF → generate full text.",
)
st.markdown("</div>", unsafe_allow_html=True)

st.caption("Model: Qwen2.5-0.5B (FP16)")

# ── Output area (below controls) ──────────────────────────────────────────
st.markdown("<div class='section-block'>", unsafe_allow_html=True)
if benchmark_mode:
    st.subheader("Benchmark Results")
    st.info("Benchmark mode: fixed 32-token decode loop. Metrics shown after run completes.")
else:
    st.subheader("Generated Output")
output_placeholder = st.empty()
if not benchmark_mode:
    output_placeholder.text_area(
        "Generated Output",
        value=st.session_state.generated_output,
        height=260,
        label_visibility="collapsed",
    )
st.markdown("</div>", unsafe_allow_html=True)

# ── Metrics panel — shown ONLY after generation completes ─────────────────
# spec: Display metrics ONLY after generation completes.
if st.session_state.metrics.tokens_per_sec > 0:
    st.markdown("### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    for col, title, value in [
        (col1, "Speed (tokens/sec)",     f"{st.session_state.metrics.tokens_per_sec:.2f}"),
        (col2, "Latency (ms/token)",     f"{st.session_state.metrics.latency_ms_per_token:.2f}"),
        (col3, "GPU Compute (ms/token)", f"{st.session_state.metrics.gpu_ms_per_token:.2f}"),
        (col4, "CPU Overhead (ms/token)",f"{st.session_state.metrics.cpu_overhead_ms_per_token:.2f}"),
    ]:
        with col:
            st.markdown(
                f"<div class='panel-card'><div class='panel-title'>{title}</div>"
                f"<div class='panel-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )

if clear_clicked:
    st.session_state.generated_output = ""
    st.session_state.metrics = RunMetrics(0.0, 0.0, 0.0, 0.0)
    st.session_state.stop_requested = False
    st.rerun()

if stop_clicked:
    st.session_state.stop_requested = True
    st.warning("Stop requested. It will be applied on the next generation run.")

if send_clicked:
    if not user_input.strip():
        st.warning("Please enter a prompt.")
    else:
        st.session_state.stop_requested = False
        model, tokenizer = load_model_and_tokenizer()

        # ── CUDA Graph validation (10 steps, spec §6) ─────────────────────
        # Run once per session; auto-disables graph if max_diff > 1e-2.
        if not st.session_state.graph_validated:
            with st.spinner("Validating CUDA Graph correctness (10 steps)..."):
                val_ids = tokenizer(
                    _build_qwen_prompt(user_input), return_tensors="pt"
                ).input_ids.to(next(model.parameters()).device)
                is_valid, max_diff = validate_graph_correctness(
                    model, val_ids, steps=10  # spec: 10-step validation
                )
            st.session_state.graph_validated = True
            st.session_state.graph_enabled   = bool(is_valid)
            if not is_valid:
                st.warning(
                    f"⚠ CUDA Graph correctness FAIL (max_diff={max_diff:.4f} > 1e-2). "
                    "CUDA Graph automatically disabled."
                )
            else:
                st.success(f"✓ CUDA Graph valid (max_diff={max_diff:.2e}). Graph enabled.")

        # ── Mode dispatch — strict separation (spec §2) ────────────────────
        with st.spinner("Running..." if benchmark_mode else "Generating..."):
            if benchmark_mode:
                # Strict separation: benchmark ON → measure ONLY, no text output.
                # Runs BENCHMARK_DECODE_STEPS-token loop averaged over BENCHMARK_REPEATS
                # repeats, with BENCHMARK_WARMUP_STEPS untimed GPU warmup steps.
                requested_graph = (execution_mode == "CUDA Graph")
                if requested_graph and st.session_state.graph_enabled:
                    metrics = _benchmark_graph(
                        model, tokenizer, user_input,
                        steps   = BENCHMARK_DECODE_STEPS,
                        repeats = BENCHMARK_REPEATS,
                        warmup  = BENCHMARK_WARMUP_STEPS,
                    )
                    mode_label = "CUDA Graph"
                else:
                    metrics = _benchmark_eager(
                        model, tokenizer, user_input,
                        steps   = BENCHMARK_DECODE_STEPS,
                        repeats = BENCHMARK_REPEATS,
                        warmup  = BENCHMARK_WARMUP_STEPS,
                    )
                    mode_label = "Eager"

                # Benchmark mode: no generated text
                st.session_state.generated_output = ""
                st.session_state.metrics = metrics

                # Rich summary displayed in the output panel
                output_placeholder.markdown(
                    f"""
**Benchmark complete — {mode_label}**

| Parameter | Value |
|---|---|
| Tokens per repeat | `{BENCHMARK_DECODE_STEPS}` |
| Repeats averaged  | `{BENCHMARK_REPEATS}` |
| Warmup steps      | `{BENCHMARK_WARMUP_STEPS}` |
| Speed             | `{metrics.tokens_per_sec:.2f}` tok/s |
| Latency           | `{metrics.latency_ms_per_token:.3f}` ms/tok |
| GPU compute       | `{metrics.gpu_ms_per_token:.3f}` ms/tok |
| CPU overhead      | `{metrics.cpu_overhead_ms_per_token:.3f}` ms/tok |
"""
                )
            else:
                # spec: benchmark OFF → normal eager generation ONLY, no CUDA Graph
                text, metrics = _decode_eager(
                    model, tokenizer, user_input, int(max_new_tokens)
                )
                # Normal mode: GPU event metrics are populated by _decode_eager;
                # pass them through as-is (cpu_overhead already clamped in _safe_metrics)
                st.session_state.generated_output = text
                st.session_state.metrics = metrics
                output_placeholder.text_area(
                    "Generated Output",
                    value=text,
                    height=260,
                    label_visibility="collapsed",
                )

        st.rerun()
