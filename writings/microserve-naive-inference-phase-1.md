---
title: Building an LLM Inference Engine (MicroServe) — Phase 1: The Naive Inference Path 
description: Part of MicroServe, a from-scratch LLM inference engine built one serving concept at a time. This article covers the naive, no-cache decode loop — why recomputing the whole sequence every step doesn't scale, and what it costs in practice.

date: August 26, 2026
modified: August 26, 2026

author: Arpon Kapuria
category: Dev Journal
tags: LLM Inference, Production AI
---

## Motivation

Serving engines like vLLM, SGLang, and TensorRT-LLM are full of optimizations that look arbitrary from the outside — paged KV caches, continuous batching, speculative decoding, kernel fusion, CUDA graphs. Reading their source code without having felt the problems these solve firsthand means memorizing a list of tricks instead of understanding a system.

So I'm building **MicroServe**: a minimal LLM inference engine, from scratch, one serving topic at a time. For each topic: survey the real techniques, implement one by hand, measure it, and only move to the next topic once the numbers make the next optimization's necessity obvious. This isn't an attempt to rebuild vLLM — it's an attempt to earn the right to read vLLM's source and immediately recognize why every ugly line is there.

> **This post is Phase 1:** the naive inference path. No cache, no batching, no cleverness. Just a baseline, and the receipts showing exactly why it has to change.

## Configuration

- **Hardware**: MacBook Air M1 (2020), 8GB unified memory
- **Framework**: PyTorch, device-agnostic (`cuda` / `mps` / `cpu`, auto-detected — the same code has to run on a CUDA box later without modification)
- **Model**: `Qwen/Qwen3-1.7B`, fp16 (~3.2 GB footprint)
- **Decoding**: greedy (deterministic — makes runs comparable across phases without sampling noise as a confound), chat-template formatted, thinking mode off, `max_new_tokens=128`

## How a Transformer Predicts the Next Token

A language model generates text one token at a time. To predict the next token, every layer of the model needs to look back at every token that came before it — that's what "attention" means. Concretely, each token gets turned into three vectors: a **Query** (what this token is looking for), a **Key** (what this token represents, so other tokens can decide whether to attend to it), and a **Value** (the actual information this token carries). The model compares the current position's Query against every earlier position's Key to decide how much weight to give each position's Value. Stack that across every layer, and the final layer's output gets turned into a probability over the entire vocabulary — the model's guess at what comes next.

The detail that matters for everything in this project: **a token's Key and Value never change once computed.** Token 5's Key and Value depend only on token 5 and what came before it — not on what gets generated afterward. So computing them twice is pure waste. That's the whole idea behind a KV cache: compute each token's Key/Value once, keep them around, reuse them for every future step.

This phase does the opposite on purpose — it throws them away every time — so the cost of *not* caching is something you can actually see in the numbers, not just take on faith.

## Generating Text Without a Cache

The simplest possible generation loop looks like this:

1. Run the prompt through the model.
2. Look at the model's prediction for the very last position, and pick the most likely next token.
3. Append that token to the sequence.
4. **Run the model again — from scratch — over the whole sequence: prompt + everything generated so far.**
5. Repeat until the model produces an end-of-text token or a length limit is hit.

Step 4 is the naive part. Here's the actual loop (`src/microserve/engine/naive.py`):

```python
for _ in range(config.max_new_tokens):
    step_start = time.perf_counter()

    outputs = model(input_ids=generated_ids, use_cache=False)   # recompute everything, every step
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    step_end = time.perf_counter()
    per_token_latencies_ms.append((step_end - step_start) * 1000)

    if next_token.item() == tokenizer.eos_token_id:
        break
```

`use_cache=False` is the one line doing all the damage. It forces the model to recompute attention over `generated_ids` — which grows by one token every iteration — from zero, every single time, discarding all the Key/Value pairs it just computed.

Here's why that's expensive in a way that compounds: the cost of one forward pass grows roughly with how many tokens it has to process. If generating token `k` means processing `prompt_len + k` tokens from scratch, then the total work to generate `n` tokens is roughly `1 + 2 + 3 + ... + n`, which grows proportionally to `n²`. A cached loop, by contrast, only ever processes *one new token* per step — its total work grows proportionally to `n`, not `n²`. That gap between "grows like n" and "grows like n²" is the entire reason Phase 2 exists.

To make the Key/Value structure concrete rather than abstract, `naive.py` also has a small tracer that runs one forward pass *with* caching enabled just to print what the model is actually holding onto:

```python
outputs = model(**inputs, output_hidden_states=True, use_cache=True)
k0 = outputs.past_key_values.key_cache[0]
print(f"KV cache, layer 0 K shape: {tuple(k0.shape)}  (batch, num_kv_heads, seq_len, head_dim)")
```

On the 17-token prompt used to trace it, that shape comes out to `(1, 8, 17, 128)` — 1.86 MB across all 28 layers. The `8` there (num_kv_heads) is smaller than the model's actual number of attention heads — Qwen3-1.7B uses grouped-query attention, where multiple query heads share the same Key/Value heads, a detail that'll matter for sizing Phase 2's cache and Phase 5's paged allocator. Every layer produces its own Key/Value pair in this shape. That's literally the data structure Phase 2 takes over managing by hand, instead of letting Hugging Face hold it internally.

## Running the Same Code on Apple Silicon and CUDA

Since this whole project has to run on both an M1 laptop and (eventually) CUDA hardware, device selection lives in exactly one place (`src/microserve/config.py`):

```python
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-1.7B"
    device: str = field(default_factory=get_device)
    max_new_tokens: int = 128
```

Everything downstream — model loading, tensor placement — asks `get_device()` instead of checking `torch.cuda.is_available()` itself, so the same code runs unmodified on either machine. Model loading follows the same one-function-does-it-all rule (`src/microserve/models/loader.py`):

```python
def load_model_and_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=config.dtype)
    model.to(config.device)
    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer
```

Every later phase (KV cache, batching, quantization...) calls this same function, so every phase's model starts from an identical place, which is what makes their benchmark numbers comparable to each other.

## Measuring What Actually Matters

Raw "it feels slow" isn't a number. This project tracks the same handful of metrics for every phase (`src/microserve/engine/metrics.py`), so later phases can be compared directly against this baseline:

- **TTFT (time to first token)** — how long you wait before anything appears at all.
- **TPOT (time per output token)** — once it starts responding, how long between each word after that.
- **Inter-token latency (ITL)** — the *individual* per-token gaps, not just their average, so you can tell a steady pace apart from a bursty one.
- **TPS (tokens per second)** — overall generation speed.
- **Memory** — how much memory the process is using while generating, both current allocation and peak.

```python
def build_request_metrics(prompt, generated_text, prompt_tokens,
                           per_token_latencies_ms, device) -> RequestMetrics:
    ttft_ms = per_token_latencies_ms[0]
    inter_token_latencies_ms = per_token_latencies_ms[1:]
    token_generation_time_ms = sum(inter_token_latencies_ms)
    total_latency_ms = ttft_ms + token_generation_time_ms
    tpot_ms = statistics.mean(inter_token_latencies_ms) if inter_token_latencies_ms else 0.0
    tps = generated_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0.0
    ...
```

The first recorded latency is TTFT; everything after it is inter-token latency, and TPOT is just their average. For memory calculation, `torch.accelerate` is used to track peak allocated memory on both CUDA and Apple Silicon and it shows a signal that current-allocated memory alone never could: peak memory pulls further and further ahead of steady allocation as generations get longer, which lines up exactly with where this baseline is expected to hurt.

```python
# Memory helpers — device-agnostic (using torch.accelerator.memory). This gives real peak-since-reset tracking on MPS

def reset_peak_memory(device: str) -> None:
    """Reset the peak-memory tracking baseline. No-op for cpu."""
    if device in ("cuda", "mps"):
        torch.accelerator.memory.reset_peak_memory_stats()


def current_memory_mb(device: str) -> float:
    """Current allocated memory in MB. Returns 0.0 on cpu or any device
    with no accelerator memory API."""
    if device in ("cuda", "mps"):
        return torch.accelerator.memory.memory_allocated() / (1024 ** 2)
    return 0.0


def peak_memory_mb(device: str) -> float:
    """True peak allocated memory in MB since the last reset_peak_memory()
    call. Accurate on both CUDA and MPS."""
    if device in ("cuda", "mps"):
        return torch.accelerator.memory.max_memory_allocated() / (1024 ** 2)
    return 0.0
```

## Serving One Request at a Time

Wrapping the loop in a server is almost mechanical — load the model once, expose one endpoint (`src/microserve/server/app_naive.py`):

```python
@app.on_event("startup")
def startup():
    global _model, _tokenizer
    _model, _tokenizer = load_model_and_tokenizer(CONFIG)

@app.post("/generate")
def generate(req: GenerateRequest):
    cfg = replace(CONFIG, max_new_tokens=req.max_new_tokens) if req.max_new_tokens else CONFIG
    result = naive_generate(_model, _tokenizer, req.prompt, cfg)
    return result.to_dict()
```

The important thing is what's *not* here: no queue, no batching, no concurrency handling of any kind. If a second request arrives while the first is still generating, it just waits — like a single cashier who fully finishes one customer's order before even looking at the next person in line, no matter how long that order takes. That's not an oversight; it's the control condition for the batching phase, which exists specifically because the accelerator sits mostly idle waiting on one request's token-by-token generation when it could be doing useful work for other requests in that same stretch of time.

## The Benchmark

A small script (`benchmarks/run_naive.py`) ties all of the above together: run the tracer once, then run the decode loop against four fixed prompts, collecting metrics for each and an aggregate across all four.

```python
wall_start = time.perf_counter()
for prompt in PROMPTS:
    result = naive_generate(model, tokenizer, prompt, bench_config)
    results.append(result)
wall_end = time.perf_counter()

aggregate = compute_aggregate(results, wall_clock_seconds=wall_end - wall_start)
```

Results get written to `benchmarks/results/naive.json` — the raw data behind everything below.

**Warmup**: right after loading the model, the script runs one throwaway generation and discards it, to pay the first-call kernel warm-up cost (MPS/CUDA compiling kernels for the first time) before anything gets timed. Without it, request 1 unfairly absorbs that one-time cost.

```python
warmup_config = replace(CONFIG, max_new_tokens=8)
_ = naive_generate(model, tokenizer, "Hello", warmup_config)
```

## Results

4 prompts, `Qwen/Qwen3-1.7B`, fp16, mps, greedy, chat-template formatted, `max_new_tokens=128`.

| Prompt | Gen. tokens | TTFT (ms) | Total latency (ms) | TPOT (ms) | TPS |
|---|---|---|---|---|---|
| "The capital of Bangladesh is" | 12 | 22.07 | 611.50 | 53.58 | 19.62 |
| "What is 2 + 2?" | 2 | 14.60 | 49.01 | 34.41 | 40.81 |
| "Explain a KV cache" | 101 | 43.52 | 10,367.77 | 103.24 | 9.74 |
| "Ocean poem, four lines" | 37 | 105.06 | 1,744.20 | 45.53 | 21.21 |

**Aggregate (4 sequential requests)**: mean latency 3,193.12ms, median 1,177.85ms, mean TTFT 46.31ms, mean TPS 22.85, RPS 0.06. p99 is still just the max at n=4 — not a real tail estimate.

### Plotting the Inter-Token Latencies

![Inter-token latency per request, naive no-cache decode|600](../images/blogs/microserve/naive_itl.png)

Each panel is one request's per-token latency, log-scaled. The two short requests stay close to a flat line; the two longer ones show sharp, isolated spikes rather than a smoothly worsening curve.

### Warmup helps, but the request right after the heaviest one still pays a tax

TTFT is low for the first three requests (22.07ms, 14.60ms, 43.52ms), then jumps to 105.06ms on request 4 — despite request 4 having the exact same prompt length (23 tokens) as request 3. Request 4 comes immediately after the heaviest generation in the batch (101 tokens, over 10 seconds), which is consistent with residual system load bleeding into the next request's first forward pass, not a cold start (warmup already handled that) or prompt length (ruled out by the exact length match).

### The long request breaks in two sharp bursts, not a gradual decline

The 101-token generation stays steady for its first ~36 tokens (12–91ms). Then four consecutive tokens spike hard — 1,488ms, **2,794ms** (the worst stall of the run), 717ms, 240ms — before dropping straight back to the same 20–100ms band for another 40 tokens. A second, smaller isolated spike hits at token 80 (1,382ms), then it's steady to the end. Not a worsening tail — two sharp, separated bursts.

### Memory tracking shows something real — and it's exactly what the theory predicted

`memory_peak_mb` is a genuinely different, higher number than `memory_allocated_mb` for every request, and the *size* of that gap tracks generation length almost perfectly:

| Prompt | Generated tokens | Peak − allocated |
|---|---|---|
| "What is 2 + 2?" | 2 | 7.38 MB |
| "The capital of Bangladesh is" | 12 | 7.96 MB |
| "Ocean poem" | 37 | 17.04 MB |
| "Explain a KV cache" | 101 | 35.83 MB |

The longest generation shows roughly 4.5x the peak-memory overshoot of the shortest. That's hard evidence — not just an inference from the stall pattern — that the naive loop's *transient* memory footprint grows with how long a generation runs, exactly as the "no reuse, so both compute and memory grow every step" theory predicted from the start.

### Wall clock is still well beyond the summed request latencies

67.05s wall clock vs. 12.77s summed request latencies — a 54.28s gap across 3 inter-request intervals (~18.09s each). A large, real gap sitting outside the timed window. Flagged as an open question — worth instrumenting what happens between requests specifically, rather than guessing further.

## What's Next

The naive loop's flaw — recomputing everything, every step, with zero Key/Value reuse — creates real instability on memory-constrained hardware, visible on two independent fronts: the ITL stalls, and real peak-memory tracking that grows in lockstep with generation length.

Phase 2 builds a hand-rolled KV cache and re-runs these same four prompts. The prediction, based on everything above: per-step compute and memory should stop growing with sequence length, which should flatten both the ITL stalls and the peak-memory overshoot down to something small and roughly constant regardless of how long the generation runs.

---

> Code: [https://github.com/arponkapuria/MicroServe](https://github.com/arponkapuria/MicroServe)
>
> Full metrics and analysis: `reports/01_naive_inference_report.md`