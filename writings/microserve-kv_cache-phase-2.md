---
title: Building an LLM Inference Engine (MicroServe) — Phase 2: The KV Cache
description: Part of MicroServe, a from-scratch LLM inference engine built one serving concept at a time. This article covers the hand-threaded KV cache — why reusing Key/Value tensors instead of recomputing them turns a quadratic decode loop into a linear one, and what that's worth in practice.

date: August 29, 2026
modified: August 29, 2026

author: Arpon Kapuria
category: Dev Journal
tags: LLM Inference, Production AI
---

## Motivation

Phase 1 built the naive decode loop on purpose: recompute the entire sequence from scratch on every single step, throw the Key/Value tensors away immediately after using them once. It was slow by design, so the cost of *not* caching would show up as real numbers instead of received wisdom. It showed up — a multi-second response for a 101-token answer, with sharp latency spikes appearing mid-generation, and a transient memory footprint that grew right alongside those spikes.

This phase does the obvious next thing: stop throwing away Key/Value tensors that haven't changed, and reuse them instead. No custom memory layout, no block allocation — that's Phase 5's job. This phase is isolated to one question, and only one: what happens when recomputation goes away?

> **This post is Phase 2:** the KV cache. Same four prompts as Phase 1, same everything else — only the engine changes (adds KV Cache). The receipts showing exactly what that one change buys.
>
> Phase 01: [Building an LLM Inference Engine (MicroServe) — Phase 1: The Naive Inference Path](/blogs/posts/microserve-naive-inference-phase-1/)


## Why a Token's Key/Value Never Changes

A token's Key and Value depend only on that token and everything before it — not on anything generated afterward. So token 5's Key/Value at decode step 20 is identical to what it was at decode step 6. Computing it again is pure waste. The KV cache is the direct consequence of that fact: compute each token's K/V once, keep them, reuse them.

## Prefill and Decode as Two Different Steps

Unlike the naive loop — where every step is shaped like a full prefill — the cached loop treats the first forward pass differently from every step after it:

1. **Prefill**: run the entire prompt through the model once, with `use_cache=True`. This produces the initial `past_key_values` — one Key and one Value tensor per layer, covering every prompt token.
2. **Decode**: for each new token, run the model on *only* that one token, passing in the cache from the previous step. The model reuses the cached K/V for every earlier position and only computes K/V for the new token.

```python
outputs = model(input_ids=input_ids, use_cache=True)
past_key_values = outputs.past_key_values          # prefill: full prompt in, full cache out

for _ in range(config.max_new_tokens):
    next_token = pick_next_token(outputs.logits[:, -1, :], config)
    outputs = model(
        input_ids=next_token,                        # decode: exactly one new token in
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values         # cache grows by one token's K/V
```

This mirrors how real serving engines think about the two phases: prefill is compute-bound (it touches every prompt token once, in parallel), decode is bandwidth-bound (it touches exactly one new token per step, but has to read the whole growing cache to do it). Making that split explicit in the code — not just implicit in performance — is the point of this phase.

## Why This Turns O(n²) Into O(n)

The naive loop's cost for generating `n` tokens grows like `1 + 2 + 3 + ... + n ≈ n²`, because every step recomputes the full growing prefix. The cached loop's decode steps each do a fixed amount of new work — one token's worth — regardless of how long the sequence has gotten. Total decode cost grows like `n`, not `n²`. The gap between those two curves is what showed up as Phase 1's latency spikes on the long generation, and, as shown below, as real, measurable transient memory pressure too.

## Running the Same Four Prompts Again

Same benchmark script structure as Phase 1  — same 4 prompts, same warmup, same `max_new_tokens=128`, only the generate function differs:

```python
warmup_config = replace(CONFIG, max_new_tokens=8)
_ = cached_generate(model, tokenizer, "Hello", warmup_config)
```

## Results

4 prompts, `Qwen/Qwen3-1.7B`, fp16, mps, greedy, chat-template formatted, `max_new_tokens=128`.

| Prompt | Gen. tokens | TTFT (ms) | Total latency (ms) | TPOT (ms) | TPS |
|---|---|---|---|---|---|
| "The capital of Bangladesh is" | 12 | 33.33 | 189.36 | 14.18 | 63.37 |
| "What is 2 + 2?" | 2 | 34.41 | 50.27 | 15.86 | 39.78 |
| "Explain a KV cache" | 101 | 32.78 | 1,827.27 | 17.94 | 55.27 |
| "Ocean poem, four lines" | 37 | 14.41 | 649.19 | 17.63 | 56.99 |

**Aggregate**: mean latency 679.02ms (Phase 1: 3,193.12ms), mean TTFT 28.73ms (Phase 1: 46.31ms), mean TPS 53.86 (Phase 1: 22.85), wall clock 14.32s for all 4 requests (Phase 1: 67.05s).

### Phase 1 vs Phase 2, side by side

![Naive vs KV Cache comparison](../benchmarks/results/naive_vs_kv_cache.png)

### The 101-token request is the headline number

Phase 1's worst case — the "explain a KV cache" prompt, 101 generated tokens — took 10.37 seconds, with two sharp multi-second bursts appearing mid-generation. With the cache: **1.83 seconds**, an 82.4% reduction, with TPOT dropping from 103.24ms to 17.94ms — a number now in line with the other three requests instead of a multi-x outlier.

### The stalls are gone, not just smaller

![Inter-token latency per request, KV cache decode](../benchmarks/results/kv_cache_itl.png)

Phase 1's naive P3 destabilized in two isolated bursts — 1,488ms and 2,794ms back-to-back around token 37, then a smaller 1,382ms spike near token 80 — against an otherwise steady 12–91ms band. The cached version shows neither: a bounded band, roughly 9–36ms across all 100 inter-token gaps, no spikes anywhere. Good evidence the diagnosis was right, not just plausible — removing only the recomputation removed the entire failure mode.

### The cache isn't free — but the tradeoff splits cleanly by generation length

Time to first token got worse under caching for the two shortest requests (22.07→33.33ms, 14.60→34.41ms) — real, structural cache-init overhead. But it got *better* for the two longer ones (43.52→32.78ms, and 105.06→14.41ms). That second case is worth unpacking: P4 (the ocean poem) immediately follows P3, the heaviest request, and Phase 1's report attributes P4's elevated naive TTFT to residual system load bleeding in from the request before it. Under caching, P3 itself finishes in 1.83s instead of 10.37s — so P4 inherits far less residual load, and ends up with the *lowest* TTFT of all four requests. Two effects are at work here: cache-init cost raises TTFT on its own, while finishing the preceding request faster lowers it for whatever comes next.

### Peak memory shows what the cache is actually saving

With MPS peak-memory tracking now real (not a flat proxy), the difference between engines is stark:

| Prompt | Naive peak − allocated | KV cache peak − allocated |
|---|---|---|
| **P2:** "What is 2 + 2?" (2 tok) | 7.38 MB | 7.25 MB |
| **P1:** "Capital of Bangladesh" (12 tok) | 7.96 MB | 2.52 MB |
| **P4:** "Ocean poem" (37 tok) | 17.04 MB | 1.57 MB |
| **P3:** "Explain a KV cache" (101 tok) | 35.83 MB | **0.36 MB** |

On the long request — the one that struggled most under the naive engine — the cache doesn't just fix latency, it nearly eliminates the transient memory overshoot: from 35.83 MB down to 0.36 MB. That's independent confirmation, from a completely different signal than latency, that the naive loop's problem really was recomputation-driven memory pressure.

### Where caching doesn't pay off

The shortest generation (2 tokens) is the one prompt where caching doesn't help — 49.01ms → 50.27ms, a small 2.6% increase. The cache buys back decode cost, and a 2-token generation has almost no decode cost to buy back, but still pays the full cache-initialization cost during prefill. A real, worth-knowing tradeoff: the payoff is proportional to generation length.

## What's Next

The cache eliminates the recomputation problem cleanly on two independent fronts — latency and transient memory. The server is still fully sequential, one request at a time, so short requests still can't share cache-init cost with anything else happening concurrently. Phase 3 replaces that one-at-a-time model with continuous batching, which is where that remaining gap gets a real answer: overlapping requests share the accelerator's idle time instead of each one paying its own fixed costs in isolation.

---
> Code: [https://github.com/arponkapuria/MicroServe](https://github.com/arponkapuria/MicroServe)
>
> Full metrics and analysis: `reports/02_kv_cache_report.md`