---
title: Building an LLM Inference Engine (MicroServe) — Phase 3: Continuous Batching
description: Part of MicroServe, a from-scratch LLM inference engine built one serving concept at a time. This article covers continuous (iteration-level) batching — why letting several requests share the GPU's idle time, instead of queuing behind each other, is the natural next step after caching, and what it actually buys in practice.

date: August 31, 2026
modified: August 31, 2026

author: Arpon Kapuria
category: Dev Journal
tags: LLM Inference, Production AI
---

## Motivation

Phase 2's KV cache fixed the biggest problem with the naive engine: recomputing the whole sequence every step. What it didn't fix is that the server still only does one thing at a time. If a second request arrives while the first is generating, it just waits — like a single cashier fully finishing one customer's order before even looking at the next person in line, no matter how long that order takes. The GPU sits mostly idle during any one request's token-by-token decode. Continuous batching is the fix: instead of one request's decode step using a sliver of the GPU while others wait in a queue, run several requests' decode steps together, in the same forward pass.

> **This post is Phase 3:** continuous batching. Requests get admitted and evicted from a shared batch every single step, not once per batch — and this post includes two benchmarks, not one, because the obvious one turned out not to be enough on its own.
> 
> **Previously in this series:** [Phase 1 — The Naive Inference Path](https://github.com/arponkapuria/MicroServe/blob/main/notes/01_naive_inference.md), [Phase 2 — The KV Cache](https://github.com/arponkapuria/MicroServe/blob/main/notes/02_kv_cache.md).

## Three Ways to Batch, and Why Continuous Wins

**Static batching**: collect a fixed group of requests and run them together from start to finish. The problem is that all requests have to stay in the batch until the *slowest request finishes*. For example, if three requests finish in 2 seconds but one takes 8 seconds, the GPU still has to keep the batch around for those extra 6 seconds. This can leave GPU resources underutilized and increase latency for requests that could have finished earlier.

![Static Batching|600](/images/blogs/microserve/phase-3-static-batching.jpg)

*Source: [Ashish Bamania](https://www.intoai.pub/p/llm-inference-batching-strategies)*

**Dynamic batching**: instead of waiting for a fixed batch to be created beforehand, the system collects requests for a short period of time and forms a batch when a *timer or request-count limit is reached*. This allows incoming requests to be grouped together more efficiently and reduces the need to process every request individually. However, once the batch starts running, its requests still generally move together — a request that finishes early does not immediately get replaced by a new request.

![Dynamic Batching|600](/images/blogs/microserve/phase-3-dynamic-batching.jpg)
![Dynamic Batching|500](/images/blogs/microserve/phase-3-dynamic-batching-2.jpg)

*Source: [Ashish Bamania](https://www.intoai.pub/p/llm-inference-batching-strategies)*

**Continuous (iteration-level) batching**: this takes batching one step further. Instead of deciding the batch only once at the beginning, the system *re-evaluates the batch after every generation step*. When one request finishes generating its response, it is immediately removed from the batch and another waiting request can take its place in the *very next step*. This keeps the GPU slots filled as much as possible, avoiding the wasted time where fast requests would otherwise sit idle while waiting for a slower request to finish.

![Continuous Batching|600](/images/blogs/microserve/phase-3-continuous-batching.jpg)

*Source: [Ashish Bamania](https://www.intoai.pub/p/llm-inference-batching-strategies)*

This project takes the third option, and for a good reason grounded in Phase 2's own results: the one prompt that didn't benefit much from caching was the shortest one, because a 2-token generation has almost no decode cost to amortize its own cache-init overhead against. Static batching would make that worse — chaining a short request's fate to whatever slow request happened to share its batch. Continuous batching is the only one of the three where a short request's slot frees up the moment it's actually done.

## The Scheduler Loop

```python
def step(self) -> None:
    self._admit()
    prefill_seqs = [s for s in self.running if s.past_key_values is None]
    decode_seqs = [s for s in self.running if s.past_key_values is not None]
    if prefill_seqs:
        self._run_prefill(prefill_seqs)
    if decode_seqs:
        self._run_decode(decode_seqs)
    self._evict_finished()
```

Every call to `step()` does four things, every time: pull new requests off the queue if there's room, run one batched forward pass for anyone who just joined (prefill), run one batched forward pass for everyone already decoding, and evict anyone who's finished. That admit/evict decision happening *every iteration*, not once per batch, is the entire mechanism this phase exists to build.

## The One Subtle Bug That Actually Matters Here

Batching sequences of different lengths together means padding the shorter ones — and padding creates a real correctness trap. A model needs to know each token's *true* position in its sequence to compute attention correctly (this is what RoPE, Qwen3's positional encoding, depends on). If a padded row's real tokens don't start at position 0, but the model isn't told that explicitly, it can silently compute the wrong positions — not crash, just quietly produce garbled text.

```python
attention_mask = encoded["attention_mask"]
position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 0)
```

This one calculation — the real position of every token, derived from the attention mask rather than assumed from tensor shape — is what keeps every row in a padded batch correct regardless of how much padding it carries. It's easy to skip by accident (the model will still *run* without it, just wrong), which is exactly why it's worth calling out explicitly here rather than leaving it implicit.

## Two Benchmarks, Not One

The obvious benchmark — same 4 prompts as every prior phase, arriving staggered a half-second apart — is the right tool for testing one thing: does a request get admitted into an already-running batch quickly, instead of waiting for the batch to fully drain? It confirmed that cleanly. But with only 4 requests and room for 4 in the batch, nothing ever actually has to *wait* for a slot — which means this benchmark structurally can't test whether batching helps throughput, since throughput gains only show up under contention.

So there's a second benchmark: 8 requests (the same 4 prompts, twice each) submitted all at once, with room for only 4 at a time — guaranteeing real queueing — compared directly against the same 8 requests run one at a time through Phase 2's engine.

## Results

### Main benchmark (staggered arrival) — vs Phase 2

![Inter-token latency per request, continuous batching|600](/images/blogs/microserve/batching_itl.png)

| Metric | Phase 2 (sequential) | Phase 3 (batched) |
|---|---|---|
| Wall clock | 15.17 s | 14.89 s |
| Mean TPOT | ~97.6 ms | ~145.7 ms |
| System TPS | 10.02 | 10.21 |
| Mean queue time | 0.0 ms | 0.04 ms |

![KV Cache vs Continuous Batching comparison|600](/images/blogs/microserve/kv_cache_vs_batching.png)

Every request's per-token latency got noticeably worse under batching — expected, since a batched step does more work than a single-sequence step. But aggregate throughput barely moved, because nothing in this benchmark ever actually contends for a slot; every request is admitted the instant it arrives. This isn't a disappointing result — it's the correct result for a benchmark that was never designed to create contention in the first place.

### Stress test (burst arrival, real contention)

| | Sequential | Continuous batching |
|---|---|---|
| Wall clock | 30.42 s | 26.91 s |
| System TPS | 9.995 | 11.296 |
| Mean queue time | 0.0 ms | 1,835.71 ms |

![Stress test — 8 requests, burst arrival|600](/images/blogs/microserve/batching_stress_comparison.png)

**1.13x speedup**, with real queueing this time (individual requests waited up to several seconds for a slot) and generated text matching the sequential baseline exactly, word for word, on all 8 requests. This is the number that actually answers "does continuous batching help" — and it does, genuinely, once there's real demand for it to help with.

The size of that win is worth being honest about too: a simple "batching shares one weight-memory read across several sequences" argument would suggest something much larger than 1.13x. Some of that gap is measurable directly — the cost of padding and reassembling a shared cache every iteration roughly doubled under the stress test's sustained load (12.66ms/iteration vs 5.27ms/iteration in the lighter benchmark) — but that alone doesn't close the whole gap. There's very likely additional cost in how a padded, masked batched forward pass runs on this hardware that this phase hasn't fully isolated. A real, open question, not a hidden one.

## What's Next

Two concrete gaps came out of this phase, each pointing at a specific later topic. Admission here is plain FIFO — a short request arriving after a long one still waits behind it, even though it could finish almost instantly once given a slot. Phase 4 (scheduling) is where that gets a real fix. And the padding/reassembly overhead measured above is a direct, quantified argument for Phase 5: paged KV cache removes the need to pad and rebuild a shared cache tensor every single iteration, and this phase's numbers are the concrete "before" picture Phase 5 gets to improve on.

---
> Code: [https://github.com/arponkapuria/MicroServe](https://github.com/arponkapuria/MicroServe)
>
> Full metrics and analysis: `reports/03_batching_report.md`