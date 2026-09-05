---
title: From Naive Decode to KV Cache: Why LLM Serving Gets Faster With One Idea
description: Part 1 of NanoServe, a tiny LLM inference engine build series. This article covers the naive decode loop to hand-threaded KV cache — why reusing Key/Value tensors instead of recomputing them turns a quadratic decode loop into a linear one, and what that's worth in practice.

date: September 03, 2026
modified: September 03, 2026

author: Arpon Kapuria
category: Dev Journal
tags: LLM Inference, AI Systems
---

I've been building a tiny LLM inference engine from scratch (**NanoServe**) — no vLLM, no SGLang, just PyTorch and a Qwen3-1.7B model running on a MacBook Air's GPU (MPS, not CUDA, so no shortcuts either). The goal is to actually understand what's happening under the hood every time we hit "generate" on a chatbot and later converting it into a production-grade tiny llm inference engine.

Step one was the dumbest possible version: generate text with zero optimizations. Step two was adding a KV cache — probably the single most important idea in LLM serving. This is the story of both steps: what I built and what the numbers actually looked like.

## Prefill vs Decode

Before touching any code, there's one distinction you need in your head, because everything else in this article — the slowness of naive decode, why KV cache fixes it, why TTFT and TPOT behave so differently — traces back to it. Every generation request an LLM handles is actually *two* different jobs wearing one trenchcoat: **prefill** and **decode**.

**Prefill** happens once, at the very start. You hand the model your whole prompt — every token of it — and it processes all of them *in one shot*. Not one at a time: all at once, in parallel, as a single matrix operation. The model isn't generating anything yet during prefill. It's just "reading" — building up an understanding of your prompt so it has something to generate from. Think of it like handing someone a full page to read before they answer you. They read the whole page in one go.

**Decode** is everything after that. This is the actual generation: producing token #1, then #2, then #3, one at a time, in a strict sequence. You cannot produce token #5 before token #4 exists, because token #5 depends on it. Unlike prefill, decode is inherently sequential — there's no way to parallelize "guess the next word" across multiple steps, because each step needs the previous step's answer first.

![LLM Inference - Prefill vs Decode|700](/images/blogs/nanoserve/inference-prefill-vs-decode.png)

Here's the part that actually matters for performance, and it's a little counterintuitive: **prefill is fast per-token, decode is slow per-token — even though prefill is doing "more work."**

Why? It comes down to how a GPU actually spends its time. During prefill, the model runs one big matrix-multiply across all prompt tokens simultaneously. GPUs are extremely good at exactly this — dense, parallel matrix math — so even though it's processing, say, 200 tokens, it can use nearly all its compute capacity at once. It's *compute-bound*, and compute-bound work is what GPUs are built for.

Decode is the opposite. At each step, you're feeding the model exactly one token, so instead of one big efficient matrix-matrix multiply, you get a much smaller matrix-vector multiply. The GPU spends most of its time not crunching numbers but *moving data* — pulling the Key/Value cache and model weights through memory — for comparatively little actual compute. It's *memory-bound*. This is why decode, token for token, is fundamentally more expensive than prefill: it can't use the hardware efficiently no matter how you write the code, because the job itself doesn't parallelize the way prefill does.

This maps directly onto the two metrics from the previous section, and it's worth being explicit about it: **TTFT is (almost entirely) prefill time. TPOT is (almost entirely) one decode step's time.** When you see TTFT and TPOT numbers, you're really looking at "how long did the parallel, compute-bound phase take" vs. "how long does the sequential, memory-bound phase take, per token." Keep that mapping in mind — it explains basically every result in this article.

## Naive Decode

Here's the thing about text generation: a language model doesn't spit out a whole sentence at once. It predicts one token at a time. You give it a prompt, it predicts the next token, you append that token to the prompt, and you ask it again. And again. Until it decides to stop.

So if you want to generate 128 tokens, you're calling the model 128 times. The question is: what do you feed it each time?

The naive answer — and I mean *naive* in the technical sense, not an insult to past-me — is: feed it everything. Every single time.

```python
generated = input_ids

for _ in range(max_new_tokens):
    logits = self.model(generated).logits          # full forward pass, every token
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
```

Look closely at that loop. On step 1, the model processes, say, 40 tokens — this step is effectively a prefill. On step 2, it processes 41 tokens — the same 40, plus the new one, *all over again*. On step 3, 42 tokens. By the time you're generating your 128th token, the model is re-processing a sequence of over 150 tokens, purely to produce one more.

![Without KV Cache - Prefill vs Decode|400](/images/blogs/nanoserve/naive-decode.png)

This is the real problem with naive decode, stated precisely: it never actually settles into a proper decode phase. Every "decode" step secretly redoes a full prefill-sized pass over the *entire* sequence so far, instead of the cheap, incremental, single-token step decode is supposed to be. It doesn't remember anything. Every step starts from zero and recomputes the entire sequence's attention, layer by layer, from the very first token.

If you know a bit of algorithmic complexity, alarm bells should be going off. Generating N tokens this way costs roughly 1 + 2 + 3 + ... + N work-units, which is O(N²). Double your output length, and the cost roughly quadruples, not doubles.

## TTFT and TPOT, Measured

I ran this on a medium-length prompt (a "explain binary search trees" question, ~128 tokens generated), averaged over a few runs after a warmup pass to shake out any cold-start noise. Here's what came back:

| Metric | Value |
|---|---|
| TTFT (prefill time) | 218.9 ms |
| TPOT (one decode step, averaged) | 385.5 ms |
| Throughput | 2.60 tokens/sec |
| Peak memory | 3447.5 MB |

At 385ms per token, that's roughly 2.6 tokens a second — slower than a slow typist. For a 128-token answer, you're waiting nearly 50 seconds total. But look at the relationship between TTFT and TPOT here: they're almost the same order of magnitude (218ms vs 385ms). Given what you now know about prefill vs decode, that should look wrong immediately — TPOT (one memory-bound decode step) shouldn't cost *more* than TTFT (a whole compute-bound prefill pass over the entire prompt). A single token step outweighing the full-prompt read is naive decode's O(N²) problem showing up directly in the numbers: every "decode" step here is secretly paying a prefill-sized cost.

## KV Cache

Here's the insight that unlocks everything: when a transformer processes token by token, most of what it computes at each layer doesn't actually change once a token has been seen. Specifically, for every previous token, the model computes a **Key** and a **Value** vector at each layer (that's the K and V in "KV cache" — they come from the attention mechanism's Query/Key/Value setup). Let's say once token #7 has been processed, its Key and Value vectors are fixed forever. They don't depend on what gets generated *after* token #7.

*So why were we recomputing them every single step?*

We weren't wasting compute for a subtle reason — we were doing it because the naive loop above just... didn't save anything. It threw away every intermediate result after each forward pass and started fresh. The fix is almost embarrassingly simple in concept: **save the Keys and Values as you go, and only compute new ones for the newest token.**

![KV Cache - Prefill and Decode|400](/images/blogs/nanoserve/kv-cache-prefill-and-decode.png)

Instead of feeding the model the *entire* growing sequence every step, you do the expensive full pass exactly once (the prompt), then feed it just the *one new token* per step, along with the cache of everything computed so far. The model uses the cache to "remember" the rest without redoing the work.

```python
outputs = self.model(input_ids, use_cache=True)
past_key_values = outputs.past_key_values         # the whole cache, built once
next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

for _ in range(max_new_tokens - 1):
    outputs = self.model(next_token, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values      # cache just grew by one token
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
```

Notice the size of what's being fed to the model in the loop now: `next_token` is a single token, not the whole sequence. That's the whole trick. This code is also, for the first time, actually honest about the prefill/decode split from earlier: the first call (`self.model(input_ids, ...)`) *is* the prefill — one shot, full prompt. The loop after it *is* decode — genuinely one token at a time, genuinely incremental, no longer secretly redoing prefill work on every step. Every step is now roughly constant-cost instead of growing linearly, which takes the total cost from O(N²) down to O(N).

## Naive vs KV Cache

Here's the same benchmark, same prompt, same hardware. All four core metrics, Naive decode vs. KV cache, side by side:

| Metric | Naive | KV Cache | Change |
|---|---|---|---|
| TTFT | 218.9 ms | 217.5 ms | ~unchanged |
| TPOT | 385.5 ms | 97.1 ms | **~4x faster** |
| Throughput | 2.60 tok/s | 10.14 tok/s | **~4x higher** |
| Peak memory | 3447.5 MB | 3309.8 MB | slightly *lower* |

## Findings

A few things worth actually understanding here, not just glancing at:

![Naive Decode vs KV Cache — TTFT, TPOT, Throughput, Peak Memory|600](/images/blogs/nanoserve/naive-vs-kv-cache-plot.png)

**TTFT barely moved, and that's exactly correct.** TTFT is dominated by the prefill — processing the full prompt for the first time — and that step is *identical* in both versions. The KV cache doesn't help you get the first token faster; it helps you get every token *after* that faster. If your use case is mostly about that initial response latency (like a quick classification task), KV cache alone won't move the needle much. If it's about generating long responses, this is everything.

**Throughput is higher for the same reason TPOT is lower — they're two views of the same fact.** Throughput just measures tokens produced per second across the whole request, and once each token costs ~4x less time to produce, the model can obviously churn out ~4x more of them in the same window. Nothing new is happening here beyond the TPOT improvement — it's the same win, restated as "how much did I get done" instead of "how long did each step take."

**4x, not 100x — and that's an honest number, not a disappointing one.** In theory, avoiding O(N²) recomputation should save you a lot more as sequences get longer. But this is running on an Apple M1 GPU (MPS) at batch size 1, generating one token at a time. At that scale, a big chunk of the per-step cost isn't the actual matrix math — it's the fixed overhead of *launching* each GPU operation in the first place. That overhead doesn't go away just because you're doing less compute per step. So the realistic speedup you see is smaller than the theoretical one, and that gap is itself a useful thing to have measured rather than assumed.

**Memory went down, which seems backwards until you think about what's actually being stored.** You'd expect a cache to cost *more* memory, not less — you're storing something that didn't exist before. But naive decode was repeatedly allocating full-sequence attention buffers at every step, and by the end of a 128-token generation those buffers cover a sequence over 200 tokens long, recreated from scratch each time. The KV cache, by contrast, only needs to compute activations for exactly one new token per step, plus a steadily growing (but much smaller, more efficient) persistent cache. The transient waste from naive decode's repeated full recomputation actually outweighs the cache's storage cost.



## What's Next

The core idea here — remember what you've already computed instead of redoing it — is the foundation almost every other LLM serving optimization builds on top of. Continuous batching, paged attention, prefix caching: all of them are, at some level, smarter ways of managing this same cache across multiple requests at once instead of just one.

Right now this KV cache is a single unbroken tensor that just grows — fine for one request at a time, but it doesn't share memory across requests, and it doesn't handle the "many sequences of different, changing lengths" problem that a real server faces constantly and that's the next problem: *how do you manage this cache efficiently when you're not just serving one person, but many, with memory that has to be allocated, reused, and freed on the fly* ? That's a block allocator away, and it's next.

---

> Code: [https://github.com/arponkapuria/NanoServe](https://github.com/arponkapuria/NanoServe)