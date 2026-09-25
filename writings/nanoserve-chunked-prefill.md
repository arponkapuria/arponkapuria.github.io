---
title: "Chunked Prefill: Stopping One Long Prompt From Freezing Everyone Else"
description: Part 6 of NanoServe, a tiny LLM inference engine build series. This article covers chunked prefill — splitting a long prompt's prefill into pieces small enough to interleave with other requests' decode steps — and what it actually bought, and cost, on Apple Silicon.

date: September 25, 2026
modified: September 25, 2026

author: Arpon Kapuria
category: Dev Journal
tags: LLM Inference, AI Systems
---

> Part 6 of the NanoServe build series. Part 5 covered prefix caching — a radix tree that lets requests sharing a prompt reuse each other's already-computed KV; this part assumes you've read that, or at least know what `admit_one` does when a new request arrives.
>
> **Part 5:** [Prefix Caching: Reusing the Work You've Already Paid For With RadixAttention](/blogs/posts/nanoserve-prefix-caching)

Every request handled so far, no matter how it arrived, gets prefilled the same way: one forward pass over the entire prompt, start to finish, before it ever produces a token. That's been fine because the earlier tests never put a long prompt next to a busy batch. This part does exactly that, and shows what happens when one request's prefill is 400+ tokens long and three other people are mid-conversation when it shows up.

## The Problem: One Big Prefill Blocks Everyone

Continuous batching's loop, since Part 3, does one thing per iteration: run a decode step for every active request, then check who's waiting to admit. `admit_one` — the function that turns a waiting request into an active one — runs a request's *entire* prefill inside that same iteration, as one uninterrupted forward pass. For a 20-token prompt that's invisible. For a 400-token prompt, it's roughly the same wall-clock cost as ten ordinary decode steps, and every request already streaming has to sit through all of it before its next token shows up.

![One Long Prefill Freezes Every Active Decoder|700](/images/blogs/nanoserve/chunked-prefill-freeze-problem.png)

Nothing built so far — not the scheduler, not the radix cache — touches this. The scheduler decides *whether* a request gets admitted; once it's in, `admit_one` still runs its prefill as one atomic block. A request three tokens into its own long prompt gets exactly as much of a head start as a request three hundred tokens in: none, because there's no such thing as "partway through admission" yet.

## Chunked Prefill

The general idea is direct: don't run a long prefill as one call. Split it into fixed-size pieces, and give the decode loop a turn between pieces. A prompt that used to cost one 400-token forward pass now costs several smaller ones, each followed by everyone else's decode step.

Sarathi-Serve (Agrawal et al., OSDI 2024) is the paper this comes from, and it names the failure mode directly: a **generation stall**, a decode step delayed by a prefill sharing its batch. Its scheduler admits decodes first, then in-flight prefills, then new prefills, all packed into one fused forward pass under a fixed per-iteration token budget — decode tokens and a prefill chunk, computed together. vLLM V1 runs the same decode-first, token-budget policy by default. SGLang chunks prompts via `--chunked-prefill-size`, and by a 2024 maintainer discussion, doesn't fuse prefill and decode into one batch unless `--enable-mixed-chunk` is explicitly set — closer to what NanoServe (our approach) builds. vLLM's Spyre accelerator backend goes further in the same direction: it can only compile prefill for one fixed chunk size, so it interleaves separate decode and prefill passes to get the same effect, because fusing isn't an option there either.

## Our Approach 

The fused version needs an attention kernel that can process a mixed batch — some rows decoding, one row mid-prefill, all in a single call. NanoServe's decode batch is a padded `(B, 1)` gather with no such kernel on MPS, the same limitation Part 3 and Part 4 already flagged for batched decode itself. Building one is a real project on its own, not a change to this step.

So NanoServe alternates instead of fusing: **one decode step, then one prefill chunk, as two separate forward calls**, decode always going first. This is the same design vLLM's Spyre backend and SGLang's default both land on when fusion isn't available or isn't enabled.

![Alternating vs Fused Chunked Prefill|800](/images/blogs/nanoserve/chunked-prefill-alternating-vs-fused.png)

A few more decisions, each deliberate:

- **Chunk size is a fixed 128 tokens**, and must be a multiple of the paged allocator's 16-token block size — a chunk boundary that didn't align with a block boundary would need the same kind of copy-on-write machinery Part 5 avoided by keeping the radix tree block-aligned.
- **One prompt prefills at a time.** It holds a batch slot for the whole time it's chunking, exactly like a normally-admitted request. Other arrivals wait behind it, and that wait counts as ordinary queue time.
- **Chunking lives only in the batch loop.** The single-request `generate()` path is unchanged — there's nothing to interleave with when only one request exists, so the flag simply doesn't apply there.
- **The radix cache still only sees complete blocks, inserted after the last chunk** — the same rule Part 5 set for the ragged tail, unaffected by how many calls it took to get there.

## Implementation

`admit_one` splits into two functions. `start_prefill` does the radix match and sets up a `PagedKVCache`, but runs no model code:

```python
def start_prefill(req, queue_time_s):
    input_ids = self._build_input_ids(req["prompt"])
    prompt_tokens = input_ids[0].tolist()
    seq_len = input_ids.shape[1]
    cache = PagedKVCache(self.pool)

    matched_block_ids, matched_nodes = self.radix_cache.match(prompt_tokens)
    # ... same max_m / seed logic as Part 5 ...
    cache.block_table = list(matched_block_ids)
    cache.num_tokens = matched_len

    prefilling = {"cache": cache, "seq_len": seq_len, "next_pos": matched_len, ...}
```

`advance_prefill` runs exactly one chunk per call — the entire remainder in one call when chunking is off, which keeps this identical to Parts 3–5 by default:

```python
def advance_prefill():
    p = prefilling
    start, seq_len = p["next_pos"], p["seq_len"]
    end = min(start + chunk_size, seq_len) if chunked else seq_len

    cache.begin_step(end - start)
    outputs = self.model(
        p["input_ids"][:, start:end], past_key_values=cache, use_cache=True,
        cache_position=torch.arange(start, end, device=self.device),
    )
    cache.commit_step()
    p["next_pos"] = end
    if end < seq_len:
        return  # more chunks left; the decode batch gets the next turn

    # last chunk: sample token 1, insert complete blocks into the radix
    # tree, hand the request to the decode batch — identical to Part 5's
    # admit_one tail from here down
    ...
```

Chunk *k* attends to the KV that chunks `0..k-1` already wrote into the paged cache — the exact mechanism Part 5's radix remainder path already relies on, just called once per chunk instead of once per request. Nothing in `PagedKVCache` itself changed.

The loop's shape is the same two-line change either way:

```python
if active:
    # ... one decode step, same as Part 3 ...

if prefilling is not None:      # decode-first: chunk runs AFTER the decode step
    advance_prefill()

try_admit()
```

`slots_free()` returns 0 while a prompt is mid-prefill, so nothing else gets admitted into that slot until it finishes — the same one-at-a-time discipline as ordinary admission, just spread across more iterations. TTFT is measured from admission to first token, which now honestly includes every decode step it waited behind.

> Code: <https://github.com/arponkapuria/NanoServe>

## Findings

Same method as every earlier part: real requests, real timing on the same M1, nothing simulated. Two tests, matching the two places a long prefill actually matters.

### Single Long Prompt, Alone

No other requests running — this test isolates the *cost* of chunking, not the benefit. A 433-token prompt, unchunked vs. chunked at 128.

![Single Long Prompt: Unchunked vs Chunked|650](/images/blogs/nanoserve/chunked-prefill-single-long-prompt.png)

**This is a cost, and it's supposed to be.** Four forward calls instead of one means paying MPS's fixed per-call dispatch overhead four times instead of once — TTFT goes up about 37%. TPOT doesn't move, because decode itself is unchanged; peak memory drops slightly, since each chunk only ever materializes logits for its own tokens instead of the whole prompt at once. The identical output text is the correctness check this test exists for: chunking changes *when* work happens, never *what* gets computed.

### Three Streaming Requests, Then a Long Prompt Arrives

This is the test the whole feature is for. Three short requests start streaming, then a 433-token prompt arrives mid-decode. Every gap between consecutive tokens on the three streaming requests gets logged.

![Streaming Requests' Token Gaps While a Long Prompt Arrives|600](/images/blogs/nanoserve/chunked-prefill-decode-stall-timeline.png)

**The freeze is the whole point, and it's visibly gone.** Unchunked, the three streams show one tall spike lasting the entire prefill — over a second and a half of dead air. Chunked, that same window becomes a plateau of much smaller gaps, and the single worst one drops 49%. Nobody freezes for the length of the whole prompt anymore; the longest anyone waits is bounded by one chunk plus one decode step, not by how long the prompt happens to be.

| Metric                    | Unchunked  | Chunked (128) |
| -------------------------- | ---------- | ------------- |
| Worst decoder gap           | 1,574 ms   | 801 ms        |
| Long request TTFT          | 1,115 ms   | 2,408 ms      |
| Throughput                 | 11.31 tok/s| 10.76 tok/s   |
| Peak memory                | 3,866.5 MB | 3,808.3 MB    |

**The long request pays for it, honestly.** Its own TTFT roughly doubles — a decode step for everyone else now runs between every one of its chunks. Throughput drops a little too, the fixed per-call overhead again. This isn't a free win; it's the exact trade the technique is named for, and it's the same shape of number Sarathi-Serve's own paper reports: bounded tail latency for concurrent streams, paid for out of one request's own completion time.

### The Chunk-Size Trade-off

![Chunk Size Sweep: Freeze vs Long-Request TTFT|700](/images/blogs/nanoserve/chunked-prefill-sweep.png)

Smaller chunks shorten the freeze and lengthen the wait for the long request — exactly the shape vLLM's own tuning guidance describes for its token budget. 256 barely helps the freeze (down 34% instead of roughly half) while still costing the long request 62% more TTFT, which is a worse trade than 128 on both axes. 64 and 128 land close enough on worst-gap.

## An Honest Anomaly

**The worst gap isn't set by the chunk — it's set by the join.** In every chunked run, the single largest gap isn't a chunk-sized pause; it's the last chunk *plus* the first decode step right after the long request joins the batch. That step runs noticeably slower — 475–535 ms instead of the usual 220–240 ms — for as long as the newly-joined request stays in the batch, and it happens even in the unchunked run. It's the same padded, gather-based attention cost Part 3 flagged for batched decode: every row in the batch gets recomputed over the full padded length regardless of how short it actually is, so a longer sequence joining the batch makes every step momentarily heavier for everyone in it. Chunking doesn't cause this and doesn't fix it — it's logged here, not patched, the same way Part 4 logged its idle-GPU TTFT spike instead of building a keep-warm mechanism that was out of scope.

That also explains why 64 and 128's worst gaps (689 ms vs. 801 ms) don't cleanly separate: both are dominated by the same join-step cost, not by the chunk size underneath it. The gaps that *only* contain a chunk plus an ordinary decode step scale the way you'd expect — roughly 483 ms at 64, 652 ms at 128, 1,005 ms at 256 — it's just that the join spike sits on top and is currently the taller number in every row.

## What's Next

That closes out the must-have build order from Part 1: naive decode, KV cache, paged KV cache, continuous batching, the scheduler, prefix caching, and now chunked prefill — seven techniques, each one a flag on the same engine, composable together rather than living as seven separate scripts. Speculative decoding and quantization remain on the stretch list, attempted opportunistically and cut without guilt if they don't land cleanly on Apple Silicon's constraints.

What's left is tying it together: a full-stack evaluation with everything switched on at once, a stage-by-stage comparison isolating what each technique actually bought over the one before it, and an honest look at where NanoServe sits next to vLLM and SGLang — not a claim of matching them, but a measured account of the gap and why it exists.