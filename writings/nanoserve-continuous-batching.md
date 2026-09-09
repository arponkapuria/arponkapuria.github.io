---
title: Continuous Batching: Getting More LLM Throughput From the Same GPU
description: Part 3 of NanoServe, a tiny LLM inference engine build series. This article covers continuous batching — why serving requests one at a time wastes the GPU, how an iteration-level scheduler fixes it, and what the numbers actually showed on Apple Silicon.

date: September 09, 2026
modified: September 09, 2026

author: Arpon Kapuria
category: Dev Journal
tags: LLM Inference, AI Systems
---

> Part 3 of the NanoServe build series. Part 1 covered naive decode and KV cache. Part 2 covered paged KV cache — block-based memory that avoids waste, built specifically so this part could use it. This part assumes you've read both, or at least know what a KV cache and a block pool are.
>
> **Part 2:** [Paged KV Cache on Apple Silicon: Fixing Memory Fragmentation Without CUDA](/blogs/posts/nanoserve-paged-kv-cache)

Part 2 ended with a promise it deliberately didn't cash in: paged KV cache doesn't make one request faster, it just avoids wasting memory — and that waste only turns into a real payoff once *multiple* requests actually share the pool at the same time. This part is where that payoff gets collected.

## The Idle GPU Problem

Every engine built so far in this series - serves exactly one request, start to finish, before touching the next one. If request 2 takes 10 seconds to fully generate its answer, request 1 sits and waits, even though the GPU has spare capacity the whole time.

![Idle GPU Problem|600](/images/blogs/nanoserve/idle-gpu-problem.png)

That spare capacity is the actual problem. Decoding one token for one request is a small operation — the GPU reads the whole model's weights from memory just to produce a single next word. Reading those weights costs roughly the same whether the result is used to produce 1 token or 4. So if 4 independent requests are willing to wait for their next token at the same moment, running them *together* gets you 4 tokens for close to the price of 1 — the accelerator was mostly idle at batch size 1 anyway.

Getting several requests to share the accelerator's attention, instead of taking turns, is the entire goal of this step.

## Continuous Batching

The obvious first idea — collect a few requests, run them together until they're *all* done — is called **static batching**, and it breaks the moment requests differ in length, which they always do. If one request needs 20 tokens and another needs 200, the whole batch is stuck waiting on the slow one; a finished request's slot sits empty instead of picking up new work.

![Head of Line Blocking|600](/images/blogs/nanoserve/head-of-line-blocking.png)

A step up from that is **dynamic batching**: instead of a fixed schedule, the system waits for enough requests to arrive (or a short timeout to pass), forms a batch from whatever showed up, and only then runs it. This is more flexible about *when* a batch starts, but once it starts, it behaves just like static batching — no mid-run swapping, so a slow request in the group still holds up everyone else's slot.

**Continuous batching**, introduced by a 2022 paper called **Orca**, fixes this by scheduling at the *iteration* level instead of the *request* level: after every single token is generated, re-decide who's in the batch. Drop anyone who just finished. Let in anyone new who's waiting. Run the next step with the new lineup. The batch's membership can change every token — that's the "continuous" part.

![Continuous Batching|700](/images/blogs/nanoserve/continuous-batching.png)

This is exactly why paged KV cache from Part 2 was a hard prerequisite, not a nice-to-have. Requests that come and go at different times, with different lengths, sharing one memory budget — a single contiguous buffer per request can't do that cheaply. Block-based storage can: freeing a request just returns its blocks to the pool, instantly reusable by whoever's admitted next.

## What Gets Batched

There's a real design fork worth walking through here, because getting it wrong either overcomplicates this step or quietly breaks something two steps down the line.

A brand-new request needs a **prefill** — its whole prompt processed at once. A request that's already running needs a **decode** step — one new token. These don't share a shape: prefill might be 300 tokens wide, decode is 1. You can't naturally combine "one request needs 300 tokens of work" and "another needs 1" into a single clean batched call.

Two ways this gets solved in real systems: Orca's way splits the operation apart internally so prefill and decode can technically share an iteration, at real implementation cost. vLLM's original way doesn't mix them at all — a new request prefills alone, as its own call, and the moment it produces its first token, it joins the shared decode batch.

We went with vLLM's version: **sequential prefill, batched decode**. It's simpler, it's what real systems shipped first, and it leaves a clean problem for later — a very long prompt's prefill currently blocks the rest of the batch until it's done. Fixing that specific problem is its own separate technique, **chunked prefill**, coming in a later part of this series. Solving it now would mean solving two problems at once and being unable to measure either one cleanly.

## The Shared Decode Cache

Part 2's `PagedKVCache` was built for exactly one request. Continuous batching needs something that can hold several requests' state at once and step them together. Rather than replace the old class, we built a second one next to it: `BatchedDecodeCache`. The core difference: instead of one `block_table` and one token count, it keeps a dictionary of them, one entry per active request.

```python
class BatchedDecodeCache(Cache):
    def __init__(self, pool: PagedKVPool):
        self.pool = pool
        self.requests: dict[int, dict] = {}
        self.active_ids: list[int] = []

    def add_request(self, req_id, block_table, num_tokens):
        self.requests[req_id] = {"block_table": list(block_table), "num_tokens": num_tokens}

    def remove_request(self, req_id):
        state = self.requests.pop(req_id)
        self.pool.release(state["block_table"])
```

Every decode step, `begin_step` runs once for the *whole batch*, not once per request — the same "compute the indices once, reuse across every layer" pattern Part 2 already established, just with a batch dimension added. The one genuinely new problem it solves: requests in the same batch are usually at different lengths — one might be on its 12th token, another on its 90th. To run them through the model in one call, every row has to look the same shape, so shorter requests get **padded** up to the longest one currently in the batch, with a mask marking exactly which positions are real tokens versus padding.

That padding only matters if the model actually knows to ignore it, which is what the mask below is for — built once per step and reused by every one of the model's layers, the same way Part 2's indices were.

```python
    def begin_step(self, active_ids: list[int]) -> torch.Tensor:
        self.active_ids = active_ids
        block_size = self.pool.block_size
        device = self.pool.k_pool.device

        write_phys, write_slot, positions = [], [], []
        max_len = 0
        for rid in active_ids:
            st = self.requests[rid]
            pos = st["num_tokens"]
            needed_blocks = -(-(pos + 1) // block_size)
            if needed_blocks > len(st["block_table"]):
                new_blocks = self.pool.allocate(needed_blocks - len(st["block_table"]))
                st["block_table"] += new_blocks
                new_tensor = torch.tensor(new_blocks, dtype=torch.long, device=device)
                st["block_table_tensor"] = torch.cat([st["block_table_tensor"], new_tensor])
            write_phys.append(st["block_table"][pos // block_size])
            write_slot.append(pos % block_size)
            positions.append(pos)
            max_len = max(max_len, pos + 1)

        self._write_phys = torch.tensor(write_phys, device=device)
        self._write_slot = torch.tensor(write_slot, device=device)
        self._max_len = max_len

        blocks_needed = -(-max_len // block_size)
        table = torch.zeros((len(active_ids), blocks_needed), dtype=torch.long, device=device)
        for i, rid in enumerate(active_ids):
            bt_tensor = self.requests[rid]["block_table_tensor"][:blocks_needed]
            table[i, :bt_tensor.shape[0]] = bt_tensor
        self._read_table = table

        real_lens = torch.tensor([self.requests[rid]["num_tokens"] + 1 for rid in active_ids], device=device)
        idx = torch.arange(blocks_needed * block_size, device=device).unsqueeze(0)
        pad_mask = (idx < real_lens.unsqueeze(1))[:, :max_len]  # (batch, max_len) True=real token

        neg_inf = torch.finfo(torch.float16).min
        self._attn_bias = torch.zeros((len(active_ids), 1, 1, max_len), dtype=torch.float16, device=device)
        self._attn_bias.masked_fill_(~pad_mask.view(len(active_ids), 1, 1, max_len), neg_inf)

        return torch.tensor(positions, device=device).unsqueeze(1)  # (batch, 1)
```

`_attn_bias` is that mask, turned into the exact form the attention math wants: a large negative number wherever a position is padding, so it contributes nothing after softmax. `position_ids` is the other new piece — since every request sits at a different point in its own sequence, each row needs its own position for the model's positional encoding to stay correct.

## The Scheduler Loop

The actual scheduler is a loop: whenever there's room and a new request is waiting, run its prefill and add it to the batch. Every iteration after that, run one batched decode step for whoever's currently active, hand each request its new token, and drop anyone who just finished.

```python
while active or pending:
    if not active:
        wait = pending[0]["arrival_delay"] - (time.perf_counter() - wall_start)
        if wait > 0:
            time.sleep(wait)
        try_admit()
        continue

    active_ids = list(active.keys())
    occupancy_samples.append(len(active_ids))
    batched_tokens = torch.cat([active[rid]["next_token"] for rid in active_ids], dim=0)

    position_ids = decode_cache.begin_step(active_ids)
    blocks_needed = decode_cache._read_table.shape[1]   
    cache_position = torch.tensor([decode_cache._max_len - 1], device=self.device)
    attn_mask = decode_cache._attn_bias

    utils.sync()    
    step_t0 = time.perf_counter()   

    outputs = self.model(
        batched_tokens, past_key_values=decode_cache, use_cache=True,
        cache_position=cache_position, position_ids=position_ids, attention_mask=attn_mask,
    )
    utils.sync()
    step_duration = time.perf_counter() - step_t0
    decode_cache.commit_step()

    step_trace.append({
        "step_index": step_index, "num_active": len(active_ids),
        "blocks_needed": blocks_needed, "max_len": decode_cache._max_len,
        "step_duration_ms": step_duration * 1000,
    })
    step_index += 1

    now = time.perf_counter()

    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for i, rid in enumerate(active_ids):
        st = active[rid]
        st["token_times"].append(now - st["last_time"])
        st["last_time"] = now
        tok = next_tokens[i:i + 1]
        st["next_token"] = tok
        st["generated_ids"].append(tok.clone())
        st["num_generated"] += 1
        is_eos = tok.item() == self.tokenizer.eos_token_id
        if is_eos or st["num_generated"] >= st["max_new_tokens"]:
            st["done"] = True

    for rid in list(active.keys()):
        if active[rid]["done"]:
            active[rid]["finish_time"] = time.perf_counter() - wall_start
            finished[rid] = active.pop(rid)
            decode_cache.remove_request(rid)

    try_admit()    
```

That's the entire mechanism, stripped to its logic — no special case for a batch growing or shrinking, because every iteration already recomputes membership from scratch. Laid out as a diagram, one full iteration looks like this:

![One Iteration of the Scheduler Loop|700](/images/blogs/nanoserve/continuous-batching-iteration-loop.png)

This loop is deliberately a fixed skeleton. Later steps in this series — a real priority scheduler, then chunked prefill — don't replace it; they plug smarter logic into specific points inside it (admission, and the prefill call) without touching the loop itself.

## Sizing the Batch

`MAX_BATCH_SIZE` caps how many requests can share a decode step at once. On a machine with higher processing power, this number can simply go up.

What actually limits it here isn't the KV pool — that's already sized generously from Part 2, at 4096 tokens across 256 blocks. It's the *transient* memory a batched forward pass needs, which grows with batch size in a way the persistent pool doesn't. We tested this directly rather than guessing: `MAX_BATCH_SIZE = 4` ran safely, and so did `6` — no crash, only a small peak-memory increase, since the pool's own cost stays fixed regardless of how many requests share it. Past some point this would eventually hit a real OOM on 8GB of unified memory; 6 turned out comfortably under that ceiling for this workload.

## A Fair Test

A first attempt at testing this — a handful of requests arriving once and running to completion — turned out to be misleading. Two short requests finished early, and for a large chunk of the run only 1-2 requests were actually sharing the batch, even though the whole point was to test real concurrency.

The fix: after an initial run, we recorded the **actual moment** each short request finished, and scheduled new "backfill" requests to arrive at exactly that moment — refilling the slot the instant it opens, the way a real server's queue would. This is measured, not guessed: guessing from an average tokens-per-second number would likely land the backfill at the wrong time, since a request's real speed depends on how full the batch already is when it runs.

## The Efficiency Ceiling

With the test fixed, throughput still moved less than expected. Rather than guess at another cause, we logged the exact duration of every decode step, tagged with how many requests were active during it. That data says a far more precise story than any single aggregate number could:

| Requests sharing the step | Time to produce each token |
|---|---|
| 1 | 102.9 ms |
| 2 | 93.3 ms |
| 3 | 78.3 ms |
| 4 | 72.7 ms |
| 6 | 72.4 ms |

Two things fall out of this. First, the improvement is real and it's exactly the mechanism this step exists to prove — each additional request sharing a step costs *less* extra time than the one before it. Second, it **flattens out around 72 milliseconds per token**, and going from 4 requests to 6 barely moved it further. That floor is genuine, explainable compute cost, not a bug: this project's substitute for a fused attention kernel (established in Part 2) pads every request up to the longest one in the batch and computes attention over that full padded width for every row, every step — real work a fused kernel would skip for the shorter rows. Chasing a bigger batch size past this point wouldn't buy more speed, just more memory risk.

## Findings

Same method as every previous part: a discarded warmup run, then a timed run, comparing the same requests run one at a time against running them through the new scheduler.

![Sequential vs Continuous Batching — TTFT, TPOT, Throughput, Peak Memory|700](/images/blogs/nanoserve/sequential-vs-continuous-batching-plot.png)

The chart shows the shape of it; the table underneath has the exact numbers behind each bar.

| Metric | Sequential | Continuous Batching |
|---|---|---|
| TTFT (mean / p99) | 259.8 / 463.7 ms | 206.5 / 351.6 ms |
| TPOT (mean / p99) | 99.0 / 106.4 ms | 428.2 / 767.7 ms |
| Throughput | 9.72 tok/s | **12.81 tok/s (+31.8%)** |
| Peak memory | 3746.3 MB | 3794.0 MB |
| Mean requests sharing each step | 1.0 | 5.19 |

**TTFT actually improved slightly, and that's a side effect worth explaining, not the main point.** Prefill is still sequential and isolated per request either way — it didn't get faster. What changed is that the sequential baseline's TTFT includes waiting behind whichever request ran before it, while under batching a new request's prefill can start the moment there's room.

**TPOT going up per request is the honest cost of the win, not a regression.** Each request now waits for a step doing roughly 5x the total work, so its own tokens arrive a bit slower — the trade-off described earlier, now measured instead of assumed.

**Throughput is the number this step exists to move, and it moved.** +31.8%, with an average of about 5 requests genuinely sharing the accelerator at once, instead of exactly 1.

The chart below is the run's actual timeline, and it makes that average concrete rather than abstract.

![Batch Occupancy Over the Run|700](/images/blogs/nanoserve/batch-occupancy-plot.png)

Active requests climb as the first batch fills in, dip when a couple of short requests finish close together, climb back up the moment the timed backfill requests land, and taper off only at the very end once nothing is left to refill the last slots. The dashed line marks the 5.19 average; the dotted line marks the 6-request cap.

## Paged KV Cache's Payoff

Part 2 measured paged KV cache's memory efficiency in isolation and explicitly deferred the real payoff to this step. Here's that payoff, made concrete: the same 4096-token pool that holds exactly 4 requests under a naive fixed-reservation scheme (1024 tokens set aside per request, regardless of how short it actually is) held **36 requests** under paged allocation in this run, because paging only spends blocks on tokens a request actually generates.

![Concurrent Capacity — Same Memory, Same Pool|700](/images/blogs/nanoserve/concurrent-capacity-plot.png)

That 4-to-36 gap is the concrete, request-counted version of the fragmentation percentages Part 2 measured abstractly — the reason block-based storage was worth building before batching existed to use it.

## What's Next

Continuous batching's admission policy so far is the simplest one possible: if there's room, let the next waiting request in, first-come-first-served, no other consideration. That's fine for a controlled test with a handful of requests, but it falls apart the moment requests arrive faster than the system can serve them, or some requests genuinely matter more than others.

That's a **scheduler** — a real admission policy sitting on top of this exact loop, deciding *who* gets in and *when*, including what happens when the queue outgrows what the system can currently handle. The loop, the batching, the shared cache all stay exactly as built here. Only the admission decision gets replaced.

---

> Code: [https://github.com/arponkapuria/NanoServe](https://github.com/arponkapuria/NanoServe)