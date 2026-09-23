---
title: "The Scheduler: Backpressure, Timeouts, and Knowing When to Say No"
description: Part 4 of NanoServe, a tiny LLM inference engine build series. This article covers how a real scheduler replaces continuous batching's naive admission policy with a bounded queue, backpressure, and timeouts.

date: September 13, 2026
modified: September 13, 2026

author: Arpon Kapuria
category: Dev Journal
tags: LLM Inference, AI Systems
---

> Part 4 of the NanoServe build series. Part 3 covered continuous batching — a loop that lets several requests share one GPU by re-checking who's active after every token; this part assumes you've read that, or at least know what it means to admit a request into a shared batch.
>
> **Part 3:** [Continuous Batching: Getting More LLM Throughput From the Same GPU](/blogs/posts/nanoserve-continuous-batching)

Part 3 fixed a *throughput* problem: get more requests decoding at once instead of one at a time. This part fixes a different problem continuous batching never touches — what happens when more requests show up than the batch, or the GPU, can actually handle. Nothing in Part 3's loop tracked how many requests were waiting outside the batch, or for how long, so a burst of traffic just queued forever with no way to say no. This article covers what a real scheduler needs — a bounded queue, a rejection rule, a timeout rule — why it's kept deliberately separate from the loop's own timing so it stays testable, and how a two-wave burst test proves the whole thing actually holds up.

## The Unbounded Queue Problem

Here's the specific gap. Continuous batching's loop already re-decides, every iteration, who's active. What it never checks is how many requests are sitting outside the batch waiting for a turn. If six are already decoding and ten more arrive, all ten simply wait — with nothing tracking how long, and nothing stopping an eleventh or a thousandth request from joining the same line.

![The Unbounded Queue Problem|650](/images/blogs/nanoserve/unbounded-queue-problem.png)

A real server needs three things this loop doesn't have yet: (1) a queue with an actual size limit, (2) a way to reject new arrivals once that limit is hit, and (3) a way to give up on a request that's been waiting too long to still be worth serving. Each of those is its own small idea, and it's worth being precise about what each one is for before writing any code.

## Three Mechanisms, One Job

#### Queue discipline 

Decides the order in which waiting requests are processed. Initally FCFS/FIFO **(First In, First Out)** comes to our mind but if we think about an alternative to "first come, first served" is serving whoever needs the least work first, since that tends to minimize how long people wait on average. The problem: at the moment a request arrives, you know how long its prompt is, but not how many tokens the model is about to generate in response — so "least work" isn't something you can actually compute in advance. Real production systems like vLLM and TGI don't try to guess around this. They use plain first-come-first-served, and reserve any notion of "priority" for something set from outside — a paying customer's request jumping ahead of a free one — not a length prediction. NanoServe does the same: **FCFS/FIFO.**

#### Backpressure 

Decides when to stop accepting new work entirely, instead of letting the queue grow without bound. The tempting version of this rejects new arrivals once their *estimated wait time* crosses some threshold — but that estimate depends on how full the batch currently is, which itself depends on requests that haven't shown up yet. Too many moving, uncertain parts to trust. The simpler and more common approach, the one vLLM actually uses, is to bound something you can directly count: a fixed cap on how many requests are allowed to be active or waiting at once. NanoServe's queue works the same way — a **fixed maximum depth**. Past that, new arrivals are rejected immediately, tagged `queue_full`.

#### Timeout

Decides when to give up on a request that's already waiting, rather than let it sit indefinitely. The specific time limit is a promise about latency — "nobody waits more than this long before we either admit them or tell them no" — not a number derived from a formula. NanoServe uses a **fixed maximum wait**, and it tracks this rejection as a *different* reason than `queue_full`. That distinction matters for a very practical reason: a spike in timeouts means the system is too slow; a spike in `queue_full` rejections means it's simply out of room. Lumping both into one generic "rejected" counter would hide which of those two very different problems you actually have.

![Scheduler Admission Pipeline|700](/images/blogs/nanoserve/scheduler-admission-pipeline.png)

## A Scheduler Without a Clock

Continuous batching's loop already runs on real wall-clock time — checks against `time.perf_counter()` scattered through a `while` loop, compared against actual GPU step durations that vary somewhat from run to run (Part 3 measured that variation directly). If admission and rejection logic got folded straight into that same loop, a scheduling decision would depend on *exactly when*, in real time, the loop happened to check — which means two runs with identical requests could, near a timing boundary, make different decisions purely from GPU timing noise. That would make bugs nearly impossible to reproduce.

The fix is to give the scheduler no clock of its own. It's a separate `Scheduler` class, and every one of its methods takes the current time as a plain argument instead of reading a real clock internally:

```python
class Scheduler:
    def __init__(self, max_queue_depth: int, max_wait_s: float):
        self.max_queue_depth = max_queue_depth
        self.max_wait_s = max_wait_s
        self.queue: list[dict] = []
        self.rejections = {"queue_full": 0, "timeout": 0}

    def submit(self, request: dict, now: float) -> str:
        if len(self.queue) >= self.max_queue_depth:
            self.rejections["queue_full"] += 1
            return "rejected_queue_full"
        self.queue.append({"request": request, "arrival_time": now})
        return "queued"

    def expire(self, now: float) -> list[dict]:
        keep, expired = [], []
        for entry in self.queue:
            if now - entry["arrival_time"] > self.max_wait_s:
                expired.append(entry)
                self.rejections["timeout"] += 1
            else:
                keep.append(entry)
        self.queue = keep
        return expired

    def pull(self, now: float, num_slots: int) -> list[dict]:
        if num_slots <= 0:
            return []
        admitted, self.queue = self.queue[:num_slots], self.queue[num_slots:]
        for entry in admitted:
            entry["queue_time_s"] = now - entry["arrival_time"]
        return admitted
```

Three methods, matching the three jobs from the last section exactly: `submit` is where backpressure gets checked, `expire` is where stale requests get dropped, `pull` is where FCFS admission actually happens. None of them look at a real clock or call the model — they only know whatever `now` they were handed. That means the whole class can be tested by feeding it made-up timestamps and checking the outcomes, with zero GPU time spent and zero waiting involved.

Wiring it into Part 3's loop is a small, single addition per iteration: check new arrivals against backpressure, drop anything that's timed out, then pull in as many as there's room for.

```python
def try_admit():
    now = time.perf_counter() - wall_start
    intake(now)                                   # backpressure checked here
    for entry in scheduler.expire(now):            # timeouts dropped here
        rejected.append({"reason": "timeout", **entry["request"]})
    free_slots = settings.MAX_BATCH_SIZE - len(active)
    for entry in scheduler.pull(now, free_slots):  # FCFS admission here
        admit_one(entry["request"], entry["queue_time_s"])
```

Everything else about Part 3's loop — the shared decode cache, the padded batch step — stays exactly as it was. Only this one function, the admission decision, changed.

## The Two-Wave Test

A single burst of requests can prove backpressure and timeout both fire correctly. It can't prove the scheduler *recovers* afterward — that once a pile of requests has been rejected and timed out, the system goes back to behaving completely normally, with nothing left over from the burst. Those are two different claims, and a test that only checks the first one would leave the more important one unverified.

![Two-Wave Burst Test Design|700](/images/blogs/nanoserve/two-wave-burst-test.png)

So the test needed two separate waves: a burst large enough to exceed both the batch and the queue, then a full drain until nothing is left running or waiting, and only then a second, smaller wave arriving into what should now be a clean, empty system.

## Findings

Same method as every earlier part: real requests, real timing on real hardware, nothing simulated. To see what the scheduler actually buys, the same 22-request, two-wave burst was run twice — once with it on, once switched off entirely, falling back to Part 3's original behavior of eventually admitting everyone, no matter how long that takes.

| Metric | Scheduler ON | Scheduler OFF |
|---|---|---|
| TTFT (mean) | 231.6 ms | 174.6 ms |
| TPOT (mean) | 436.7 ms | 414.4 ms |
| Throughput | 11.68 tok/s | 13.34 tok/s |
| Peak memory | 3859.2 MB | 3902.8 MB |
| Rejections | 6 queue_full, 3 timeout | 0 |
| Mean queue time | 116.5 ms | 0.0 ms |

### Scheduler On vs. Off

Every one of the 22 requests is accounted for: 13 admitted, 6 rejected as `queue_full`, 3 as `timeout` — both rejection reasons firing for exactly the situations they were built to catch. The scheduler-off run is the control that makes this comparison mean something: same 22 requests, zero rejections, all served eventually — proof Part 3's fallback behavior is genuinely unchanged, not just assumed to be.

![Scheduler ON vs OFF|700](/images/blogs/nanoserve/scheduler-on-vs-off-plot.png)

TTFT and TPOT both come out a little worse with the scheduler on, worth being upfront about rather than glossing over. It isn't a regression in how tokens get decoded — the batched step itself is identical — it's the honest cost of the extra bookkeeping on every iteration. Peak memory barely moves either way, which makes sense: memory is only allocated at admission, and that moment looks the same regardless of who let the request through.

### The Burst, Second by Second

![Two-Wave Burst Test Timeline|700](/images/blogs/nanoserve/wave-admission-timeline-plot.png)

Six requests are admitted the instant Wave 1 arrives and run straight through. A seventh — deliberately capped at one output token — finishes fast and frees its slot early, which is the only reason the queue's front-most entry gets pulled in with a real, measured wait instead of an instant one. Three requests further back in that same queue never get a turn: the batch won't free another slot for a long while, so they expire first. Six more arrivals find the queue already full and are turned away on the spot. By the time Wave 2 lands, Wave 1 has fully drained, and its six requests run at completely ordinary TTFT and TPOT — no leftover state, nothing carried over from the burst.

### An Honest Anomaly

One more thing worth explaining rather than averaging away: the first request admitted after Wave 1 drains shows a TTFT several times higher than everything around it, and every request right after it goes straight back to normal, like nothing happened. That looked like a bug at first. But it isn't the scheduler's fault: the scheduler-off run is never idle, so it never hits this, and when the test was rerun a few more times the same pattern kept showing up — a brief pause and nothing happens, a pause of a few seconds and the slowdown comes back, every time. Basically the GPU dozes off a bit when there's nothing to do, and the next request pays a small tax waking it back up. That's a real cost of letting the accelerator sit idle between bursts, not a flaw in the scheduler's logic — a production server would pay the same price, and fixing it would mean a keep-warm mechanism, a different feature entirely from backpressure or timeouts.

## What's Next

Every request handled so far is treated as a total stranger to every other one — same prompt or not, it's computed completely from scratch, no matter how many times an identical or overlapping prefix has already passed through the engine. That's wasted work the moment real traffic has any shared structure at all: the same system prompt, the same few-shot examples, the same long document, prefixed onto many different user questions.

That's what prefix caching solves, and NanoServe's version of it is a **radix cache** — a tree structure built directly on top of the block allocator from Part 2, letting requests that share a prefix share its already-computed KV blocks too, instead of recomputing keys and values that already exist somewhere in the pool.

---

> Code: [https://github.com/arponkapuria/NanoServe](https://github.com/arponkapuria/NanoServe)
> 
> **Part 5:** [Prefix Caching: Reusing the Work You’ve Already Paid For With RadixAttention](/blogs/posts/nanoserve-prefix-caching)