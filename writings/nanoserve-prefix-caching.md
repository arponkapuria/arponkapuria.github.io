---
title: "Prefix Caching: Reusing the Work You’ve Already Paid For With RadixAttention"
description: Part 5 of NanoServe, a tiny LLM inference engine build series. This article covers prefix caching (SGLang's RadixAttention), why NanoServe builds a block-aligned simplification of it on top of the existing paged allocator, and what skipping already-computed prefill actually bought on Apple Silicon.

date: September 23, 2026
modified: September 23, 2026

author: Arpon Kapuria
category: Dev Journal
tags: LLM Inference, AI Systems
---

> Part 5 of the NanoServe build series. Part 4 covered the scheduler — backpressure and timeouts on top of continuous batching's admission loop; this part assumes you've read that, or at least know what it means for a request to sit in a queue waiting for a batch slot.
>
> **Part 4:** [The Scheduler: Backpressure, Timeouts, and Knowing When to Say No](/blogs/posts/nanoserve-scheduler)

Every request handled so far in this series is treated like a total stranger to every other one. Same system prompt, same instructions, same few-shot examples pasted in front of a different question — doesn't matter. The engine recomputes all of it, every single time, from token zero. That's not a bug in anything built so far; nothing built so far had any way to notice two requests share anything. This part gives it that ability.

## The Problem: Redundant Prefill

Picture a support chatbot. Every request it gets starts the same way — a paragraph of system instructions, maybe some account context — followed by whatever the user actually typed. Four different users ask four different questions, but the first eighty-or-so tokens of every single prompt are byte-for-byte identical.

Nothing built in this series so far can tell. Every request runs through the exact same code path as if it were the first and only request the engine has ever seen: full prefill, every token, no exceptions. The shared prefix gets computed multiple times to produce multiple copies of the exact same Key/Value vectors.

![Redundant Prefill Work|600](/images/blogs/nanoserve/redundant-prefill.png)

This is exactly the kind of waste our KV cache didn't touch — it stops a *single* request from redoing its own past tokens, but has nothing to say about two *different* requests sharing tokens. That's a different problem, and it needs a different fix.

## Prefix Caching

The general idea — detect a shared prefix across requests and reuse its already-computed KV instead of recomputing it — is called **prefix caching**. Two real systems implement it differently, and the difference matters for what we build next.

![Prefix Caching Techniques|700](/images/blogs/nanoserve/prefix-caching-techniques.png)

### Hashing

vLLM's approach is hash-based. Chop every sequence into the same fixed-size blocks the paged allocator already uses. Hash each block's token content together with its parent block's hash, so identical token sequences produce identical hash chains. A new request walks its own blocks, looks each hash up in a table, and reuses any physical block that already exists under that hash instead of allocating and writing a new one.

Hashing only ever matches at block boundaries — two sequences either share an entire block's worth of tokens or they don't, no partial credit.

### RadixAttention

SGLang's approach is RadixAttention, and it's worth spending a little more time on since it's the one NanoServe borrows from. Cached sequences live in a **radix tree** — a trie whose edges are labeled with token spans, branching at exactly the point where two sequences start to differ, down to a single token if that's where the split happens. Looking up a match means walking down from the root, following the longest chain of edges that still agrees with the new request's tokens; whatever that reaches is already computed, and only the tokens past that point are new work. Unlike hashing, a radix tree can match down to the exact token where sequences diverge, because it isn't tied to any fixed chunk size — that's specifically what SGLang's paper calls out as the technique's edge over block-hashing.

The name comes from pairing that radix tree with **cache-aware scheduling**: SGLang's scheduler actively prefers running requests that share a cached prefix close together, so the tree stays "hot" and hit rates stay high, rather than treating cache locality as an accident of arrival order. And because a KV cache is finite, the tree also runs an **LRU eviction policy** — every node tracks how recently it was touched, and when the pool runs low, the least-recently-used leaves get reclaimed first, freeing their physical blocks back to the pool. Two more things worth knowing: SGLang's tree caches **entire sequences, prompt and output both**, so a later request can match into a previous request's *generated* text too, not just its prompt; and matching happens at **single-token granularity**, so a split can land anywhere, not just at some fixed chunk boundary. NanoServe's version only does the first half of that: it caches and matches **prompt tokens, prefill only** — a generated token is never inserted into the tree, for reasons the next section gets into.

## Our Approach

Here's the actual design fork, and it's worth being upfront that we don't build the version that matches SGLang's own numbers most closely.

True RadixAttention wants token-level splitting — a page size of 1 — so a match can end anywhere, not just at a block edge. NanoServe already has a paged allocator from Part 2, fixed at 16 tokens per block, and every block in that pool is written and read as one atomic unit. Letting a match end mid-block would mean two different requests sharing *part* of a physical block while diverging in the rest of it — which needs copy-on-write inside a single block, real added machinery neither Part 2's allocator nor `PagedKVCache`'s `update()` method has.

So NanoServe builds a **block-aligned radix tree**: same tree structure — branching, longest-prefix match, refcounted nodes — but every edge is exactly one block (16 tokens), and a match is all-or-nothing per block. This is the same trade-off SGLang itself accepts when it runs with a page size greater than 1 for compatibility with a paged backend; it isn't a shortcut unique to this project, it's a documented, real point in the design space.

![Block Aligned Radix Tree|750](/images/blogs/nanoserve/block-aligned-radix-tree.png)

One direct consequence, worth naming rather than hiding: because matches only ever happen at complete block boundaries, there's never a need to *split* an existing edge the way a true token-level radix tree sometimes must. Structurally, this makes NanoServe's tree closer to a trie over fixed-size symbols than the fully general radix tree SGLang describes — a real simplification that follows directly from reusing the block allocator, not an oversight.

## Caching Scope

Three scope decisions, each one deliberate rather than assumed.

**Prompt tokens only — never generated tokens.** SGLang's production deployments cache entire sequences, prompt and output both, which pays off when many parallel completions branch from a shared, growing context. NanoServe's pool is 256 blocks total, shared with everything else the engine does. Caching every generated token of every request churns through that budget fast, on text that's usually a one-off nobody else will ask for again. Shared *system prompts* — the case this article's example is built around — recur across requests constantly; shared *completions* mostly don't. Prompt-only caching spends the budget where the hit rate actually lives.

**Only complete blocks are ever inserted — the ragged tail never is.** A prompt's length is rarely an exact multiple of 16. Whatever's left over after the last complete block — say, 3 tokens — has to be computed fresh no matter what, and it's about to receive this request's own generated tokens right afterward, written into that same physical block. If that block were shared, a second request matching into it would be reading a block that's actively being mutated by someone else's generation. So the ragged tail always stays private, and only whole, finished blocks ever get handed to the tree.

![Why Ragged Tail Stays Private|700](/images/blogs/nanoserve/ragged-tail-requests.png)

**Eviction is deferred, not built.** Unlike SGLang's LRU policy described above, once a block is handed to NanoServe's tree, it stays there for the rest of the process's life — nothing reclaims it yet. Each node does track a `ref_count`, so the bookkeeping a real eviction pass would need already exists; nothing currently acts on it hitting zero. At this project's scale — a handful of distinct prefixes in testing — that's a real, stated scope cut, the same kind of honest gap Part 4 logged for its idle-GPU TTFT spike rather than quietly working around.

## Implementation: RadixCache

Two small classes, `RadixNode` and `RadixCache`, sitting next to `PagedKVPool` and `PagedKVCache` from Part 2 rather than replacing anything in them.

```
class RadixNode:
    def __init__(self, tokens=(), block_id=None):
        self.tokens = tokens
        self.block_id = block_id
        self.children: dict[tuple, "RadixNode"] = {}
        self.ref_count = 0
        self.last_access = 0.0

class RadixCache:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.root = RadixNode()

    def match(self, tokens: list[int]) -> tuple[list[int], list[RadixNode]]:
        node = self.root
        block_ids, nodes = [], []
        B = self.block_size
        for i in range(len(tokens) // B):
            chunk = tuple(tokens[i * B:(i + 1) * B])
            child = node.children.get(chunk)
            if child is None:
                break
            block_ids.append(child.block_id)
            nodes.append(child)
            node = child
        return block_ids, nodes

    def insert(self, tokens, parent, start_block, new_block_ids):
        node = parent
        inserted = []
        B = self.block_size
        for j, block_id in enumerate(new_block_ids):
            i = start_block + j
            chunk = tuple(tokens[i * B:(i + 1) * B])
            child = node.children.get(chunk)
            if child is None:
                child = RadixNode(tokens=chunk, block_id=block_id)
                node.children[chunk] = child
            child.ref_count += 1
            inserted.append(child)
            node = child
        return inserted

    def release(self, nodes: list[RadixNode]) -> None:
        for n in nodes:
            n.ref_count = max(0, n.ref_count - 1)
```

`match()` walks complete block-chunks from the root, using each chunk's own token content as the dictionary key — an exact-match lookup, not a hash chosen ahead of time, so there's zero ambiguity about whether two chunks are "the same." It stops the instant a chunk doesn't have a matching child, and returns exactly how far it got. `insert()` does the mirror operation for whatever wasn't matched: walk forward from wherever matching stopped, creating a new child node for each newly-computed complete block. A node's `ref_count` goes up every time a request's own match or insert touches it, and back down in `release()` — the scaffolding a future eviction pass would read, unused for now.

## Integration with the Engine

This is the part that could have meant real pain on Part 2's `PagedKVCache`, and didn't need to. The trick is in what gets handed to `begin_step` *before* it's called.

```
matched_block_ids, matched_nodes = self.radix_cache.match(prompt_tokens)

# Logits, unlike KV, are never cached -- always leave at least one token
# to actually forward, even on a 100%-matched repeat request.
max_m = (seq_len - 1) // block_size
m = min(len(matched_block_ids), max_m)
matched_block_ids, matched_nodes = matched_block_ids[:m], matched_nodes[:m]
matched_len = m * block_size

cache = PagedKVCache(self.pool)
cache.block_table = list(matched_block_ids)   # seed with the matched blocks
cache.num_tokens = matched_len                # pretend we're already this far in

cache.begin_step(seq_len - matched_len)       # only the unmatched remainder
outputs = self.model(
    input_ids[:, matched_len:], past_key_values=cache, use_cache=True,
    cache_position=torch.arange(matched_len, seq_len, device=self.device),
)
```

Part 2's `begin_step` was already written to compute "how many blocks does this request need, given how many it already has" — it doesn't care *why* `block_table` already has entries in it, only that it does. Seeding `block_table` and `num_tokens` with the matched prefix before calling `begin_step` means the existing block-allocation math just naturally allocates for the unmatched remainder only. Nothing in `PagedKVCache` itself changed to make this work.

The one real gotcha: **logits aren't cached, only KV is.** If a request's entire prompt matched — a verbatim repeat — there'd be nothing left to run forward on, and no way to know what token comes next. `max_m` exists purely to guarantee at least one token always gets forwarded, even on a 100% match.

![Skip-Prefill Flow|700](/images/blogs/nanoserve/skip-prefill-flow.png)

Continuous batching's `admit_one` gets the identical treatment — match, seed, forward the remainder, insert whatever's newly complete — since it's the same operation, just called from a different place. The one thing that has to change on the *release* side: a finished request can no longer hand its entire `block_table` back to the pool, because the leading blocks might belong to the tree now, not to it.

```
def remove_request(self, req_id: int, radix_cache=None) -> None:
    state = self.requests.pop(req_id)
    self.pool.release(state["block_table"][state["cached_blocks"]:])
    if radix_cache is not None:
        radix_cache.release(state["radix_path"])
```

Only the blocks *past* `cached_blocks` — the private tail and whatever decode added — go back to the free list. The shared prefix blocks stay exactly where they are, for the next request that shows up asking the same question a different way.

## Findings

Two separate tests, matching the two places this actually got wired in: a single request repeatedly hitting a shared prefix, and several requests sharing one under real continuous batching.

### Single-Request

Same method as every earlier part: warmup discarded, five timed runs per condition, averaged. Three conditions, same prompt shape, varying only how much of it the tree has already seen: a guaranteed miss, a partial hit (roughly one of two blocks), and a near-total hit (everything but the one token `max_m` always holds back).

![Radix Prefix Cache: TTFT by Cache-Hit Condition|600](/images/blogs/nanoserve/radix-ttft-by-condition.png)

**This isn't a memory trick — it's a real compute saving, and the gradient proves it.** A miss forwards every token through the model; a partial hit forwards roughly half; a near-full hit forwards almost none. TTFT tracks that shape directly, not just "on beats off" as a binary. That's the actual claim prefix caching makes: it skips real prefill work, not just avoids re-storing something.

### Continuous-Batching

The single-request test proves the mechanism. This one proves it under the condition it's actually built for: several different requests, arriving close together, genuinely overlapping in decode, sharing one prefix none of them typed identically on purpose — a system prompt, in this case an 80-token support-bot instruction block.

| Metric | Radix OFF | Radix ON |
| ------ | --------- | -------- |
| TTFT (mean) | 334.5 ms | 209.8 ms |
| TPOT (mean) | 558.8 ms | 469.5 ms |
| Throughput | 8.27 tok/s | 9.52 tok/s |
| Cache hit rate | — | 73.7% |

**Per-request, the win is bigger than the aggregate suggests, and consistent.** The very first request always misses — nothing to hit yet — and every request after it that shares the prefix drops by roughly half: four separate requests each landed between -52.5% and -52.9% TTFT, run after run. One request came in lower, at -26.6% — see the anomaly note below.

![Per-Request TTFT: Radix Cache ON vs OFF|700](/images/blogs/nanoserve/radix-ttft-per-request.png)

**TPOT moved too, and it isn't decode getting cheaper.** The actual per-token decode cost is identical with the cache on or off — the same step-by-step trace confirms it, step for step. What moved is how long each *admission* takes: a cache-hit prefill finishes faster, and that admission runs synchronously inside the same loop that's also stepping everyone else's decode. A shorter admission is a shorter pause for everyone already running — a real, if secondary, effect of the same mechanism, not a second optimization.

### Memory and Capacity

This is the number the compute-saving numbers above don't show on their own: how much of the pool this actually frees up.

![Concurrent Capacity: Radix Sharing|600](/images/blogs/nanoserve/radix-concurrent-capacity.png)

Nine requests sharing one prefix, each needing 7 blocks independently, need 56 blocks total with no sharing. With the shared prefix paid for exactly once — 5 blocks — and each request only privately holding its own 2-block remainder, the same nine requests cost 21 blocks: a 62.5% reduction. On the same 256-block pool that fits 36 requests this size with no sharing, that reduction buys room for **125** — a 3.47x jump in how many concurrent requests the exact same memory fits, purely from requests having something in common.

This is the same shape of result as Part 3's paged-vs-naive capacity finding — a memory-efficiency win that scales with how many requests share the pool, not a speed number. It's also directionally the same claim vLLM and SGLang's own published prefix-caching results make; the concrete multiplier here is specific to this synthetic nine-request workload, not a reproduction of anyone's benchmark.

### An Honest Anomaly

One request's numbers didn't behave the first time. The second hit in the ramp — right after the first request finished writing the shared blocks — came out *slower* with caching on than off, twice in a row, before a third run showed the expected direction. Every other hit request, across all three runs, improved by roughly half every single time. Only this one position wobbled.

"Twice slower, then once faster" is a coin flip, not a pattern — a real mechanism should reproduce, not reverse on a third try. The more likely explanation is ordinary single-call timing noise landing on the same slot twice by coincidence, not a genuine first-touch cost on Apple Silicon.

One thing that *did* hold up: the scheduler admitted and rejected a slightly different set of requests run to run, from small real timing differences pushing one request's wait just over or under its 2500ms timeout — the same hard-cutoff-meets-jitter effect Part 4 had to give margin for. Whatever got admitted, the radix numbers behaved identically regardless — confirming the scheduler and prefix caching are genuinely independent layers.

## What's Next

Prefix caching solves the problem of requests that *start* the same way. It has nothing to say about a request whose prompt is simply very long on its own — one prefill call, hundreds or thousands of tokens, admitted the way `admit_one` has worked since Part 3: all at once, blocking everyone else's decode step until it's done. A short request arriving right behind a long one waits the entire time, batch slot or not.

That's **chunked prefill**: splitting a long prompt's prefill into pieces small enough to interleave with other requests' decode steps instead of running it as one uninterrupted block. The scheduler, the batching loop, the shared cache, this tree — all of it stays exactly as built. Only how a single big prefill gets sliced changes.

---

> Code: <https://github.com/arponkapuria/NanoServe>
>
> **Part 6:** [Chunked Prefill: Stopping One Long Prompt From Freezing Everyone Else](/blogs/posts/nanoserve-chunked-prefill)