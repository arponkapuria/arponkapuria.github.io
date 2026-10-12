---
title: NanoServe
page_title: NanoServe: A Tiny LLM Inference Engine Built From Scratch on Apple Silicon
description: Project walkthrough of NanoServe, a tiny LLM inference engine built phase by phase on Hugging Face Transformers, with KV caching, paged attention, continuous batching, a scheduler, prefix caching and chunked prefill, each benchmarked on Apple Silicon.
tagline: A Tiny LLM inference engine where every serving optimization is built from scratch
author: Arpon Kapuria
author_url: https://arponkapuria.github.io
stack: Python, PyTorch (MPS), Hugging Face Transformers, Qwen3-1.7B, Matplotlib, uv
cover: Michelangelo - Creation of Adam (cropped).jpg
cover_position: center 45%
---

## System diagram {#system}

```diagram
id: n-system
size: 920 600
alt: System architecture: benchmark scripts build a NanoServeEngine from an EngineConfig; the engine has a continuous batching loop with a scheduler, radix cache, chunked prefill and batched decode, all backed by a paged KV pool; the Hugging Face model runs every forward pass; results go to JSON files and plots
caption: One engine, six feature flags. Every phase plugs into the same loop and the same paged block pool.
---
{
  groups: [
    [215, 20, 480, 540, 'NanoServeEngine (src/nanoserve/engine.py)'],
    [225, 135, 460, 215, 'run_continuous_batch() loop', 60]
  ],
  nodes: [
    ['bench', 'Benchmark scripts\none per phase', 25, 250, 150, 80, 'white'],
    ['cfg', 'EngineConfig\nfeature flags', 235, 60, 190, 55, 'blue'],
    ['gen', 'generate()\nradix > paged > kv > naive', 470, 60, 210, 55, 'yellow'],
    ['sch', 'Scheduler\nFCFS queue, backpressure,\ntimeouts', 240, 170, 180, 70, 'pink'],
    ['radix', 'RadixCache\nblock-aligned prefix trie', 500, 165, 170, 60, 'purple'],
    ['pre', 'Prefill\nstart_prefill, then one\nchunk per iteration', 240, 265, 180, 70, 'orange'],
    ['dec', 'Batched decode step\nmasked, padded', 500, 262, 170, 65, 'green'],
    ['pkv', 'PagedKVCache\nper request', 245, 385, 140, 60, 'blue'],
    ['bdc', 'BatchedDecodeCache\nshared by active requests', 455, 385, 210, 60, 'blue'],
    ['pool', 'PagedKVPool\nk_pool, v_pool, free list', 300, 480, 260, 70, 'grey', 'db'],
    ['hf', 'Hugging Face\nQwen3-1.7B\nfp16 on MPS', 745, 235, 150, 90, 'white'],
    ['res', 'results/*.json\nplot scripts, PNGs', 745, 440, 150, 90, 'grey', 'db']
  ],
  edges: [
    ['bench', 'cfg', '', 'r', 'l', { via: [[200, 290], [200, 87]] }],
    ['cfg', 'gen', '', 'r', 'l'],
    ['cfg', 'sch', '', 'b', 't'],
    ['sch', 'pre', 'admit', 'b', 't'],
    ['pre', 'radix', 'match, insert', 'r.2', 'l.8'],
    ['pre', 'dec', 'joins batch', 'r.7', 'l.8'],
    ['dec', 'hf', '', 'r', 'l'],
    ['pre', 'pkv', 'writes KV', 'b', 't'],
    ['dec', 'bdc', '', 'b.3', 't.35'],
    ['pkv', 'bdc', 'hand-off', 'r', 'l'],
    ['pkv', 'pool', '', 'b', 't.25'],
    ['bdc', 'pool', '', 'b', 't.8'],
    ['bench', 'res', 'metrics JSON, plots', 'b', 'b', { via: [[100, 585], [820, 585]], lx: 460, ly: 575 }]
  ],
  notes: [
    [580, 128, 'single-request paths', 14]
  ]
}
```

## Overview {#overview}

NanoServe is a small LLM inference engine for Apple Silicon, built on top of Hugging Face Transformers and running Qwen3-1.7B in fp16 on an M1 MacBook Air's GPU (PyTorch's MPS backend, 8 GB of unified memory). Engines like vLLM, SGLang and TensorRT-LLM are full of optimizations that look arbitrary from the outside: paged KV caches, continuous batching, prefix caching. I wanted to feel the problem each one solves before reading how it is solved. So every technique here is implemented by hand, in isolation, and benchmarked against the step before it. The next optimization's necessity then shows up in the numbers instead of being asserted.

The engine grows in seven steps, and each step is a flag on one `EngineConfig`: naive decode, KV cache, paged KV cache, continuous batching, a scheduler, prefix caching and chunked prefill. They compose. With every flag on, a request is admitted by the scheduler, its prompt is matched against a radix tree of already-computed KV blocks, the remainder is prefilled in 128-token chunks that interleave with other requests' decode steps, and the request then joins a shared decode batch that reads and writes one pool of fixed-size KV blocks.

Because MPS has no way to write a fused paged-attention kernel from Python, the engine uses a gather-based substitute (copy the blocks into a contiguous tensor with `index_select`, then run ordinary attention). The memory benefits come from the block bookkeeping, not the kernel, so they still hold. The slower attention path is measured and reported instead of hidden. Every step also has a blog post with the full story, linked in the matching section below:

1. [From Naive Decode to KV Cache: Why LLM Serving Gets Faster With One Idea](/blogs/posts/nanoserve-naive-decode-to-kv-caching)
2. [KV Cache on Apple Silicon: Why Memory Fragments and How PagedAttention Helps](/blogs/posts/nanoserve-paged-kv-cache)
3. [Continuous Batching: Getting More LLM Throughput From the Same GPU](/blogs/posts/nanoserve-continuous-batching)
4. [The Scheduler: Backpressure, Timeouts, and Knowing When to Say No](/blogs/posts/nanoserve-scheduler)
5. [Prefix Caching: Reusing the Work You've Already Paid For With RadixAttention](/blogs/posts/nanoserve-prefix-caching)
6. [Chunked Prefill: Stopping One Long Prompt From Freezing Everyone Else](/blogs/posts/nanoserve-chunked-prefill)

The headline numbers, all measured on that one machine:

:::table stack 22% 38%
| Phase | What it fixed | Measured result |
| --- | --- | --- |
| [KV cache](/blogs/posts/nanoserve-naive-decode-to-kv-caching) | Every decode step recomputed the whole sequence | TPOT 385.5 ms to 97.1 ms (about 4x), throughput 2.60 to 10.14 tok/s |
| [Paged KV cache](/blogs/posts/nanoserve-paged-kv-cache) | Memory reserved but unused, and scattered free space | 1.2 to 4.7% waste vs 45 to 92% for a 1024-token reservation; speed at parity |
| [Continuous batching](/blogs/posts/nanoserve-continuous-batching) | One request at a time left the GPU idle | +31.8% throughput; 36 vs 4 requests fit in the same pool |
| [Scheduler](/blogs/posts/nanoserve-scheduler) | Unbounded queue, no way to say no | 22-request burst: 6 rejected as `queue_full`, 3 as `timeout`, clean recovery |
| [Prefix caching](/blogs/posts/nanoserve-prefix-caching) | Shared prompt prefixes recomputed per request | 73.7% hit rate, TTFT -37.3%, 62.5% fewer blocks, 3.47x capacity |
| [Chunked prefill](/blogs/posts/nanoserve-chunked-prefill) | One long prefill froze every active stream | Worst decoder gap 1,574 ms to 801 ms, paid for by the long request's TTFT |
:::

## File structure by component {#files}

The repo is split by what each piece does at runtime. Each tree below lists only the files that belong to that component. The deep dives further down walk through them one phase at a time.

#### Engine core

```
src/nanoserve/
  config.py         EngineConfig feature flags + AggregateMetrics dataclass
  settings.py       model, device, block size, pool size, batch / queue limits,
                    chunk size, benchmark prompt presets
  engine.py         NanoServeEngine: naive, kv-cache, paged, radix generate paths
                    + run_continuous_batch() (scheduler, chunked prefill, decode loop)
  utils.py          device-agnostic sync, peak memory, percentile, save_results
```

#### Memory: paged KV cache

```
src/nanoserve/
  paged_cache.py    PagedKVPool (block allocator), PagedKVCache (one request),
                    BatchedDecodeCache (many requests, one decode call)
  naive_baseline.py fixed-reservation waste + simulated contiguous arena,
                    the baseline paging is measured against
test_fragmentation.py   internal + external fragmentation tests
```

#### Scheduling and prefix reuse

```
src/nanoserve/
  scheduler.py      Scheduler: FCFS queue, fixed-depth backpressure, wait timeout
                    (pure state transitions, no clock, no model calls)
  radix_cache.py    RadixCache / RadixNode: block-aligned prefix trie, refcounted
```

#### Benchmarks

```
benchmark.py                   naive, KV cache and paged runs on one request
continuous_batch_benchmark.py  sequential vs batched, same requests, same pool
scheduler_benchmark.py         two-wave burst, scheduler on vs off
radix_benchmark.py             single request: miss / partial / full prefix hit
radix_ramp_benchmark.py        batched ramp, radix on vs off
radix_capacity_savings.py      block and capacity savings from a saved run
chunked_prefill_benchmark.py   long-prompt overhead + decode-stall test
```

#### Results and write-ups

```
results/
  *.json            one file per run (config, metrics, per-request, step traces)
  plot_*.py         one plot script per experiment
  images/           generated plots
notes/              the long-form write-ups, published as the blog series
```

## Tech stack and design choices {#stack}

The stack is deliberately small, because the point of the project is the serving logic and not the framework around it. The one design choice that shaped everything else is the hardware: an Apple GPU, which rules out the usual CUDA-only shortcuts. The [paged KV cache post](/blogs/posts/nanoserve-paged-kv-cache) goes through that decision in detail.

:::table stack 20% 35%
| Choice | Why I picked it | Why not the alternatives |
| --- | --- | --- |
| **PyTorch on MPS** | Runs on the machine I own, with `index_select`, scatter writes and softmax already implemented in Metal. | **CUDA kernels**: not portable to Apple GPUs.<br>**MLX / vllm-metal**: a different tensor framework and hand-written Metal shaders, a bigger project than this one. |
| **Hugging Face Transformers** | The model code stays unmodified. Every attention layer calls `cache.update()`, so a custom `Cache` subclass plugs in without touching the model. | **Writing the model by hand**: it would bury the serving work under model code. |
| **Qwen3-1.7B, fp16** | Small enough to leave room for a KV pool on 8 GB (the model takes about 3.4 GB), big enough to behave like a real LLM. 28 layers, 8 KV heads, head dimension 128. | **Larger models**: no memory left for the pool. **Tiny toy models**: the timing wouldn't mean anything. |
| **Gather-based paged attention** | Real scattered blocks plus an `index_select` gather: the same memory behavior as vLLM, with a slower attention step. | **Fused paged kernel**: not available on MPS. |
| **Block size 16, 256 blocks** | 16 is vLLM's default, so numbers stay comparable. 256 blocks is 4,096 tokens, about 450 MB, which leaves headroom on 8 GB. | **8 or 32 tokens**: more tracking overhead, or coarser reuse. **512 blocks**: roughly 900 MB, a wider margin worth revisiting once there are real concurrency numbers. |
| **Feature flags on one engine** | Each phase can be switched on or off, so every benchmark has a control run. | **Separate scripts per phase**: they stop composing and can't be compared fairly. |
| **uv, hatchling, Python 3.12+** | Fast, reproducible installs from a lockfile. | **pip + requirements.txt**: no lockfile. |
:::

## Naive decode and the KV cache {#c1}

**Blog post:** [From Naive Decode to KV Cache: Why LLM Serving Gets Faster With One Idea](/blogs/posts/nanoserve-naive-decode-to-kv-caching)

Every request is two different jobs: **prefill** reads the whole prompt in one parallel, compute-bound pass, and **decode** produces tokens one at a time in a memory-bound loop. TTFT is almost entirely prefill, and TPOT is almost entirely one decode step. The naive baseline has no cache at all: each step feeds the model the entire sequence so far, so generating N tokens costs about 1 + 2 + ... + N units of work. The KV cache fixes this by saving each token's Keys and Values the first time they are computed and feeding only the newest token afterwards.

```diagram
id: n-kv
size: 760 260
alt: Prompt goes through one prefill pass that fills the KV cache; each decode step reads the cache, feeds one token, and appends one entry
caption: With a cache, prefill happens once and every decode step touches one new token.
---
{
  nodes: [
    ['prompt', 'Prompt\nn tokens', 20, 90, 120, 70, 'white'],
    ['pf', 'Prefill\none forward pass', 200, 90, 150, 70, 'orange'],
    ['kv', 'KV cache\nK, V for every\ntoken, per layer', 410, 75, 160, 100, 'grey', 'db'],
    ['dec', 'Decode step\nfeed 1 new token', 620, 90, 120, 70, 'green']
  ],
  edges: [
    ['prompt', 'pf', '', 'r', 'l'],
    ['pf', 'kv', 'store K, V', 'r', 'l'],
    ['kv', 'dec', 'read past', 'r', 'l'],
    ['dec', 'kv', 'append 1 token', 'b', 'b', { via: [[680, 225], [490, 225]], lx: 585, ly: 240 }]
  ],
  notes: [
    [380, 40, 'naive decode has no cache, so every step re-feeds the whole sequence: O(N^2) work', 15]
  ]
}
```

##### Flow

1. The engine builds the chat-formatted prompt with thinking turned off and tokenizes it.
2. Prefill runs once over the whole prompt and returns the cache plus the first token (greedy `argmax`).
3. Each decode step feeds only the previous token together with the cache.
4. The cache grows by one entry per layer, the next token is picked, and the loop stops at EOS or `max_new_tokens`.

##### Explanation

The two loops from `engine.py` differ by one idea. In the naive version the model is called on `generated`, the full growing sequence. In the cached version the first call is the prefill and the loop feeds `next_token` alone.

```py src/nanoserve/engine.py
# naive: full forward pass over every token, every step
for _ in range(max_new_tokens):
    logits = self.model(generated).logits
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

# KV cache: prefill once, then one token per step
outputs = self.model(input_ids, use_cache=True)
past_key_values = outputs.past_key_values
next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
for _ in range(max_new_tokens - 1):
    outputs = self.model(next_token, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
```

:::table stack 28% 24%
| Metric (medium prompt, 128 tokens) | Naive | KV cache |
| --- | --- | --- |
| TTFT | 218.9 ms | 217.5 ms |
| TPOT | 385.5 ms | **97.1 ms** |
| Throughput | 2.60 tok/s | **10.14 tok/s** |
| Peak memory | 3,447.5 MB | 3,309.8 MB |
:::

![Naive decode vs KV cache: TTFT, TPOT, throughput and peak memory](/images/blogs/nanoserve/naive-vs-kv-cache-plot.png)

:::qa Why didn't TTFT change?
TTFT is the prefill, and the prefill is identical in both versions. The cache only helps tokens after the first one, so a workload that is mostly about first-response latency won't gain much from it alone.
:::

:::qa Why only about 4x when the algorithmic saving is much bigger?
The benchmark runs at batch size 1 on an M1 GPU, where a large part of each step is the fixed cost of launching GPU operations, and that cost doesn't shrink when there is less math. The measured gap between the theoretical and the real speedup is a result in itself.
:::

:::qa Why did peak memory go down after adding a cache?
Naive decode allocates full-sequence attention buffers on every step, and by the end of a 128-token run those cover a sequence of over 200 tokens, recreated each time. The cached version computes activations for one token per step, so the transient waste it removes outweighs the storage it adds.
:::

:::gaps
##### Trade-offs and gaps

- Decoding is greedy `argmax` only. There is no sampling, temperature or top-p, so quality metrics aren't part of any benchmark.
- The cache is one growing tensor per request, which is exactly what the next phase has to fix.
- All numbers come from one machine and a handful of timed runs after a warmup, with no confidence intervals.
:::

## Paged KV cache {#c2}

**Blog post:** [KV Cache on Apple Silicon: Why Memory Fragments and How PagedAttention Helps](/blogs/posts/nanoserve-paged-kv-cache)

A contiguous per-request cache breaks down once many requests share one memory budget. **Internal fragmentation** is memory reserved but never used (reserve 1,024 tokens, generate 80). **External fragmentation** is free memory scattered into pieces too small to use. PagedAttention borrows the operating system's answer: fixed-size blocks, a per-request block table that maps logical blocks to wherever they landed physically, and a free list. Because every block is interchangeable, allocation is "pop any index", and external fragmentation can't happen.

```diagram
id: n-paged
size: 860 390
alt: Two requests each have a block table; their logical blocks point to scattered physical blocks in one shared pool, while unused blocks sit on a free list
caption: Logical blocks (top) land on arbitrary physical blocks (bottom). White blocks are free and can go to any request.
---
{
  groups: [
    [25, 15, 290, 95, 'Request A: block_table = [3, 0, 5]'],
    [455, 15, 215, 95, 'Request B: [1, 4]'],
    [10, 190, 840, 150]
  ],
  nodes: [
    ['a0', 'logical 0', 40, 50, 80, 45, 'blue'],
    ['a1', 'logical 1', 140, 50, 80, 45, 'blue'],
    ['a2', 'logical 2', 240, 50, 80, 45, 'blue'],
    ['b0', 'logical 0', 470, 50, 80, 45, 'green'],
    ['b1', 'logical 1', 570, 50, 80, 45, 'green'],
    ['p0', 'block 0', 20, 235, 80, 60, 'blue'],
    ['p1', 'block 1', 120, 235, 80, 60, 'green'],
    ['p2', 'block 2\nfree', 220, 235, 80, 60, 'white'],
    ['p3', 'block 3', 320, 235, 80, 60, 'blue'],
    ['p4', 'block 4', 420, 235, 80, 60, 'green'],
    ['p5', 'block 5', 520, 235, 80, 60, 'blue'],
    ['p6', 'block 6\nfree', 620, 235, 80, 60, 'white'],
    ['p7', 'block 7\nfree', 720, 235, 80, 60, 'white']
  ],
  edges: [
    ['a0', 'p3', '', 'b', 't', { via: [[80, 125], [360, 125]] }],
    ['a1', 'p0', '', 'b', 't', { via: [[180, 138], [60, 138]] }],
    ['a2', 'p5', '', 'b', 't', { via: [[280, 151], [560, 151]] }],
    ['b0', 'p1', '', 'b', 't', { via: [[510, 164], [160, 164]] }],
    ['b1', 'p4', '', 'b', 't', { via: [[610, 177], [460, 177]] }]
  ],
  notes: [
    [430, 322, 'PagedKVPool: k_pool and v_pool, 256 blocks of 16 tokens (8 shown)', 16],
    [430, 368, 'allocate = pop a block id off the free list, release = push it back, both O(1)', 15]
  ]
}
```

##### Flow

1. At startup the engine allocates the whole pool once: two tensors, `k_pool` and `v_pool`, shaped (layers, blocks, kv heads, block size, head dim), plus a free list of block ids.
2. Before each forward pass, `begin_step(n)` works out how many blocks the new tokens need, allocates any missing ones, and precomputes the write and read indices once.
3. The model runs normally. Each of the 28 layers calls `update()`, which scatters the new K and V into their blocks and gathers the sequence's blocks into one contiguous tensor for attention.
4. `commit_step()` advances the token count. The same three calls handle prefill (`n` is the prompt length) and decode (`n` is 1).
5. When a request finishes, `free()` returns its blocks to the free list immediately.

##### Explanation

The pool is plain bookkeeping with no GPU-specific logic. `PagedKVCache` subclasses Hugging Face's `Cache` and implements `update()`, so the unmodified model calls it without knowing paging exists.

```py src/nanoserve/paged_cache.py
class PagedKVPool:
    def __init__(self, num_layers, num_kv_heads, head_dim, block_size, num_blocks, device, dtype):
        self.block_size = block_size
        shape = (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        self.k_pool = torch.zeros(shape, device=device, dtype=dtype)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.free_blocks = list(range(num_blocks))

    def allocate(self, n: int) -> list[int]:
        if len(self.free_blocks) < n:
            raise RuntimeError(f"KV pool exhausted: need {n} blocks, {len(self.free_blocks)} free")
        return [self.free_blocks.pop() for _ in range(n)]
```

`update()` is where the substitute for vLLM's fused CUDA kernel lives: a vectorized write of all new tokens in one indexed assignment, then a gather with `index_select`.

```py src/nanoserve/paged_cache.py
def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
    self.pool.k_pool[layer_idx, self._write_phys, :, self._write_slot, :] = key_states[0].permute(1, 0, 2)
    self.pool.v_pool[layer_idx, self._write_phys, :, self._write_slot, :] = value_states[0].permute(1, 0, 2)

    k = self.pool.k_pool[layer_idx].index_select(0, self._read_phys)   # the gather
    v = self.pool.v_pool[layer_idx].index_select(0, self._read_phys)
    ...
    return k, v   # contiguous (1, kv_heads, seq_len, head_dim) for ordinary attention
```

:::table stack 30% 22%
| Internal fragmentation (tokens held) | Paged waste | 1,024-token reservation waste |
| --- | --- | --- |
| 79 (short) | 1.25% | 92.29% |
| 183 (medium) | 4.69% | 82.13% |
| 561 (long) | 2.60% | 45.21% |
:::

For external fragmentation I drove the real pool through an adversarial pattern: allocate A (4 blocks), allocate B (3 blocks), free A, then ask for C (6 blocks). The paged pool succeeds. A simulated first-fit contiguous arena fails, because its largest free span is 4 and C needs 6, even though more than enough memory is free in total.

![Internal fragmentation: paged vs fixed reservation](/images/blogs/nanoserve/internal-fragmentation-waste-plot.png)

![Single-request speed: plain KV cache vs paged KV cache](/images/blogs/nanoserve/kv-cache-vs-paged-kv-cache-plot.png)

:::qa Why a gather instead of a fused kernel?
The fused CUDA kernel can't be ported. It is compiled for NVIDIA's architecture, while Apple GPUs run Metal Shading Language, and PyTorch's MPS backend only exposes a fixed set of pre-written operations. The memory win comes from the block bookkeeping, not the kernel, so a slower gather still delivers what paging is for. The cost is measured instead of hidden.
:::

:::qa Does paging make a single request faster?
No, and it can't. It does the same attention math as a plain cache plus a gather. TTFT and TPOT land within noise of the plain cache, and peak memory rises by about 446 MB, which is almost exactly the pool's fixed upfront allocation. The payoff only appears when several requests share the pool, which is what the next phase measures.
:::

:::qa Why does `begin_step` exist separately from `update`?
The model calls `update()` once per layer, 28 times per forward pass, but blocks must be allocated once per pass or they would be allocated 28 times. `begin_step` does allocation and index math once, and every layer reuses the result.
:::

:::qa Why 16 tokens per block?
It is vLLM's default, so numbers stay comparable. Smaller blocks waste less per sequence but add tracking and more, smaller gathers. Larger blocks mean fewer gathers but coarser reuse: a 3-token sequence would still occupy a full 32-slot block.
:::

:::gaps
##### Trade-offs and gaps

- Attention is a two-step gather and compute, so every step copies the sequence's blocks. A fused kernel would skip that pass of memory movement.
- Internal waste is bounded by `block_size - 1` tokens per sequence, not zero.
- The pool is a fixed 256 blocks. Running out raises an error, and there is no preemption or swapping.
- The fragmentation comparison uses an isolated naive allocator, not a full competing engine, so it corroborates vLLM's published figures and doesn't reproduce them.
:::

## Continuous batching {#c3}

**Blog post:** [Continuous Batching: Getting More LLM Throughput From the Same GPU](/blogs/posts/nanoserve-continuous-batching)

Decoding one token reads the entire model's weights from memory regardless of whether that produces one token or four, so a GPU running a single request is mostly idle. Static batching waits for the slowest request, and dynamic batching only improves when a batch starts. **Continuous batching** (from the Orca paper) schedules at the iteration level: after every token, drop the finished requests, admit waiting ones, and run the next step with the new lineup.

```diagram
id: n-cb
size: 860 330
alt: Loop of try_admit, prefill, batched decode step, finish check and remove_request, repeating every iteration
caption: One iteration of run_continuous_batch(). Batch membership is recomputed from scratch each time, so growing or shrinking needs no special case.
---
{
  nodes: [
    ['adm', 'try_admit()\nscheduler or free slot', 30, 120, 170, 70, 'pink'],
    ['pre', 'Prefill\nnew request, own call', 260, 120, 170, 70, 'orange'],
    ['dec', 'Batched decode step\nall active requests', 490, 120, 190, 70, 'green'],
    ['fin', 'Sample + finish check\nargmax per row, EOS\nor max tokens', 490, 245, 190, 70, 'yellow'],
    ['rem', 'remove_request()\nblocks back to pool', 240, 250, 190, 60, 'blue']
  ],
  edges: [
    ['adm', 'pre', 'free slot', 'r', 'l'],
    ['pre', 'dec', 'joins batch', 'r', 'l'],
    ['dec', 'fin', '', 'b', 't'],
    ['fin', 'rem', 'done', 'l', 'r'],
    ['rem', 'adm', 'next iteration', 'b', 'b', { via: [[335, 295], [115, 295]], lx: 230, ly: 285 }]
  ],
  notes: [
    [440, 55, 'membership is re-decided after every decode step', 16]
  ]
}
```

##### Flow

1. A new request prefills on its own (vLLM's original design: sequential prefill, batched decode), then hands its block table and token count to the shared `BatchedDecodeCache`.
2. Each iteration, `begin_step` runs once for the whole batch. It allocates a block for any request that needs one and builds the batched write indices, read table and attention mask.
3. One model call decodes a token for every active request. Each row has its own `position_ids`.
4. Each request gets its token. Anyone who hit EOS or `max_new_tokens` is removed and their blocks return to the pool.
5. `try_admit()` fills any freed slot, up to `MAX_BATCH_SIZE` (6 here).

##### Explanation

Requests in one batch sit at different lengths, so shorter rows are padded up to the longest and a mask marks which positions are real. The mask is built once per step and reused by all 28 layers, the same compute-once pattern as paging.

```py src/nanoserve/paged_cache.py
real_lens = torch.tensor([self.requests[rid]["num_tokens"] + 1 for rid in active_ids], device=device)
idx = torch.arange(blocks_needed * block_size, device=device).unsqueeze(0)
pad_mask = (idx < real_lens.unsqueeze(1))[:, :max_len]   # True = real token

neg_inf = torch.finfo(torch.float16).min
self._attn_bias = torch.zeros((len(active_ids), 1, 1, max_len), dtype=torch.float16, device=device)
self._attn_bias.masked_fill_(~pad_mask.view(len(active_ids), 1, 1, max_len), neg_inf)

return torch.tensor(positions, device=device).unsqueeze(1)   # per-request position_ids
```

To test it fairly, I first logged the real finish time of each short request and scheduled backfill requests to arrive at exactly those moments, because a first test with a few one-shot arrivals left only 1 to 2 requests sharing the batch for much of the run. Then I logged every decode step with the number of requests in it:

:::table stack 40% 30%
| Requests sharing the step | Time per token |
| --- | --- |
| 1 | 102.9 ms |
| 2 | 93.3 ms |
| 3 | 78.3 ms |
| 4 | 72.7 ms |
| 6 | 72.4 ms |
:::

:::table stack 30% 25%
| Metric | Sequential | Continuous batching |
| --- | --- | --- |
| TTFT (mean / p99) | 259.8 / 463.7 ms | 206.5 / 351.6 ms |
| TPOT (mean / p99) | 99.0 / 106.4 ms | 428.2 / 767.7 ms |
| Throughput | 9.72 tok/s | **12.81 tok/s (+31.8%)** |
| Peak memory | 3,746.3 MB | 3,794.0 MB |
| Mean requests per step | 1.0 | 5.19 |
:::

![Sequential vs continuous batching](/images/blogs/nanoserve/sequential-vs-continuous-batching-plot.png)

![Batch occupancy over the run](/images/blogs/nanoserve/batch-occupancy-plot.png)

![Concurrent capacity in the same memory pool](/images/blogs/nanoserve/concurrent-capacity-plot.png)

This is where paging pays off. The 4,096-token pool holds exactly 4 requests under a naive scheme that sets aside 1,024 tokens each, and 36 requests of this workload's size under paged allocation.

:::qa Why sequential prefill with batched decode, instead of mixing them?
A prefill might be 300 tokens wide and a decode step is 1, so they don't share a shape. Orca splits operations internally to mix them, at real implementation cost. vLLM's first design didn't mix them at all. I took the simpler one, and it leaves a clean problem for later: a long prefill blocks the batch, which is exactly what [chunked prefill](#c6) fixes.
:::

:::qa TPOT got worse. Isn't that a regression?
Per request, yes: each token now waits for a step doing about 5x the total work. System throughput is what rose (+31.8%), with about 5 requests sharing the accelerator instead of 1. It is the trade-off batching makes, measured instead of assumed.
:::

:::qa Why does the speedup flatten at about 72 ms per token?
The gather-based attention pads every row to the longest sequence and computes over the full padded width every step, which a fused kernel would skip for shorter rows. Going from 4 to 6 requests barely changed the step time, so a bigger batch would only add memory risk.
:::

:::gaps
##### Trade-offs and gaps

- Admission is limited by `MAX_BATCH_SIZE`, not by free blocks, so the pool can still run dry mid-run with nothing to catch it.
- Padding to the longest sequence wastes compute that grows with length variance.
- `BatchedDecodeCache.begin_step` builds its index tensors with Python loops over requests, and the loop syncs the device every step.
- The 36 vs 4 capacity figure is a capacity calculation for this workload. The engine's actual batch is capped at 6.
:::

## The scheduler {#c4}

**Blog post:** [The Scheduler: Backpressure, Timeouts, and Knowing When to Say No](/blogs/posts/nanoserve-scheduler)

Continuous batching tracks who is active, but it never checked how many requests were waiting outside the batch, or for how long. A burst just queued forever. A real server needs three things the loop didn't have: a queue with a size limit, a rule for rejecting arrivals once it is full, and a rule for giving up on a request that has waited too long. Each one answers a different question, and the two rejections are tagged with different reasons on purpose: a spike in `timeout` means the system is too slow, a spike in `queue_full` means it is out of room.

```diagram
id: n-sched
size: 920 340
alt: A request arrives and is checked against the queue depth; if the queue is full it is rejected as queue_full, otherwise it joins the FCFS queue; waiting longer than 2.5 seconds gets it rejected as timeout, otherwise it is pulled into a free batch slot
caption: Three methods, three jobs: submit() applies backpressure, expire() applies the timeout, pull() does FCFS admission.
---
{
  nodes: [
    ['arr', 'Arrival\nsubmit(req, now)', 15, 125, 120, 60, 'white'],
    ['full', 'queue\nfull?', 165, 105, 140, 100, 'yellow', 'diamond'],
    ['rej1', 'Rejected:\nqueue_full', 185, 245, 100, 55, 'pink'],
    ['q', 'FCFS queue\nmax depth 10', 355, 120, 130, 70, 'blue'],
    ['old', 'waited\n> 2.5 s?', 525, 105, 140, 100, 'yellow', 'diamond'],
    ['rej2', 'Rejected:\ntimeout', 545, 245, 100, 55, 'pink'],
    ['pull', 'pull(), then admit\nprefill + join batch', 715, 120, 190, 70, 'green']
  ],
  edges: [
    ['arr', 'full', '', 'r', 'l'],
    ['full', 'rej1', 'yes', 'b', 't'],
    ['full', 'q', 'no', 'r', 'l'],
    ['q', 'old', '', 'r', 'l'],
    ['old', 'rej2', 'yes', 'b', 't'],
    ['old', 'pull', 'no', 'r', 'l']
  ],
  notes: [
    [460, 50, 'every method takes now as an argument: no real clock and no model calls inside', 15],
    [460, 325, 'MAX_QUEUE_DEPTH = 10, MAX_QUEUE_WAIT_MS = 2500, MAX_BATCH_SIZE = 6', 15]
  ]
}
```

##### Flow

1. When a request's arrival time passes, `intake()` calls `scheduler.submit(req, now)`. If the queue already holds `MAX_QUEUE_DEPTH` requests, it returns `rejected_queue_full`.
2. Before every admission pass, `expire(now)` drops anything that has waited longer than `MAX_QUEUE_WAIT_MS` and counts it as a `timeout`.
3. `pull(now, free_slots)` takes up to that many requests from the front of the queue (FCFS) and records each one's queue time.
4. Each pulled request goes through the normal prefill path and joins the decode batch. With the scheduler off, the engine falls back to the old unbounded behavior exactly.

##### Explanation

The scheduler is a pure state machine with no clock of its own. If admission logic read `time.perf_counter()` inside the loop, two identical runs could make different decisions near a timing boundary purely from GPU jitter, and bugs would be nearly impossible to reproduce. Passing `now` in makes the whole class testable with made-up timestamps and zero GPU time.

```py src/nanoserve/scheduler.py
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
```

The test is a two-wave burst. A single burst can prove that backpressure and timeouts fire, but not that the system recovers afterwards. So Wave 1 (16 requests, one deliberately capped at one output token so a slot frees early) exceeds both the batch and the queue, the system drains completely, and only then does Wave 2 (6 requests) arrive into what should be a clean, empty engine. The same 22 requests were also run with the scheduler off, as the control.

:::table stack 28% 28%
| Metric | Scheduler ON | Scheduler OFF |
| --- | --- | --- |
| TTFT (mean) | 231.6 ms | 174.6 ms |
| TPOT (mean) | 436.7 ms | 414.4 ms |
| Throughput | 11.68 tok/s | 13.34 tok/s |
| Peak memory | 3,859.2 MB | 3,902.8 MB |
| Rejections | 6 `queue_full`, 3 `timeout` | 0 |
| Mean queue time | 116.5 ms | 0.0 ms |
:::

All 22 requests are accounted for: 13 admitted, 6 rejected as `queue_full`, 3 as `timeout`. The off run serves everyone eventually, which shows the old fallback behavior is genuinely unchanged.

![Scheduler on vs off](/images/blogs/nanoserve/scheduler-on-vs-off-plot.png)

![Two-wave burst: what happened to each request](/images/blogs/nanoserve/wave-admission-timeline-plot.png)

:::qa Why FCFS and not shortest-job-first?
At arrival you know the prompt length but not how many tokens the model will generate, so "least work" can't be computed in advance. vLLM and TGI use plain first-come-first-served and keep any priority for something set from outside, like a paying customer's request. NanoServe does the same.
:::

:::qa Why a fixed queue depth instead of rejecting on estimated wait time?
An estimated wait depends on how full the batch is, which depends on requests that haven't arrived yet. Bounding something you can directly count is simpler and more trustworthy, and it is what vLLM does too.
:::

:::qa Why is TTFT worse with the scheduler on?
The batched decode step is identical. The scheduler adds bookkeeping on every iteration, and the requests that wait in the queue count toward queue time. It is a small, honest cost that the control run makes visible.
:::

:::qa What was the TTFT spike on the first request after Wave 1?
The first request admitted after the system drained showed a TTFT several times higher than its neighbours, then everything went back to normal. The scheduler-off run is never idle, so it never showed this, and a pause of a few seconds brought the slowdown back each time. The most likely explanation is the GPU idling between bursts and the next request paying a wake-up cost. A keep-warm mechanism would be a different feature, so I logged it instead of working around it.
:::

:::gaps
##### Trade-offs and gaps

- FCFS only: no priorities, fairness classes or preemption.
- `expire()` runs once per scheduling pass, so timeout resolution is one loop iteration (hundreds of milliseconds under load), not exact.
- The 2,500 ms timeout is a hard cutoff, so identical runs can admit a slightly different set of requests when jitter pushes one over or under it.
- The queue is bounded by count, not by tokens or memory.
- The idle-GPU TTFT spike is logged but not fixed.
:::

## Prefix caching with a radix tree {#c5}

**Blog post:** [Prefix Caching: Reusing the Work You've Already Paid For With RadixAttention](/blogs/posts/nanoserve-prefix-caching)

Until now every request was a stranger to every other one. A support bot's requests all start with the same system prompt, yet each one recomputed identical Keys and Values from token zero. A **radix tree** over already-computed blocks lets a new request find the longest prefix it shares with earlier requests and reuse those physical blocks, so only the unmatched remainder is prefilled. This is SGLang's RadixAttention idea, built on top of the paged allocator from the earlier phase.

```diagram
id: n-radix
size: 900 350
alt: The radix cache root leads to one shared prefix of five blocks; three requests with different questions branch from it, each with its own unmatched remainder and a private ragged tail that is never cached
caption: The shared system prompt is computed and stored once. Each request only pays for what comes after it.
---
{
  nodes: [
    ['root', 'RadixCache\nroot', 20, 140, 110, 60, 'white'],
    ['sh', 'Shared prefix\n5 blocks = 80 tokens\nsystem prompt', 180, 125, 190, 90, 'purple'],
    ['r1', 'Request 1: France\nunmatched remainder', 430, 35, 210, 60, 'yellow'],
    ['r2', 'Request 2: Japan\nunmatched remainder', 430, 145, 210, 60, 'yellow'],
    ['r3', 'Request 3: Brazil\nunmatched remainder', 430, 255, 210, 60, 'yellow'],
    ['t1', 'ragged tail + decode\nprivate, never cached', 690, 35, 185, 60, 'grey'],
    ['t2', 'ragged tail + decode\nprivate, never cached', 690, 145, 185, 60, 'grey'],
    ['t3', 'ragged tail + decode\nprivate, never cached', 690, 255, 185, 60, 'grey']
  ],
  edges: [
    ['root', 'sh', '', 'r', 'l'],
    ['sh', 'r1', '', 'r', 'l'],
    ['sh', 'r2', '', 'r', 'l'],
    ['sh', 'r3', '', 'r', 'l'],
    ['r1', 't1', '', 'r', 'l'],
    ['r2', 't2', '', 'r', 'l'],
    ['r3', 't3', '', 'r', 'l']
  ],
  notes: [
    [450, 338, 'a hit raises ref_count on the shared nodes: no recompute and no new blocks for the prefix', 15]
  ]
}
```

##### Flow

1. `match()` walks the prompt's complete 16-token chunks from the root and returns the longest run of cached blocks.
2. The match is capped so at least one token is always forwarded. Logits aren't cached, so a verbatim repeat would otherwise have nothing to compute the next token from.
3. The request's `block_table` and `num_tokens` are seeded with the matched blocks, so the existing allocation math only allocates for the remainder.
4. Only the unmatched remainder is prefilled, attending to the cached blocks already sitting in the pool.
5. Newly computed complete blocks are inserted into the tree. The ragged tail never is, because it is about to receive this request's generated tokens.
6. On finish, only blocks past the cached prefix go back to the free list. The shared blocks stay for the next request.

##### Explanation

NanoServe builds a **block-aligned** radix tree: every edge is exactly one block of 16 tokens, so a block either matches whole or not at all. True token-level splitting would let two requests share part of a physical block while diverging in the rest, which needs copy-on-write inside a block that the allocator doesn't have. This is the same trade-off SGLang accepts when it runs with a page size above 1. A side effect is that the tree never needs to split an existing edge.

```py src/nanoserve/radix_cache.py
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
```

The engine integration needed no change to `PagedKVCache`. `begin_step` already computes how many blocks a request needs given how many it has, so seeding the table is enough.

```py src/nanoserve/engine.py
matched_block_ids, matched_nodes = self.radix_cache.match(prompt_tokens)
max_m = (seq_len - 1) // block_size          # always leave >= 1 token to forward
m = min(len(matched_block_ids), max_m)
matched_len = m * block_size

cache.block_table = list(matched_block_ids)  # seed with the shared blocks
cache.num_tokens = matched_len               # pretend we're already this far in
cache.begin_step(seq_len - matched_len)      # allocates for the remainder only
```

Two tests match the two places it is wired in. The single-request test varies only how much of the prompt the tree has seen (five timed runs each, TTFT in milliseconds):

:::table stack 28% 22%
| Condition | Prompt tokens served from cache | TTFT |
| --- | --- | --- |
| Miss | 0% | 367.6 ms |
| Partial hit | 42% | 170.5 ms |
| Near-full hit | all but one token | 95.9 ms |
:::

The gradient proves it is a real compute saving, because TTFT tracks how many tokens still go through the model. The second test is a nine-request ramp under continuous batching, all sharing an 80-token support-bot system prompt:

:::table stack 28% 24%
| Metric | Radix OFF | Radix ON |
| --- | --- | --- |
| TTFT (mean) | 334.5 ms | 209.8 ms (-37.3%) |
| TPOT (mean) | 558.8 ms | 469.5 ms |
| Throughput | 8.27 tok/s | 9.52 tok/s |
| Cache hit rate | n/a | 73.7% |
:::

![TTFT by cache-hit condition](/images/blogs/nanoserve/radix-ttft-by-condition.png)

![Per-request TTFT with the radix cache on vs off](/images/blogs/nanoserve/radix-ttft-per-request.png)

Four of the hit requests each landed between -52.5% and -52.9% TTFT, run after run. The capacity effect is bigger than the speed effect: nine requests that each need 7 blocks cost 56 blocks without sharing. With the 5-block prefix paid for once and a 2-block private remainder each, they cost 21 blocks (-62.5%), which on the 256-block pool takes concurrent capacity from 36 requests to 125 (3.47x).

![Concurrent capacity with prefix sharing](/images/blogs/nanoserve/radix-concurrent-capacity.png)

:::qa Why cache prompt tokens only, never generated ones?
SGLang caches whole sequences, which pays off when many completions branch from a shared context. Here the pool is 256 blocks shared with everything else, and generated text is usually a one-off that nobody asks for again. Shared system prompts recur constantly, so prompt-only caching spends the budget where the hit rate is.
:::

:::qa Why is the ragged tail never inserted?
A prompt is rarely an exact multiple of 16, and whatever is left over is about to receive the request's own generated tokens in the same physical block. If that block were shared, a second request could read a block someone else is actively writing. So only complete, finished blocks go into the tree.
:::

:::qa Why did TPOT improve if decode is unchanged?
The per-step decode cost is identical with the cache on or off, and the step trace confirms it. What moved is how long each admission takes. A cache-hit prefill is shorter, and admission runs synchronously inside the loop that is also stepping everyone else's decode, so every other request pauses for less time.
:::

:::qa One request got slower with caching on. What happened?
The second hit in the ramp came out slower with the cache on, twice in a row, then faster on a third run, while every other hit improved by about half every time. Twice slower and once faster looks like timing noise landing on the same slot, not a mechanism. I logged it as an anomaly instead of averaging it away.
:::

:::gaps
##### Trade-offs and gaps

- **No eviction.** Nodes persist for the life of the process, so the free pool only shrinks. `ref_count` and `last_access` are tracked, but nothing acts on them yet. LRU eviction over `ref_count == 0` leaves is the obvious next step.
- Block alignment means a shared prefix that ends mid-block loses that partial block.
- Prompt tokens only: later requests can't match into earlier generated text.
- The multipliers (-62.5% blocks, 3.47x capacity) are specific to this synthetic nine-request workload and not a reproduction of anyone's benchmark.
:::

## Chunked prefill {#c6}

**Blog post:** [Chunked Prefill: Stopping One Long Prompt From Freezing Everyone Else](/blogs/posts/nanoserve-chunked-prefill)

Since continuous batching, a request's whole prefill has run as one uninterrupted forward pass inside the loop. For a 20-token prompt that is invisible. For a 433-token prompt it costs about as much as ten decode steps, and every stream already running waits through all of it. Sarathi-Serve calls this a **generation stall**. Chunked prefill splits the prefill into fixed-size pieces and gives the decode loop a turn between pieces.

```diagram
id: n-chunk
size: 900 300
alt: Timeline comparing unchunked prefill, where one long forward pass freezes the decoders, with chunked prefill, where one decode step and one 128-token chunk alternate
caption: The same long prompt, sliced. Decoders never wait longer than one chunk plus one decode step.
---
{
  nodes: [
    ['u0', 'decode', 160, 70, 70, 50, 'green'],
    ['u1', 'one 433-token prefill, decoders frozen', 235, 70, 300, 50, 'pink'],
    ['u2', 'decode', 540, 70, 70, 50, 'green'],
    ['c0', 'decode', 160, 180, 70, 50, 'green'],
    ['c1', 'chunk 0', 235, 180, 70, 50, 'orange'],
    ['c2', 'decode', 310, 180, 70, 50, 'green'],
    ['c3', 'chunk 1', 385, 180, 70, 50, 'orange'],
    ['c4', 'decode', 460, 180, 70, 50, 'green'],
    ['c5', 'chunk 2', 535, 180, 70, 50, 'orange'],
    ['c6', 'decode', 610, 180, 70, 50, 'green'],
    ['c7', 'chunk 3', 685, 180, 70, 50, 'orange'],
    ['c8', 'decode', 760, 180, 70, 50, 'green']
  ],
  edges: [],
  notes: [
    [78, 100, 'Unchunked', 17],
    [78, 210, 'Chunked (128)', 17],
    [450, 30, 'decode-first: one decode step, then one prefill chunk, as two separate forward calls', 16],
    [450, 275, 'worst gap between decoder tokens: 1,574 ms unchunked vs 801 ms chunked', 16]
  ]
}
```

##### Flow

1. `start_prefill` tokenizes the prompt, does the radix match and creates the request's `PagedKVCache`, but runs no model code.
2. Each loop iteration runs the decode step for all active requests first.
3. `advance_prefill` then runs exactly one chunk (`PREFILL_CHUNK_SIZE`, 128 tokens), which attends to the earlier chunks already written into the paged cache.
4. After the last chunk, the first token is sampled, complete blocks go into the radix tree, and the request joins the decode batch.
5. While a prompt is mid-prefill it holds a batch slot, and `slots_free()` returns 0 so only one prompt prefills at a time.

##### Explanation

Sarathi-Serve and vLLM V1 fuse decode tokens and a prefill chunk into one forward pass under a token budget. That needs an attention kernel that handles a mixed batch, and NanoServe's decode batch is a padded `(B, 1)` gather, with no such kernel on MPS. So it alternates instead of fusing, as SGLang's default does. Chunk size must be a multiple of the 16-token block size, or a chunk boundary would need the same copy-on-write machinery the radix tree avoided.

```py src/nanoserve/engine.py
def advance_prefill():
    p = prefilling
    cache, seq_len, start = p["cache"], p["seq_len"], p["next_pos"]
    end = min(start + chunk_size, seq_len) if chunked else seq_len   # whole remainder when off

    cache.begin_step(end - start)
    outputs = self.model(
        p["input_ids"][:, start:end], past_key_values=cache,
        use_cache=True, cache_position=torch.arange(start, end, device=self.device),
    )
    cache.commit_step()
    p["next_pos"] = end
    if end < seq_len:
        return   # more chunks to go; the decode batch gets its turn first
    ...          # last chunk: sample token 1, insert blocks into the radix tree, join the batch
```

```py src/nanoserve/engine.py
if active:
    ...                       # one batched decode step, same as before
if prefilling is not None:    # decode-first: the chunk runs AFTER the decode step
    advance_prefill()
try_admit()
```

The first test runs the 433-token prompt alone, which isolates the cost of chunking. The second is the one the feature exists for: three short requests stream about 100 tokens each, then the long prompt arrives at t = 3 s, and every gap between consecutive tokens on the three streams is logged.

:::table stack 28% 22%
| Single long prompt, alone | Unchunked | Chunked (128) |
| --- | --- | --- |
| TTFT | 1,124 ms | 1,539 ms (+37%) |
| TPOT | 142.3 ms | 142.7 ms |
| Peak memory | 3,858.6 MB | 3,767.7 MB |
| Output text | baseline | identical |
:::

That cost is expected: four forward calls instead of one pay MPS's fixed dispatch overhead four times. The identical output text is the correctness check, since chunking changes when work happens and never what is computed.

:::table stack 28% 22%
| Three streams + long arrival | Unchunked | Chunked (128) |
| --- | --- | --- |
| Worst decoder gap | 1,574 ms | **801 ms (-49%)** |
| Long request TTFT | 1,115 ms | 2,408 ms |
| Throughput | 11.31 tok/s | 10.76 tok/s |
| Peak memory | 3,866.5 MB | 3,808.3 MB |
:::

![Single long prompt: unchunked vs chunked](/images/blogs/nanoserve/chunked-prefill-single-long-prompt.png)

![Token gaps on the streaming requests while the long prompt arrives](/images/blogs/nanoserve/chunked-prefill-decode-stall-timeline.png)

Unchunked, the streams show one tall spike lasting the entire prefill. Chunked, that window becomes a plateau of much smaller gaps. The long request pays for it, since a decode step for everyone else now runs between each of its chunks.

:::table stack 24% 25%
| Chunk size | Worst decoder gap | Long request TTFT |
| --- | --- | --- |
| Unchunked | 1,574 ms | 1,115 ms |
| 64 | 689 ms | 3,260 ms |
| 128 | 801 ms | 2,408 ms |
| 256 | 1,072 ms | 1,804 ms |
:::

![Chunk size sweep: freeze vs long-request TTFT](/images/blogs/nanoserve/chunked-prefill-sweep.png)

:::qa Why alternate decode and prefill instead of fusing them?
Fusing needs an attention kernel that can process one batch where some rows decode and one row is mid-prefill. On MPS that is a real project on its own, and the same limitation already shaped batched decode. Alternating is what vLLM's Spyre backend and SGLang's default both fall back to when fusion isn't available or enabled.
:::

:::qa Why does the long request's TTFT roughly double?
A decode step for every other request now runs between each of its chunks, and TTFT is measured from admission to first token, so it honestly includes every decode step it waited behind. Bounded tail latency for concurrent streams is paid out of one request's completion time, which is the same shape of trade-off the Sarathi-Serve paper reports.
:::

:::qa Why isn't the worst gap set by the chunk size?
In every chunked run the largest gap is the last chunk plus the first decode step after the long request joins the batch. That step runs at 475 to 535 ms instead of the usual 220 to 240 ms, because the padded gather recomputes every row over the full padded length, so one longer sequence makes every step heavier. It happens in the unchunked run too. That is why the worst gaps at 64 and 128 don't separate cleanly. The gaps that contain only a chunk and an ordinary decode step do scale as expected, at roughly 483 ms for 64, 652 ms for 128 and 1,005 ms for 256.
:::

:::qa Why must the chunk size be a multiple of 16?
A chunk boundary that didn't land on a block boundary would leave a block partly written by one chunk and finished by the next, and the radix cache would then need copy-on-write inside a block. Keeping chunks block-aligned avoids it.
:::

:::gaps
##### Trade-offs and gaps

- One prompt prefills at a time and holds a batch slot while it does, so other arrivals queue behind it.
- Alternating isn't fusing: the fused version amortizes dispatch overhead and avoids the extra forward call per chunk.
- The worst gap is dominated by the join step, which chunking doesn't fix. It is logged, not patched.
- Chunking applies only in the batch loop. The single-request `generate()` path ignores the flag.
- The benchmark is one 433-token prompt on one machine, and the 64 vs 128 comparison is within the noise of the join spike.
:::

## Benchmarking and measurement {#bench}

Every phase is evaluated the same way, so numbers are comparable from one phase to the next. A discarded warmup run absorbs MPS's first-call overhead, then 3 to 5 timed runs are averaged. The device is synchronized before every timestamp, because MPS work is asynchronous and an unsynchronized `perf_counter()` would only time the kernel launch. Each phase also keeps a control run: the previous behavior, selected by flipping a flag, on the same requests.

```py src/nanoserve/config.py
@dataclass
class AggregateMetrics:
    ttft_mean: float
    tpot_mean: float
    throughput_tps: float
    peak_memory_mb: float
    cache_hit_rate: float | None = None
    ttft_p99: float | None = None
    tpot_p99: float | None = None
    batch_occupancy_mean: float | None = None
    rejected_queue_full: int | None = None
    rejected_timeout: int | None = None
    tpot_max: float | None = None
    ...
```

:::table stack 26% 30%
| Experiment | Question it answers | Control |
| --- | --- | --- |
| `benchmark.py` | How fast is one request with each cache type? | Previous step's flag |
| `test_fragmentation.py` | How much memory is wasted, and can a fragmented pool still serve a large request? | Fixed 1,024-token reservation, simulated contiguous arena |
| `continuous_batch_benchmark.py` | Does batching raise system throughput? | Same requests run sequentially |
| `scheduler_benchmark.py` | Do backpressure and timeouts fire, and does the system recover? | Scheduler off, same 22 requests |
| `radix_benchmark.py`, `radix_ramp_benchmark.py` | Does the shared prefix save compute and memory? | Radix off, same requests |
| `chunked_prefill_benchmark.py` | Does chunking remove the freeze, and what does it cost? | Chunking off, same requests |
:::

Each run writes a JSON file to `results/` with its config, aggregate metrics, per-request records, and (for batched runs) a per-step trace of how many requests were active and how long each step took. The `plot_*.py` scripts read those files, so every chart on this page and in the blog series can be regenerated from the saved data.

:::qa Why is `utils.sync()` called around every measurement?
GPU work on MPS is queued asynchronously. Without a synchronize, the timer would stop when the operation was launched and not when it finished. `utils.py` wraps `torch.accelerator.synchronize()` so the same code works on CUDA, MPS and CPU.
:::

:::qa How reproducible are the numbers?
Run-to-run variation is real on this hardware. Scheduler timeouts depend on step times, so identical runs can admit and reject slightly different sets of requests, and a couple of results (the scheduler's idle-GPU spike, one radix hit that wobbled) are logged as anomalies. What reproduced reliably were the shapes: the TTFT gradient across cache-hit conditions, the roughly 52% TTFT drop on shared-prefix hits, and the decode freeze disappearing under chunking.
:::

## Trade-offs and gaps, in one place {#limits}

These are the trade-offs of building each technique in isolation on a small machine, grouped by area. Each one gets addressed when a real need shows up. The [scheduler](/blogs/posts/nanoserve-scheduler), [prefix caching](/blogs/posts/nanoserve-prefix-caching) and [chunked prefill](/blogs/posts/nanoserve-chunked-prefill) posts each end with an "honest anomaly" section covering the surprising results.

- **Memory:** a fixed 256-block pool; admission is limited by `MAX_BATCH_SIZE`, not free blocks, so there is no preemption or swapping when the pool runs out.
- **Attention:** a gather-based substitute for a fused kernel, with padding to the longest sequence in the batch.
- **Prefix cache:** block-aligned only, prompts only, no eviction, and `ref_count` is tracked but not yet used.
- **Scheduler:** FCFS only, a count-based queue bound, and timeouts checked once per loop iteration.
- **Chunked prefill:** one prompt at a time, alternating instead of fused, and a worst gap dominated by the batch-join step.
- **Engine:** greedy decoding only, one model, direct calls from benchmark scripts with no HTTP serving layer. The `use_quantization` and `use_speculative_decoding` flags exist in `EngineConfig` but aren't implemented.
- **Measurement:** one M1 with 8 GB, short runs and synthetic workloads (a nine-request ramp, a 22-request burst, a 433-token prompt), with no confidence intervals and no head-to-head against vLLM or SGLang yet.

## What's next? {#next}

The must-have build order is finished, and the [chunked prefill post](/blogs/posts/nanoserve-chunked-prefill) lays out the rest.

- A full-stack evaluation with every flag on at once
- A stage-by-stage comparison isolating what each technique bought over the one before it
- An honest, measured account of where NanoServe sits next to vLLM and SGLang, and why the gap exists
- LRU eviction for the radix cache, using the `ref_count` and `last_access` it already tracks
- Memory-aware admission, so the scheduler looks at free blocks and not only batch slots
- Speculative decoding and quantization, attempted opportunistically within Apple Silicon's constraints

---

:::note
Source code: [github.com/arponkapuria/NanoServe](https://github.com/arponkapuria/NanoServe)
:::

:::note
About the painting: ["The Creation of Adam" (c. 1508 to 1512) by Michelangelo (public domain), from Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg)
:::

:::note
N.B: This page was drafted with AI assistance and then reviewed and verified by me.
:::