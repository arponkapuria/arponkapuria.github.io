---
title: Paged KV Cache on Apple Silicon: Fixing Memory Fragmentation Without CUDA
description: Part 2 of NanoServe, a tiny LLM inference engine build series. This article covers how paged KV cache eliminates GPU memory fragmentation, the CUDA-based design behind it, and the block-based cache built for Apple Silicon instead.

date: September 05, 2026
modified: September 05, 2026

author: Arpon Kapuria
category: Dev Journal
tags: LLM Inference, AI Systems
---

> Part 2 of the NanoServe build series. Part 1 covered naive decode and KV cache; this part assumes you've read that, or at least know what a KV cache is and why it speeds up decoding.
>
> **Part 1:** [From Naive Decode to KV Cache: Why LLM Serving Gets Faster With One Idea](/blogs/posts/nanoserve-paged-kv-cache)
 
Part 1 fixed a *compute* problem: stop recomputing Keys and Values you already know. This part fixes a different problem that KV cache never touches — *how that cache is stored in memory*. Because this project runs on an Apple Silicon GPU instead of an NVIDIA one, the version we can actually build looks different from the industry-standard one. This article covers what the standard version does, why we can't copy it directly, what we built instead, and how we tested it.

## The Memory Problem
 
A KV cache is a running record of every token's Key and Value vectors, computed once and reused for every later step. That's what made decoding faster in Part 1 - roughly 4x, in our own numbers.
 
But look at how that record is stored. In Part 1, and in HuggingFace's default `DynamicCache`, each request gets one growing, contiguous tensor. It starts small and grows by one slot per token. That's fine for a single request running alone. It breaks down the moment a server has to handle many requests at once — which is exactly where this project is headed next, with continuous batching.
 
Two separate problems show up here:
 
**Internal fragmentation** — memory reserved for a sequence but never used. Since you don't know how long a sequence will run, a system typically reserves space for some worst case upfront, say 1024 tokens. If the sequence only generates 80 tokens, the other 944 reserved slots sit unused for its whole lifetime.
 
**External fragmentation** — memory that's genuinely free, but scattered into pieces too small to use. Picture three sequences sitting back-to-back in memory. The middle one finishes and frees its space. Now there's a free gap in the middle, but if a new, larger sequence needs more contiguous space than that gap holds, it can't use it — even though enough total free memory exists elsewhere.

![The Fragmentation Problem|600](/images/blogs/nanoserve/fragmentation-problem.png)
 
This isn't hypothetical. The team behind **vllm** measured it on real serving systems and found they were using only 20–38% of their reserved KV cache memory for actual data — meaning 60–80% of it sat wasted. On a resource-constrained machine, that can not be ignored.

## PagedAttention
 
The fix **vllm** introduced — called **PagedAttention**, borrows an idea from operating systems that's decades old: virtual memory management.
 
An OS gives every running program the illusion of owning a private, contiguous block of memory. In reality, that memory is scattered across arbitrary physical locations. A **page table** maps "the program's virtual page N" to "wherever it actually landed physically." The program never notices.
 
PagedAttention applies the same idea to tokens instead of memory pages:
 
- **Block**: a fixed-size chunk that holds the K/V vectors for a fixed number of tokens (say, 16 tokens' worth), across every attention head and every layer.
- **Physical block pool**: one big pre-allocated set of such blocks, shared across *every* sequence the whole engine is serving — not one pool per sequence.
- **Block table**: a small, per-sequence lookup list mapping "this sequence's logical block 0, 1, 2..." to "physical block #47, #12, #203..." wherever those actually happen to sit in the pool.
- **Free list**: a list of which physical blocks are currently unused. Allocating is just popping an index off this list — instant, no searching for a contiguous run of the right size.

![Logical Blocks vs Physical Blocks|600](/images/blogs/nanoserve/logical-blocks-vs-physical-blocks.png)
 
Because every block is the same size, allocation stops being "find a contiguous region of the right size" — which fragments badly, as shown above — and becomes "grab any block, from anywhere." External fragmentation can't happen. You still lose a little memory internally — up to `block_size − 1` tokens per sequence, whatever doesn't fill the last block — but that waste is small and bounded, not the unpredictable waste of upfront reservation.

## The CUDA Implementation
 
It's worth being precise about what vLLM's actual production implementation does, it has two separate pieces, and our version builds only one of them.
 
**(1) The block bookkeeping** described above — pool, block table, free list. It's plain data-structure logic, nothing GPU-specific. You could write it in any language.
 
**(2) A custom hand-written CUDA kernel that computes attention directly against the scattered blocks** — Say a sequence's block table points to blocks 47, 12, and 203. The simple approach would be to first copy those three blocks into one contiguous buffer, then run ordinary attention math on it — two steps. vLLM's kernel instead reads straight from blocks 47, 12, and 203 wherever they live, computes the attention scores, runs softmax, and blends the values, all inside one fused GPU kernel launch. No intermediate copy is ever created.
 
![Fused CUDA Kernel|600](/images/blogs/nanoserve/fused-cuda-kernel-paged-kv.png)
 
That fusion is the actual source of vLLM's speed advantage over a plain "gather, then compute" approach — skipping the intermediate copy avoids an entire extra pass of memory movement.

## No Fused Kernel on MPS
 
That CUDA kernel isn't portable code — it can't be adapted, only replaced. CUDA kernels are compiled by NVIDIA's toolchain for NVIDIA's specific GPU architecture: thread blocks, warps, shared memory banks. None of that exists on an Apple GPU. It's the same reason x86 machine code can't run on an ARM chip — different instruction set entirely.
 
Apple GPUs run **Metal Shading Language**, a completely separate language and execution model. PyTorch's MPS backend gives us a fixed set of pre-written operations — matrix multiply, softmax, `index_select`, and so on — already implemented in Metal. It doesn't let us write our own fused kernel from Python the way CUDA extensions do. Getting a fused paged-attention kernel on Apple Silicon means hand-writing it in Metal directly.
 
A community project called `vllm-metal` ports vLLM to Apple Silicon with real, hand-written Metal kernels, integrated with Apple's **MLX** framework (a separate tensor library from PyTorch). Their published numbers show real gains — over 80x on time-to-first-token, 3–4x on throughput, versus their own earlier non-paged baseline.
 
*So why aren't we just using that?* Look at what it actually took to build: a dedicated open-source project, multiple contributors, hand-written Metal compute shaders, integration with a separate tensor framework (MLX, not PyTorch), and — per their own public issue tracker — still-unresolved engineering problems (for instance, kernel calls not yet fully integrated into MLX's lazy computation graph, requiring explicit CPU-GPU synchronization barriers). That is a specialized, ongoing systems-programming effort. It's not that it's impossible on Apple hardware; it's that doing it *properly* is a substantially larger, different project than this one.

## Our Approach
 
Given that, the choice was between pulling in `vllm-metal`'s MLX-based kernels for one component of a learning project, or building the block management ourselves in plain PyTorch and substituting something simpler for the fused kernel.
 
We chose the Naive Two-Step approach, and it holds up for a specific reason: the memory-management benefit — no external fragmentation, bounded internal fragmentation — comes entirely from the block bookkeeping, not the kernel. The fused kernel only affects *how fast* attention runs once the blocks already exist; it doesn't change whether fragmentation gets solved. So building the bookkeeping properly and swapping in a slower substitute for the kernel still delivers what this technique is actually for.

![Naive Two Step Approach|700](/images/blogs/nanoserve/two-step-paged-kv.png)
 
**The substitute:** store KV data in real, scattered, fixed-size blocks, exactly as described above, but at attention time, explicitly **gather** the sequence's blocks into a temporary contiguous tensor using PyTorch's `index_select`, then run ordinary attention math on that tensor. This is the two-step version from the diagram earlier — deliberately slower than a fused kernel, with that cost measured honestly rather than hidden.
 
This also fits HuggingFace's `transformers` library with almost no extra code. Every attention layer calls a method named `update()` on whatever cache object it's given, and expects a Key/Value tensor back — it has no idea what happens inside that call. Writing our own class with that same `update()` method, doing the block write and gather internally, lets it plug straight into the existing, unmodified model. The model never needs to know paging is happening underneath it.

## Implementation: The Pool and The Cache
 
Two pieces, matching the two roles above.
 
**`PagedKVPool`** is the engine-owned physical storage, created once at startup and shared across every request.
 
Two single large tensors, `k_pool` and `v_pool` , the entire memory budget, reserved once and never resized. `free_blocks` is a plain list used as a stack — allocating pops indices off it, releasing pushes them back. Both are O(1), with no searching involved, which is exactly why external fragmentation can't happen here.
 
```python
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
 
    def release(self, block_ids: list[int]) -> None:
        self.free_blocks.extend(block_ids)
```
 
**`PagedKVCache`** is a lightweight per-request object that borrows blocks from the pool, built to match the `Cache` interface HuggingFace's model already expects.
 
```python
class PagedKVCache(Cache):
    def __init__(self, pool: PagedKVPool):
        self.pool = pool
        self.block_table: list[int] = []
        self.num_tokens = 0
        self._step_new_tokens = 0
 
    def begin_step(self, num_new_tokens: int) -> None:
        self._step_new_tokens = num_new_tokens
        block_size = self.pool.block_size
        start, end = self.num_tokens, self.num_tokens + num_new_tokens
        needed_blocks = -(-end // block_size)  # ceiling division
        if needed_blocks > len(self.block_table):
            self.block_table += self.pool.allocate(needed_blocks - len(self.block_table))
 
        table = torch.tensor(self.block_table, device=self.pool.k_pool.device)
        positions = torch.arange(start, end, device=self.pool.k_pool.device)
        self._write_phys = table[positions // block_size]
        self._write_slot = positions % block_size
        self._read_phys = table[:needed_blocks]
        self._n_blocks = needed_blocks
        self._end = end
 
    def commit_step(self) -> None:
        self.num_tokens += self._step_new_tokens
        self._step_new_tokens = 0
 
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        block_size = self.pool.block_size
        self.pool.k_pool[layer_idx, self._write_phys, :, self._write_slot, :] = key_states[0].permute(1, 0, 2)
        self.pool.v_pool[layer_idx, self._write_phys, :, self._write_slot, :] = value_states[0].permute(1, 0, 2)
 
        k = self.pool.k_pool[layer_idx].index_select(0, self._read_phys)
        v = self.pool.v_pool[layer_idx].index_select(0, self._read_phys)
        num_kv_heads, head_dim = k.shape[1], k.shape[3]
        k = k.permute(1, 0, 2, 3).reshape(1, num_kv_heads, self._n_blocks * block_size, head_dim)[:, :, :self._end, :]
        v = v.permute(1, 0, 2, 3).reshape(1, num_kv_heads, self._n_blocks * block_size, head_dim)[:, :, :self._end, :]
        return k, v
 
    def free(self) -> None:
        self.pool.release(self.block_table)
        self.block_table = []
        self.num_tokens = 0
```
 
`begin_step` runs once per forward pass, not once per layer — an important distinction covered in the next section. It works out how many blocks this step's tokens need using ceiling division, allocates new blocks if needed, and precomputes the read/write indices once, so every layer can reuse them.
 
`update()` is what HuggingFace's model calls once per layer. Using the indices `begin_step` already computed, it does a vectorized write — every new token's Key/Value goes into its assigned block and slot in one indexed assignment, not a loop — and a gather, pulling the sequence's blocks out of the pool with `index_select` and reshaping them into the standard shape attention expects. That gather is the actual stand-in for vLLM's fused kernel — the one place in this implementation doing something the real kernel avoids.


## Estimating the Pool Size

To size the pool, we need to know how much memory one token costs. Qwen3-1.7B has 28 layers, 8 Key/Value heads per layer, and a 128-number head dimension, stored in 16-bit floats.

![KV Cache Memory Calculation|700](/images/blogs/nanoserve/kv-cache-memory-calculation.png)

> With this configuration, the KV cache pool reserves **~458 MB**, holding up to **4096 tokens** across **256 blocks** of **16 tokens** each.

**Why 16 tokens per block?** This is the value vLLM itself defaults to, and it's a genuine trade-off, not an arbitrary choice: Smaller blocks (8) waste less per sequence in the worst case, but mean more blocks to track and more, smaller gather operations. Larger blocks (32) mean fewer, bigger gathers, but coarser reuse — a 3-token sequence still occupies a full 32-slot block. 16 sits in the middle, and keeps our numbers comparable to published results.

**Why 256 blocks?** This was a memory budget decision for this specific machine: 8GB unified memory, shared with the OS, with the model itself already using roughly 3.4GB. It leaves headroom while staying small enough that block reuse is actually exercised. Doubling to 512 blocks would cost about 920MB instead of 458MB — a wider margin, worth revisiting once real concurrency numbers exist.

## Inference Flow

HuggingFace's model calls `update()` once per layer — 28 times per forward pass. But block allocation has to happen once per forward pass, not 28 times, or it would double-allocate.
 
So the engine calls `begin_step(num_new_tokens)` once, before the model runs, handling allocation and index computation up front. The model then runs its normal forward pass, calling `update()` 28 times internally, each call reusing the same precomputed indices and touching only its own layer's slice of the pool. After the forward pass, the engine calls `commit_step()` once, advancing the token count so the next step's `begin_step` builds on the right number.
 
![Inference: The Per-Step Flow|700](/images/blogs/nanoserve/per-step-flow-paged-inference.png)
 
The same code handles both **prefill** — the first call, `begin_step(prompt_length)`, processing the whole prompt at once — and **decode** — every call after, `begin_step(1)`, one token at a time. Only the number changes.
 
When a sequence finishes, `cache.free()` returns every block it held straight back to the pool's free list, instantly available to the next request. No cleanup pass is needed, since every block is interchangeable by design.

## Findings
 
Paged KV cache exists to eliminate memory waste — both kinds of fragmentation — by replacing per-sequence contiguous allocation with fixed-size, reusable blocks, so more sequences can eventually share the same memory budget.
 
That goal doesn't include making a single sequence faster. It can't: paging does the same attention math as a plain cache, plus an extra gather step, so at best it matches plain KV cache's speed and never beats it while only one sequence is running. The real payoff — more sequences fitting in the same memory, better throughput under load — only shows up once multiple sequences actually compete for the pool at once, which is continuous batching's job, not this step's. What follows checks that the infrastructure is correct and the memory properties are real, not that it's faster yet.

### TTFT, TPOT, Throughput & Memory 
 
Same method as Part 1: a discarded warmup run, three timed runs averaged, comparing paged KV cache against Part 1's plain KV cache on the same prompt.
 
![KV Cache vs Paged KV Cache|700](/images/blogs/nanoserve/kv-cache-vs-paged-kv-cache-plot.png)
 
TTFT and TPOT land within noise of plain KV cache — expected, since the underlying attention math is identical and the added gather step turned out to be cheap once properly vectorized. Peak memory is about 446MB higher, and that number isn't mysterious: it's almost exactly the pool's fixed upfront allocation from the sizing math above (~458MB theoretical). That cost is paid once at startup regardless of how much of it any single request actually uses — a cost that only turns into a good trade-off once multiple sequences share the same pre-paid pool, which a single-request test can't show.
 
### Internal Fragmentation
 
To measure this, we ran real generations at three different lengths through the paged engine and, before freeing each one, computed how many token-slots were reserved but unused. For a naive-side comparison, we ran the same lengths through a real allocation reserving a fixed 1024 tokens upfront regardless of actual length, and measured that the same way.
 
![Internal Fragmentation - Waste Comparison|700](/images/blogs/nanoserve/internal-fragmentation-waste-plot.png)
 
Paged waste stays low and bounded — 1.2% to 4.7% — regardless of sequence length, consistent with the `block_size − 1` bound. Naive waste drops from 92.3% to 45.2% as sequences get longer, but that's arithmetic, not naive "improving": its waste is `1 − actual/1024`, so it shrinks mechanically as actual length approaches the fixed reservation. Even at naive's best case here, it still wastes roughly 17x more than paged's worst case. A real system serving mostly short, unpredictable-length requests would sit much closer to the high-waste end of this range, not the favorable long-sequence case.
 
This is also directionally consistent with vLLM's own published numbers — 60–80% waste for naive contiguous systems, under 4% for PagedAttention, measured on their own production workloads. Our setup differs (different model, hardware, and a real-but-isolated naive allocator rather than a full competing engine), so treat this as corroborating evidence, not a reproduction of their exact experiment.
 
### External Fragmentation
 
This test needs a different method, because external fragmentation is a property of an allocator's bookkeeping, not of physical memory. We drove our real `PagedKVPool` through a deliberately adversarial pattern: allocate A (4 blocks), allocate B (3 blocks), free A, then try to allocate C (6 blocks). Seven blocks are free in total, more than enough — but under a naive contiguous allocator, A's freed space and any other free space aren't necessarily adjacent, so no single free span of 6 may exist. We ran the identical pattern through a small simulated contiguous allocator for comparison, rather than real GPU tensors — real allocations here would only add noise from PyTorch's own underlying memory manager, not test the strategy itself.
 
| Allocator | Result |
|---|---|
| Paged block pool | Success |
| Naive contiguous arena | Failed — largest free span was 4, needed 6 |
 
The paged pool succeeds because it never needs contiguity in the first place; the naive arena fails exactly as predicted, despite having more free memory in total than the request needed.

## What's Next
 
Everything measured here is groundwork, and its payoff is intentionally deferred. Block-addressable storage exists to make continuous batching possible: serving many sequences of different, changing lengths, arriving and finishing unpredictably, sharing one fixed memory budget without heavy padding waste. A single contiguous buffer per sequence makes that either impossible or badly wasteful — pad everyone to the batch's longest sequence, or manage a mess of independently-growing allocations that fragments exactly as shown earlier. Block-based storage avoids both problems by construction.
 
The real performance story — throughput under concurrent load, actual utilization of the memory set aside here — gets measured next, not in this step. This step's job was proving the foundation is correct and provably better on the specific properties it targets, ahead of the step that will actually make use of it.
 
---
 
> Code: [https://github.com/arponkapuria/NanoServe](https://github.com/arponkapuria/NanoServe)