---
title: "Optimizing an NVFP4 Group GEMM Kernel on Blackwell"
date: 2026-02-16
permalink: /posts/nvfp4_group_gemm/
tags:
  - CUDA
  - blog
---

Happy Chinese New Year! Besides celebrating CNY this weekend, I also spent some time working on NVFP4 block-scaled group GEMM kernel optimizations on NVIDIA B200. The code is written in [CuTeDSL](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass/cute), CUTLASS's Python DSL for Blackwell kernels.

Here is my optimization journey!

## Background: What is NVFP4 Block-Scaled Group GEMM?

NVFP4 (FP4 E2M1) is a 4-bit floating point format used in quantized inference. Each element is only half a byte, so the dynamic range is tiny. To compensate, **block scaling** groups every 16 elements and multiplies them by a shared FP8 scale factor. The GEMM becomes:

$$C = (A \cdot SFA) \times (B \cdot SFB)$$

where $SFA$ and $SFB$ are per-block scale factor tensors in FP8 (E4M3FN).

**Group GEMM** means we batch multiple independent GEMM problems (each with potentially different M, N, K) into a single kernel launch. Each CTA (thread block) handles one 128x128 output tile from one problem in the group.

On Blackwell (SM100), this uses:
- **TMA (Tensor Memory Accelerator)** for async global-to-shared memory loads
- **UMMA (Unified Matrix Multiply-Accumulate)** via `tcgen05` tensor core instructions
- **TMEM (Tensor Memory)** for accumulator storage
- **Tensormap descriptors** that tell TMA where and how to load data

## The Reference Kernel

The [reference kernel](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/grouped_blockscaled_gemm.py) from CUTLASS is a fully-featured persistent kernel. For this competition, I started from a simplified non-persistent version that launches one CTA per output tile.

The baseline design is **single-warp sequential**: warp 0 does everything — TMA loads, pipeline waits, S2T copies, MMA — all in one loop body:

```python
# Baseline: single-warp design — warp 0 does everything sequentially
if warp_idx == 0:
    acc_empty = acc_producer.acquire_and_advance()
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

    for k_tile in range(k_tile_cnt):
        # 1. Wait for empty buffer, issue TMA loads
        ab_empty = ab_producer.acquire_and_advance()
        cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, ab_empty.index)],
                  tma_bar_ptr=ab_empty.barrier, ...)
        cute.copy(tma_atom_b, ...)
        cute.copy(tma_atom_sfa, ...)
        cute.copy(tma_atom_sfb, ...)

        # 2. Wait for data to arrive, then compute
        ab_full = ab_consumer.wait_and_advance()  # blocks until TMA done
        cute.copy(tiled_copy_s2t_sfa, ...)  # S2T scale factors
        cute.copy(tiled_copy_s2t_sfb, ...)
        cute.gemm(tiled_mma, tCtAcc, tCrA[...], tCrB[...], tCtAcc)  # MMA

        ab_full.release()
    acc_empty.commit()
```

The pipeline used `make_participants()` which returns producer/consumer handle objects with `acquire_and_advance()` and `wait_and_advance()` methods:

```python
ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    num_stages=num_ab_stage,  # was 1
    ...
).make_participants()
```

With `num_ab_stage = 1`, there's only one shared memory buffer. TMA must finish loading before MMA can start, and MMA must finish before the next TMA load. **Zero overlap.**

The host code also re-created all metadata tensors (problem sizes, pointer arrays, tensormap buffers) on every invocation — pure overhead.

## Optimization 1: Host-Side Caching

**Problem:** Every call to `custom_kernel()` allocated new CUDA tensors for metadata, even when the same tensors were being reused across benchmark iterations.

**Solution:** Cache all host-side allocations keyed by tensor identity. On subsequent calls with the same tensors, skip all the `torch.tensor()` and `torch.empty()` allocations:

```python
_compiled_kernel_cache = {}
_host_cache = {}

def custom_kernel(data: input_t) -> output_t:
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    num_groups = len(problem_sizes)
    compiled_func = compile_kernel(problem_sizes)

    # Cache host-side tensors keyed by first tensor identity
    cache_key = id(abc_tensors[0][0])
    cache_hit = False
    if cache_key in _host_cache:
        cached = _host_cache[cache_key]
        if cached[0] is abc_tensors[0][0]:  # identity check, not equality
            cache_hit = True

    if not cache_hit:
        # ... build tensors only on first call ...
        _host_cache[cache_key] = (
            abc_tensors[0][0],           # sentinel for identity check
            tensor_of_problem_sizes,
            tensor_of_abc_ptrs,
            tensor_of_sfasfb_ptrs,
            tensor_of_tensormap,
            cute_ptrs,
            total_num_clusters,
        )
        cached = _host_cache[cache_key]

    # Fast path: reuse cached tensors
    cute_ptrs = cached[5]
    total_num_clusters = cached[6]
    compiled_func(cute_ptrs[0], cute_ptrs[1], cute_ptrs[2], cute_ptrs[3],
                  total_num_clusters, problem_sizes, num_groups)
```

The compiled kernel is also cached by group count, so JIT compilation only happens once per unique number of groups.

## Optimization 2: Warp Specialization

**Problem:** In the single-warp design, TMA and MMA execute sequentially within the same warp. The tensor cores sit idle while waiting for memory, and TMA sits idle while computing.

**Solution:** Split TMA and MMA into separate warps so they can overlap:
- **Warp 0** — TMA Producer: issues async TMA loads
- **Warp 1** — MMA Consumer: waits for data, computes GEMM

This requires switching from the `make_participants()` API (which assumes uniform control flow) to explicit `PipelineState` objects (which support divergent warp paths):

```python
# Before: make_participants() — assumes all threads share control flow
ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(...).make_participants()
acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(...).make_participants()

# After: raw pipeline objects — each warp creates its own PipelineState
ab_pipeline = pipeline.PipelineTmaUmma.create(
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    num_stages=num_ab_stage,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
    tx_count=num_tma_load_bytes,
)
acc_pipeline = pipeline.PipelineUmmaAsync.create(
    barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    num_stages=num_acc_stage,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
)
```

The main loop splits into two separate `if warp_idx` blocks:

```python
# Warp 0 — TMA Producer: loads data into shared memory
if warp_idx == 0:
    ab_producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_ab_stage
    )
    for k_tile in range(k_tile_cnt):
        ab_pipeline.producer_acquire(ab_producer_state)
        cute.copy(tma_atom_a, tAgA[(None, k_tile)],
                  tAsA[(None, ab_producer_state.index)],
                  tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                  tma_desc_ptr=tensormap_manager.get_tensormap_ptr(...))
        cute.copy(tma_atom_b, ...)
        cute.copy(tma_atom_sfa, ...)
        cute.copy(tma_atom_sfb, ...)
        ab_producer_state.advance()
    ab_pipeline.producer_tail(ab_producer_state)

# Warp 1 — MMA Consumer: computes on data as it arrives
if warp_idx == 1:
    ab_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, num_ab_stage
    )
    acc_producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_acc_stage
    )
    acc_pipeline.producer_acquire(acc_producer_state)
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

    for k_tile in range(k_tile_cnt):
        ab_pipeline.consumer_wait(ab_consumer_state)
        # S2T copy scale factors from smem to tmem
        cute.copy(tiled_copy_s2t_sfa,
                  tCsSFA_compact_s2t[(None, None, None, None, ab_consumer_state.index)],
                  tCtSFA_compact_s2t)
        cute.copy(tiled_copy_s2t_sfb, ...)
        # MMA
        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kblock_idx)].iterator)
            tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kblock_idx)].iterator)
            cute.gemm(tiled_mma, tCtAcc, tCrA[...], tCrB[...], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        ab_pipeline.consumer_release(ab_consumer_state)
        ab_consumer_state.advance()
    acc_pipeline.producer_commit(acc_producer_state)
```

The epilogue also switches from `make_participants` handles to explicit `PipelineState`:

```python
# Before
acc_full = acc_consumer.wait_and_advance()
...
acc_full.release()

# After
acc_consumer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Consumer, num_acc_stage
)
acc_pipeline.consumer_wait(acc_consumer_state)
...
acc_pipeline.consumer_release(acc_consumer_state)
```

## Optimization 3: Multi-Stage Pipelining (Double Buffering)

**Problem:** Warp specialization with `num_ab_stage = 1` is *slower* than the baseline. There's only one shared memory buffer, so the producer must wait for the consumer to release it before loading the next tile. The warps just take turns — no overlap.

**Solution:** Increase `num_ab_stage` from 1 to 2 (and later 4). With multiple buffer slots, the TMA producer can fill stage N+1 while the MMA consumer computes on stage N:

```python
# Before
num_ab_stage = 1  # single buffer — no overlap possible

# After
num_ab_stage = 4  # quad buffering — TMA can stay 3 stages ahead of MMA
```

This is the change that makes warp specialization actually pay off. The `PipelineState` tracks which stage each warp is working on, and the `producer_acquire` / `consumer_wait` calls handle the synchronization automatically.

## What's Next

The path forward:

- **Persistent kernel**: One CTA processes multiple tiles across groups, amortizing all setup costs. This is what the [CUTLASS reference](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/grouped_blockscaled_gemm.py) does.
- ...
