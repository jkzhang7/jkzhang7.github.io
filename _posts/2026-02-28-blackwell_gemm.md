---
title: "Understanding GEMM on Blackwell with CuTeDSL"
date: 2026-02-28
permalink: /posts/blackwell_gemm/
tags:
  - CUDA
  - blog
---

In this post I want to walk through how a high-performance GEMM is structured on Blackwell, using the official CUTLASS [dense GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_alpha_beta_persistent.py) as a case study. This kernel is written in [CuTeDSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) — NVIDIA's Python DSL for authoring CUTLASS kernels — and it packs in nearly every Blackwell-specific optimization available.

The operation being computed is:

$$C = \alpha \cdot A \cdot B^T + \beta \cdot C$$

where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{N \times K}$, $C \in \mathbb{R}^{M \times N}$. The kernel supports flexible input and output data types.

---

## 1. The Souls of CuTeDSL

Before diving in, a quick note on CuTeDSL. Its core idea is that memory layouts are first-class values. Rather than manually computing flat indices, you describe data as a `cute.Tensor` — a pointer paired with a layout (shape + strides) — and operations like `cute.copy`, `cute.gemm`, and `cute.partition` work generically over any layout. This makes developer life much easier when expressing complex tiling, swizzling, and partitioning schemes.

For a deeper introduction I recommend [this post by Veitner](https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/) (and his other posts about CuTeDSL!) and the [official docs](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html). I also find the [examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) are quite helpful to understand.

---

## 2. Kernel Structure in the Code

The kernel is implemented as a single class `SM100PersistentDenseGemmAlphaBetaKernel`. The key methods to understand are:

- **`__init__`** — stores hyperparameters (MMA tile shape, cluster shape, dtypes, stage counts). Does no GPU work.
- **`__call__(a, b, c, alpha, beta, max_active_clusters, stream, epilogue_op)`** — the host-side entry point. It sets up TMA descriptors, computes layouts, allocates shared memory descriptors, and calls `cute.compile` + launches the kernel. `epilogue_op` is an optional elementwise lambda applied to the output (e.g. ReLU), defaulting to identity.
- **`kernel`** — the actual `@cute.kernel` GPU function. This is where all the warp-specialized logic lives.
- **`can_implement(a, b, c)`** — checks alignment and shape constraints before committing to this kernel.

The host-side `__call__` does the heavy lifting of layout arithmetic: it calls helpers like `make_smem_layout_a/b`, `make_tiled_tma_atom`, and `compute_epilogue_tile_shape` to produce all the layout objects that get passed into the GPU kernel as compile-time constants. This is a key CuTeDSL pattern — move as much as possible to compile time so the GPU kernel only sees simple, specialized code.

Inside `kernel`, control flow splits immediately by warp ID:

```python
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

if warp_idx == self.tma_warp_id:   # warp 5 — TMA producer
    ...
if warp_idx == self.mma_warp_id:   # warp 4 — MMA
    ...
if warp_idx < self.mma_warp_id:    # warps 0–3 + warp 6 — epilogue
    ...
```

Each branch runs an independent persistent loop over output tiles, synchronized through shared-memory barriers.

---

## 3. Optimizations

### 3.1 TMEM: Accumulator off the Register File

Blackwell adds a new on-chip memory tier called **TMEM** (tensor memory), sitting between SMEM and registers. It is used exclusively to store MMA accumulators.

This matters because the 256×128 f32 accumulator tile is 128KB — impossible to keep in registers. With TMEM, the accumulator lives entirely off the register file during the mainloop, and the epilogue warps drain it out with `tcgen05.ld` after MMA finishes. The MMA warp retrieves a TMEM pointer at startup and constructs a `cute.Tensor` over it:

```python
tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
```

The result: the mainloop warps carry virtually no register pressure from accumulators, freeing registers for pipeline state and address computations.

### 3.2 tcgen05: 2-CTA MMA Instructions

Blackwell's MMA instruction (`tcgen05`) can operate across **two CTAs simultaneously** as a single logical instruction. With `use_2cta_instrs=True`, the 256×128 output tile is computed jointly by both CTAs in the cluster — each CTA owns a 128×128 slice but executes the same `tcgen05.mma` with the hardware coordinating the data routing between them:

```python
for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
    cute.gemm(
        tiled_mma,
        tCtAcc,
        tCrA[(None, None, k_block_idx, ab_consumer_state.index)],
        tCrB[(None, None, k_block_idx, ab_consumer_state.index)],
        tCtAcc,
    )
    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
```

The `ACCUMULATE` field starts as `False` for the first K-block of each tile (overwrite mode) and is set to `True` after the first block (accumulate mode), avoiding an explicit zero-initialization of the TMEM accumulator.

### 3.3 2-CTA Cluster and TMA Multicast

The kernel is launched with a **cluster shape of `(2, 1)`**: two CTAs grouped into a cluster, sitting on the same GPC and able to access each other's shared memory.

```
Cluster (2×1)
┌───────────────┬───────────────┐
│    CTA 0      │    CTA 1      │
│  rows 0..127  │ rows 128..255 │   ← split along M
└───────────────┴───────────────┘
         Both need the same B tile (K×128)
```

Since both CTAs need the same B tile but different A tiles, the kernel uses **TMA multicast**: a single TMA transaction loads B from L2 and delivers it to both CTAs' SMEM simultaneously:

```python
cute.copy(
    tma_atom_b,
    tBgB_slice[(None, ab_producer_state.count)],  # global B at current k-tile
    tBsB[(None, ab_producer_state.index)],          # SMEM stage slot
    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
    mcast_mask=b_full_mcast_mask,   # broadcast to both CTAs in cluster
)
```

The TMA descriptor for B uses the `sm_100_2sm` variant, which tells the hardware to deliver the data to two CTAs in one shot. The result: B's L2 bandwidth cost is halved compared to running two independent CTAs.

### 3.4 Warp Specialization

The kernel launches with **224 threads per CTA** — exactly 7 warps — each with a fixed, non-overlapping role:

| Warp | Role |
|---|---|
| 0, 1, 2, 3 | **Epilogue** — drain accumulators, scale, convert, store |
| 4 | **MMA** — issue `tcgen05` matrix multiply instructions |
| 5 | **TMA producer** — load A and B tiles from global memory into SMEM |
| 6 | **Epilogue C loader** — load C tiles from global memory into SMEM for beta scaling |

The three roles overlap in time: while warp 4 is computing the current tile's MMA, warp 5 is already loading the next tile's A and B into a different SMEM buffer, and warps 0–3 are draining the previous tile's accumulator to global memory. This is the core idea behind warp specialization: instead of all warps doing the same work sequentially, different warps pipeline different stages of the computation.

### 3.5 Multi-Stage A/B Pipeline

With warp 5 (loader) and warp 4 (MMA) running concurrently, the kernel hides memory latency by **staging multiple A/B tile buffers in shared memory**. The number of stages is computed automatically from available SMEM:

```python
(self.num_acc_stage,
 self.num_ab_stage,   # typically 2–4
 self.num_c_stage,
 self.num_d_stage) = self._compute_stages(...)
```

The pipeline uses `mbarrier`-based producer/consumer synchronization:

- **Producer (warp 5)** calls `ab_pipeline.producer_acquire(state)` to claim an empty slot, issues the non-blocking TMA load into it, then advances to the next slot.
- **Consumer (warp 4)** calls `ab_pipeline.consumer_wait(state)` which spins until the TMA completes, computes MMA, then calls `consumer_release` to free the slot.

Because there are multiple buffer slots, warp 5 can be loading tile `k+1` while warp 4 is computing tile `k` — pure latency hiding. The producer also uses `producer_try_acquire` (a non-blocking peek) one step ahead to avoid stalling on the acquire in the common case.

### 3.6 Epilogue: TMEM → Registers → SMEM → GMEM

After the mainloop, the 256×128 f32 accumulator in TMEM must be drained, scaled, type-converted, and written to global memory. The epilogue warps (0–3) do this in subtiles to keep register pressure low:

```python
for subtile_idx in cutlass.range(subtile_cnt):
    # 1. TMEM → registers (drain accumulator)
    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

    # 2. Wait for warp 6 to finish loading C into SMEM, then copy SMEM → registers
    c_pipeline.consumer_wait(c_pipeline_consumer_state)
    cute.copy(tiled_copy_s2r,
              tSR_sC[(None, None, None, c_pipeline_consumer_state.index)],
              tSR_rC)
    c_pipeline.consumer_release(c_pipeline_consumer_state)

    # 3. Scale, apply epilogue op, convert epi_dtype → c_dtype
    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
    c_vec   = tiled_copy_r2s.retile(tSR_rC).load()
    d_vec = epilogue_op(
        alpha.to(self.epi_dtype) * acc_vec.to(self.epi_dtype)
        + beta.to(self.epi_dtype) * c_vec.to(self.epi_dtype)
    ).to(self.c_dtype)

    # 4. registers → SMEM → GMEM (via TMA store)
    tRS_rD.store(d_vec)
    cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[(None, None, None, d_buffer)])
```

None of these steps have a hardware accelerator like `tcgen05.mma` — they are purely scalar/memory-bound, parallelized across threads. With 256×128 = 32,768 elements, 4 warps (128 threads) are needed to maintain throughput. Warp 6 runs ahead as a dedicated C tile loader to pre-fill the SMEM buffer with the beta·C term, keeping the epilogue warps from stalling on C loads.

### 3.7 Persistent Tile Scheduling

Rather than launching one thread block per output tile, the kernel is **persistent**: it launches a fixed grid sized to the number of active SMs and has each block loop over multiple tiles.

```python
tile_sched = utils.StaticPersistentTileScheduler.create(
    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
)
work_tile = tile_sched.initial_work_tile_info()

while work_tile.is_valid_tile:
    # process this tile ...
    tile_sched.advance_to_next_work()
    work_tile = tile_sched.get_current_work()
```

Tile-index → (M, N) coordinate remapping uses `cute.fast_divmod` (a precomputed divisor trick) to avoid expensive integer divisions in the hot loop.

The benefits: blocks stay resident on SMs across tiles, SMEM and TMEM remain allocated, barriers are reused, and the TMA descriptor prefetch at the start of the kernel (`cpasync.prefetch_descriptor`) amortizes over many tiles.

### 3.8 SMEM Swizzling for Bank Conflict Avoidance

All A, B, and C SMEM buffers use swizzled layouts (`S<2,4,3>` / `S<3,4,3>`). A swizzled layout permutes the byte address of each element such that 32 threads accessing a contiguous tile slice land on 32 distinct SMEM banks — no bank conflicts on the SMEM → register copies feeding the MMA. CuTeDSL handles this transparently; the kernel just calls `make_smem_layout_a/b` with the tile shape and dtype, and gets back the appropriate swizzled layout.

---

## 4. Summary

| Optimization | Mechanism | HW Generation |
|---|---|---|
| Accumulator in TMEM | Off-register-file storage, `tcgen05.ld` drain | Blackwell |
| 2-CTA MMA (`tcgen05`) | 256×128 joint instruction across cluster | Blackwell |
| TMA multicast | Single load → both CTAs' SMEM | Hopper (extended in Blackwell) |
| Warp specialization | Separate TMA / MMA / epilogue warps | Hopper |
| Multi-stage A/B pipeline | SMEM double-buffering + `mbarrier` | Hopper |
| Persistent scheduling | Fixed grid, loop over tiles, `fast_divmod` | Hopper |
| Bank-conflict-free SMEM | Swizzled layouts via CuTeDSL | — |

This kernel is a textbook exercise in using every level of Blackwell's memory hierarchy in concert: HBM → L2 → SMEM (via TMA multicast) → TMEM (accumulators) → registers (epilogue) → SMEM → HBM (via TMA store). What I find most impressive is that CuTeDSL lets you express all of this without manually computing a single byte offset — the layout algebra handles the indexing, and the compiler handles the rest.

---

**PS.** Writing high-performance GPU kernels is hard enough — doing it without access to the target hardware (B200, H100, etc.) makes it even harder. I've been using [Modal](https://modal.com) to prototype locally and execute remotely on Blackwell B200s. If you're in a similar situation, check out my repo [cuda_on_modal](https://github.com/jkzhang7/cuda_on_modal), which has templates for CUDA C, CuTeDSL, and debug artifact dumps (PTX, IR).
