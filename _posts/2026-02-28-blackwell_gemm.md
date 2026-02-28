---
title: "Understanding GEMM on Blackwell with CuTeDSL"
date: 2026-02-28
permalink: /posts/blackwell_gemm/
tags:
  - CUDA
  - blog
---

Writing kernels in CuteDsl is fascinating. I would like to use this post to share my understanding of GEMM on Blackwell, specifically one official [example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_alpha_beta_persistent.py) of GEMM on Blackwell.
