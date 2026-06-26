#!/usr/bin/env python3
"""
op_bench.py — Microbenchmark individual RPAm kernel ops on v6e-1.

Measures wall-clock time for each constituent operation so that
kernel_timeline.py constants can be derived from first principles rather
than back-solved from end-to-end benchmarks.

Each op is measured in isolation via JAX JIT + jax.profiler, using the
exact tensor shapes and dtypes from the kernel.

Operations measured
-------------------
  q_matmul        x_bq  [bq_sz,  D]  @  W_q  [D,  Nq*H]          float32 → bf16
  kv_matmul       x_bkv [bkv_sz, D]  @  W_kv [D,  2*Nkv*H]       float32 → bf16
  k_matmul        x_bkv [bkv_sz, D]  @  W_k  [D,  Nkv*H]         float32 → bf16
  v_matmul        x_bkv [bkv_sz, D]  @  W_v  [D,  Nkv*H]         float32 → bf16
  q_norm_rope     rms_norm + rope on  Q [bq_sz,  Nq,  H]
  k_norm_rope     rms_norm + rope on  K [bkv_sz, Nkv, H]
  softmax_store   acc / l + bitcast + strided store  [bq_sz, Nkv, Nq_per_kv, H]
  flash_attn_qk   step1: q [bq_csz*Nq_per_kv, H] @ k [bkv_csz, H]^T + online softmax
  flash_attn_pv   step2: p [bq_csz*Nq_per_kv, bkv_csz] @ v [bkv_csz, H] + acc update

Derived constants
-----------------
  T_q_projection_wallclock  = max(q_matmul, q_norm_rope)   [MXU and VPU run in parallel in Mosaic]
  T_kv_projection_wallclock = kv_matmul + k_norm_rope       [K matmul, then V matmul, then K norm — sequential]
  T_flash_attention_per_loop = flash_attn_qk + flash_attn_pv (pipelined, so roughly max, not sum)
  T_output_norm_per_bq_tile  = softmax_store

Usage
-----
  python3 scripts/kernel_eval/op_bench.py
  python3 scripts/kernel_eval/op_bench.py --bq-sz 256 --bkv-sz 1024 --bkv-csz 512
  python3 scripts/kernel_eval/op_bench.py --n-reps 200 --n-warmup 10
"""

import argparse
import gzip
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Profiler-based timing (same approach as kernel_bench.py)
# ---------------------------------------------------------------------------

def bench_jit(fn, fn_args, n_warmup=5, n_reps=100, tag="op"):
    """JIT + profile fn(*fn_args), return mean µs from device-side trace."""
    import tempfile

    # Warmup (also triggers JIT compilation)
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*fn_args))

    trace_dir = tempfile.mkdtemp(prefix=f"op_bench_{tag}_")
    try:
        with jax.profiler.trace(trace_dir):
            for _ in range(n_reps):
                out = fn(*fn_args)
            jax.block_until_ready(out)

        # Find trace file
        trace_file = None
        for suffix in ["*.trace.json.gz", "trace.json"]:
            matches = sorted(Path(trace_dir).rglob(suffix))
            if matches:
                trace_file = matches[-1]
                break
        if trace_file is None:
            raise FileNotFoundError(f"No trace found in {trace_dir}")

        opener = gzip.open if trace_file.suffix == ".gz" else open
        with opener(trace_file, "rt") as fh:
            data = json.load(fh)

        durations = []
        for e in data.get("traceEvents", []):
            dur = e.get("dur", 0)
            name = e.get("name", "")
            if dur > 0 and ("custom_call" in name.lower() or
                            "dot" in name.lower() or
                            "matmul" in name.lower() or
                            "fused" in name.lower() or
                            "fusion" in name.lower() or
                            "broadcast" in name.lower() or
                            tag.replace("_", "") in name.lower().replace("_", "")):
                durations.append(dur / 1e3)   # ns → µs

        # Fallback: collect all non-tiny device events
        if not durations:
            for e in data.get("traceEvents", []):
                dur = e.get("dur", 0)
                cat = e.get("cat", "")
                if dur > 5000 and "XLA" in cat or cat == "":   # > 5µs
                    durations.append(dur / 1e3)

        return durations

    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


def bench_loop(make_loop_fn, n_iters=1000, n_warmup=3, n_trials=5):
    """Accurate device-side timing: run n_iters iterations in a single lax.fori_loop,
    time the whole thing, divide by n_iters. This eliminates Python dispatch overhead
    (~100µs per Python-level call) and forces XLA to not hoist loop-invariant ops.

    make_loop_fn() returns a jitted (loop_fn, args) pair where loop_fn runs n_iters
    iterations internally and returns a scalar so XLA can't elide the computation.
    """
    loop_fn, args = make_loop_fn()
    # Warmup (triggers compilation)
    for _ in range(n_warmup):
        jax.block_until_ready(loop_fn(*args))
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(loop_fn(*args))
        times.append((time.perf_counter() - t0) * 1e6 / n_iters)
    return np.mean(times), np.std(times)


# ---------------------------------------------------------------------------
# Op definitions — exact shapes from the kernel
# ---------------------------------------------------------------------------
# DMA bandwidth benchmark
# ---------------------------------------------------------------------------

def bench_dma_bandwidth(sizes_kb=(256, 512, 1024, 2048, 4096),
                        n_extra=40, n_trials=10):
    """Measure HBM->VMEM async DMA bandwidth using pltpu.make_async_copy.

    Anti-caching design:
      x_buf shape = [loop_size, n_elems]  (total > VMEM so XLA cannot cache it).
      Python for loop with static row index x_buf[i, :] -> each call gets a
      different HBM offset; XLA sees distinct static slices, cannot unify them.

    Anti-hoisting / anti-CSE:
      Each iteration passes x_buf[i, :] (Python int i -> static HBM slice offset).
      XLA sees distinct custom_call inputs -> cannot CSE. No fori_loop -> no LICM.
      No counter input needed: the different HBM address per call is sufficient.

    Two-point slope:
      Run at loop_size=N_LO and N_HI=N_LO+n_extra; divide difference by n_extra.
      This cancels Python dispatch overhead and all fixed per-call overhead.
      Both N_LO and N_HI arrays are > VMEM (min_loops = VMEM/n_bytes + 1).

    Returns (bandwidth_gbs, latency_us, (slope, intercept)) or (None, None, None).
    """
    try:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
    except ImportError:
        print("  [skip] jax.experimental.pallas not available")
        return None, None, None

    dtype = jnp.bfloat16
    VMEM_BYTES = 32 * 1024 * 1024   # conservative 32 MB estimate

    def _dma_kernel(src_hbm_ref, out_ref, dst_vmem_ref, sem_ref):
        cp = pltpu.make_async_copy(src_hbm_ref, dst_vmem_ref, sem_ref)
        cp.start()
        cp.wait()
        # Read from dst_vmem_ref creates a read-after-write dependency on the DMA.
        # Mosaic's liveness analysis sees dst_vmem_ref as live -> cannot DCE the DMA.
        # No counter needed: each call already has a distinct x_buf[i] slice (different
        # HBM address), so XLA cannot CSE calls and LICM doesn't apply (no fori_loop).
        out_ref[...] = dst_vmem_ref[0:1].astype(jnp.float32)

    def _make_bench(n_elems, loop_size):
        raw_fn = pl.pallas_call(
            _dma_kernel,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
            scratch_shapes=[pltpu.VMEM((n_elems,), dtype), pltpu.SemaphoreType.DMA],
            grid=(),
        )
        # 2D buffer: total size = loop_size * n_bytes > VMEM -> cannot be cached.
        x_buf = jax.random.normal(jax.random.PRNGKey(n_elems * loop_size),
                                   (loop_size, n_elems), dtype=dtype)

        @jax.jit
        def bench(x=x_buf, _fn=raw_fn):
            acc = jnp.float32(0.0)
            for i in range(loop_size):          # Python loop = compile-time unroll
                # x[i] is a static HBM slice (i is a Python int).
                # Each row has a different HBM address -> XLA cannot CSE or cache.
                out = _fn(x[i])
                acc = acc + out[0]
            return acc

        return bench

    # ── HLO verification ──────────────────────────────────────────────────
    _n_elems_v = (sizes_kb[0] * 1024) // 2
    _n_lo_v    = max(40, VMEM_BYTES // (_n_elems_v * 2) + 1)
    _bench_v   = _make_bench(_n_elems_v, _n_lo_v)
    try:
        hlo = _bench_v.lower().as_text()
        has_while  = 'while' in hlo.lower()
        n_hbm_refs = hlo.lower().count('hbm')
        n_cc       = hlo.lower().count('custom_call') + hlo.lower().count('custom-call')
        print(f"  HLO check (size={sizes_kb[0]}KB, loop={_n_lo_v}):")
        print(f"    while loop     : {has_while}  (False = no carry promotion)")
        print(f"    hbm refs       : {n_hbm_refs}  (expect {_n_lo_v})")
        print(f"    custom_call ct : {n_cc}  (expect ~{_n_lo_v})")
        total_mb = _n_lo_v * _n_elems_v * 2 / 1e6
        print(f"    x_buf total    : {total_mb:.1f} MB  ({'> VMEM' if total_mb*1e6 > VMEM_BYTES else 'WARNING < VMEM'})")
    except Exception as e:
        print(f"  HLO check failed: {e}")

    print(f"\n  {'Size':>8}  {'N_lo':>6}  {'N_hi':>6}  {'xbuf MB':>8}  {'dma µs':>8}  {'GB/s':>8}")
    print("  " + "-" * 58)

    bytes_list, time_list = [], []
    for kb in sizes_kb:
        n_bytes  = kb * 1024
        n_elems  = n_bytes // 2
        n_lo     = max(40, VMEM_BYTES // n_bytes + 1)
        n_hi     = n_lo + n_extra
        x_mb_hi  = n_hi * n_bytes / 1e6

        bench_lo = _make_bench(n_elems, n_lo)
        bench_hi = _make_bench(n_elems, n_hi)

        for b in [bench_lo, bench_hi]:
            for _ in range(2): jax.block_until_ready(b())

        def run(fn):
            ts = [((t := time.perf_counter()), jax.block_until_ready(fn()),
                   (time.perf_counter()-t)*1e6)[2] for _ in range(n_trials)]
            return float(np.median(ts))

        t_lo = run(bench_lo)
        t_hi = run(bench_hi)
        dma_us = (t_hi - t_lo) / n_extra
        bw     = n_bytes / dma_us / 1e3 if dma_us > 0 else 0
        print(f"  {kb:7d}KB  {n_lo:6d}  {n_hi:6d}  {x_mb_hi:8.1f}  {dma_us:8.2f}  {bw:8.0f}")
        bytes_list.append(n_bytes)
        time_list.append(dma_us)

    b_arr = np.array(bytes_list, dtype=float)
    t_arr = np.array(time_list,  dtype=float)
    slope, intercept = np.polyfit(b_arr, t_arr, 1)
    bw_gbs  = 1.0 / slope / 1e3
    latency = float(intercept)

    print()
    print(f"  Fit:  t_us = {intercept:.2f} + {slope:.4e} x bytes")
    print(f"  -> bandwidth = {bw_gbs:.0f} GB/s   setup_latency = {latency:.2f} us")
    return bw_gbs, latency, (slope, intercept)

# ---------------------------------------------------------------------------
# XLA-external projection ops (full seq_len, not per-tile)
# ---------------------------------------------------------------------------

def make_xla_external_loop_fns(n_tok=2048, D=2560, Nq=32, Nkv=8, H=128,
                                rope_dim=64, dtype=jnp.bfloat16, n_iters=500):
    """Op factories for the XLA Q/K/V projection pipeline at full seq_len.

    x is the lax.fori_loop carry (shape [n_tok, D]), so it stays in VMEM across
    iterations — matching the intra-layer x-reuse that XLA achieves when K and V
    matmuls follow the Q matmul within the same compiled function.
    """
    key = jax.random.PRNGKey(42)
    W_q      = jax.random.normal(key, (D, Nq * H),       dtype=dtype)
    W_k      = jax.random.normal(key, (D, Nkv * H),      dtype=dtype)
    W_v      = jax.random.normal(key, (D, Nkv * H),      dtype=dtype)
    qn_scale = jax.random.normal(key, (H,),               dtype=jnp.float32)
    kn_scale = jax.random.normal(key, (H,),               dtype=jnp.float32)
    sin_q    = jax.random.normal(key, (n_tok, rope_dim),  dtype=jnp.float32)
    cos_q    = jax.random.normal(key, (n_tok, rope_dim),  dtype=jnp.float32)
    sin_k    = jax.random.normal(key, (n_tok, rope_dim),  dtype=jnp.float32)
    cos_k    = jax.random.normal(key, (n_tok, rope_dim),  dtype=jnp.float32)
    pad_k = D - Nkv * H   # 2560 - 1024 = 1536

    def make_xla_q_matmul():
        x0 = jax.random.normal(key, (n_tok, D), dtype=dtype)
        @jax.jit
        def loop(x):
            def body(i, x):
                out = jnp.dot(x, W_q, preferred_element_type=jnp.float32).astype(dtype)
                return out[:, :D]
            return jax.lax.fori_loop(0, n_iters, body, x)
        return loop, (x0,)

    def make_xla_k_matmul():
        x0 = jax.random.normal(key, (n_tok, D), dtype=dtype)
        @jax.jit
        def loop(x):
            def body(i, x):
                out = jnp.dot(x, W_k, preferred_element_type=jnp.float32).astype(dtype)
                return jnp.pad(out, ((0, 0), (0, pad_k)))
            return jax.lax.fori_loop(0, n_iters, body, x)
        return loop, (x0,)

    def make_xla_v_matmul():
        x0 = jax.random.normal(key, (n_tok, D), dtype=dtype)
        @jax.jit
        def loop(x):
            def body(i, x):
                out = jnp.dot(x, W_v, preferred_element_type=jnp.float32).astype(dtype)
                return jnp.pad(out, ((0, 0), (0, pad_k)))
            return jax.lax.fori_loop(0, n_iters, body, x)
        return loop, (x0,)

    def make_xla_q_norm_rope():
        x0 = jax.random.normal(key, (n_tok, D), dtype=dtype)
        @jax.jit
        def loop(x):
            def body(i, x):
                q_f32 = jnp.dot(x, W_q, preferred_element_type=jnp.float32)
                q_3d  = q_f32.reshape(n_tok * Nq, H)
                rms   = jax.lax.rsqrt(jnp.mean(q_3d ** 2, axis=-1, keepdims=True) + 1e-6)
                q_n   = (q_3d * rms * qn_scale).astype(dtype).reshape(n_tok, Nq, H)
                q1 = q_n[..., :rope_dim]; q2 = q_n[..., rope_dim:2 * rope_dim]
                q_rot = jnp.concatenate(
                    [q1 * cos_q[:, None, :] - q2 * sin_q[:, None, :],
                     q2 * cos_q[:, None, :] + q1 * sin_q[:, None, :]], axis=-1)
                return q_rot.reshape(n_tok, Nq * H).astype(dtype)[:, :D]
            return jax.lax.fori_loop(0, n_iters, body, x)
        return loop, (x0,)

    def make_xla_k_norm_rope():
        x0 = jax.random.normal(key, (n_tok, D), dtype=dtype)
        @jax.jit
        def loop(x):
            def body(i, x):
                k_f32 = jnp.dot(x, W_k, preferred_element_type=jnp.float32)
                k_3d  = k_f32.reshape(n_tok * Nkv, H)
                rms   = jax.lax.rsqrt(jnp.mean(k_3d ** 2, axis=-1, keepdims=True) + 1e-6)
                k_n   = (k_3d * rms * kn_scale).astype(dtype).reshape(n_tok, Nkv, H)
                k1 = k_n[..., :rope_dim]; k2 = k_n[..., rope_dim:2 * rope_dim]
                k_rot = jnp.concatenate(
                    [k1 * cos_k[:, None, :] - k2 * sin_k[:, None, :],
                     k2 * cos_k[:, None, :] + k1 * sin_k[:, None, :]], axis=-1)
                out = k_rot.reshape(n_tok, Nkv * H).astype(dtype)
                return jnp.pad(out, ((0, 0), (0, pad_k)))
            return jax.lax.fori_loop(0, n_iters, body, x)
        return loop, (x0,)

    return {
        "xla_q_matmul":    make_xla_q_matmul,
        "xla_k_matmul":    make_xla_k_matmul,
        "xla_v_matmul":    make_xla_v_matmul,
        "xla_q_norm_rope": make_xla_q_norm_rope,
        "xla_k_norm_rope": make_xla_k_norm_rope,
    }


# ---------------------------------------------------------------------------
# Per-tile kernel ops
# ---------------------------------------------------------------------------

def make_loop_fns(bq_sz, bkv_sz, bq_csz, bkv_csz, D, Nq, Nkv, H, rope_dim, dtype,
                  n_iters=500):
    """Return dict of {name: make_loop_fn} where each make_loop_fn() builds a
    jitted function that runs n_iters iterations of the op in a single lax.fori_loop.

    The loop carries a small state that depends on each iteration's output, preventing
    XLA from hoisting the computation out of the loop as loop-invariant code.
    Total device time / n_iters = per-op device execution time.
    """
    key = jax.random.PRNGKey(42)
    Nq_per_kv = Nq // Nkv

    # Fixed weights and auxiliary inputs (loop-invariant, not in carry)
    W_q    = jax.random.normal(key, (D, Nq*H),          dtype=dtype)
    W_k    = jax.random.normal(key, (D, Nkv*H),         dtype=dtype)
    W_v    = jax.random.normal(key, (D, Nkv*H),         dtype=dtype)
    W_kv   = jax.random.normal(key, (D, 2*Nkv*H),       dtype=dtype)
    qn_scale = jax.random.normal(key, (H,),              dtype=jnp.float32)
    kn_scale = jax.random.normal(key, (H,),              dtype=jnp.float32)
    sin_q  = jax.random.normal(key, (bq_sz, rope_dim),  dtype=jnp.float32)
    cos_q  = jax.random.normal(key, (bq_sz, rope_dim),  dtype=jnp.float32)
    sin_k  = jax.random.normal(key, (bkv_sz, rope_dim), dtype=jnp.float32)
    cos_k  = jax.random.normal(key, (bkv_sz, rope_dim), dtype=jnp.float32)
    k_fa   = jax.random.normal(key, (bkv_csz, H),       dtype=dtype)
    v_fa   = jax.random.normal(key, (bkv_csz, H),       dtype=dtype)
    sm_scale = H ** -0.5

    q_rows = bq_csz * Nq_per_kv

    def make_q_matmul():
        # Carry: x_bq[bq_sz, D] — each iter outputs [bq_sz, Nq*H], slice back to D cols
        x0 = jax.random.normal(key, (bq_sz, D), dtype=dtype)
        @jax.jit
        def loop(x_init):
            def body(i, x):
                out = jnp.dot(x, W_q, preferred_element_type=jnp.float32).astype(dtype)
                return out[:, :D]   # slice [bq_sz, Nq*H] → [bq_sz, D]
            return jax.lax.fori_loop(0, n_iters, body, x_init)
        return loop, (x0,)

    def make_kv_matmul():
        # Carry: x_bkv[bkv_sz, D]; output [bkv_sz, 2*Nkv*H], zero-pad back to D
        x0 = jax.random.normal(key, (bkv_sz, D), dtype=dtype)
        pad_kv = D - 2*Nkv*H   # 2560 - 2048 = 512
        @jax.jit
        def loop(x_init):
            def body(i, x):
                out = jnp.dot(x, W_kv, preferred_element_type=jnp.float32).astype(dtype)
                return jnp.pad(out, ((0,0),(0, pad_kv)))   # [bkv_sz, D]
            return jax.lax.fori_loop(0, n_iters, body, x_init)
        return loop, (x0,)

    def make_k_matmul_only():
        x0 = jax.random.normal(key, (bkv_sz, D), dtype=dtype)
        pad_k = D - Nkv*H   # 2560 - 1024 = 1536
        @jax.jit
        def loop(x_init):
            def body(i, x):
                out = jnp.dot(x, W_k, preferred_element_type=jnp.float32).astype(dtype)
                return jnp.pad(out, ((0,0),(0, pad_k)))
            return jax.lax.fori_loop(0, n_iters, body, x_init)
        return loop, (x0,)

    def make_v_matmul_only():
        x0 = jax.random.normal(key, (bkv_sz, D), dtype=dtype)
        pad_v = D - Nkv*H
        @jax.jit
        def loop(x_init):
            def body(i, x):
                out = jnp.dot(x, W_v, preferred_element_type=jnp.float32).astype(dtype)
                return jnp.pad(out, ((0,0),(0, pad_v)))
            return jax.lax.fori_loop(0, n_iters, body, x_init)
        return loop, (x0,)

    def make_q_norm_rope():
        # Carry: x_bq[bq_sz, D]; do Q matmul + norm + rope, feed output[:, :D] back
        x0 = jax.random.normal(key, (bq_sz, D), dtype=dtype)
        @jax.jit
        def loop(x_init):
            def body(i, x):
                q_f32 = jnp.dot(x, W_q, preferred_element_type=jnp.float32)
                q_3d  = q_f32.reshape(bq_sz * Nq, H)
                rms   = jax.lax.rsqrt(jnp.mean(q_3d ** 2, axis=-1, keepdims=True) + 1e-6)
                q_n   = (q_3d * rms * qn_scale).astype(dtype).reshape(bq_sz, Nq, H)
                q1    = q_n[..., :rope_dim]
                q2    = q_n[..., rope_dim:2*rope_dim]
                q_rot = jnp.concatenate(
                    [q1 * cos_q[:, None, :] - q2 * sin_q[:, None, :],
                     q2 * cos_q[:, None, :] + q1 * sin_q[:, None, :]], axis=-1)
                out   = q_rot.reshape(bq_sz, Nq * H).astype(dtype)
                return out[:, :D]   # [bq_sz, D], dtype=bf16
            return jax.lax.fori_loop(0, n_iters, body, x_init)
        return loop, (x0,)

    def make_k_norm_rope():
        x0 = jax.random.normal(key, (bkv_sz, D), dtype=dtype)
        pad_k = D - Nkv*H
        @jax.jit
        def loop(x_init):
            def body(i, x):
                kv_f32 = jnp.dot(x, W_kv, preferred_element_type=jnp.float32)
                k_3d   = kv_f32[:, :Nkv*H].reshape(bkv_sz * Nkv, H)
                rms    = jax.lax.rsqrt(jnp.mean(k_3d ** 2, axis=-1, keepdims=True) + 1e-6)
                k_n    = (k_3d * rms * kn_scale).astype(dtype).reshape(bkv_sz, Nkv, H)
                k1     = k_n[..., :rope_dim]
                k2     = k_n[..., rope_dim:2*rope_dim]
                k_rot  = jnp.concatenate(
                    [k1 * cos_k[:, None, :] - k2 * sin_k[:, None, :],
                     k2 * cos_k[:, None, :] + k1 * sin_k[:, None, :]], axis=-1)
                out    = k_rot.reshape(bkv_sz, Nkv * H).astype(dtype)
                return jnp.pad(out, ((0,0),(0, pad_k)))   # [bkv_sz, D], bf16
            return jax.lax.fori_loop(0, n_iters, body, x_init)
        return loop, (x0,)

    def make_flash_attn_qk():
        # step1_mxu + step1_vpu combined (old benchmark, kept for comparison)
        q0 = jax.random.normal(key, (q_rows, H), dtype=dtype)
        @jax.jit
        def loop(q_init):
            def body(i, q):
                s        = jnp.matmul(q, k_fa.T, preferred_element_type=jnp.float32) * sm_scale
                s_rowmax = jnp.max(s, axis=1, keepdims=True)
                p        = jnp.exp(s - s_rowmax)
                return p[:, :H].astype(dtype)
            return jax.lax.fori_loop(0, n_iters, body, q_init)
        return loop, (q0,)

    def make_flash_attn_pv():
        # step2_mxu + step2_vpu combined (old benchmark, kept for comparison)
        acc0 = jax.random.normal(key, (q_rows, H), dtype=jnp.float32)
        @jax.jit
        def loop(acc_init):
            def body(i, acc):
                p  = jnp.broadcast_to(acc[:, :1].astype(dtype), (q_rows, bkv_csz))
                pv = jnp.matmul(p, v_fa, preferred_element_type=jnp.float32)
                return acc + pv
            return jax.lax.fori_loop(0, n_iters, body, acc_init)
        return loop, (acc0,)

    def make_flash_attn_step1_mxu():
        # step1 MXU only: q @ k^T  (no softmax VPU work)
        q0 = jax.random.normal(key, (q_rows, H), dtype=dtype)
        @jax.jit
        def loop(q):
            def body(i, q):
                s = jnp.matmul(q, k_fa.T, preferred_element_type=jnp.float32) * sm_scale
                return s[:, :H].astype(dtype)   # slice [q_rows, H] for carry
            return jax.lax.fori_loop(0, n_iters, body, q)
        return loop, (q0,)

    def make_flash_attn_step1_vpu():
        # step1 VPU only: online softmax (rowmax + exp + sum)  — no matmul
        # carry is s [q_rows, bkv_csz] float32; output same shape
        s0 = jax.random.normal(key, (q_rows, bkv_csz), dtype=jnp.float32)
        @jax.jit
        def loop(s):
            def body(i, s):
                m = jnp.max(s, axis=1, keepdims=True)
                p = jnp.exp(s - m)
                l = jnp.sum(p, axis=1, keepdims=True)
                return p / (l + 1e-8)   # [q_rows, bkv_csz] float32
            return jax.lax.fori_loop(0, n_iters, body, s)
        return loop, (s0,)

    def make_flash_attn_step2_mxu():
        # step2 MXU only: p @ v  (no acc-update VPU work)
        p0 = jax.random.normal(key, (q_rows, H), dtype=dtype)
        @jax.jit
        def loop(p):
            def body(i, p):
                p_full = jnp.broadcast_to(p[:, :1].astype(dtype), (q_rows, bkv_csz))
                pv = jnp.matmul(p_full, v_fa, preferred_element_type=jnp.float32)
                return pv[:, :H].astype(dtype)   # [q_rows, H] for carry
            return jax.lax.fori_loop(0, n_iters, body, p)
        return loop, (p0,)

    def make_softmax_store():
        # acc / l for a full bq tile: [Nkv, bq_sz*Nq_per_kv, H] / [Nkv, bq_sz*Nq_per_kv, 1]
        acc0 = jax.random.normal(key, (Nkv, bq_sz * Nq_per_kv, H),    dtype=jnp.float32)
        l0   = jnp.ones(          (Nkv, bq_sz * Nq_per_kv, 1),         dtype=jnp.float32)
        @jax.jit
        def loop(acc_init, l_init):
            def body(i, carry):
                acc, l = carry
                out = (acc / l).astype(dtype)
                return acc + out.astype(jnp.float32), l   # force dependency on out
            acc_f, l_f = jax.lax.fori_loop(0, n_iters, body, (acc_init, l_init))
            return acc_f
        return loop, (acc0, l0)

    Nq_per_kv  = Nq // Nkv
    bkv_stride = Nkv          # cdiv(Nkv*2, kv_packing=2) = Nkv = 8

    # ── New ops: VPU/packing overhead not captured by the benchmarks above ──

    def make_rope_sincos_q():
        # kernel: load_rope_sincos computes jnp.sin+cos on-the-fly every Q tile
        # op_bench q_norm_rope uses pre-computed sin/cos — misses this cost
        sinusoid0 = jax.random.normal(key, (bq_sz, rope_dim), dtype=jnp.float32)
        @jax.jit
        def loop(s):
            def body(i, s): return jnp.sin(s) + jnp.cos(s)
            return jax.lax.fori_loop(0, n_iters, body, s)
        return loop, (sinusoid0,)

    def make_rope_sincos_k():
        # same as above but for K tiles: sinusoid [bkv_sz, rope_dim]
        sinusoid0 = jax.random.normal(key, (bkv_sz, rope_dim), dtype=jnp.float32)
        @jax.jit
        def loop(s):
            def body(i, s): return jnp.sin(s) + jnp.cos(s)
            return jax.lax.fori_loop(0, n_iters, body, s)
        return loop, (sinusoid0,)

    def make_kv_pack_store():
        # kernel: compute_kv_from_x_bkv packs K+V into strided VMEM layout
        # for Nkv=8 heads × strided_store([bkv_sz*bkv_stride=8192, H], step=bkv_stride=8)
        # kv_packing=2 (bf16 on v6e): k_bits | (v_bits << 16) → uint32
        kv_ref0 = jnp.zeros((bkv_sz * bkv_stride, H), dtype=jnp.uint32)
        k_u32   = jax.random.randint(key, (bkv_sz, H), 0, 65535, dtype=jnp.uint32)
        v_u32   = jax.random.randint(key, (bkv_sz, H), 0, 65535, dtype=jnp.uint32)
        packed0 = k_u32 | (v_u32 << jnp.uint32(16))
        @jax.jit
        def loop(kv_ref):
            def body(i, kv_ref):
                delta = kv_ref[0:1, 0:1]   # tiny carry dependency prevents XLA hoisting
                for h in range(Nkv):
                    kv_ref = kv_ref.at[h::bkv_stride, :].set(packed0 + delta)
                return kv_ref
            return jax.lax.fori_loop(0, n_iters, body, kv_ref)
        return loop, (kv_ref0,)

    def make_out_pack_store():
        # kernel: after softmax_store (acc/l), packs output via pltpu.bitcast + strided_store
        # strided_store with step=1 is sequential; main cost is bitcast + reshape
        q_rows = bq_sz * Nq_per_kv
        acc0   = jax.random.normal(key, (Nkv, q_rows, H), dtype=jnp.float32)
        l0     = jnp.ones((Nkv, q_rows, 1), dtype=jnp.float32)
        @jax.jit
        def loop(acc):
            def body(i, acc):
                out_bf16 = (acc / l0).astype(dtype)
                out_u16  = jax.lax.bitcast_convert_type(out_bf16, jnp.uint16)
                out_u32  = out_u16.astype(jnp.uint32).reshape(Nkv * q_rows, H)
                return acc + out_u32.astype(jnp.float32).reshape(Nkv, q_rows, H) * 1e-7
            return jax.lax.fori_loop(0, n_iters, body, acc)
        return loop, (acc0,)

    return {
        "q_matmul":        make_q_matmul,
        "kv_matmul":       make_kv_matmul,
        "k_matmul_only":   make_k_matmul_only,
        "v_matmul_only":   make_v_matmul_only,
        "q_norm_rope":     make_q_norm_rope,
        "k_norm_rope":     make_k_norm_rope,
        "flash_attn_qk":        make_flash_attn_qk,
        "flash_attn_pv":        make_flash_attn_pv,
        "flash_attn_step1_mxu": make_flash_attn_step1_mxu,
        "flash_attn_step1_vpu": make_flash_attn_step1_vpu,
        "flash_attn_step2_mxu": make_flash_attn_step2_mxu,
        "softmax_store":        make_softmax_store,
        "rope_sincos_q":   make_rope_sincos_q,
        "rope_sincos_k":   make_rope_sincos_k,
        "kv_pack_store":   make_kv_pack_store,
        "out_pack_store":  make_out_pack_store,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bq-sz",    type=int, default=256)
    ap.add_argument("--bkv-sz",   type=int, default=1024)
    ap.add_argument("--bq-csz",   type=int, default=128,   help="Q compute chunk size per KV head per attn step")
    ap.add_argument("--bkv-csz",  type=int, default=512,   help="KV compute chunk size per attn loop")
    ap.add_argument("--D",        type=int, default=2560)
    ap.add_argument("--Nq",       type=int, default=32)
    ap.add_argument("--Nkv",      type=int, default=8)
    ap.add_argument("--H",        type=int, default=128)
    ap.add_argument("--n-tok",    type=int, default=2048,  help="sequence length for XLA-external benchmarks")
    ap.add_argument("--rope-dim", type=int, default=64,    help="half head-dim; rope rotates first rope_dim and second rope_dim dims")
    ap.add_argument("--n-warmup", type=int, default=3,  help="warmup runs before timing")
    ap.add_argument("--n-reps",   type=int, default=5,  help="number of timed trials (each runs n_iters=500 device iters)")
    ap.add_argument("--skip-dma", action="store_true",  help="skip DMA bandwidth benchmark (Pallas)")
    ap.add_argument("--skip-xla", action="store_true",  help="skip XLA-external projection benchmark")
    args = ap.parse_args()

    dtype = jnp.bfloat16
    print(f"Device: {jax.devices()[0]}")
    print(f"bq_sz={args.bq_sz}  bkv_sz={args.bkv_sz}  bq_csz={args.bq_csz}  bkv_csz={args.bkv_csz}")
    print(f"D={args.D}  Nq={args.Nq}  Nkv={args.Nkv}  H={args.H}  rope_dim={args.rope_dim}")
    print(f"n_warmup={args.n_warmup}  n_reps={args.n_reps}  dtype={dtype}")
    print()

    N_ITERS = 500
    loop_fns = make_loop_fns(
        bq_sz=args.bq_sz, bkv_sz=args.bkv_sz, bq_csz=args.bq_csz,
        bkv_csz=args.bkv_csz, D=args.D, Nq=args.Nq, Nkv=args.Nkv,
        H=args.H, rope_dim=args.rope_dim, dtype=dtype, n_iters=N_ITERS,
    )

    Nq_per_kv  = args.Nq // args.Nkv
    bkv_stride = args.Nkv   # cdiv(Nkv*2, kv_packing=2)
    shapes_desc = {
        "q_matmul":       f"x_bq[{args.bq_sz},{args.D}] @ W_q[{args.D},{args.Nq*args.H}]",
        "kv_matmul":      f"x_bkv[{args.bkv_sz},{args.D}] @ W_kv[{args.D},{2*args.Nkv*args.H}]",
        "k_matmul_only":  f"x_bkv[{args.bkv_sz},{args.D}] @ W_k[{args.D},{args.Nkv*args.H}]",
        "v_matmul_only":  f"x_bkv[{args.bkv_sz},{args.D}] @ W_v[{args.D},{args.Nkv*args.H}]",
        "q_norm_rope":    f"matmul+norm+rope [{args.bq_sz},{args.Nq},{args.H}] (pre-computed sin/cos)",
        "k_norm_rope":    f"matmul+norm+rope [{args.bkv_sz},{args.Nkv},{args.H}] (pre-computed sin/cos)",
        "flash_attn_qk":        f"q[{args.bq_csz*Nq_per_kv},{args.H}] @ k^T + softmax  (step1 combined)",
        "flash_attn_pv":        f"p @ v + acc update  (step2 combined)",
        "flash_attn_step1_mxu": f"q[{args.bq_csz*Nq_per_kv},{args.H}] @ k[{args.bkv_csz},{args.H}]^T  (MXU only)",
        "flash_attn_step1_vpu": f"rowmax+exp+sum [{args.bq_csz*Nq_per_kv},{args.bkv_csz}]  (VPU only)",
        "flash_attn_step2_mxu": f"p[{args.bq_csz*Nq_per_kv},{args.bkv_csz}] @ v[{args.bkv_csz},{args.H}]  (MXU only)",
        "softmax_store":  f"acc[{args.Nkv},{args.bq_sz*Nq_per_kv},{args.H}] / l",
        "rope_sincos_q":  f"sin+cos [{args.bq_sz},{args.rope_dim}] float32  (on-the-fly per Q tile)",
        "rope_sincos_k":  f"sin+cos [{args.bkv_sz},{args.rope_dim}] float32  (on-the-fly per K tile)",
        "kv_pack_store":  f"{args.Nkv}×scatter [{args.bkv_sz},{args.H}] into [{args.bkv_sz*bkv_stride},{args.H}] uint32",
        "out_pack_store": f"bitcast bf16→uint32 + reshape [{args.Nkv*args.bq_sz*Nq_per_kv},{args.H}]",
    }

    results = {}
    print(f"{'Operation':<22}  {'µs/iter':>8}  {'std':>6}  shape / note")
    print(f"  (each measured as {N_ITERS} iters in lax.fori_loop / {N_ITERS}  — device time only)")
    print("-" * 82)
    for name, make_fn in loop_fns.items():
        mean_us, std_us = bench_loop(make_fn, n_iters=N_ITERS,
                                     n_warmup=args.n_warmup, n_trials=args.n_reps)
        results[name] = mean_us
        print(f"  {name:<20}  {mean_us:>8.2f}  {std_us:>6.2f}  {shapes_desc.get(name,'')}")

    # ── Derive kernel_timeline.py constants ──────────────────────────────────
    r = results

    # In Mosaic, Q matmul (MXU) and Q norm+rope (VPU) run in parallel.
    # The kernel ALSO computes sin/cos on-the-fly (load_rope_sincos), which adds
    # to the VPU time but was NOT in the q_norm_rope benchmark (used pre-computed).
    T_q_proj_vpu              = r["q_norm_rope"] + r["rope_sincos_q"]
    T_q_projection_wallclock  = max(r["q_matmul"], T_q_proj_vpu)

    # KV: fused K+V matmul (MXU) overlaps with K norm+rope+sin/cos (VPU).
    # After that, pack K+V into strided VMEM layout (kv_pack_store) — sequential.
    T_kv_proj_vpu             = r["k_norm_rope"] + r["rope_sincos_k"]
    T_kv_projection_wallclock = max(r["kv_matmul"], T_kv_proj_vpu) + r["kv_pack_store"]

    # Flash attention pipeline (Mosaic: MXU and VPU run in parallel across iterations):
    #   step1_mxu[i] → { step1_vpu[i] (VPU) || step2_mxu[i-1] (MXU) }
    #   wall-clock per steady-state iteration = T_step1_mxu + max(T_step1_vpu, T_step2_mxu)
    #
    # IMPORTANT: step1_vpu standalone benchmark is unreliable — the [q_rows, bkv_csz] carry
    # traverses HBM in JAX but stays in VMEM registers in Mosaic. Use inferred value instead:
    #   T_step1_vpu_inferred = flash_attn_qk - step1_mxu  (both measured in same carry regime)
    Nkv = args.Nkv
    n_bq_chunks = args.bq_sz // args.bq_csz
    T_step1_mxu          = r["flash_attn_step1_mxu"]
    T_step1_vpu_raw      = r["flash_attn_step1_vpu"]   # unreliable (HBM carry overhead)
    T_step1_vpu_inferred = max(0.0, r["flash_attn_qk"] - T_step1_mxu)
    T_step2_mxu          = r["flash_attn_step2_mxu"]
    T_one_step  = T_step1_mxu + max(T_step1_vpu_inferred, T_step2_mxu)
    T_flash_attention_per_loop = T_one_step * Nkv * n_bq_chunks

    # Output norm: acc/l normalize + bitcast/pack for HBM write.
    T_output_norm_per_bq_tile  = r["softmax_store"] + r["out_pack_store"]

    print()
    print("=" * 80)
    print("Derived kernel_timeline.py constants")
    print("(Mosaic pipelines MXU and VPU in parallel: wallclock = max, not sum)")
    print("=" * 80)
    print()
    print(f"  q_matmul={r['q_matmul']:.2f}µs  q_norm_rope={r['q_norm_rope']:.2f}µs"
          f"  rope_sincos_q={r['rope_sincos_q']:.2f}µs")
    print(f"  VPU(Q) = q_norm_rope + rope_sincos_q = {T_q_proj_vpu:.2f}µs")
    print(f"  → T_q_projection_wallclock   = max(q_matmul, VPU_q)"
          f" = {T_q_projection_wallclock:.2f}µs")
    print()
    print(f"  kv_matmul={r['kv_matmul']:.2f}µs  k_norm_rope={r['k_norm_rope']:.2f}µs"
          f"  rope_sincos_k={r['rope_sincos_k']:.2f}µs  kv_pack_store={r['kv_pack_store']:.2f}µs")
    print(f"  VPU(K) = k_norm_rope + rope_sincos_k = {T_kv_proj_vpu:.2f}µs")
    print(f"  → T_kv_projection_wallclock  = max(kv_matmul, VPU_k) + kv_pack"
          f" = {T_kv_projection_wallclock:.2f}µs")
    print()
    print(f"  step1_mxu={T_step1_mxu:.2f}µs  step1_vpu(raw)={T_step1_vpu_raw:.2f}µs ← HBM carry overhead, unreliable")
    print(f"  step1_vpu(inferred) = flash_attn_qk({r['flash_attn_qk']:.2f}) - step1_mxu({T_step1_mxu:.2f}) = {T_step1_vpu_inferred:.2f}µs")
    print(f"  step2_mxu={T_step2_mxu:.2f}µs")
    print(f"  pipeline: step1_mxu + max(step1_vpu_inf, step2_mxu) = {T_step1_mxu:.2f} + max({T_step1_vpu_inferred:.2f}, {T_step2_mxu:.2f}) = {T_one_step:.2f}µs")
    print(f"  T_one_step={T_one_step:.2f}µs × {Nkv} KV heads × {n_bq_chunks} bq_chunks")
    print(f"  → T_flash_attention_per_loop = {T_flash_attention_per_loop:.2f}µs  (back-calc=11.85µs is more reliable)")
    print()
    print(f"  softmax_store={r['softmax_store']:.2f}µs  out_pack_store={r['out_pack_store']:.2f}µs")
    print(f"  → T_output_norm_per_bq_tile  = softmax_store + out_pack = {T_output_norm_per_bq_tile:.2f}µs")
    print()
    print("Paste into kernel_timeline.py (compute ops — update after running with --skip-dma):")
    print()
    print(f"T_flash_attention_per_loop  = {T_flash_attention_per_loop:.2f}")
    print(f"T_q_projection_wallclock    = {T_q_projection_wallclock:.1f}")
    print(f"T_q_matmul_mxu              = {T_q_proj_vpu - r['q_norm_rope']:.2f}  # q_matmul corrected (for MXU bar)")
    print(f"T_kv_projection_wallclock   = {T_kv_projection_wallclock:.1f}")
    print(f"T_kv_matmul_mxu             = {r['kv_matmul']:.2f}  # kv_matmul corrected (for MXU bar)")
    print(f"T_output_norm_per_bq_tile   = {T_output_norm_per_bq_tile:.1f}")
    print(f"# DMA constants: run without --skip-dma to update these:")
    print(f"# DMA_SETUP_LATENCY = <intercept>")
    print(f"# DMA_BANDWIDTH_GBS = <1/slope/1e3>")
    print()
    print("Sanity check (compare to measured 253µs baseline, 545µs fused):")
    print("(Note: fused model also needs T_extra_dma_barrier_per_bkv=3.0µs × 12 bkv tiles)")
    n_loops = 1+1+2+2+3+3+4+4   # causal mask, 2048 tokens
    n_bkv_tiles = 1+1+1+1+2+2+2+2   # causal mask bkv tile count
    n_bq    = 8
    T_barrier = 3.0   # µs — extra DMA barrier per bkv tile in fused kernel
    baseline_model = n_loops * T_flash_attention_per_loop + n_bq * T_output_norm_per_bq_tile
    fused_model    = (baseline_model
                      + n_bq  * T_q_projection_wallclock
                      + 2     * T_kv_projection_wallclock
                      + n_bkv_tiles * T_barrier)
    print(f"  baseline = {n_loops}×{T_flash_attention_per_loop:.2f} + {n_bq}×{T_output_norm_per_bq_tile:.1f}"
          f" = {baseline_model:.0f}µs  (measured 253µs)")
    print(f"  fused    = baseline + 8×{T_q_projection_wallclock:.1f} + 2×{T_kv_projection_wallclock:.1f}"
          f" + 12×{T_barrier:.1f} = {fused_model:.0f}µs  (measured 545µs)")

    # ── Launch-overhead correction (two-point fit) ───────────────────────────
    # Each bench_loop call has a fixed Python→XLA dispatch cost T_fixed that
    # gets divided by N_ITERS.  Measure at N_LO and N_HI, fit a line, extract
    # the true per-iteration time as the slope.
    print()
    print("=" * 80)
    print("Launch-overhead correction  (T_total = T_fixed + T_op × N, solved at two N)")
    print("=" * 80)
    N_LO, N_HI = 100, 1000
    kw = dict(bq_sz=args.bq_sz, bkv_sz=args.bkv_sz, bq_csz=args.bq_csz,
              bkv_csz=args.bkv_csz, D=args.D, Nq=args.Nq, Nkv=args.Nkv,
              H=args.H, rope_dim=args.rope_dim, dtype=dtype)
    fns_lo = make_loop_fns(n_iters=N_LO, **kw)
    fns_hi = make_loop_fns(n_iters=N_HI, **kw)
    print(f"\n  {'Operation':<22}  {'naive/500':>9}  {'corrected':>9}  {'overhead':>9}")
    print("  " + "-" * 56)
    for name in fns_lo:
        lo_fn, lo_args = fns_lo[name]()
        hi_fn, hi_args = fns_hi[name]()
        for _ in range(args.n_warmup):
            jax.block_until_ready(lo_fn(*lo_args))
            jax.block_until_ready(hi_fn(*hi_args))
        t_lo_list, t_hi_list = [], []
        for _ in range(args.n_reps):
            t0 = time.perf_counter(); jax.block_until_ready(lo_fn(*lo_args))
            t_lo_list.append((time.perf_counter() - t0) * 1e6)
            t0 = time.perf_counter(); jax.block_until_ready(hi_fn(*hi_args))
            t_hi_list.append((time.perf_counter() - t0) * 1e6)
        T_lo = np.mean(t_lo_list)   # total µs for N_LO iters
        T_hi = np.mean(t_hi_list)   # total µs for N_HI iters
        T_op   = (T_hi - T_lo) / (N_HI - N_LO)   # µs per iter (slope)
        T_fixed = T_lo - T_op * N_LO              # µs fixed overhead (intercept)
        naive = results.get(name, float("nan"))
        print(f"  {name:<22}  {naive:>9.2f}  {T_op:>9.2f}  {T_fixed:>9.2f}  µs")

    # ── DMA bandwidth ────────────────────────────────────────────────────────
    if not args.skip_dma:
        print()
        print("=" * 80)
        print("DMA bandwidth  (Pallas HBM copy kernel, affine fit  t = a + b×bytes)")
        print("=" * 80)
        print()
        bw_gbs, lat_us, _ = bench_dma_bandwidth()
        if bw_gbs is not None:
            print()
            print(f"  → Update kernel_timeline.py: dma_us(B, bw={bw_gbs:.0f})")
            print(f"    + add {lat_us:.1f} µs fixed setup latency per DMA transfer")

    # ── XLA-external projection ops ──────────────────────────────────────────
    if not args.skip_xla:
        print()
        print("=" * 80)
        print(f"XLA-external projection ops  (n_tok={args.n_tok}, full seq_len)")
        print("=" * 80)
        xla_shapes = {
            "xla_q_matmul":    f"x[{args.n_tok},{args.D}] @ W_q[{args.D},{args.Nq*args.H}]",
            "xla_k_matmul":    f"x[{args.n_tok},{args.D}] @ W_k[{args.D},{args.Nkv*args.H}]",
            "xla_v_matmul":    f"x[{args.n_tok},{args.D}] @ W_v[{args.D},{args.Nkv*args.H}]",
            "xla_q_norm_rope": f"matmul+norm+rope  [{args.n_tok},{args.Nq},{args.H}]",
            "xla_k_norm_rope": f"matmul+norm+rope  [{args.n_tok},{args.Nkv},{args.H}]",
        }
        xla_fns = make_xla_external_loop_fns(
            n_tok=args.n_tok, D=args.D, Nq=args.Nq, Nkv=args.Nkv, H=args.H,
            rope_dim=args.rope_dim, dtype=dtype, n_iters=N_ITERS,
        )
        xla_r = {}
        print(f"\n  {'Operation':<22}  {'µs/iter':>8}  {'std':>6}  shape")
        print("  " + "-" * 80)
        for name, make_fn in xla_fns.items():
            mean_us, std_us = bench_loop(make_fn, n_iters=N_ITERS,
                                         n_warmup=args.n_warmup, n_trials=args.n_reps)
            xla_r[name] = mean_us
            print(f"  {name:<22}  {mean_us:>8.2f}  {std_us:>6.2f}  {xla_shapes.get(name, '')}")

        T_Q = xla_r.get("xla_q_norm_rope", 0)
        T_K = xla_r.get("xla_k_norm_rope", 0)
        T_V = xla_r.get("xla_v_matmul",    0)
        print()
        print("  XLA pipeline (sequential, x reused Q→K→V from VMEM carry):")
        print(f"    Q (matmul+norm+rope) = {T_Q:.1f} µs")
        print(f"    K (matmul+norm+rope) = {T_K:.1f} µs")
        print(f"    V (matmul only)      = {T_V:.1f} µs")
        print(f"    Critical path total  = {T_Q + T_K + T_V:.1f} µs")
        print()
        print("  Comparison to kernel_timeline.py XLA constants (estimated from HLO):")
        print(f"    Q: T_Qmat+T_Qpost = 30+95 = 125 µs   measured {T_Q:.1f} µs")
        print(f"    K: T_Kmat+T_Kpost =  7+14 =  21 µs   measured {T_K:.1f} µs")
        print(f"    V: T_Vmat         =    14 µs          measured {T_V:.1f} µs")


if __name__ == "__main__":
    main()
