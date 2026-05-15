"""Benchmark: padded vs tight KV packing for group (multi-seq) prefill.

With padded packing each seq's KV slot in VMEM is rounded up to page_size,
wasting MXU capacity.  With tight packing KV is concatenated exactly, allowing
more sequences per group (fewer MXU calls).

Usage:
    python benchmarks/group_kv_packing_benchmark.py
"""

import gzip
import json
import os
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention,
)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing,
)

# ---------------------------------------------------------------------------
# Scenario: many short seqs where padding is expensive.
# kv=17 → padded=32 → 4 seqs fill bkv_csz=128; tight=17 → 7 seqs fit.
# With 28 seqs: padded needs 7 groups, tight needs 4 groups → ~1.75× fewer.
# ---------------------------------------------------------------------------
PAGE_SIZE = 16
BQ_CSZ = 128
BKV_CSZ = 128
BQ_SZ = 128
BKV_SZ = 256

Q_LEN = 17
KV_LEN = 17
NUM_SEQS = 28          # 28 seqs × kv=17: padded→7 groups (4+4+4+4+4+4+4), tight→4 groups (7+7+7+7)
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
NUM_PAGES = 256

NUM_WARMUP = 3
NUM_ITERS = 20


def build_inputs(rng: np.random.Generator, dtype):
    packing = get_dtype_packing(dtype)
    padded_head_dim = align_to(HEAD_DIM, 128)
    num_kv_heads_x2 = align_to(NUM_KV_HEADS * 2, packing)
    pages_per_seq = cdiv(KV_LEN, PAGE_SIZE)
    max_num_tokens = align_to(NUM_SEQS * Q_LEN, 128)
    max_num_seq = align_to(NUM_SEQS, 8)

    q = jnp.array(rng.random((max_num_tokens, NUM_Q_HEADS, HEAD_DIM),
                               dtype=np.float32)).astype(dtype)
    k = jnp.array(rng.random((max_num_tokens, NUM_KV_HEADS, HEAD_DIM),
                               dtype=np.float32)).astype(dtype)
    v = jnp.array(rng.random((max_num_tokens, NUM_KV_HEADS, HEAD_DIM),
                               dtype=np.float32)).astype(dtype)

    # Build paged KV cache with one entry per seq.
    page_cnt = 0
    page_indices_list = []
    kv_pages_list = []
    kv_lens = []
    cu_q_lens = [0]
    for _ in range(NUM_SEQS):
        kv_lens.append(KV_LEN)
        cu_q_lens.append(cu_q_lens[-1] + Q_LEN)
        kv = jnp.array(rng.random((KV_LEN, num_kv_heads_x2 // packing,
                                    packing, padded_head_dim),
                                   dtype=np.float32)).astype(dtype)
        n_pages = cdiv(KV_LEN, PAGE_SIZE)
        kv = jnp.pad(kv, ((0, n_pages * PAGE_SIZE - KV_LEN), (0, 0), (0, 0),
                           (0, 0))).reshape(n_pages, PAGE_SIZE,
                                            num_kv_heads_x2 // packing,
                                            packing, padded_head_dim)
        indices = page_cnt + jnp.arange(n_pages, dtype=jnp.int32)
        indices = jnp.pad(indices, (0, pages_per_seq - n_pages))
        page_indices_list.append(indices)
        kv_pages_list.append(kv)
        page_cnt += n_pages

    kv_cache = jnp.concatenate(kv_pages_list, axis=0)
    kv_cache = jnp.pad(kv_cache,
                        ((0, NUM_PAGES - kv_cache.shape[0]), (0,0),(0,0),(0,0),(0,0)))
    page_indices = jnp.stack(page_indices_list, axis=0)
    page_indices = jnp.pad(page_indices,
                            ((0, max_num_seq - NUM_SEQS), (0, 0))).reshape(-1)

    cu_q_lens_arr = jnp.array(cu_q_lens, dtype=jnp.int32)
    cu_q_lens_arr = jnp.pad(cu_q_lens_arr, (0, max_num_seq + 1 - len(cu_q_lens)))
    kv_lens_arr = jnp.array(kv_lens, dtype=jnp.int32)
    kv_lens_arr = jnp.pad(kv_lens_arr, (0, max_num_seq - NUM_SEQS))
    distribution = jnp.array([0, 0, NUM_SEQS], dtype=jnp.int32)

    return (q, k, v, kv_cache, kv_lens_arr, page_indices, cu_q_lens_arr,
            distribution)


def run_kernel(inputs, tight: bool):
    q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution = inputs
    out, _ = ragged_paged_attention(
        q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution,
        m_block_sizes=(BQ_SZ, BKV_SZ, BQ_CSZ, BKV_CSZ),
        tight_kv_packing=tight,
    )
    return out


def get_tpu_time_ms(trace_dir: str) -> float:
    """Parse a JAX Perfetto trace directory and return total TPU active time in ms."""
    trace_file = os.path.join(trace_dir, "plugins", "profile",
                               next(d for d in os.listdir(
                                   os.path.join(trace_dir, "plugins", "profile"))
                                    if os.path.isdir(os.path.join(
                                        trace_dir, "plugins", "profile", d))),
                               "trace.json.gz")
    if not os.path.exists(trace_file):
        # Fallback: search recursively.
        for root, _, files in os.walk(trace_dir):
            for f in files:
                if f == "trace.json.gz":
                    trace_file = os.path.join(root, f)
                    break

    with gzip.open(trace_file, "rt") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])
    # Sum durations of all complete events on the "TFRT" / XLA device stream.
    # Category "Op" or pid containing "tpu" / "StreamExecutor".
    tpu_us = 0
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        pid = str(ev.get("pid", ""))
        name = ev.get("name", "")
        # XLA device kernel events appear as "Op" or in the "XlaModule" category.
        if ("XlaModule" in cat or cat == "Op" or
                "tpu" in pid.lower() or "StreamExecutor" in name):
            tpu_us += ev.get("dur", 0)

    return tpu_us / 1000.0  # µs → ms


def benchmark_variant(inputs, tight: bool, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # JIT-compile + warmup.
    jitted = jax.jit(lambda inp: run_kernel(inp, tight))
    print("  Warming up ...", end="", flush=True)
    for _ in range(NUM_WARMUP):
        out = jitted(inputs)
        out.block_until_ready()
    print(" done")

    # Wall-clock timing.
    times = []
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        out = jitted(inputs)
        out.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    wall_ms = np.median(times)
    print(f"  Wall time (median of {NUM_ITERS}): {wall_ms:.3f} ms")

    # Profiler trace.
    with tempfile.TemporaryDirectory() as trace_dir:
        with jax.profiler.trace(trace_dir, create_perfetto_trace=True):
            for _ in range(5):
                out = jitted(inputs)
                out.block_until_ready()

        try:
            tpu_ms = get_tpu_time_ms(trace_dir)
            print(f"  TPU time (5 iters, from trace): {tpu_ms:.3f} ms  "
                  f"(~{tpu_ms/5:.3f} ms/iter)")
        except Exception as e:
            print(f"  [trace parse failed: {e}]")

    return wall_ms


def main():
    rng = np.random.default_rng(42)
    dtype = jnp.bfloat16
    inputs = build_inputs(rng, dtype)

    seqs_per_group_padded = BKV_CSZ // align_to(KV_LEN, PAGE_SIZE)
    seqs_per_group_tight  = BKV_CSZ // KV_LEN
    print(f"Config: {NUM_SEQS} seqs, q={Q_LEN}, kv={KV_LEN}, page={PAGE_SIZE}")
    print(f"  padded kv step = {align_to(KV_LEN, PAGE_SIZE)} "
          f"→ {seqs_per_group_padded} seqs/group "
          f"→ {-(-NUM_SEQS // seqs_per_group_padded)} groups")
    print(f"  tight  kv step = {KV_LEN} "
          f"→ {seqs_per_group_tight} seqs/group "
          f"→ {-(-NUM_SEQS // seqs_per_group_tight)} groups")

    t_padded = benchmark_variant(inputs, tight=False, label="Padded KV packing")
    t_tight  = benchmark_variant(inputs, tight=True,  label="Tight KV packing")

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Padded: {t_padded:.3f} ms")
    print(f"  Tight:  {t_tight:.3f} ms")
    print(f"  Speedup: {t_padded/t_tight:.2f}x")


if __name__ == "__main__":
    main()
