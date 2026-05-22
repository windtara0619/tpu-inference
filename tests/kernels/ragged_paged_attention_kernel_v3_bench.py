"""Benchmark: merged vs non-merged ragged paged attention.

Setup:
  - 1000 sequences, q_len=20, kv_len=20 (mixed/encode path)
  - num_q_heads=4, num_kv_heads=1  (GQA ratio 4)
  - head_dim=128
  - compute_size in {128, 256, 512}

Timing method:
  jax.profiler trace → parse XLA Module events on /device:TPU:0
  (pid=3, tid=2) to extract pure TPU execution time, discarding
  host dispatch and compilation overhead.
"""

import gzip
import json
import os
import shutil
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import test_util as jtu

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_SEQS    = 1000
Q_LEN       = 20
KV_LEN      = 20
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1           # GQA ratio = 4
HEAD_DIM    = 128
DTYPE       = jnp.bfloat16
PAGE_SIZE   = 16
WARMUP_ITERS = 3
PROFILE_ITERS = 5          # iterations inside one profiler trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_merged_group_cu_seqs(seq_lens, max_num_seqs, compute_size,
                                num_q_heads_per_kv_head):
    q_limit = compute_size // num_q_heads_per_kv_head
    group_boundaries = [0]
    cur_q = cur_kv = cur_n = 0
    for q_len, kv_len in seq_lens:
        fits = (cur_q + q_len <= q_limit and cur_kv + kv_len <= compute_size)
        if fits and cur_n > 0:
            cur_q += q_len; cur_kv += kv_len; cur_n += 1
        else:
            if cur_n > 0:
                group_boundaries.append(group_boundaries[-1] + cur_n)
            cur_q = q_len; cur_kv = kv_len; cur_n = 1
    if cur_n > 0:
        group_boundaries.append(group_boundaries[-1] + cur_n)
    arr = np.array(group_boundaries, dtype=np.int32)
    result = np.full(max_num_seqs + 1, arr[-1], dtype=np.int32)
    result[:len(arr)] = arr
    return jnp.array(result)


def _build_inputs(num_seqs, q_len, kv_len, num_q_heads, num_kv_heads,
                  head_dim, page_size, dtype):
    rng = np.random.default_rng(0)
    kv_packing  = get_dtype_packing(dtype)
    padded_hd   = align_to(head_dim, 128)
    num_kv_x2   = align_to(num_kv_heads * 2, kv_packing)
    pages_per_seq = cdiv(kv_len, page_size)

    total_q      = num_seqs * q_len
    max_tokens   = align_to(total_q, 128)
    max_num_seq  = align_to(num_seqs, 8)

    def rand(shape):
        return jnp.array(rng.random(shape, dtype=np.float32)).astype(dtype)

    q = rand((max_tokens, num_q_heads, head_dim))
    k = rand((max_tokens, num_kv_heads, head_dim))
    v = rand((max_tokens, num_kv_heads, head_dim))

    # Build paged KV cache: all seqs share identical kv_len pages.
    kv_page = rand((kv_len, num_kv_x2 // kv_packing, kv_packing, padded_hd))
    kv_page_padded = jnp.pad(
        kv_page,
        ((0, pages_per_seq * page_size - kv_len), (0,0), (0,0), (0,0))
    ).reshape(pages_per_seq, page_size, num_kv_x2 // kv_packing, kv_packing,
               padded_hd)

    num_pages = num_seqs * pages_per_seq + 1
    kv_cache  = jnp.tile(kv_page_padded, (num_seqs, 1, 1, 1, 1)).reshape(
        num_seqs * pages_per_seq, page_size, num_kv_x2 // kv_packing,
        kv_packing, padded_hd)
    kv_cache  = jnp.pad(kv_cache, ((0, num_pages - kv_cache.shape[0]),
                                    (0,0),(0,0),(0,0),(0,0)))

    page_indices = jnp.arange(num_seqs * pages_per_seq,
                               dtype=jnp.int32).reshape(num_seqs, pages_per_seq)
    page_indices = jnp.pad(page_indices,
                            ((0, max_num_seq - num_seqs), (0, 0))).reshape(-1)

    cu_q = np.arange(num_seqs + 1, dtype=np.int32) * q_len
    cu_q_arr = jnp.pad(jnp.array(cu_q), (0, max_num_seq + 1 - len(cu_q)))

    kv_lens_arr = jnp.pad(
        jnp.full((num_seqs,), kv_len, dtype=jnp.int32),
        (0, max_num_seq - num_seqs))

    # All seqs in MIXED range: distribution = [0, 0, num_seqs]
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    return (q, k, v, kv_cache, kv_lens_arr, page_indices, cu_q_arr,
            distribution, num_pages, max_num_seq, pages_per_seq)


def _parse_tpu_module_us(trace_gz_path):
    """Return list of XLA-Module durations (µs) on /device:TPU:0."""
    with gzip.open(trace_gz_path, 'rb') as f:
        trace = json.load(f)
    events = trace.get('traceEvents', [])

    # pid=3 → /device:TPU:0, tid=2 → XLA Modules
    tpu_pid = None
    xla_module_tid = None
    for e in events:
        if e.get('ph') == 'M' and e.get('name') == 'process_name':
            if '/device:TPU' in e.get('args', {}).get('name', ''):
                tpu_pid = e['pid']
        if tpu_pid and e.get('ph') == 'M' and e.get('pid') == tpu_pid \
                and e.get('name') == 'thread_name':
            if 'XLA Modules' in e.get('args', {}).get('name', ''):
                xla_module_tid = e['tid']

    if tpu_pid is None or xla_module_tid is None:
        return []

    return [e['dur'] for e in events
            if e.get('ph') == 'X'
            and e.get('pid') == tpu_pid
            and e.get('tid') == xla_module_tid
            and 'dur' in e]


def _find_trace_gz(trace_dir):
    for root, _, files in os.walk(trace_dir):
        for f in files:
            if f.endswith('.trace.json.gz'):
                return os.path.join(root, f)
    return None


def _run_and_time(fn, warmup, profile_iters, label):
    """Warm up, then collect pure TPU time via jax.profiler.

    fn() must return a fresh output each call without holding onto donated
    buffers across calls (the caller should copy donated tensors internally).
    """
    # Warm-up: JIT-compile and fill caches
    for _ in range(warmup):
        fn().block_until_ready()

    # Profile
    trace_dir = tempfile.mkdtemp(prefix='rpa_bench_')
    try:
        jax.profiler.start_trace(trace_dir)
        for _ in range(profile_iters):
            fn().block_until_ready()
        jax.profiler.stop_trace()

        gz = _find_trace_gz(trace_dir)
        if gz is None:
            print(f"  {label}: trace file not found")
            return None

        durations = _parse_tpu_module_us(gz)
        if not durations:
            print(f"  {label}: no XLA Module events found in trace")
            return None

        # There may be multiple module events (e.g. one per iteration).
        # Take the median to be robust to outliers.
        median_us = float(np.median(durations))
        min_us    = float(np.min(durations))
        max_us    = float(np.max(durations))
        print(f"  {label}: median={median_us/1e3:.3f}ms  "
              f"min={min_us/1e3:.3f}ms  max={max_us/1e3:.3f}ms  "
              f"(over {len(durations)} XLA-module events)")
        return median_us
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    if not jtu.is_device_tpu_at_least(version=4):
        print("Skipping: requires TPUv4+")
        return

    num_q_per_kv = NUM_Q_HEADS // NUM_KV_HEADS  # = 4

    print(f"\n{'='*70}")
    print(f"RPA v3 Benchmark: merged vs non-merged")
    print(f"  {NUM_SEQS} seqs × q_len={Q_LEN}, kv_len={KV_LEN}")
    print(f"  num_q_heads={NUM_Q_HEADS}, num_kv_heads={NUM_KV_HEADS}  "
          f"(GQA ratio={num_q_per_kv})")
    print(f"  head_dim={HEAD_DIM}, dtype={DTYPE}, page_size={PAGE_SIZE}")
    print(f"  warmup={WARMUP_ITERS}, profile_iters={PROFILE_ITERS}")
    print(f"{'='*70}\n")

    (q, k, v, kv_cache, kv_lens_arr, page_indices, cu_q_arr, distribution,
     num_pages, max_num_seq, pages_per_seq) = _build_inputs(
        NUM_SEQS, Q_LEN, KV_LEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        PAGE_SIZE, DTYPE)

    seq_lens = [(Q_LEN, KV_LEN)] * NUM_SEQS
    bkv_sz = max(128, KV_LEN)   # must be >= kv_len and a power of 2 multiple

    results = {}

    # q, k, v and kv_cache are donated/aliased in the kernel's
    # input_output_aliases, so we must supply fresh copies each call.
    # Keep host copies and device_put each iteration.
    q_host       = np.array(q)
    k_host       = np.array(k)
    v_host       = np.array(v)
    kv_cache_host = np.array(kv_cache)

    def fresh():
        return (jax.device_put(q_host),
                jax.device_put(k_host),
                jax.device_put(v_host),
                jax.device_put(kv_cache_host))

    # -----------------------------------------------------------------------
    # Non-merged baseline
    # -----------------------------------------------------------------------
    print("--- Non-merged (baseline) ---")
    def run_non_merged():
        fq, fk, fv, fkv = fresh()
        out, _ = ragged_paged_attention(
            fq, fk, fv, fkv, kv_lens_arr, page_indices, cu_q_arr,
            distribution,
            use_causal_mask=True,
            merge_mixed_seqs=False,
            m_block_sizes=(128, bkv_sz, 128, bkv_sz),
        )
        return out

    t_base = _run_and_time(run_non_merged, WARMUP_ITERS, PROFILE_ITERS,
                           "non-merged")
    results['non-merged'] = t_base

    # -----------------------------------------------------------------------
    # Merged with different compute_size
    # -----------------------------------------------------------------------
    for compute_size in [128, 256, 512]:
        q_limit = compute_size // num_q_per_kv
        seqs_per_group = q_limit // Q_LEN
        print(f"\n--- Merged, compute_size={compute_size} "
              f"(q_limit={q_limit}, ~{seqs_per_group} seqs/group) ---")

        merged_group_cu_seqs = _build_merged_group_cu_seqs(
            seq_lens, max_num_seq, compute_size, num_q_per_kv)

        cs = compute_size  # capture for closure
        mcu = merged_group_cu_seqs

        def run_merged(cs=cs, mcu=mcu):
            fq, fk, fv, fkv = fresh()
            out, _ = ragged_paged_attention(
                fq, fk, fv, fkv, kv_lens_arr, page_indices, cu_q_arr,
                distribution,
                use_causal_mask=True,
                merge_mixed_seqs=True,
                compute_size=cs,
                merged_group_cu_seqs=mcu,
                m_block_sizes=(cs, bkv_sz, cs, bkv_sz),
            )
            return out

        t = _run_and_time(run_merged, WARMUP_ITERS, PROFILE_ITERS,
                          f"merged compute_size={cs}")
        results[f'merged-{cs}'] = t

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Summary (median pure TPU time):")
    base = results.get('non-merged')
    for label, t in results.items():
        if t is None:
            print(f"  {label:30s}  N/A")
        else:
            speedup = f"  {base/t:.2f}x speedup" if base and label != 'non-merged' else ""
            print(f"  {label:30s}  {t/1e3:.3f} ms{speedup}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    run_benchmark()
