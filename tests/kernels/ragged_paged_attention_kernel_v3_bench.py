"""Benchmark: merged vs non-merged ragged paged attention.

Setup:
  - 1000 sequences, q_len=16, kv_len=16 (pure encode: kv_left_frm_cache=0)
  - num_q_heads=4, num_kv_heads=1  (GQA ratio 4)
  - head_dim=128
  - compute_size in {128, 256, 512}

Experiments (run for each compute_size):
  A. baseline     — update_kv_cache=True,  skip_kv_mask=False
  B. no_kv_update — update_kv_cache=False, skip_kv_mask=False  (H3: KV write overhead)
  C. no_mask      — update_kv_cache=True,  skip_kv_mask=True   (H2: mask build overhead)

Hypotheses:
  H1 (pipeline): inspect per-event durations below — uniform sizes = pipeline works;
                 first event much larger = stall waiting for first-group DMA.
  H2 (mask):     no_mask variant skips kv_start_2d/kv_end_2d construction.
  H3 (kv write): no_kv_update variant skips writing new tokens to KV cache.
  H4 (KV tile):  K loaded as [compute_size] rows but only max_q_tokens=compute_size//nhd
                 rows hold valid data (pure encode). MXU pads [max_q_tokens, cs] matmul
                 to [128, 128] tile → (cs/max_q_tokens)x wasted MXU compute.

Timing method:
  jax.profiler trace → parse XLA Module events on /device:TPU:0
  to extract pure TPU execution time, discarding host dispatch and compilation overhead.
"""

import gzip
import json
import os
import shutil
import tempfile

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
NUM_SEQS     = 1000
Q_LEN        = 16
KV_LEN       = 16           # == Q_LEN → pure encode: kv_left_frm_cache = 0
NUM_Q_HEADS  = 4
NUM_KV_HEADS = 1            # GQA ratio = 4
HEAD_DIM     = 128
DTYPE        = jnp.bfloat16
PAGE_SIZE    = 128          # pages_per_seq = cdiv(16, 128) = 1
WARMUP_ITERS  = 3
PROFILE_ITERS = 5           # iterations inside one profiler trace


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
    kv_packing   = get_dtype_packing(dtype)
    padded_hd    = align_to(head_dim, 128)
    num_kv_x2    = align_to(num_kv_heads * 2, kv_packing)
    pages_per_seq = cdiv(kv_len, page_size)

    total_q     = num_seqs * q_len
    max_tokens  = align_to(total_q, 128)
    max_num_seq = align_to(num_seqs, 8)

    def rand(shape):
        return jnp.array(rng.random(shape, dtype=np.float32)).astype(dtype)

    q = rand((max_tokens, num_q_heads, head_dim))
    k = rand((max_tokens, num_kv_heads, head_dim))
    v = rand((max_tokens, num_kv_heads, head_dim))

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

    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    return (q, k, v, kv_cache, kv_lens_arr, page_indices, cu_q_arr,
            distribution, num_pages, max_num_seq, pages_per_seq)


def _parse_tpu_module_events(trace_gz_path):
    """Return list of (name, dur_us) XLA-Module events on /device:TPU:0."""
    with gzip.open(trace_gz_path, 'rb') as f:
        trace = json.load(f)
    events = trace.get('traceEvents', [])

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

    return [(e.get('name', '<unnamed>'), e['dur']) for e in events
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

    Prints all XLA-module events with names and individual durations.
    H1 (pipeline check): if events are uniformly sized the data pipeline is
    working; if the first event is much larger there is a stall before the
    double-buffer kicks in.
    """
    for _ in range(warmup):
        fn().block_until_ready()

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

        events = _parse_tpu_module_events(gz)
        if not events:
            print(f"  {label}: no XLA Module events found in trace")
            return None

        durations = [dur for _, dur in events]
        print(f"\n  [{label}]  ({len(events)} XLA-module events, "
              f"{profile_iters} profile iters)")
        for name, dur in events:
            print(f"    {name[:64]:64s}  {dur/1e3:7.3f} ms")

        median_us = float(np.median(durations))
        min_us    = float(np.min(durations))
        max_us    = float(np.max(durations))
        print(f"    --> median={median_us/1e3:.3f}ms  "
              f"min={min_us/1e3:.3f}ms  max={max_us/1e3:.3f}ms")
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
    print(f"RPA v3  merged vs non-merged  (overhead diagnosis)")
    print(f"  {NUM_SEQS} seqs x q_len={Q_LEN}, kv_len={KV_LEN}  "
          f"(pure encode: kv_left_frm_cache=0)")
    print(f"  num_q_heads={NUM_Q_HEADS}, num_kv_heads={NUM_KV_HEADS} "
          f"(GQA={num_q_per_kv})")
    print(f"  head_dim={HEAD_DIM}, dtype={DTYPE}, page_size={PAGE_SIZE}")
    print(f"  warmup={WARMUP_ITERS}, profile_iters={PROFILE_ITERS}")
    print(f"Experiments per compute_size:")
    print(f"  A. baseline     update_kv_cache=True  skip_kv_mask=False")
    print(f"  B. no_kv_update update_kv_cache=False skip_kv_mask=False "
          f"(H3: KV write)")
    print(f"  C. no_mask      update_kv_cache=True  skip_kv_mask=True  "
          f"(H2: mask build)")
    print(f"H1 pipeline check: inspect per-event durations "
          f"(uniform = good pipeline)")
    print(f"{'='*70}")

    (q, k, v, kv_cache, kv_lens_arr, page_indices, cu_q_arr, distribution,
     num_pages, max_num_seq, pages_per_seq) = _build_inputs(
        NUM_SEQS, Q_LEN, KV_LEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        PAGE_SIZE, DTYPE)

    seq_lens = [(Q_LEN, KV_LEN)] * NUM_SEQS
    bkv_sz   = max(128, KV_LEN)

    q_host        = np.array(q)
    k_host        = np.array(k)
    v_host        = np.array(v)
    kv_cache_host = np.array(kv_cache)

    def fresh():
        return (jax.device_put(q_host),
                jax.device_put(k_host),
                jax.device_put(v_host),
                jax.device_put(kv_cache_host))

    results = {}

    # -----------------------------------------------------------------------
    # Non-merged baselines (with and without KV cache update)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("NON-MERGED")
    print(f"{'='*70}")
    for update_kv, tag in [(True, "A-baseline"), (False, "B-no_kv_update")]:
        label = f"non-merged [{tag}]"

        def run_non_merged(ukv=update_kv):
            fq, fk, fv, fkv = fresh()
            out, _ = ragged_paged_attention(
                fq, fk, fv, fkv, kv_lens_arr, page_indices, cu_q_arr,
                distribution,
                use_causal_mask=True,
                update_kv_cache=ukv,
                merge_mixed_seqs=False,
                m_block_sizes=(128, bkv_sz, 128, bkv_sz),
            )
            return out

        t = _run_and_time(run_non_merged, WARMUP_ITERS, PROFILE_ITERS, label)
        results[label] = t

    # -----------------------------------------------------------------------
    # Merged: three experiment variants per compute_size
    # -----------------------------------------------------------------------
    for compute_size in [128, 256, 512]:
        q_limit       = compute_size // num_q_per_kv
        seqs_per_group = q_limit // Q_LEN
        print(f"\n{'='*70}")
        print(f"MERGED  compute_size={compute_size}  "
              f"max_q_tokens={q_limit}  ~{seqs_per_group} seqs/group")
        print(f"{'='*70}")

        merged_group_cu_seqs = _build_merged_group_cu_seqs(
            seq_lens, max_num_seq, compute_size, num_q_per_kv)

        cs  = compute_size
        mcu = merged_group_cu_seqs

        # C-no_mask uses skip_kv_mask=True which requires use_causal_mask=False.
        # Correctness is irrelevant here; we only want the timing delta.
        for update_kv, skip_mask, causal, tag in [
            (True,  False, True,  "A-baseline"),
            (False, False, True,  "B-no_kv_update"),
            (True,  True,  False, "C-no_mask"),
        ]:
            label = f"merged-{cs} [{tag}]"

            def run_merged(cs=cs, mcu=mcu, ukv=update_kv, sm=skip_mask,
                           csl=causal):
                fq, fk, fv, fkv = fresh()
                out, _ = ragged_paged_attention(
                    fq, fk, fv, fkv, kv_lens_arr, page_indices, cu_q_arr,
                    distribution,
                    use_causal_mask=csl,
                    update_kv_cache=ukv,
                    skip_kv_mask=sm,
                    merge_mixed_seqs=True,
                    compute_size=cs,
                    merged_group_cu_seqs=mcu,
                    m_block_sizes=(cs, bkv_sz, cs, bkv_sz),
                )
                return out

            t = _run_and_time(run_merged, WARMUP_ITERS, PROFILE_ITERS, label)
            results[label] = t

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Summary (median pure TPU time):")
    base = results.get("non-merged [A-baseline]")
    for label, t in results.items():
        if t is None:
            print(f"  {label:48s}  N/A")
            continue
        suffix = ""
        if base and "non-merged" not in label and "[A-baseline]" in label:
            suffix = f"  ({base/t:.2f}x vs non-merged-A)"
        elif base and "non-merged" not in label:
            nm_key = label.split("[")[0].strip() + " [A-baseline]"
            nm_key_alt = "non-merged [A-baseline]"
            # show delta vs same-cs baseline
            cs_base_key = label.split("[")[0].strip() + " [A-baseline]"
            cs_base = results.get(cs_base_key)
            if cs_base and cs_base > 0:
                suffix = f"  ({cs_base/t:.2f}x vs {cs_base_key})"
        print(f"  {label:48s}  {t/1e3:.3f} ms{suffix}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    run_benchmark()
