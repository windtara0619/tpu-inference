"""Benchmark: merged vs non-merged ragged paged attention.

Setup:
  - 1000 sequences, kv_len == q_len (pure encode: kv_left_frm_cache=0)
  - num_kv_heads=1, head_dim=128
  - Parameters swept: num_q_heads in {1, 4}, q_len in {16, 20}
  - compute_size in {128, 256, 512}

Experiments per config:
  A. default      — VMEM [mq,cs] mask; fill_q_bounds writes 1 row/Q-token; repeat for GQA
  B. no_kv_update — update_kv_cache=False
  C. no_mask      — debug_disable_merged_mask=True  (mask overhead lower bound)

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
NUM_KV_HEADS = 1
HEAD_DIM     = 128
DTYPE        = jnp.bfloat16
PAGE_SIZE    = 128
WARMUP_ITERS  = 3
PROFILE_ITERS = 5

# Parameter sweep
PARAM_CONFIGS = [
    (num_q_heads, q_len)
    for num_q_heads in [1, 4, 8]
    for q_len      in [16, 20]
]
COMPUTE_SIZES = [128, 256, 512]


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


def _parse_top_ops(trace_gz_path, top_n=20):
    """Return top ops by total duration across all non-module threads on TPU."""
    with gzip.open(trace_gz_path, 'rb') as f:
        trace = json.load(f)
    events = trace.get('traceEvents', [])

    tpu_pid = None
    xla_module_tid = None
    thread_names = {}
    for e in events:
        if e.get('ph') == 'M' and e.get('name') == 'process_name':
            if '/device:TPU' in e.get('args', {}).get('name', ''):
                tpu_pid = e['pid']
        if tpu_pid and e.get('ph') == 'M' and e.get('pid') == tpu_pid \
                and e.get('name') == 'thread_name':
            tid = e['tid']
            tname = e.get('args', {}).get('name', '')
            thread_names[tid] = tname
            if 'XLA Modules' in tname:
                xla_module_tid = tid

    if tpu_pid is None:
        return []

    # Aggregate duration per op name across all non-module threads.
    totals = {}
    counts = {}
    for e in events:
        if (e.get('ph') == 'X'
                and e.get('pid') == tpu_pid
                and e.get('tid') != xla_module_tid
                and 'dur' in e):
            name = e.get('name', '<unnamed>')
            totals[name] = totals.get(name, 0) + e['dur']
            counts[name] = counts.get(name, 0) + 1

    ranked = sorted(totals.items(), key=lambda x: -x[1])[:top_n]
    return [(name, totals[name], counts[name]) for name, _ in ranked]


def _profile_and_print_ops(fn, warmup, label, top_n=20):
    """Warm up then print top ops by total duration for one profiler pass,
    broken down per thread (DMA, MXU, VPU, etc.)."""
    for _ in range(warmup):
        fn().block_until_ready()

    trace_dir = tempfile.mkdtemp(prefix='rpa_ops_')
    try:
        jax.profiler.start_trace(trace_dir)
        fn().block_until_ready()
        jax.profiler.stop_trace()

        gz = _find_trace_gz(trace_dir)
        if gz is None:
            print(f"  {label}: trace file not found")
            return

        with gzip.open(gz, 'rb') as f:
            trace = json.load(f)
        all_events = trace.get('traceEvents', [])

        # Find TPU pid and all thread names.
        tpu_pid = None
        thread_names = {}
        for e in all_events:
            if e.get('ph') == 'M' and e.get('name') == 'process_name':
                if '/device:TPU' in e.get('args', {}).get('name', ''):
                    tpu_pid = e['pid']
            if tpu_pid and e.get('ph') == 'M' and e.get('pid') == tpu_pid \
                    and e.get('name') == 'thread_name':
                thread_names[e['tid']] = e.get('args', {}).get('name', '')

        if tpu_pid is None:
            print(f"  {label}: no TPU device found")
            return

        print(f"\n  [{label}]  threads on /device:TPU:0:")
        for tid, tname in sorted(thread_names.items()):
            print(f"    tid={tid}  {tname}")

        # Per-thread top ops.
        for tid, tname in sorted(thread_names.items()):
            thread_events = [e for e in all_events
                             if e.get('ph') == 'X'
                             and e.get('pid') == tpu_pid
                             and e.get('tid') == tid
                             and 'dur' in e]
            if not thread_events:
                continue
            totals, counts = {}, {}
            for e in thread_events:
                n = e.get('name', '<unnamed>')
                totals[n] = totals.get(n, 0) + e['dur']
                counts[n] = counts.get(n, 0) + 1
            ranked = sorted(totals.items(), key=lambda x: -x[1])[:top_n]
            thread_total = sum(totals.values())
            print(f"\n    -- {tname} (tid={tid})  total={thread_total/1e3:.3f}ms --")
            print(f"    {'op name':56s}  {'ms':>7}  {'cnt':>5}  {'avg µs':>7}")
            for n, dur in ranked:
                print(f"    {n[:56]:56s}  {dur/1e3:7.3f}  {counts[n]:5d}  {dur/counts[n]:7.1f}")
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


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
def _run_one_config(num_q_heads, q_len, all_results):
    """Run non-merged and merged benchmarks for one (num_q_heads, q_len) config."""
    num_q_per_kv = num_q_heads // NUM_KV_HEADS
    kv_len = q_len  # pure encode
    bkv_sz = max(128, kv_len)
    tag_cfg = f"nqh={num_q_heads} qlen={q_len}"

    print(f"\n{'='*70}")
    print(f"CONFIG  num_q_heads={num_q_heads}  q_len=kv_len={q_len}  "
          f"GQA={num_q_per_kv}  (pure encode)")
    print(f"{'='*70}")

    (q, k, v, kv_cache, kv_lens_arr, page_indices, cu_q_arr, distribution,
     num_pages, max_num_seq, _) = _build_inputs(
        NUM_SEQS, q_len, kv_len, num_q_heads, NUM_KV_HEADS, HEAD_DIM,
        PAGE_SIZE, DTYPE)

    seq_lens = [(q_len, kv_len)] * NUM_SEQS
    q_host        = np.array(q)
    k_host        = np.array(k)
    v_host        = np.array(v)
    kv_cache_host = np.array(kv_cache)

    def fresh():
        return (jax.device_put(q_host), jax.device_put(k_host),
                jax.device_put(v_host), jax.device_put(kv_cache_host))

    # Non-merged baseline: same block sizes as before (unchanged).
    print(f"\n--- non-merged  bq_csz=128 bkv_csz={bkv_sz} ---")
    for update_kv, exp in [(True, "A"), (False, "B")]:
        label = f"[{tag_cfg}] non-merged [{exp}]"

        def run_nm(ukv=update_kv):
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

        t = _run_and_time(run_nm, WARMUP_ITERS, PROFILE_ITERS, label)
        all_results[label] = t

    # Per compute_size: non-merged with matching block size, then merged variants.
    for compute_size in COMPUTE_SIZES:
        q_limit        = compute_size // num_q_per_kv
        seqs_per_group = q_limit // q_len
        mcu = _build_merged_group_cu_seqs(
            seq_lens, max_num_seq, compute_size, num_q_per_kv)

        cs = compute_size
        # Non-merged matching merged: bkv_sz=bkv_csz=cs, bq_csz=bq_sz=cs//nhd
        # so Q tile=[cs, head_dim] matches merged's compute tile.
        d_bkv_sz  = cs
        d_bkv_csz = cs
        d_bq_csz  = max(1, cs // num_q_per_kv)
        d_bq_sz   = d_bq_csz
        print(f"\n--- cs={cs}  max_q_tokens={q_limit}  ~{seqs_per_group} seqs/group "
              f" nm: bq_csz={d_bq_csz} bkv_csz={d_bkv_csz} ---")

        # Non-merged with the same effective compute block size as merged (experiment D).
        label_d = f"[{tag_cfg}] non-merged-{cs} [D]"
        def run_nm_cs(bsz=(d_bq_sz, d_bkv_sz, d_bq_csz, d_bkv_csz)):
            fq, fk, fv, fkv = fresh()
            out, _ = ragged_paged_attention(
                fq, fk, fv, fkv, kv_lens_arr, page_indices, cu_q_arr,
                distribution,
                use_causal_mask=True,
                update_kv_cache=True,
                merge_mixed_seqs=False,
                m_block_sizes=bsz,
            )
            return out
        t = _run_and_time(run_nm_cs, WARMUP_ITERS, PROFILE_ITERS, label_d)
        all_results[label_d] = t

        for update_kv, no_mask, exp in [
            (True,  False, "A"),
            (False, False, "B-no_kv_update"),
            (True,  True,  "C-no_merged_mask"),
        ]:
            label = f"[{tag_cfg}] merged-{cs} [{exp}]"

            def run_m(cs=cs, mcu=mcu, ukv=update_kv, nmm=no_mask):
                fq, fk, fv, fkv = fresh()
                out, _ = ragged_paged_attention(
                    fq, fk, fv, fkv, kv_lens_arr, page_indices, cu_q_arr,
                    distribution,
                    use_causal_mask=True,
                    update_kv_cache=ukv,
                    debug_disable_merged_mask=nmm,
                    merge_mixed_seqs=True,
                    compute_size=cs,
                    merged_group_cu_seqs=mcu,
                    m_block_sizes=(cs, bkv_sz, cs, bkv_sz),
                )
                return out

            t = _run_and_time(run_m, WARMUP_ITERS, PROFILE_ITERS, label)
            all_results[label] = t


def run_benchmark():
    if not jtu.is_device_tpu_at_least(version=4):
        print("Skipping: requires TPUv4+")
        return

    print(f"\n{'='*70}")
    print(f"RPA v3  merged vs non-merged")
    print(f"  {NUM_SEQS} seqs, num_kv_heads={NUM_KV_HEADS}, "
          f"head_dim={HEAD_DIM}, dtype={DTYPE}, page_size={PAGE_SIZE}")
    print(f"  warmup={WARMUP_ITERS}, profile_iters={PROFILE_ITERS}")
    print(f"  Experiments: A=default  B=no_kv_update  C=no_mask")
    print(f"{'='*70}")

    all_results = {}
    for num_q_heads, q_len in PARAM_CONFIGS:
        _run_one_config(num_q_heads, q_len, all_results)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    W = 110
    print(f"\n{'='*W}")
    print("SUMMARY  (ms — nm=non-merged baseline | nm-cs=non-merged matching cs | mgd=merged [A] | mgd-C=merged [C no-mask])")
    print(f"  speedup ratios vs nm: >1 means variant is faster than nm baseline")
    print(f"  mask-ms = mgd - mgd-C (mask overhead); mask% = mask-ms/mgd*100")
    print(f"{'='*W}")
    hdr = (f"  {'config':22s}  {'cs':>4}  {'seq/grp':>7}  "
           f"{'nm':>6}  {'nm-cs':>6}  {'mgd':>6}  {'mgd-C':>6}  "
           f"{'nm-cs/nm':>9}  {'mgd/nm':>7}  {'mgd-C/nm':>9}  "
           f"{'mask-ms':>8}  {'mask%':>6}")
    print(hdr)
    print(f"  {'-'*22}  {'-'*4}  {'-'*7}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  "
          f"{'-'*9}  {'-'*7}  {'-'*9}  "
          f"{'-'*8}  {'-'*6}")
    for (num_q_heads, q_len) in PARAM_CONFIGS:
        tag_cfg      = f"nqh={num_q_heads} qlen={q_len}"
        num_q_per_kv = num_q_heads // NUM_KV_HEADS
        cfg_label    = f"{tag_cfg} GQA={num_q_per_kv}"
        nm = all_results.get(f"[{tag_cfg}] non-merged [A]")
        nm_str = f"{nm/1e3:.3f}" if nm else "  N/A"
        for i, cs_val in enumerate(COMPUTE_SIZES):
            spg = max(1, (cs_val // num_q_per_kv) // q_len)
            d  = all_results.get(f"[{tag_cfg}] non-merged-{cs_val} [D]")
            ma = all_results.get(f"[{tag_cfg}] merged-{cs_val} [A]")
            mc = all_results.get(f"[{tag_cfg}] merged-{cs_val} [C-no_merged_mask]")
            d_str  = f"{d/1e3:.3f}"  if d  else "  N/A"
            ma_str = f"{ma/1e3:.3f}" if ma else "  N/A"
            mc_str = f"{mc/1e3:.3f}" if mc else "  N/A"
            dnm  = f"{nm/d:.2f}x"  if (nm and d)  else ""
            manm = f"{nm/ma:.2f}x" if (nm and ma) else ""
            mcnm = f"{nm/mc:.2f}x" if (nm and mc) else ""
            if ma and mc:
                mask_ms  = (ma - mc) / 1e3
                mask_pct = (ma - mc) / ma * 100
                mask_ms_str  = f"{mask_ms:.3f}"
                mask_pct_str = f"{mask_pct:.1f}%"
            else:
                mask_ms_str = mask_pct_str = ""
            cfg_col = cfg_label if i == 0 else ""
            nm_col  = nm_str    if i == 0 else ""
            print(f"  {cfg_col:22s}  {cs_val:>4}  {spg:>7}  "
                  f"{nm_col:>6}  {d_str:>6}  {ma_str:>6}  {mc_str:>6}  "
                  f"{dnm:>9}  {manm:>7}  {mcnm:>9}  "
                  f"{mask_ms_str:>8}  {mask_pct_str:>6}")
        print()
    print(f"{'='*W}\n")


def run_op_comparison():
    """Print top ops for non-merged vs merged-128 baseline to find the 0.05ms gap."""
    if not jtu.is_device_tpu_at_least(version=4):
        print("Skipping: requires TPUv4+")
        return

    num_q_per_kv = NUM_Q_HEADS // NUM_KV_HEADS

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
        return (jax.device_put(q_host), jax.device_put(k_host),
                jax.device_put(v_host), jax.device_put(kv_cache_host))

    print(f"\n{'='*70}")
    print("Top-ops comparison: non-merged vs merged-128 [A-baseline]")
    print(f"{'='*70}")

    def run_non_merged():
        fq, fk, fv, fkv = fresh()
        out, _ = ragged_paged_attention(
            fq, fk, fv, fkv, kv_lens_arr, page_indices, cu_q_arr, distribution,
            use_causal_mask=True, update_kv_cache=True, merge_mixed_seqs=False,
            m_block_sizes=(128, bkv_sz, 128, bkv_sz),
        )
        return out

    _profile_and_print_ops(run_non_merged, WARMUP_ITERS,
                           "non-merged [A-baseline]", top_n=25)

    cs  = 128
    mcu = _build_merged_group_cu_seqs(seq_lens, max_num_seq, cs, num_q_per_kv)

    def run_merged():
        fq, fk, fv, fkv = fresh()
        out, _ = ragged_paged_attention(
            fq, fk, fv, fkv, kv_lens_arr, page_indices, cu_q_arr, distribution,
            use_causal_mask=True, update_kv_cache=True, merge_mixed_seqs=True,
            compute_size=cs, merged_group_cu_seqs=mcu,
            m_block_sizes=(cs, bkv_sz, cs, bkv_sz),
        )
        return out

    _profile_and_print_ops(run_merged, WARMUP_ITERS,
                           "merged-128 [A-baseline]", top_n=25)


if __name__ == '__main__':
    run_benchmark()
