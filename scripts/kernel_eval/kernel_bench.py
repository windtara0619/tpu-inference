#!/usr/bin/env python3
"""
kernel_bench.py -- Microbenchmark for the RPA Pallas kernel at configurable shapes.

Builds realistic kernel inputs, compiles, warms up, then runs under
jax.profiler to get accurate device-side per-custom-call timings.  Supports
inline ablations (edit kernel.py before the run, auto-revert after).

Usage examples
--------------
  # Baseline: no RoPE fusion
  python kernel_bench.py --q-len 2048 --kv-len 2048 --tag no_rope

  # RoPE fused, full 2048-token fresh prefill
  python kernel_bench.py --q-len 2048 --kv-len 2048 --has-rope --tag rope_fused

  # Ablation: skip sin/cos transcendentals
  python kernel_bench.py --q-len 2048 --kv-len 2048 --has-rope \\
      --ablation skip_sincos --tag skip_sincos

Built-in ablations (--ablation NAME)
-------------------------------------
  skip_sincos   Replace jnp.sin/cos with zeros/ones (measures sin/cos compute cost)
  skip_apply_q  Skip Q rotation body in load_bq        (measures Q rotation cost)
  skip_apply_k  Skip rotate_inplace_bkv_k call         (measures K rotation+writeback cost)

Custom ablations
----------------
  --ablation-file path/to/ablation.json
  JSON format: [{"old": "...", "new": "...", "description": "..."}]
"""

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Built-in ablations for the RPA kernel
# ---------------------------------------------------------------------------

KERNEL_PATH = Path(__file__).parents[2] / \
    "tpu_inference/kernels/ragged_paged_attention/v3/kernel.py"

BUILTIN_ABLATIONS = {
    "skip_sincos": {
        "description": "Replace sin/cos transcendentals with zeros/ones to measure their compute cost",
        "old": (
            "        sinusoid_inp = positions.astype(jnp.float32) * timescale\n"
            "        sin = jnp.sin(sinusoid_inp).astype(dtype)\n"
            "        cos = jnp.cos(sinusoid_inp).astype(dtype)"
        ),
        "new": (
            "        sinusoid_inp = positions.astype(jnp.float32) * timescale\n"
            "        sin = jnp.zeros_like(sinusoid_inp, dtype=dtype)  # ABLATION\n"
            "        cos = jnp.ones_like(sinusoid_inp, dtype=dtype)   # ABLATION"
        ),
    },
    "skip_apply_q": {
        "description": "Skip Q rotation in load_bq to measure its cost (sin/cos still computed)",
        "old":  "        q_flat = strided_load(q_ref, load_start, load_sz, 1, dtype=q_dtype)\n        if has_rope:",
        "new":  "        q_flat = strided_load(q_ref, load_start, load_sz, 1, dtype=q_dtype)\n        if False and has_rope:  # ABLATION",
    },
    "skip_apply_k": {
        "description": "Skip rotate_inplace_bkv_k to measure K rotation+writeback cost",
        "old":  "                if has_rope:\n                    # Rotate the \"new\" (not-yet-cached) k tokens in-place,",
        "new":  "                if False and has_rope:  # ABLATION\n                    # Rotate the \"new\" (not-yet-cached) k tokens in-place,",
    },
}


def apply_ablation(spec: dict) -> None:
    text = KERNEL_PATH.read_text()
    if spec["old"] not in text:
        raise RuntimeError(f"Ablation pattern not found in {KERNEL_PATH}:\n{spec['old'][:80]}")
    KERNEL_PATH.write_text(text.replace(spec["old"], spec["new"], 1))


def revert_kernel() -> None:
    subprocess.run(["git", "checkout", "--", str(KERNEL_PATH)], check=True,
                   capture_output=True)


# ---------------------------------------------------------------------------
# Kernel input construction
# ---------------------------------------------------------------------------

def build_inputs(q_len, kv_len, num_q_heads, num_kv_heads, head_dim, page_size, dtype_str,
                 num_seqs=1):
    """
    Build kernel inputs for a batch of `num_seqs` sequences, each with
    q_len/num_seqs query tokens and kv_len/num_seqs KV tokens (fresh prefill).
    q_len and kv_len must be divisible by num_seqs.
    """
    import jax.numpy as jnp
    import numpy as np
    from tpu_inference.kernels.ragged_paged_attention.v3.util import (
        align_to, cdiv, get_dtype_packing)

    assert q_len % num_seqs == 0, f"q_len {q_len} must be divisible by num_seqs {num_seqs}"
    assert kv_len % num_seqs == 0, f"kv_len {kv_len} must be divisible by num_seqs {num_seqs}"

    seq_q_len  = q_len  // num_seqs
    seq_kv_len = kv_len // num_seqs

    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[dtype_str]
    rng = np.random.default_rng(1234)

    def gen(shape):
        return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)

    max_tokens = align_to(q_len, 128)
    max_num_seq = max(8, num_seqs + 1)
    pages_per_seq = cdiv(seq_kv_len, page_size)
    total_pages = pages_per_seq * num_seqs
    num_pages = max(64, total_pages + 8)

    kv_packing = get_dtype_packing(dtype)
    padded_head = align_to(head_dim, 128)
    num_kv_x2 = align_to(num_kv_heads * 2, kv_packing)

    q = gen((max_tokens, num_q_heads, head_dim))
    k = gen((max_tokens, num_kv_heads, head_dim))
    v = gen((max_tokens, num_kv_heads, head_dim))

    # Build KV cache: num_seqs independent caches
    kv_raw_seq = gen((seq_kv_len, num_kv_x2 // kv_packing, kv_packing, padded_head))
    kv_padded = jnp.pad(kv_raw_seq,
                        ((0, pages_per_seq * page_size - seq_kv_len), (0,0), (0,0), (0,0)))
    kv_pages = kv_padded.reshape(-1, page_size, num_kv_x2 // kv_packing, kv_packing, padded_head)
    kv_cache = jnp.pad(
        jnp.tile(kv_pages, (num_seqs, 1, 1, 1, 1)),
        ((0, num_pages - total_pages), (0,0),(0,0),(0,0),(0,0)))

    # Page indices: each sequence gets its own contiguous pages
    page_idx_rows = []
    for s in range(num_seqs):
        row = jnp.arange(s * pages_per_seq, (s + 1) * pages_per_seq, dtype=jnp.int32)
        page_idx_rows.append(row)
    page_indices_2d = jnp.stack(page_idx_rows, axis=0)  # [num_seqs, pages_per_seq]
    page_indices = jnp.pad(page_indices_2d,
                           ((0, max_num_seq - num_seqs), (0, 0))).reshape(-1)

    # cu_q: cumulative query lengths [0, seq_q, 2*seq_q, ...]
    cu_q_vals = jnp.array([i * seq_q_len for i in range(num_seqs + 1)], dtype=jnp.int32)
    cu_q = jnp.pad(cu_q_vals, (0, max_num_seq + 1 - (num_seqs + 1)))

    # kv_lens: each sequence has seq_kv_len tokens
    kv_lens = jnp.pad(jnp.array([seq_kv_len] * num_seqs, dtype=jnp.int32),
                      (0, max_num_seq - num_seqs))

    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    return q, k, v, kv_cache, kv_lens, page_indices, cu_q, distribution


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def run_and_profile(fn, fn_args, n_reps, trace_dir) -> dict:
    """Run fn(*fn_args) n_reps times under jax.profiler; return per-family stats."""
    import jax

    if os.path.exists(trace_dir):
        shutil.rmtree(trace_dir)

    with jax.profiler.trace(trace_dir):
        for _ in range(n_reps):
            out = fn(*fn_args)
        jax.block_until_ready(out)

    # find trace file (may be nested under plugins/profile/... on TPU)
    trace_file = None
    for suffix in ["*.trace.json.gz", "trace.json"]:
        matches = sorted(Path(trace_dir).rglob(suffix))
        if matches:
            trace_file = matches[0]
            break
    if trace_file is None:
        raise FileNotFoundError(f"No trace found in {trace_dir}")

    opener = gzip.open if trace_file.suffix == ".gz" else open
    with opener(trace_file, "rt") as fh:
        data = json.load(fh)
    events = data.get("traceEvents", [])

    by_family = defaultdict(list)
    for e in events:
        a = e.get("args", {})
        if not isinstance(a, dict):
            continue
        if a.get("hlo_category") != "custom-call":
            continue
        name = e.get("name", "")
        if not (name.startswith("RPAm") or name.startswith("RPAd")):
            continue
        family = re.sub(r"\.\d+$", "", name)
        by_family[family].append(e.get("dur", 0))

    result = {}
    for family, durs in by_family.items():
        d = np.array(durs)
        result[family] = {
            "n": len(d),
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "std": float(d.std()),
            "min": float(d.min()),
            "max": float(d.max()),
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--q-len", type=int, default=2048)
    p.add_argument("--kv-len", type=int, default=2048,
                   help="kv_len == q_len means a fresh prefill (all tokens new)")
    p.add_argument("--num-q-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=128)
    p.add_argument("--block-sizes", default="256,1024,128,512",
                   help="bq_sz,bkv_sz,bq_csz,bkv_csz  (default matches p_128-bq_256_128-bkv_1024_512)")
    p.add_argument("--num-seqs", type=int, default=1,
                   help="Number of sequences in the batch (q_len and kv_len split equally)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--rope-theta", type=float, default=10000.0,
                   help="RoPE theta (used when --mega-kernel is set; rope is always applied with projections)")
    p.add_argument("--mega-kernel", action="store_true", default=False,
                   help="Fuse Q+KV projection into kernel (requires --hidden-size)")
    p.add_argument("--hidden-size", type=int, default=0,
                   help="Hidden dim for fused Q/KV projection (e.g. 2560 for Qwen3-4B)")
    p.add_argument("--vmem-limit-mb", type=int, default=100)
    p.add_argument("--n-reps", type=int, default=20)
    p.add_argument("--ablation", choices=list(BUILTIN_ABLATIONS), default=None,
                   help="Built-in kernel ablation to apply before benchmarking")
    p.add_argument("--ablation-file",
                   help="JSON file with custom ablation specs [{old,new,description}]")
    p.add_argument("--tag", default="bench", help="Label for trace directory and output")
    p.add_argument("--trace-dir", default="/tmp/rpa_bench_{tag}")
    p.add_argument("--json-out", help="Write results as JSON to this file")
    args = p.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from tpu_inference.kernels.ragged_paged_attention.v3.kernel import ragged_paged_attention

    bq_sz, bkv_sz, bq_csz, bkv_csz = [int(x) for x in args.block_sizes.split(",")]
    trace_dir = args.trace_dir.replace("{tag}", args.tag)

    seq_q = args.q_len // args.num_seqs
    print(f"Building inputs: {args.num_seqs} seq × {seq_q} tokens = {args.q_len} total, "
          f"heads=({args.num_q_heads},{args.num_kv_heads}), head_dim={args.head_dim}, "
          f"page_size={args.page_size}, dtype={args.dtype}", flush=True)

    q, k, v, kv_cache, kv_lens, page_indices, cu_q, distribution = build_inputs(
        args.q_len, args.kv_len, args.num_q_heads, args.num_kv_heads,
        args.head_dim, args.page_size, args.dtype, num_seqs=args.num_seqs)

    fn_kwargs = dict(
        rope_theta=args.rope_theta,
        m_block_sizes=(bq_sz, bkv_sz, bq_csz, bkv_csz),
        vmem_limit_bytes=args.vmem_limit_mb * 1024 * 1024,
    )

    if args.mega_kernel:
        if args.hidden_size <= 0:
            raise ValueError("--hidden-size must be set when using --mega-kernel")
        if bkv_sz % bq_sz != 0:
            raise ValueError(f"--mega-kernel requires bkv_sz % bq_sz == 0, got {bkv_sz} % {bq_sz} != 0")
        dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[args.dtype]
        rng = np.random.default_rng(9999)
        max_tokens = q.shape[0]
        D, H = args.hidden_size, args.head_dim
        x   = jnp.array(rng.random((max_tokens, D), dtype=np.float32)).astype(dtype)
        wq  = jnp.array(rng.random((D, args.num_q_heads * H), dtype=np.float32)).astype(dtype)
        qns = jnp.array(rng.random((H,), dtype=np.float32)).astype(dtype)
        wk  = jnp.array(rng.random((D, args.num_kv_heads * H), dtype=np.float32)).astype(dtype)
        kns = jnp.array(rng.random((H,), dtype=np.float32)).astype(dtype)
        wv  = jnp.array(rng.random((D, args.num_kv_heads * H), dtype=np.float32)).astype(dtype)
        fn_kwargs["mega_kernel"] = True
        fn_kwargs["qn_scale"] = qns
        # Pass wk, kns, wv as fn args (NOT in fn_kwargs) so they are treated as
        # dynamic inputs, not compile-time constants.  Closing over large JAX
        # arrays in fn_kwargs embeds them as jaxpr literals, causing XLA to
        # constant-fold 10MB+ of weights, which hangs for minutes.

        @jax.jit
        def fn(q, k, v, kv_cache, kv_lens, page_indices, cu_q, dist,
               x, wq, wk, kns, wv):
            return ragged_paged_attention(q, k, v, kv_cache, kv_lens, page_indices,
                                         cu_q, dist, x=x, w_q=wq,
                                         w_k=wk, kn_scale=kns, w_v=wv,
                                         **fn_kwargs)

        fn_args = (q, k, v, kv_cache, kv_lens, page_indices, cu_q, distribution,
                   x, wq, wk, kns, wv)
    else:
        @jax.jit
        def fn(q, k, v, kv_cache, kv_lens, page_indices, cu_q, dist):
            return ragged_paged_attention(q, k, v, kv_cache, kv_lens, page_indices,
                                         cu_q, dist, **fn_kwargs)

        fn_args = (q, k, v, kv_cache, kv_lens, page_indices, cu_q, distribution)

    ablation_spec = None
    if args.ablation:
        ablation_spec = BUILTIN_ABLATIONS[args.ablation]
    elif args.ablation_file:
        with open(args.ablation_file) as fh:
            ablation_spec = json.load(fh)

    if ablation_spec:
        print(f"Applying ablation: {ablation_spec.get('description','')}", flush=True)
        apply_ablation(ablation_spec)

    try:
        print("Compiling...", flush=True)
        jax.block_until_ready(fn(*fn_args))

        print(f"Profiling {args.n_reps} reps → {trace_dir}", flush=True)
        stats = run_and_profile(fn, fn_args, args.n_reps, trace_dir)
    finally:
        if ablation_spec:
            revert_kernel()
            print("Kernel reverted.", flush=True)

    print(f"\n{'op family':<55}  {'n':>4}  {'mean µs':>10}  {'median µs':>10}  {'std µs':>8}")
    print("-" * 95)
    for family, s in sorted(stats.items()):
        print(f"{family:<55}  {s['n']:>4}  {s['mean']:>10.3f}  {s['median']:>10.3f}  {s['std']:>8.3f}")

    result = {
        "tag": args.tag,
        "q_len": args.q_len,
        "kv_len": args.kv_len,
        "mega_kernel": args.mega_kernel,
        "ablation": args.ablation,
        "stats": stats,
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nJSON written to {args.json_out}")

    return result


if __name__ == "__main__":
    main()
