# Pallas Kernel Development Workflow

A self-contained toolbox for proposing, evaluating, and measuring Pallas kernel
ideas against the XLA-compiled baseline.  Every step produces a JSON artefact
that the next step can consume.

---

## Tool Overview

| Script | What it does |
|---|---|
| `trace_tool.py` | Analyse jax.profiler traces: per-family stats, two-trace delta, time-window dump, source attribution |
| `kernel_bench.py` | Microbenchmark the RPA kernel at configurable shapes; supports built-in and custom ablations |
| `idea_check.py` | Roofline feasibility check before writing code; ablation-summary table after |

---

## The Flow

```
1. UNDERSTAND THE BASELINE
        │
        ▼
2. PROPOSE THE IDEA  →  3. FEASIBILITY CHECK  →  (if positive)
                                                        │
                                          4. PROTOTYPE  +  ABLATIONS
                                                        │
                                          5. SUMMARISE FINDINGS
```

---

## Step 1 — Understand the baseline

Capture two profiler traces (one per configuration) and compare them:

```bash
# Start vLLM server with FUSE_ROPE_INTO_ATTN_KERNEL=false, profile, stop.
# Repeat with =true.  (Or use kernel_bench.py for standalone kernel traces.)

python trace_tool.py compare \
    --base /tmp/trace_false/trace.json \
    --exp  /tmp/trace_true/trace.json  \
    --pattern "^RPAm|^RPAd" --category custom-call
```

Drill into the time window around a specific kernel call to see what changed
before and after:

```bash
python trace_tool.py window \
    --trace /tmp/trace_true/trace.json \
    --landmark "RPAm-p_128-bq_256_128-bkv_1024_512.72" \
    --before 500 --after 800
```

Attribute ops to a specific Python source file:

```bash
python trace_tool.py source \
    --trace /tmp/trace_false/trace.json \
    --source rope_interface
```

---

## Step 2 — Propose an idea

Write it down as three numbers:

1. **DMA saved** — which tensor reads/writes disappear?
2. **DMA added** — which new loads/stores appear?
3. **FLOPs added** — what extra compute does the new path do?

Example for "fuse RoPE into the attention kernel":

| | Baseline (FALSE) | Proposed (TRUE) |
|---|---|---|
| apply_rope(q) HBM | Q read 16.8 MB + write 16.8 MB = 33.6 MB | eliminated (Q loaded once into VMEM) |
| apply_rope(k) HBM | K read 4.2 MB + write 4.2 MB = 8.4 MB | eliminated |
| Rotation FLOPs | ~2.3 GFLOP (standalone fused op) | ~2.7 GFLOP (inside kernel, same math) |

---

## Step 3 — Feasibility check (before writing any code)

```bash
python idea_check.py evaluate \
    --name "fuse_rope_into_rpa" \
    --hw tpuv6e \
    --baseline-dma  "Q_read=16.8,Q_write=16.8,K_read=4.2,K_write=4.2,sincos_write=1.05" \
    --baseline-flops 2.3e9 \
    --baseline-time  320 \
    --proposed-dma  "Q_read=16.8,K_read=4.2" \
    --proposed-flops 2.7e9
```

This prints a roofline table.  If `proposed roofline time < baseline roofline time`,
the idea has a ceiling for improvement and is worth pursuing.

**Critical questions the check surfaces:**

- Is the baseline memory-bound or compute-bound?
  If compute-bound already, saving DMA bytes won't help much.
- Does the proposed path flip from memory-bound to compute-bound?
  If so, the new FLOPs added must stay below the compute roofline.
- What is the arithmetic intensity ridge point?
  Operations below it are memory-bound; above it, compute-bound.

**Red flags:**

- `proposed_dma > baseline_dma` — the "optimisation" adds more HBM traffic.
- `proposed_flops` large enough to push past the MXU roofline.
- Proposed path requires the rotation inside a nested loop (redundant work).

---

## Step 4 — Prototype and ablate

### 4a. Establish baselines with `kernel_bench.py`

```bash
# No-rope baseline
python kernel_bench.py \
    --q-len 2048 --kv-len 2048 \
    --num-q-heads 32 --num-kv-heads 8 --head-dim 128 \
    --page-size 128 --block-sizes 256,1024,128,512 \
    --tag no_rope --json-out results/no_rope.json

# Fused rope (proposed)
python kernel_bench.py \
    --q-len 2048 --kv-len 2048 \
    --has-rope \
    --tag rope_fused --json-out results/rope_fused.json
```

### 4b. Run ablations to attribute costs to components

```bash
for ablation in skip_sincos skip_apply_q skip_apply_k; do
    python kernel_bench.py \
        --q-len 2048 --kv-len 2048 --has-rope \
        --ablation $ablation \
        --tag $ablation --json-out results/${ablation}.json
done
```

Built-in ablations:

| `--ablation` | What is stubbed | Measures |
|---|---|---|
| `skip_sincos` | `jnp.sin/cos` → zeros/ones | Cost of sin/cos transcendentals |
| `skip_apply_q` | Q rotation body in `load_bq` | Cost of applying rotation to Q |
| `skip_apply_k` | `rotate_inplace_bkv_k` call | Cost of K rotation + VMEM writeback |

Custom ablation (JSON file):

```json
{
  "description": "Skip the Q rotation broadcast step",
  "old": "        q_3d = jnp.concatenate([q_first_rot, q_second_rot], axis=-1)",
  "new": "        q_3d = jnp.concatenate([q_first,     q_second],     axis=-1)  # ABLATION"
}
```

```bash
python kernel_bench.py --has-rope \
    --ablation-file my_ablation.json \
    --tag my_ablation --json-out results/my_ablation.json
```

### 4c. Summarise component breakdown

```bash
python idea_check.py ablation-summary \
    --baseline   results/no_rope.json \
    --fused      results/rope_fused.json \
    --ablations  results/skip_sincos.json \
                 results/skip_apply_q.json \
                 results/skip_apply_k.json \
    --kernel-family "RPAm-p_128-bq_256_128-bkv_1024_512"
```

Output:
```
=== Ablation summary for RPAm-p_128-bq_256_128-bkv_1024_512 ===
Baseline (no rope):  252.93 µs
Fused (has_rope):    448.23 µs
Total rope overhead: +195.30 µs

Component                       w/ ablation (µs)    cost (µs)  % of total
-----------------------------------------------------------------------
sin/cos compute                           319.10     +129.13       66.1%
apply rope to Q                           325.20     +123.03       63.0%
apply rope to K + writeback               367.87      +80.36       41.1%
```

---

## Step 5 — Summarise findings

After running the ablations, fill in the experiment template:

```
## Experiment: <idea name>
Date: YYYY-MM-DD

### Hypothesis
<What DMA is being saved, what compute is being added, why the roofline
predicts an improvement.>

### Roofline prediction
Baseline: <t_pred> µs (<bound>-bound, <intensity> FLOP/B)
Proposed: <t_pred> µs (<bound>-bound)
Expected improvement: <delta> µs

### Measurements
| Config | RPAm mean (µs) | Δ vs baseline |
|---|---|---|
| no_rope  | 252.93 | — |
| rope_fused | 448.23 | +195.30 |
| skip_sincos | 319.10 | kernel saves 129.13 |
| skip_apply_q | 325.20 | kernel saves 123.03 |
| skip_apply_k | 367.87 | kernel saves 80.36 |

### Root cause (if negative result)
<Which loop-level decision made the compute cost exceed the DMA savings.>

### Next steps
<Either fix the root cause, or conclude the idea is not viable.>
```

---

## What we learned from the RoPE fusion investigation

This workflow was developed during the investigation of `FUSE_ROPE_INTO_ATTN_KERNEL`.

**The idea** was correct: eliminate the apply_rope(q/k) HBM round-trip (33.6 + 8.4 = 42 MB,
~143 µs at 2048 tokens on v6e-1) by rotating Q and K inside the kernel while they're
already in VMEM.

**What went wrong** — three loop-placement decisions that each exceeded the DMA saving:

| Root cause | Where in kernel.py | Extra cost |
|---|---|---|
| Q rotation inside bkv loop (4× redundant) | `load_bq` at innermost loop level | 123 µs |
| K rotation of full bkv tile × num_bq times | `rotate_inplace_bkv_k` called 16× | 80 µs |
| sin/cos recomputed per bq tile (no XLA CSE) | `load_rope_sincos` 16× per layer | 129 µs (overlaps) |

Total overhead: +195 µs vs 143 µs saved → **net −52 µs regression per call**, plus
a secondary −8.36 µs/layer in the MLP cluster from VMEM pressure caused by the
longer attention kernel shifting the gate_proj prefetch into fusion.255's execution window.

**The fix** would be:
1. Hoist Q rotation above the bkv loop (rotate once per bq tile, buffer the result).
2. Rotate K only for new tokens (not full bkv tiles), or rotate once per bkv tile
   (not once per bq × bkv).
3. Accept that sin/cos will be recomputed per bq tile (unavoidable inside the kernel),
   but ensure it's not the bottleneck by keeping each call large (one call per bq tile,
   not per bq_chunk × kv_head).
