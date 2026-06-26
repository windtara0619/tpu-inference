# kernel_eval — Auto-research instructions for Claude

These three scripts form a self-contained loop for evaluating Pallas kernel ideas.
When the user describes a kernel idea, or asks why a kernel is slow, use this workflow
without waiting to be asked step-by-step.

---

## When to engage this workflow

Trigger on any of these:
- User proposes a Pallas/XLA optimization ("what if we fuse X into the attention kernel")
- User asks why a kernel regressed or is slower than expected
- User asks how much a specific step costs inside a kernel
- User asks for a roofline estimate before implementing something

---

## Tool reference

All scripts live in `scripts/kernel_eval/` and must be run from the repo root:

```bash
python scripts/kernel_eval/trace_tool.py <subcommand> [args]
python scripts/kernel_eval/kernel_bench.py [args]
python scripts/kernel_eval/idea_check.py <subcommand> [args]
```

---

## Step-by-step auto-research protocol

### 0. Gather context first

Before running anything, answer these questions from the code or conversation:
- What kernel family? (e.g. `RPAm-p_128-bq_256_128-bkv_1024_512`)
- What serving shape? (q_len, kv_len, num_q_heads, num_kv_heads, head_dim, page_size)
- Are there existing traces to start from? If so, run Step 1. If not, skip to Step 2.

---

### 1. Analyse existing traces (if available)

```bash
# Full delta between two configurations
python scripts/kernel_eval/trace_tool.py compare \
    --base <path_to_base_trace> \
    --exp  <path_to_exp_trace>  \
    --category custom-call

# Drill into ops immediately before/after the regressed kernel call
python scripts/kernel_eval/trace_tool.py window \
    --trace <path> \
    --landmark "<kernel_name>.<instance_number>" \
    --before 500 --after 800

# Attribute costs to a specific Python source file
python scripts/kernel_eval/trace_tool.py source \
    --trace <path> --source <filename_substring>
```

**What to extract:** baseline mean µs, experiment mean µs, any secondary regressions
in nearby ops (look for prefetch / DMA ops shifting after the landmark).

---

### 2. Estimate DMA and FLOPs for the idea

Before writing any kernel code, quantify the idea as three numbers.
Use the tensor shapes from Step 0:

```
bytes_q  = q_len * num_q_heads * head_dim * 2     # bfloat16
bytes_k  = q_len * num_kv_heads * head_dim * 2
bytes_sincos = q_len * head_dim / 2 * 2           # half head_dim, bfloat16
```

Write down:
- **DMA saved**: which HBM reads/writes disappear in the proposed path
- **DMA added**: any new loads/stores the proposed path requires
- **FLOPs added**: extra arithmetic (typically rotation ops: 6 muls + 4 adds per element)

---

### 3. Roofline feasibility check

```bash
python scripts/kernel_eval/idea_check.py evaluate \
    --name "<idea_name>" \
    --hw tpuv6e \
    --baseline-dma  "<label=MB,label=MB,...>" \
    --baseline-flops <float> \
    --baseline-time  <measured_µs or 0 if unknown> \
    --proposed-dma  "<label=MB,...>" \
    --proposed-flops <float> \
    --proposed-time  0
```

**Decision rule:**
- `WORTH EXPLORING` → proceed to Step 4
- `LIKELY NOT WORTH IT` → report the roofline math to the user and stop; do not
  spend time prototyping something the model says cannot win

**Red flags to report immediately:**
- Proposed path flips from memory-bound to compute-bound with more FLOPs than headroom
- `proposed_dma > baseline_dma` (the idea adds, not removes, HBM traffic)
- Rotation placed inside a nested loop (multiply element count by loop trip count)

---

### 4. Run kernel microbenchmarks

Always run `no_rope` baseline first, then `rope_fused`, sequentially (never parallel —
two JAX processes on the same TPU die with a lockfile error).

```bash
mkdir -p /tmp/kernel_eval_results

python scripts/kernel_eval/kernel_bench.py \
    --q-len <Q> --kv-len <KV> \
    --num-q-heads <NQ> --num-kv-heads <NKV> --head-dim <D> \
    --page-size <P> --block-sizes <bq,bkv,bq_c,bkv_c> \
    --tag no_rope --json-out /tmp/kernel_eval_results/no_rope.json

python scripts/kernel_eval/kernel_bench.py \
    --q-len <Q> --kv-len <KV> \
    --num-q-heads <NQ> --num-kv-heads <NKV> --head-dim <D> \
    --page-size <P> --block-sizes <bq,bkv,bq_c,bkv_c> \
    --has-rope \
    --tag rope_fused --json-out /tmp/kernel_eval_results/rope_fused.json
```

Default serving shape (Llama-3 70B on v6e-1 prefill):
`--q-len 2048 --kv-len 2048 --num-q-heads 32 --num-kv-heads 8 --head-dim 128 --page-size 128 --block-sizes 256,1024,128,512`

---

### 5. Run ablations to attribute component costs

Run these sequentially, one at a time:

```bash
for ablation in skip_sincos skip_apply_q skip_apply_k; do
    python scripts/kernel_eval/kernel_bench.py \
        --q-len <Q> --kv-len <KV> \
        --num-q-heads <NQ> --num-kv-heads <NKV> --head-dim <D> \
        --page-size <P> --block-sizes <bq,bkv,bq_c,bkv_c> \
        --has-rope \
        --ablation $ablation \
        --tag $ablation --json-out /tmp/kernel_eval_results/${ablation}.json
done
```

If the user's idea requires a new ablation not in the built-in list, write a JSON patch
and use `--ablation-file`:

```json
{"description": "...", "old": "<exact text from kernel.py>", "new": "<replacement>"}
```

After each `apply_ablation`, the kernel is auto-reverted. Verify with
`git diff tpu_inference/kernels/ragged_paged_attention/v3/kernel.py` — it should be empty.

---

### 6. Print the summary table

```bash
python scripts/kernel_eval/idea_check.py ablation-summary \
    --baseline  /tmp/kernel_eval_results/no_rope.json \
    --fused     /tmp/kernel_eval_results/rope_fused.json \
    --ablations /tmp/kernel_eval_results/skip_sincos.json \
                /tmp/kernel_eval_results/skip_apply_q.json \
                /tmp/kernel_eval_results/skip_apply_k.json \
    --kernel-family "RPAm-p_128-bq_256_128-bkv_1024_512"
```

---

### 7. Report findings

Summarise in this order:
1. **Verdict**: did the idea win or lose, by how many µs?
2. **Roofline prediction vs actual**: was the model accurate? Why or why not?
3. **Dominant cost component** from the ablation table
4. **Root cause** (which loop-level decision drove the cost — be specific about line numbers)
5. **Proposed fix** or next experiment if the idea is salvageable

---

## Common pitfalls

| Pitfall | What happens | How to avoid |
|---|---|---|
| Running two JAX procs in parallel | ABORTED: libtpu lockfile | Always run benchmarks sequentially |
| `jnp.concatenate` in rotate_half for odd shapes | XLA fusion_emitter crash | Use `jnp.stack(...).reshape(x.shape)` instead |
| Ablation pattern not found | `RuntimeError` from `apply_ablation` | Read the current kernel.py before writing the patch; strings must match exactly |
| Comparing traces from different token lengths | Apparent 8–12× gap that is pure scale | Always benchmark at the serving shape (2048 tokens) |
| Assuming XLA CSEs across opaque pallas calls | Incorrect cost accounting | Each `tpu_custom_call` is opaque; no cross-kernel CSE |

---

## Known baseline numbers (v6e-1, q_len=2048, fresh prefill)

| Config | RPAm mean (µs) |
|---|---|
| no_rope (FALSE) | 252.93 |
| rope_fused (TRUE) | 448.23 |
| skip_sincos ablation | 319.10 |
| skip_apply_q ablation | 325.20 |
| skip_apply_k ablation | 367.87 |

Standalone apply_rope costs at 2048 tokens (XLA-compiled, FALSE path):
- apply_rope(q): ~110.5 µs
- apply_rope(k): ~32.4 µs
