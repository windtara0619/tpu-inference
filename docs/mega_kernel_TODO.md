# mega_kernel — Open TODOs

Collected from the auto-research sessions (2026-07-13 … 2026-07-15). Numbers are from
kernel_bench on v6e-1 (Qwen3-4B shapes: 32/8 heads, head_dim 128, D=2560, bf16) unless noted.
Current state: prefill RPAm 424 µs (base 253), decode RPAd 746 µs median (base 694),
empty kernel calls ~0.5 µs.

## Tier 1 — clear wins, moderate effort

- [ ] **Hide the decode batch-head bubble (~+54 µs/kernel, the whole remaining decode gap).**
  Batch b+1's Q/KV GEMM chain currently runs exactly when its first seq needs it, stalling the
  cache-read pipe ~13 µs per batch boundary. Move the batch-head compute earlier — trigger it
  during batch b's tail seqs (the x prefetch already fires at `off_in_batch == bq_sz-2`),
  overlapped with their DMA-bound attention. Requires double-buffering `kv_batch_ref`
  (single-slot today; `bq_x2_ref` is already double). Expected: decode ≈ 703 µs ≈ base parity
  while absorbing all projection work.

- [ ] **Decode-only mega mode (`MEGA_KERNEL=decode`).**
  Fusion wins in the memory-bound decode kernel, loses in the compute-bound prefill kernel
  (projection GEMM serializes with attention on the one MXU; real mixed steps pay ~+11.7 ms).
  Fused projections for the DECODE kernel only; XLA projections + plain kernel for prefill.
  Complication: mixed steps need XLA to compute q/k/v for the prefill token range while the
  kernel computes them for decode tokens. Fastest route to net-positive end-to-end.

## Tier 2 — in-kernel prefill efficiency (if prefill fusion stays)

- [ ] **Fuse RMS-norm statistics into the Q-projection GEMM pass** (the XLA epilogue trick seen
  in the baseline HLO dump: one `kOutput` fusion returns `(f32 sum-of-squares, bf16 proj)`).
  Merge `compute_q_proj` + `compute_q_rope_norm` to kill the store→reload→cast round-trip
  through `bq_x2_ref`. ⚠️ Measure carefully: the current split exists for pipelining, and the
  `no_qnorm` ablation showed the pipeline is delicate (removing the norm made the kernel
  65 µs SLOWER — its store pattern is load-bearing for Mosaic).

- [ ] **Share sin/cos between Q and K rope.** Computed twice per tile today
  (`compute_q_rope_norm` and `compute_kv_from_x_tile`); for gap-0 prefill the positions are
  identical. XLA computes one table and feeds both rope fusions. ~10–15 µs.

## Tier 3 — outside the kernel

- [ ] **Fold the output transpose into o_proj.** The per-layer 5D `prepare_outputs` copy
  (`[K,T,...]→[T,...]`, ~27 µs/layer at 4096 tokens, 36×/step, exists in base mode too) might
  vanish if o_proj consumed the kernel's head-major layout directly
  (`einsum("KTnpH,KnpHD->TD")`). Same idea might kill the x-side relayout (`copy.220`).
  Experiments, not sure things — the producer-side x reshape attempt failed (XLA kept the
  `{0,1}` residual layout regardless; don't retry that variant).

- [ ] **int8/fp8 W_qkv and quantized KV cache.** Decode is at the HBM roofline (694 µs ≈
  1.07 GB reads / 1.6 TB/s); the only way below is moving fewer bytes. Halves the 4.2 MB/seq
  cache reads, the ~20 µs w_qkv copy, and speeds the GEMMs. Big feature, touches correctness.

## Investigations / verification

- [ ] **End-to-end profiling rerun** (user runs `MEGA_KERNEL=true python3
  examples/tpu_profiling.py --model Qwen/Qwen3-4B --input-len 1000 --output-len 1000
  --batch-size 256`) to confirm the decode prefetch fix + async w_qkv copy in the real
  workload. Expected vs the 2026_07_14_05_15_30 profile: decode step −0.8 ms from empty-RPAm
  staging plus the RPAd pipeline gain.

- [ ] **Real-workload RPAm gap mystery.** Real mixed-step RPAm is 1498 µs where matched-shape
  benches predict ~850 µs (base inflates too: 553 vs 325). Straddle/misalignment was ruled out
  (gap-1000 vs aligned: same delta). To reproduce, kernel_bench needs a heterogeneous batch
  (e.g. 250 decode seqs + 4 prefill chunk seqs in one call — `build_inputs` currently only
  builds uniform seqs).

- [ ] **Reconstruct `gen_timeline.py` and check it into `scripts/kernel_eval/`.** The
  calibrated event-scheduler that generates `docs/rpa_timeline_v2.html` lived in the session
  scratchpad, which was cleaned. Rebuild from the model constants documented in the diagram
  (T_Q=18.65, T_KV=3.5, T_AQK=5.793, T_APV=6.057, DMA overhead≈1 µs @1.6 TB/s, decode
  batch-head bubble `dma = max(dma, qready-2)`) so the diagram stays regenerable.

## Housekeeping

- [ ] Decide whether to commit or delete `docs/autoresearch_kernel_blog.md` (markdown draft
  superseded by `docs/blog_autoresearch_kernelsage.html`).
- [ ] Commit the staged docs files (blog HTML, rpa_timeline_v2.html).

## Standing caveats (don't lose these)

- **Decode early-prefetch safety** rests on decode seqs owning disjoint writable pages (writes
  go only to each seq's own tail page; prefix-cached shared pages are read-only). If a future
  feature lets two decode seqs in one batch share a writable tail block, the cross-slot
  `_wait_prior_write` must come back for that case (kernel.py ~676, ~1388, ~1450).
- **Never use `kernel_bench.py --ablation/--ablation-file` with uncommitted kernel work** — its
  revert is `git checkout -- kernel.py`. Patch manually with a backup copy instead.
- **Dead ends, don't re-chase**: straddle recompute (delta identical aligned vs misaligned),
  K-norm VPU cost (~0), positions gather (18 µs, dies with the batch-head fix anyway),
  KV packing (~4 µs), producer-side x reshape for layout (no effect on XLA layout assignment).
