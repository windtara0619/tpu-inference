# How We Cut a Fused TPU Attention Kernel from 545 µs to 424 µs with an Auto-Research Loop

*Optimizing a Pallas TPU kernel usually looks like this: stare at a profile, form a theory, rewrite some code, re-profile, repeat. Each iteration costs hours and most theories are wrong. For our fused attention kernel we built something better: an auto-research loop that benchmarks every unit operation, reconstructs the kernel's internal pipeline as a calibrated timeline diagram, and lets the diagram tell us where the bubbles are. This post walks through the method and the 22% latency win it produced.*

## The kernel

Our ragged paged attention (RPA) kernel serves mixed prefill/decode batches for LLM inference on TPU. The experiment: fuse the **QKV projection, RMS-norm, and RoPE** — normally separate XLA ops between transformer layers — directly into the attention kernel (`MEGA_KERNEL`). The payoff on paper is fewer HBM round-trips for Q/K/V and less per-layer XLA overhead. The first working version told a different story:

| | attention only (baseline) | first fused version |
|---|---|---|
| RPA prefill kernel, 2048 tokens, v6e-1 | 253 µs | **545 µs** |

The fusion added 292 µs to absorb work that XLA did in ~200 µs on the same chip. Somewhere inside the kernel, time was leaking. But a Pallas kernel is opaque to the profiler — the trace shows one 545 µs custom-call block and nothing inside it.

## Step 1: benchmark every unit op

If the profiler can't see inside the kernel, build the picture from parts. `op_bench.py` measures each primitive the kernel is made of, at the kernel's exact shapes:

- the Q projection GEMM (`[256,2560] @ [2560,4096]`, bf16 → 18.65 µs)
- the KV projection GEMM per tile (3.5 µs)
- Q RMS-norm + RoPE on the VPU (6.35 µs), K-side (3.3 µs)
- sin/cos generation, the packed KV store, the output norm
- every DMA: x tile fetch (1.4 µs), KV-cache tile read/write (2.7 µs), the 31 MB fused weight load (19.6 µs)
- the attention inner loop itself: QK matmul 5.79 µs and PV matmul 6.06 µs per 512-token chunk, with the softmax accumulation on the VPU (1.14 µs) hiding under PV

These constants are the vocabulary. Alone they don't explain 545 µs — summing them gives far less. The gap between "sum of parts" and "measured whole" *is* the optimization opportunity: it's all stalls.

## Step 2: reconstruct the pipeline as a diagram

The TPU core has three resources that can run concurrently: the **MXU** (matrix unit), the **VPU** (vector unit), and the **DMA engines**. A kernel is fast exactly when all three are busy at once. So we wrote a small event scheduler that replays the kernel's program order — every GEMM, every norm, every DMA, with their true dependencies — using the unit-op constants, and renders it as an interactive timeline (`docs/rpa_timeline.html`): one swimlane per resource, one hoverable block per op.

The critical discipline: **the model's total must match the measured total.** Our first fused model summed to 545 µs against a 545 µs measurement — which meant the model's stalls were the kernel's stalls, and we could trust what the picture showed:

- The MXU lane had gaps after every Q projection: the GEMM finished, then the MXU **sat idle while the VPU ran the norm+RoPE**, because attention couldn't start until the normed Q was stored.
- K/V for a whole 1024-token block was computed in one shot from a *separate* x fetch, stalling attention behind a second DMA and a 13 µs GEMM + 13 µs VPU norm, serially.
- The same x tile was fetched twice — once for the Q projection, once for the KV projection.

None of this is visible in a profiler trace. All of it is obvious in the diagram.

## Step 3: fix a bubble, re-measure, re-calibrate

Each fix follows the same loop: roofline-check the idea (`idea_check.py` — if the model says it can't win, don't prototype it), patch the kernel, benchmark (`kernel_bench.py`), and re-fit the diagram. Some of the wins, in the order we landed them:

**Overlap the Q projection with attention (−28 µs).** Issue tile *t+1*'s projection GEMM before tile *t*'s attention loops. The MXU processes them back-to-back while the VPU norms tile *t+1* in the shadow of the attention matmuls. The diagram's MXU lane closed up; the VPU lane went from bursty to continuously occupied.

**Compute K/V per bq tile from the already-resident x (removes a DMA + dedups the fetch).** Instead of a separate 1024-row KV pass with its own x fetch, each 256-row x tile — already in VMEM for the Q projection — produces its K/V rows, which are written eagerly to the KV cache; later tiles re-read them as ordinary cached tokens. One x load instead of two, and the 3.5 µs KV GEMM tucks between attention loops instead of damming them.

**Fuse the weights once, at weight-load time.** The kernel wants `[W_q | W_k | W_v]` as one flat matrix. Concatenating it inside the jitted step made XLA re-materialize ~30 MB per layer per step. Building it once after checkpoint load turned that into a pass-through operand. (Found not by the diagram but by its XLA-side sibling: dumping the optimized HLO and reading what ops surround the kernel.)

**Load the weights with an async prologue copy (−21 µs of hidden entry cost).** Declaring the fused weights as a VMEM operand made Pallas block-copy 31 MB on *every kernel entry — including empty ones* (a mixed-batch step launches both a decode and a prefill kernel; one is often empty). We caught this because an "empty" kernel measured 21 µs ≈ 31.4 MB ÷ 1.6 TB/s — the roofline arithmetic identified the culprit when an attempted fix (guarding the prologue) measurably did nothing. Now the weights are an HBM operand, copied to VMEM by an async DMA issued as the prologue's first instruction, overlapping the x and KV fetches, with a one-shot semaphore wait before first use. Empty kernels: 21 µs → 0.4 µs.

**Ablations keep us honest.** `kernel_bench.py` can patch any code region out of the kernel, measure, and auto-revert. This attributed the remaining fused cost precisely: Q GEMM 67 µs, KV GEMM 29 µs, RoPE ~20 µs — all near roofline. It also produced our favorite negative result: *removing* the Q norm made the kernel **65 µs slower**. The norm's store pattern was load-bearing for Mosaic's pipeline. Intuition would never have guessed that; the harness measured it in two minutes.

## The result

| | baseline | first fused | after auto-research loop |
|---|---|---|---|
| RPA prefill kernel | 253 µs | 545 µs | **424 µs** |
| fused work absorbed | — | proj+norm+rope | proj+norm+rope |
| same work done in XLA | ~200 µs | — | — |

The fused kernel now performs the projection+norm+RoPE work *faster than XLA does it standalone* (~170 µs marginal vs ~200 µs), while also eliminating the Q/K/V HBM round-trips and the per-layer XLA op overhead. The same method, pointed at the decode kernel, found an entirely different class of bug — a DMA prefetch serialized behind a cache write that a race-condition fix had over-applied — and recovered 325 µs there; the diagram's DMA lane showed reads queuing behind writes they didn't depend on.

## Takeaways

1. **A calibrated model beats a profiler for opaque kernels.** When the model total matches the measured total, every gap in the model's swimlanes is a real bubble with a name.
2. **Benchmark unit ops at exact shapes.** GEMM efficiency at `M=256` is not GEMM efficiency at `M=64`; the constants must come from the kernel's actual tile sizes.
3. **Gate ideas with a roofline before prototyping.** Half our ideas died in `idea_check.py` for the price of a division.
4. **Ablate, don't assume.** The component you'd optimize on intuition may be free — or negative.
5. **Watch the edges of the kernel.** Two of the biggest wins (weight fusion at load time, async weight copy) were about how operands cross the XLA↔Pallas boundary, not the kernel body at all.

The whole loop — unit benchmarks, ablation harness, roofline checker, trace differ, and the timeline generator — lives in `scripts/kernel_eval/` and runs unattended: describe a kernel idea, and the tooling benchmarks it, models it, and reports whether the silicon agrees.
