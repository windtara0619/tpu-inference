# RPAm Kernel DMA/MXU/VPU Timeline

## Workload: 1560 tokens, fused (has_qproj + has_kvproj)
- bq_sz=256 → 7 bq tiles (Q tiles)
- bkv_sz=1024 → 2 fresh KV tiles (with eager cache write)
- Qwen3-4B: D=2560, N_q=32, N_kv=8, H=128

```
µs:   0        45        108      141       204       237     326    415    504    593    682    771
      ├────────┼─────────┼────────┼─────────┼─────────┼───────┼──────┼──────┼──────┼──────┼──────┤

DMA:  [B][VVV][C]        [b][vvv][c]         [W─────] [W─────]

MXU:  ··[Q─────][KV────────][A──][KV────────][A──]   [q──][a][a] [q──][a][a] [q──][a][a] ×4 more
         bq[0]   KV tile 0  attn  KV tile 1  attn      bq[1]       bq[2]       bq[3..6]

VPU:  [r][N──]            [n─]              [n─]       [n─]        [n─]  ...
```

## Symbol Legend

### DMA track
| Symbol | Meaning |
|--------|---------|
| `B` | DMA start: x_bq[0] HBM→VMEM (1.3MB, hidden states for Q projection, bq tile 0) |
| `b` | DMA start: x_bq[1] HBM→VMEM (double-buffered, starts during bkv loop of bq[0]) |
| `V` / `VVV` | DMA start: x_bkv[0] HBM→VMEM (5.24MB, hidden states for KV projection, bkv tile 0) |
| `v` / `vvv` | DMA start: x_bkv[1] HBM→VMEM (5.24MB, bkv tile 1, starts during bq[0]'s bkv[0] loop) |
| `C` | DMA start: KV cache pages HBM→VMEM (bkv tile 0 cache read, bkv_p page DMAs) |
| `c` | DMA start: KV cache pages HBM→VMEM (bkv tile 1 cache read) |
| `W─────` | DMA start: KV cache write VMEM→HBM (bkv_x2_ref → kv_cache_hbm, async) |

### MXU track (Matrix Multiply Unit)
| Symbol | Meaning |
|--------|---------|
| `Q─────` | Q projection: `x_bq @ W_q` [256×2560×4096] ~17µs (bq[0]) |
| `q──` | Same Q projection for bq[1..6] — smaller because fewer bkv tiles follow |
| `KV────` | KV projection: `x_bkv @ W_kv` [1024×2560×2048] ~34µs per tile |
| `A──` | Flash attention: QK^T softmax (step1) interleaved with PV accumulate (step2) across KV heads |
| `a` | Flash attention for bq[1..6] (reads KV from cache, no KV projection needed) |
| `··` | MXU idle: waiting for x_bq[0] DMA to complete (started in prologue, ~1.5µs) |

### VPU track (Vector Processing Unit)
| Symbol | Meaning |
|--------|---------|
| `r` | RoPE timescale computation (tiny, runs in prologue during DMA) |
| `N──` | Q norm + Q RoPE + strided_store to bq_x2_ref (~15µs, after Q MXU) |
| `n─` | K norm + K RoPE (~10µs, after KV MXU; also includes Q norm for bq[1..6]) |

## The ~30µs gap between `[a]` and `[q]`

Between the end of attention (`[a]`) and the start of the next Q matmul (`[q]`),
there is a ~30µs gap where the MXU is idle. This is **NOT** because Q is waiting
for the KV cache write — it is because of **unavoidable VPU work** at the
bq-iteration boundary:

```
[bkv attention loop ends]
  ↓ ~30µs of pure VPU — no MXU to overlap with
  acc = acc_ref[...]                   VPU read: [8, 1024, 128] f32 = 4.2MB
  out = acc * reciprocal(l)           VPU: 1M float32 multiply+reciprocal ~10µs
  bitcast + strided_store → bo_x2_ref VPU write: 4.2MB strided ~7µs
  start_send_bo                        DMA start: non-blocking
  [end of bq[i]]
  l_ref[...] = 0                       VPU fill: [8, 1024, 128] × 3 ~10µs
  m_ref[...] = -inf                    VPU fill
  acc_ref[...] = 0                     VPU fill
  [begin of bq[i+1]]
  ↓
  [Q─────] MXU starts
```

**Why MXU is blocked**: The output norm (`acc * recip(l)`) reads `acc_ref`,
then it must be zeroed before the bkv attention loop of the next tile can write
into it. These sequential VPU steps must all complete before Q MXU can be
followed by any attention — there is no MXU work to issue in this window.

**Why KV cache write is NOT the cause**: The `W─────` KV cache write (VMEM→HBM)
is a non-blocking DMA started after `compute_kv_from_x_bkv`. It is tracked by
`wait_update_kv_cache(bkv_sem_idx)` which is called at the start of the next
`_fetch_bkv`. This wait completes long before bq[1] starts (the DMA takes ~5µs
but bq[0]'s attention runs for ~46µs after the write DMA starts).

**Why moving l/m/acc init after Q MXU did not help**:
Mosaic already overlaps the init with Q MXU regardless of code order because
`l_ref/m_ref/acc_ref` (init) and `x_tile_x2_ref/bq_x2_ref` (Q MXU) are
different buffers with no data dependency. Mosaic detects this at compile time
and schedules them concurrently. The init (~10µs) is already hidden inside Q MXU
(~17µs) — moving the code changes nothing.

**Why output norm is the real bottleneck**:
The output norm (`acc * reciprocal(l)`) runs at the END of iteration i and Q MXU
runs at the START of iteration i+1. `@pl.loop(unroll=False)` creates a dynamic
loop boundary that Mosaic treats as opaque — it cannot see across iterations to
discover that these two ops are independent. The step1/step2 pipeline works
because the inner `for kv_head_idx` loop is statically unrolled by Python, so
Mosaic sees ALL ops in a single flat IR block and can interleave freely.

**Attempted fix and why it regressed**:
Moving `finalize_prev_bq_output` to the start of bq[i+1] (before Q MXU) and
guarding with `@pl.when(bq_idx > 0)` caused a ~16µs regression. The dynamic
`@pl.when` condition forces Mosaic to conservatively treat it as potentially
dependent on Q MXU output, serializing rather than parallelizing them.

## Key timing estimates (Qwen3-4B, bq_sz=256, bkv_sz=1024)

| Operation | Unit | Time |
|-----------|------|------|
| x_bq DMA (1.3MB) | DMA | ~1.5µs |
| x_bkv DMA (5.24MB) | DMA | ~5.8µs |
| KV cache page DMAs (~4.2MB) | DMA | ~4.7µs |
| Q matmul `[256,2560]×[2560,4096]` | MXU | ~17µs |
| Q norm + Q RoPE + store | VPU | ~15µs |
| KV matmul `[1024,2560]×[2560,2048]` | MXU | ~34µs |
| K norm + K RoPE | VPU | ~10µs |
| Attention (1 bkv tile, causal) | MXU | ~23µs |
| Output norm (acc * recip(l)) | VPU | ~10µs |
| strided_store output | VPU | ~7µs |
| l/m/acc init (×3 fills) | VPU | ~10µs |

## Overall structure

```
PROLOGUE (bq[0] only):
  Start DMAs: x_bq[0], x_bkv[0], KV_cache[0]  ← all concurrent
  VPU: RoPE timescale computation                ← overlaps with DMAs

bq[0] — the expensive first tile (computes KV for 2 fresh bkv tiles):
  MXU: Q proj        ← x_bkv DMAs already done (started in prologue)
  VPU: Q norm+rope
  [bkv tile 0]:
    MXU: KV proj     ← uses x_bkv[0] from VMEM
    VPU: K norm+rope
    async DMA: KV tile → cache
    MXU: attention
  [bkv tile 1]:
    MXU: KV proj     ← uses x_bkv[1] from VMEM
    VPU: K norm+rope
    async DMA: KV tile → cache
    MXU: attention
  VPU: output norm + bo store   ← 30µs gap before bq[1]

bq[1..6] — cached tiles (KV already in cache, no KV projection):
  MXU: Q proj
  VPU: Q norm+rope
  [bkv tile 0]: cache DMA → attention MXU
  [bkv tile 1]: cache DMA → attention MXU
  VPU: output norm + bo store   ← 30µs gap before next bq

EPILOGUE:
  wait_send_bo (both slots)
  wait_update_kv_cache (both slots)
```
