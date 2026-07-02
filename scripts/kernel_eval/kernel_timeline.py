"""
kernel_timeline.py — DMA/MXU/VPU timeline for the RPAm kernel.

Calibrated to match measured benchmarks on v6e-1:
  baseline (no proj) : 253µs
  fused (qproj+kvproj): 545µs

Constants (measured via scripts/kernel_eval/op_bench.py — lax.fori_loop, two-point launch-overhead correction):
  T_flash_attention_per_loop  = 11.85µs  back-calculated: (253 - 8×T_output_norm) / 20 causal-mask loops
  T_q_projection_wallclock    = 25.0µs   max(q_matmul=18.65, q_norm_rope+rope_sincos_q=24.98) — VPU bottleneck
                                          rope_sincos_q=0.28µs is on-the-fly sin/cos not in old benchmark
  T_kv_projection_wallclock   = 26.6µs   max(kv_matmul=13.63, k_norm_rope+rope_sincos_k=23.63) + kv_pack_mosaic=3µs
                                          rope_sincos_k=1.08µs; kv_pack_mosaic≈3µs (VMEM strided scatter)
                                          NOTE: XLA kv_pack_store benchmark=43.5µs is HBM scatter (~15× too slow)
  T_extra_dma_barrier_per_bkv =  3.0µs   extra self-DMA barrier per bkv TILE in fused vs baseline
                                          fused: 2 barriers/bkv (cache + x_bkv); baseline: 1 barrier/bkv
                                          12 bkv tiles × 3µs = 36µs dominant gap source
                                          (NOTE: 12 bkv tiles, not 20 flash-attention loops)
  T_output_norm_per_bq_tile   =  2.0µs   softmax_store; out_pack_store ~0µs in Mosaic (step=1 sequential write)

  Fused model accuracy: 253 + 8×25.0 + 2×26.6 + 12×3.0 = 544µs  (measured 545µs, residual 1µs ✓)

DMA constants (op_bench.py bench_dma_bandwidth — pltpu.make_async_copy, HBM→VMEM):
  bandwidth     = 2217 GB/s  fit slope; method: 2D x_buf[loop_size, n_elems] > VMEM so XLA cannot
                             cache it; static row slicing x[i,:] gives different HBM offset per iter;
                             two-point slope (N_LO, N_HI) cancels Python + kernel-launch overhead.
  setup latency =  0.83 µs  fit intercept (fixed per-DMA overhead)

XLA-external constants (measured via op_bench.py xla_* ops, n_tok=2048, x from VMEM carry):
  Q matmul+norm+rope = 150.1µs  (xla_q_norm_rope; split: matmul=52.2µs, norm+rope=97.9µs)
  K matmul+norm+rope =  32.0µs  (xla_k_norm_rope; split: matmul=15.8µs, norm+rope=16.2µs)
  V matmul           =  15.8µs  (xla_v_matmul)
  Critical path      = 197.9µs  sequential Q→K→V (benchmark measured 197.8µs ✓)

Causal mask: bq[i] attends ceil(min(n_tok, (i+1)*bq_sz) / bkv_sz) bkv tiles.

Usage:
  python3 scripts/kernel_eval/kernel_timeline.py                  # fused (default)
  python3 scripts/kernel_eval/kernel_timeline.py --baseline       # no projection
  python3 scripts/kernel_eval/kernel_timeline.py --xla-external   # XLA proj pipeline
  python3 scripts/kernel_eval/kernel_timeline.py --n-tok 2048
"""

import argparse, math

# ── Calibrated constants ──────────────────────────────────────────────────────
# Derived from two benchmark measurements: baseline=253µs, fused=545µs.
# See docstring above for the calibration algebra.

# Flash attention step components (for timeline visualization)
# step1_mxu and step2_mxu measured via op_bench; step1_vpu inferred = flash_attn_qk - step1_mxu
# Pipeline per iteration: step1_mxu → { step1_vpu(VPU) || step2_mxu(MXU) } in parallel
T_flash_step1_mxu = 0.66   # µs — q @ k^T matmul (MXU)
T_flash_step1_vpu = 0.13   # µs — online softmax (VPU); inferred = qk_combined(0.79) - step1_mxu(0.66)
T_flash_step2_mxu = 0.69   # µs — p @ v matmul (MXU); overlaps with step1_vpu of next iter
# T_one_step = T_flash_step1_mxu + max(T_flash_step1_vpu, T_flash_step2_mxu)
#            = 0.66 + max(0.13, 0.69) = 1.35µs/step_pair
# T_flash_attention_per_loop = T_one_step × Nkv(8) × n_bq_chunks(2) = 1.35×16 = 21.6µs
# BUT: back-calculated from baseline (11.85µs) is more reliable — op_bench overestimates
# because benchmarks run ops in isolation without Mosaic's cross-op scheduling.
#
# Fractions of attn block: MXU is fully occupied (QK then PV back-to-back).
# acc (VPU) starts after QK and overlaps PV; it is shorter since T_step1_vpu < T_step2_mxu.
_ATTN_T_MXU_TOTAL = T_flash_step1_mxu + T_flash_step2_mxu          # 1.35µs
_ATTN_FRAC_QK  = T_flash_step1_mxu / _ATTN_T_MXU_TOTAL             # 0.489
_ATTN_FRAC_PV  = T_flash_step2_mxu / _ATTN_T_MXU_TOTAL             # 0.511
_ATTN_FRAC_ACC = T_flash_step1_vpu  / _ATTN_T_MXU_TOTAL             # 0.096
T_flash_attention_per_loop  = 11.85  # µs — (253 - 8×2.0) / 20; back-calc from baseline
T_q_projection_wallclock    = 25.0   # µs — wall-clock (VPU bottleneck: q_norm+rope+sincos=24.98)
T_q_matmul_mxu              = 18.65  # µs — MXU-only Q matmul (finishes before VPU; for timeline bar width)
T_kv_projection_wallclock   = 26.6   # µs — wall-clock: max(kv_mat=13.63, k_norm+sincos=23.63) + kv_pack≈3µs
T_kv_matmul_mxu             = 13.63  # µs — MXU-only KV matmul (finishes before VPU; for timeline bar width)
T_extra_dma_barrier_per_bkv =  3.0   # µs — extra x_bkv barrier per bkv tile; fused only, not in baseline
T_output_norm_per_bq_tile   =  2.0   # µs — op_bench corrected: softmax_store

DMA_SETUP_LATENCY = 0.83   # µs — fixed per-transfer setup (op_bench.py make_async_copy fit)
DMA_BANDWIDTH_GBS = 2217   # GB/s — HBM→VMEM async DMA bandwidth (op_bench.py, v6e-1)
                           # Method: 2D x_buf[loop_size, n_elems] > VMEM; static row slicing;
                           # two-point slope cancels Python overhead. Fit: t = 0.83 + 4.51e-7 × B

def dma_us(B, bw=DMA_BANDWIDTH_GBS):
    """Return DMA time in µs for B bytes at bw GB/s, including setup latency."""
    return DMA_SETUP_LATENCY + B / (bw * 1e3)


def make_dims(D=2560, Nq=32, Nkv=8, H=128, bq_sz=256, bkv_sz=1024, bkv_csz=512):
    return dict(D=D, Nq=Nq, Nkv=Nkv, H=H,
                bq_sz=bq_sz, bkv_sz=bkv_sz, bkv_csz=bkv_csz)


def bkv_info(bq_idx, n_tok, bq_sz, bkv_sz, bkv_csz):
    """Return list of (bkv_idx, n_loops, eff_sz, needs_proj) for a given bq tile.

    Causal mask: bq[i] only attends tokens 0..min(n_tok, (i+1)*bq_sz).
    needs_proj=True when this bkv tile is computed fresh (not already in cache).
    """
    eff_kv = min(n_tok, (bq_idx + 1) * bq_sz)
    result = []
    for bkv in range(math.ceil(eff_kv / bkv_sz)):
        esz    = min(bkv_sz, eff_kv - bkv * bkv_sz)
        nloops = math.ceil(esz / bkv_csz)
        kv_start = bkv * bkv_sz
        needs_proj = kv_start >= bq_idx * bq_sz   # first bq that covers this bkv
        result.append((bkv, nloops, esz, needs_proj))
    return result


# ── Op collectors ─────────────────────────────────────────────────────────────

def collect_baseline(n_tok, dims):
    """Ops for the no-projection kernel (mega_kernel=False)."""
    D=dims['D']; Nq=dims['Nq']; Nkv=dims['Nkv']; H=dims['H']
    bq_sz=dims['bq_sz']; bkv_sz=dims['bkv_sz']; bkv_csz=dims['bkv_csz']

    D_q_hbm  = dma_us(Nkv * bq_sz  * (Nq // Nkv // 2) * 2 * H * 2)
    D_kv_hbm = dma_us(bkv_sz * (Nkv * 2 // 2) * 2 * H * 2)
    D_kvcw   = D_kv_hbm
    n_bq     = math.ceil(n_tok / bq_sz)

    ops = []
    def add(t0, t1, track, sub, key, lbl=''):
        if t1 - t0 > 0.01:
            ops.append((round(t0, 3), round(t1, 3), track, sub, key, lbl))

    # Prologue DMAs (double-buffered, hidden)
    add(0, D_q_hbm,  'DMA', 'q_reads',  'DQ', 'q_hbm[bq=0]')
    add(0, D_kv_hbm, 'DMA', 'kv_reads', 'DKV','kv_hbm[bkv=0]')
    add(0, 2,        'VPU', 'vpu',       'r',  'rope_ts')

    t = 0.0
    for bq in range(n_bq):
        tiles = bkv_info(bq, n_tok, bq_sz, bkv_sz, bkv_csz)
        # Prefetch next q tile (double-buffered)
        if bq < n_bq - 1:
            add(t, t + D_q_hbm, 'DMA', 'q_reads', 'DQ', f'q_hbm[{bq+1}]')
        for i, (bi, nloops, esz, _) in enumerate(tiles):
            # Prefetch next kv tile (double-buffered)
            if i < len(tiles) - 1:
                add(t, t + D_kv_hbm, 'DMA', 'kv_reads', 'DKV',
                    f'kv_hbm[bq{bq},bkv{bi+1}]')
            elif bq < n_bq - 1:
                add(t, t + D_kv_hbm, 'DMA', 'kv_reads', 'DKV',
                    f'kv_hbm[bq{bq+1},bkv0]')
            attn_time = nloops * T_flash_attention_per_loop
            # Show nloops interleaved QK→PV||AAC cycles.
            # Each outer loop iteration covers Nkv×n_bq_chunks step-pairs.
            # Real pipeline per cycle: step1_mxu[t] → {step2_mxu[t-1](MXU) || step1_vpu[t](VPU)}
            t_qk_loop  = T_flash_attention_per_loop * _ATTN_FRAC_QK
            t_pv_loop  = T_flash_attention_per_loop * _ATTN_FRAC_PV
            t_acc_loop = T_flash_attention_per_loop * _ATTN_FRAC_ACC
            for loop_idx in range(nloops):
                tl = t + loop_idx * T_flash_attention_per_loop
                first = bq == 0 and bi == 0 and loop_idx == 0
                add(tl,             tl + t_qk_loop,            'MXU', 'mxu', 'AQK' if first else 'aqk', f'Attn[{bq},{bi}] QK loop{loop_idx}')
                add(tl + t_qk_loop, tl + t_qk_loop + t_pv_loop,'MXU', 'mxu', 'APV' if first else 'apv', f'Attn[{bq},{bi}] PV loop{loop_idx}')
                add(tl + t_qk_loop, tl + t_qk_loop + t_acc_loop,'VPU','vpu', 'AAC' if first else 'aac', f'Attn[{bq},{bi}] acc loop{loop_idx}')
            t += attn_time
        # KV cache write (only after last attention, on last bq tile that covers it)
        if bq == n_bq - 1:
            add(t, t + D_kvcw, 'DMA', 'kv_writes', 'DKW', 'KVcache_wr[0]')
            add(t, t + D_kvcw, 'DMA', 'kv_writes', 'DKW', 'KVcache_wr[1]')
        add(t, t + T_output_norm_per_bq_tile, 'VPU', 'vpu', 'O' if bq == 0 else 'o', f'out[{bq}]')
        t += T_output_norm_per_bq_tile

    return ops, round(t, 2)


def collect_fused(n_tok, dims):
    """Ops for the fused kernel (mega_kernel=True)."""
    D=dims['D']; bq_sz=dims['bq_sz']; bkv_sz=dims['bkv_sz']; bkv_csz=dims['bkv_csz']
    Nkv=dims['Nkv']; H=dims['H']

    D_x_bq   = dma_us(bq_sz  * D * 2)
    D_x_bkv  = dma_us(bkv_sz * D * 2)
    D_kv_hbm = dma_us(bkv_sz * (Nkv * 2 // 2) * 2 * H * 2)
    D_kvcw   = D_kv_hbm
    n_bq     = math.ceil(n_tok / bq_sz)

    ops = []
    def add(t0, t1, track, sub, key, lbl=''):
        if t1 - t0 > 0.01:
            ops.append((round(t0, 3), round(t1, 3), track, sub, key, lbl))

    add(0, D_x_bq,  'DMA', 'q_reads',  'DXQ', 'x_bq[0]')
    add(0, D_x_bkv, 'DMA', 'kv_reads', 'DXK', 'x_bkv[0]')
    add(0, 2,       'VPU', 'vpu',       'r',   'rope_ts')

    t = 0.0
    # KVcache DMAs are prefetched one bkv iteration ahead (double-buffered).
    # prev_tile_t tracks the start time of the previous bkv tile so the DMA
    # can be shown at its actual issue time rather than at consumption time.
    # Initialised to 0.0 = prologue: first bkv DMA is always issued there.
    prev_tile_t = 0.0

    for bq in range(n_bq):
        # Q matmul waits for x_bq DMA on bq[0]
        t0q = t + D_x_bq if bq == 0 else t
        # MXU: q_matmul (18.65µs). VPU: norm+rope+sincos starts AFTER matmul output is ready.
        # VPU cannot overlap with MXU here — norm reads matmul output (data dependency).
        add(t0q,                  t0q + T_q_matmul_mxu,          'MXU', 'mxu', 'Q' if bq == 0 else 'q', f'Q_mat[{bq}]')
        add(t0q + T_q_matmul_mxu, t0q + T_q_projection_wallclock, 'VPU', 'vpu', 'N' if bq == 0 else 'n', f'Q_norm[{bq}]')
        t = t0q + T_q_projection_wallclock
        if bq < n_bq - 1:
            add(t, t + D_x_bq, 'DMA', 'q_reads', 'DXQ', f'x_bq[{bq+1}]')

        tiles = bkv_info(bq, n_tok, bq_sz, bkv_sz, bkv_csz)
        for i, (bi, nloops, esz, needs_proj) in enumerate(tiles):
            t_tile_start = t   # save: becomes prev_tile_t for the next bkv iteration

            if needs_proj:
                if i < len(tiles) - 1:
                    add(t, t + D_x_bkv, 'DMA', 'kv_reads', 'DXK', f'x_bkv[{bi+1}]')
                # MXU: kv_matmul (13.63µs). VPU: norm+rope+sincos+pack starts AFTER matmul.
                add(t,                   t + T_kv_matmul_mxu,          'MXU', 'mxu', 'K' if bi == 0 else 'k', f'KV[{bi}]')
                add(t + T_kv_matmul_mxu, t + T_kv_projection_wallclock, 'VPU', 'vpu', 'n', f'K_norm[{bi}]')
                add(t + T_kv_projection_wallclock, t + T_kv_projection_wallclock + D_kvcw,
                    'DMA', 'kv_writes', 'DKW', f'KVcache_wr[{bi}]')
                t += T_kv_projection_wallclock
            else:
                # KVcache DMA was issued by prefetch_next_bkv at the START of the
                # PREVIOUS bkv iteration (double-buffered), not at the current t.
                add(prev_tile_t, prev_tile_t + D_kv_hbm, 'DMA', 'kv_reads', 'DKC',
                    f'KVcache_rd[{bq},{bi}]')

            # T_extra_dma_barrier_per_bkv: fused kernel waits for x_bkv DMA barrier
            # in addition to the cache-pages barrier that baseline already has.
            attn_time = nloops * T_flash_attention_per_loop + T_extra_dma_barrier_per_bkv
            # barrier overhead sits before the first QK cycle; then nloops interleaved cycles
            t_qk_loop  = T_flash_attention_per_loop * _ATTN_FRAC_QK
            t_pv_loop  = T_flash_attention_per_loop * _ATTN_FRAC_PV
            t_acc_loop = T_flash_attention_per_loop * _ATTN_FRAC_ACC
            t_loop_start = t + T_extra_dma_barrier_per_bkv
            for loop_idx in range(nloops):
                tl = t_loop_start + loop_idx * T_flash_attention_per_loop
                first = bq == 0 and bi == 0 and loop_idx == 0
                add(tl,             tl + t_qk_loop,             'MXU', 'mxu', 'AQK' if first else 'aqk', f'Attn[{bq},{bi}] QK loop{loop_idx}')
                add(tl + t_qk_loop, tl + t_qk_loop + t_pv_loop, 'MXU', 'mxu', 'APV' if first else 'apv', f'Attn[{bq},{bi}] PV loop{loop_idx}')
                add(tl + t_qk_loop, tl + t_qk_loop + t_acc_loop,'VPU', 'vpu', 'AAC' if first else 'aac', f'Attn[{bq},{bi}] acc loop{loop_idx}')
            t += attn_time

            prev_tile_t = t_tile_start   # advance for next bkv iteration

        add(t, t + T_output_norm_per_bq_tile, 'VPU', 'vpu', 'O' if bq == 0 else 'o', f'out[{bq}]')
        t += T_output_norm_per_bq_tile

    return ops, round(t, 2)


def collect_xla_external():
    """XLA external Q/K/V projection ops (baseline path, outside Pallas kernel).

    XLA loads x once to VMEM and reuses it for K and V projections, hiding the
    HBM bandwidth cost. Calibrated from HLO estimated_cycles at 2.5 GHz.

    Key insight: K and V projections reuse x already in VMEM after Q matmul,
    so they are effectively compute-bound (not HBM-bound like Q).
    Critical path: ~197.9µs isolated (measured 197.8µs via op_bench.py xla_* ops).
    """
    T_Qmat  = 52.2   # Q matmul  [2048,2560]@[2560,4096]  op_bench xla_q_matmul (x from VMEM)
    T_Qpost = 97.9   # Q norm+rope VPU  = xla_q_norm_rope(150.1) - xla_q_matmul(52.2)
    T_Kmat  = 15.8   # K matmul  [2048,2560]@[2560,1024]  x reused from VMEM
    T_Kpost = 16.2   # K norm+rope VPU  = xla_k_norm_rope(32.0) - xla_k_matmul(15.8)
    T_Vmat  = 15.8   # V matmul  [2048,2560]@[2560,1024]  x reused from VMEM
    D_xWk   = 15.0   # async DMA: x(10MB)+W_k(5MB)→VMEM, hidden in T_Qpost (97.9µs)
    D_Wv    =  6.0   # async DMA: W_v(5MB)→VMEM, hidden in T_Kpost (16.2µs)

    ops = []
    def add(t0, t1, track, sub, key, lbl=''):
        if t1 - t0 > 0.01:
            ops.append((round(t0, 3), round(t1, 3), track, sub, key, lbl))

    # W_q already in VMEM via cross_program_prefetch from previous layer (free)
    add(0, 5, 'DMA', 'free', 'XWQP', 'W_q from prev layer (free)')

    # Q matmul: x from HBM (cold load), W_q from VMEM
    add(0, T_Qmat, 'MXU', 'mxu', 'XQ', 'Q matmul x@W_q')

    # Async DMA x+W_k starts right after Q matmul, hidden during Q post
    add(T_Qmat, T_Qmat + D_xWk, 'DMA', 'dma_read', 'XDMA', 'x+W_k → VMEM')

    # Q norm+rope: VPU, overlaps with DMA above (done at T_Qmat+D_xWk = 45µs)
    add(T_Qmat, T_Qmat + T_Qpost, 'VPU', 'vpu', 'XQP', 'Q norm+rope')

    t = T_Qmat + T_Qpost   # ~125µs — x and W_k in VMEM since ~45µs

    # K matmul: x[VMEM] @ W_k[VMEM] — no HBM read, very fast
    add(t, t + T_Kmat, 'MXU', 'mxu', 'XK', 'K matmul x@W_k')

    # Async DMA W_v, hidden during K norm+rope
    add(t + T_Kmat, t + T_Kmat + D_Wv, 'DMA', 'dma_read', 'XDMV', 'W_v → VMEM')

    # K norm+rope: VPU
    add(t + T_Kmat, t + T_Kmat + T_Kpost, 'VPU', 'vpu', 'XKP', 'K norm+rope')

    t += T_Kmat + T_Kpost   # ~146µs

    # V matmul: x[VMEM] @ W_v[VMEM]
    add(t, t + T_Vmat, 'MXU', 'mxu', 'XV', 'V matmul x@W_v')
    t += T_Vmat   # ~160µs

    return ops, round(t, 2)


# ── ASCII renderer ─────────────────────────────────────────────────────────────

_LABELS = {
    'DQ': 'q_hbm', 'DKV': 'kv_hbm', 'DKC': 'KVcache_rd', 'DKW': 'KVcache_wr',
    'DXQ': 'x_bq',  'DXK': 'x_bkv',
    'Q': 'Q_mat',   'q': 'Q_mat',   'K': 'KV_mat',  'k': 'KV_mat',
    'AQK': 'Attn_QK',  'aqk': 'Attn_QK',
    'APV': 'Attn_PV',  'apv': 'Attn_PV',
    'AAC': 'Attn_acc', 'aac': 'Attn_acc',
    'N': 'Q_norm',  'n': 'Q/K_norm',
    'O': 'out_norm','o': 'out_norm', 'r': 'rope_ts',
    'XQ': 'Q_mat',  'XK': 'K_mat', 'XV': 'V_mat',
    'XQP': 'Q_norm+rope', 'XKP': 'K_norm+rope',
    'XDMA': 'x+W_k_DMA', 'XDMV': 'W_v_DMA', 'XWQP': 'W_q(free)',
}

_DMA_SUBS = {
    'q_reads': 0, 'kv_reads': 1, 'kv_writes': 2,
    'dma_read': 0, 'free': 2,
}

_CHARS = {
    'DQ': 'Q', 'DKV': 'V', 'DKC': 'c', 'DKW': 'W',
    'DXQ': 'B', 'DXK': 'V',
    'Q': 'Q', 'q': 'q', 'K': 'K', 'k': 'k',
    'AQK': 'Q', 'aqk': 'q', 'APV': 'P', 'apv': 'p', 'AAC': 'S', 'aac': 's',
    'N': 'N', 'n': 'n',
    'O': 'O', 'o': 'o', 'r': 'r',
    'XQ': 'Q', 'XK': 'K', 'XV': 'V',
    'XQP': 'N', 'XKP': 'n',
    'XDMA': 'D', 'XDMV': 'd', 'XWQP': 'w',
}


def _render_ascii(title, ops, total_us, zoom_us=None):
    """Render an ASCII timeline of ops."""
    end_us = zoom_us or total_us
    WIDTH  = 100
    PAD_L  = 7   # chars reserved for track label

    def px(t):
        return PAD_L + int(min(1.0, max(0.0, t / end_us)) * (WIDTH - PAD_L - 1))

    # Track buffers: 3 DMA lanes + MXU + VPU
    tracks = [[' '] * WIDTH for _ in range(5)]

    def put(buf, col, text):
        for i, ch in enumerate(text):
            if 0 <= col + i < WIDTH:
                buf[col + i] = ch

    def box(buf, t0, t1, char):
        c0, c1 = px(t0), px(t1)
        w = max(3, c1 - c0)
        put(buf, c0, '[' + char + '─' * (w - 3) + ']')

    for t0, t1, track, sub, key, lbl in ops:
        if t0 >= end_us:
            continue
        ch = _CHARS.get(key, '?')
        if track == 'DMA':
            lane = _DMA_SUBS.get(sub, 0)
            box(tracks[lane], t0, min(t1, end_us), ch)
        elif track == 'MXU':
            box(tracks[3], t0, min(t1, end_us), ch)
        else:
            box(tracks[4], t0, min(t1, end_us), ch)

    # Ruler
    ruler = [' '] * WIDTH
    ticks = list(range(0, int(end_us) + 1, max(10, int(end_us) // 8)))
    if int(end_us) not in ticks:
        ticks.append(int(end_us))
    for t in ticks:
        col = px(t)
        s = str(t)
        for i, ch in enumerate(s):
            if col + i < WIDTH:
                ruler[col + i] = ch

    suffix = f'  [zoom 0–{zoom_us:.0f}µs]' if zoom_us else ''
    print(f'\n  {title}{suffix}')
    print(f'  total={total_us:.0f}µs')
    print()
    print(f"  µs:    {''.join(ruler)}")
    print(f"  DMA_q: {''.join(tracks[0])}")
    print(f"  DMA_k: {''.join(tracks[1])}")
    print(f"  DMA_w: {''.join(tracks[2])}")
    print(f"  MXU:   {''.join(tracks[3])}")
    print(f"  VPU:   {''.join(tracks[4])}")
    print()
    print('  Legend')
    print('  ──────')
    print('  DMA_q: Q reads (q_hbm/x_bq/x+W_k)  DMA_k: KV reads  DMA_w: KV writes/W_v')
    print('  MXU:   Q=Q_matmul  K=KV_matmul  A=Attn')
    print('  VPU:   r=rope_ts  N/n=Q/K_norm+rope  O/o=out_norm')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tok',      type=int,   default=2048)
    ap.add_argument('--bq-sz',      type=int,   default=256)
    ap.add_argument('--bkv-sz',     type=int,   default=1024)
    ap.add_argument('--bkv-csz',    type=int,   default=512)
    ap.add_argument('--D',          type=int,   default=2560)
    ap.add_argument('--Nq',         type=int,   default=32)
    ap.add_argument('--Nkv',        type=int,   default=8)
    ap.add_argument('--H',          type=int,   default=128)
    ap.add_argument('--baseline',   action='store_true', help='no projection')
    ap.add_argument('--xla-external', action='store_true',
                    help='XLA external Q/K/V projection pipeline')
    ap.add_argument('--zoom',       type=float, default=0,
                    help='also show a zoomed view up to this µs value')
    args = ap.parse_args()

    dims = make_dims(D=args.D, Nq=args.Nq, Nkv=args.Nkv, H=args.H,
                     bq_sz=args.bq_sz, bkv_sz=args.bkv_sz, bkv_csz=args.bkv_csz)

    if args.xla_external:
        ops, total = collect_xla_external()
        title = 'XLA EXTERNAL (proj_q/proj_k/proj_v/rope — outside Pallas kernel)'
        _render_ascii(title, ops, total)
        if args.zoom:
            _render_ascii(title, ops, total, zoom_us=args.zoom)
    elif args.baseline:
        ops, total = collect_baseline(args.n_tok, dims)
        title = f'BASELINE (no projection) — {args.n_tok} tokens'
        _render_ascii(title, ops, total)
        if args.zoom:
            _render_ascii(title, ops, total, zoom_us=args.zoom)
    else:
        ops, total = collect_fused(args.n_tok, dims)
        title = f'FUSED (mega_kernel=True) — {args.n_tok} tokens'
        _render_ascii(title, ops, total)
        if args.zoom:
            _render_ascii(title, ops, total, zoom_us=args.zoom)


if __name__ == '__main__':
    main()
