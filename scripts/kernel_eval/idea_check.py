#!/usr/bin/env python3
"""
idea_check.py -- Roofline feasibility estimator for a proposed Pallas kernel idea.

Before implementing anything, this tool answers: "Is this idea worth building?"
It quantifies the DMA saved, the compute added, and whether the roofline model
predicts an improvement over the XLA-compiled baseline.

Concept
-------
Every kernel is either HBM-bandwidth-bound or compute-bound:
  roofline_time = max(bytes_transferred / hbm_bw, flops / peak_flops)

A kernel idea is feasible when:
  proposed_roofline < baseline_roofline
  AND measured improvement >= 80% of the roofline prediction (sanity gate)

Hardware profiles
-----------------
  tpuv6e  HBM 459 GB/s,  bf16 MXU 918 TFLOP/s,  VMEM ~10 TB/s
  tpuv5e  HBM 819 GB/s,  bf16 MXU 197 TFLOP/s,  VMEM ~10 TB/s

Usage
-----
  # Evaluate the "fuse RoPE into attention" idea
  python idea_check.py evaluate \\
    --name "fuse_rope_into_rpa" \\
    --hw tpuv6e \\
    --baseline-dma  "Q_read=16.8,Q_write=16.8,K_read=4.2,K_write=4.2,sincos_write=1.05" \\
    --baseline-flops 2.3e9 \\
    --baseline-time  320 \\
    --proposed-dma  "Q_read=16.8,K_read=4.2" \\
    --proposed-flops 2.7e9 \\
    --proposed-time  595

  # Evaluate component costs from ablation JSON files
  python idea_check.py ablation-summary \\
    --baseline results/no_rope.json \\
    --fused    results/rope_fused.json \\
    --ablations results/skip_sincos.json results/skip_apply_q.json results/skip_apply_k.json \\
    --kernel-family "RPAm-p_128-bq_256_128-bkv_1024_512"
"""

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Hardware profiles
# ---------------------------------------------------------------------------

HW_PROFILES = {
    "tpuv6e": {
        "hbm_bw_GBs":    459,    # GB/s HBM bandwidth per chip
        "vmem_bw_GBs":  8000,    # GB/s VMEM bandwidth (estimated)
        "mxu_tflops":    918,    # bf16 peak MXU TFLOP/s
        "vpu_tflops":      2,    # bf16 VPU TFLOP/s (scalar/transcendental)
    },
    "tpuv5e": {
        "hbm_bw_GBs":    819,
        "vmem_bw_GBs":  8000,
        "mxu_tflops":    197,
        "vpu_tflops":    1.5,
    },
}


def roofline(bytes_hbm: float, flops: float, hw: dict) -> dict:
    """
    Return predicted time (µs) for an operation that transfers `bytes_hbm` bytes
    of HBM and performs `flops` floating-point operations.
    """
    t_mem_us    = bytes_hbm / (hw["hbm_bw_GBs"] * 1e9) * 1e6
    t_mxu_us    = flops / (hw["mxu_tflops"] * 1e12) * 1e6
    t_pred_us   = max(t_mem_us, t_mxu_us)
    intensity   = flops / bytes_hbm if bytes_hbm > 0 else float("inf")
    ridge_point = hw["mxu_tflops"] * 1e12 / (hw["hbm_bw_GBs"] * 1e9)
    return {
        "t_mem_us":   t_mem_us,
        "t_mxu_us":   t_mxu_us,
        "t_pred_us":  t_pred_us,
        "bound":      "memory" if t_mem_us >= t_mxu_us else "compute",
        "arith_intensity": intensity,
        "ridge_point_flop_per_byte": ridge_point,
    }


def parse_dma_spec(spec: str) -> float:
    """
    Parse "label=MB,label=MB,..." or plain float and return total bytes.
    Example: "Q_read=16.8,Q_write=16.8,K_read=4.2" -> (16.8+16.8+4.2)*1e6
    """
    total_mb = 0.0
    for part in spec.split(","):
        part = part.strip()
        if "=" in part:
            _, val = part.split("=", 1)
            total_mb += float(val)
        else:
            total_mb += float(part)
    return total_mb * 1e6   # return bytes


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args):
    hw = HW_PROFILES[args.hw]

    base_bytes = parse_dma_spec(args.baseline_dma)
    prop_bytes = parse_dma_spec(args.proposed_dma)

    base_rf = roofline(base_bytes, args.baseline_flops, hw)
    prop_rf = roofline(prop_bytes, args.proposed_flops, hw)

    delta_measured  = args.proposed_time - args.baseline_time
    delta_roofline  = prop_rf["t_pred_us"] - base_rf["t_pred_us"]
    dma_saved_mb    = (base_bytes - prop_bytes) / 1e6
    flops_added     = args.proposed_flops - args.baseline_flops
    verdict         = "WORTH EXPLORING" if prop_rf["t_pred_us"] < base_rf["t_pred_us"] else "LIKELY NOT WORTH IT"

    print(f"=== Idea: {args.name} ===")
    print(f"Hardware: {args.hw}  (HBM {hw['hbm_bw_GBs']} GB/s, MXU {hw['mxu_tflops']} TFLOP/s)")
    print()
    print(f"{'':30} {'Baseline':>14} {'Proposed':>14} {'Delta':>14}")
    print("-" * 75)
    print(f"{'HBM transferred (MB)':30} {base_bytes/1e6:>14.2f} {prop_bytes/1e6:>14.2f} {(prop_bytes-base_bytes)/1e6:>+14.2f}")
    print(f"{'FLOPs':30} {args.baseline_flops:>14.3e} {args.proposed_flops:>14.3e} {flops_added:>+14.3e}")
    print(f"{'Arith intensity (FLOP/B)':30} {base_rf['arith_intensity']:>14.2f} {prop_rf['arith_intensity']:>14.2f}")
    print(f"{'Roofline bound':30} {base_rf['bound']:>14} {prop_rf['bound']:>14}")
    print(f"{'Roofline time (µs)':30} {base_rf['t_pred_us']:>14.1f} {prop_rf['t_pred_us']:>14.1f} {delta_roofline:>+14.1f}")
    print(f"{'Measured time (µs)':30} {args.baseline_time:>14.1f} {args.proposed_time:>14.1f} {delta_measured:>+14.1f}")
    print()

    efficiency_base  = base_rf["t_pred_us"] / args.baseline_time * 100 if args.baseline_time else 0
    efficiency_prop  = prop_rf["t_pred_us"] / args.proposed_time * 100 if args.proposed_time else 0
    print(f"Roofline efficiency:  baseline={efficiency_base:.0f}%  proposed={efficiency_prop:.0f}%")

    if args.proposed_time > 0:
        actual_gain     = args.baseline_time - args.proposed_time
        roofline_gain   = base_rf["t_pred_us"] - prop_rf["t_pred_us"]
        realised_pct    = (actual_gain / roofline_gain * 100) if roofline_gain != 0 else float("nan")
        print(f"Actual gain vs roofline-predicted gain: {actual_gain:.1f} vs {roofline_gain:.1f} µs "
              f"({realised_pct:.0f}% realised)")

    print()
    print(f"VERDICT: {verdict}")
    if delta_roofline > 0:
        print(f"  The proposed idea adds more bottleneck ({delta_roofline:+.1f} µs roofline) "
              f"than it removes.")
        print(f"  DMA saved: {dma_saved_mb:.1f} MB  |  FLOPs added: {flops_added:.2e}")
        if dma_saved_mb < 0:
            print(f"  ↑ Proposed path has MORE DMA than baseline — re-check design.")
    else:
        print(f"  Roofline improves by {-delta_roofline:.1f} µs.")


# ---------------------------------------------------------------------------
# Subcommand: ablation-summary
# ---------------------------------------------------------------------------

def cmd_ablation_summary(args):
    def load(path):
        with open(path) as fh:
            return json.load(fh)

    base = load(args.baseline)
    fused = load(args.fused)
    ablations = [load(p) for p in (args.ablations or [])]

    fam = args.kernel_family

    def get_mean(result):
        stats = result.get("stats", {})
        # try exact match first, then prefix match
        if fam in stats:
            return stats[fam]["mean"]
        for k, v in stats.items():
            if k.startswith(fam.split("-")[0]):
                return v["mean"]
        return float("nan")

    base_mean  = get_mean(base)
    fused_mean = get_mean(fused)
    total_delta = fused_mean - base_mean

    print(f"=== Ablation summary for {fam} ===")
    print(f"Baseline (no rope):  {base_mean:.2f} µs")
    print(f"Fused (has_rope):    {fused_mean:.2f} µs")
    print(f"Total rope overhead: {total_delta:+.2f} µs")
    print()
    print(f"{'Component':<30}  {'w/ ablation (µs)':>18}  {'cost (µs)':>11}  {'% of total':>10}")
    print("-" * 75)
    for abl_result in ablations:
        abl_mean = get_mean(abl_result)
        cost     = fused_mean - abl_mean
        pct      = cost / total_delta * 100 if total_delta else 0
        name     = abl_result.get("ablation") or abl_result.get("tag", "?")
        desc     = {
            "skip_sincos":  "sin/cos compute",
            "skip_apply_q": "apply rope to Q",
            "skip_apply_k": "apply rope to K + writeback",
        }.get(name, name)
        print(f"{desc:<30}  {abl_mean:>18.2f}  {cost:>+11.2f}  {pct:>9.1f}%")
    print()
    print("Note: components overlap (non-additive) when cos/sin ablation also")
    print("      simplifies rotation math.  The 'skip all' baseline recovers exactly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- evaluate --
    p_eval = sub.add_parser("evaluate", help="Roofline feasibility for a proposed idea")
    p_eval.add_argument("--name", default="unnamed_idea")
    p_eval.add_argument("--hw", default="tpuv6e", choices=list(HW_PROFILES))
    p_eval.add_argument("--baseline-dma",   required=True,
                        help="MB spec for baseline: 'Q_read=16.8,Q_write=16.8,...'")
    p_eval.add_argument("--baseline-flops", required=True, type=float)
    p_eval.add_argument("--baseline-time",  required=True, type=float, help="Measured µs")
    p_eval.add_argument("--proposed-dma",   required=True)
    p_eval.add_argument("--proposed-flops", required=True, type=float)
    p_eval.add_argument("--proposed-time",  default=0.0,   type=float,
                        help="Measured µs (0 = not yet implemented)")

    # -- ablation-summary --
    p_abl = sub.add_parser("ablation-summary",
                           help="Summarise component costs from kernel_bench.py JSON outputs")
    p_abl.add_argument("--baseline",       required=True, help="JSON from kernel_bench --tag no_rope")
    p_abl.add_argument("--fused",          required=True, help="JSON from kernel_bench --tag rope_fused")
    p_abl.add_argument("--ablations",      nargs="+",     help="JSON files from each ablation run")
    p_abl.add_argument("--kernel-family",  default="RPAm-p_128-bq_256_128-bkv_1024_512")

    args = parser.parse_args()
    {"evaluate": cmd_evaluate, "ablation-summary": cmd_ablation_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
