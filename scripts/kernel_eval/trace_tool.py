#!/usr/bin/env python3
"""
trace_tool.py -- CLI for analysing XLA profiler traces produced by jax.profiler.trace().

Subcommands
-----------
  stats    Per-op-family statistics from one trace.
  compare  Delta table between two traces (base vs experiment).
  window   All ops in a time window around a named landmark op.
  source   All ops attributed to a given source file.

Trace format
------------
The tool accepts both plain trace.json and gzipped *.trace.json.gz files that
jax.profiler writes into the trace directory.  Point it at either the
top-level directory or the individual file.

Examples
--------
  # Op-family stats for RPA custom-calls
  python trace_tool.py stats \\
      --trace /tmp/trace_true/trace.json \\
      --pattern "^RPAm|^RPAd" --category custom-call

  # Compare FALSE vs TRUE for all custom-calls
  python trace_tool.py compare \\
      --base /tmp/trace_false/trace.json \\
      --exp  /tmp/trace_true/trace.json  \\
      --pattern "^RPAm|^RPAd" --category custom-call

  # 1 ms window centred on RPAm.72
  python trace_tool.py window \\
      --trace /tmp/trace_false/trace.json \\
      --landmark "RPAm-p_128-bq_256_128-bkv_1024_512.72" \\
      --before 500 --after 500

  # All ops sourced from rope_interface.py
  python trace_tool.py source \\
      --trace /tmp/trace_false/trace.json \\
      --source rope_interface
"""

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_trace_file(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    # directory: look for trace.json or *.trace.json.gz recursively
    for candidate in sorted(p.rglob("*.trace.json.gz")):
        return candidate
    for candidate in sorted(p.rglob("trace.json")):
        return candidate
    raise FileNotFoundError(f"No trace file found under {path_str!r}")


def load_trace(path_str: str) -> list[dict]:
    p = find_trace_file(path_str)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    return [e for e in events if isinstance(e, dict)]


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _name(e):
    return e.get("name", "")


def _args(e):
    a = e.get("args", {})
    return a if isinstance(a, dict) else {}


def _family(e):
    """Strip the trailing .NNN instance counter from HLO instruction names."""
    name = _name(e)
    return re.sub(r"\.\d+$", "", name)


def _is_xla_op(e):
    return e.get("pid") == 3 and e.get("tid") == 3


def filter_events(events, pattern=None, category=None, xla_ops_only=True):
    result = []
    pat = re.compile(pattern) if pattern else None
    for e in events:
        if xla_ops_only and not _is_xla_op(e):
            continue
        if category and _args(e).get("hlo_category") != category:
            continue
        if pat and not pat.search(_name(e)):
            continue
        result.append(e)
    return result


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------

def cmd_stats(args):
    events = load_trace(args.trace)
    filtered = filter_events(events, pattern=args.pattern,
                             category=args.category,
                             xla_ops_only=not args.all_pids)

    by_family = defaultdict(list)
    for e in filtered:
        by_family[_family(e)].append(e.get("dur", 0))

    if not by_family:
        print("No matching events found.", file=sys.stderr)
        return

    rows = []
    for fam, durs in sorted(by_family.items(), key=lambda x: -sum(x[1])):
        d = np.array(durs)
        rows.append((fam, len(d), d.mean(), np.median(d), d.min(), d.max(), d.sum()))

    w = max(len(r[0]) for r in rows)
    hdr = f"{'op family':<{w}}  {'n':>5}  {'mean µs':>10}  {'median µs':>10}  {'min µs':>10}  {'max µs':>10}  {'total µs':>12}"
    print(hdr)
    print("-" * len(hdr))
    for fam, n, mean, med, mn, mx, total in rows:
        print(f"{fam:<{w}}  {n:>5}  {mean:>10.3f}  {med:>10.3f}  {mn:>10.3f}  {mx:>10.3f}  {total:>12.3f}")


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------

def cmd_compare(args):
    base_events = load_trace(args.base)
    exp_events  = load_trace(args.exp)

    def aggregate(events):
        filtered = filter_events(events, pattern=args.pattern,
                                 category=args.category,
                                 xla_ops_only=not args.all_pids)
        by_family = defaultdict(list)
        for e in filtered:
            by_family[_family(e)].append(e.get("dur", 0))
        return {k: np.array(v) for k, v in by_family.items()}

    base_agg = aggregate(base_events)
    exp_agg  = aggregate(exp_events)

    all_families = sorted(set(base_agg) | set(exp_agg),
                          key=lambda k: -abs((exp_agg.get(k, np.zeros(1)).mean()
                                             - base_agg.get(k, np.zeros(1)).mean())))

    def stat(d):
        return (len(d), d.mean(), d.sum()) if len(d) else (0, float("nan"), 0.0)

    rows = []
    for fam in all_families:
        bn, bmean, btot = stat(base_agg.get(fam, np.array([])))
        en, emean, etot = stat(exp_agg.get(fam, np.array([])))
        delta = emean - bmean
        rows.append((fam, bn, bmean, btot, en, emean, etot, delta))

    w = max(len(r[0]) for r in rows)
    hdr = (f"{'op family':<{w}}  "
           f"{'base n':>6}  {'base mean µs':>13}  {'base total':>12}  "
           f"{'exp n':>5}  {'exp mean µs':>12}  {'exp total':>11}  "
           f"{'Δ mean µs':>11}")
    print(hdr)
    print("-" * len(hdr))
    for fam, bn, bmean, btot, en, emean, etot, delta in rows:
        sign = "+" if delta >= 0 else ""
        print(f"{fam:<{w}}  "
              f"{bn:>6}  {bmean:>13.3f}  {btot:>12.3f}  "
              f"{en:>5}  {emean:>12.3f}  {etot:>11.3f}  "
              f"{sign}{delta:>10.3f}")


# ---------------------------------------------------------------------------
# Subcommand: window
# ---------------------------------------------------------------------------

def cmd_window(args):
    events = load_trace(args.trace)
    xla_ops = [e for e in events if _is_xla_op(e) and "dur" in e and "ts" in e]
    xla_ops.sort(key=lambda e: e["ts"])

    landmark = next((e for e in xla_ops if _name(e) == args.landmark), None)
    if landmark is None:
        # try family match
        landmark = next((e for e in xla_ops
                         if _family(e) == args.landmark), None)
    if landmark is None:
        print(f"Landmark {args.landmark!r} not found.", file=sys.stderr)
        sys.exit(1)

    t_start = landmark["ts"] - args.before
    t_end   = landmark["ts"] + landmark["dur"] + args.after

    window_events = [e for e in xla_ops if t_start <= e["ts"] <= t_end]

    w_name = max(len(_name(e)) for e in window_events) if window_events else 20
    w_cat  = max(len(_args(e).get("hlo_category", "?")) for e in window_events) if window_events else 10
    print(f"{'ts (µs)':>12}  {'dur (µs)':>10}  {'category':<{w_cat}}  {'name':<{w_name}}  source")
    print("-" * (12 + 10 + w_cat + w_name + 20))
    for e in window_events:
        a = _args(e)
        marker = "  ◄ LANDMARK" if e is landmark else ""
        print(f"{e['ts']:>12.3f}  {e.get('dur',0):>10.3f}  "
              f"{a.get('hlo_category','?'):<{w_cat}}  "
              f"{_name(e):<{w_name}}  "
              f"{a.get('source','')}{marker}")


# ---------------------------------------------------------------------------
# Subcommand: source
# ---------------------------------------------------------------------------

def cmd_source(args):
    events = load_trace(args.trace)
    xla_ops = [e for e in events if _is_xla_op(e)]

    by_source = defaultdict(list)
    for e in xla_ops:
        src = _args(e).get("source", "")
        if args.source in src:
            key = (src, _family(e), _args(e).get("hlo_category", "?"))
            by_source[key].append(e.get("dur", 0))

    if not by_source:
        print(f"No ops with source containing {args.source!r}.", file=sys.stderr)
        return

    rows = []
    for (src, fam, cat), durs in sorted(by_source.items(), key=lambda x: -sum(x[1])):
        d = np.array(durs)
        rows.append((src, fam, cat, len(d), d.mean(), d.sum()))

    w_src = max(len(r[0]) for r in rows)
    w_fam = max(len(r[1]) for r in rows)
    hdr = f"{'source':<{w_src}}  {'op family':<{w_fam}}  {'category':<20}  {'n':>5}  {'mean µs':>10}  {'total µs':>12}"
    print(hdr)
    print("-" * len(hdr))
    for src, fam, cat, n, mean, total in rows:
        print(f"{src:<{w_src}}  {fam:<{w_fam}}  {cat:<20}  {n:>5}  {mean:>10.3f}  {total:>12.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- stats --
    p_stats = sub.add_parser("stats", help="Per-family statistics from one trace")
    p_stats.add_argument("--trace", required=True)
    p_stats.add_argument("--pattern", default=None, help="Regex filter on op name")
    p_stats.add_argument("--category", default=None, help="hlo_category filter")
    p_stats.add_argument("--all-pids", action="store_true")

    # -- compare --
    p_cmp = sub.add_parser("compare", help="Delta table between two traces")
    p_cmp.add_argument("--base", required=True, help="Base trace (e.g. rope=false)")
    p_cmp.add_argument("--exp",  required=True, help="Experiment trace (e.g. rope=true)")
    p_cmp.add_argument("--pattern", default=None)
    p_cmp.add_argument("--category", default=None)
    p_cmp.add_argument("--all-pids", action="store_true")

    # -- window --
    p_win = sub.add_parser("window", help="Ops in a time window around a landmark")
    p_win.add_argument("--trace", required=True)
    p_win.add_argument("--landmark", required=True, help="Exact op name (e.g. RPAm.72)")
    p_win.add_argument("--before", type=float, default=500, help="µs before landmark")
    p_win.add_argument("--after",  type=float, default=500, help="µs after landmark end")

    # -- source --
    p_src = sub.add_parser("source", help="All ops attributed to a source file")
    p_src.add_argument("--trace", required=True)
    p_src.add_argument("--source", required=True, help="Substring of source path")

    args = parser.parse_args()
    {"stats": cmd_stats, "compare": cmd_compare,
     "window": cmd_window,  "source": cmd_source}[args.cmd](args)


if __name__ == "__main__":
    main()
