#!/usr/bin/env python3
"""
mosaic_llo_to_sheets.py

Parse Mosaic post-finalize LLO text dump and generate:
  - events.csv : one row per op (time index, lane, op, line)
  - grid.csv   : swimlane grid (rows=lanes, cols=time steps), each cell is one "circle"

Assumptions:
  - Every op takes exactly 1 time unit ("circle")
  - Timeline order = textual order in the file
  - We only count lines that look like operations (contain 'llo.' or 'scf.' etc.)

Usage:
  python mosaic_llo_to_sheets.py \
    --input /path/to/post-finalize-llo.txt \
    --out_dir out \
    --max_ops 5000

Then in Google Sheets:
  File -> Import -> Upload -> grid.csv -> Insert new sheet
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

LANES = ["DMA", "MXU", "VPU", "CTRL", "OTHER"]

# --- Heuristic lane classification rules ---
# Ordered by priority: first match wins.
LANE_RULES: List[Tuple[str, re.Pattern]] = [
    # DMA
    ("DMA", re.compile(r"\bllo\.(enqueue_dma|dma_done|dma_wait|dma_start|copy|memcpy)\b")),
    # MXU / matrix pipeline
    ("MXU", re.compile(r"\bllo\.(vmatmul|vmatprep|vmatres|vlatchi|vlatch|vmat)\b")),
    # VPU / vector ALU / math-heavy
    ("VPU", re.compile(r"\bllo\.(vexp|vlog|vtanh|vsin|vcos|vrecip|vrsqrt|vsqrt|vdiv|vmul|vadd|vsub|vmax|vmin|vand|vor|vxor|vshl|vshr|vsel|vconvert|vcast)\b")),
    # Control / scalar / flow
    ("CTRL", re.compile(r"\b(scf\.(if|for|while)|llo\.error\.if|llo\.assert|llo\.trace_(start|stop)|llo\.(constant|sadd|ssub|smul|sdiv|srem|shl|shr|icmp|fcmp|select|cast|bitcast|reshape|saddr|index|tid|pid))\b")),
]

# Lines to ignore (noise): loc tables, module headers, type attrs, etc.
IGNORE_PREFIXES = (
    "#loc", "#map", "#tpu.", "#vector.", "#affine_map", "module ", "func.func ", "llvm.", "builtin."
)

# Very loose op detector: capture 'llo.xxx' or 'scf.xxx' token
OP_TOKEN_RE = re.compile(r"\b(llo\.[A-Za-z0-9_.]+|scf\.[A-Za-z0-9_.]+)\b")


@dataclass
class Event:
    t: int
    lane: str
    op: str
    line_no: int
    text: str


def guess_lane(op: str, text: str) -> str:
    blob = f"{op} {text}"
    for lane, pat in LANE_RULES:
        if pat.search(blob):
            return lane
    return "OTHER"


def is_ignorable_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    for p in IGNORE_PREFIXES:
        if s.startswith(p):
            return True
    # Skip pure braces / attributes
    if s in ("{", "}", "})", "})", ");", ")", "];"):
        return True
    # Skip lines that are just comments
    if s.startswith("//"):
        return True
    return False


def extract_op_token(line: str) -> Optional[str]:
    m = OP_TOKEN_RE.search(line)
    return m.group(1) if m else None


def parse_events(path: str, max_ops: int) -> List[Event]:
    events: List[Event] = []
    t = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f, start=1):
            if is_ignorable_line(raw):
                continue
            op = extract_op_token(raw)
            if not op:
                continue

            lane = guess_lane(op, raw)
            events.append(Event(t=t, lane=lane, op=op, line_no=i, text=raw.rstrip("\n")))
            t += 1
            if t >= max_ops:
                break

    return events


def write_events_csv(events: List[Event], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "lane", "op", "line_no", "text"])
        for e in events:
            w.writerow([e.t, e.lane, e.op, e.line_no, e.text])


def build_grid(events: List[Event], lanes: List[str]) -> List[List[str]]:
    """
    Returns grid as rows:
      row0: header (Time 0..N-1)
      row1..: each lane row has a marker when an event is in that lane at time t.

    Cell content is a short token for easy coloring and reading:
      DMA: "D" (plus op suffix)
      MXU: "M"
      VPU: "V"
      CTRL:"C"
      OTHER:"O"
    """
    n = len(events)
    header = ["lane\\t"] + [str(t) for t in range(n)]
    rows = [header]

    lane_to_row = {lane: [""] * (n + 1) for lane in lanes}
    for lane in lanes:
        lane_to_row[lane][0] = lane

    def marker(e: Event) -> str:
        # compact marker; you can change to just "●" if you want
        base = {"DMA": "D", "MXU": "M", "VPU": "V", "CTRL": "C", "OTHER": "O"}[e.lane]
        # Add tiny op hint so it's not a completely opaque dot
        # e.g. vmatmul -> "M:matmul"
        short = e.op.split(".")[-1]
        short = short.replace("vmatmul", "matmul").replace("enqueue_dma", "dma").replace("dma_done", "done")
        if len(short) > 10:
            short = short[:10]
        return f"{base}:{short}"

    for e in events:
        lane_to_row[e.lane][e.t + 1] = marker(e)

    for lane in lanes:
        rows.append(lane_to_row[lane])

    return rows


def write_grid_csv(grid: List[List[str]], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to post-finalize-llo.txt")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    ap.add_argument("--max_ops", type=int, default=10000, help="Max ops to parse (safety cap)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    events = parse_events(args.input, args.max_ops)
    if not events:
        raise SystemExit("No ops found. Check input path or tweak OP_TOKEN_RE / IGNORE rules.")

    events_csv = os.path.join(args.out_dir, "events.csv")
    grid_csv = os.path.join(args.out_dir, "grid.csv")

    write_events_csv(events, events_csv)
    grid = build_grid(events, LANES)
    write_grid_csv(grid, grid_csv)

    print(f"Wrote {len(events)} ops")
    print(f"  events: {events_csv}")
    print(f"  grid:   {grid_csv}")
    print("\nGoogle Sheets import:")
    print("  Sheets -> File -> Import -> Upload -> grid.csv -> Insert new sheet")
    print("Optional: use Conditional formatting to color by starting letter (D/M/V/C/O).")


if __name__ == "__main__":
    main()
