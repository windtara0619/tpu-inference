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
  - Only ops that can be mapped to _ragged_paged_attention_kernel are included

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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mosaic_dump_to_loc import (
    LOC_REF_RE,
    LOC_USE_RE,
    SIMPLE_FILE_LOC_RE,
    compact_loc_payload,
    parse_loc_map,
)

# Target kernel function name for filtering
TARGET_KERNEL = "_ragged_paged_attention_kernel"

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
    source_loc: Optional[str] = None  # e.g., "kernel.py:412:9"
    merged_count: int = 1  # number of ops merged into this event
    merged_ops: List[str] = field(default_factory=list)  # original ops if merged


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


def extract_loc_id(line: str) -> Optional[str]:
    """Extract #locNNN from a line like 'llo.foo ... loc(#loc123)'."""
    m = LOC_USE_RE.search(line)
    return m.group(1) if m else None


def resolve_to_target_kernel(
    loc_id: str,
    loc_map: Dict[str, str],
    target_kernel: str,
    max_depth: int = 50,
) -> Optional[str]:
    """
    Resolve loc chain and return the source location (file:line:col) if it
    references the target kernel function. Returns None if not found.
    """
    seen: set[str] = set()
    current = loc_id
    depth = 0

    while depth < max_depth:
        if current in seen:
            break
        seen.add(current)

        payload = loc_map.get(current)
        if payload is None:
            break

        # Check if this payload references the target kernel
        if target_kernel in payload:
            # Extract file:line:col
            match = SIMPLE_FILE_LOC_RE.search(payload)
            if match:
                path, line, col = match.group(1), match.group(2), match.group(3)
                return f"{path}:{line}:{col}"
            # Fallback to compacted payload
            return compact_loc_payload(payload)

        # Follow nested loc references
        refs = LOC_REF_RE.findall(payload)
        if not refs:
            break
        current = refs[0]
        depth += 1

    return None


def merge_consecutive_events(events: List[Event]) -> List[Event]:
    """
    Merge consecutive events that have the same source_loc (same line in the
    original kernel). Add a [merged:N] tag to identify merged groups.
    """
    if not events:
        return events

    merged: List[Event] = []
    current_group: List[Event] = [events[0]]

    for e in events[1:]:
        # Check if this event can be merged with the current group
        if (
            e.source_loc is not None
            and current_group[0].source_loc is not None
            and e.source_loc == current_group[0].source_loc
        ):
            current_group.append(e)
        else:
            # Finalize current group
            merged.append(_create_merged_event(current_group))
            current_group = [e]

    # Don't forget the last group
    merged.append(_create_merged_event(current_group))

    # Reassign time indices
    for i, e in enumerate(merged):
        e.t = i

    return merged


def _create_merged_event(group: List[Event]) -> Event:
    """Create a single event from a group of events to be merged."""
    first = group[0]
    if len(group) == 1:
        return first

    # Collect all ops in the group
    all_ops = [e.op for e in group]
    unique_ops = list(dict.fromkeys(all_ops))  # preserve order, remove dups

    # Create merged op name with tag
    merged_op = f"{unique_ops[0]}[merged:{len(group)}]"

    # Combine text from all events (truncated)
    combined_text = " | ".join(e.text.strip()[:50] for e in group[:3])
    if len(group) > 3:
        combined_text += f" | ... (+{len(group) - 3} more)"

    return Event(
        t=first.t,
        lane=first.lane,
        op=merged_op,
        line_no=first.line_no,
        text=combined_text,
        source_loc=first.source_loc,
        merged_count=len(group),
        merged_ops=all_ops,
    )


def parse_events(
    path: str, max_ops: int, target_kernel: str = TARGET_KERNEL
) -> List[Event]:
    """
    Parse events from LLO dump, filtering to only ops that map to target_kernel.
    """
    # First pass: read all lines and build loc map
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    loc_map = parse_loc_map(lines)

    # Second pass: extract events, filtering by target kernel
    events: List[Event] = []
    t = 0
    skipped = 0

    for i, raw in enumerate(lines, start=1):
        if is_ignorable_line(raw):
            continue
        op = extract_op_token(raw)
        if not op:
            continue

        # Check if this op maps to the target kernel
        loc_id = extract_loc_id(raw)
        source_loc = None
        if loc_id:
            source_loc = resolve_to_target_kernel(loc_id, loc_map, target_kernel)

        if source_loc is None:
            skipped += 1
            continue

        lane = guess_lane(op, raw)
        events.append(
            Event(
                t=t,
                lane=lane,
                op=op,
                line_no=i,
                text=raw.rstrip("\n"),
                source_loc=source_loc,
            )
        )
        t += 1
        if t >= max_ops:
            break

    if skipped > 0:
        print(f"Filtered out {skipped} ops not mapping to {target_kernel}")

    return events


def write_events_csv(events: List[Event], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "lane", "op", "line_no", "source_loc", "merged_count", "text"])
        for e in events:
            w.writerow(
                [e.t, e.lane, e.op, e.line_no, e.source_loc or "", e.merged_count, e.text]
            )


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
        # Handle merged ops - extract the base op name before [merged:N]
        if "[merged:" in short:
            short = short.split("[merged:")[0]
        short = short.replace("vmatmul", "matmul").replace("enqueue_dma", "dma").replace("dma_done", "done")
        if len(short) > 10:
            short = short[:10]
        # Add merged indicator if applicable
        if e.merged_count > 1:
            return f"{base}:{short}[x{e.merged_count}]"
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
    ap.add_argument(
        "--target_kernel",
        default=TARGET_KERNEL,
        help=f"Filter ops to those mapping to this kernel function (default: {TARGET_KERNEL})",
    )
    ap.add_argument(
        "--no_merge",
        action="store_true",
        help="Disable merging of consecutive ops from the same source line",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    events = parse_events(args.input, args.max_ops, args.target_kernel)
    if not events:
        raise SystemExit(
            f"No ops found mapping to {args.target_kernel}. "
            "Check input path or target kernel name."
        )

    original_count = len(events)

    # Merge consecutive events from the same source line
    if not args.no_merge:
        events = merge_consecutive_events(events)
        merged_count = original_count - len(events)
        if merged_count > 0:
            print(f"Merged {merged_count} ops into {len(events)} events")

    events_csv = os.path.join(args.out_dir, "events.csv")
    grid_csv = os.path.join(args.out_dir, "grid.csv")

    write_events_csv(events, events_csv)
    grid = build_grid(events, LANES)
    write_grid_csv(grid, grid_csv)

    print(f"Wrote {len(events)} events (from {original_count} ops mapping to {args.target_kernel})")
    print(f"  events: {events_csv}")
    print(f"  grid:   {grid_csv}")
    print("\nGoogle Sheets import:")
    print("  Sheets -> File -> Import -> Upload -> grid.csv -> Insert new sheet")
    print("Optional: use Conditional formatting to color by starting letter (D/M/V/C/O).")
    print("Look for [xN] suffix to identify merged ops from the same source line.")


if __name__ == "__main__":
    main()
