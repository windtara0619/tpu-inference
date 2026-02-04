#!/usr/bin/env python3
"""
expand_mlir_loc_refs.py

Replace loc(#locNNN) references with concrete locations like kernel_old.py:412:9,
using #locNNN = loc(...) definitions at the top of an MLIR text dump.

Usage:
  python expand_mlir_loc_refs.py \
    --input  post-finalize-llo.txt \
    --output post-finalize-llo.expanded.txt

Optional:
  --keep_wrapper   Replace loc(#locNNN) -> loc(kernel_old.py:412:9) instead of bare kernel_old.py:412:9
  --drop_loc_defs  Remove the #locNNN = ... definition lines from the output (makes file smaller)
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, Tuple

# Match a loc definition line like:
#   #loc2719 = loc("kernel_old.py":412:9)
# or more complex:
#   #loc123 = loc(fused["..."](#loc1, #loc2))
LOC_DEF_RE = re.compile(r'^(#loc\d+)\s*=\s*loc\((.*)\)\s*$')

# Match uses like: loc(#loc2719)
LOC_USE_RE = re.compile(r'loc\(\s*(#loc\d+)\s*\)')

# Extract a simple file:line:col pattern from a loc payload:
#   "kernel_old.py":412:9
SIMPLE_FILE_LOC_RE = re.compile(r'"([^"]+)"\s*:\s*(\d+)\s*:\s*(\d+)')
LOC_REF_RE = re.compile(r'#loc\d+')


def compact_loc_payload(payload: str) -> str:
    """
    Convert loc(...) payload into a compact human-readable string.
    Prefer file:line:col if present; otherwise fall back to a shortened payload.
    """
    match = SIMPLE_FILE_LOC_RE.search(payload)
    if match:
        path, line, col = match.group(1), match.group(2), match.group(3)
        return f"{path}:{line}:{col}"

    # Fallback: keep something informative but short (strip whitespace).
    # Examples: fused["..."](#loc1, #loc2), callsite(...), unknown, etc.
    shortened = " ".join(payload.split())
    if len(shortened) > 120:
        shortened = shortened[:117] + "..."
    return shortened


def parse_loc_map(lines: list[str]) -> Dict[str, str]:
    loc_map: Dict[str, str] = {}
    for line in lines:
        match = LOC_DEF_RE.match(line.strip())
        if not match:
            continue
        loc_id, payload = match.group(1), match.group(2)
        loc_map[loc_id] = payload
    return loc_map


def resolve_loc_chain(
    loc_id: str,
    loc_map: Dict[str, str],
    keep_wrapper: bool,
    max_depth: int,
) -> str:
    chain: list[str] = []
    seen: set[str] = set()
    current = loc_id
    depth = 0
    while depth < max_depth:
        if current in seen:
            chain.append(f"{current}=<cycle>")
            break
        seen.add(current)
        payload = loc_map.get(current)
        if payload is None:
            chain.append(f"{current}=<unresolved>")
            break
        compacted = compact_loc_payload(payload)
        chain.append(compacted)
        refs = LOC_REF_RE.findall(payload)
        if not refs:
            break
        current = refs[0]
        depth += 1
    if keep_wrapper:
        return f"loc({' -> '.join(chain)})"
    return " -> ".join(chain)


def rewrite_file(
    lines: list[str],
    loc_map: Dict[str, str],
    keep_wrapper: bool,
    drop_loc_defs: bool,
    max_depth: int,
) -> Tuple[list[str], int, int]:
    """
    Returns (new_lines, replaced_count, unresolved_count)
    """
    replaced = 0
    unresolved = 0
    out: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Optionally drop loc definition lines
        if drop_loc_defs and LOC_DEF_RE.match(stripped):
            continue

        def _sub(match: re.Match) -> str:
            nonlocal replaced, unresolved
            loc_id = match.group(1)
            if loc_id not in loc_map:
                unresolved += 1
                return match.group(0)  # leave loc(#locNNN) as-is
            replaced += 1
            return resolve_loc_chain(loc_id, loc_map, keep_wrapper, max_depth)

        new_line = LOC_USE_RE.sub(_sub, line)
        out.append(new_line)

    return out, replaced, unresolved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input MLIR text file")
    ap.add_argument("--output", required=True, help="Output rewritten file")
    ap.add_argument(
        "--keep_wrapper",
        action="store_true",
        help="Replace loc(#locNNN) with loc(file:line:col) instead of bare file:line:col",
    )
    ap.add_argument(
        "--drop_loc_defs",
        action="store_true",
        help="Remove #locNNN = loc(...) definition lines from output",
    )
    ap.add_argument(
        "--max_depth",
        type=int,
        default=50,
        help="Maximum depth for resolving nested loc chains (default: %(default)s).",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    loc_map = parse_loc_map(lines)
    new_lines, replaced, unresolved = rewrite_file(
        lines, loc_map, args.keep_wrapper, args.drop_loc_defs, args.max_depth
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"Parsed {len(loc_map)} #loc definitions")
    print(f"Replaced {replaced} loc(#locNNN) references")
    if unresolved:
        print(
            f"Left {unresolved} loc(#locNNN) references unresolved (no definition found)"
        )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
