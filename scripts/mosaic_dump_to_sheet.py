#!/usr/bin/env python3
"""Parse a Mosaic dump and emit a CSV suitable for Google Sheets diagrams."""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class OpRecord:
    name: str
    raw: str
    op_type: str


DMA_PATTERNS = re.compile(r"\b(dma|async_copy|copy)\b", re.IGNORECASE)


def classify_op(raw_line: str) -> str:
    if DMA_PATTERNS.search(raw_line):
        return "DMA"
    return "Compute"


def extract_last_step(lines: Iterable[str]) -> List[str]:
    line_list = list(lines)
    indices = [
        idx for idx, line in enumerate(line_list)
        if "post-finalize-llo" in line
    ]
    if not indices:
        raise ValueError("No 'post-finalize-llo' step found in dump.")
    start = indices[-1] + 1
    return line_list[start:]


def parse_ops(lines: Iterable[str]) -> List[OpRecord]:
    ops: List[OpRecord] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("//", "#")):
            continue
        if stripped.startswith("-----"):
            continue
        if " = " in stripped or stripped.startswith("%"):
            name = stripped.split(" = ", 1)[0].strip()
            name = name.lstrip("%")
            ops.append(OpRecord(name=name, raw=stripped, op_type=classify_op(stripped)))
    if not ops:
        raise ValueError("No operations found in the last post-finalize-llo step.")
    return ops


def write_csv(ops: List[OpRecord], output_path: Path) -> None:
    header = ["Type"] + [op.name or f"op_{idx + 1}" for idx, op in enumerate(ops)]
    dma_row = ["DMA"]
    compute_row = ["Compute"]
    for op in ops:
        if op.op_type == "DMA":
            dma_row.append("●")
            compute_row.append("")
        else:
            dma_row.append("")
            compute_row.append("●")
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(dma_row)
        writer.writerow(compute_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the last post-finalize-llo step from a Mosaic dump and "
            "emit a CSV for a DMA/Compute diagram."
        ))
    parser.add_argument(
        "dump_path",
        type=Path,
        help="Path to post-finalize-llo.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mosaic_dma_compute_diagram.csv"),
        help="Output CSV path for Google Sheets (default: %(default)s).",
    )
    args = parser.parse_args()

    lines = args.dump_path.read_text(encoding="utf-8").splitlines()
    last_step_lines = extract_last_step(lines)
    ops = parse_ops(last_step_lines)
    write_csv(ops, args.output)

    print(f"Wrote {len(ops)} operations to {args.output}")


if __name__ == "__main__":
    main()
