#!/usr/bin/env python3
"""Run the 10,368-MiB FP16 TurboQuant scaling study into one CSV."""
import argparse
import csv
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluator", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("turboquant_memory_scaling.csv"))
    p.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    p.add_argument("--bits", type=int, nargs="+", default=[4, 3, 2])
    a = p.parse_args()
    rows = []
    for context in a.contexts:
        for bits in a.bits:
            temporary = a.output.parent / f"_tq_{bits}_{context}.csv"
            subprocess.run([str(a.evaluator), "--bit-width", str(bits), "--layers", "36", "--kv-heads", "72", "--head-dim", "144", "--context", str(context), "--baseline-mb", "10368", "--baseline-context", "8192", "--csv", str(temporary)], check=True)
            with temporary.open(newline="", encoding="utf-8") as f:
                rows.extend(r for r in csv.DictReader(f) if r["kind"] == "compression")
            temporary.unlink()
    fields = ["bit_width", "context", "baseline_mb", "compressed_mb", "ratio", "ideal_compressed_mb", "ideal_ratio"]
    with a.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    print(f"wrote {a.output} ({len(rows)} measurements)")


if __name__ == "__main__":
    main()
