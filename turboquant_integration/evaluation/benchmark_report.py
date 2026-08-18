#!/usr/bin/env python3
"""Aggregate standard long-context benchmark outputs into one reproducible report.

Each input is a JSON or JSONL file with at least `benchmark`, `task`, and
`score` fields.  Scores must already be computed by the official benchmark
adapter; this script never substitutes a synthetic score for a dataset result.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

VALID = {"LongBench", "Needle In A Haystack", "ZeroSCROLLS", "RULER", "L-Eval"}


def records(path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    if text.startswith("{"):
        return [json.loads(text)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path, help="official benchmark JSON/JSONL output files")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    by_benchmark = defaultdict(list)
    for source in a.inputs:
        for r in records(source):
            required = {"benchmark", "task", "score"}
            if not required <= r.keys():
                raise ValueError(f"{source}: every record needs {sorted(required)}")
            if r["benchmark"] not in VALID:
                raise ValueError(f"unknown benchmark: {r['benchmark']}")
            by_benchmark[r["benchmark"]].append(r)
    summary = {"benchmarks": {}}
    for name in VALID:
        rs = by_benchmark.get(name, [])
        summary["benchmarks"][name] = {
            "status": "measured" if rs else "not_run",
            "aggregate_score": sum(float(x["score"]) for x in rs) / len(rs) if rs else None,
            "tasks": rs,
        }
    a.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
