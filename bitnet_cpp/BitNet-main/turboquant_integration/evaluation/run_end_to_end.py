#!/usr/bin/env python3
"""Collect PPL and tokens/s from actual FP16 and TurboQuant BitNet runs."""
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path


def invoke(template, mode, bits, output):
    subprocess.run(template.format(mode=mode, bit_width=bits, output=output), shell=True, check=True)
    data = json.loads(Path(output).read_text(encoding="utf-8"))
    required = {"tokens_generated", "elapsed_seconds", "nll_sum", "nll_tokens"}
    if missing := required - data.keys():
        raise ValueError(f"{output}: missing {sorted(missing)}")
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-command", required=True, help="template with {mode}, {bit_width}, {output}")
    p.add_argument("--turboquant-command", required=True, help="must run the TurboQuant KV-enabled BitNet executable")
    p.add_argument("--bits", type=int, nargs="+", default=[4, 3, 2])
    p.add_argument("--output", type=Path, default=Path("turboquant_end_to_end.json"))
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="tq-e2e-") as temp:
        baseline = invoke(a.baseline_command, "fp16", 16, Path(temp) / "baseline.json")
        results = []
        for bits in a.bits:
            q = invoke(a.turboquant_command, "turboquant", bits, Path(temp) / f"tq{bits}.json")
            results.append({"bit_width": bits, "perplexity": math.exp(q["nll_sum"] / q["nll_tokens"]), "baseline_perplexity": math.exp(baseline["nll_sum"] / baseline["nll_tokens"]), "tokens_per_second": q["tokens_generated"] / q["elapsed_seconds"], "baseline_tokens_per_second": baseline["tokens_generated"] / baseline["elapsed_seconds"], "logits_path": q.get("logits_path"), "baseline_logits_path": baseline.get("logits_path")})
    a.output.write_text(json.dumps({"baseline": baseline, "turboquant": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(a.output)


if __name__ == "__main__":
    main()
