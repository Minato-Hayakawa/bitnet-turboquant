#!/usr/bin/env python3
"""Compare baseline and TurboQuant model logits without third-party packages.

Inputs are little-endian float32 files shaped [tokens, vocab_size], written by
the same prompt set with the FP16 cache and with the TurboQuant cache.
"""
import argparse
import json
import math
import struct
from pathlib import Path


def read_logits(path: Path, vocab: int):
    raw = path.read_bytes()
    if len(raw) % (4 * vocab):
        raise ValueError(f"{path}: byte length is not divisible by 4*vocab_size")
    return struct.iter_unpack("<" + "f" * vocab, raw)


def topk(row, k):
    return [i for i, _ in sorted(enumerate(row), key=lambda x: x[1], reverse=True)[:k]]


def cosine(a, b):
    ab = sum(x * y for x, y in zip(a, b))
    aa = sum(x * x for x in a)
    bb = sum(y * y for y in b)
    return 0.0 if not aa or not bb else ab / math.sqrt(aa * bb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--turboquant", required=True, type=Path)
    p.add_argument("--vocab-size", required=True, type=int)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    base, quant = list(read_logits(a.baseline, a.vocab_size)), list(read_logits(a.turboquant, a.vocab_size))
    if len(base) != len(quant):
        raise ValueError("baseline and turboquant token counts differ")
    n = len(base)
    report = {
        "tokens": n,
        "cosine_similarity": sum(cosine(x, y) for x, y in zip(base, quant)) / n if n else 0.0,
        "top1_token_match_rate": sum(topk(x, 1)[0] == topk(y, 1)[0] for x, y in zip(base, quant)) / n if n else 0.0,
        "top5_token_match_rate": sum(topk(x, 1)[0] in topk(y, 5) for x, y in zip(base, quant)) / n if n else 0.0,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if a.output:
        a.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
