# BitNet + TurboQuant integration and evaluation

`bitnet_turboquant_kv.hpp` is the KV-cache boundary adapter.  It stores one
TurboQuant payload per token/head and reconstructs FP32 buffers for the
attention operation.  It is deliberately an **offload/readback adapter**: it
does not claim that quantized data are consumed directly by a ggml attention
kernel.

The checked-out BitNet repository does not contain `3rdparty/llama.cpp`, so a
direct patch to `llama-kv-cache.cpp` cannot be compiled here.  Once that
submodule is initialized, wire `KVCacheAdapter::append()` at the cache
offload/eviction write boundary and `read_keys()` / `read_values()` immediately
before the relevant attention graph reads the offloaded range.  Keep the recent
window as FP16 if low decode latency matters.

## Build and measure

From `bitnet_cpp/BitNet` (after its llama.cpp submodule has been initialized):

```powershell
cmake -S . -B build -DBITNET_TURBOQUANT=ON
cmake --build build --target bitnet-turboquant-eval --config Release
.\build\bin\Release\bitnet-turboquant-eval.exe --bit-width 3 --layers 36 --kv-heads 72 --context 8192 --baseline-mb 289 --csv turboquant_3bit.csv
```

Run the executable once for 4, 3, and 2 bit.  The CSV/stdout report the actual
TurboQuant payload size (including per-vector FP32 scale metadata) relative to
the requested baseline.  It also runs a deterministic synthetic attention
probe.  Measure real token fidelity by dumping little-endian FP32 logits from
the identical prompts (baseline and quantized cache) and running:

```powershell
python .\turboquant_integration\evaluation\analyze_logits.py --baseline fp16_logits.f32 --turboquant tq3_logits.f32 --vocab-size 128256 --output fidelity_3bit.json
```

Long-context benchmark adapters should invoke the model through the same
TurboQuant cache mode and record the normal task metric:

| Benchmark | Record |
| --- | --- |
| LongBench | task-specific score and aggregate |
| Needle In A Haystack | retrieval accuracy by context length/depth |
| ZeroSCROLLS | zero-shot task score |
| RULER | pass rate by length and synthetic task |
| L-Eval | aggregate and per-task score |

The CLI validates cache compression and attention-output fidelity.  It cannot
run these dataset benchmarks without the model checkpoint, tokenizer, and
datasets.  Run each official adapter with the TurboQuant cache enabled, export
records with `benchmark`, `task`, and `score`, then produce one report:

```powershell
python .\turboquant_integration\evaluation\benchmark_report.py longbench.jsonl needle.jsonl zeroscrolls.jsonl ruler.jsonl leval.jsonl --output long_context_report.json
```

## 10,368 MiB / long-context experiment

The 10,368 MiB FP16 reference at 8K corresponds to 36 layers, 72 KV heads,
and head dimension 144. Build the evaluator, then run all 15 combinations
(2/3/4 bit × 2K/4K/8K/16K/32K):

```powershell
python .\turboquant_integration\evaluation\run_memory_scaling.py --evaluator .\bitnet_turboquant_eval.exe --output turboquant_memory_scaling.csv
```

`compressed_mb` includes TurboQuant's padding and per-vector metadata;
`ideal_compressed_mb` is the raw-bit theoretical lower bound.

## End-to-End PPL and speed experiment

Use a baseline and TurboQuant command template that write JSON containing
`tokens_generated`, `elapsed_seconds`, `nll_sum`, and `nll_tokens` to
`{output}`. Both must use exactly the same model, prompts, and decoding setup.

```powershell
python .\turboquant_integration\evaluation\run_end_to_end.py --baseline-command "<FP16 BitNet command> --report {output}" --turboquant-command "<TurboQuant-enabled BitNet command> --bit-width {bit_width} --report {output}" --output turboquant_end_to_end.json
```

The command runner only aggregates measurements from a genuine TurboQuant KV
backend; it does not label an unmodified FP16 BitNet execution as quantized.
