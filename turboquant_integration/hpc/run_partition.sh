#!/usr/bin/env bash
# Build and launch the BitNet + TurboQuant experiment on one RIKEN partition.
set -euo pipefail

PARTITION=${PARTITION:?set PARTITION (for example: a100)}
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
MODEL=${MODEL:-"$ROOT/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"}
BUILD=${BUILD:-"$ROOT/build-${PARTITION}"}
CONTEXT=${CONTEXT:-8192}
BITS=${BITS:-3}
GPU_LAYERS=${GPU_LAYERS:-999}

case "$PARTITION" in
  a100|b300|ai-h100l|ai-h100l-pu|ai-h200-brc|ai-l40s|qc-a100|qc-gh200|ng-dgx-s|ng-dgx-m0|ng-dgx-m1|ng-dgx-m2|ng-dgx-m3)
    module load "system/${PARTITION%%-*}" nvhpc 2>/dev/null || module load "system/$PARTITION" nvhpc
    BACKEND=(-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA=ON)
    ;;
  mi100|qc-mi210|qc-mi250|fs-mi300a|fs-mi300x)
    module load "system/$PARTITION" rocm
    BACKEND=(-DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON)
    ;;
  qc-pvc)
    module load system/qc-pvc 2>/dev/null || source /opt/intel/oneapi/setvars.sh
    BACKEND=(-DGGML_SYCL=ON -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_TARGET=INTEL)
    ;;
  fx700)
    module load system/fx700 FJSVstclanga
    BACKEND=(-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON)
    ;;
  genoa|genoa-m)
    module load system/genoa mpi/openmpi-x86_64
    BACKEND=(-DGGML_NATIVE=ON -DGGML_CPU_ALL_VARIANTS=ON)
    GPU_LAYERS=0
    ;;
  r340|qc-pro6000)
    BACKEND=(-DGGML_NATIVE=ON)
    GPU_LAYERS=0
    ;;
  *) echo "Unsupported partition: $PARTITION" >&2; exit 2;;
esac

test -f "$MODEL" || { echo "Model not found: $MODEL" >&2; exit 2; }
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBITNET_TURBOQUANT=ON "${BACKEND[@]}"
cmake --build "$BUILD" --config Release --parallel "${SLURM_CPUS_PER_TASK:-16}"

# The memory-only experiment is available immediately.
python3 "$ROOT/turboquant_integration/evaluation/run_memory_scaling.py" \
  --evaluator "$BUILD/bin/bitnet-turboquant-eval" \
  --output "turboquant_memory_${PARTITION}.csv"

# End-to-end mode must be provided by the forthcoming tqkv ggml backend. Do
# not silently substitute the FP16 cache: that would invalidate PPL/NIAH data.
if [[ ! -x "$BUILD/bin/llama-turboquant-bench" ]]; then
  echo "TurboQuant ggml backend is not built; memory study completed, E2E intentionally not run." >&2
  exit 3
fi
"$BUILD/bin/llama-turboquant-bench" --model "$MODEL" --ctx-size "$CONTEXT" \
  --gpu-layers "$GPU_LAYERS" --kv-turboquant "$BITS" --all-experiments \
  --report "turboquant_e2e_${PARTITION}_${BITS}bit.json"
