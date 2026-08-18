#!/bin/bash
# TurboQuant KV evaluator: CPU-only reference run on R-CCS genoa.
# Submit from the repository root with:
#   sbatch bitnet_cpp/bitnet/turboquant_integration/hpc/submit_eval_genoa.sbatch
#SBATCH --job-name=bitnet-tq-eval
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 01:00:00
#SBATCH --output=bitnet-tq-eval-%j.out
#SBATCH --error=bitnet-tq-eval-%j.err

set -euo pipefail
source /etc/profile
module load system/genoa mpi/openmpi-x86_64

# When submitted from the repository root, this is the project root. Override
# PROJECT_ROOT when the source was copied to a different cluster directory.
PROJECT_ROOT=${PROJECT_ROOT:-"$SLURM_SUBMIT_DIR"}
RESULT_DIR="$PROJECT_ROOT/results/turboquant/${SLURM_JOB_ID}"
EVALUATOR="$RESULT_DIR/bitnet_turboquant_eval"
mkdir -p "$RESULT_DIR"

cd "$PROJECT_ROOT"
g++ -std=c++17 -O3 -march=native \
  -I turboquant_cpp/turboquant_cpp/include \
  -I bitnet_cpp/bitnet/turboquant_integration/include \
  bitnet_cpp/bitnet/turboquant_integration/src/bitnet_turboquant_eval.cpp \
  -o "$EVALUATOR"

for BITS in 4 3 2; do
  "$EVALUATOR" --bit-width "$BITS" --layers 36 --kv-heads 72 --head-dim 144 \
    --context 8192 --baseline-mb 10368 --baseline-context 8192 \
    --csv "$RESULT_DIR/turboquant_${BITS}bit_8k.csv"
done

python3 bitnet_cpp/bitnet/turboquant_integration/evaluation/run_memory_scaling.py \
  --evaluator "$EVALUATOR" \
  --output "$RESULT_DIR/turboquant_memory_scaling.csv"

echo "Results: $RESULT_DIR"
