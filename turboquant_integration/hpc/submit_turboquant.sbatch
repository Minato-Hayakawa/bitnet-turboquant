#!/usr/bin/env bash
# Usage: sbatch --partition=a100 --gpus=1 --export=ALL,PARTITION=a100 submit_turboquant.sbatch
# For CPU partitions omit --gpus. Choose walltime/memory according to policy.
#SBATCH --job-name=bitnet-tq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=bitnet-tq-%j.out
#SBATCH --error=bitnet-tq-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch}"
bash bitnet_cpp/bitnet/turboquant_integration/hpc/run_partition.sh
