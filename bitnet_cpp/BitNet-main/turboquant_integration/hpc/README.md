# Cluster launch profiles

Run a GPU partition, for example:

```bash
source bitnet_cpp/bitnet/turboquant_integration/hpc/profiles/nvidia.env
sbatch --partition=a100 --gpus=1 --export=ALL,PARTITION=a100 bitnet_cpp/bitnet/turboquant_integration/hpc/submit_turboquant.sbatch
```

Use `amd.env` for MI100/MI250/MI300, `intel.env` for PVC, and `cpu.env` for
FX700/EPYC/R340. The launcher selects the documented module and backend. For
`ai-h100l-pu`, set `--time=00:30:00`; for B300, H200, L40S and QC partitions,
request GPU count with `--gpus=N`.

The memory-scaling CSV runs now. E2E deliberately requires the planned
`llama-turboquant-bench` binary, which only exists after the ggml fixed-block
type and backend kernels are implemented. This avoids invalid FP16-as-TurboQuant
results.

## Current evaluator (ready now)

The current C++ evaluator is CPU-only, so submit it to `genoa` without a GPU:

```bash
sbatch bitnet_cpp/bitnet/turboquant_integration/hpc/submit_eval_genoa.sbatch
```

Check it with `squeue -j <job-id>`. Results are written below
`results/turboquant/<job-id>/`, including all 2/3/4-bit 8K CSVs and the
2K–32K scaling CSV. `submit_eval_a100.sbatch` is only for node comparison and
does not request an A100 because this particular evaluator has no CUDA kernel.
