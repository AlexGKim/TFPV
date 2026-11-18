#!/bin/bash
# Backwards-compatible wrapper, kept because BATCH_NERSC.md documents this name.
# The real script is slurm/compile_gpu.sh, which takes MODEL (default 2color):
#   sbatch --export=MODEL=2color_v10 slurm/compile_gpu.sh
exec bash slurm/compile_gpu.sh "$@"
