#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J compile_gpu
#SBATCH --mail-type=FAIL

# One-time per model: compile a Stan model with GPU (OpenCL) support.
# Run from the ariel/ directory:
#
#   sbatch --export=MODEL=2color     slurm/compile_gpu.sh
#   sbatch --export=MODEL=2color_v10 slurm/compile_gpu.sh
#
# MODEL defaults to 2color. slurm/compile_2color_gpu.sh is kept as a one-line
# wrapper for that default, so BATCH_NERSC.md's documented command still works.
#
# The target is ${MODEL}_g, built from ${MODEL}_g.stan -- a tracked symlink to
# ${MODEL}.stan. The separate _g name exists so the GPU and CPU binaries of the
# same model can coexist in one directory; the symlinks are tracked because git
# stores them as 20-byte blobs and a fresh checkout without them cannot compile
# at all.
#
# STAN_OPENCL=true is passed on the make line and deliberately NOT put in
# cmdstan/make/local. That file DOES exist on Perlmutter (checked 2026-11-18)
# but leaves STAN_OPENCL commented out; its only active settings are
# `STANCFLAGS+= --warn-pedantic` and `CXXFLAGS += -fpermissive`. Keep it that
# way: a global flag would also turn OpenCL on for the CPU ./${MODEL} that
# run_dr2_onepop.sh builds, and would break the macOS arm64 build.

set -e

MODEL=${MODEL:-2color}

case "$MODEL" in
    *[!A-Za-z0-9_]*|"")
        echo "ERROR: MODEL='$MODEL' is not a plain identifier ([A-Za-z0-9_]+)." >&2
        exit 1
        ;;
esac

if [ ! -e "${MODEL}_g.stan" ]; then
    echo "ERROR: ${MODEL}_g.stan not found in $(pwd)." >&2
    echo "       Create it as a symlink to the model source and track it:" >&2
    echo "         ln -s ${MODEL}.stan ${MODEL}_g.stan && git add ${MODEL}_g.stan" >&2
    exit 1
fi

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia

# Workaround for CUDA OpenCL link error on Perlmutter
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

cd ../../cmdstan
make -j4 STAN_OPENCL=true ../TFPV/ariel/${MODEL}_g
cd ../TFPV/ariel

echo "Compiled: $(ls -lh ${MODEL}_g)"
