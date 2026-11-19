#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -J compile_cpu
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/compile_cpu_%j.out

# Compile a Stan model for CPU (no OpenCL). Run from the ariel/ directory:
#
#   sbatch --export=MODEL=2color     slurm/compile_cpu.sh
#   sbatch --export=MODEL=2color_v10 slurm/compile_cpu.sh
#
# ONE AT A TIME -- every model shares the ../../cmdstan tree and its artifacts,
# so two concurrent makes race. See slurm/compile_gpu.sh.
#
# Why this exists
# ---------------
# 1. Plan section 6d's GPU-vs-CPU agreement gate needs an x86-64 CPU reference
#    binary. A laptop build cannot serve: NERSC is x86-64, the Mac is arm64.
# 2. As of 2026-11-19 the GPU binaries segfault at OpenCL initialisation --
#    ./2color_g and ./2color_v10_g both die with SIGSEGV in `optimize` and in
#    `pathfinder`, immediately after the argument dump (jobs 61583060,
#    61583061, 61584352, 61584353). A CPU build isolates the cause: if CPU
#    runs, the fault is in the OpenCL path; if CPU crashes too, it is the
#    toolchain or the model/data.
# 3. It is a usable fallback. The validated abacus run took ~6.4 h/chain on
#    CPU, so Phase 1 can proceed without GPUs if it has to.
#
# This target is ${MODEL}, built from the tracked ${MODEL}.stan -- NOT the
# ${MODEL}_g symlink, and with no STAN_OPENCL, so the OpenCL code path is
# absent from the binary rather than merely unused.
#
# NOTE: `2color` is a tracked file in this repo (a historical mistake -- it is
# a macOS arm64 binary that should never have been committed). Building it
# here overwrites it, so `git status` on NERSC will show it modified. That is
# expected; `git checkout -- 2color` restores the committed one.

set -e
source /global/common/software/desi/perlmutter/desiconda/20260227-2.3.1/conda/etc/profile.d/conda.sh
conda activate /global/cfs/projectdirs/desi/users/akim/conda/envs/TFPV

MODEL=${MODEL:-2color}

case "$MODEL" in
    *[!A-Za-z0-9_]*|"")
        echo "ERROR: MODEL='$MODEL' is not a plain identifier ([A-Za-z0-9_]+)." >&2
        exit 1
        ;;
esac

if [ ! -e "${MODEL}.stan" ]; then
    echo "ERROR: ${MODEL}.stan not found in $(pwd)." >&2
    exit 1
fi

# Same compiler as the GPU build, for comparability: Perlmutter's default g++
# is 14.3.0, which has a reproducible ICE on 2color.stan (see
# slurm/compile_gpu.sh). /usr/bin/g++-13 is 13.2.1 and is the compiler this
# cmdstan tree's prebuilt headers were made with.
STAN_CXX=${STAN_CXX:-g++-13}
MAKE_J=${MAKE_J:-1}

mkdir -p slurm/logs
echo "Compiling CPU binary: MODEL=$MODEL CXX=$STAN_CXX -j$MAKE_J"

cd ../../cmdstan
make -j${MAKE_J} CXX=${STAN_CXX} ../TFPV/ariel/${MODEL}
cd ../TFPV/ariel

echo "Compiled: $(ls -lh ${MODEL})"
file ${MODEL} 2>/dev/null || true
