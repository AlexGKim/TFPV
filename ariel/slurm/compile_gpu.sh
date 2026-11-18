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
# ONE AT A TIME. Do not submit two of these concurrently: every model is built
# in the SAME ../../cmdstan tree and shares its artifacts (the model_header
# precompiled header, libtbb, bin/stanc), so two makes race on the same
# targets. Submitting both together on 2026-11-18 put them on one node, where
# they reached MaxRSS 235 GB against a 229 GB request and the 2color job was
# OOM-killed compiling 2color_g.o (job 61544394) while the v10 job, which is
# the larger model, had finished alone in 1 m 59 s (job 61544396). Wait for
# one to finish before submitting the next.
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

# Modules: craype-accel-nvidia80 + cudatoolkit ONLY. Do NOT add
# `nvidia PrgEnv-nvidia` -- see the note below.
module load craype-accel-nvidia80 cudatoolkit

# Workaround for CUDA OpenCL link error on Perlmutter
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

# Why PrgEnv-nvidia is gone (2026-11-18)
# -------------------------------------
# compile_2color_gpu.sh loaded `nvidia PrgEnv-nvidia` as well. That now sets
# CXX to nvc++ (nvidia/26.5) and the build dies twice over:
#
#   1. stan/lib/stan_math/make/libraries:114: *** "Need to set TBB_CXX_TYPE
#      for non-standard compiler other than gcc or clang."
#   2. with TBB_CXX_TYPE set, it gets as far as the precompiled header and
#      then: nvc++-Error-Unknown switch: -Wno-ignored-attributes  (and -MG).
#      Stan's OpenCL build passes gcc flags that nvc++ does not accept.
#
# The fix is not to coax nvc++ through: it is that this build never used
# nvc++. cmdstan's existing precompiled headers are
# model_header_opencl_13_2.hpp.gch and model_header_opencl_14_3.hpp.gch --
# g++ 13.2 and 14.3, not nvidia 26.5 -- so the working 2color_g of 2026-08-06
# was produced by g++. PrgEnv-gnu is Perlmutter's default (Lmod reports it is
# "replacing PrgEnv-gnu/8.7.0" when the nvidia modules load), g++ is now
# 14.3.0, and its OpenCL PCH is already built.
#
# Stan's OpenCL backend needs only the bundled OpenCL headers
# (stan/lib/stan_math/lib/opencl_3.0.0) and libOpenCL to link against. The
# latter comes from cudatoolkit -- $CUDATOOLKIT_HOME/lib64/libOpenCL.so, which
# the LIBRARY_PATH line above already puts on the search path. The NVIDIA host
# compiler plays no part.
#
# NOTE the asymmetry: the step5d/5e/6 scripts still load `nvidia
# PrgEnv-nvidia`. Those are RUNTIME loads for an already-built binary, they
# are what the delivered runs used, and they are left alone deliberately.
# Only the compile needed changing.
#
# TBB_CXX_TYPE is left unset by default: with g++, Stan detects CXX_TYPE=gcc
# and sets it itself. Pass --export=...,TBB_CXX_TYPE=... only if a future
# toolchain change needs it.
TBB_ARG=""
[ -n "${TBB_CXX_TYPE:-}" ] && TBB_ARG="TBB_CXX_TYPE=${TBB_CXX_TYPE}"

# Build parallelism. 4 is fine for a model that compiles cleanly, but the
# single translation unit ${MODEL}_g.o is by far the largest thing here and
# -j lets it run alongside other heavy compiles. 2color_g.o was OOM-killed at
# -j4 while sharing a node (job 61544394) and then hit
# "g++: internal compiler error: Segmentation fault ... cc1plus" at -j4 alone
# (job 61545173). Use MAKE_J=1 when a model will not compile:
#   sbatch --export=MODEL=2color,MAKE_J=1 slurm/compile_gpu.sh
# Remember --export without ALL restricts the job environment, so MAKE_J only
# reaches the job if it is in the --export string.
MAKE_J=${MAKE_J:-4}

# Host compiler. Unset means Perlmutter's default g++, currently 14.3.0.
#
# g++ 14.3 has a REPRODUCIBLE internal compiler error on 2color.stan:
#   g++: internal compiler error: Segmentation fault signal terminated
#   program cc1plus      -> make/program:77: ../TFPV/ariel/2color_g.o  Error 4
# seen at -j4 (job 61545173, 37 s) and again at -j1 (job 61545756, 35 s,
# MaxRSS 93 MB) -- so it is a compiler bug, not a resource problem. Oddly it
# does NOT affect 2color_v10.stan, the larger model, which builds cleanly.
#
# g++ 13.2 is the known-good compiler for this tree: cmdstan already holds
# model_header_opencl_13_2.hpp.gch, so 13.2 has built OpenCL models here
# before -- almost certainly what produced the 2026-08-06 2color_g. Use:
#   sbatch --export=MODEL=2color,STAN_CXX=g++-13 slurm/compile_gpu.sh
# /usr/bin/g++-13 is 13.2.1 and needs no module load. The precompiled header
# is named after the compiler version, so this reuses the existing 13_2 PCH.
#
# NOTE for paired runs: the two arms of a v10/v5 likelihood-ratio comparison
# should be built by the SAME compiler, or they differ at roundoff for an
# uninteresting reason. If 2color is built with g++-13, rebuild 2color_v10
# with g++-13 too.
CXX_ARG=""
[ -n "${STAN_CXX:-}" ] && CXX_ARG="CXX=${STAN_CXX}"

cd ../../cmdstan
make -j${MAKE_J} STAN_OPENCL=true $TBB_ARG $CXX_ARG ../TFPV/ariel/${MODEL}_g
cd ../TFPV/ariel

echo "Compiled: $(ls -lh ${MODEL}_g)"
