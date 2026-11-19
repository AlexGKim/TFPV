# Shared GPU runtime environment, sourced by every script that EXECUTES a
# ${MODEL}_g binary: step5d_map.sh, step5e_metric.sh, step6_node.sh and
# step6_chain.sh. One file, so step6_node and step6_chain cannot drift apart.
#
# The LD_LIBRARY_PATH line is load-bearing. Without it every OpenCL binary
# segfaults at startup on Perlmutter, for every model:
#
#   ./2color_g     optimize   -> SIGSEGV   (job 61584352)
#   ./2color_g     pathfinder -> SIGSEGV   (job 61584353)
#   ./2color_v10_g optimize   -> SIGSEGV   (job 61583060)
#   ./2color_v10_g pathfinder -> SIGSEGV   (job 61583061)
#
# Why. The OpenCL ICD loader (libOpenCL.so.1, from cudatoolkit) reads
# /etc/OpenCL/vendors/nvidia.icd, which names "libnvidia-opencl.so.1" with no
# path, and dlopens it. On the GPU nodes that library exists --
# /usr/lib64/libnvidia-opencl.so.1 -> libnvidia-opencl.so.580.196.07 -- but it
# is NOT in the ldconfig cache, so the dlopen fails and the crash follows at
# context creation, before any computation. /etc/OpenCL/vendors and the driver
# libraries were both last touched in the 2026-10/11 node image refresh, which
# is consistent with this having worked in August and not now.
#
# Verified on a GPU node (job 61587991), same binary, same data, three runs:
#   baseline                               -> SIGSEGV, exit 139
#   LD_LIBRARY_PATH=/usr/lib64:...         -> exit 0, optimizes normally
#   ... plus explicit opencl platform/device -> exit 0
# The binaries were never at fault; nothing needs rebuilding.
#
# Note the pre-existing `export LIBRARY_PATH=...` line these scripts carried is
# NOT the same thing and never did anything here: LIBRARY_PATH is the
# compile/link-time search path, meaningful in slurm/compile_gpu.sh where it
# was copied from, and inert at runtime. It is kept, harmlessly, so the
# compile-time workaround stays discoverable in one idiom across the scripts.

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia

# Link-time path (inert at runtime; see above).
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

# Runtime path: lets the ICD loader find the NVIDIA OpenCL driver library.
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
