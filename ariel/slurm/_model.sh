# Shared resolution of the Stan model name for the SLURM pipeline.
# Sourced (not executed) by step5d_map.sh, step6_node.sh, step6_chain.sh,
# step7_diagnose.sh and step8_predict.sh, after each has set CONFIG.
#
# Precedence, highest first:
#   1. the MODEL environment variable
#   2. the config's "model" key
#   3. "2color"
#
# Why a shared file rather than a line in each script: step6_node.sh and
# step6_chain.sh must agree exactly -- step6_chain.sh exists to resubmit one
# failed chain alongside three that step6_node.sh already produced, so a
# different model name there would silently mix posteriors from two models.
# Sourcing one resolver is a stronger guarantee than the duplicated "2color"
# literals it replaces.
#
# SLURM gotcha: `sbatch --export=CONFIG=$CFG` without ALL restricts the job
# environment, so MODEL does NOT reach the job unless it is appended to the
# --export string, exactly the way DEBUG and MAX_DEPTH already are in
# batch_submit.sh. If a job logs a model you did not ask for, check --export
# before anything else.
#
# `exe` is deliberately not consulted. It is dead for steps 4-8, still carries
# a stale "tophat" default in config_utils.PIPELINE_DEFAULTS, and duplicates
# "model".

if [ -z "${CONFIG:-}" ]; then
    echo "ERROR: slurm/_model.sh sourced before CONFIG was set" >&2
    exit 1
fi

# A missing or unparsable config yields "", so the default below applies. This
# is deliberate: every SLURM caller reads the config's "run" key *before*
# sourcing this file, so a genuinely broken config has already failed loudly by
# the time we get here. run_dr2_onepop.sh, by contrast, sources this before the
# config exists (export_config.py writes it in step 3b), and must not die.
if [ -z "${MODEL:-}" ]; then
    MODEL=$(python3 -c "
import json, sys
try:
    print(json.load(open('$CONFIG')).get('model') or '')
except Exception:
    print('')
")
fi
MODEL=${MODEL:-2color}

# MODEL is interpolated into a command name (./${MODEL}_g), a CSV glob and an
# output filename, so refuse anything that is not a plain identifier. This is
# the actual safety guard; the allowlist below is only a tested-set warning.
case "$MODEL" in
    *[!A-Za-z0-9_]*|"")
        echo "ERROR: MODEL='$MODEL' is not a plain identifier ([A-Za-z0-9_]+);" \
             "refusing to interpolate it into a command name." >&2
        exit 1
        ;;
esac

# The tested set. Anything else is allowed through -- the pipeline has other
# model sources (2color_v7, 2color_v8, sb_v9) and a config naming one of them
# should run that model rather than silently running 2color, which is what the
# hardcoded literals used to do -- but it is called out in the log, because the
# only models with a GPU build are the ones ${MODEL}_g.stan exists for.
case "$MODEL" in
    2color|2color_v10) ;;
    *)
        echo "WARNING: MODEL=$MODEL is outside this pipeline's tested set" \
             "(2color, 2color_v10). It needs ${MODEL}_g.stan and a GPU build" \
             "from slurm/compile_gpu.sh." >&2
        ;;
esac

export MODEL
