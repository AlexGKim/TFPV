#!/bin/zsh
# Run the 2color pipeline locally (no SLURM) for one or more subset configs.
#
# This is the laptop/no-scheduler counterpart to slurm/batch_submit.sh. It
# mirrors that chain step for step so results are comparable to a NERSC batch
# run: step4 -> step6 (4 chains) -> step7 -> step8. There is no step5d when the
# config sets "fixed_init" (step4 writes init_MAP.json directly); see
# BATCH_MOCKS.md decision #2b.
#
# Differences from the SLURM path that are inherent to running locally:
#   - uses the CPU binary ./2color, not the GPU ./2color_g
#   - the 4 chains run as background processes on this machine rather than
#     one-per-GPU on an allocated node
#
# Usage:
#   zsh run_subsets.sh                # subsets 01 02 03 04 (default)
#   zsh run_subsets.sh 00             # just s00
#   zsh run_subsets.sh 00 01 02 03 04 # all five
#
# s00 is excluded from the default list because it is normally run standalone
# first as the validation run (BATCH_MOCKS.md "Step 0"). Pass it explicitly to
# include it.

set -e

SUBSETS=("$@")
if [ ${#SUBSETS[@]} -eq 0 ]; then
    SUBSETS=(01 02 03 04)
fi

# Sampler settings: keep in sync with slurm/step6_node.sh's non-debug defaults
# so local runs are science-grade and comparable to batch runs. 1000 warmup and
# max_depth=10 replaced an earlier 250/8, which produced 14.75% max-treedepth
# transitions and 2.5% divergences on the abacus mock. delta=0.9 (up from
# Stan's 0.8) controls the divergence rate without 0.95's steep wall-clock
# blow-up. No metric_file= : metric seeding was measured ~2.7x slower overall
# than adapting dense_e from identity, with no quality gain (decision #3).
NUM_WARMUP=${NUM_WARMUP:-1000}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
MAX_DEPTH=${MAX_DEPTH:-10}
DELTA=${DELTA:-0.9}

echo "Subsets: ${SUBSETS[@]}"
echo "Sampler: num_warmup=$NUM_WARMUP num_samples=$NUM_SAMPLES max_depth=$MAX_DEPTH delta=$DELTA"
echo

for i in "${SUBSETS[@]}"; do
    CONFIG="configs/abacus_subsets/abacus_2color_s${i}.json"
    RUN=$(python -c "import json; print(json.load(open('$CONFIG'))['run'])")
    echo "================ subset $i (run=$RUN) ================"

    # --- Step 4: data prep -------------------------------------------------
    # Always re-run so input.json matches the *current* config (n_subsets,
    # subset_index, n_objects). Skipping it after a config edit silently reuses
    # a stale partition; desi_data.py warns on a metadata mismatch, but that
    # check cannot detect a change to the partitioning *code*, so re-running
    # unconditionally is the safe default.
    echo "[step4] desi_data.py"
    python desi_data.py --config "$CONFIG"

    # --- Step 5d: MAP, only when the config has no fixed_init --------------
    # With "fixed_init" set, step4 already wrote init_MAP.json by transforming
    # frozen physical-unit values into this run's standardized coordinates —
    # no optimizer needed. Otherwise fall back to the MAP fit, using
    # make_map_init.py rather than an inline reimplementation (an earlier
    # inline copy here only overwrote keys that appear as optimize.csv
    # columns, so slope_orig/intercept_orig silently kept their stale pre-MAP
    # values and the rank-1 loading w was never refined).
    if [ -f "output/$RUN/init_MAP.json" ]; then
        echo "[step5d] skipped — init_MAP.json already written by step4 (fixed_init)"
    else
        echo "[step5d] 2color optimize + make_map_init.py"
        ./2color optimize \
            data file="output/$RUN/input.json" \
            init="output/$RUN/init.json" \
            output file="output/$RUN/optimize.csv"
        python make_map_init.py --run "$RUN"
    fi

    # --- Step 6: MCMC, 4 chains in parallel --------------------------------
    echo "[step6] 4 chains"
    for CHAIN in 1 2 3 4; do
        ./2color sample num_warmup=$NUM_WARMUP num_samples=$NUM_SAMPLES \
            adapt delta=$DELTA save_metric=1 \
            algorithm=hmc engine=nuts max_depth=$MAX_DEPTH metric=dense_e \
            id=$CHAIN \
            data file="output/$RUN/input.json" \
            init="output/$RUN/init_MAP.json" \
            output file="output/$RUN/2color_${CHAIN}.csv" &
    done
    wait
    echo "[step6] done"

    # --- Step 7: diagnostics ----------------------------------------------
    echo "[step7] stansummary / diagnose / corner"
    ../../cmdstan/bin/stansummary output/$RUN/2color_?.csv > "output/$RUN/stansummary.txt"
    ../../cmdstan/bin/diagnose    output/$RUN/2color_?.csv > "output/$RUN/diagnose.txt"
    python corner.py --run "$RUN" --model 2color

    # --- Step 8: predictions, then residuals ------------------------------
    # Order matters: color_predict.py first so the catalog and covariance are
    # already written before any plotting is attempted. Under `set -e` a
    # residual-plot failure would otherwise cost the run its science output.
    # Set NO_COV=1 to skip the O(G^2) covariance (~1.17 GB per slice-scoped
    # run) when only the plots are wanted.
    echo "[step8] color_predict.py"
    if [ "${NO_COV:-0}" = "1" ]; then
        python color_predict.py --config "$CONFIG" --model 2color --no-cov
    else
        python color_predict.py --config "$CONFIG" --model 2color
    fi
    echo "[step8] explore_residuals.py"
    python explore_residuals.py --config "$CONFIG" --kind 2color

    echo "================ subset $i complete ================"
    echo
done

echo "All requested subsets complete: ${SUBSETS[@]}"
