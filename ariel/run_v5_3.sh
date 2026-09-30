#!/bin/bash
# v5_3 = the DR2_TF v5 two-population 2color run with z_obs_min raised 0.01 -> 0.03.
# Configs differ from v5 in "run" and "z_obs_min" only.
#
# Step 4 (desi_data.py) has ALREADY been run for both populations:
# output/$RUN/{input.json,init.json,config.json,data.png} exist.
# This script runs Steps 5d, 5e (Pathfinder metric), 6, 7, 8 and the combine.
#
# METRIC NOTE — why this differs from DR2_TWOPOP.md Step 6.
# The doc says Step 6 runs from the identity metric and that make_pf_metric.py
# "is not a step here". But the scripts that actually produced v5 --
# output/DR2_TF_{spirals,irrs}_v5_2color_*/run_step6.sh -- both pass
#   metric=dense_e metric_file=output/$RUN/pf_metric.json
# and their chain logs confirm it. v5 spiral chain1: 4089 s warmup + 2258 s
# sampling = 6347 s (3.17 s/iteration). An identity-metric v5_3 attempt on
# 2026-09-28 had not reached iteration 100 after 27 min (>16 s/iteration, ~5x
# slower). So this script reproduces v5's ACTUAL procedure, which is both the
# correct comparison and far faster.
#
# Pathfinder arguments are taken from the v5 pathfinder.csv headers. Note the two
# populations genuinely differed in v5 and that difference is preserved here:
#   spiral    pathfinder init = init.json
#   irregular pathfinder init = init_MAP.json
# Both then sample from init_MAP.json.
#
# MUST run on macOS: ./2color and ../../cmdstan are arm64 Mach-O and cannot
# execute in the Claude device shell (Linux aarch64).
#
#   cd ~/Projects/TFPV/ariel
#   pkill -f '2color sample'          # stop any earlier attempt first
#   nohup bash run_v5_3.sh > output/run_v5_3.log 2>&1 &
set -u
cd "$(dirname "$0")"
CMDSTAN=../../cmdstan

if pgrep -f '2color sample' > /dev/null; then
    echo "ERROR: '2color sample' is already running. Stop it first:  pkill -f '2color sample'"
    exit 1
fi

for SPEC in spirals:init.json irrs:init_MAP.json; do
  TAG=${SPEC%%:*}; PF_INIT=${SPEC##*:}
  if [ "$TAG" = spirals ]; then RUN=DR2_TF_spirals_v5_3_2color_spiral
  else                          RUN=DR2_TF_irrs_v5_3_2color_irregular; fi
  CONFIG=configs/$(echo $RUN | tr '[:upper:]' '[:lower:]').json
  echo "=== [$(date)] $RUN ==="

  # ---- Step 5d: MAP init
  ./2color optimize \
      data file=output/$RUN/input.json \
      init=output/$RUN/init.json \
      output file=output/$RUN/optimize.csv          > output/$RUN/step5d.log 2>&1
  python3 make_map_init.py --run $RUN              >> output/$RUN/step5d.log 2>&1
  echo "[$(date)] $RUN MAP done"

  # ---- Step 5e: Pathfinder-seeded dense_e metric (as v5 actually ran)
  ./2color pathfinder \
      num_paths=1 num_draws=200 num_elbo_draws=10 num_psis_draws=200 \
      history_size=30 max_lbfgs_iters=100 \
      data file=output/$RUN/input.json \
      init=output/$RUN/$PF_INIT \
      output file=output/$RUN/pathfinder.csv        > output/$RUN/step5e.log 2>&1
  python3 make_pf_metric.py --run $RUN --stepsize 0.1 >> output/$RUN/step5e.log 2>&1
  if [ ! -s output/$RUN/pf_metric.json ]; then
      echo "ERROR: pf_metric.json not produced for $RUN - see output/$RUN/step5e.log"; exit 1
  fi
  echo "[$(date)] $RUN pathfinder metric done"

  # ---- Step 6: 4 parallel chains, Pathfinder dense_e metric
  rm -f output/$RUN/2color_?.csv output/$RUN/2color_?_metric.json output/$RUN/chain*.log
  PIDS=()
  for CHAIN_ID in 1 2 3 4; do
      ./2color sample num_warmup=1000 num_samples=1000 \
          adapt delta=0.9 save_metric=1 \
          algorithm=hmc engine=nuts max_depth=10 \
          metric=dense_e metric_file=output/$RUN/pf_metric.json \
          id=$CHAIN_ID \
          data file=output/$RUN/input.json \
          init=output/$RUN/init_MAP.json \
          output file=output/$RUN/2color_${CHAIN_ID}.csv \
          > output/$RUN/chain${CHAIN_ID}.log 2>&1 &
      PIDS+=($!)
  done
  FAIL=0; for PID in "${PIDS[@]}"; do wait "$PID" || FAIL=1; done
  [ "$FAIL" = 1 ] && { echo "ERROR: a chain failed in $RUN"; exit 1; }
  echo "[$(date)] $RUN chains done"

  # ---- Step 7: diagnostics
  $CMDSTAN/bin/stansummary output/$RUN/2color_?.csv > output/$RUN/stansummary.txt
  $CMDSTAN/bin/diagnose    output/$RUN/2color_?.csv > output/$RUN/diagnose.txt
  python3 corner.py --run $RUN --model 2color       > output/$RUN/step7_corner.log 2>&1 || \
      echo "  (corner.py failed - chainconsumer missing? non-fatal)"

  # ---- Step 8: posterior-predictive catalog + covariance
  python3 color_predict.py --config $CONFIG --model 2color > output/$RUN/step8.log 2>&1
  grep "Loaded d_err_r" output/$RUN/step8.log || \
      echo "  WARNING: d_err_r did not come from the dust pickle - check step8.log"
done

# ---- Combine the two populations (adds the cross-population systematic terms)
python3 combine_color_xonly.py \
    --spiral-run    DR2_TF_spirals_v5_3_2color_spiral \
    --irregular-run DR2_TF_irrs_v5_3_2color_irregular \
    --out-run       DR2_TF_v5_3_2color_combined \
    --catalog-name  DESI-DR2_TF_pv_cat_v5_3.fits \
    --spiral-config    configs/dr2_tf_spirals_v5_3_2color_spiral.json \
    --irregular-config configs/dr2_tf_irrs_v5_3_2color_irregular.json \
    > output/combine_v5_3.log 2>&1
echo "=== [$(date)] v5_3 complete -> output/DR2_TF_v5_3_2color_combined/ ==="
