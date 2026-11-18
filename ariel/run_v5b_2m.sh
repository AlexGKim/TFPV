#!/bin/bash
# v5b_2m = the DR2_TF v5 two-population 2color run (which produced DESI-DR2_TF_pv_cat_v5b.fits)
# with the TRAINING absolute magnitudes moved to 2M++ peculiar-velocity-corrected distances.
#
# Inputs:  data/DR2_TF_{spirals,irrs}_v5_2m_{spiral,irregular}.fits   (make_pv_corrected_input.py)
# Configs: configs/dr2_tf_{spirals,irrs}_v5b_2m_2color_{spiral,irregular}.json
#          = the v5 configs with only "run" and "fits_file" changed (all cuts identical).
#
# Procedure is run_v5_3.sh's (which reproduces v5's ACTUAL steps: Pathfinder dense_e metric,
# spirals' pathfinder from init.json, irregulars' from init_MAP.json, sampling from init_MAP.json)
# plus Step 4 (desi_data.py) at the start, and the two populations run IN PARALLEL (8 chains on
# 12 cores). After the combine, finalize_v5b_2m.py converts the raw catalogue to the agreed
# convention: LOGDIST against the OBSERVED redshift (as v5b) with PV-corrected extras.
#
# MUST run on macOS (arm64 ./2color and ../../cmdstan).
#   cd ~/Projects/TFPV/ariel
#   nohup bash run_v5b_2m.sh > output/run_v5b_2m.log 2>&1 &
set -u
cd "$(dirname "$0")"
CMDSTAN=../../cmdstan

if pgrep -f '2color sample' > /dev/null; then
    echo "ERROR: '2color sample' is already running. Stop it first:  pkill -f '2color sample'"
    exit 1
fi

run_population () {
  local TAG=$1 PF_INIT=$2 RUN CONFIG
  if [ "$TAG" = spirals ]; then RUN=DR2_TF_spirals_v5b_2m_2color_spiral
  else                          RUN=DR2_TF_irrs_v5b_2m_2color_irregular; fi
  CONFIG=configs/$(echo $RUN | tr '[:upper:]' '[:lower:]').json
  mkdir -p output/$RUN
  echo "=== [$(date)] $RUN start ==="

  # ---- Step 4: Stan data build (cuts, 5000-galaxy training draw with seed 42, init.json)
  python3 desi_data.py --config $CONFIG                 > output/$RUN/step4.log 2>&1 || { echo "ERROR step4 $RUN"; return 1; }
  [ -s output/$RUN/input.json ] || { echo "ERROR: input.json not produced for $RUN"; return 1; }
  echo "[$(date)] $RUN step4 done"

  # ---- Step 5d: MAP init
  ./2color optimize \
      data file=output/$RUN/input.json \
      init=output/$RUN/init.json \
      output file=output/$RUN/optimize.csv              > output/$RUN/step5d.log 2>&1
  python3 make_map_init.py --run $RUN                  >> output/$RUN/step5d.log 2>&1 || { echo "ERROR step5d $RUN"; return 1; }
  echo "[$(date)] $RUN MAP done"

  # ---- Step 5e: Pathfinder-seeded dense_e metric
  ./2color pathfinder \
      num_paths=1 num_draws=200 num_elbo_draws=10 num_psis_draws=200 \
      history_size=30 max_lbfgs_iters=100 \
      data file=output/$RUN/input.json \
      init=output/$RUN/$PF_INIT \
      output file=output/$RUN/pathfinder.csv            > output/$RUN/step5e.log 2>&1
  python3 make_pf_metric.py --run $RUN --stepsize 0.1 >> output/$RUN/step5e.log 2>&1
  [ -s output/$RUN/pf_metric.json ] || { echo "ERROR: pf_metric.json not produced for $RUN - see step5e.log"; return 1; }
  echo "[$(date)] $RUN pathfinder metric done"

  # ---- Step 6: 4 parallel chains
  rm -f output/$RUN/2color_?.csv output/$RUN/2color_?_metric.json output/$RUN/chain*.log
  local PIDS=() CHAIN_ID PID FAIL=0
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
  for PID in "${PIDS[@]}"; do wait "$PID" || FAIL=1; done
  [ "$FAIL" = 1 ] && { echo "ERROR: a chain failed in $RUN"; return 1; }
  echo "[$(date)] $RUN chains done"

  # ---- Step 7: diagnostics
  $CMDSTAN/bin/stansummary output/$RUN/2color_?.csv > output/$RUN/stansummary.txt
  $CMDSTAN/bin/diagnose    output/$RUN/2color_?.csv > output/$RUN/diagnose.txt
  python3 corner.py --run $RUN --model 2color       > output/$RUN/step7_corner.log 2>&1 || \
      echo "  (corner.py failed - non-fatal)"

  # ---- Step 8: posterior-predictive catalogue + covariance
  python3 color_predict.py --config $CONFIG --model 2color > output/$RUN/step8.log 2>&1 || { echo "ERROR step8 $RUN"; return 1; }
  grep "Loaded d_err_r" output/$RUN/step8.log || echo "  WARNING: d_err_r did not come from the dust pickle - check step8.log"
  echo "=== [$(date)] $RUN complete ==="
}

run_population spirals init.json     > output/run_v5b_2m_spirals.log 2>&1 &
P1=$!
run_population irrs    init_MAP.json > output/run_v5b_2m_irrs.log    2>&1 &
P2=$!
FAIL=0
wait $P1 || FAIL=1
wait $P2 || FAIL=1
cat output/run_v5b_2m_spirals.log output/run_v5b_2m_irrs.log
[ "$FAIL" = 1 ] && { echo "ERROR: a population failed; not combining"; exit 1; }

# ---- Step 9: combine the two populations (adds the cross-population systematic terms)
python3 combine_color_xonly.py \
    --spiral-run    DR2_TF_spirals_v5b_2m_2color_spiral \
    --irregular-run DR2_TF_irrs_v5b_2m_2color_irregular \
    --out-run       DR2_TF_v5b_2m_2color_combined \
    --catalog-name  DESI-DR2_TF_pv_cat_v5b_2m_raw.fits \
    --spiral-config    configs/dr2_tf_spirals_v5b_2m_2color_spiral.json \
    --irregular-config configs/dr2_tf_irrs_v5b_2m_2color_irregular.json \
    > output/combine_v5b_2m.log 2>&1 || { echo "ERROR: combine failed - see output/combine_v5b_2m.log"; exit 1; }

# ---- Step 10: convert to the agreed convention (LOGDIST vs observed z + PV-corrected extras)
python3 finalize_v5b_2m.py > output/finalize_v5b_2m.log 2>&1 || { echo "ERROR: finalize failed - see output/finalize_v5b_2m.log"; exit 1; }
echo "=== [$(date)] v5b_2m complete -> output/DR2_TF_v5b_2m_2color_combined/DESI-DR2_TF_pv_cat_v5b_2m.fits (copied to catalogs/) ==="
