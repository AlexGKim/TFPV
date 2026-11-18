#!/bin/bash
# Steps 7-10 of run_v5b_2m.sh, for use after the chains were (re)launched by hand.
# Waits for all '2color sample' processes to exit, then: diagnostics, prediction, combine, finalize.
set -u
cd "$(dirname "$0")"
CMDSTAN=../../cmdstan
while pgrep -f '2color sample' > /dev/null; do sleep 60; done
echo "=== [$(date)] all chains finished ==="
FAIL=0
for RUN in DR2_TF_spirals_v5b_2m_2color_spiral DR2_TF_irrs_v5b_2m_2color_irregular; do
  CONFIG=configs/$(echo $RUN | tr '[:upper:]' '[:lower:]').json
  for c in 1 2 3 4; do
    grep -q "Elapsed Time" output/$RUN/chain$c.log || { echo "ERROR: $RUN chain $c did not finish"; FAIL=1; }
  done
  [ "$FAIL" = 1 ] && continue
  $CMDSTAN/bin/stansummary output/$RUN/2color_?.csv > output/$RUN/stansummary.txt
  $CMDSTAN/bin/diagnose    output/$RUN/2color_?.csv > output/$RUN/diagnose.txt
  python3 corner.py --run $RUN --model 2color       > output/$RUN/step7_corner.log 2>&1 || echo "  (corner.py failed - non-fatal)"
  echo "[$(date)] $RUN step7 done"
  python3 color_predict.py --config $CONFIG --model 2color > output/$RUN/step8.log 2>&1 || { echo "ERROR step8 $RUN"; FAIL=1; continue; }
  grep "Loaded d_err_r" output/$RUN/step8.log || echo "  WARNING: d_err_r did not come from the dust pickle - check step8.log"
  echo "[$(date)] $RUN step8 done"
done
[ "$FAIL" = 1 ] && { echo "ERROR: a population failed; not combining"; exit 1; }
python3 combine_color_xonly.py \
    --spiral-run    DR2_TF_spirals_v5b_2m_2color_spiral \
    --irregular-run DR2_TF_irrs_v5b_2m_2color_irregular \
    --out-run       DR2_TF_v5b_2m_2color_combined \
    --catalog-name  DESI-DR2_TF_pv_cat_v5b_2m_raw.fits \
    --spiral-config    configs/dr2_tf_spirals_v5b_2m_2color_spiral.json \
    --irregular-config configs/dr2_tf_irrs_v5b_2m_2color_irregular.json \
    > output/combine_v5b_2m.log 2>&1 || { echo "ERROR: combine failed - see output/combine_v5b_2m.log"; exit 1; }
echo "[$(date)] combine done"
python3 finalize_v5b_2m.py --compare > output/finalize_v5b_2m.log 2>&1 || { echo "ERROR: finalize failed - see output/finalize_v5b_2m.log"; exit 1; }
echo "=== [$(date)] v5b_2m complete -> output/DR2_TF_v5b_2m_2color_combined/DESI-DR2_TF_pv_cat_v5b_2m.fits (copied to catalogs/) ==="
