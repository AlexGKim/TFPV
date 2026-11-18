#!/bin/bash
# run_disks_single.sh — Automates DR2_DISKS_SINGLE.md Steps 1-8 for a single
# population: the DR2 disk catalogs stacked into one file and fit without any
# split. Same steps and same sampler arguments as run_dr2_onepop.sh — only the
# population machinery is gone, so this run is directly comparable to the
# morphology-split one.
#
# Usage:
#   ./run_disks_single.sh --fits data/DR2_TF_disks_v5_all.fits [options]
#
# Steps 1-3b run synchronously in the foreground, including Step 3's
# interactive set_fiducial.py prompts (stdin is your real terminal). Once
# Step 3b (export_config.py) completes, this script re-execs itself detached
# (nohup) to run Steps 4-8 unattended in the background -- that phase takes
# hours (mainly Step 6's MCMC sampling).
#
# See DR2_DISKS_SINGLE.md for the full narrative / rationale.
#
# Step 6 runs from the identity metric directly, with no separate metric
# warm-start step. This is a rank-1 consequence, not a general claim about
# preconditioning: the warmup funnel that made a fixed metric necessary came
# from the earlier rank-2 parameterization (a vanishing second scale plus a
# sphere-constrained unit_vector null direction, which no fixed metric could
# precondition). Under the current rank-1 S = w w^T the funnel is gone, and
# warmup adapts a dense metric from identity in ~4.4 h/chain without stalling
# at max treedepth -- measured on the abacus validation run, see BATCH_MOCKS.md.
# It also keeps this script algorithmically identical to the NERSC mock batch
# (slurm/step6_node.sh), which is what lets mock-derived uncertainties
# calibrate the real measurement.
#
# (Do not confuse this with the ~2.7x figure quoted in 2COLOR.md: that measured
# the *old short-MCMC* metric builder -- a covariance from ~100 post-warmup
# draws -- and says nothing about Pathfinder, which was a separate mechanism
# retired here for the rank-1 reason above.)
#
# Options:
#   --fits PATH                       stacked disk FITS file (written by
#                                      make_disk_union.py; its basename drives
#                                      RUN/CONFIG naming)
#   --run-suffix NAME                 override the filename-derived RUN name
#                                      (RUN becomes NAME_2color)
#   --from-step N                     clear sentinels for step N onward and
#                                      rerun from there (N in: 1 2 3 3b 4 5 5d 6 7 8)
#   --chains N                        number of parallel Step 6 chains (default 4)
#   --warmup N                        Step 6 num_warmup (default 1000; raised from
#                                      250 after an abacus-mock experiment showed a
#                                      non-trivial divergence/max-treedepth rate and
#                                      a systematic residual bias at 250)
#   --delta D                         adapt delta (default 0.9, matching the
#                                     NERSC batch; see slurm/step6_node.sh)
#   --model NAME                      Stan model (default 2color). 2color_v10
#                                      adds a redshift-dependent TF slope; its
#                                      CSVs and PNG are named after it, and
#                                      desi_data.py emits slope_std_0 /
#                                      slope_std_1 instead of slope_std.
#   --z_obs_min Z / --z_obs_max Z     training redshift window for Steps 1 and
#                                      2 (defaults 0.01 / 0.065). This is NOT
#                                      cosmetic: selection_ellipse.py fits the
#                                      GMM *under* the cut, and set_fiducial.py
#                                      derives slope_plane/intercept_plane/
#                                      intercept_plane2 from the resulting
#                                      selection_ellipse.json, so the z cut
#                                      propagates into the cut geometry. A
#                                      z>0.03 run must pass --z_obs_min 0.03
#                                      here, not only set it in the config, and
#                                      must take haty_min/haty_max from THIS
#                                      sample's own select_v2_pull.png.
#   --max-depth N                     Step 6 max_depth (default 10, Stan's own
#                                      default; lowering to 8 roughly halves the
#                                      worst-case per-iteration cost but was found
#                                      to cut off trajectories prematurely)
#
# Resumable: each step is guarded by a sentinel file under output/$RUN/
# (.step{N}_done, matching slurm/check_status.sh's convention). Re-running
# the script skips completed steps. Step 6 tracks per-chain sentinels, so a
# partial failure only reruns the missing chains.

set -e

STEP_ORDER=(1 2 3 3b 4 5 5d 6 7 8)

usage() {
    echo "Usage: $0 --fits PATH [--run-suffix NAME] [--from-step N] [--chains N] [--warmup N] [--max-depth N] [--delta D] [--model NAME] [--z_obs_min Z] [--z_obs_max Z]"
    exit 1
}

FITS=""
RUN_SUFFIX=""
FROM_STEP=""
N_CHAINS=4
N_WARMUP=1000
MAX_DEPTH=10
# adapt delta. MUST match slurm/step6_node.sh's non-debug default so the local
# real-data fit and the NERSC mock batch sample the same posterior with the same
# algorithm -- mock-derived uncertainties only calibrate the real measurement if
# the sampler is identical. 0.9 (up from Stan's default 0.8) is the value
# DR2_TWOPOP.md validated against 1.9%-8.4% divergence rates at 0.8.
DELTA=0.9
PHASE2=0
# Stan model. Empty so slurm/_model.sh resolves it from the config's "model"
# key (falling back to 2color), the same precedence the NERSC path uses.
MODEL=""
# Training redshift window for Steps 1-2. See the --z_obs_min note above.
Z_OBS_MIN=0.01
Z_OBS_MAX=0.065

while [ $# -gt 0 ]; do
    case "$1" in
        --fits) FITS="$2"; shift 2 ;;
        --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
        --from-step) FROM_STEP="$2"; shift 2 ;;
        --chains) N_CHAINS="$2"; shift 2 ;;
        --warmup) N_WARMUP="$2"; shift 2 ;;
        --max-depth) MAX_DEPTH="$2"; shift 2 ;;
        --delta) DELTA="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --z_obs_min) Z_OBS_MIN="$2"; shift 2 ;;
        --z_obs_max) Z_OBS_MAX="$2"; shift 2 ;;
        --phase2) PHASE2=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[ -z "$FITS" ] && { echo "ERROR: --fits is required"; usage; }

# --- Naming: derived from the FITS filename, not a fixed DR2_v0/v1 label ---
FITS_STEM=$(basename "$FITS" .fits)
if [ -n "$RUN_SUFFIX" ]; then
    RUN="${RUN_SUFFIX}_2color"
else
    RUN="${FITS_STEM}_2color"
fi
CONFIG="configs/$(echo "$RUN" | tr '[:upper:]' '[:lower:]').json"
RUN_DIR="output/$RUN"

mkdir -p "$RUN_DIR"

# MODEL: --model > the config's "model" key > 2color. The "_2color" in RUN
# above is a NAMING convention, deliberately not parameterized.
source slurm/_model.sh

echo "RUN=$RUN"
echo "CONFIG=$CONFIG"
echo "FITS=$FITS"
echo "MODEL=$MODEL"
echo "z window (Steps 1-2)=[$Z_OBS_MIN, $Z_OBS_MAX]"

# --- Sentinel helpers ---
sentinel() { echo "$RUN_DIR/.step${1}_done"; }

step_index() {
    local i=0 s
    for s in "${STEP_ORDER[@]}"; do
        i=$((i + 1))
        [ "$s" = "$1" ] && { echo "$i"; return; }
    done
    echo "ERROR: unknown step '$1' (expected one of: ${STEP_ORDER[*]})" >&2
    exit 1
}

skip_if_done() {
    local step="$1"
    if [ -f "$(sentinel "$step")" ]; then
        echo "Step $step: already done, skipping (rm $(sentinel "$step") to force)."
        return 0
    fi
    return 1
}

mark_done() { touch "$(sentinel "$1")"; }

# --from-step: clear sentinels for that step and everything after it
if [ -n "$FROM_STEP" ]; then
    FROM_IDX=$(step_index "$FROM_STEP")
    echo "Clearing sentinels from step $FROM_STEP onward..."
    idx=0
    for s in "${STEP_ORDER[@]}"; do
        idx=$((idx + 1))
        if [ "$idx" -ge "$FROM_IDX" ]; then
            rm -f "$(sentinel "$s")"
            [ "$s" = "6" ] && rm -f "$RUN_DIR"/.step6_chain*_done
        fi
    done
fi

# ============================================================
# Phase 1 (foreground, interactive): Steps 1-3b
# ============================================================
if [ "$PHASE2" -eq 0 ]; then

    # Step 0: build the stacked, VI-filtered catalog if it is missing. There
    # is no population split here — make_disk_union.py's only cut is "not
    # VI-rejected", which nothing downstream applies.
    if [ ! -f "$FITS" ]; then
        echo "=== Building $FITS from the DR2 disk catalogs ==="
        python make_disk_union.py --out "$FITS"
    fi

    if ! skip_if_done 1; then
        echo "=== Step 1: selection_ellipse.py ==="
        python selection_ellipse.py --file "$FITS" --run "$RUN" --source DESI \
            --z_obs_min "$Z_OBS_MIN" --z_obs_max "$Z_OBS_MAX" \
            --haty_min -23 --haty_max -18
        mark_done 1
        echo "Inspect: open $RUN_DIR/selection_ellipse.png"
    fi

    if ! skip_if_done 2; then
        echo "=== Step 2: select_v2.py (always uses tophat, not the final model) ==="
        python select_v2.py --run "$RUN" --fits_file "$FITS" --exe ./tophat \
            --z_obs_min "$Z_OBS_MIN" --z_obs_max "$Z_OBS_MAX"
        mark_done 2
        echo "Inspect: open $RUN_DIR/select_v2_pull.png"
    fi

    if ! skip_if_done 3; then
        echo "=== Step 3: set_fiducial.py — INTERACTIVE ==="
        echo "Look at $RUN_DIR/select_v2_pull.png before answering."
        python set_fiducial.py --run "$RUN"
        mark_done 3
        echo "Inspect: open $RUN_DIR/select_v2_fiducial_pull.png"
    fi

    if ! skip_if_done 3b; then
        echo "=== Step 3b: export_config.py — INTERACTIVE ==="
        echo "Suggested answers: exe=$MODEL, source=DESI, model=$MODEL, n_sigma=3.0"
        echo "Step 3 answers come from THIS sample's select_v2_pull.png; do not"
        echo "reuse another run's haty_min/haty_max."
        python export_config.py --run "$RUN" --out "$CONFIG"
        # export_config.py only picks up the correct fits_file from
        # output/$RUN/config.json, which desi_data.py (Step 4) hasn't
        # written yet at this point -- it silently falls back to a
        # hardcoded placeholder otherwise. Force-correct it here.
        python -c "
import json
cfg = json.load(open('$CONFIG'))
cfg['fits_file'] = '$FITS'
json.dump(cfg, open('$CONFIG', 'w'), indent=2)
print('fits_file set to', cfg['fits_file'])
"
        mark_done 3b
        echo "Config written: $CONFIG (commit it to git)"
        echo "Optional: add \"train_fraction\": 0.4 / \"dust_pickle\": \"...\" to $CONFIG now if needed."
    fi

    echo ""
    echo "=== Steps 1-3b complete. Backgrounding Steps 4-8 (this takes hours). ==="
    nohup "$0" --fits "$FITS" \
        ${RUN_SUFFIX:+--run-suffix "$RUN_SUFFIX"} --chains "$N_CHAINS" \
        --warmup "$N_WARMUP" --max-depth "$MAX_DEPTH" --delta "$DELTA" \
        ${MODEL:+--model "$MODEL"} \
        --z_obs_min "$Z_OBS_MIN" --z_obs_max "$Z_OBS_MAX" --phase2 \
        > "$RUN_DIR/run_dr2_phase2.log" 2>&1 &
    echo "Backgrounded as PID $!. Follow progress with:"
    echo "  tail -f $RUN_DIR/run_dr2_phase2.log"
    exit 0
fi

# ============================================================
# Phase 2 (backgrounded, unattended): Steps 4-8
# ============================================================

if ! skip_if_done 4; then
    echo "=== Step 4: desi_data.py ==="
    # --model is passed explicitly so that --model on THIS script's command
    # line wins over the config's own "model" key. It decides which init keys
    # desi_data.py writes: 2color_v10 declares slope_std_0 / slope_std_1 where
    # earlier models declare slope_std, and a mismatch is silent -- CmdStan
    # ignores unknown init keys and random-inits what is missing.
    python desi_data.py --config "$CONFIG" --model "$MODEL"
    mark_done 4
fi

if ! skip_if_done 5; then
    echo "=== Step 5: compile $MODEL (CPU) ==="
    # No STAN_OPENCL here: this is the CPU binary. The GPU build is the
    # separate ${MODEL}_g target -- see slurm/compile_gpu.sh.
    ( cd ../../cmdstan && make ../TFPV/ariel/${MODEL} )
    mark_done 5
fi

if ! skip_if_done 5d; then
    echo "=== Step 5d: MAP estimate ==="
    ./${MODEL} optimize \
        data file="$RUN_DIR/input.json" \
        init="$RUN_DIR/init.json" \
        output file="$RUN_DIR/optimize.csv"
    python3 make_map_init.py --run "$RUN"
    mark_done 5d
fi

if ! skip_if_done 6; then
    echo "=== Step 6: MCMC sampling ($N_CHAINS parallel chains, identity metric, delta=$DELTA) ==="
    PIDS=()
    CHAIN_IDS=()
    for CHAIN_ID in $(seq 1 "$N_CHAINS"); do
        if [ -f "$RUN_DIR/.step6_chain${CHAIN_ID}_done" ]; then
            echo "  chain $CHAIN_ID: already done, skipping"
            continue
        fi
        ./${MODEL} sample num_warmup="$N_WARMUP" num_samples=1000 \
            adapt delta="$DELTA" save_metric=1 \
            algorithm=hmc engine=nuts max_depth="$MAX_DEPTH" metric=dense_e \
            id=$CHAIN_ID \
            data file="$RUN_DIR/input.json" \
            init="$RUN_DIR/init_MAP.json" \
            output file="$RUN_DIR/${MODEL}_${CHAIN_ID}.csv" &
        PIDS+=($!)
        CHAIN_IDS+=("$CHAIN_ID")
    done

    FAIL=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            touch "$RUN_DIR/.step6_chain${CHAIN_IDS[$i]}_done"
        else
            echo "  ERROR: chain ${CHAIN_IDS[$i]} failed"
            FAIL=1
        fi
    done

    if [ "$FAIL" = "1" ]; then
        echo "Step 6: one or more chains failed. Rerun with --from-step 6 to retry only the missing chains."
        exit 1
    fi
    mark_done 6
    echo "Step 6: all $N_CHAINS chains done."
fi

if ! skip_if_done 7; then
    echo "=== Step 7: diagnostics ==="
    ../../cmdstan/bin/stansummary "$RUN_DIR"/${MODEL}_?.csv > "$RUN_DIR/stansummary.txt"
    ../../cmdstan/bin/diagnose    "$RUN_DIR"/${MODEL}_?.csv > "$RUN_DIR/diagnose.txt"
    # --model is passed explicitly: corner.py hard-defaults to "tophat".
    python corner.py --run "$RUN" --model "$MODEL"
    # explore_residuals.py runs in step 8, after color_predict.py — see below.
    mark_done 7
fi

if ! skip_if_done 8; then
    echo "=== Step 8: prediction (x-only, the default model; pass --full for the full model too) ==="
    # --model deliberately omitted: color_predict.py's precedence is explicit
    # flag > config "model" > "color", so the config stays authoritative.
    # Note --full is NOT available for 2color_v10: color_predict.py's
    # _reject_v10 blocks the full-colour predictors, and only x-only
    # supports v10.
    python color_predict.py --config "$CONFIG"

    # Order matters, and it is the same order slurm/step8_predict.sh uses:
    # color_predict.py writes the catalog and covariance (the science output),
    # explore_residuals.py only writes diagnostic plots. Running the plots first
    # -- as this script used to, from step 7 -- means a plotting failure under
    # `set -e` costs the run its science output.
    echo "=== Step 8b: residual diagnostics ==="
    python explore_residuals.py --config "$CONFIG" --kind "$MODEL" \
        || echo "WARNING: explore_residuals.py failed (non-fatal)"
    mark_done 8
fi

echo ""
echo "=== DONE: $RUN complete. ==="
