#!/bin/bash
# run_dr2_onepop.sh — Automates DR2_TWOPOP.md Steps 1-8 for one population.
#
# Usage:
#   ./run_dr2_onepop.sh --population {spiral,irregular} --fits data/<official_dr2>.fits [options]
#
# Steps 1-3b run synchronously in the foreground, including Step 3's
# interactive set_fiducial.py prompts (stdin is your real terminal). Once
# Step 3b (export_config.py) completes, this script re-execs itself detached
# (nohup) to run Steps 4-8 unattended in the background -- that phase takes
# hours (mainly Step 6's MCMC sampling).
#
# See DR2_TWOPOP.md for the full narrative / rationale.
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
#   --population {spiral,irregular}   required
#   --fits PATH                       parent/official DR2 FITS file (passed to
#                                      make_population_subsets.py --input; its
#                                      basename drives RUN/CONFIG naming)
#   --run-suffix NAME                 override the filename-derived RUN name
#                                      (RUN becomes NAME_2color_<population>)
#   --from-step N                     clear sentinels for step N onward and
#                                      rerun from there (N in: 1 2 3 3b 4 5 5d 6 7 8)
#   --chains N                        number of parallel Step 6 chains (default 4)
#   --warmup N                        Step 6 num_warmup (default 1000; raised from
#                                      250 after an abacus-mock experiment showed a
#                                      non-trivial divergence/max-treedepth rate and
#                                      a systematic residual bias at 250)
#   --delta D                         adapt delta (default 0.9, matching the
#                                     NERSC batch; see slurm/step6_node.sh)
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
    echo "Usage: $0 --population {spiral,irregular} --fits PATH [--run-suffix NAME] [--from-step N] [--chains N] [--warmup N] [--max-depth N] [--delta D]"
    exit 1
}

POPULATION=""
PARENT_FITS=""
RUN_SUFFIX=""
FROM_STEP=""
N_CHAINS=4
N_WARMUP=1000
MAX_DEPTH=10
# adapt delta. MUST match slurm/step6_node.sh's non-debug default so the local
# real-data fit and the NERSC mock batch sample the same posterior with the same
# algorithm -- mock-derived uncertainties only calibrate the real measurement if
# the sampler is identical. 0.9 (up from Stan's default 0.8) is the value
# DR2_TWOPOP.md validated against 1.9%-8.4% divergence rates at 0.8; this script
# previously passed no delta at all and so silently sampled at 0.8.
DELTA=0.9
PHASE2=0

while [ $# -gt 0 ]; do
    case "$1" in
        --population) POPULATION="$2"; shift 2 ;;
        --fits) PARENT_FITS="$2"; shift 2 ;;
        --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
        --from-step) FROM_STEP="$2"; shift 2 ;;
        --chains) N_CHAINS="$2"; shift 2 ;;
        --warmup) N_WARMUP="$2"; shift 2 ;;
        --max-depth) MAX_DEPTH="$2"; shift 2 ;;
        --delta) DELTA="$2"; shift 2 ;;
        --phase2) PHASE2=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[ -z "$POPULATION" ] && { echo "ERROR: --population is required"; usage; }
[ -z "$PARENT_FITS" ] && { echo "ERROR: --fits is required"; usage; }
case "$POPULATION" in
    spiral|irregular) ;;
    *) echo "ERROR: --population must be 'spiral' or 'irregular'"; exit 1 ;;
esac

# --- Naming: derived from the parent FITS filename, not a fixed DR2_v0/v1 label ---
PARENT_STEM=$(basename "$PARENT_FITS" .fits)
if [ -n "$RUN_SUFFIX" ]; then
    RUN="${RUN_SUFFIX}_2color_${POPULATION}"
else
    RUN="${PARENT_STEM}_2color_${POPULATION}"
fi
CONFIG="configs/$(echo "$RUN" | tr '[:upper:]' '[:lower:]').json"
FITS="${PARENT_FITS%.fits}_${POPULATION}.fits"
RUN_DIR="output/$RUN"

mkdir -p "$RUN_DIR"

echo "RUN=$RUN"
echo "CONFIG=$CONFIG"
echo "FITS=$FITS"

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

    # Step 0: population split (idempotent — make_population_subsets.py
    # skips existing files unless --force; safe to call every invocation)
    if [ ! -f "$FITS" ]; then
        echo "=== Splitting $PARENT_FITS into per-population subsets ==="
        python make_population_subsets.py --input "$PARENT_FITS"
    fi

    if ! skip_if_done 1; then
        echo "=== Step 1: selection_ellipse.py ==="
        python selection_ellipse.py --file "$FITS" --run "$RUN" --source DESI \
            --z_obs_min 0.01 --z_obs_max 0.065 --haty_min -23 --haty_max -18
        mark_done 1
        echo "Inspect: open $RUN_DIR/selection_ellipse.png"
    fi

    if ! skip_if_done 2; then
        echo "=== Step 2: select_v2.py (always uses tophat, not the final model) ==="
        python select_v2.py --run "$RUN" --fits_file "$FITS" --exe ./tophat \
            --z_obs_min 0.01 --z_obs_max 0.065
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
        echo "Suggested answers: exe=2color, source=DESI, model=2color, n_sigma=3.0"
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
    nohup "$0" --population "$POPULATION" --fits "$PARENT_FITS" \
        ${RUN_SUFFIX:+--run-suffix "$RUN_SUFFIX"} --chains "$N_CHAINS" \
        --warmup "$N_WARMUP" --max-depth "$MAX_DEPTH" --delta "$DELTA" --phase2 \
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
    python desi_data.py --config "$CONFIG"
    mark_done 4
fi

if ! skip_if_done 5; then
    echo "=== Step 5: compile 2color ==="
    ( cd ../../cmdstan && make ../TFPV/ariel/2color )
    mark_done 5
fi

if ! skip_if_done 5d; then
    echo "=== Step 5d: MAP estimate ==="
    ./2color optimize \
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
        ./2color sample num_warmup="$N_WARMUP" num_samples=1000 \
            adapt delta="$DELTA" save_metric=1 \
            algorithm=hmc engine=nuts max_depth="$MAX_DEPTH" metric=dense_e \
            id=$CHAIN_ID \
            data file="$RUN_DIR/input.json" \
            init="$RUN_DIR/init_MAP.json" \
            output file="$RUN_DIR/2color_${CHAIN_ID}.csv" &
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
    ../../cmdstan/bin/stansummary "$RUN_DIR"/2color_?.csv > "$RUN_DIR/stansummary.txt"
    ../../cmdstan/bin/diagnose    "$RUN_DIR"/2color_?.csv > "$RUN_DIR/diagnose.txt"
    python corner.py --run "$RUN" --model 2color
    python explore_residuals.py --config "$CONFIG" --kind 2color
    mark_done 7
fi

if ! skip_if_done 8; then
    echo "=== Step 8: prediction (x-only, the default model; pass --full for the full model too) ==="
    python color_predict.py --config "$CONFIG" --model 2color
    mark_done 8
fi

echo ""
echo "=== DONE: $RUN complete. ==="
