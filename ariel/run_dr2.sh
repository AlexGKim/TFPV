#!/bin/bash
# run_dr2.sh — Automates DR2_2COLOR.md Steps 1-8 for one population.
#
# Usage:
#   ./run_dr2.sh --population {spiral,irregular} --fits data/<official_dr2>.fits [options]
#
# Steps 1-3b run synchronously in the foreground, including Step 3's
# interactive set_fiducial.py prompts (stdin is your real terminal). Once
# Step 3b (export_config.py) completes, this script re-execs itself detached
# (nohup) to run Steps 4-8 unattended in the background -- that phase takes
# hours (mainly Step 6's MCMC sampling).
#
# See DR2_2COLOR.md for the full narrative / rationale (in particular why
# Step 6 runs from the identity metric directly, with no separate metric
# warm-start step -- an empirical test found that ~2.7x slower overall than
# just letting Step 6 adapt from scratch).
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
#
# Resumable: each step is guarded by a sentinel file under output/$RUN/
# (.step{N}_done, matching slurm/check_status.sh's convention). Re-running
# the script skips completed steps. Step 6 tracks per-chain sentinels, so a
# partial failure only reruns the missing chains.

set -e

STEP_ORDER=(1 2 3 3b 4 5 5d 6 7 8)

usage() {
    echo "Usage: $0 --population {spiral,irregular} --fits PATH [--run-suffix NAME] [--from-step N] [--chains N]"
    exit 1
}

POPULATION=""
PARENT_FITS=""
RUN_SUFFIX=""
FROM_STEP=""
N_CHAINS=4
PHASE2=0

while [ $# -gt 0 ]; do
    case "$1" in
        --population) POPULATION="$2"; shift 2 ;;
        --fits) PARENT_FITS="$2"; shift 2 ;;
        --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
        --from-step) FROM_STEP="$2"; shift 2 ;;
        --chains) N_CHAINS="$2"; shift 2 ;;
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
        echo "=== Step 2: select_v2.py ==="
        python select_v2.py --run "$RUN" --fits_file "$FITS" --exe ./2color \
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
        mark_done 3b
        echo "Config written: $CONFIG (commit it to git)"
        echo "Optional: add \"train_fraction\": 0.4 / \"dust_pickle\": \"...\" to $CONFIG now if needed."
    fi

    echo ""
    echo "=== Steps 1-3b complete. Backgrounding Steps 4-8 (this takes hours). ==="
    nohup "$0" --population "$POPULATION" --fits "$PARENT_FITS" \
        ${RUN_SUFFIX:+--run-suffix "$RUN_SUFFIX"} --chains "$N_CHAINS" --phase2 \
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
    echo "=== Step 6: MCMC sampling ($N_CHAINS parallel chains, identity metric) ==="
    PIDS=()
    CHAIN_IDS=()
    for CHAIN_ID in $(seq 1 "$N_CHAINS"); do
        if [ -f "$RUN_DIR/.step6_chain${CHAIN_ID}_done" ]; then
            echo "  chain $CHAIN_ID: already done, skipping"
            continue
        fi
        ./2color sample num_warmup=250 num_samples=1000 \
            adapt save_metric=1 \
            algorithm=hmc engine=nuts max_depth=8 metric=dense_e \
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
    echo "=== Step 8: prediction ==="
    python color_predict.py --config "$CONFIG" --model 2color --xonly
    mark_done 8
fi

echo ""
echo "=== DONE: $RUN complete. ==="
