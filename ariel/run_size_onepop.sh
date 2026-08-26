#!/bin/bash
# run_size_onepop.sh — Automates DR2_SIZESPLIT.md Steps 1-8 for one population.
#
# Identical to run_dr2_onepop.sh except for the population split: the sample is
# partitioned by physical isophotal diameter (D26_kpc > 30 kpc vs <= 30 kpc)
# instead of by MORPHTYPE. Steps 1-8 are the same commands with the same
# arguments, so the two chains sample the same posterior with the same
# algorithm and their results are directly comparable.
#
# Usage:
#   ./run_size_onepop.sh --population {large,small} --fits data/DR2_TF_disks_v5.fits [options]
#
# Steps 1-3b run synchronously in the foreground, including Step 3's
# interactive set_fiducial.py prompts (stdin is your real terminal). Once
# Step 3b (export_config.py) completes, this script re-execs itself detached
# (nohup) to run Steps 4-8 unattended in the background -- that phase takes
# hours (mainly Step 6's MCMC sampling).
#
# See DR2_SIZESPLIT.md for the full narrative / rationale.
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
#   --population {large,small}        required
#   --fits PATH                       stacked parent FITS file (written by
#                                      make_size_subsets.py; its basename
#                                      drives RUN/CONFIG naming)
#   --run-suffix NAME                 override the filename-derived RUN name
#                                      (RUN becomes NAME_2color_<population>)
#   --size-input PATH                 catalog to stack into the parent, repeatable
#                                      (default: the two DR2_TF v5 files)
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
#   --dust-pickle PATH                internal-dust MCMC written into $CONFIG as
#                                     "dust_pickle" at Step 3b (default
#                                     data/loa_internalDust_nokcorr_mcmc.pickle,
#                                     giving the DR2 value d_err_r = 0.21734862).
#                                     "none" leaves it unset, for inputs whose
#                                     FITS header carries A_R_ERR/DSTCFF_R_ERR/
#                                     CFFR_ERR instead.
#
# Resumable: each step is guarded by a sentinel file under output/$RUN/
# (.step{N}_done, matching slurm/check_status.sh's convention). Re-running
# the script skips completed steps. Step 6 tracks per-chain sentinels, so a
# partial failure only reruns the missing chains.

set -e

STEP_ORDER=(1 2 3 3b 4 5 5d 6 7 8)

usage() {
    echo "Usage: $0 --population {large,small} --fits PATH [--run-suffix NAME] [--from-step N] [--chains N] [--warmup N] [--max-depth N] [--delta D] [--dust-pickle PATH|none]"
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
# DR2_TWOPOP.md validated against 1.9%-8.4% divergence rates at 0.8.
DELTA=0.9
PHASE2=0
# Internal-dust MCMC supplying d_err_r, the dust-slope uncertainty that scales
# the v_dust systematic in step 8's covariance. resolve_d_err_r() takes
# cfg["dust_pickle"] first, then the FITS header keywords, then a built-in iron
# default of 0.17680325 -- and the DR2 disk subsets carry NO dust keyword at all,
# so an unset dust_pickle lands silently on that default. For DR2 that is wrong
# by a factor of 0.8135 in d_err_r, i.e. 0.66x in dust variance. It emits one
# WARNING, which is exactly the line that scrolled past unnoticed in the mock
# batch. Set it here rather than leaving it to the reminder this script used to
# print: --dust-pickle none opts out for inputs that do carry a header keyword.
DUST_PICKLE="data/loa_internalDust_nokcorr_mcmc.pickle"
# Catalogs stacked into $PARENT_FITS by make_size_subsets.py. The DR2 disk
# sample ships as two morphology files; the size split is a repartition of
# exactly those galaxies, which is what makes it comparable to the
# spiral/irregular result.
SIZE_INPUTS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --population) POPULATION="$2"; shift 2 ;;
        --fits) PARENT_FITS="$2"; shift 2 ;;
        --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
        --size-input) SIZE_INPUTS+=("$2"); shift 2 ;;
        --from-step) FROM_STEP="$2"; shift 2 ;;
        --chains) N_CHAINS="$2"; shift 2 ;;
        --warmup) N_WARMUP="$2"; shift 2 ;;
        --max-depth) MAX_DEPTH="$2"; shift 2 ;;
        --delta) DELTA="$2"; shift 2 ;;
        --dust-pickle) DUST_PICKLE="$2"; shift 2 ;;
        --phase2) PHASE2=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[ -z "$POPULATION" ] && { echo "ERROR: --population is required"; usage; }
[ -z "$PARENT_FITS" ] && { echo "ERROR: --fits is required"; usage; }
case "$POPULATION" in
    large|small) ;;
    *) echo "ERROR: --population must be 'large' or 'small'"; exit 1 ;;
esac

# Fail now, not in step 8 hours later: a dust_pickle path that does not exist
# resolves to the iron default with only a WARNING, which is the silent-wrong
# outcome this setting exists to prevent.
if [ "$DUST_PICKLE" != "none" ] && [ ! -f "$DUST_PICKLE" ]; then
    echo "ERROR: --dust-pickle '$DUST_PICKLE' does not exist."
    echo "       Pass a valid path, or --dust-pickle none if this input carries"
    echo "       A_R_ERR / DSTCFF_R_ERR / CFFR_ERR in its FITS header instead."
    exit 1
fi

if [ ${#SIZE_INPUTS[@]} -eq 0 ]; then
    SIZE_INPUTS=(data/DR2_TF_spirals_v5.fits data/DR2_TF_irrs_v5.fits)
fi

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

    # Step 0: size split, skipped once this population's subset exists.
    # Unlike the morphology chain the parent is itself built here, by stacking
    # the spiral and irregular catalogs, so the inputs are the two v5 files
    # rather than $PARENT_FITS.
    if [ ! -f "$FITS" ]; then
        echo "=== Building $PARENT_FITS and splitting it by D26_kpc ==="
        # --force: the subset is missing, so the whole split is rebuilt from
        # the inputs; without it the run would abort on the parent file that a
        # previous invocation already wrote. The split is deterministic, so
        # rewriting it changes nothing.
        python make_size_subsets.py --input "${SIZE_INPUTS[@]}" --force \
            --parent-stem "$PARENT_STEM" --outdir "$(dirname "$PARENT_FITS")"
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
dust = '$DUST_PICKLE'
if dust != 'none':
    cfg['dust_pickle'] = dust
json.dump(cfg, open('$CONFIG', 'w'), indent=2)
print('fits_file set to', cfg['fits_file'])
print('dust_pickle set to', cfg.get('dust_pickle', '<unset: header keyword or iron default>'))
"
        mark_done 3b
        echo "Config written: $CONFIG (commit it to git)"
        echo "Optional: add \"train_fraction\": 0.4 to $CONFIG now if needed."
    fi

    echo ""
    echo "=== Steps 1-3b complete. Backgrounding Steps 4-8 (this takes hours). ==="
    SIZE_INPUT_ARGS=()
    for f in "${SIZE_INPUTS[@]}"; do SIZE_INPUT_ARGS+=(--size-input "$f"); done
    nohup "$0" --population "$POPULATION" --fits "$PARENT_FITS" \
        "${SIZE_INPUT_ARGS[@]}" \
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
    # explore_residuals.py runs in step 8, after color_predict.py — see below.
    mark_done 7
fi

if ! skip_if_done 8; then
    echo "=== Step 8: prediction (x-only, the default model; pass --full for the full model too) ==="
    python color_predict.py --config "$CONFIG" --model 2color

    # Order matters, and it is the same order slurm/step8_predict.sh uses:
    # color_predict.py writes the catalog and covariance (the science output),
    # explore_residuals.py only writes diagnostic plots. Running the plots first
    # -- as this script used to, from step 7 -- means a plotting failure under
    # `set -e` costs the run its science output.
    echo "=== Step 8b: residual diagnostics ==="
    python explore_residuals.py --config "$CONFIG" --kind 2color
    mark_done 8
fi

echo ""
echo "=== DONE: $RUN complete. ==="
