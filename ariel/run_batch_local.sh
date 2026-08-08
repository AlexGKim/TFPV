#!/bin/bash
# run_batch_local.sh — Run the mock pipeline LOCALLY, with no scheduler.
#
# This is the no-SLURM counterpart to slurm/batch_submit.sh and runs the SAME
# steps, in the same order, with the same sampler arguments:
#
#     step4 → step6 (N chains in parallel) → step7 → step8
#
# The point of the parity is scientific, not cosmetic: mock-derived
# uncertainties only calibrate the real DR2 measurement if both are fit by the
# same algorithm. Every cmdstan/python command line below is copied from its
# slurm/step*.sh counterpart, with exactly two changes -- the CPU binary
# ./2color instead of the OpenCL build ./2color_g, and no CUDA_VISIBLE_DEVICES
# (chains go one per CPU core rather than one per GPU). If you change a sampler
# argument here, change slurm/step6_node.sh to match, or the two stop being
# comparable.
#
# Steps deliberately NOT run, matching the batch (see BATCH_MOCKS.md):
#   1/2/3/3b  selection ellipse, MLE, fiducial, export_config -- the mock
#             trapezoid cuts are FROZEN in the base config, not re-derived
#             per file. That is what makes 125 mocks comparable to each other.
#   5d        MAP optimize -- configs set "fixed_init", so step4 writes
#             init_MAP.json itself. Run as a fallback only if it is missing.
#   5e        metric seeding -- chains start from the identity metric.
#
# Usage:
#   bash run_batch_local.sh <config_dir | config.json> [options]
#   bash run_batch_local.sh --fits-dir <dir_of_mock_fits> [options]
#
# Options:
#   --fits-dir DIR    Generate one config per mock FITS in DIR first, via
#                     make_batch_configs.py, then run them. Its pre-flight
#                     header validation (dust keyword, PHOTSYS_ERR) therefore
#                     protects local runs too.
#   --base CONFIG     Base config to clone for --fits-dir
#                     (default: configs/abacus_2color.json)
#   --outdir DIR      Where --fits-dir writes configs
#                     (default: configs/batch_local)
#   --chains N        Chains per run, launched in parallel (default 4)
#   --jobs N          Runs to process concurrently (default 1). Each run uses
#                     --chains cores, so --jobs 2 --chains 4 needs 8 cores; and
#                     each concurrent step8 holds a ~1.19 GB covariance plus
#                     ~0.55 GB of temporaries.
#   --from-step S     Clear sentinels from step S onward for every run, then
#                     re-run from there. S in: 4 5d 6 7 8
#   --warmup N        num_warmup      (default 1000, matching the batch)
#   --samples N       num_samples     (default 1000)
#   --max-depth N     max_depth       (default 10)
#   --delta D         adapt delta     (default 0.9)
#   --debug           Fast plumbing test: no adaptation, 15 samples,
#                     max_depth=1, fixed stepsize, and step8 skips the
#                     covariance. Mirrors step6_node.sh's DEBUG=1 branch.
#                     Results are NOT science-grade.
#   --no-cov          Skip the O(G^2) covariance in step8 (implied by --debug)
#   --force           Re-run even where .step8_done already exists
#   -h | --help
#
# Resumable: every step is guarded by a sentinel under output/<run>/, using the
# SAME names as slurm/check_status.sh, so both status scripts work on local runs
# with no changes:
#   bash slurm/check_status.sh configs/abacus_2color.json
#   bash slurm/batch_status.sh configs/batch_local
# Step 6 tracks per-chain sentinels, so a partial failure re-runs only the
# chains that are actually missing.
#
# Cost: at target_main_count=17234 a mock is roughly 6.4 h/chain on CPU
# (~4.4 h of that is warmup), 4 chains in parallel, so ~6-7 h wall-clock per
# mock plus ~5-10 min for step8. Use --debug to exercise the plumbing first.

# Deliberately NOT `set -e`: one failing run must not abort the whole batch --
# each run's status is checked explicitly and recorded. `pipefail` is required
# because every step is piped through `tee` for logging, and without it the
# pipeline would report tee's exit status and every real failure would pass
# silently.
set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TARGET=""
FITS_DIR=""
BASE_CONFIG="configs/abacus_2color.json"
OUTDIR="configs/batch_local"
N_CHAINS=4
N_JOBS=1
FROM_STEP=""
NUM_WARMUP=1000
NUM_SAMPLES=1000
MAX_DEPTH=10
DELTA=0.9
DEBUG=0
NO_COV=0
FORCE=0

STEP_ORDER=(4 5d 6 7 8)
TRACKER="batch/local_tracker.csv"

usage() {
    sed -n '2,70p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-1}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --fits-dir)   FITS_DIR="$2"; shift 2 ;;
        --base)       BASE_CONFIG="$2"; shift 2 ;;
        --outdir)     OUTDIR="$2"; shift 2 ;;
        --chains)     N_CHAINS="$2"; shift 2 ;;
        --jobs)       N_JOBS="$2"; shift 2 ;;
        --from-step)  FROM_STEP="$2"; shift 2 ;;
        --warmup)     NUM_WARMUP="$2"; shift 2 ;;
        --samples)    NUM_SAMPLES="$2"; shift 2 ;;
        --max-depth)  MAX_DEPTH="$2"; shift 2 ;;
        --delta)      DELTA="$2"; shift 2 ;;
        --debug)      DEBUG=1; shift ;;
        --no-cov)     NO_COV=1; shift ;;
        --force)      FORCE=1; shift ;;
        -h|--help)    usage 0 ;;
        --*)          echo "ERROR: unknown option: $1" >&2; usage ;;
        *)
            if [ -z "$TARGET" ]; then TARGET="$1"; shift
            else echo "ERROR: unexpected extra argument: $1" >&2; usage; fi
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve the config list
# ---------------------------------------------------------------------------
if [ -n "$FITS_DIR" ]; then
    if [ -n "$TARGET" ]; then
        echo "ERROR: pass either a config dir/file or --fits-dir, not both." >&2
        exit 1
    fi
    [ -d "$FITS_DIR" ] || { echo "ERROR: --fits-dir not found: $FITS_DIR" >&2; exit 1; }
    [ -f "$BASE_CONFIG" ] || { echo "ERROR: --base not found: $BASE_CONFIG" >&2; exit 1; }
    echo "=== Generating configs from $FITS_DIR ==="
    # make_batch_configs.py validates every file's dust keyword and PHOTSYS_ERR
    # column and writes nothing if any file fails, so a bad input costs seconds
    # here rather than a 6-hour run followed by a broken step 8.
    if ! python3 make_batch_configs.py --dir "$FITS_DIR" --base "$BASE_CONFIG" \
            --outdir "$OUTDIR" --overwrite; then
        echo "ERROR: config generation failed — nothing was run." >&2
        exit 1
    fi
    TARGET="$OUTDIR"
    echo ""
fi

[ -n "$TARGET" ] || { echo "ERROR: pass a config directory, a config file, or --fits-dir." >&2; usage; }

CONFIGS=()
if [ -d "$TARGET" ]; then
    for CFG in "$TARGET"/*.json; do
        [ -e "$CFG" ] || { echo "ERROR: no *.json configs in $TARGET" >&2; exit 1; }
        CONFIGS+=("$CFG")
    done
elif [ -f "$TARGET" ]; then
    CONFIGS+=("$TARGET")
else
    echo "ERROR: not a directory or file: $TARGET" >&2
    exit 1
fi

case "$N_CHAINS" in ''|*[!0-9]*) echo "ERROR: --chains must be a positive integer" >&2; exit 1 ;; esac
case "$N_JOBS"   in ''|*[!0-9]*) echo "ERROR: --jobs must be a positive integer" >&2; exit 1 ;; esac
[ "$N_CHAINS" -ge 1 ] || { echo "ERROR: --chains must be >= 1" >&2; exit 1; }
[ "$N_JOBS"   -ge 1 ] || { echo "ERROR: --jobs must be >= 1" >&2; exit 1; }

if [ -n "$FROM_STEP" ]; then
    _ok=0
    for s in "${STEP_ORDER[@]}"; do [ "$s" = "$FROM_STEP" ] && _ok=1; done
    [ "$_ok" = "1" ] || { echo "ERROR: --from-step must be one of: ${STEP_ORDER[*]}" >&2; exit 1; }
fi

# ---------------------------------------------------------------------------
# Sampler arguments — copied from slurm/step6_node.sh's two branches so the
# local and batch fits are argument-for-argument identical.
# ---------------------------------------------------------------------------
if [ "$DEBUG" = "1" ]; then
    NUM_WARMUP=0
    NUM_SAMPLES=15
    ADAPT_ARGS="adapt engaged=0"
    ENGINE_ARGS="engine=nuts max_depth=1"
    STEPSIZE_ARG="stepsize=${STEPSIZE:-0.08}"
    NO_COV=1
    echo "*** DEBUG mode: no adaptation, max_depth=1, $NUM_SAMPLES samples, no covariance."
    echo "*** Results are NOT science-grade — this only exercises the plumbing."
else
    ADAPT_ARGS="adapt delta=$DELTA save_metric=1"
    ENGINE_ARGS="engine=nuts max_depth=$MAX_DEPTH"
    STEPSIZE_ARG=""
fi

# ---------------------------------------------------------------------------
# Prerequisites. The batch gets these for free (a one-time compile job, and a
# conda env baked into every step script); locally they are the two things most
# likely to be silently wrong.
# ---------------------------------------------------------------------------
check_prereqs() {
    local missing=0

    # Rebuild ./2color when 2color.stan is newer, or the binary is absent.
    # Deliberately mtime-based rather than sentinel-based: run_dr2_onepop.sh
    # guards its compile with .step5_done, which means an edited .stan is never
    # picked up and you silently fit the previous model.
    if [ ! -x ./2color ] || [ 2color.stan -nt ./2color ]; then
        if [ ! -x ./2color ]; then
            echo "=== ./2color missing — compiling ==="
        else
            echo "=== 2color.stan is newer than ./2color — recompiling ==="
        fi
        if ! ( cd ../../cmdstan && make ../TFPV/ariel/2color ); then
            echo "ERROR: failed to compile ./2color." >&2
            missing=1
        fi
    else
        echo "./2color is up to date with 2color.stan"
    fi

    # step7 needs these; without the check they fail mid-run, hours in.
    for tool in ../../cmdstan/bin/stansummary ../../cmdstan/bin/diagnose; do
        if [ ! -x "$tool" ]; then
            echo "ERROR: $tool not found or not executable." >&2
            echo "       Build it with: ( cd ../../cmdstan && make build )" >&2
            missing=1
        fi
    done

    [ "$missing" = "0" ] || exit 1
}

# ---------------------------------------------------------------------------
# Per-run driver. Runs in a subshell (so --jobs can background it) and returns
# non-zero on failure without taking the batch down.
# ---------------------------------------------------------------------------
run_one() {
    local CFG="$1"
    local RUN RUN_DIR

    RUN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['run'])" "$CFG" 2>/dev/null)
    if [ -z "$RUN" ]; then
        echo "[$CFG] ERROR: could not read \"run\" from config — skipping" >&2
        return 1
    fi
    RUN_DIR="output/$RUN"
    mkdir -p "$RUN_DIR"

    # These read $RUN_DIR from run_one's local scope, so they are only valid
    # while run_one is on the stack.
    sentinel_of() { echo "$RUN_DIR/.step${1}_done"; }
    skip_if_done() { [ -f "$(sentinel_of "$1")" ]; }
    mark_done() { touch "$(sentinel_of "$1")"; }

    # --from-step: clear this step's sentinel and every later one, matching
    # run_dr2_onepop.sh's clearing loop including the per-chain special case.
    if [ -n "$FROM_STEP" ]; then
        local from_idx=0 idx=0 s
        for s in "${STEP_ORDER[@]}"; do
            idx=$((idx + 1))
            [ "$s" = "$FROM_STEP" ] && from_idx=$idx
        done
        idx=0
        for s in "${STEP_ORDER[@]}"; do
            idx=$((idx + 1))
            if [ "$idx" -ge "$from_idx" ]; then
                rm -f "$(sentinel_of "$s")"
                [ "$s" = "6" ] && rm -f "$RUN_DIR"/.step6_chain*_done
            fi
        done
    fi

    # Same skip rule as slurm/batch_submit.sh: .step8_done means done.
    if [ "$FORCE" = "0" ] && [ -f "$RUN_DIR/.step8_done" ]; then
        echo "[$RUN] skip (done) — pass --force or --from-step to re-run"
        return 0
    fi

    local started
    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "=========================================================="
    echo "[$RUN] start $started   config=$CFG"
    echo "=========================================================="

    # --- Step 4: prepare data (slurm/step4_data.sh) ---
    if skip_if_done 4; then
        echo "[$RUN] step4: already done, skipping"
    else
        echo "[$RUN] step4: desi_data.py"
        if ! python3 desi_data.py --config "$CFG" 2>&1 | tee "$RUN_DIR/local_step4.log"; then
            echo "[$RUN] ERROR: step4 failed — see $RUN_DIR/local_step4.log" >&2
            _track "$RUN" "$CFG" "$started" "FAILED_step4"
            return 1
        fi
        mark_done 4
    fi

    # --- Step 5d fallback: only when step4 did not write init_MAP.json.
    # Mock configs set "fixed_init" so this never fires; it keeps the driver
    # usable on a config that does not. Mirrors run_subsets.sh's guard.
    if [ ! -f "$RUN_DIR/init_MAP.json" ]; then
        echo "[$RUN] step5d: no init_MAP.json (config sets no fixed_init) — running MAP optimize"
        {
            ./2color optimize \
                data file="$RUN_DIR/input.json" \
                init="$RUN_DIR/init.json" \
                output file="$RUN_DIR/optimize.csv" \
                || echo "[$RUN] step5d: optimizer exited nonzero (using best iterate)"
            python3 make_map_init.py --run "$RUN" || true
        } 2>&1 | tee "$RUN_DIR/local_step5d.log"
        if [ ! -f "$RUN_DIR/init_MAP.json" ]; then
            echo "[$RUN] ERROR: step5d produced no init_MAP.json — cannot sample" >&2
            _track "$RUN" "$CFG" "$started" "FAILED_step5d"
            return 1
        fi
        mark_done 5d
    fi

    # --- Step 6: sampling (slurm/step6_node.sh, one chain per core) ---
    local PIDS=() CHAIN_IDS=() CHAIN_ID i FAIL=0
    for CHAIN_ID in $(seq 1 "$N_CHAINS"); do
        if [ -f "$RUN_DIR/.step6_chain${CHAIN_ID}_done" ]; then
            echo "[$RUN] step6 chain $CHAIN_ID: already done, skipping"
            continue
        fi
        (
            ./2color sample num_warmup=$NUM_WARMUP num_samples=$NUM_SAMPLES \
                $ADAPT_ARGS \
                algorithm=hmc $ENGINE_ARGS metric=dense_e $STEPSIZE_ARG \
                id=$CHAIN_ID \
                data file="$RUN_DIR/input.json" \
                init="$RUN_DIR/init_MAP.json" \
                output file="$RUN_DIR/2color_${CHAIN_ID}.csv"
        ) > "$RUN_DIR/local_step6_chain${CHAIN_ID}.log" 2>&1 &
        PIDS+=($!)
        CHAIN_IDS+=("$CHAIN_ID")
    done
    if [ "${#PIDS[@]}" -gt 0 ]; then
        echo "[$RUN] step6: ${#PIDS[@]} chain(s) running in parallel (chains ${CHAIN_IDS[*]})"
        # Touch each chain's sentinel only if that chain actually succeeded, so
        # --from-step 6 re-runs exactly the missing ones. run_subsets.sh's bare
        # `wait` could not tell which chain failed.
        for i in "${!PIDS[@]}"; do
            if wait "${PIDS[$i]}"; then
                touch "$RUN_DIR/.step6_chain${CHAIN_IDS[$i]}_done"
                echo "[$RUN] step6 chain ${CHAIN_IDS[$i]}: done"
            else
                echo "[$RUN] ERROR: step6 chain ${CHAIN_IDS[$i]} failed — see $RUN_DIR/local_step6_chain${CHAIN_IDS[$i]}.log" >&2
                FAIL=1
            fi
        done
    fi
    if [ "$FAIL" = "1" ]; then
        echo "[$RUN] ERROR: step6 incomplete. Re-run with --from-step 6 to retry only the missing chains." >&2
        _track "$RUN" "$CFG" "$started" "FAILED_step6"
        return 1
    fi

    # --- Step 7: diagnostics (slurm/step7_diagnose.sh) ---
    if skip_if_done 7; then
        echo "[$RUN] step7: already done, skipping"
    else
        echo "[$RUN] step7: stansummary, diagnose, corner"
        # In DEBUG mode corner.py is allowed to fail: 15 unadapted draws give a
        # degenerate posterior that chainconsumer cannot summarize (IndexError
        # inside get_parameter_summary_max). stansummary/diagnose still work and
        # are what a plumbing test actually needs to see. Production keeps it
        # fatal. slurm/step7_diagnose.sh has the matching branch.
        local CORNER_RC=0
        if ! {
            ../../cmdstan/bin/stansummary "$RUN_DIR"/2color_?.csv > "$RUN_DIR/stansummary.txt" &&
            ../../cmdstan/bin/diagnose    "$RUN_DIR"/2color_?.csv > "$RUN_DIR/diagnose.txt"
        } 2>&1 | tee "$RUN_DIR/local_step7.log"; then
            echo "[$RUN] ERROR: step7 (stansummary/diagnose) failed — see $RUN_DIR/local_step7.log" >&2
            _track "$RUN" "$CFG" "$started" "FAILED_step7"
            return 1
        fi
        python3 corner.py --run "$RUN" --model 2color 2>&1 | tee -a "$RUN_DIR/local_step7.log" || CORNER_RC=1
        if [ "$CORNER_RC" != "0" ]; then
            if [ "$DEBUG" = "1" ]; then
                echo "[$RUN] WARNING: corner.py failed, tolerated in --debug ($NUM_SAMPLES draws is too few to plot)"
            else
                echo "[$RUN] ERROR: step7 (corner.py) failed — see $RUN_DIR/local_step7.log" >&2
                _track "$RUN" "$CFG" "$started" "FAILED_step7"
                return 1
            fi
        fi
        mark_done 7
    fi

    # --- Step 8: prediction, then residual plots (slurm/step8_predict.sh) ---
    # Order matters and is not negotiable: color_predict.py writes the catalog
    # and covariance (the science output); explore_residuals.py only writes
    # plots. Plots first would let a plotting failure cost the run its output.
    if skip_if_done 8; then
        echo "[$RUN] step8: already done, skipping"
    else
        local COV_ARG=""
        [ "$NO_COV" = "1" ] && COV_ARG="--no-cov"
        echo "[$RUN] step8: color_predict.py ${COV_ARG}"
        if ! python3 color_predict.py --config "$CFG" --model 2color $COV_ARG \
                2>&1 | tee "$RUN_DIR/local_step8.log"; then
            echo "[$RUN] ERROR: step8 (color_predict.py) failed — see $RUN_DIR/local_step8.log" >&2
            _track "$RUN" "$CFG" "$started" "FAILED_step8"
            return 1
        fi
        echo "[$RUN] step8b: explore_residuals.py"
        if ! python3 explore_residuals.py --config "$CFG" --kind 2color \
                2>&1 | tee "$RUN_DIR/local_step8b.log"; then
            echo "[$RUN] ERROR: step8b (explore_residuals.py) failed — see $RUN_DIR/local_step8b.log" >&2
            echo "[$RUN] NOTE: the catalog and covariance were already written." >&2
            _track "$RUN" "$CFG" "$started" "FAILED_step8b"
            return 1
        fi
        mark_done 8
    fi

    echo "[$RUN] COMPLETE"
    _track "$RUN" "$CFG" "$started" "COMPLETE"
    return 0
}

# batch/local_tracker.csv is the analogue of the batch's batch/job_tracker.csv:
# what makes an overnight multi-mock run auditable after the fact.
_track() {
    printf '%s,%s,%s,%s,%s\n' "$1" "$2" "$3" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$4" >> "$TRACKER"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p batch
[ -f "$TRACKER" ] || echo "run,config,started,finished,status" > "$TRACKER"

check_prereqs

echo ""
echo "Configs      : ${#CONFIGS[@]}"
echo "Chains/run   : $N_CHAINS"
echo "Concurrent   : $N_JOBS run(s)  (~$((N_JOBS * N_CHAINS)) cores)"
echo "Sampler      : num_warmup=$NUM_WARMUP num_samples=$NUM_SAMPLES $ADAPT_ARGS $ENGINE_ARGS $STEPSIZE_ARG"
echo "Tracker      : $TRACKER"
[ -n "$FROM_STEP" ] && echo "From step    : $FROM_STEP (later sentinels cleared per run)"
echo ""

N_OK=0
N_BAD=0
if [ "$N_JOBS" = "1" ]; then
    for CFG in "${CONFIGS[@]}"; do
        if run_one "$CFG"; then N_OK=$((N_OK + 1)); else N_BAD=$((N_BAD + 1)); fi
    done
else
    # Concurrent runs: background each whole chain and throttle to --jobs.
    RUN_PIDS=()
    for CFG in "${CONFIGS[@]}"; do
        while [ "$(jobs -rp | wc -l)" -ge "$N_JOBS" ]; do sleep 5; done
        run_one "$CFG" &
        RUN_PIDS+=($!)
    done
    for PID in "${RUN_PIDS[@]}"; do
        if wait "$PID"; then N_OK=$((N_OK + 1)); else N_BAD=$((N_BAD + 1)); fi
    done
fi

echo ""
echo "=========================================================="
echo "Runs complete/skipped : $N_OK"
echo "Runs failed           : $N_BAD"
echo "Tracker               : $TRACKER"
if [ -d "$TARGET" ]; then
    echo "Status                : bash slurm/batch_status.sh $TARGET"
else
    echo "Status                : bash slurm/check_status.sh $TARGET"
fi
echo "=========================================================="

[ "$N_BAD" = "0" ] || exit 1
