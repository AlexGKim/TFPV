#!/bin/zsh
# Run the 2color pipeline for subsets 01-04.
#
# s00 is intentionally NOT included here: it was already run standalone
# end-to-end (step4 -> step5d -> step6 -> step8) and its output/abacus_2color_s00/
# is current. Re-run s00 through this script too if its config or the mock FITS
# file ever changes.
#
# Usage: zsh run_subsets.sh

set -e

# Step 4: data prep. Always re-run so output/abacus_2color_sNN/input.json
# matches the *current* configs/abacus_subsets/abacus_2color_sNN.json (n_subsets,
# subset_index, n_objects). Skipping this after editing a config silently reuses
# a stale partition — desi_data.py now warns if it detects that mismatch, but
# re-running step4 unconditionally here avoids it entirely.
for i in 01 02 03 04; do
    echo "Preparing data for subset $i..."
    python desi_data.py --config "configs/abacus_subsets/abacus_2color_s${i}.json"
done

# Step 5d: MAP optimization -> init_MAP.json (must be redone whenever step4
# regenerates input.json, since the MAP depends on the training rows).
for i in 01 02 03 04; do
    echo "MAP optimization for subset $i..."
    ./2color optimize \
        data file="output/abacus_2color_s${i}/input.json" \
        init="output/abacus_2color_s${i}/init.json" \
        output file="output/abacus_2color_s${i}/optimize.csv"
    python - "output/abacus_2color_s${i}" <<'EOF'
import pandas as pd, json, sys
run_dir = sys.argv[1]
df = pd.read_csv(f'{run_dir}/optimize.csv', comment='#')
row = df.iloc[0]
old = json.load(open(f'{run_dir}/init.json'))
new = {}
for k in old.keys():
    if k == 'intercept_std':
        cols = sorted([c for c in df.columns if c.startswith('intercept_std.')],
                      key=lambda s: int(s.split('.')[1]))
        new[k] = [float(row[c]) for c in cols]
    elif k in df.columns:
        new[k] = float(row[k])
    else:
        new[k] = old[k]
with open(f'{run_dir}/init_MAP.json', 'w') as f:
    json.dump(new, f, indent=2)
print(f'MAP init written to {run_dir}/init_MAP.json')
EOF
done

# Copy metric to all subset directories
for i in 01 02 03 04; do
    cp -f output/abacus_2color/metric.json "output/abacus_2color_s${i}/metric.json"
done
echo "Metric copied to all subsets."

# Step 6: MCMC sampling. num_warmup=250/num_samples=1000 matches 2COLOR.md Step 6
# and what output/abacus_2color_s00 was actually fit with — use the same settings
# here so s01-s04 are science-grade and comparable to s00, not a quick-look debug
# run at reduced samples.
for i in 01 02 03 04; do
    echo "Sampling subset $i..."
    for CHAIN in 1 2 3 4; do
        ./2color sample num_warmup=250 num_samples=1000 \
            adapt save_metric=1 \
            algorithm=hmc metric=dense_e \
            metric_file="output/abacus_2color_s${i}/metric.json" \
            id=$CHAIN \
            data file="output/abacus_2color_s${i}/input.json" \
            init="output/abacus_2color_s${i}/init_MAP.json" \
            output file="output/abacus_2color_s${i}/2color_${CHAIN}.csv" &
    done
    wait
    echo "Subset $i sampling done."
done

# Step 7+8: Diagnostics and predictions
for i in 01 02 03 04; do
    echo "Processing subset $i diagnostics and predictions..."
    python corner.py --run "abacus_2color_s${i}" --model 2color
    python color_predict.py --run-dir "output/abacus_2color_s${i}" --model 2color --no-cov --no-catalog
done

echo "All subsets complete."
