#!/bin/zsh
for i in {1..10}
do
        echo ./segev sample algorithm=hmc engine=nuts max_depth=15 adapt delta=0.95 num_warmup=1000 num_samples=1000 num_chains=4 data file=data/SGA_TFR_simtest_${(l:3::0:)i}.json output file=output/simtest_sym_${(l:3::0:)i}.csv
done
