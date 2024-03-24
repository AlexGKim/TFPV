#!/bin/zsh
for i in {1..10}
do
        ./segev sample num_warmup=2000 num_chains=4 data file=data/SGA_TFR_simtest_${(l:3::0:)i}.json output file=output_${(l:3::0:)i}.csv
done
