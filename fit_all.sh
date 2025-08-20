#!/bin/zsh

for i in {95..8309}
do
  ./fit sample algorithm=hmc engine=nuts num_warmup=1000 num_samples=1000 num_chains=4 init="data_fit/$RELEASE_DIR/fit_init_one.json"  data file="data_fit/$RELEASE_DIR/fit_$i.json" output file="output_fit/$RELEASE_DIR/fit_$i.csv"
done
