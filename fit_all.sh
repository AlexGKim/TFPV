#!/bin/zsh

for i in {0..8716}
do
  ./fit sample algorithm=hmc engine=nuts num_warmup=1000 num_samples=1000 num_chains=4 init="data_fit/Y1/fit_init.json"  data file="data_fit/Y1/fit_$i.json" output file="output_fit/Y1/fit_$i.csv"
done