#!/bin/bash
python /opt/TFPV-docker/fitstojson.py
/opt/TFPV-docker/cluster sample algorithm=hmc engine=nuts \
    max_depth=17 adapt delta=0.999 num_warmup=3000 num_samples=1000 num_chains=4 \
    init=$OUTPUT_DIR/$RELEASE_DIR/iron_cluster_init.json data file=$OUTPUT_DIR/$RELEASE_DIR/iron_cluster.json output file=$OUTPUT_DIR/$RELEASE_DIR/cluster.csv
