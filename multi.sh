#!/bin/zsh
coma_batch(){
        for i in {1..10}
        do
                ./coma sample algorithm=hmc engine=nuts max_depth=18 adapt delta=0.95 num_warmup=1000 num_samples=1000 num_chains=4 data file=data/SGA_TFR_simtest_${(l:3::0:)i}.json output file=output/coma_410_${(l:3::0:)i}.csv
        done
}


fuji(){
        ./coma sample algorithm=hmc engine=nuts max_depth=19 adapt delta=0.95 num_warmup=1000 num_samples=1000 num_chains=4 \ 
                data file=data/SGA-2020_fuji_Vrot.json output file=output/fuji_400.csv
}

diagnose_batch(){
        for i in {1..10}
        do
                ../bin/diagnose output/coma_410_${(l:3::0:)i}_*.csv
        done
}

stansummary_batch(){
        for i in {1..10}
        do
                ../bin/stansummary -c coma_ss_410_${(l:3::0:)i}.csv -p 32,68 -s 3 output/coma_410_${(l:3::0:)i}_*.csv
        done
}

iron(){
        ./iron sample algorithm=hmc engine=nuts max_depth=16 adapt delta=0.95 num_warmup=1000 num_samples=1000 num_chains=4 data file=data/SGA-2020_iron_Vrot_sub.json output file=output/iron_210_sub.csv
}
