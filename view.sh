#!/bin/zsh
for i in {1..10}
do
        ../bin/stansummary output/simtest_perp_${(l:3::0:)i}_*.csv | grep aR
done
