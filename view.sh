#!/bin/zsh
for i in {1..10}
do
        ../bin/stansummary output_${(l:3::0:)i}.csv | grep aR
done
