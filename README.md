# TFPV
DESI Tully Fisher Peculiar Velocity

I cloned the repository in the my cmdstan directory to have cmdstan/TFPV

Create the json file needed by STAN from Kelly's FITS file

python fitstojson

Compile code

make TFPV/tfpv  

Run code from TFPV

./tfpv sample data file=SGA-2020_iron_Vrot.json

See results

../bin/stansummary output.csv

