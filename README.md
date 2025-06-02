# TFPV
DESI Tully Fisher Peculiar Velocity

## Instructions

### Data

#### Data Environmental variables
Location of input and output data files are communicated through environmental variables.  For example on my local computer I do
```
export DESI_SGA_DIR=/Users/akim/Projects/DESI_SGA
export OUTPUT_DIR=/Users/akim/Projects/TFPV/output 
export DATA_DIR=/Users/akim/Projects/TFPV/data
export RELEASE=Y1
```

#### Input data
- Assume that there is a public github repo and a private NERSC filesystem that has all the needed data
- Different sets of data in RELEASE subdirectories
- Download DESI_SGA repository https://github.com/DESI-UR/DESI_SGA
  - Set environmental variable DESI_SGA_DIR to the directory
  - Tully's table of cluster information $DESI_SGA_DIR/Tully15-Table3.fits
  - Set environmental variable RELEASE to the appropriate release.  Input data is in $DESI_SGA_DIR/TF/$RELEASE. Default is Y1.
  - Clusters and which galaxies are in them $DESI_SGA_DIR/TF/$RELEASE/output_??????.txt
- Other required private data
  - Data is at NERSC /global/cfs/cdirs/desi/science/td/pv/tfgalaxies/
  - If working locally retrieve the needed RELEASEs
  - Set environmental variable DATA_DIR to the directory where these files will be located
  - Relevent data in $DATA_DIR/$RELEASE/
    - SGA-2020_iron_Vrot_VI.fits Note that this name should presumably be changed to something more generic.
   
#### Ouput data
- Set environmental variable OUTPUT_DIR to where created data go $OUTPUT_DIR/$RELEASE/. 

### Docker image
- The docker image is alexgkim/tfpv:dev
- docker run -v /Users/akim/Projects/TFPV/data:/data -v /Users/akim/Projects/TFPV/output:/output -v /Users/akim/Projects/DESI_SGA:/DESI_SGA alexgkim/tfpv:dev ./command.sh
- shifter --volume=/global/homes/a/akim/tmpdata:/data --volume=/global/homes/a/akim/tmpoutput:/output --volume=/global/homes/a/akim/tmpdesi_sga:/DESI_SGA --image=alexgkim/tfpv:dev ./command/sh

### Local install

- Install cmdstan
- Download TFPV repository https://github.com/AlexGKim/TFPV.  Personally I cloned the repository to live inside the cmdstan installation cmdstan/TFPV given that conformed to the STAN examples.
- I don't have a "requirements.txt" file with all the python packages needed to run the software.  Just create a new environment, run, crash, install, repeat...
- Create the data and initial condition files input to STAN.  This is done with the command "python fitstojson.py".  If you look at the file filtstojson.py, you will find several functions that can be called through __main__.
  - iron_cluster_json() creates the data for IRON, coma_json() creates the data for EDR.
  - coma_json() has a keyword for cuts, which in my analysis is set to True.
- The STAN code to fit IRON is cluster.stan, the code to fit Coma is coma.stan.
- Each code can be run with different configurations.  The one to care about is controled by the variable "dispersion_case".  "dispersion_case=4" is the perpendicular fit, "dispersion_case=3" is the ITF fit despite what is written in the comments.
- Use standard cmdstan to compile the STAN code.  From cmdstan/ "make TFPV/cluster" for example.  You will find a new executable TFPV/cluster. Personally I immediately rename it to include the dispersion case.  So I have files "cluster311", "cluster411", "coma411", "coma311"
- At the top of cluster.stan, coma.stan are commented out command line commands I used to run the fit.  Run them and hope for the best.  Note it puts the output into a directory "output" you have to create.
- STAN provides two useful tools to check the result.  The first checks for the convergence. from TFPV  "../bin/diagnose output/cluster_311_?.csv" for example.  The second shows statistics on the posterior "../bin/stansummary output/cluster_311_?.csv"
- Plots for both cluster and coma runs are made using methods in plot_iron.py
