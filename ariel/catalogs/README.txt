These represent catalogs generated from the same data and different algorithms.  They have overlapping galaxies.  The critical column is log_distance.There are systematically different solutions, particularly at low-redshift.

There are two analysis frameworks, Hyper and Hier.
Hyper is
DESI-DR2_TF_pv_cat_v5a.fits
DESI-DR2_TF_pv_cat_v7a.fits
DESI-DR2_TF_pv_test_cat_v9.fits
Hier is
DESI-DR2_TF_pv_cat_v10b.fits
DESI-DR2_TF_pv_cat_v5_3.fits
DESI-DR2_TF_pv_cat_v5b.fits

There are analyses using z>0.01
DESI-DR2_TF_pv_cat_v5a.fits
DESI-DR2_TF_pv_cat_v5b.fits

There are analyses using z>0.03
DESI-DR2_TF_pv_cat_v7a.fits
DESI-DR2_TF_pv_cat_v5_3.fits

There are analyses allowing redshift-dependent Tully-Fisher Relation slopes
DESI-DR2_TF_pv_cat_v10b.fits
DESI-DR2_TF_pv_test_cat_v9.fits

Two corrections to the above, made 2026-11-18:
  - the z>0.03 list said "v5c", which is not a catalogue in this directory;
    the z>0.03 Hier rerun is v5_3.
  - v5_3 was listed under the redshift-dependent-slope analyses. It is not
    one: its configs differ from v5's in "run" and "z_obs_min" only and both
    set "model": "2color" (a fixed slope), and run_v5_3.sh's header says the
    same. The z-dependent-slope Hier catalogue is v10b. lowz_common.py
    repeated this error in CatInfo(zslope=True) and has been corrected too.

