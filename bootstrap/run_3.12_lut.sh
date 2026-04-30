WORKING=focus_scan/v3.12
PM=../StarSharp/data/pointing_model/default.ecsv
PROD=../StarSharp/data/rtp_lookup
mkdir -p $WORKING/

# # Test
# NX=1
# NRAD=3

# Production
NX=4
NRAD=6


# Step 1: 3-DOF runs per band
for band in u g r i z y; do
    python focus_scan.py \
        --version 3.12 \
        --band $band \
        --dofs Cam_dz Cam_rx Cam_ry \
        --nx $NX \
        --nrad $NRAD \
        --pointing-model $PM \
        --output $WORKING/raw_v3.12_$band.ecsv
done

# Step 2: average Cam_rx/Cam_ry across bands (wavelength-independent tilt)
python combine_lut_bands.py \
    $WORKING/raw_v3.12_*.ecsv \
    --cols Cam_rx Cam_ry \
    --output $WORKING/avg_rxry_v3.12.ecsv

# Step 3: fit sin to Cam_rx/Cam_ry, output at 3 deg resolution
python refit_lut.py \
    $WORKING/avg_rxry_v3.12.ecsv \
    --sin-cols Cam_rx Cam_ry \
    --step 3 \
    --output $WORKING/refit_rxry_v3.12.ecsv

# Step 4: refocus Cam_dz per band with tilt pre-applied (preserves band-dependence)
for band in u g r i z y; do
    python focus_scan.py \
        --version 3.12 \
        --band $band \
        --dofs Cam_dz \
        --nx $NX \
        --nrad $NRAD \
        --rtp-lookup $WORKING/refit_rxry_v3.12.ecsv \
        --pointing-model $PM \
        --output $WORKING/refocus_v3.12_$band.ecsv
done

# Step 5: refit Cam_rx, Cam_ry to sin, resample Cam_dz to 3 deg grid
for band in u g r i z y; do
    python refit_lut.py \
        $WORKING/refocus_v3.12_$band.ecsv \
        --sin-cols Cam_rx Cam_ry \
        --const-cols Cam_dz \
        --step 3 \
        --output $WORKING/v3.12_$band.ecsv
done

# Step 6: copy to production
for band in u g r i z y; do
    cp -f $WORKING/v3.12_$band.ecsv $PROD/
done
