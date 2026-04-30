WORKING=focus_scan/v3.14
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
        --version 3.14 \
        --band $band \
        --dofs Cam_dz Cam_rx Cam_ry \
        --nx $NX \
        --nrad $NRAD \
        --pointing-model $PM \
        --output $WORKING/raw_v3.14_$band.ecsv
done

# Step 2: average Cam_rx/Cam_ry across bands (wavelength-independent tilt)
python combine_lut_bands.py \
    $WORKING/raw_v3.14_*.ecsv \
    --cols Cam_rx Cam_ry \
    --output $WORKING/avg_rxry_v3.14.ecsv

# Step 3: fit sin to Cam_rx/Cam_ry, output at 3 deg resolution
python refit_lut.py \
    $WORKING/avg_rxry_v3.14.ecsv \
    --sin-cols Cam_rx Cam_ry \
    --step 3 \
    --output $WORKING/refit_rxry_v3.14.ecsv

# Step 4: optimize only Cam_dx/Cam_dy with Cam_dz/Cam_rx/Cam_ry pre-applied
for band in u g r i z y; do
    python focus_scan.py \
        --version 3.14 \
        --band $band \
        --dofs Cam_dx Cam_dy \
        --nx $NX \
        --nrad $NRAD \
        --rtp-lookup $WORKING/refit_rxry_v3.14.ecsv \
        --pointing-model $PM \
        --output $WORKING/refocus1_v3.14_$band.ecsv
done

# Step 5: combine again, but also dx, dy
python combine_lut_bands.py \
    $WORKING/refocus1_v3.14_*.ecsv \
    --cols Cam_rx Cam_ry Cam_dx Cam_dy \
    --output $WORKING/avg_rxrydxdy_v3.14.ecsv

# Step 6: fit to sin
python refit_lut.py \
    $WORKING/avg_rxrydxdy_v3.14.ecsv \
    --sin-cols Cam_rx Cam_ry Cam_dx Cam_dy \
    --step 3 \
    --output $WORKING/refit_rxrydxdy_v3.14.ecsv

# Step 7: refocus Cam_dz per band with shift+tilt pre-applied
for band in u g r i z y; do
    python focus_scan.py \
        --version 3.14 \
        --band $band \
        --dofs Cam_dz \
        --nx $NX \
        --nrad $NRAD \
        --rtp-lookup $WORKING/refit_rxrydxdy_v3.14.ecsv \
        --pointing-model $PM \
        --output $WORKING/refocus2_v3.14_$band.ecsv
done

# Step 8: refit rx, ry, dx, dy to sin, resample Cam_dz to 3 deg grid
for band in u g r i z y; do
    python refit_lut.py \
        $WORKING/refocus2_v3.14_$band.ecsv \
        --sin-cols Cam_rx Cam_ry Cam_dx Cam_dy \
        --const-cols Cam_dz \
        --step 3 \
        --output $WORKING/v3.14_$band.ecsv
done

# Step 9: copy to production
for band in u g r i z y; do
    cp -f $WORKING/v3.14_$band.ecsv $PROD/
done
