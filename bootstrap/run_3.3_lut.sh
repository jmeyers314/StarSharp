WORKING=focus_scan/v3.3
PM=../StarSharp/data/pointing_model/default.ecsv
PROD=../StarSharp/data/rtp_lookup
mkdir -p $WORKING/

# # Test
# NX=1
# NRAD=3

# Production
NX=4
NRAD=6

# Step 1: 1-DOF runs per band
for band in u g r i z y; do
    python focus_scan.py \
    --version 3.3 \
    --band $band \
    --dofs Cam_dz \
    --nx $NX \
    --nrad $NRAD \
    --rtp-step 30 \
    --pointing-model $PM \
    --output $WORKING/raw_v3.3_$band.ecsv
done

# Step 2: fit Cam_dz with a constant per band (wavelength-dependent, no cross-band averaging)
for band in u g r i z y; do
    python refit_lut.py \
        $WORKING/raw_v3.3_$band.ecsv \
        --const-cols Cam_dz \
        --step 30 \
        --output $WORKING/v3.3_$band.ecsv
done

# Step 3: copy to production
for band in u g r i z y; do
    cp -f $WORKING/v3.3_$band.ecsv $PROD/
done
