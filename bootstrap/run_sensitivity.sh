PM=../StarSharp/data/pointing_model/default.ecsv
OUTDIR=sensitivity
mkdir -p "$OUTDIR"

# 3.3: Cam_dz only per band
for band in u g r i z y; do
    python compute_sensitivity.py --version 3.3 --band $band \
        --pointing-model $PM \
        --rtp-lookup focus_scan/v3.3/v3.3_$band.ecsv \
        --algorithm zk \
        --output $OUTDIR/3.3_$band.fits
done

# 3.12: shared Cam_rx/Cam_ry + per-band Cam_dz
for band in u g r i z y; do
    python compute_sensitivity.py --version 3.12 --band $band \
        --pointing-model $PM \
        --rtp-lookup focus_scan/v3.12/v3.12_$band.ecsv \
        --algorithm zk \
        --output $OUTDIR/3.12_$band.fits
done

# 3.14: shared Cam_dx/Cam_dy + per-band Cam_dz/rx/ry
for band in u g r i z y; do
    python compute_sensitivity.py --version 3.14 --band $band \
        --pointing-model $PM \
        --rtp-lookup focus_scan/v3.14/v3.14_$band.ecsv \
        --algorithm zk \
        --output $OUTDIR/3.14_$band.fits
done
