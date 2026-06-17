#!/usr/bin/env bash
# Phase A sufficiency check: is the existing v3.14 RTP lookup good enough for the
# v1000 optic, or do we need to bootstrap v1000 from scratch?
#
# For each band we run a focus scan on the v1000 optic with the *v3.14* lookup
# applied as the model offset.  focus_scan reports:
#   rms_nominal   = spot RMS with the v3.14 lookup applied, no residual fit
#   rms_optimized = spot RMS after fitting a fresh 5-DOF residual correction
# A small (rms_nominal - rms_optimized) gap across all bands/angles means the
# v3.14 lookup already lands near the v1000 optimum -> v3.14 is sufficient.
set -euo pipefail

WORKING=focus_scan/v1000_check
PROD=../StarSharp/data/rtp_lookup
PM=../StarSharp/data/pointing_model/default.ecsv
mkdir -p $WORKING/

NX=4
NRAD=6

for band in u g r i z y; do
    python focus_scan.py \
        --version 1000 \
        --band $band \
        --dofs Cam_dz Cam_dx Cam_dy Cam_rx Cam_ry \
        --nx $NX \
        --nrad $NRAD \
        --rtp-lookup $PROD/v3.14_$band.ecsv \
        --pointing-model $PM \
        --output $WORKING/check_v1000_$band.ecsv
done

# Aggregate the residual-blur gap across all bands and RTP angles.
python summarize_v1000_check.py $WORKING/check_v1000_*.ecsv
