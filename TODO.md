# TODO List


## High Priority
- [ ] FAM donut -> spot
  • [ ] model.zernikes() and model.spot() get zk, dzk args
- [ ] Compare StarSharp/MTAOS sensitivity
- [ ] Compare StarSharp/MTAOS vmodes
- [ ] Check that sensitivity in zernikeGQ is consistent with sensitivity in zernikeTA
- [ ] Compare on-the-fly computed intrinsic to value from butler

## Medium Priority
- [ ] Data access classes
- [ ] Think through requiring dz.single(field) to have dz.rtp == field.rtp...
- [ ] Run more nights
- [ ] Create tutorial notebooks

## Low Priority
- [ ] Hook into danish


## Resolved
- [x] Add fitting for Zernikes -> State
- [x] Add zernikeTA option in RaytracedOpticalModel.zernikes
- [x] Verify that batoid.RayVector.fromStop() is independent of state.
    See sandbox/verify_fromStop_indep.py
- [x] Add chief/mean option in RaytracedOpticalModel.spots
- [x] Check that chief ray displacements are reasonable and continuous across FOV
    Chief ray is problematic!  Replacing with reference="ring" works wonders.
    See sandbox/verify_cr_continuity.py
