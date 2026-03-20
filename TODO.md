# TODO List


## High Priority
- [ ] Check that sensitivity in zernikeGQ is consistent with sensitivity in zernikeTA
- [ ] Compare StarSharp/MTAOS sensitivity
- [ ] Compare StarSharp/MTAOS vmodes

## Medium Priority
- [ ] Data access classes

## Low Priority


## Resolved
- [x] Add fitting for Zernikes -> State
- [x] Add zernikeTA option in RaytracedOpticalModel.zernikes
- [x] Verify that batoid.RayVector.fromStop() is independent of state.
    See sandbox/verify_fromStop_indep.py
- [x] Add chief/mean option in RaytracedOpticalModel.spots
- [x] Check that chief ray displacements are reasonable and continuous across FOV
    Chief ray is problematic!  Replacing with reference="ring" works wonders.
    See sandbox/verify_cr_continuity.py
