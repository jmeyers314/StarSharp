# TODO List


## High Priority
- [ ] LinearModel(RaytracedModel) for a single RTP.  Could be useful for focal-plane fitting.
      Could be hybrid of RaytracedModel + pre-built sensitivity matrix for dof.
- [ ] Single focal plane star images for test dataset.
- [ ] Spot "convolution" by outer sum with Gaussian distribution.
- [ ] Weighted moments both in simulation and data.  Use the same 2D weight.
- [ ] Fitter for moments
- [ ] Need to linearize Spots.field too since that's an output.

## Medium Priority
- [ ] Compare zernikeGQ sensitivity to zernikeTA sensitivity
- [ ] Compare StarSharp/MTAOS sensitivity
- [ ] FAM donut -> spot
  • [ ] model.zernikes() and model.spot() get zk, dzk args
- [ ] Compare on-the-fly computed intrinsic to value from butler
- [ ] Create tutorial notebooks
- [ ] Data access classes
- [ ] Think through requiring dz.single(field) to have dz.rtp == field.rtp...
- [ ] Run more nights
- [ ] Check sensitivity to plate scale!

## Low Priority
- [ ] Hook into danish
- [ ] Predict in-focus images by outer sum with atm?
- [ ] Implies could simultaneously fit donuts and in-focus at pixel level!


## Resolved
- [x] Add fitting for Zernikes -> State
- [x] Add zernikeTA option in RaytracedOpticalModel.zernikes
- [x] Verify that batoid.RayVector.fromStop() is independent of state.
    See sandbox/verify_fromStop_indep.py
- [x] Add chief/mean option in RaytracedOpticalModel.spots
- [x] Check that chief ray displacements are reasonable and continuous across FOV
    Chief ray is problematic!  Replacing with reference="ring" works wonders.
    See sandbox/verify_cr_continuity.py
- [x] Compare StarSharp/MTAOS vmodes
