import numpy as np
import astropy.units as u
import asdf
from lsst.daf.butler import Butler, DatasetNotFoundError
from tqdm import tqdm
from astropy.table import vstack


# Known collections
# repo: "LSSTCam"  # summit
#   LSSTCam/runs/quickLook

# "/repo/main"  # USDF
#   "u/brycek/aosBaseline_lsstcam_triplets_inFocus",  # in-focus science 2nd moments
#   "u/brycek/aos_lsstcam_triplets_step2/danish/wep_v14_13_1/donut_viz_v2_0_4",  # FAM
#   "u/brycek/aos_lsstcam_triplets_step2/danish_dense/wep_v14_13_1/donut_viz_v2_0_4",  # FAM
#   "u/brycek/aos_lsstcam_triplets_inFocusVisits_wavefront_step2/danish/wep_v14_13_1/donut_viz_v2_0_4",  # corner mode
#   "u/brycek/aos_lsstcam_triplets_inFocusVisits_wavefront_step2/danish_dense/wep_v14_13_1/donut_viz_v2_0_4",  # corner mode


def makeTableFromSourceCatalogs(srcs, visitInfo, desc):
    tables = []
    for det, src in tqdm(srcs.items(), desc=desc, leave=False):
        tab = src.asAstropy()
        w = tab["calib_psf_candidate"]
        tab = tab[w]

        pix_to_arcsec = 0.2 * u.arcsec / u.pix
        # Rename some columns
        tab["x"] = tab["base_FPPosition_x"]
        tab["y"] = tab["base_FPPosition_y"]
        tab["Ixx"] = tab["ext_shapeHSM_HsmSourceMoments_xx"] * pix_to_arcsec**2
        tab["Ixy"] = tab["ext_shapeHSM_HsmSourceMoments_xy"] * pix_to_arcsec**2
        tab["Iyy"] = tab["ext_shapeHSM_HsmSourceMoments_yy"] * pix_to_arcsec**2
        tab["PSF_Ixx"] = tab["ext_shapeHSM_HsmPsfMoments_xx"] * pix_to_arcsec**2
        tab["PSF_Ixy"] = tab["ext_shapeHSM_HsmPsfMoments_xy"] * pix_to_arcsec**2
        tab["PSF_Iyy"] = tab["ext_shapeHSM_HsmPsfMoments_yy"] * pix_to_arcsec**2
        tab["detector"] = det
        tab = tab[["detector", "x", "y", "Ixx", "Ixy", "Iyy", "PSF_Ixx", "PSF_Ixy", "PSF_Iyy"]]
        tables.append(tab)
    table = vstack(tables, metadata_conflicts="silent")

    table.meta["rotTelPos"] = (
        (visitInfo.boresightParAngle - visitInfo.boresightRotAngle).asRadians() - np.pi / 2
    )
    table.meta["rotSkyPos"] = visitInfo.boresightRotAngle.asRadians()

    return table


def main(args):
    butler = Butler(
        args.butler,
        collections=args.collections,
        instrument="LSSTCam",
    )
    records = butler.query_dimension_records(
        "exposure",
        where=args.where,
    )

    for record in tqdm(records, desc="Records"):
        day_obs = record.day_obs
        seq_num = record.seq_num
        visitInfo = None

        try:
            aos_corner_avg = butler.get("aggregateAOSVisitTableAvg", day_obs=day_obs, seq_num=seq_num)
            aos_corner_raw = butler.get("aggregateAOSVisitTableRaw", day_obs=day_obs, seq_num=seq_num)
        except DatasetNotFoundError:
            continue

        desc = f"{day_obs=} {seq_num=}"
        srcs = {}
        # Outer tqdm for detectors, nested inside records tqdm
        for det in tqdm(range(189), desc=f"Reading detectors {desc}", leave=False):
            dataId = dict(
                day_obs=day_obs,
                seq_num=seq_num,
                detector=det,
            )
            try:
                srcs[det] = butler.get("single_visit_star_footprints", **dataId)
            except DatasetNotFoundError:
                continue
            if visitInfo is None:
                try:
                    visitInfo = butler.get("post_isr_image.visitInfo", **dataId)
                except DatasetNotFoundError:
                    continue
        if len(srcs) == 0:
            continue
        src = makeTableFromSourceCatalogs(srcs, visitInfo, desc=f"Consolidating detectors {desc}")

        with asdf.AsdfFile(
            {
                "src":src,
                "aos_corner_avg":aos_corner_avg,
                "aos_corner_raw":aos_corner_raw,
            }
        ) as af:
            af.write_to(f"{day_obs}{seq_num:05d}.asdf")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--butler", type=str, default="LSSTCam")
    parser.add_argument("--collections", nargs='+', type=str, default=["LSSTCam/runs/quickLook"])
    parser.add_argument("--where", type=str, default=None)

    args = parser.parse_args()

    main(args)
