from itertools import chain
from lsst.afw.geom.ellipses import Quadrupole
from lsst.geom import LinearTransform
from tqdm import tqdm
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
import asdf
import warnings
warnings.filterwarnings(
    "ignore",
    category=Warning,
    message=r".*The unit 'erg' has been deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=asdf.exceptions.AsdfPackageVersionWarning
)
from star_sharp import StarSharp, grid_measurements
from figure import layout_singlet_figure, layout_triplet_figure, AxisText
from scipy.optimize import least_squares
import astropy.units as u

SIGMA_TO_FWHM = np.sqrt(np.log(256))
MM_TO_DEGREE = 100*0.2/3600

ALL_DOFS = []
for pod in ["M2", "Cam"]:
    ALL_DOFS.append((f"{pod} dz", "μm"))
    ALL_DOFS.append((f"{pod} dx", "μm"))
    ALL_DOFS.append((f"{pod} dy", "μm"))
    ALL_DOFS.append((f"{pod} rx", "arcsec"))
    ALL_DOFS.append((f"{pod} ry", "arcsec"))
for mirror in ["M1M3", "M2"]:
    for i in range(1, 21):
        ALL_DOFS.append((f"{mirror} B{i}", "μm"))

def apply_transform(table, transform, prefix):
    transform = LinearTransform(transform)
    rotShapes = [
        Quadrupole(row["Ixx"], row["Iyy"], row["Ixy"]).transform(transform)
        for row in table
    ]
    table[prefix + "_Ixx"] = [sh.getIxx() for sh in rotShapes]
    table[prefix + "_Iyy"] = [sh.getIyy() for sh in rotShapes]
    table[prefix + "_Ixy"] = [sh.getIxy() for sh in rotShapes]
    table[prefix + "_x"] = transform[0, 0] * table["x"] + transform[0, 1] * table["y"]
    table[prefix + "_y"] = transform[1, 0] * table["x"] + transform[1, 1] * table["y"]
    return table

def augment_moments(table, prefix):
    table[prefix + "T"] = table[prefix + "Ixx"] + table[prefix + "Iyy"]
    # w for whisker = unnormalized ellipticity moments
    table[prefix + "w1"] = table[prefix + "Ixx"] - table[prefix + "Iyy"]
    table[prefix + "w2"] = 2 * table[prefix + "Ixy"]
    table[prefix + "w"] = np.sqrt(table[prefix + "w1"]**2 + table[prefix + "w2"]**2)
    # e for ellipticity
    table[prefix + "e1"] = table[prefix + "w1"] / table[prefix + "T"]
    table[prefix + "e2"] = table[prefix + "w2"] / table[prefix + "T"]
    table[prefix + "e"] = np.sqrt(table[prefix + "e1"]**2 + table[prefix + "e2"]**2)
    # beta for the sky-angle of ellipticity (distinct from the phase angle)
    table[prefix + "beta"] = 0.5 * np.arctan2(table[prefix + "e2"], table[prefix + "e1"])
    # Whisker components for quiver plotting
    table[prefix + "wx"] = table[prefix + "w"] * np.cos(table[prefix + "beta"])
    table[prefix + "wy"] = table[prefix + "w"] * np.sin(table[prefix + "beta"])
    # Ellipticity components for quiver plotting
    table[prefix + "ex"] = table[prefix + "e"] * np.cos(table[prefix + "beta"])
    table[prefix + "ey"] = table[prefix + "e"] * np.sin(table[prefix + "beta"])
    table[prefix + "FWHM"] = np.sqrt(0.5 * table[prefix + "T"]) * SIGMA_TO_FWHM

    return table

def conv_moments_residual(
    x, data, model,
):
    Ixx, Iyy, Ixy = x
    w = np.isfinite(data["Ixx"]) & np.isfinite(data["Iyy"]) & np.isfinite(data["Ixy"])
    out = np.concatenate([
        data["Ixx"][w] - model["Ixx"][w] - Ixx,
        data["Iyy"][w] - model["Iyy"][w] - Iyy,
        data["Ixy"][w] - model["Ixy"][w] - Ixy
    ])
    return out


def main(args):
    with asdf.open(args.filename) as af:
        src = af["src"]
        aos_corner_avg = af["aos_corner_avg"]
        try:
            aos_fam_avg = af["aos_fam_avg"]
        except KeyError:
            aos_fam_avg = None
        visit_id = src.meta["LSST BUTLER DATAID VISIT"]  # workaround
    if len(src) == 0:
        raise ValueError("No source measurements found in the input file.")

    # Parse the use_dof argument
    # which will in general be something like: "3-6,9,10,20-24"
    # first strip whitespace
    dof_str = args.use_dof.strip()
    use_dof = []
    for part in dof_str.split(","):
        if "-" in part:
            start, end = [int(p) for p in part.split("-")]
            use_dof.extend(range(start, end + 1))
        else:
            use_dof.append(int(part))
    use_dof = np.sort(use_dof)
    nkeep = args.nkeep
    if nkeep < 0:
        nkeep = None

    w = np.isfinite(src["Ixx"])
    w &= np.isfinite(src["Ixy"])
    w &= np.isfinite(src["Iyy"])
    src = src[w]

    rtp = src.meta["rotTelPos"]
    srtp, crtp = np.sin(rtp), np.cos(rtp)
    aaRot = np.array([[crtp, srtp], [-srtp, crtp]]) @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]])
    src = apply_transform(src, aaRot, "aa")

    src = augment_moments(src, "")
    src = augment_moments(src, "aa_")

    ssh = StarSharp(
        band = "r",
        use_dof = use_dof,
        nkeep = nkeep,
        transverse_pupil_radii = args.transverse_pupil_radii,
        transverse_field_radii = args.transverse_field_radii,
        wf_kmax = args.wf_kmax,
        wf_jmax = args.wf_jmax,
        tqdm = tqdm,
    )

    gridded = grid_measurements(
        src["aa_x"] * MM_TO_DEGREE,
        -src["aa_y"] * MM_TO_DEGREE,  # +ve alt is -ve y_OCS
        {
            "Ixx":src["aa_Ixx"],
            "Ixy":-src["aa_Ixy"], # +ve alt is -ve y_OCS
            "Iyy":src["aa_Iyy"]
        },
        ssh.field_u, ssh.field_v,
    )
    gridded = augment_moments(gridded, "")
    gridded["u"] = ssh.field_u
    gridded["v"] = ssh.field_v

    Ixx = Ixy = Iyy = None
    # If either wavefront is selected, start assembling inputs
    if args.fit_corners or args.fit_fam:
        thx = None
        if args.fit_corners:
            thx = np.rad2deg(aos_corner_avg["thx_OCS"])
            thy = np.rad2deg(aos_corner_avg["thy_OCS"])
            zk = aos_corner_avg["zk_OCS"]
            use_zk = aos_corner_avg.meta["nollIndices"]
        if args.fit_fam:
            if aos_fam_avg is None:
                raise ValueError("No full array mode data available.")
            if thx is None:
                thx = np.rad2deg(aos_fam_avg["thx_OCS"])
                thy = np.rad2deg(aos_fam_avg["thy_OCS"])
                zk = aos_fam_avg["zk_OCS"]
                use_zk = aos_fam_avg.meta["nollIndices"]
            else:
                assert np.all(aos_fam_avg.meta["nollIndices"] == use_zk)
                thx = np.concatenate([thx, np.rad2deg(aos_fam_avg["thx_OCS"])])
                thy = np.concatenate([thy, np.rad2deg(aos_fam_avg["thy_OCS"])])
                zk = np.concatenate([zk, aos_fam_avg["zk_OCS"]])
        if args.fit_moments:
            result = ssh.fit_both(
                gridded,
                thx,
                thy,
                zk,
                use_zk=use_zk,
            )
            Ixx = result["Ixx"]
            Ixy = result["Ixy"]
            Iyy = result["Iyy"]
        else:  # Only WF
            result = ssh.fit_wf(thx, thy, zk, use_zk=use_zk)
            model_moments = ssh.moments_model(result["state"]) # optics only
            # Fit for Ixx, Ixy, Iyy holding optics fixed
            guess = [np.sqrt(1.0/2.35), np.sqrt(1.0/2.35), 0.0]
            Ixx, Iyy, Ixy = least_squares(conv_moments_residual, guess, args=(gridded, model_moments)).x
    else:
        if not args.fit_moments:
            raise ValueError("At least one of --fit_moments, --fit_fam, or --fit_corners must be specified.")
        result = ssh.fit_moments(gridded)
        Ixx = result["Ixx"]
        Ixy = result["Ixy"]
        Iyy = result["Iyy"]
    model_moments = ssh.moments_model(result["state"], dIxx=Ixx, dIyy=Iyy, dIxy=Ixy)
    opt_moments = ssh.moments_model(result["state"])
    # resid_moments = {
    #     "Ixx": gridded["Ixx"] - model_moments["Ixx"],
    #     "Iyy": gridded["Iyy"] - model_moments["Iyy"],
    #     "Ixy": gridded["Ixy"] - model_moments["Ixy"]
    # }
    # resid_moments = augment_moments(resid_moments, "")

    if args.plot is not None:
        # We'll use these a lot, so unpack here
        u = gridded["u"]
        v = gridded["v"]

        do_triplet_layout = args.plot_triplet or (aos_fam_avg is not None and not args.plot_singlet)
        if do_triplet_layout:
            layout = layout_triplet_figure()
            zk_axs = layout["zk_axs"]
        else:
            layout = layout_singlet_figure()

        # Unpack
        fig = layout["fig"]
        fig.suptitle(f"Visit {visit_id}")

        corner_axs=layout["corner_axs"]
        char_axs=layout["char_axs"]
        dz_ax=layout["dz_ax"]
        text_ax=layout["text_ax"]

        shape_axs = layout["shape_axs"]
        fwhm_cax=layout["fwhm_cax"]
        ellip_cax=layout["ellip_cax"]

        moments_axs = layout["moments_axs"]
        Tsqr_cax = layout["Tsqr_cax"]
        w_cax = layout["w_cax"]

        # Moments panels
        s = 40 if not do_triplet_layout else 10
        T_kwargs = dict(s=s, vmin=0.0, vmax=2.0, cmap="turbo")
        w_kwargs = dict(s=s, vmin=-0.5, vmax=0.5, cmap="bwr")
        T_scatter = moments_axs[0, 0].scatter(
            u,
            v,
            c=gridded["T"],
            **T_kwargs,
        )
        w_scatter = moments_axs[0, 1].scatter(
            u,
            v,
            c=gridded["w1"],
            **w_kwargs,
        )
        moments_axs[0, 2].scatter(
            u,
            v,
            c=2 * gridded["w2"],
            **w_kwargs,
        )

        moments_axs[1, 0].scatter(
            u,
            v,
            c=model_moments["T"],
            **T_kwargs,
        )
        moments_axs[1, 1].scatter(
            u,
            v,
            c=model_moments["w1"],
            **w_kwargs,
        )
        moments_axs[1, 2].scatter(
            u,
            v,
            c=2 * model_moments["w2"],
            **w_kwargs,
        )
        moments_axs[2, 0].scatter(
            u,
            v,
            c=gridded["T"] - model_moments["T"],
            **w_kwargs,
        )
        moments_axs[2, 1].scatter(
            u,
            v,
            c=gridded["w1"] - model_moments["w1"],
            **w_kwargs,
        )
        moments_axs[2, 2].scatter(
            u,
            v,
            c=2 * (gridded["w2"] - model_moments["w2"]),
            **w_kwargs,
        )

        # Shape panels
        FWHM_kwargs = dict(s=s, vmin=0.5, vmax=2.0, cmap="turbo")
        e_kwargs = dict(s=s, vmin=-0.3, vmax=0.3, cmap="bwr")

        shape_axs[0, 0].quiver(
            u,
            v,
            gridded["wx"],
            gridded["wy"],
            headlength=0,
            headaxislength=0,
            scale=1.0,
            pivot="middle",
        )
        shape_axs[1, 0].quiver(
            u,
            v,
            model_moments["wx"],
            model_moments["wy"],
            headlength=0,
            headaxislength=0,
            scale=1.0,
            pivot="middle",
        )

        fwhm_scatter = shape_axs[0, 1].scatter(
            u,
            v,
            c=gridded["FWHM"],
            **FWHM_kwargs
        )
        shape_axs[1, 1].scatter(
            u,
            v,
            c=model_moments["FWHM"],
            **FWHM_kwargs
        )
        shape_axs[2, 1].scatter(
            u,
            v,
            c=gridded["FWHM"] - model_moments["FWHM"],
            **e_kwargs  # Centered around 0.0
        )

        e_scatter = shape_axs[0, 2].scatter(
            u,
            v,
            c=gridded["e1"],
            **e_kwargs
        )
        shape_axs[1, 2].scatter(
            u,
            v,
            c=model_moments["e1"],
            **e_kwargs
        )
        shape_axs[2, 2].scatter(
            u,
            v,
            c=gridded["e1"] - model_moments["e1"],
            **e_kwargs
        )

        shape_axs[0, 3].scatter(
            u,
            v,
            c=gridded["e2"],
            **e_kwargs
        )
        shape_axs[1, 3].scatter(
            u,
            v,
            c=model_moments["e2"],
            **e_kwargs
        )
        shape_axs[2, 3].scatter(
            u,
            v,
            c=gridded["e2"] - model_moments["e2"],
            **e_kwargs
        )

        fig.colorbar(T_scatter, cax=Tsqr_cax, label="arcsec$^2$")
        fig.colorbar(w_scatter, cax=w_cax, label="arcsec$^2$")
        fig.colorbar(fwhm_scatter, cax=fwhm_cax, label="arcsec")
        fig.colorbar(e_scatter, cax=ellip_cax)

        # FAM Zernikes panels
        if do_triplet_layout and aos_fam_avg is not None:
            model_fam_zks = ssh.wf_model(
                np.rad2deg(aos_fam_avg["thx_OCS"]),
                np.rad2deg(aos_fam_avg["thy_OCS"]),
                result["state"],
            )
            zk_kwargs = dict(s=20, cmap="bwr")
            for [ax0, ax1, ax2], j in zip(zk_axs.T, range(4, 28+1)):
                # Use this vmax initially, if there's data, we'll reset it.
                vmax = np.nanquantile(np.abs(model_fam_zks[:, j]), 0.95)*1.2
                sc1 = ax1.scatter(
                    np.rad2deg(aos_fam_avg["thx_OCS"]),
                    np.rad2deg(aos_fam_avg["thy_OCS"]),
                    c=model_fam_zks[:, j],
                    **zk_kwargs
                )
                ax0.text(
                    0.74, 0.89, f"{vmax:.2f}", font="monospace", fontsize=12, color="black",
                    transform=ax0.transAxes,
                )
                try:
                    i = np.where(aos_fam_avg.meta["nollIndices"] == j)[0][0]
                except IndexError:
                    sc1.set_clim(-vmax, vmax)
                    continue
                all_zks = np.concatenate(
                    [
                        model_fam_zks[:, j],
                        aos_fam_avg["zk_OCS"][:, i]
                    ]
                )
                vmax = np.nanquantile(np.abs(all_zks), 0.95)*1.2
                sc0 = ax0.scatter(
                    np.rad2deg(aos_fam_avg["thx_OCS"]),
                    np.rad2deg(aos_fam_avg["thy_OCS"]),
                    c=aos_fam_avg["zk_OCS"][:, i],
                    **zk_kwargs
                )
                sc2 = ax2.scatter(
                    np.rad2deg(aos_fam_avg["thx_OCS"]),
                    np.rad2deg(aos_fam_avg["thy_OCS"]),
                    c=model_fam_zks[:, j] - aos_fam_avg["zk_OCS"][:, i],
                    **zk_kwargs
                )
                for sc in [sc0, sc1, sc2]:
                    sc.set_clim(-vmax, vmax)

        # Corner Zernikes panels
        model_corner_zks = ssh.wf_model(
            np.rad2deg(aos_corner_avg["thx_OCS"]),
            np.rad2deg(aos_corner_avg["thy_OCS"]),
            result["state"],
        )
        for row, model_zk in zip(aos_corner_avg, model_corner_zks):
            i = int(row["detector"][1])//4
            j = int(row["detector"][2])//4
            ax = corner_axs[i, j]
            w = 0.3
            noll = aos_corner_avg.meta["nollIndices"]
            ax.bar(noll-w/2, row["zk_OCS"], width=w, color="r", label="data")
            ax.bar(np.arange(4, 29)+w/2, model_zk[4:], width=w, color="b", label="model")
            ax.legend()

        # Text output panel
        with AxisText(text_ax, ncols=3) as at:
            fit_str = "fit:"
            if args.fit_moments:
                fit_str += " moments"
            if args.fit_fam:
                fit_str += " fam"
            if args.fit_corners:
                fit_str += " corners"
            at.write(fit_str)
            at.write(f"opt FWHM {np.nanmean(opt_moments["FWHM"]):8.3f} arcsec")
            at.write()
            at.write(f"ext Ixx  {Ixx:8.3f} arcsec²")
            at.write(f"ext Iyy  {Iyy:8.3f} arcsec²")
            at.write(f"ext Ixy  {Ixy:8.3f} arcsec²")
            at.write(f"ext FWHM {np.sqrt(0.5*(Ixx+Iyy))*SIGMA_TO_FWHM:8.3f} arcsec")
            at.write(f"ext e1   {(Ixx-Iyy)/(Ixx+Iyy):8.3f}")
            at.write(f"ext e2   {2*Ixy/(Ixx+Iyy):8.3f}")
            at.write()
            for i, dof in enumerate(result["state"]):
                name, unit = ALL_DOFS[use_dof[i]]
                at.write(f"{name:8} {dof:8.3f} {unit}")

        # vmode panel
        ssh.plot_modes(*char_axs[:,0], cmap="bwr")

        # vmode dz panel
        ssh.plot_sens_dz(dz_ax, cmap="bwr")

        fig.savefig(args.plot)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Input ASDF file")
    parser.add_argument("--plot", type=str, default=None, help="Filename for plot output")
    parser.add_argument("--fit_moments", action="store_true", help="Fit moments")
    parser.add_argument("--fit_fam", action="store_true", help="Fit full array mode wavefront")
    parser.add_argument("--fit_corners", action="store_true", help="Fit corner wavefront")
    parser.add_argument("--use_dof", type=str, default="0-9", help="Degrees of freedom to use")
    parser.add_argument("--nkeep", type=int, default=6, help="Number of modes to keep in the fit.  Negative for no orthogonalization.")
    parser.add_argument("--transverse_pupil_radii", type=int, default=10, help="Number of transverse pupil radii")
    parser.add_argument("--transverse_field_radii", type=int, default=14, help="Number of transverse field radii")
    parser.add_argument("--wf_kmax", type=int, default=15, help="Maximum wavefront field Noll index")
    parser.add_argument("--wf_jmax", type=int, default=28, help="Maximum wavefront pupil Noll index")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plot_triplet", action="store_true", help="Plot as triplet figure layout")
    group.add_argument("--plot_singlet", action="store_true", help="Plot as singlet figure layout")
    args = parser.parse_args()
    main(args)
