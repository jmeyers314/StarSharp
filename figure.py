from itertools import chain

import numpy as np
from matplotlib.figure import Figure


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


class AxisText:
    def __init__(
        self,
        ax,
        x=0.01,
        y=0.99,
        width=None,
        height=None,
        max_lines=20,
        ncols=1
    ):
        """
        ax: matplotlib axis
        x, y: starting position in axis coordinates
        max_lines: maximum number of lines per column
        ncols: number of columns
        """
        self.ax = ax
        self.x = x
        self.y = y
        self.max_lines = max_lines
        self.ncols = ncols
        self.buf = []
        if width is None:
            width = 1.0 - x
        if height is None:
            height = y
        self.col_width = width / ncols
        self.line_height = height / max_lines

    def write(self, *objects, sep=" ", end="\n"):
        self.buf.append(sep.join(map(str, objects)) + end)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        text_str = ''.join(self.buf)
        lines = text_str.splitlines()

        for col in range(self.ncols):
            col_lines = lines[col*self.max_lines:(col+1)*self.max_lines]
            if not col_lines:
                continue
            for i, line in enumerate(col_lines):
                self.ax.text(
                    self.x + col*self.col_width,
                    self.y - i*self.line_height,
                    line,
                    transform=self.ax.transAxes,
                    ha='left', va='top',
                    family='monospace'
                )

def layout_triplet_figure():

    fig = Figure(figsize=(28, 15.75))

    corner_axs = fig.subplots(
        nrows=2, ncols=2, squeeze=False,
        sharex=True, sharey=True,
        gridspec_kw={
            "left": 0.03,
            "right": 0.40,
            "top": 0.95,
            "bottom": 0.75,
            "wspace": 0.03,
            "hspace": 0.1
        }
    )
    corner_axs[0,0].set_ylabel("Aberration [μm]")
    corner_axs[1,0].set_ylabel("Aberration [μm]")
    corner_axs[1,0].set_xlabel("Noll index")
    corner_axs[1,1].set_xlabel("Noll index")
    corner_axs[0,0].text(4, -1.7, "R00", transform=corner_axs[0,0].transData)
    corner_axs[0,1].text(4, -1.7, "R40", transform=corner_axs[0,1].transData)
    corner_axs[1,0].text(4, -1.7, "R04", transform=corner_axs[1,0].transData)
    corner_axs[1,1].text(4, -1.7, "R44", transform=corner_axs[1,1].transData)
    noll = np.arange(4, 28+1)
    for ax in corner_axs.flat:
        for x in np.arange(noll[0]-0.5, noll[-1]+1, 1):
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlim(3.5, 28.5)
        ax.set_ylim(-2.0, 2.0)

    char_axs = fig.subplots(
        nrows=2, ncols=1, squeeze=False,
        gridspec_kw={
            "left": 0.415,
            "right": 0.515,
            "top": 0.95,
            "bottom": 0.75,
            "wspace": 0.0,
            "hspace": 0.0,
            "height_ratios": [0.8, 0.2]
        }
    )
    for ax in char_axs.flat:
        ax.set_yticks([])
    char_axs[0, 0].set_xticks([])
    char_axs[1, 0].set_xlabel("vmode idx")

    dz_ax = fig.subplots(
        nrows=1, ncols=1, squeeze=False,
        gridspec_kw={
            "left": 0.545,
            "right": 0.645,
            "top": 0.95,
            "bottom": 0.75,
            "wspace": 0.0,
            "hspace": 0.0
        }
    )[0, 0]
    dz_ax.set_xlabel("vmode idx")

    text_ax = fig.subplots(
        nrows=1, ncols=1, squeeze=False,
        gridspec_kw={
            "left": 0.66,
            "right": 0.97,
            "top": 0.95,
            "bottom": 0.75,
            "wspace": 0.0,
            "hspace": 0.0
        }
    )[0, 0]
    text_ax.set_xticks([])
    text_ax.set_yticks([])

    # Z4 to Z15 (12 columns)
    lo_zk_axs = fig.subplots(
        nrows=3, ncols=12, squeeze=False,
        sharex=True, sharey=True,
        gridspec_kw={
            "left": 0.018,
            "right": 0.692,
            "top": 0.68,
            "bottom": 0.38,
            "wspace": 0.0,
            "hspace": 0.0
        }
    )
    lo_zk_axs[0, 0].set_ylabel("data")
    lo_zk_axs[1, 0].set_ylabel("model")
    lo_zk_axs[2, 0].set_ylabel("resid")

    # Z16 to Z28 (13 columns)
    hi_zk_axs = fig.subplots(
        nrows=3, ncols=13, squeeze=False,
        sharex=True, sharey=True,
        gridspec_kw={
            "left": 0.018,
            "right": 0.748,
            "top": 0.34,
            "bottom": 0.04,
            "wspace": 0.0,
            "hspace": 0.0
        }
    )
    hi_zk_axs[0, 0].set_ylabel("data")
    hi_zk_axs[1, 0].set_ylabel("model")
    hi_zk_axs[2, 0].set_ylabel("resid")

    zk_axs = np.hstack([lo_zk_axs, hi_zk_axs])
    for ax in zk_axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
    for ax, j in zip(zk_axs[0], range(4, 29)):
        ax.set_title(f"Z{j}")

    shape_axs = fig.subplots(
        nrows=3, ncols=4, squeeze=False,
        sharex=True, sharey=True,
        gridspec_kw={
            "left": 0.699,
            "right": 0.925,
            "top": 0.68,
            "bottom": 0.38,
            "wspace": 0.0,
            "hspace": 0.0,
        }
    )
    for ax in shape_axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
    shape_axs[0, 0].set_title("Quiver")
    shape_axs[0, 1].set_title("FWHM")
    shape_axs[0, 2].set_title("e1")
    shape_axs[0, 3].set_title("e2")
    fwhm_cax = fig.add_axes([0.935, 0.38, 0.005, 0.3])
    ellip_cax = fig.add_axes([0.965, 0.38, 0.005, 0.3])

    moments_axs = fig.subplots(
        nrows=3, ncols=3, squeeze=False,
        sharex=True, sharey=True,
        gridspec_kw={
            "left": 0.755,
            "right": 0.925,
            "top": 0.34,
            "bottom": 0.04,
            "wspace": 0.0,
            "hspace": 0.0,
        }
    )
    for ax in moments_axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
    moments_axs[0, 0].set_title("Ixx + Iyy")
    moments_axs[0, 1].set_title("Ixx - Iyy")
    moments_axs[0, 2].set_title("2 Ixy")
    Tsqr_cax = fig.add_axes([0.935, 0.04, 0.005, 0.3])
    w_cax = fig.add_axes([0.965, 0.04, 0.005, 0.3])

    for ax in chain(moments_axs.flat, shape_axs.flat, zk_axs.flat):
        th = np.linspace(0, 2 * np.pi, 100)
        x = 1.75 * np.cos(th)
        y = 1.75 * np.sin(th)
        ax.plot(x, y, color="black", lw=0.25)

    return dict(
        fig=fig,

        corner_axs=corner_axs,
        char_axs=char_axs,
        dz_ax=dz_ax,
        text_ax=text_ax,

        zk_axs=zk_axs,
        shape_axs=shape_axs,
        fwhm_cax=fwhm_cax,
        ellip_cax=ellip_cax,

        moments_axs=moments_axs,
        Tsqr_cax=Tsqr_cax,
        w_cax=w_cax,
    )


def layout_singlet_figure():

    fig = Figure(figsize=(28, 15.75))

    corner_axs = fig.subplots(
        nrows=2, ncols=2, squeeze=False,
        sharex=True, sharey=True,
        gridspec_kw={
            "left": 0.03,
            "right": 0.40,
            "top": 0.95,
            "bottom": 0.75,
            "wspace": 0.03,
            "hspace": 0.1
        }
    )
    corner_axs[0,0].set_ylabel("Aberration [μm]")
    corner_axs[1,0].set_ylabel("Aberration [μm]")
    corner_axs[1,0].set_xlabel("Noll index")
    corner_axs[1,1].set_xlabel("Noll index")
    corner_axs[0,0].text(4, -1.7, "R00", transform=corner_axs[0,0].transData)
    corner_axs[0,1].text(4, -1.7, "R40", transform=corner_axs[0,1].transData)
    corner_axs[1,0].text(4, -1.7, "R04", transform=corner_axs[1,0].transData)
    corner_axs[1,1].text(4, -1.7, "R44", transform=corner_axs[1,1].transData)
    noll = np.arange(4, 28+1)
    for ax in corner_axs.flat:
        for x in np.arange(noll[0]-0.5, noll[-1]+1, 1):
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlim(3.5, 28.5)
        ax.set_ylim(-2.0, 2.0)

    char_axs = fig.subplots(
        nrows=2, ncols=1, squeeze=False,
        gridspec_kw={
            "left": 0.415,
            "right": 0.515,
            "top": 0.95,
            "bottom": 0.75,
            "wspace": 0.0,
            "hspace": 0.0,
            "height_ratios": [0.8, 0.2]
        }
    )
    for ax in char_axs.flat:
        ax.set_yticks([])
    char_axs[0, 0].set_xticks([])
    char_axs[1, 0].set_xlabel("vmode idx")

    dz_ax = fig.subplots(
        nrows=1, ncols=1, squeeze=False,
        gridspec_kw={
            "left": 0.545,
            "right": 0.645,
            "top": 0.95,
            "bottom": 0.75,
            "wspace": 0.0,
            "hspace": 0.0
        }
    )[0, 0]
    dz_ax.set_xlabel("vmode idx")

    text_ax = fig.subplots(
        nrows=1, ncols=1, squeeze=False,
        gridspec_kw={
            "left": 0.66,
            "right": 0.97,
            "top": 0.95,
            "bottom": 0.75,
            "wspace": 0.0,
            "hspace": 0.0
        }
    )[0, 0]
    text_ax.set_xticks([])
    text_ax.set_yticks([])

    shape_axs = fig.subplots(
        nrows=3, ncols=4, squeeze=False,
        sharex=True, sharey=True,
        gridspec_kw={
            "left": 0.03,
            "right": 0.48,
            "top": 0.66,
            "bottom": 0.06,
            "wspace": 0.0,
            "hspace": 0.0,
        }
    )
    for ax in shape_axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
    shape_axs[0, 0].set_title("Quiver")
    shape_axs[0, 1].set_title("FWHM")
    shape_axs[0, 2].set_title("e1")
    shape_axs[0, 3].set_title("e2")
    shape_axs[0, 0].set_ylabel("Data")
    shape_axs[1, 0].set_ylabel("Model")
    shape_axs[2, 0].set_ylabel("Resid")
    fwhm_cax = fig.add_axes([0.49, 0.06, 0.01, 0.6])
    ellip_cax = fig.add_axes([0.52, 0.06, 0.01, 0.6])

    moments_axs = fig.subplots(
        nrows=3, ncols=3, squeeze=False,
        sharex=True, sharey=True,
        gridspec_kw={
            "left": 0.56,
            "right": 0.9,
            "top": 0.66,
            "bottom": 0.06,
            "wspace": 0.0,
            "hspace": 0.0,
        }
    )
    for ax in moments_axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
    moments_axs[0, 0].set_title("Ixx + Iyy")
    moments_axs[0, 1].set_title("Ixx - Iyy")
    moments_axs[0, 2].set_title("2 Ixy")
    moments_axs[0, 0].set_ylabel("Data")
    moments_axs[1, 0].set_ylabel("Model")
    moments_axs[2, 0].set_ylabel("Resid")
    Tsqr_cax = fig.add_axes([0.915, 0.06, 0.01, 0.6])
    w_cax = fig.add_axes([0.95, 0.06, 0.01, 0.6])

    for ax in chain(moments_axs.flat, shape_axs.flat):
        th = np.linspace(0, 2 * np.pi, 100)
        x = 1.75 * np.cos(th)
        y = 1.75 * np.sin(th)
        ax.plot(x, y, color="black", lw=0.25)

    return dict(
        fig=fig,

        corner_axs=corner_axs,
        char_axs=char_axs,
        dz_ax=dz_ax,
        text_ax=text_ax,

        shape_axs=shape_axs,
        fwhm_cax=fwhm_cax,
        ellip_cax=ellip_cax,

        moments_axs=moments_axs,
        Tsqr_cax=Tsqr_cax,
        w_cax=w_cax,
    )