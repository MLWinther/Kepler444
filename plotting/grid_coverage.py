import h5py
import numpy as np
import matplotlib.pyplot as plt

import basta.constants as bc

#####################################################################
# Corner plot of basis parameters for grid, for appendix C of paper #
# out: grid_coverage.pdf                                            #
#####################################################################

def base_corner(baseparams, base, outbasename):
    _, parlab, _, _ = bc.parameters.get_keys([par for par in baseparams])
    
    # Size of figure, stolen from basta/corner.py
    K = len(baseparams) - 1
    factor = 2.0 if K > 1 else 3.0
    whspace = 0.05
    lbdim = 0.5 * factor
    trdim = 0.2 * factor
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    # Format figure
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )
    # Some index magic in the following, as we are not intereset in the 'diagonal'
    for j in range(K):
        for i in range(j, K + 1):
            if j == i:
                continue
            elif K == 1:
                ax = axes
            else:
                ax = axes[i - 1, j]
            # Old subgrid
            ax.plot(
                base[:, j],
                base[:, i],
                "X",
                color="k",
                markersize=1,
                zorder=10,
                rasterized=True,
            )
            if i == K:
                # Set xlabel and rotate ticklabels for no overlap
                ax.set_xlabel(parlab[j], fontsize=16)
                ax.tick_params(axis="x", labelsize=14)
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if not K == 1:
                    ax.xaxis.set_label_coords(0.5, -0.35)
            else:
                # Remove xticks from non-bottom subplots
                ax.set_xticks([])
            if j == 0:
                # Set ylabel
                ax.set_ylabel(parlab[i], fontsize=16)
                ax.tick_params(axis="y", labelsize=14)
                if not K == 1:
                    ax.yaxis.set_label_coords(-0.35, 0.5)
            else:
                # Remove yticks from non-leftmost subplots
                ax.set_yticks([])

        for i in range(K):
            # Remove the empty subplots
            if i < j:
                ax = axes[i, j]
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])

    # Easiest pretty layout for single-subplot
    if K == 1:
        fig.tight_layout()
    # Save and close
    fig.savefig(outbasename)
    plt.close()

def main():
    gridname = "/home/au540782/Kepler444/input/Garstec_AS09_Kepler444_diff.hdf5"
    grid = h5py.File(gridname, "r")
    baseparams = ["massini", "FeHini", "yini", "alphaMLT", "ove", "gcut"]
    N = len(grid["header/massini"])
    base = np.zeros((N, len(baseparams)))
    for i, par in enumerate(baseparams):
        base[:, i] = grid["header"][par][:]
    
    outname = "plots/grid_coverage.pdf"

    base_corner(baseparams, base, outname)

if __name__ == "__main__":
    main()