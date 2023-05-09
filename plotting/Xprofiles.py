import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from garstec_helpers import read_fgong

plt.style.use(os.path.join(os.environ["BASTADIR"], "basta/plots.mplstyle"))

#######################################################################################
# Read an plot specified profiles of sound speed and hydrogen content, paper figure 7 #
# out: Xprofiles.pdf                                                                  #
#######################################################################################


def main():

    xlims  = [[0,0.3], [0, 0.3]]
    ylims  = [[0.35, 0.75], [4.1e7, 4.9e7]]
    ylabs  = [r"X", r"$c_s$ (cm/s)"]
    ylpad  = [50, 4]
    widths = [0.7, 1.]
    # tracks = ["0488", "0568"]
    # nfit   = ["#1", "#5"]
    # agefit = [11.890, 13.562]
    tracks = ["1500", "2116"]
    nfit   = ["#2", "#4"]
    agefit = [11.148, 12.823]
    cols   = ["#88CCEE", "#117733"]
    titles = [r"$\mathrm{Age}\approx 8 \,\mathrm{Gyr}$", r"Age of HLM"]
    lstyle = ['-', '--']

    path = "gongfiles_Xprofiles/track{}-*.fgong"

    fig, ax = plt.subplots(2,2, figsize=(8.47,6), sharey="row", sharex=True)

    for i, track in enumerate(tracks):
        gongs = np.sort(glob.glob(path.format(track)))
        
        for j, gong in enumerate(gongs):
            starg, starl = read_fgong(gong)

            cs = np.sqrt(starl["gamma1"] * starl["P"] / starl["Rho"])
            xx = starl["X"]
            for k, yval in enumerate([xx, cs]):
                if k == 0 and j == 1:
                    lab = "{:.1f}".format(agefit[i]) + "$\,\mathrm{Gyr}$"
                else:
                    lab = nfit[i]
                ax[k,j].plot(np.exp(starl["ln(m/M)"]), yval, 
                             lstyle[i], color=cols[i], label=lab, lw=2)

    for i, title in enumerate(titles):
        w = widths[i]
        ax[0,i].legend(title=title, bbox_to_anchor=((1-w)/2., 1.02, w, 0.102),
                       loc = 8, ncol=2, mode="expand", borderaxespad=0, 
                       frameon=False, title_fontsize=18, fontsize=16,
                       handletextpad=0.2)
        ax[-1,i].set_xlabel(r'm/M')
        ax[i,0].set_xlim(xlims[i])
        ax[i,0].set_ylim(ylims[i])
        ax[i,0].set_ylabel(ylabs[i], labelpad=ylpad[i])
        if i==1:
            ax[i,0].set_yticks([4.2e7, 4.4e7, 4.6e7, 4.8e7])
            ax[i,0].set_yticklabels([r"$4.2\times 10^7$", r"$4.4\times 10^7$", 
                                     r"$4.6\times 10^7$", r"$4.8\times 10^7$"])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig("plots/Xprofiles.pdf", dpi=300)
    plt.close(fig)
    

if __name__ == "__main__":
    main()
