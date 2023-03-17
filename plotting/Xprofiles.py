import os
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(os.path.join(os.environ["BASTADIR"], "basta/plots.mplstyle"))

def main():

    xlims  = [[0,0.3], [0, 0.3]]
    ylims  = [[0.35, 0.75], [4.1e7, 4.9e7]]
    ylabs  = [r"X", r"$c_s$ (cm/s)"]
    ylpad  = [50, 4]
    widths = [0.7, 1.]
    tracks = ["0488", "0568"]
    nfit   = ["#1", "#5"]
    agefit = [11.890, 13.562]
    cols   = ["#44AA99", "#882255"]
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
    

def read_fgong(filename):
    """
    Read in an FGONG file.
    This function can read in FONG versions 250 and 300.
    The output gives the center first, surface last.
    Parameters
    ----------
    filename : str
        Absolute path an name of the FGONG file to read
    Returns
    -------
    starg : dict
        Global parameters of the model
    starl : dict
        Local parameters of the model
    """
    # Standard definitions and units for FGONG
    glob_pars = [
        ("mass", "g"),
        ("Rphot", "cm"),
        ("Lphot", "erg/s"),
        ("Zini", None),
        ("Xini", None),
        ("alphaMLT", None),
        ("phi", None),
        ("xi", None),
        ("beta", None),
        ("dp", "s"),
        ("ddP_drr_c", None),
        ("ddrho_drr_c", None),
        ("Age", "yr"),
        ("teff", "K"),
        ("Gconst", "cm3/gs2"),
    ]
    loc_pars = [
        ("radius", "cm"),  # Var 1
        ("ln(m/M)", None),
        ("Temp", "K"),
        ("P", "kg/m/s2"),
        ("Rho", "g/cm3"),
        ("X", None),
        ("Lumi", "erg/s"),
        ("opacity", "cm2/g"),
        ("eps_nuc", None),
        ("gamma1", None),  # Var 10
        ("grada", None),
        ("delta", None),
        ("cp", None),
        ("free_e", None),
        ("brunt_A", None),  # Var 15 (!!)
        ("rx", None),
        ("Z", None),
        ("R-r", "cm"),
        ("eps_logg", None),
        ("Lg", "erg/s"),  # Var 20
        ("xhe3", None),
        ("xc12", None),
        ("xc13", None),
        ("xn14", None),
        ("xo16", None),
        ("dG1_drho", None),
        ("dG1_dp", None),
        ("dG1_dY", None),
        ("xh2", None),
        ("xhe4", None),  # Var 30
        ("xli7", None),
        ("xbe7", None),
        ("xn15", None),
        ("xo17", None),
        ("xo18", None),
        ("xne20", None),  # Var 36 (end of format spec.)
        ("xmg24", None),  # Extension. Added in Garstec
        ("xsi28", None),  # ibid.
        ("NA1", None),  # NOT IN USE
        ("NA2", None),
    ]  # NOT IN USE  -- Var 40

    # Start reading the file
    with open(filename, "r") as ff:
        lines = ff.readlines()

    # Read file definitions from the fifth line (first four is comments)
    NN, ICONST, IVAR, IVERS = [int(i) for i in lines[4].strip().split()]
    if not ICONST == 15:
        raise ValueError("cannot interpret FGONG file: wrong ICONST")

    # Data storage
    data = []
    starg = {}

    # Read the file from the fifth line onwards
    # Change in the format for storing the numbers (February 2017):
    #  - If IVERS <= 1000, 1p5e16.9
    #  - If IVERS  > 1000, 1p,5(x,e26.18e3)
    if IVERS <= 1000:
        for line in lines[5:]:
            data.append(
                [
                    line[0 * 16 : 1 * 16],
                    line[1 * 16 : 2 * 16],
                    line[2 * 16 : 3 * 16],
                    line[3 * 16 : 4 * 16],
                    line[4 * 16 : 5 * 16],
                ]
            )
    else:
        for line in lines[5:]:
            data.append(
                [
                    line[0 * 27 : 1 * 27],
                    line[1 * 27 : 2 * 27],
                    line[2 * 27 : 3 * 27],
                    line[3 * 27 : 4 * 27],
                    line[4 * 27 : 5 * 27],
                ]
            )

    # Put the data into arrays
    data = np.ravel(np.array(data, float))
    for i in range(ICONST):
        starg[glob_pars[i][0]] = data[i]
    data = data[15:].reshape((NN, IVAR)).T

    # Reverse the profile to get center ---> surface
    data = data[:, ::-1]

    # Make it into a record array and return the data
    starl = np.rec.fromarrays(data, names=[lp[0] for lp in loc_pars])

    # Exclude the center r = 0. mesh (MESA includes it)
    if starl["radius"][0] < 1.0e-14:
        starl = starl[1:]

    return starg, starl


if __name__ == "__main__":
    main()
