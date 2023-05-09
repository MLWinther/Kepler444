from __future__ import print_function, with_statement, division
import os
import glob
import json
import h5py

import numpy as np
import matplotlib.pyplot as plt

import basta.constants as bc
import basta.fileio as fio
import basta.freq_fit as freq_fit
import basta.utils_seismic as su
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

import helpers

plt.style.use(os.path.join(os.environ['BASTADIR'], 'basta/plots.mplstyle'))
rcset = {'xtick.direction': u'in', 'ytick.direction': u'in', 'xtick.top':True, 'ytick.right':True, 'font.family': "sans-serif", "font.size":18}
plt.rcParams.update(rcset)

########################################################################
# Extracts statistics from fits, and produces plots of best-fit-tracks #
# out: *fitnumber*/{agecore.pdf,ratios.pdf}                            #
########################################################################

top = '/home/au540782/Kepler444/'

def main(Nbest=0, 
         only=[], 
         sets=[], 
         bayfits=[],
):

    if len(only) or len(sets):
        only = list(np.unique(np.append(np.array(only), np.array(sets).flatten()).flatten()))
    
    # Compare BASTA and Buldgen fits
    top = '/home/au540782/Kepler444/'
    

    # Define grids and determine core types of tracks
    grids = [h5py.File(os.path.join(top, "input/Garstec_AS09_Kepler444_nodiff.hdf5"), 'r'),
             h5py.File(os.path.join(top, "input/Garstec_AS09_Kepler444_diff.hdf5"), 'r')]
    
    core_types = [helpers.determine_core_type(grids[0], interval=[1,9]),
                  helpers.determine_core_type(grids[1], interval=[1,9])]
    bins = [0.5, 1.5, 2.5, 3.5]
    hist1, b = np.histogram(list(core_types[0].values()), bins=bins)
    hist2, b = np.histogram(list(core_types[1].values()), bins=bins)
    print("""
            |    1 |    2 |    3 
    No diff | {0} | {1} | {2}
    Diff    | {3} | {4} | {5}
    """.format(str(hist1[0]).rjust(4),
               str(hist1[1]).rjust(4),
               str(hist1[2]).rjust(4),
               str(hist2[0]).rjust(4),
               str(hist2[1]).rjust(4),
               str(hist2[2]).rjust(4)))

    #############################
    # Prepare figure for sets
    #############################
    if len(sets):
        figs, axs = plt.subplots(max([len(l) for l in sets]), len(sets), figsize=(16,12))
        parameters = ["agecore", "Teff", "rho", "massfin", "FeH", "age"]
        _, paramlabels, _, _ = bc.parameters.get_keys(parameters)
        xlabel = paramlabels[0]
    
    #############################
    # Loop over all combinations
    #############################
    path = "all_combinations"
    outdirs = glob.glob(os.path.join(top, path, "output*"))
    outdirs = np.sort(outdirs)
    
    for outdir in outdirs:
        # Extract number and keys for fit from directory name
        n = int(outdir.split("/")[-1].split("_")[1])
        if n not in only and len(only):
            continue
        keys = outdir.split("/")[-1].split("_")[2:]
        if "r01" in keys and "nottrusted" in keys:
            continue
        if "parallax" in keys and False:
            continue
        
        if not os.path.isdir(os.path.join("plots", "{:03d}".format(n))):
            os.mkdir(os.path.join("plots", "{:03d}".format(n)))
        # Diffusion and thereby grid to work with
        diff = True if "diffusion" in keys else False
        if diff:
            grid = grids[1]
        else:
            grid = grids[0]

        # Get fit data
        bastajson = os.path.join(outdir, "Kepler444.json")
        selected = fio.load_selectedmodels(bastajson)
        trackmodmax = helpers.get_nmax(selected, Nbest, pdf=True)
        
        plot_ratios(grid, trackmodmax, n, top)

        print(35*"#")
        print(" Fit nr {1}: Stats of {0} best".format(len(trackmodmax), n))
        print(35*"#")
        print("  n |  mass |  [Fe/H] |   rho |    age |   Gcut |    ove |   Yini |   aMLT |   a/Fe |   agecor | track")
        wstr = " {0} | {1} |  {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} "
        for i, (track, mod, _) in enumerate(trackmodmax):
            mass = grid[track]["massfin"][mod]
            FeH  = grid[track]["FeH"][mod]
            rho  = grid[track]["rho"][mod]
            age  = grid[track]["age"][mod]*1e-3
            gcut = grid[track]["gcut"][mod]
            fove = grid[track]["ove"][mod]
            yini = grid[track]["yini"][mod]
            amlt = grid[track]["alphaMLT"][mod]
            aFe  = grid[track]["alphaFe"][mod]
            acor = grid[track]["agecore"][mod]
            fmod = grid[track]["name"][mod]
            mass = "{:.3f}".format(mass).rjust(5)
            FeH  = "{:.3f}".format( FeH).rjust(6)
            rho  = "{:.3f}".format( rho).rjust(5)
            age  = "{:.3f}".format( age).rjust(6)
            gcut = "{:.3f}".format(gcut).rjust(6)
            fove = "{:.3f}".format(fove).rjust(6)
            yini = "{:.3f}".format(yini).rjust(6)
            amlt = "{:.3f}".format(amlt).rjust(6)
            aFe  = "{:.3f}".format(aFe).rjust(6)
            acor = "{:8.3f}".format(acor).rjust(8)
            print(wstr.format(str(i+1).rjust(2), mass, FeH, rho, age, gcut, fove, yini, amlt, aFe, acor, track.split("/")[-1], fmod))

        plot_agecore_posterior(grid, trackmodmax, n, bayfits[n])

    if len(sets):
        figs.tight_layout()
        figs.savefig("plots/agecore_sets.pdf")
        plt.close("all")

def plot_ratios(Grid, trackmodmax, nfit, top):
    N = len(trackmodmax)
    freqfile = os.path.join(top, 'input/Kepler444.xml')
    obskey, obs, _ = fio.read_freq(freqfile)
    obsr012, covr012 = freq_fit.compute_ratios(obskey, obs, "r012")
    errr012 = np.sqrt(np.diag(covr012))
    mask = (obsr012[2, :] == float(2)) & ((obsr012[3, :] == 25) | (obsr012[3, :] == 26))

    fig, ax = plt.subplots(N,1,figsize=(8.47,N*4))
    for i in range(N):
        ax[i].errorbar(obsr012[1, ~mask], obsr012[0, ~mask], yerr=errr012[~mask], 
                       fmt='d', color='k',label=r'Observed')
        ax[i].errorbar(obsr012[1, mask], obsr012[0, mask], yerr=errr012[mask], 
                       fmt='d', color='grey',label=r'Removed')
    
        track, index, _ = trackmodmax[i]
        osc = su.transform_obj_array(Grid[track + '/osc'][index])
        osckey = su.transform_obj_array(Grid[track + '/osckey'][index])
        dnufit = float(Grid[track + '/dnufit'][index])

        joinkeys, join = freq_fit.calc_join(osc, osckey, obs, obskey)

        bestr012 = freq_fit.compute_ratioseqs(joinkeys, join, "r012")

        ax[i].plot(bestr012[1, :], bestr012[0, :], "^", color="#DDCC77", label=r"Best fit")

        ax[i].set_ylabel(r"$r_{012}$")
        ax[i].set_xlim([3400, 5400])

        title  = "Track {:04d}".format(int(track.split("/")[-1][-4:]))
        title += "\n" + r"$\Delta\nu = $" + "{:.2f}".format(dnufit)
        if i == 0:
            ax[i].legend(title=title)
        else:
            ax[i].legend([], [], title=title)

    for a in ax[:-1]:
        a.set_xticklabels([])
    ax[-1].set_xlabel(r'$\nu\,(\mu \mathrm{Hz})$')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig("plots/{0:03d}/ratios.pdf".format(nfit), dpi=300)
    plt.close(fig)



def plot_agecore_posterior(grid, trackmodmax, n, bayfit):
    cols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#882255', 
            '#DDCC77', '#CC6677', '#999933', '#AA4499']
    # Buldgen profile for overlay
    Buldgen = np.loadtxt(os.path.join(top, 'input/profile_Buldgen.txt'), comments='#', delimiter=",")
    

    fig, ax = plt.subplots(1,1, figsize=(8.47,4))
    ax.plot(Buldgen[:,0], Buldgen[:,1], '--', alpha=1, 
                 color=cols[-2], label=r'$\mathcal{M}_{13}$',zorder=2, lw=2)

    # Plot each individual track in axes corresponding to core type
    for i, (track, mod, _) in enumerate(trackmodmax):
        if i >= len(cols):
            col = 'darkgrey'
            lab = '__nolabel__'
        else:
            col = cols[i]
            lab = "#" + str(i+1)
        age = np.asarray(grid[track]["age"])*1e-3
        mcore = np.asarray(grid[track]["Mcore"])
        
        ax.plot(age, mcore, color=col, label=lab, zorder=3, lw=2)
        ax.plot(age[mod], mcore[mod], '.', color=col, zorder=5, markersize=13)
    ax.set_ylim([-0.005,0.105])
    ax.set_xlim([-0.1, 15])
    
    ax.plot(np.ones(2)*bayfit[0], [-2,1.5], '-k' , zorder=10)
    ax.plot(np.ones(2)*bayfit[1], [-2,1.5], '--k', zorder=10)
    ax.plot(np.ones(2)*bayfit[2], [-2,1.5], '--k', zorder=10)

    ax.set_xlabel(r'Age (Gyr)')
    ax.set_ylabel(r'$M_{\mathrm{core}}$ (m/M)')
    
    # Beautyfication
    ax.legend(fontsize=16,title_fontsize=16)
    fig.tight_layout()
    fig.savefig("plots/{:03d}/agecore.pdf".format(n), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    Nbest = 5
    only = [16, 22]
    sets = []
    bayfits = {16: [0.625, 0.625-0.248, 0.625+2.123],
               22: [0.592, 0.592-0.244, 0.592+2.150]}
    main(Nbest, only, sets, bayfits)
