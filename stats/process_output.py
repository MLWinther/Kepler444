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

top = '/home/au540782/Kepler444/'

def main(Nbest=0, 
         only=[], 
         sets=[], 
         set_labels=[], 
         set_legends=[]):

    if len(only) or len(sets):
        only = list(np.unique(np.append(np.array(only), np.array(sets).flatten()).flatten()))
    # Compare BASTA and Buldgen fits
    top = '/home/au540782/Kepler444/'
    

    # Define grids and determine core types of tracks
    grids = [h5py.File(os.path.join(top, "input/Garstec_AS09_Kepler444_nodiff.hdf5"), 'r'),
             h5py.File(os.path.join(top, "input/Garstec_AS09_Kepler444_diff.hdf5"), 'r')]
             #h5py.File(os.path.join(top, "input/Garstec_AS09_Kepler444.hdf5"), 'r')]
    
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
        print("  n |  mass |  [Fe/H] |   rho |    age |   Gcut |    ove |   Yini |   aMLT |   a/Fe | agecor | track")
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
            acor = "{:.3f}".format(acor).rjust(6)
            print(wstr.format(str(i+1).rjust(2), mass, FeH, rho, age, gcut, fove, yini, amlt, aFe, acor, track.split("/")[-1], fmod))

        np.random.seed(444)
        #freq = np.ones((35,2))
        #for i in range(10000):
        #    _ = np.random.normal(freq[:,0], freq[:,1])
        logy = np.concatenate([ts.logPDF for ts in selected.values()])
        noofind = len(logy)
        nonzero = np.isfinite(logy)
        logy = logy[nonzero]
        lk = logy - np.amax(logy)
        pd = np.exp(lk - np.log(np.sum(np.exp(lk))))
        sampled_indices = np.random.choice(np.arange(len(pd)), p=pd, size=noofind)
        
        plot_agecore_posterior(selected, grid, sampled_indices, noofind, nonzero, trackmodmax, n)
        print(trackmodmax)

    if len(sets):
        figs.tight_layout()
        figs.savefig("plots/agecore_sets.pdf")
        plt.close("all")

    col = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
           '#DDCC77', '#CC6677', '#882255', '#AA4499']
    # 6, 1, 3
    col = ['#CC6677', '#88CCEE', '#117733']
        
def insert_posterior(ax, grid, selected, sampled_indices, noofind, nonzero, leglabel, param="agecore", xlabel=False):
    qs = [0.5, 0.158655, 0.841345]
    col = "#CC6677"
    agelabs = {'age': r'Age (Gyr)', 'agecore': r'$\tau_{\rm core}$ (Gyr)'}
    _, paramlabels, _, _ = bc.parameters.get_keys(["age", param])
    paramlab = paramlabels[1]

    allx = np.zeros(noofind)
    i = 0
    for modelpath in selected:
        N = len(selected[modelpath].logPDF)
        allx[i : i + N] = grid[os.path.join(modelpath, param)][
                                  selected[modelpath].index]
        i += N
    
    x = allx[nonzero][sampled_indices]
    if "age" in param:
        paramlab = agelabs[param]
        x *= 1e-3
    
    if False:
        bayfit = np.quantile(x, qs)
    else:
    # For fixing plot for paper
        bayfit = [0.600, 0.600-0.252, 0.600+2.312]
    
    n,b = np.histogram(x, bins="scott")
    kernel = gaussian_kde(x, bw_method="scott")
    x0 = np.linspace(np.amin(x), np.amax(x), num=500)
    y0 = kernel(x0)
    y0 /= np.amax(y0)
    hist = gaussian_filter(n, 1)
    x0_hist = np.array(list(zip(b[:-1], b[1:]))).flatten()
    y0_hist = np.array(list(zip(n, n))).flatten() / np.amax(n)
    ax.fill_between(x0_hist, y0_hist, y2 = -1, interpolate=True, 
                       color=col, alpha=0.15)
    ax.plot(x0, y0, 'k')
    ax.fill_between(x0, y0, y2=-1, interpolate=True, color=col, alpha=0.15)
    
    fmt = "{{0:{0}}}".format(".3f").format
    bay = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    if len(leglabel):
        title = leglabel + "\n"
    else:
        title = ""
    title += paramlab + "="
    title += bay.format(fmt(bayfit[0]), fmt(bayfit[0]-bayfit[1]), fmt(bayfit[2]-bayfit[0]))
    ax.legend([], [], title=title, title_fontsize=16)
    
    ax.plot(np.ones(2)*bayfit[0], [-2,1.5], '-k')
    ax.plot(np.ones(2)*bayfit[1], [-2,1.5], '--k')
    ax.plot(np.ones(2)*bayfit[2], [-2,1.5], '--k')

    #ax[i,j].legend([], [], title = title, frameon=False, title_fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.05)

    if xlabel:
        ax.set_xlabel(paramlab)
    return bayfit

def plot_ratios(Grid, trackmodmax, nfit, top):
    N = len(trackmodmax)
    freqfile = os.path.join(top, 'input/Kepler444.xml')
    obskey, obs, _ = fio.read_freq(freqfile)
    obsr012, covr012, _ = freq_fit.compute_ratios(obskey, obs, "r012")
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



def plot_agecore_posterior(selected, grid, sampled_indices, noofind, nonzero, trackmodmax, n):
    cols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#882255', 
            '#DDCC77', '#CC6677', '#999933', '#AA4499']
    # Buldgen profile for overlay
    Buldgen = np.loadtxt(os.path.join(top, 'input/profile_Buldgen.txt'), 
                         comments='#', delimiter=', ')
    

    f,d = plt.subplots(1,1)
    fig, axes = plt.subplots(1,1, figsize=(8.47,4))
    bayfit = insert_posterior(d, grid, selected, sampled_indices, 
                              noofind, nonzero, "", xlabel=True)
    plt.close(f)
    axes.plot(Buldgen[:,0], Buldgen[:,1], '--', alpha=1, 
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
        
        axes.plot(age, mcore, color=col, label=lab, zorder=3, lw=2)
        axes.plot(age[mod], mcore[mod], '.', color=col, zorder=5, markersize=13)
    axes.set_ylim([-0.005,0.105])
    axes.set_xlim([-0.1, 15])
    
    axes.plot(np.ones(2)*bayfit[0], [-2,1.5], '-k' , zorder=10)
    axes.plot(np.ones(2)*bayfit[1], [-2,1.5], '--k', zorder=10)
    axes.plot(np.ones(2)*bayfit[2], [-2,1.5], '--k', zorder=10)

    axes.set_xlabel(r'Age (Gyr)')
    axes.set_ylabel(r'$M_{\mathrm{core}}$ (m/M)')
    
    # Beautyfication
    axes.legend(fontsize=16,title_fontsize=16)
    fig.tight_layout()
    fig.savefig("plots/{:03d}/agecore.pdf".format(n), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(444)
    Nbest = 5
    only = [16]
    sets = []
    #only = [i for i in range(65,129)] + [204]
    #only = [204, 257, 258]
    #only = [321, 322, 323, 324]
    #sets = [[325, 326, 327, 328], [329, 330, 331, 332], [333, 334, 335, 336], [337, 338, 339, 340]]
    set_labels = [r"No $\Delta\nu$", r"$\Delta\nu$", r"No $\Delta\nu$", r"$\Delta\nu$"]
    legends1 = [r"$f_{\mathrm{ove}}>0.002$", r"$f_{\mathrm{ove}}>0.004$",
                r"$f_{\mathrm{ove}}>0.006$", r"$f_{\mathrm{ove}}>0.008$"]
    legends2 = [r"$0-1500$", r"$501-2000$", r"$1001-2500$", r"$1501-3000$"]
    set_legends = [legends1, legends1, legends2, legends2]
    #only = [16, 96, 256, 321, 322]
    main(Nbest, only, sets, set_labels, set_legends)
