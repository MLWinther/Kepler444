import os
import copy
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

import helpers

import basta.stats as stats
import basta.utils_seismic as su
import basta.fileio as fio
import basta.freq_fit as freq_fit

##################################################################################
# Plot and test individual chi2 values for each ratio mode, for referee response #
# out: ratio_comparison.pdf                                                      #
##################################################################################


def xdiff(obsr, modr, intpol=False):
    if intpol:
        modratio = copy.deepcopy(obsr)
        # Seperate and interpolate within the separate r01, r10 and r02 sequences
        for rtype in set(obsr[2, :]):
            obsmask = obsr[2, :] == rtype
            modmask = modr[2, :] == rtype
            intfunc = interp1d(
                modr[1, modmask], modr[0, modmask], kind="linear"
            )
            modratio[0, obsmask] = intfunc(
                obsr[1, obsmask]
            )
        x = obsr[0,:] - modratio[0,:]
        return x, modratio
    else:
        x = obsr[0,:] - modr[0,:]
        return x, modr

rcset = {'xtick.direction': u'in', 'ytick.direction': u'in', 'xtick.top':True, 'ytick.right':True, 
         'font.family': "sans-serif", "font.size":18}
plt.rcParams.update(rcset)

top = '/home/au540782/Kepler444'
gridfile = os.path.join(top, "input", "Garstec_AS09_Kepler444_diff.hdf5")
grid = h5py.File(gridfile, 'r')

selmodfile = glob.glob(os.path.join(top, "all_combinations", "output_016_*", "*.json"))[0]
selectedmodels = fio.load_selectedmodels(selmodfile)
track, ind = stats.most_likely(selectedmodels)


frefile = os.path.join(top, "input", "Kepler444.xml")
notfile = os.path.join(top, "input", "nottrusted_Kepler444.fre")
vfmt = "{:.4f}"

intpol = False
incmodes = {0: [[20, 3], [21, 5], [22, 3]], 1: [[20, 3], [21, 5]]}

multiply = False

fig, ax = plt.subplots(3,1)
allprint = []
for j in range(2):
    for k, without in enumerate([False,]):# True]):
        if without:
            obskey, obs, _ = fio.read_freq(frefile, nottrustedfile=notfile)
        else:
            obskey, obs, _ = fio.read_freq(frefile)
        if j == 1:
            if multiply:
                for l in incmodes.keys():
                    for (n,mult) in incmodes[l]:
                        lmask = obskey[0, :] == l
                        nmask = obskey[1, :] == n
                        lmask &= nmask
                        nind = np.where(lmask)[0][0]
                        obs[1, nind] *= mult

        np.random.seed(444)
        obsr012, cov = freq_fit.compute_ratios(obskey, obs, "r012")
        covinv = np.linalg.pinv(cov, rcond=1e-8)

        Hod    = su.transform_obj_array(grid[track]["osc"][ind])
        Hodkey = su.transform_obj_array(grid[track]["osckey"][ind])
        if intpol:
            modkey, mod = Hodkey, Hod
        else:
            modkey, mod = freq_fit.calc_join(Hod, Hodkey, obs, obskey)
        HLMr012 = freq_fit.compute_ratioseqs(modkey, mod, "r012")
        
        modkey, mod = helpers.buldgen_freqs("12", obs, obskey, top=top, extend=intpol)
        b12r012 = freq_fit.compute_ratioseqs(modkey, mod, "r012")
        modkey, mod = helpers.buldgen_freqs("13", obs, obskey, top=top, extend=intpol)
        b13r012 = freq_fit.compute_ratioseqs(modkey, mod, "r012")

        x, HLMr012 = xdiff(obsr012, HLMr012, intpol=intpol)
        if not multiply and j == 1:
            for i in range(len(obsr012[0,:])):
                if obsr012[2, i] == 1 and obsr012[3, i] == 21:
                    x[i] = 0
        chi2cov = (x.T.dot(covinv).dot(x)) / len(x)
        chi2unc = (x.T.dot(np.identity(covinv.shape[0])*covinv).dot(x)) / len(x)
        print("HLM: " + vfmt.format(chi2cov) + ", " + vfmt.format(chi2unc))

        x, b12r012 = xdiff(obsr012, b12r012, intpol=intpol)
        if not multiply and j == 1:
            for i in range(len(obsr012[0,:])):
                if obsr012[2, i] == 1 and obsr012[3, i] == 21:
                    x[i] = 0
        chi2cov = (x.T.dot(covinv).dot(x)) / len(x)
        chi2unc = (x.T.dot(np.identity(covinv.shape[0])*covinv).dot(x)) / len(x)
        print("B12: " + vfmt.format(chi2cov) + ", " + vfmt.format(chi2unc))
        
        x, b13r012 = xdiff(obsr012, b13r012, intpol=intpol)
        if not multiply and j == 1:
            for i in range(len(obsr012[0,:])):
                if obsr012[2, i] == 1 and obsr012[3, i] == 21:
                    x[i] = 0
        chi2cov = (x.T.dot(covinv).dot(x)) / len(x)
        chi2unc = (x.T.dot(np.identity(covinv.shape[0])*covinv).dot(x)) / len(x)
        print("B13: " + vfmt.format(chi2cov) + ", " + vfmt.format(chi2unc))
        
        allchi2 = np.zeros((3,len(x)))
        for i in range(len(obsr012[0,:])):
            prtstr = []
            dbstr  = []
            prtstr.append("$" + "r_{{0{0:d}}}({1:d})".format(int(obsr012[2,i]), int(obsr012[3,i])) + "$")
            allchi2[0,i] = (obsr012[0,i]-HLMr012[0,i])**2*covinv[i,i]
            allchi2[1,i] = (obsr012[0,i]-b12r012[0,i])**2*covinv[i,i]
            allchi2[2,i] = (obsr012[0,i]-b13r012[0,i])**2*covinv[i,i]

            #dbstr.append(" " * len(prtstr[0]))
            #dbstr.append(vfmt.format(abs(obsr012[0,i]-HLMr012[0,i])))
            # dbstr.append(vfmt.format(abs(obsr012[0,i]-b12r012[0,i])))
            # dbstr.append(vfmt.format(abs(obsr012[0,i]-b13r012[0,i])))
            # dbstr.append("{:.3f}".format(cov[i,i]**0.5*1e3))
            # dbstr.append("{:.3e}".format(covinv[i,i]))

            prtstr.append(vfmt.format(cov[i,i]**0.5))
            prtstr.append(vfmt.format((obsr012[0,i]-HLMr012[0,i])**2*covinv[i,i]))
            prtstr.append(vfmt.format((obsr012[0,i]-b12r012[0,i])**2*covinv[i,i]))
            prtstr.append(vfmt.format((obsr012[0,i]-b13r012[0,i])**2*covinv[i,i]))
            if j == 0:
                allprint.append(prtstr)
            else:
                allprint[i].extend(prtstr)
            #print("   ".join(dbstr) + "\n")
            print(" & ".join(prtstr) + "\\\\")

        print(chi2cov, chi2unc)
        offset = 30.0
        obsr012[1, obsr012[2,:] == 2] += offset
        HLMr012[1, HLMr012[2,:] == 2] += offset
        b12r012[1, b12r012[2,:] == 2] += offset
        b13r012[1, b13r012[2,:] == 2] += offset
        cmax = 200 #np.max(allchi2)*100
        ylim = [-0.01, 0.08]
        if k == 0:
            ax[0].errorbar(obsr012[1,:], obsr012[0,:], yerr=np.diag(cov)**0.5,fmt='o', color='k', label=r"Obs")
            ax[1].errorbar(obsr012[1,:], obsr012[0,:], yerr=np.diag(cov)**0.5,fmt='o', color='k', label=r"Obs")
            ax[2].errorbar(obsr012[1,:], obsr012[0,:], yerr=np.diag(cov)**0.5,fmt='o', color='k', label=r"Obs")
            ax[0].errorbar(HLMr012[1,:], HLMr012[0,:], yerr=allchi2[0,:]/cmax, fmt='s', color='#88CCEE', markeredgewidth=2, label=r"HLM")
            ax[1].errorbar(b12r012[1,:], b12r012[0,:], yerr=allchi2[1,:]/cmax, fmt='s', color='#882255', markeredgewidth=2, label=r"$\mathcal{M}_{12}$")
            ax[2].errorbar(b13r012[1,:], b13r012[0,:], yerr=allchi2[2,:]/cmax, fmt='s', color='#DDCC77', label=r"$\mathcal{M}_{13}$")

            ax[-1].set_xlabel(r"$\nu\,[\mu \mathrm{Hz}]$")
            ax[1].set_ylabel(r"$r_{012}$")

            ax[0].set_ylim(ylim)
            ax[1].set_ylim(ylim)
            ax[2].set_ylim(ylim)

ax[0].set_title("Normal error")
ax[0].set_title("Increased errors")
ax[0].legend()
ax[1].legend()
ax[2].legend()

fig.tight_layout()
fig.savefig("plots/ratio_comparison.pdf")

#for prtstr in allprint:
#    print(" & ".join(prtstr) + "\\\\")