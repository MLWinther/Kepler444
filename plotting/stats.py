import os
import sys
import glob
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import basta.constants as bc
import basta.utilities as ut
import basta.ratios as ratio
import basta.freq_cor as freq_cor
import basta.seismic_utils as su
from basta.fileio import read_freq, read_obs
from helpers import *
#from utils.garstec import StarLog

# Set the style of all plots
plt.style.use(os.path.join(os.environ['BASTADIR'], 'basta/plots.mplstyle'))

def FeH_from_ZX(ZX, m):
    buldalpha = {13: "A09Pho_P02", 12: "A09Pho_P03"}
    ZXsun = bc.mixtures.SUNzx["AS09"]
    MeH = np.log10(ZX/ZXsun)
    FeH = MeH - bc.mixtures.M_H_Corr_Alpha[buldalpha[m]]
    return FeH

def chi2(modv, obs):
    chi2 = ((modv - obs[0])/obs[1])**2.
    return chi2


top = '/home/au540782/Kepler444/'
grid = h5py.File(os.path.join(top, 'input/Garstec_AS09_Kepler444.hdf5'), 'r')
frefile = os.path.join(top, 'input/Kepler444_Campante.xml')
notfile = os.path.join(top, 'input/nottrusted_Kepler444.fre')

specobs = {"FeH": [-0.52, 0.12], "Teff": [5172, 75], "LPhot": [0.4,0.04], "logg": [4.595, 0.06]}

buldgenmod = [12, 13]
outputnumber = [8]
threepoint = [False, True]

for remove in [False, notfile]:
    for tp in threepoint:
        if remove:
            sys.stdout = open(os.devnull, 'w')
            obskey, obs, _ = read_freq(frefile, nottrustedfile=remove)
            sys.stdout = sys.__stdout__
        else:
            obskey, obs, _ = read_freq(frefile)
        obsr012, covr012 = obs_ratios(frefile, remove=remove, tp=tp)
        identy = np.identity(covr012.shape[0])
        for j in range(covr012.shape[0]):
            identy[j,j] = covr012[j,j]
        for j, cov in enumerate([covr012, identy]):
            cms = "Covariance" if j==0 else "Identity"
            rmessage = "modes removed" if remove else ""
            for opn in outputnumber:
                desc = glob.glob(os.path.join(top, 'modelling/output{:02d}*'.format(opn)))[0]
                desc = desc.split('{:02d}_'.format(opn))[-1].split('/')[0]
                print('\n'+'#'*40)
                print("Fit nr: ", opn, desc, rmessage)
                print("Cov:", cms)
                selmod = get_selected(opn)
                track, index, _ = get_nmax(selmod, 1)[0]
                
                freqs = grid_freqs(grid, track, index, obs, obskey)
                modspec = {}
                for par in specobs:
                    modspec[par] = grid[os.path.join(track, par)][index]
                chispec = sum([chi2(modspec[o], specobs[o]) for o in specobs])
                chifreq = chi2_difference(obsr012, cov, freqs, tp=tp)
                chisum = (chispec + chifreq)/(len(specobs)+obsr012.shape[0])

                print("Chi2sum:\t{:1.3f}".format(chisum))
                print("Chi2spec:\t{:1.3f}".format(chispec))
                print("Chi2freq:\t{:1.3f}".format(chifreq))

                
            for bmod in buldgenmod:
                print('\n'+'#'*40)
                print('Buldgen model: ', bmod, rmessage)
                print("Cov:", cms)
                buldfreqs = buldgen_freqs(bmod, obs, obskey)
                
                logdat = buldgen_log(bmod, ['Teff', 'Zs', 'Xs', 'ZXs', 'L', 'logg'])[-1,:]
                FeH = FeH_from_ZX(logdat[3], bmod)
                bspec = {"FeH": FeH, "Teff": logdat[0], 
                         "LPhot": logdat[4], 'logg': logdat[5]}
                chispec = sum([chi2(bspec[o], specobs[o]) for o in specobs])
                chifreq = chi2_difference(obsr012, cov, buldfreqs, tp=tp)
                chisum = (chispec + chifreq)/(len(specobs)+obsr012.shape[0])

                print("Chi2sum:\t{:1.3f}".format(chisum))
                print("Chi2spec:\t{:1.3f}".format(chispec))
                print("Chi2freq:\t{:1.3f}".format(chifreq))
