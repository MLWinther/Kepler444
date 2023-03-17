import os
import glob
import h5py

import numpy as np

import basta.stats as stats
import basta.fileio as fio
import basta.freq_fit as freq_fit
import basta.plot_seismic as plt_seis
import basta.utils_seismic as su


top = "/home/au540782/Kepler444/"

numax = 4538
obsdnu = 179.64

grid = h5py.File(os.path.join(top, "input", 
                 "Garstec_AS09_Kepler444_diff.hdf5"))

obskey, obs, _ = fio.read_freq(os.path.join(top, "input", "Kepler444.xml"),
                nottrustedfile=os.path.join(top, "input", "nottrusted_Kepler444.fre"))

fitnum = 16
jsonfile = glob.glob(os.path.join(top, "all_combinations", 
                     "output_{:03d}_*/*.json".format(fitnum)))[0]
#jsonfile = glob.glob(os.path.join(top, "modelling", 
#                     "output_dnufit_in*/*.json"))[0]
                     

selmod = fio.load_selectedmodels(jsonfile)

pathPDF, indPDF = stats.get_highest_likelihood(grid, selmod, ["age", "agecore"])
pathCHI, indCHI = stats.get_highest_likelihood(grid, selmod, ["age", "agecore"])

names = ["HLM", "LCM"]
plotname = "plots/echelle_{:s}_{:s}.pdf"

for path, ind, name in zip([pathPDF, pathCHI], [indPDF, indCHI], names):
    dnu    = grid[path]["dnufit"][ind]
    mod    = su.transform_obj_array(grid[path]["osc"][ind])
    modkey = su.transform_obj_array(grid[path]["osckey"][ind])
    joinkey, join = freq_fit.calc_join(
        mod=mod,
        modkey=modkey,
        obs=obs,
        obskey=obskey,
    )
    corjoin, coeffs = freq_fit.BG14(
        joinkeys=joinkey, join=join, scalnu=numax,
    )
    print(coeffs)
    plt_seis.echelle(
        selmod,
        grid,
        obs,
        obskey,
        mod=mod,
        modkey=modkey,
        dnu=obsdnu,
        join=join,
        joinkeys=joinkey,
        pair=False,
        output=plotname.format(name, "uncorrected"),
    )

    plt_seis.echelle(
        selmod,
        grid,
        obs,
        obskey,
        mod=mod,
        modkey=modkey,
        dnu=obsdnu,
        join=corjoin,
        joinkeys=joinkey,
        freqcor="BG14",
        coeffs=coeffs,
        scalnu=numax,
        pair=True,
        output=plotname.format(name, "corrected"),
    )