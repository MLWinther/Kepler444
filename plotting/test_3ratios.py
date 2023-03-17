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

top = '/home/au540782/Kepler444/'
frefile = os.path.join(top, 'input/Kepler444_Campante.xml')
notfile = os.path.join(top, 'input/nottrusted_Kepler444.fre')

obskey, obs, _ = read_freq(frefile)
obsr012, covr012 = obs_ratios(frefile)
buldfreqs = buldgen_freqs(12, obs, obskey)

r02, r01, _ = ratio.ratios(convert_obs_freqs(obs, obskey), tp=True)
br02, br01, _ = ratio.ratios(buldfreqs, tp=True)

fig, ax = plt.subplots(1,1)
ax.plot(obsr012[:,3], obsr012[:,1], '.', label="Obs 5-point")
ax.plot(r01[:,3], r01[:,1], '.', label=" Obs 3-point")
ax.plot(br01[:,3], br01[:,1], '.', label="Buldgen")
ax.legend()
fig.tight_layout()
fig.savefig('plots/test_3ratios.png')

