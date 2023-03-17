import os
import sys
import glob
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import basta.constants as bc
import basta.utils_general as ut
import basta.freq_fit as freq_fit
import basta.utils_seismic as su
import basta.fileio as fio
import helpers
#from utils.garstec import StarLog

# Set the style of all plots
plt.style.use(os.path.join(os.environ['BASTADIR'], 'basta/plots.mplstyle'))
rcset = {'xtick.direction': u'in', 'ytick.direction': u'in', 'xtick.top':True, 'ytick.right':True, 'font.family': "sans-serif", "font.size":18}
plt.rcParams.update(rcset)

cols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']

top = '/home/au540782/Kepler444/'
freqfile = os.path.join(top, 'input/Kepler444.xml')
gridfile = os.path.join(top, 'input/Garstec_AS09_Kepler444_diff.hdf5')

Grid = h5py.File(gridfile, 'r')

obskey, obs, _ = fio.read_freq(freqfile)
obskeynot, obsnot, _ = fio.read_freq(freqfile, nottrustedfile="/home/au540782/Kepler444/input/nottrusted_Kepler444.fre")
obsr012, obsr012cov, _ = freq_fit.compute_ratios(obskey, obs, "r012")
obsr012not, obsr012notcov, _ = freq_fit.compute_ratios(obskeynot, obsnot, "r012")

outputnumber = 16
loc = glob.glob(os.path.join(top, 'all_combinations/output_{:03d}*'.format(outputnumber)))[0]
desc = loc.split('{:03d}_'.format(outputnumber))[-1].split('/')[0]

print("Fit nr: ", outputnumber, desc)

bm = [12, 13]
selmod = fio.load_selectedmodels(os.path.join(loc, "Kepler444.json"))
trackmodmax = helpers.get_nmax(selmod, 30)

track, index, _ = trackmodmax[0]
print("Ratios show track: ", track[-4:])
osc = su.transform_obj_array(Grid[track + '/osc'][index])
osckey = su.transform_obj_array(Grid[track + '/osckey'][index])

col12 = '#88CCEE'; col13 = '#882255'

modkey, mod = freq_fit.calc_join(osc, osckey, obs, obskey)
modr012 = freq_fit.compute_ratioseqs(modkey, mod, "r012")

lnmask  = obsr012[2, :] == 2
lnmask &= obsr012[3, :] > max(obsr012[3, lnmask]) - 2


fig, ax = plt.subplots(2,1,figsize=(8.47,5.23))
l1 = ax[0].errorbar(obsr012[1, ~lnmask], obsr012[0, ~lnmask], yerr=np.sqrt(np.diag(obsr012cov)[~lnmask]), fmt='d', color='k',label=r'Observed')
l2 = ax[0].errorbar(obsr012[1, lnmask], obsr012[0, lnmask], yerr=np.sqrt(np.diag(obsr012cov)[lnmask]**2), fmt='d', color='grey',label=r'Removed')
l3, = ax[0].plot(modr012[1, :], modr012[0, :], '^', color='#332288', label=r'Best fit')
l4, = ax[0].plot(0,0, '^', color=col12, label=r'$M_{12}$')
l5, = ax[0].plot(0,0, 'o', color=col13, label=r'$M_{13}$')

ax[1].errorbar(obsr012[1, ~lnmask], obsr012[0, ~lnmask], yerr=np.sqrt(np.diag(obsr012cov)[~lnmask]), fmt='d', color='k',label=r'Observed')
ax[1].errorbar(obsr012[1, lnmask], obsr012[0, lnmask], yerr=np.sqrt(np.diag(obsr012cov)[lnmask]), fmt='d', color='grey',label=r'Removed')

bmfmt = ["^", "o"]
bmcol = [col12, col13]

for buld, fmt, col in zip(bm, bmfmt, bmcol):
    osckey, osc = helpers.buldgen_freqs(buld, obs, obskey, top=top)
    buldr012 = freq_fit.compute_ratioseqs(osckey, osc, "r012")

    ax[1].plot(buldr012[1, :], buldr012[0, :], fmt, color=col, label=r'$M_{%d}$' % (buld))

ax[0].legend([l1,l2,l3,l4,l5], 
             [r'Observed', r'Removed', r'HLM', r'$\mathcal{M}_{12}$', r'$\mathcal{M}_{13}$'],
             bbox_to_anchor=(0, 1.02, 1, 0.102), loc=8, mode="expand", borderaxespad=0, 
             ncol=5,fontsize=18,handletextpad=-0.3)
#ax[1].legend([l4,l5], [r'$M_{12}$', r'$M_{13}$'], frameon=True)

ax[1].set_xlabel(r'$\nu\,(\mu \mathrm{Hz})$')
ax[0].set_ylabel(r'$r_{012}$')
ax[1].set_ylabel(r'$r_{012}$')

ax[0].set_xticks([])

ax[0].set_xlim([3400, 5400])
ax[1].set_xlim([3400, 5400])
fig.tight_layout()
fig.subplots_adjust(hspace=0)
fig.savefig('plots/ratios.pdf', dpi=300)
plt.close()
###########################################################
