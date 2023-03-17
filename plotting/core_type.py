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
from basta.fileio import read_freq
from helpers import *
#from utils.garstec import StarLog
rcset = {'xtick.direction': u'in', 'ytick.direction': u'in', "xtick.top": True, "ytick.right": True, "font.family":"sans-serif", "font.size": 18}
plt.rcParams.update(rcset)
# Set the style of all plots
#plt.style.use(os.path.join(os.environ['BASTADIR'], 'basta/plots.mplstyle'))
top = '/home/au540782/Kepler444/'

def plot_core_types(grid, selmod, core_type, hist):
    col = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', 
           '#DDCC77', '#CC6677', '#882255', '#AA4499']
    col = [col[6], col[1], col[3]]
    Buldgen = np.loadtxt(os.path.join(top, 'input/profile_Buldgen.txt'), comments='#', delimiter=', ')
    fig, axes = plt.subplots(3,2, sharey='col', figsize=(8.47,7),
                             gridspec_kw={'width_ratios': [2,1.1]})
    for track in selmod:
        ax = axes[core_type[track]-1,0]

        age = np.asarray(grid[track]["age"])*1e-3
        mcore = np.asarray(grid[track]["Mcore"])

        ax.plot(age,mcore, alpha=0.3, color=col[core_type[track]-1],zorder=3)
        ax.set_ylim([-0.005,0.105])
        ax.set_xlim([-0.1, 15])
    for i in range(3):
        axes[0,0].plot(-1, -1, '-', alpha=0.8, color=col[i], label=r'Core type {0}'.format(i+1))
    for i in range(3):
        axes[i,0].plot(Buldgen[:,0], Buldgen[:,1], '--k',alpha=1, label=r'$M_{13}$',zorder=5)
    axes[0,0].set_xticklabels([])
    axes[1,0].set_xticklabels([])
    axes[-1,0].set_xlabel(r'Age (Gyr)')
    axes[1,0].set_ylabel(r'$M_{\mathrm{core}}$ (m/M)')
    
    gs = axes[0,1].get_gridspec()
    for ax in axes[0:,-1]:
        ax.remove()
    ax = fig.add_subplot(gs[0:,-1])
    w = 0.25; pad = 0.1
    for i,h in enumerate(hist):
        ax.fill_betweenx([-1-w+i, -1+w+i], h, -1, color=col[i])
    ax.set_ylim([1.0+w+pad,-1.0-w-pad])
    ax.set_xlim([0, 1.])
    ax.set_yticks([])
    #ax.set_xticklabels([])
    ax.get_xaxis().tick_bottom()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel(r'Normalised Posterior')
    axes[0,0].legend([],[], bbox_to_anchor=(-0.1,1.02), loc='lower left', fontsize=14)
    fig.tight_layout()
    axes[0,0].legend(bbox_to_anchor=(-0.1,1.02), loc='lower left', ncol=4, fontsize=14)
    fig.savefig("plots/core_types.pdf")
    plt.close()

def inject_core_type(grid, interval=[2,9], tol=0.01, err=1e-4):
    core_type = determine_core_type(grid, interval, err)
    for group, tracks in grid['grid/'].items():
        for t,libitem in tracks.items():
            l = len(libitem['age'])
            ct = np.ones(l)*core_type[libitem.name[1:]]
            keypath = os.path.join(libitem.name, 'Ctype')
            grid[keypath] = ct
            

grid = h5py.File(os.path.join(top, 'input/Garstec_AS09_Kepler444.hdf5'), 'r+')


outputnumber = 8
selmod = get_selected(outputnumber)

core_type = determine_core_type(grid, interval=[0.5,9], tol=0.004)
#inject_core_type(grid)
logy = np.concatenate([ts.logPDF for ts in selmod.values()])
noofind = len(logy)
nonzero = np.isfinite(logy)
logy = logy[nonzero]
lk = logy - np.amax(logy)
pd = np.exp(lk - np.log(np.sum(np.exp(lk))))

x = np.concatenate([np.ones(len(t.logPDF))*core_type[p] for p,t in selmod.items()])[nonzero]

bins = [0.5, 1.5, 2.5, 3.5]
hist, bins = np.histogram(x, bins=bins, weights=pd)
plot_core_types(grid, selmod, core_type, hist)

