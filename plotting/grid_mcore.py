import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import basta.constants as constants

# Set the style of all plots
rcset = {'xtick.direction': u'in', 'ytick.direction': u'in', 
         'xtick.top': True, 'ytick.right': True,
         'font.family': 'sans-serif', 'font.size': 18}
plt.rcParams.update(rcset)

######################################################################
# Plot the variation of convective core lifetimes, figure 2 in paper #
# out: fg_mcore_sample.pdf                                           #
######################################################################

Grid = h5py.File('../input/Garstec_AS09_Kepler444_diff.hdf5','r')
Buldgen = np.loadtxt('../input/profile_Buldgen.txt', delimiter = ',')

ypar = 'FeHini'

_, labels, _, _ = constants.parameters.get_keys(['age', 'Mcore', 'gcut', 'ove', 
                                                 'massini', ypar])

cols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', 
        '#DDCC77', '#CC6677', '#882255', '#AA4499']

top = 'grid/tracks/'
tracks = list(Grid['grid/tracks/'])
markersize = 10

lifetimes = np.array([Grid[os.path.join(top, track, 'agecore')][0] for track in tracks])

offsets = [np.ones(len(cols))*0.005, np.ones(len(cols))*0.001,
           np.zeros(len(cols)), np.zeros(len(cols))]
offsets[0][5]  = -0.005
offsets[1][5]  = -0.001
offsets[2][-1] = -0.011
offsets[2][1]  = -0.002
offsets[3][0]  = -0.01

step = 2e3
dn = [0 for _ in cols]
dn[-1] = -5
lft = 0
subtracks = []
for i,_ in enumerate(cols):
    mask = np.where(np.logical_and(lifetimes > lft, lifetimes < lft + step))[0][7+dn[i]]
    subtracks.append(tracks[mask])
    lft += step




fig, ax = plt.subplot_mosaic([["ax0", "ax0"], ["ax1", "ax2"]], figsize=(8.47, 7))

ax["ax0"].plot(Buldgen[:,0], Buldgen[:,1], '--k', alpha=0.5, label=r'$\mathcal{M}_{13}$')

for i, track in enumerate(subtracks):
    path = os.path.join(top, track)
    mass = Grid[os.path.join(path,'massini')][0]
    
    age  = np.asarray(Grid[os.path.join(path, 'age')])*1e-3
    cor  = np.asarray(Grid[os.path.join(path,'Mcore')])
    mini = float(Grid[os.path.join(path, 'massini')][0])
    amlt = float(Grid[os.path.join(path, 'alphaMLT')][0])
    gcut = float(Grid[os.path.join(path, 'gcut')][0])
    ove  = float(Grid[os.path.join(path, 'ove')][0])
    yax1 = float(Grid[os.path.join(path, ypar)][0])
    acor = float(Grid[os.path.join(path, 'agecore')][0])*1e-3
    ind  = min(np.argmin(abs(age-acor)), np.argmin(abs(age-14.5)))

    ax["ax0"].plot(age , cor , '-', color=cols[i], label="__nolegend__")
    ax["ax0"].text(age[ind]+0.1, cor[ind] + 0.002, str(i+1))
    ax["ax1"].plot(mini, yax1, '.', color=cols[i], markersize=markersize)
    ax["ax1"].text(mini + offsets[0][i], yax1 + offsets[2][i], str(i+1),
                   verticalalignment='center', horizontalalignment='center')
    ax["ax2"].plot(ove , gcut, '.', color=cols[i], markersize=markersize)
    ax["ax2"].text(ove + offsets[1][i], gcut + offsets[3][i], str(i+1),
                   verticalalignment='center', horizontalalignment='center')

#ax.legend(facecolor='white', frameon=True, framealpha=1, bbox_to_anchor=(1, 1), loc='upper left',fontsize=16)
ax["ax0"].legend(facecolor='white', frameon=False, framealpha=1, fontsize=18)
ax["ax0"].set_xlabel(r'Age (Gyr)')
ax["ax0"].set_ylabel(labels[1])
ax["ax0"].set_xlim([-0.2,15])
ax["ax0"].set_ylim([-0.005,0.125])
ax["ax1"].set_xlabel(labels[4])
ax["ax1"].set_ylabel(labels[5])
ax["ax1"].set_xlim([0.695, 0.805])
ax["ax2"].set_xlabel(labels[3])
ax["ax2"].set_ylabel(labels[2])

fig.tight_layout()
fig.savefig('plots/fg_mcore_sample.pdf', dpi=300)
