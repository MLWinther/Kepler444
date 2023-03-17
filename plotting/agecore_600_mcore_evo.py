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


Grid = h5py.File('../input/Garstec_AS09_Kepler444_diff.hdf5','r')
Buldgen = np.loadtxt('../input/profile_Buldgen.txt', delimiter = ', ')

ypar = 'FeHini'

_, labels, _, _ = constants.parameters.get_keys(['age', 'Mcore', 'gcut', 'ove', 
                                                 'massini', ypar])

cols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', 
        '#DDCC77', '#CC6677', '#882255', '#AA4499']

top = 'grid/tracks/'
tracks = list(Grid['grid/tracks/'])

fig, ax = plt.subplots(1,1)
for i, track in enumerate(tracks):
    path = os.path.join(top, track)
    agecore = Grid[os.path.join(path, "agecore")][0]
    ove = Grid[os.path.join(path, "ove")][0]
    if agecore > 550 and agecore < 650:
        k = 0
    #elif ove < 1e-3:
    #    k = 1
    else:
        continue

    age  = np.asarray(Grid[os.path.join(path, 'age')])*1e-3
    cor  = np.asarray(Grid[os.path.join(path,'Mcore')])
    ax.plot(age, cor, '-', color=cols[k])

ax.set_xlim(0,1)
ax.set_ylim([-0.005,0.15])
ax.plot(Buldgen[:,0], Buldgen[:,1], '--', color='grey')
ax.set_xlabel(r"Age (Gyr)")
ax.set_ylabel(labels[1])
fig.tight_layout()
fig.savefig("plots/agecore600_test.pdf", dpi=300)
plt.close(fig)
