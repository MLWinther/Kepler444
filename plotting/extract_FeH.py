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
cols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']


Grid = h5py.File('../input/Garstec_AS09_Kepler444.hdf5','r')

_, labels, _, _ = constants.parameters.get_keys(['age', 'FeH', 'FeHini'])

gstr = r'$g_{\mathrm{cut}}$'

top = 'grid/tracks/'
tracks = list(Grid['grid/tracks/'])

data = [[], [], [], []]
ga1 = []

for trackname in tracks:
    track = Grid[os.path.join(top, trackname)]
    if not track['dif'][0]:
        continue
    if track['gcut'][0] > 1.0:
        ga1.append(trackname)
    FeHini = track['FeHini'][0]

    ind = np.where(abs(np.asarray(track['age']) - 10e3) == min(abs(np.asarray(track['age']) - 10e3)))[0][0]
    FeH10  = track['FeH'][ind]
    ind = np.where(abs(np.asarray(track['age']) - 12e3) == min(abs(np.asarray(track['age']) - 12e3)))[0][0]
    FeHend = track['FeH'][ind]

    data[0].append(FeH10)
    data[1].append(FeHend)
    data[2].append(FeH10  - FeHini)
    data[3].append(FeHend - FeHini)



lab = [labels[1], labels[1],
       labels[1] + " - " + labels[2], labels[1] + " - " + labels[2]]
title = [r"$\mathrm{Age} = 10\mathrm{Gyr}$",r"$\mathrm{Age} = 12\mathrm{Gyr}$*"]
fig, axes = plt.subplots(2,2, figsize=(12, 8))

for i,ax in enumerate([*axes[0,:], *axes[1,:]]):
    if i%2 == 0:
        bins = 'auto'
    hist, bins = np.histogram(data[i], bins=bins)
    ax.hist(data[i], bins=bins, rwidth=0.9, color=cols[0])

    ax.set_xlabel(lab[i])
    ax.set_title(title[i%2])


fig.tight_layout()
fig.savefig('plots/FeH_diff_sample.pdf')

fig, ax = plt.subplots(1,1)
for trackname in ga1:
    track = Grid[os.path.join(top, trackname)]
    
    age = np.asarray(track["age"])*1e-3
    mcore = np.asarray(track["Mcore"])

    ax.plot(age, mcore)

fig.savefig('plots/gcut_mcore.png')


