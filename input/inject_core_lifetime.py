import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

tol = 1e-4
cols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
        '#DDCC77', '#CC6677', '#882255', '#AA4499']

for d in ["diff"]:#, "nodiff"]:
    grid = h5py.File("Garstec_AS09_Kepler444_{0}.hdf5".format(d), "r+")
    
    bp = 'grid/'
    for group, tracks in grid[bp].items():
        for name, libitem in tracks.items():
            age = np.asarray(libitem['age'])
            mcore = np.asarray(libitem['Mcore'])
            rev_mcore = mcore[::-1]
            ind = len(mcore) - np.argmax(rev_mcore > tol) - 1
            if ind != len(mcore) - 1:
                lt = age[ind]
            else:
                fit1 = np.polyfit(mcore[-50:], age[-50:], 1)
                fit2 = np.polyfit(mcore[-5:], age[-5:], 1)
                if fit1[0] > -2e5:
                    fit = fit2
                else:
                    fit = fit1
                lt = fit[1]

            path = os.path.join(bp, group, name, "agecore")
            try:
                grid[path]
                del grid[path]
            except:
                None
            grid[path] = min(lt, 2e4)*np.ones(len(age))
