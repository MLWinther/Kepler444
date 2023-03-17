import h5py
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

rcset = {'xtick.direction': u'in', 'ytick.direction': u'in', 'xtick.top':True, 'ytick.right':True, 'font.family': "sans-serif", "font.size":18}
plt.rcParams.update(rcset)

bmods = [13, 12]
cols = ["#882255", "#88CCEE"]
labels = [r"$\mathcal{M}_{13}$, Overshoot", r"$\mathcal{M}_{12}$, No overshoot"]
skip = 200

fig, ax = plt.subplots(2,1, figsize=(8.47,6), sharex=True)

for i,bm in enumerate(bmods):
    dat = buldgen_log(bm, ['age', 'Mcore', 'He3'])
    dat[:, 0] = dat[:, 0]*1e-9
    ax[0].plot(dat[:, 0][skip:], dat[:, 1][skip:], '-', color=cols[i], lw=2.5, label=labels[i])
    ax[1].plot(dat[:, 0][skip:], dat[:, 2][skip:], '-', color=cols[i], lw=2.5, label=labels[i])


#ax[0].set_xlabel(r"Age (Gyr)")
ax[0].set_ylabel(r"$M_{\rm core}\, (m/M)$", labelpad=36)

ax[1].set_xlabel(r"Age (Gyr)")
ax[1].set_ylabel(r"$^3{\rm He}\,({\rm g/mol})$")

yticks = [r"$2\times 10^{-5}$", r"$4\times 10^{-5}$",r"$6\times 10^{-5}$",r"$8\times 10^{-5}$"]
ax[1].set_yticks([2e-5, 4e-5, 6e-5, 8e-5])
ax[1].set_yticklabels(yticks)


ax[0].legend(fontsize=18)

fig.tight_layout()
fig.subplots_adjust(hspace=0)
plt.savefig("plots/core_buldgen.pdf")
plt.close()
