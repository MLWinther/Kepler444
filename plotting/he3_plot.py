import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import he3_eq
from garstec_helpers import read_garlog, read_fgong
from basta.constants import parameters

# Set the style of all plots
plt.style.use(os.path.join(os.environ['BASTADIR'], 'basta/plots.mplstyle'))
rcset = {'xtick.direction': u'in', 'ytick.direction': u'in', "xtick.top": True, 
         "ytick.right": True, "font.family":"sans-serif", 'font.size': 18}
plt.rcParams.update(rcset)


path = 'He3models/Buldgen_{0}/'
mods = [path.format(i) for i in ['00', '15']]
labs = [r'$f = 0.000$', r'$f = 0.015$']

col = ["#CC6677", "#88CCEE", "#332288",]
xlim = [-0.1,12]

Buldgen = np.loadtxt('/home/au540782/Kepler444/input/profile_Buldgen.txt', delimiter=',')

_, plab, _, _ = parameters.get_keys(["age", "Mcore"])

fig, ax = plt.subplots(1,2,figsize=(8.47,4))
for i,modpath in enumerate(mods):
    fgongs = np.array(glob.glob(modpath+'*.fgong'))
    number = np.array([int(name.split('-')[-1].split('.')[0]) for name in fgongs])
    fgongs = fgongs[np.argsort(number)]
    
    age      = []
    cen_he3  = []
    cen_he3e = []

    for fgong in fgongs:
        starg, starl = read_fgong(fgong)
        
        he3 = he3_eq.he3_frac(np.array(starl['X']), np.array(starl['xhe4']), np.array(starl['Temp']))
        cen_he3e.append(he3[0])
        cen_he3.append(np.array(starl['xhe3'])[0])
        age.append(float(starg['Age']))
    
    age = np.asarray(age)*1e-9
    cen_he3 = np.asarray(cen_he3)
    cen_he3e = np.asarray(cen_he3e)

    lage, mcore = read_garlog(glob.glob(modpath + '*.log')[0], "15", "age", "mcnvc")
    lage *= 1e-3
    ax[0].plot(lage, mcore, '-', color=col[i], label=labs[i])
    ax[1].plot(age, cen_he3, '-', color=col[i], label=labs[i])
    ax[1].plot(age, cen_he3e, '--', color=col[i])


ax[0].plot(Buldgen[:,0], Buldgen[:,1], '--', color='k', label=r'$M_{13}$',zorder=0)
ax[0].set_ylim([-0.01, 0.15]); ax[0].set_xlim(xlim)

ylim = ax[1].get_ylim()
ax[1].plot(-1,-1, '--k', label=r'$(X_{^3\mathrm{He}})_e$')
ax[1].set_ylim(ylim); ax[1].set_xlim(xlim)
    
ax[0].set_xlabel(plab[0])
ax[0].set_ylabel(plab[1])
ax[1].set_xlabel(plab[0])
ax[1].set_ylabel(r'$X_{^3\mathrm{He}}$')

ax[0].legend()
ax[1].legend()
fig.tight_layout()
fig.savefig('plots/buldgen_he3.pdf')
plt.close()
