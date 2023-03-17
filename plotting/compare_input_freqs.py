import os
import numpy as np
import matplotlib.pyplot as plt

from basta.utils_seismic import ratio_and_cov
from basta.fileio import read_freqs_xml

plt.style.use(os.path.join(os.environ["BASTADIR"], "basta/plots.mplstyle"))

inp = '/home/au540782/Kepler444/input/'
filenames = [os.path.join(inp, 'Kepler444_Campante.xml'), os.path.join(inp, 'Kepler444.xml')]
labs = [r"Campante $\ell = {0}$", r"Davies $\ell = {0}$"]
cols = ["#D55E00", "#009E73", "#0072B2"]
colc = ["#D55E00", "#0072B2"]
cold = ["#009E73", "#CC3311"]
fmt  = ["D", "^", "v"]

dnufit = 179.64

fc, ec, nc, lc = read_freqs_xml(filenames[0])
fd, ed, nd, ld = read_freqs_xml(filenames[1])

xlim = [-0.5, 0.5]

fig, ax = plt.subplots(1, 4, sharey=True,
                       gridspec_kw={"width_ratios": [1, 0.2, 0.2, 0.2]})

for ll in [0, 1, 2]:
    maskc = lc == ll
    maskd = ld == ll
    
    ax[0].errorbar(fc[maskc] % dnufit, fc[maskc], xerr=ec[maskc], fmt="o", 
                   color=cols[ll], label=labs[0].format(ll))
    ax[0].errorbar(fd[maskd] % dnufit, fd[maskd], xerr=ed[maskd], fmt="D", 
                   color=cols[ll], label=labs[1].format(ll), markersize=5,
                   markeredgecolor='k', markeredgewidth=1)

    diff = fc[maskc] - fd[maskd]
    ax[ll+1].plot(diff, fc[maskc], 'v', color=cols[ll])
    ax[ll+1].errorbar(np.zeros(len(fc[maskc])), fc[maskc], xerr=ec[maskc], fmt='.',
                      color="darkgrey", zorder=-4, alpha=0.7, markersize=0)
    ax[ll+1].set_xlim(xlim)
    #ax[ll+1].set_title(r"$\ell={0}$".format(ll))

ylim = list(ax[0].get_ylim())
ylim[1] += 50
for ll in [0, 1, 2]:
    ax[ll+1].plot([0,0], ylim, '--', color="darkgrey", zorder=-5, alpha=0.7)
    ax[ll+1].legend(title=r"$\ell={0}$".format(ll), title_fontsize=14)
ax[0].set_ylim(ylim)

ax[0].legend()
ax[0].set_xlabel(r"Frequency modulo $\Delta\nu=%.2f\mu\mathrm{Hz}$" % (dnufit))
ax[0].set_ylabel(r"Frequency ($\mu$Hz)")
ax[2].set_title(r"Campante Uncertainty")
ax[2].set_xlabel(r"Campante - Davies ($\mu$Hz)")
fig.tight_layout()
fig.savefig("plots/Echelle_comparison.pdf")
plt.close(fig)



names = ["l", "n", "freq", "err"]
fmts  = [int, int, float, float]
freqc = np.zeros(len(fc), dtype={"names": names, "formats": fmts})
freqc[:]["l"] = lc
freqc[:]["n"] = nc
freqc[:]["freq"] = fc
freqc[:]["err"] = ec

freqd = np.zeros(len(fc), dtype={"names": names, "formats": fmts})
freqd[:]["l"] = ld
freqd[:]["n"] = nd
freqd[:]["freq"] = fd
freqd[:]["err"] = ed


labc = [r"Campante $r_{01}$", "Campante $r_{02}$"]
labd = [r"Davies $r_{01}$", "Davies $r_{02}$"]

ratc, _ = ratio_and_cov(freqc, rtype="R012")
ratd, _ = ratio_and_cov(freqd, rtype="R012")

ylim = [-0.003, 0.003]

offset = 15
fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1,0.2,0.2]})
for ll in [0, 1]:
    maskc = ratc[:,1] > 0.03 if ll else ratc[:,1] < 0.03 
    maskd = ratd[:,1] > 0.03 if ll else ratd[:,1] < 0.03 
    ax[0].errorbar(ratc[:,3][maskc], ratc[:,1][maskc], yerr=ratc[:,2][maskc], 
                   fmt='o', color=colc[ll], label=labc[ll])
    ax[0].errorbar(ratd[:,3][maskd]+offset, ratd[:,1][maskd], yerr=ratd[:,2][maskd], 
                   fmt='D', color=cold[ll], markersize=5, markeredgecolor='k', 
                   markeredgewidth=1, label=labd[ll])

    ax[ll+1].plot(ratc[:,3][maskc], ratc[:,1][maskc]-ratd[:,1][maskd], 'v',
                  color=colc[ll], zorder=1)
    ax[ll+1].errorbar(ratc[:,3][maskc], np.zeros(ratc[maskc].shape[0]), yerr=ratc[:,2][maskc],
                      fmt='.', color='darkgrey', alpha=0.7, zorder=-4, markersize=0)
    ax[ll+1].set_ylim(ylim)

xlim = list(ax[0].get_xlim())
for ll in [0, 1]:
    ax[ll+1].plot(xlim, [0,0], '--', color='darkgrey', alpha=0.7, zorder=-5)
ax[0].set_xlim(xlim)

ax[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.02), loc=8, ncol=4)
ax[1].legend(title=r"$r_{01}$", title_fontsize=14)
ax[2].legend(title=r"$r_{02}$", title_fontsize=14)
ax[0].set_ylabel(r"$r_{012}$")
ax[1].set_ylabel(r"Campante - Davies$\qquad\qquad\quad$")
ax[2].set_xlabel(r"Frequency ($\mu$Hz) (+%.0f$\mu$Hz for Davies)" % (offset))
fig.tight_layout()
fig.savefig("plots/Ratios_comparison.pdf")
plt.close(fig)
