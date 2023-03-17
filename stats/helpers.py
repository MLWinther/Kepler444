import os
import numpy as np
import matplotlib.pyplot as plt
import basta.constants as bc
import basta.fileio as fio
import basta.utils_general as ut
import basta.freq_fit as freq_fit
import basta.utils_seismic as su
from basta.fileio import read_freq
from basta.stats import quantile_1D

#from utils.garstec import StarLog

top = '/home/au540782/Kepler444/'
names = ['l', 'n', 'freq', 'err']
fmts = [int, int, float, float]

# Set the style of all plots
plt.style.use(os.path.join(os.environ['BASTADIR'], 'basta/plots.mplstyle'))

class mixtures:
    """
    Quantities for the solar mixtures
    """
    # Alpha enhancement constants for converting observations
    # KRISTOFFER: PUT NUMBERS HERE
    A_alpha = {'GN93': 0.684, 'GS98': 0.683, 'AS09': 0.654}
    B_alpha = {'GN93': 0.316, 'GS98': 0.317, 'AS09': 0.346}

    # Alpha enhancement constants (derived from the full list of abundances)
    # [Kristoffer B. Nielsen, Feb 2019]
    M_H_Corr_Alpha = {'ALEN': 0.29204658638268377,
                      'GN93': 0.29204658638268377,
                      'A09Pho_N02': -0.11979672909034456,
                      'A09Pho_N01': -0.06252186498465928,
                      'A09Pho_P01': +0.06770139351077731,
                      'A09Pho_P02': +0.1401981518283577,
                      'A09Pho_P03': +0.2175562018314121,
                      'A09Pho_P04': +0.29819712008410715,
                      'A09Pho_P05': +0.38236406391679,
                      'A09Pho_P06': +0.4692735380620521}

    # Z/X for the solar mixtures (ALEN = GN93 w. alpha enhancement)
    SUNzx = {'GN93': 0.0245, 'GS98': 0.0230, 'AS09': 0.01811, 'CA11': 0.0209,
             'RT16': 0.0245, 'ALEN': 0.0245}

    # dYdZ and Yp used for abundance calculation in BaSTI-isochrones
    dYdZ = {'v2018': 1.31,  'v2004': 1.31}
    Yp   = {'v2018': 0.247, 'v2004': 0.247}


def buldgen_freqs(m, obs, obskey, top, extend=False):
    """
    Reads frequencies of Buldgen models, and maps to observed frequencies
    """
    bobs = np.loadtxt(os.path.join(top, 'input/Model{0}.freq'.format(m)))
    l = bobs[:,0]; n = bobs[:,1]; f = bobs[:,2]; e = bobs[:,3]
    joins = freq_fit.calc_join(np.array([f, e], dtype=np.float),
                               np.array([l, n], dtype=np.int),
                               obs, obskey)
    joinkeys, join = joins
    if extend:
        joinkeys, join = su.extend_modjoin(joinkeys, 
                                           join, 
                                           np.array([l, n], dtype=np.int), 
                                           np.array([f, e], dtype=np.float))
    return joinkeys, join

def chi2_difference(obskey, obs, modkey, mod, tp=False, rtype='r012', corr=True, inc="", weight=True):
    """
    Calculates the chi2 difference between observations and model ratios given
    the fitting options.
    """
    obsratio, cov, covinv = freq_fit.compute_ratios(
        obskey, obs, rtype, threepoint=tp
    )
    if not corr:
        cov = np.identity(cov.shape[0]) * cov
        covinv = np.linalg.pinv(cov, rcond=1e-8)
    
    modratio = freq_fit.compute_ratioseqs(modkey, mod, rtype, threepoint=tp)
    x = obsratio[0, :] - modratio[0, :]
    if inc == "exclude":
        for i in range(len(obsratio[0,:])):
            if obsratio[2, i] == 1 and obsratio[3, i] == 21:
                x[i] = 0

    if weight:
        w = len(x)
    else:
        w = 1
    chi2 = (x.T.dot(covinv).dot(x))/w
    return chi2


def get_mins(select):
    """
    Extracts the chi2 value and model identifier of the highest
    likelihood model and the lowest chi2 model.
    """

    # Get highest likelihood model
    mins = []; mods = []
    pathmax = []
    for path, trackstats in select.items():
        i = np.argmax(trackstats.logPDF)
        PDF = float(trackstats.logPDF[i])
        chi = float(trackstats.chi2[i])
        ind = trackstats.index.nonzero()[0][i]
        if PDF != -np.inf:
            pathmax.append([path, PDF, chi, ind])
    pathmax = np.asarray(pathmax, dtype=object)
    mins.append(pathmax[np.argmax(pathmax[:,1]), 2])
    mods.append([pathmax[np.argmax(pathmax[:,1]), 0], pathmax[np.argmax(pathmax[:,1]), 3]])

    # Get lowest chi2 model
    pathmax = []
    for path, trackstats in select.items():
        i = np.argmin(trackstats.chi2)
        PDF = float(trackstats.logPDF[i])
        chi = float(trackstats.chi2[i])
        ind = trackstats.index.nonzero()[0][i]
        if PDF != -np.inf:
            pathmax.append([path, PDF, chi, ind])
    pathmax = np.asarray(pathmax, dtype=object)
    mins.append(pathmax[np.argmin(pathmax[:,2]), 2])
    mods.append([pathmax[np.argmin(pathmax[:,2]), 0], pathmax[np.argmin(pathmax[:,2]), 3]])
    return mins, mods


def get_nmax(select, n, pdf=True):
    """
    Extract the n highest-likelihood/lowest chi2 models/tracks
    """
    pathmax = []
    for path, trackstats in select.items():
        if pdf:
            i = np.argmax(trackstats.logPDF)
            val = float(trackstats.logPDF[i])
        else:
            i = np.argmin(trackstats.chi2)
            val = float(trackstats.chi2[i])
        ind = trackstats.index.nonzero()[0][i]
        if val != -np.inf:
            pathmax.append([path, ind, val])
    pathmax = np.asarray(pathmax, dtype = object)
    mask = np.argsort(pathmax[:,2])
    if pdf:
        pathmax = np.flipud(pathmax[mask][-n:])
    else:
        pathmax = pathmax[mask][:n]
    return pathmax


#############
# UNCHECKED #
#############


def read_obs(filename, maxl=2):
    """
    Read obs-file from ADIPLS. Can handle negative orders. Will only return
    modes where the radial order is l <= maxl .

    Parameters
    ----------
    filename : str
        Absolute path and name of the file to read
    maxl : int, optional
        Highest radial order to include (no larger than 3!)

    Returns
    -------
    l : array
        Angular degree of the modes
    n: array
        Radial order of the modes
    f : array
        Frequency of the modes
    e : array
        Inertia of the modes
    """
    assert 0 <= maxl < 4, "Invalid value of l!"

    with open(filename, 'r') as f:
        lines = f.readlines()

    values = np.zeros((len(lines), 4))
    for i in range(len(lines)):
        values[i, 0] = lines[i][0:5]
        values[i, 1] = lines[i][5:12]
        values[i, 2] = lines[i][12:22]
        values[i, 3] = lines[i][22:35]

    l, n, f, e = np.transpose(values)
    lmask = l.astype(int) <= maxl

    return l.astype(int)[lmask], n.astype(int)[lmask], f[lmask], e[lmask]


def obs_ratios(freqfile, remove=False, tp=False, rtype='R012'):
    obskey, obs, _ = read_freq(freqfile)
    if remove:
        for n in [24, 25]:
            mask = (obskey[0] == 2) & (obskey[1] == n)
            obskey = obskey[:, ~mask]
            obs = obs[:, ~mask]
    obsr, covr, _ = freq_fit.compute_ratios(obskey, obs, ratiotype=rtype.lower(), threepoint=tp)
    return obsr, covr


def obs_to_freqs(path, obs, obskey):
    l, n, f, e = read_obs(path)
    joins = freq_fit.calc_join(np.array([f, e], dtype=np.float), 
                               np.array([l, n], dtype=np.int),
                               obs, obskey)
    joinkeys, join = joins
    freqs = np.zeros(len(join[0]), dtype={'names': names[:-1], 'formats': fmts[:-1]})
    freqs[:]['l'] = joinkeys[0, joinkeys[0][:] < 3]
    freqs[:]['n'] = joinkeys[1, joinkeys[0][:] < 3]
    freqs[:]['freq'] = join[0, joinkeys[0][:] < 3]
    return freqs

def convert_obs_freqs(obs, obskey):
    freqs = np.zeros(len(obs[0]), dtype={'names': names[:-1], 'formats': fmts[:-1]})
    freqs[:]['l'] = obskey[0]
    freqs[:]['n'] = obskey[1]
    freqs[:]['freq'] = obs[0]
    return freqs

def grid_freqs(grid, track, index, obs, obskey):
    osc = su.transform_obj_array(grid[track + '/osc'][index])
    osckey = su.transform_obj_array(grid[track + '/osckey'][index])

    joinkeys, join = freq_fit.calc_join(osc, osckey, obs, obskey)
    newfreqs = np.zeros(len(join[0]), dtype={'names': names[:-1], 'formats': fmts[:-1]})
    newfreqs[:]['l'] = joinkeys[0, joinkeys[0][:] < 3]
    newfreqs[:]['n'] = joinkeys[1, joinkeys[0][:] < 3]
    newfreqs[:]['freq'] = join[0, joinkeys[0][:] < 3]
    return newfreqs

def buldgen_log(m, pars):
    dat = np.loadtxt(os.path.join(top, 'input/Model{0}.log'.format(m)), comments='#')
    names = {'mod': 0, 'age': 1, 'Teff': 2, 'L': 3, 'Xc': 4, 'logg': 13, 
             'ZXs': 14, 'RConv': 16, 'Zs': 17, 'Xs': 18}
    dat[:,2] = 10**dat[:,2]
    dat[:,3] = 10**dat[:,3]
    columns = [names[p] for p in pars]
    if len(columns) == 1:
        return dat[:, columns][0]
    elif len(columns) > 1:
        return dat[:, columns]
    else:
        return dat


def determine_core_type(grid, interval = [2,9], tol=0.01, err=1e-4):
    core_type = {}; bp = 'grid/'
    for group, tracks in grid[bp].items():
        for name, libitem in tracks.items():
            age = np.asarray(libitem['age'])*1e-3
            mcore = np.asarray(libitem['Mcore'])
            m1 = mcore[np.argmin(abs(age-interval[0]))]
            m2 = mcore[np.argmin(abs(age-interval[1]))]
            if m1 < tol:
                ctyp = 1
            elif m1 > tol and m2 < err:
                ctyp = 2
            elif m2 > err:
                ctyp = 3
            else:
                raise ValueError("Error in determine_core_type")
            core_type[os.path.join(bp, group, name)] = ctyp
    return core_type

def determine_core_lifetime(grid, tol=1e-4, noffset=0):
    core_lifetime = {}; bp = 'grid/'
    for group, tracks in grid[bp].items():
        for name, libitem in tracks.items():
            age = np.asarray(libitem['age'])*1e-3
            mcore = np.asarray(libitem['Mcore'])
            rev_mcore = mcore[::-1]
            ind = len(mcore) - np.argmax(rev_mcore > tol) - 1
            lt = age[ind]
            core_lifetime[os.path.join(bp, group, name)] = lt
    return core_lifetime

def FeH_from_ZX(ZX, m):
    buldalpha = {13: "A09Pho_P02", 12: "A09Pho_P03"}
    ZXsun = bc.mixtures.SUNzx["AS09"]
    MeH = np.log10(ZX/ZXsun)
    FeH = MeH - bc.mixtures.M_H_Corr_Alpha[buldalpha[m]]
    return FeH

def posterior_core_type(selmod, core_types):
    logy = np.concatenate([ts.logPDF for ts in selmod.values()])
    noofind = len(logy)
    nonzero = np.isfinite(logy)
    logy = logy[nonzero]
    lk = logy - np.amax(logy)
    pd = np.exp(lk - np.log(np.sum(np.exp(lk))))

    x = np.concatenate([np.ones(len(t.logPDF))*core_types[p] for p,t in selmod.items()])[nonzero]

    bins = [0.5, 1.5, 2.5, 3.5]
    hist, bins = np.histogram(x, bins=bins, weights=pd)
    ctype = np.argmax(hist) + 1
    return ctype

def posterior_core_lifet(selmod, core_lifet):
    logy = np.concatenate([ts.logPDF for ts in selmod.values()])
    noofind = len(logy)
    nonzero = np.isfinite(logy)
    logy = logy[nonzero]
    lk = logy - np.amax(logy)
    pd = np.exp(lk - np.log(np.sum(np.exp(lk))))
    sampled_indices = np.random.choice(np.arange(len(pd)), p=pd, size=noofind)

    x = np.concatenate([np.ones(len(t.logPDF))*core_lifet[p] for p,t in selmod.items()])
    x = x[nonzero][sampled_indices]

    qs = [0.5, 0.158655, 0.841345]
    out = quantile_1D(x, pd, qs)
    # Transform from quantiles to 68% Bayesian interval
    bayfit = [out[0], out[0]-out[1], out[2]-out[0]]

    return bayfit
