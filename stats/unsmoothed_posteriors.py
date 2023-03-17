import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import basta.fileio as fio

from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter


def main():
    fits = [16, 96, 256]
    top = '/home/au540782/Kepler444/'
    outdirs = glob.glob(os.path.join(top, "all_combinations/output*"))
    outdirs = np.sort(outdirs)[[i-1 for i in fits]]
    histargs = {"color": "tab:cyan", "alpha": 0.17}
    histargs_line = {"color": "tab:orange", "alpha": 0.7}
    histargs_fill = {"color": "tab:orange", "alpha": 0.1}
    
    grids = [os.path.join(top, "input/Garstec_AS09_Kepler444_nodiff.hdf5"),
             os.path.join(top, "input/Garstec_AS09_Kepler444_diff.hdf5")]
    cols = ['#332288', '#88CCEE', '#CC6677', '#117733', '#999933', '#DDCC77']
    N = 5
    parameters = ["age", "Teff", "FeH", "massfin", "rho", "agecore"]

    for n, outdir in zip(fits,outdirs):
        fig, ax = plt.subplots(3, len(parameters), figsize=(20,8.27))
        for i, param in enumerate(parameters):
            vals1, loglike1, chi21 = extract_from_json(os.path.join(outdir, "Kepler444.json"),
                                                    grids[(n // 80) % 2], param, highest=True)
            pd = np.exp(loglike1 - np.log(np.sum(np.exp(loglike1))))
            ax[0,i].plot(vals1, pd, '.')

            vals2, loglike2, chi22 = extract_from_json(os.path.join(outdir, "Kepler444.json"),
                                                    grids[(n // 80) % 2], param, highest=N)
            for j, (val, like) in enumerate(zip(vals2, loglike2)):
                ax[1,i].plot(val, np.exp(like), '-', color=cols[j], label=str(j+1), zorder=5)

            vals, loglike, chi2 = extract_from_json(os.path.join(outdir, "Kepler444.json"),
                                                    grids[(n // 80) % 2], param, highest=False)
            samples = sample_posterior(vals=vals, logys=loglike)
            counts, bins = np.histogram(a=vals, bins=100, density=True, weights=np.exp(loglike))
            counts = gaussian_filter(input=counts, sigma=1)
            kernel = gaussian_kde(samples, bw_method="silverman")
            x0 = np.linspace(np.amin(samples), np.amax(samples), num=500)
            y0 = kernel(x0)
            y0 /= np.amax(y0)
            x0_hist = np.array(list(zip(bins[:-1], bins[1:]))).flatten()
            y0_hist = np.array(list(zip(counts, counts))).flatten() / np.amax(counts)
            ax[2,i].fill_between(
                x0_hist, y0_hist, interpolate=True, label="Posterior histogram", **histargs
            )
            ax[2,i].plot(x0, y0, label="Posterior KDE", **histargs_line)
            ax[2,i].fill_between(x0, y0, interpolate=True, **histargs_fill)
            ax[2,i].set_xlabel(param)
            
            mm = [np.min(vals1), np.max(vals1)]
            for j, a in enumerate(ax[:,i]):
                bot,top = a.get_ylim()
                sort = np.argsort(loglike1)[::-1]
                for k in range(N):
                    a.plot(np.ones(2)*vals1[sort][k], [bot,top], '--k', alpha=0.3)
                a.set_xlim(mm)
                if top - bot > 1.2*bot:
                    a.set_ylim([0, top])
                else:
                    a.set_ylim([bot, top])
                if j == 0:
                    nvals = vals1[sort][:N]
                    order = np.argsort(nvals)
                    a.set_title(', '.join([str(o+1) for o in order]))
        
        for i in range(2):
            for a in ax[i,:]:
                a.set_xticks([])
        for i in range(len(parameters)-1):
            for a in ax[:,-i-1]:
                a.set_yticks([])
        ax[0,0].set_ylabel("Highest model in track")
        ax[1,0].set_ylabel("Likelihood")
        ax[2,0].set_ylabel("Posterior")
        ax[1,-1].legend(bbox_to_anchor=(1.01,1.0), loc="upper left")
        fig.suptitle("Order of {0} highest likelihood tracks".format(N))
        fig.tight_layout()
        fig.savefig("plots/{:03d}_posteriors.pdf".format(n), dpi=300)
        plt.close()

def sample_posterior(vals, logys):
    """
    Computation of the posterior of a parameter by numerical sampling. Duplication of
    how it is performed in basta/process_output.py.

    Parameters
    ----------
    vals : array_like
        Parameter values

    logys : array_like
        Corresponding log(likelihood)'s

    Returns
    ------
    samples : array_like
        Samples of the posterior. Can be used to make the posterior distribution.
    """
    lk = logys - np.amax(logys)
    post = np.exp(lk - np.log(np.sum(np.exp(lk))))
    nsamples = min(1e8, len(logys))
    sampled_indices = np.random.choice(a=np.arange(len(post)), p=post, size=nsamples)
    return vals[sampled_indices]


def extract_from_json(jsonfile, gridfile, parameter, highest=False):
    """
    Extract information from a BASTA json file.
    During a fit, if "optionaloutputs" is True, a dump of the statistics for each star
    will be stored in a .json file. The grid used to perform the fit is required to
    obtain any useful information from the json-dump.
    Parameters
    ----------
    jsonfile : str
        Path to the .json file from the BASTA fit
    gridfile : str
        Path to the grid used to compute the fit
    parameter : str
        The parameter values to extract (must be a valid name!)
    Returns
    -------
    parametervals : array_like
        Values of the given parameter for all models (with non-zero likelihood) in the
        grid
    loglikes : array_like
        The log(likelihood) of all models in the grid, excluding zero-likelihood models
    chi2 : array_like
        The chi**2 values of all models (with non-zero likelihood) in the grid
    """
    # The json file can be read into selectedmodels (in BASTA lingo), which contains
    # chi2 and log-likelihood (called logPDF internally) of all models in the grid. It
    # is stored by track (see below).
    selectedmodels = fio.load_selectedmodels(jsonfile)

    # Typically, we want to extract the likelihoods and the values of a given parameter
    # for all models in the grid/fit
    with h5py.File(gridfile, "r") as grid:
        parametervals = []
        loglikes = []
        chi2s = []

        # In selectedmodels, the information is accessed by looping over the tracks
        for trackpath, trackstats in sorted(selectedmodels.items()):
            gridval = grid["{0}/{1}".format(trackpath, parameter)]
            
            # The log-likelihood (called logPDF) and chi2 can be stored directly
            if highest and type(highest)==bool:
                ind = np.argmax(trackstats.logPDF)
                loglikes.append(trackstats.logPDF[ind])
                chi2s.append(trackstats.chi2[ind])
                parametervals.append(gridval[trackstats.index][ind])
            else:
                loglikes.append(trackstats.logPDF)
                chi2s.append(trackstats.chi2)
                parametervals.append(gridval[trackstats.index])

        # After extraction it is useful to collapse the lists into numpy arrays
        if highest and type(highest)==bool:
            parametervals = np.array(parametervals)
            loglikes = np.array(loglikes)
            chi2s = np.array(chi2s)
        elif highest:
            sort = np.argsort(np.array([np.max(likes) for likes in loglikes]))[::-1][:highest]
            parametervals = [parametervals[s] for s in sort]
            loglikes = [loglikes[s] for s in sort]
            chi2s = [chi2s[s] for s in sort]
            
        else:
            parametervals = np.concatenate(parametervals)
            loglikes = np.concatenate(loglikes)
            chi2s = np.concatenate(chi2s)


    return parametervals, loglikes, chi2s

if __name__ == "__main__":
    main()
