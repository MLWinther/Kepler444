from __future__ import print_function, with_statement, division
import os
import glob
import json
import h5py

import numpy as np
from xml.etree import ElementTree

import basta.constants as bc
import basta.freq_fit as freq_fit
import basta.fileio as fio
import basta.utils_seismic as su

# MANY functions in this
import helpers

# Compare BASTA and Buldgen fits

def FeH_from_ZX(ZX, m):
    buldalpha = {'13': "A09Pho_P02", '12': "A09Pho_P03"}
    ZXsun = helpers.mixtures.SUNzx["AS09"]
    MeH = np.log10(ZX/ZXsun)
    FeH = MeH - helpers.mixtures.M_H_Corr_Alpha[buldalpha[m]]
    return FeH, MeH

def chi2(modv, obs):
    chi2 = ((modv - obs[0])/obs[1])**2.
    #print(modv, obs[0], obs[1], chi2)
    return chi2

def compute_bchi2(top, frefile, notfile):
    # To not compute it every time, first determine buldgen chi2 of ratios
    bchi2 = {"freq": {"13": {}, "12": {}}, "spec": {}}
    incmodes = {0: [[20, 1.5], [21, 2], [22, 1.5]], 1: [[20, 1.5], [21, 2]]}
    for inc in ["normal", "exclude"]:#, "increased"]:
        for trust in ["trusted", "nottrusted"]:
            if "not" in trust:
                obskey, obs, _ = fio.read_freq(frefile, nottrustedfile=notfile)
            else:
                obskey, obs, _ = fio.read_freq(frefile)
            if inc == "increased":
                for l in incmodes.keys():
                    for (n,mult) in incmodes[l]:
                        lmask = obskey[0, :] == l
                        nmask = obskey[1, :] == n
                        lmask &= nmask
                        ind = np.where(lmask)[0][0]
                        obs[1, ind] *= mult

            for bmod in ["12", "13"]:
                modkey, mod = helpers.buldgen_freqs(bmod, obs, obskey, top)
                for ratio in ["r01", "r02", "r012"]:
                    for corr in ["Corr", "Uncorr"]:
                        for points in ["3p", "5p"]:
                            if "1" not in ratio and points=="5p":
                                continue
                            if "2" not in ratio and "not" in trust:
                                continue
                            fmode = "_".join([ratio, corr, trust, points, inc])

                            tp = True if "3" in points else False
                            
                            chi2rval = helpers.chi2_difference(
                                            obskey,
                                            obs,
                                            modkey,
                                            mod, 
                                            tp=tp,
                                            rtype=ratio,
                                            corr=False if "u" in corr.lower() else True,
                                            inc=inc,
                                            weight=True)

                            bchi2["freq"][bmod][fmode] = chi2rval

    # Package Buldgen spectroscopic info
    for bmod in ["12", "13"]:
        logdat = helpers.buldgen_log(bmod, 
                 ['Teff', 'Zs', 'Xs', 'ZXs', 'L', 'logg'])[-1,:]
        FeH, MeH = FeH_from_ZX(logdat[3], bmod)
        bspec = {
            "Teff": logdat[0], 
            "LPhot": logdat[4], 
            "FeH": FeH, 
            "MeH": MeH, 
            "logg": logdat[5], 
            "rho": 2.4925 if bmod == "12" else 2.4933}
        bchi2["spec"][bmod] = bspec

    s = json.dumps(bchi2)
    with open(Buldjson, 'w') as f:
        f.write(s)

    return bchi2
 

top = '/home/au540782/Kepler444/'
frefile = os.path.join(top, 'input/Kepler444_Campante.xml')

specobs = {"FeH": [-0.52, 0.12], "Teff": [5172, 75], "LPhot": [0.4, 0.04], 
           "logg": [4.595, 0.06], "rho": [2.496, 0.012], "MeH": [-0.37, 0.09]}

# Frequency fitting parameters
notfile = os.path.join(top, 'input/nottrusted_Kepler444.fre')
rtypes = ["r01", "r02", "r012"]

# Name of generated XML input file
xmlfile = 'input_{:03d}.xml'
# Name of ascii input file
asciifile = '/home/au540782/Kepler444/input/Kepler444_{0}.ascii'

Buldjson = "Buldchi2.json"
precomputed = False


np.random.seed(444)
if not precomputed:
    bchi2 = compute_bchi2(top, frefile, notfile)
else:
    with open(Buldjson, 'r') as f:
        bchi2 = json.load(f)


# Print of Buldgen ratio chi2 values
n = 1
for rt in rtypes:
    for corr in ["Uncorr", "Corr"]:
        for trust in ["trusted", "nottrusted"]:
            for points in ["3p", "5p"]:
                if "1" not in rt and points == "5p":
                    continue
                if "2" not in rt and "not" in trust:
                    continue
                fmode = "_".join([rt, corr, trust, points, "normal"])
                
                b12 = bchi2["freq"]["12"][fmode]
                b13 = bchi2["freq"]["13"][fmode]
                
                wrstr = "{:2d} & ".format(n)
                n += 1

                if b12 < b13:
                    wrstr += "\\textbf{" + "{:.3f}".format(b12) + "}"
                    wrstr += " & {:.3f}".format(b13)
                else:
                    wrstr += "{:.3f} & ".format(b12)
                    wrstr += "\\textbf{" + "{:.3f}".format(b13) + "}"

                wrstr += " & "

                # Ratio type
                wrstr += "$r_{" + rt[1:] + "}$ & "

                # N-point 
                if rt == "r02":
                    wrstr += "& "
                elif points == "3p":
                    wrstr += "$3p$ & "
                else:
                    wrstr += "$5p$ & "

                # Removed modes
                if "not" in trust:
                    wrstr += "\\checkmark & "
                elif "2" in rt:
                    wrstr += "\\transparent{0.4}x & "
                else:
                    wrstr += " & "

                # Covariance
                if "U" not in corr:
                    wrstr += "\\checkmark "
                else:
                    wrstr += "\\transparent{0.4}x "
                
                # wrstr += " & "

                # fmode = "_".join([rt, corr, trust, points, "increased"])
                
                # b12 = bchi2["freq"]["12"][fmode]
                # b13 = bchi2["freq"]["13"][fmode]
                
                # if b12 < b13:
                #     wrstr += "\\textbf{" + "{:.3f}".format(b12) + "}"
                #     wrstr += " & {:.3f}".format(b13)
                # else:
                #     wrstr += "{:.3f} & ".format(b12)
                #     wrstr += "\\textbf{" + "{:.3f}".format(b13) + "}"

                wrstr += " & "

                fmode = "_".join([rt, corr, trust, points, "exclude"])
                
                b12 = bchi2["freq"]["12"][fmode]
                b13 = bchi2["freq"]["13"][fmode]
                
                if b12 < b13:
                    wrstr += "\\textbf{" + "{:.3f}".format(b12) + "}"
                    wrstr += " & {:.3f}".format(b13)
                else:
                    wrstr += "{:.3f} & ".format(b12)
                    wrstr += "\\textbf{" + "{:.3f}".format(b13) + "}"

                wrstr += "\\\\"

                print(wrstr)


    
# Define grids and determine core types of tracks
grids = [h5py.File(os.path.join(top, "input/Garstec_AS09_Kepler444_nodiff.hdf5"), 'r'),
         h5py.File(os.path.join(top, "input/Garstec_AS09_Kepler444_diff.hdf5"), 'r')]
         #h5py.File(os.path.join(top, "input/Garstec_AS09_Kepler444.hdf5"), 'r')]
#core_types = [determine_core_type(grids[0], interval=[1,9]),
#              determine_core_type(grids[1], interval=[1,9])]
#core_lifet = [determine_core_lifetime(grids[0]),
#              determine_core_lifetime(grids[1])]


path = "all_combinations"
outdirs = glob.glob(os.path.join(top, path, "output*"))
outdirs = np.sort(outdirs)

classpars = {"logg": "$\\log g$", "LPhot": "\\lphot", 
            "FeH": "\\feh", "MeH": "\\meh", "Teff": "\\teff", 
            "parallax": "$\\varpi$", "rho": "$\\rho$"}
fmt = "{{0:{0}}}".format(".3f").format
baystr = "${{{0}}}_{{-{1}}}^{{+{2}}}$ & "

lifetimemode = True

outfile = open("comparison.txt", 'w')

recompute = False

for outdir in outdirs:
    fitparams = ()
    
    # Extract number and keys for fit from directory name
    n = int(outdir.split("/")[-1].split("_")[1])
    keys = [k.lower() for k in outdir.split("/")[-1].split("_")[2:]]
    
    # Extract classical parameters
    for key in classpars.keys():#, "r01", "r02", "r012", "parallax"]:
        if key.lower() in keys and key != "parallax":
            fitparams += (key,)
    
    # Extract ratio fit type
    for rt in rtypes:
        if rt in keys:
            rtype = rt
    
    # Extract 3 or 5 point
    threepoint = True if "3p" in keys else False

    # Correlations
    uncorr = True if "uncorr" in keys else False

    # Removed mode
    rmode = True if "nottrusted" in keys else False
    
    # Parallax
    parallax = True if "parallax" in keys else False

    # Diffusion and thereby grid to work with
    diff = True if "diffusion" in keys else False
    if diff:
        grid = grids[1]
        #ct = core_types[1]
        #lft = core_lifet[1]
    else:
        grid = grids[0]
        #ct = core_types[0]
        #lft = core_lifet[0]
    

    # Get fit data
    bastajson = os.path.join(outdir, "Kepler444.json")
    bastaxml  = os.path.join(outdir, "result.xml")
    tree = ElementTree.parse(bastaxml)
    root = tree.getroot()
    agecore = root.find("star").find("agecore")
    selected = fio.load_selectedmodels(bastajson)
    # Highest likelihood and lowest chi2 models
    bastachi, bastamods = helpers.get_mins(selected)
    if recompute:
        bastanewchi = []
        for j, (track, ind) in enumerate(bastamods):
            if rmode:
                obskey, obs, _ = fio.read_freq(frefile, nottrustedfile=notfile)
            else:
                obskey, obs, _ = fio.read_freq(frefile)
            modkey = su.transform_obj_array(grid[track]["osckey"][ind])
            mod    = su.transform_obj_array(grid[track]["osc"][ind])
            modkey, mod = freq_fit.calc_join(mod, modkey, obs, obskey)
            chi2rval = helpers.chi2_difference(
                            obskey,
                            obs,
                            modkey,
                            mod, 
                            tp=threepoint,
                            rtype=rtype,
                            corr=not uncorr,
                            inc=False)
            bastanewchi.append(chi2rval)
            for key in fitparams:
                modv = grid[track][key][ind]
                bastanewchi[j] += chi2(modv, specobs[key])

        #print(bastachi)
        #print(bastanewchi)

    
    if rtype == "r02":
        threepoint = True
    fmode = "_".join([rtype,
                      "Uncorr" if uncorr else "Corr",
                      "nottrusted" if rmode else "trusted",
                      "3p" if threepoint else "5p",
                      "normal"])
    
    
    # Determine Buldgen chi2 for this combination
    sumbchi2 = []
    for bmod in ["12", "13"]:
        spechi2 = sum([chi2(bchi2["spec"][bmod][o], specobs[o]) for o in fitparams])
        sumbchi2.append(bchi2["freq"][bmod][fmode] + spechi2)
    
    # List of all chi2s
    chi2s = [bastachi[0], bastachi[1], sumbchi2[0], sumbchi2[1]]
    
    if parallax:
        chi2l = chi2s[:2]
        indmin = [3]
    else:
        chi2l = chi2s
        indmin = np.where(np.asarray(chi2l) == np.asarray(chi2l[-2:]).min())[0]

    wrstr = "{:2d}".format(n) + " & "
    
    # Chi2 values
    for i, chi in enumerate(chi2s):
        if i > 1 and parallax:
            wrstr += " & "
            continue
        if i in indmin and False:
            wrstr += "\\textbf{" + "{:.3f}".format(chi) + "} & "
        else:
            wrstr += "{:.3f} & ".format(chi)
    
    # Basta min model lifetime
    if False:
        lft = grid[os.path.join(bastamods[0][0], 'agecore')][bastamods[0][1]]
        wrstr += "{:.3f} & ".format(lft*1e-3)

    # Inferred lifetime
    if lifetimemode:
        #baylft = posterior_core_lifet(selected, lft)
        baylft = [agecore.get("value"), agecore.get("error_minus"), agecore.get("error_plus")]
        baylft = [float(v) for v in baylft]
        wrstr += baystr.format(fmt(baylft[0]), fmt(baylft[1]), fmt(baylft[2]))
    
    # Classical parameters
    pars = []
    for par in classpars.keys():
        if par.lower() in keys and par.lower() not in ["feh", "teff"]:
            pars.append(classpars[par])
    wrstr += ",\\,".join(pars) + " & "

    # Ratio type
    wrstr += "$r_{" + rtype[1:] + "}$ & "
    
    # N-point 
    if rtype == "r02":
        wrstr += "& "
    elif threepoint:
        wrstr += "$3p$ & "
    else:
        wrstr += "$5p$ & "
    
    # Removed modes
    if rmode:
        wrstr += "\\checkmark & "
    elif "2" not in rtype:
        wrstr += " & "
    else:
        wrstr += "\\transparent{0.4}x & "
    
    # Whether to exclude diffusion in table
    if True:
        # Covariance
        if not uncorr:
            wrstr += "\\checkmark \\ts\\\\"
        else:
            wrstr += "\\transparent{0.4}x \\ts\\\\"
    else:
        # Covariance
        if not uncorr:
            wrstr += "\\checkmark & "
        else:
            wrstr += "\\transparent{0.4}x & "

        # Diffusion
        if diff:
            wrstr += "\\checkmark \\ts\\\\"
        else:
            wrstr += "\\transparent{0.4}x \\ts\\\\"
    
    #if not parallax:
    print(wrstr)

    outfile.write(outdir.split("/")[-1] + "\n")
    outfile.write(wrstr + "\n\n")

outfile.close()

