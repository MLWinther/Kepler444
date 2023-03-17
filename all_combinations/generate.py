from __future__ import print_function, with_statement, division
from basta.xml_create import generate_xml

# Generate the input XML file from an ascii file

# Name of grid input file
grid = '/home/au540782/Kepler444/input/Garstec_AS09_Kepler444_diff.hdf5'
# Solar model contained in grid
solarmodel = True
# Name of outputfile
outputfile = 'result.ascii'
# Indicator for missing value, default value
missingval = -999.999
# Frequency input files
freqpath = '/home/au540782/Kepler444/input/'
# Parameters to be outputted
outparams = ('Teff', 'FeH', 'MeH', 'LPhot', 'rho', 'age', 'massfin', 'agecore', 'gcut', 'ove', 'alphaMLT', 'distance')
# Plots/diagrams
cornerplots = ('age', 'massfin', 'ove', 'gcut', 'agecore', 'distance')
freqplots = True
kielplots = True
dustframe = 'icrs'
filters = ('Mb_TYCHO', 'Mv_TYCHO')
# Parameters ordered as in the input file
inputparams = ('starid', 'Teff', 'Teff_err', 'logg', 'logg_err',
               'FeH', 'FeH_err', 'LPhot', 'LPhot_err', 'rho', 
               'rho_err', 'dnu', 'dnu_err', 'numax', 'numax_err', 
               'parallax', 'parallax_err','Mb_TYCHO', 
               'Mb_TYCHO_err', 'Mv_TYCHO', 'Mv_TYCHO_err',
               'RA', 'RA_err', 'DEC', 'DEC_err', 'MeH', 'MeH_err',
               'LPhot', 'LPhot_err')
# Priors of the probability computation
priors = {'IMF': 'IMF', "dnufit": {"min": "175", "max": "185"} }
# Frequency fitting parameters
fcor = 'BG14'
bexp = 4.5
dnufrac = 0.50
notrust = {"Kepler444": '/home/au540782/Kepler444/input/nottrusted_Kepler444.fre'}

# Name of generated XML input file
xmlfile = 'input_{:03d}.xml'
# Name of ascii input file
asciifile = '/home/au540782/Kepler444/input/Kepler444.ascii'

fpbase = {'freqpath': freqpath, 'fcor': fcor, 'bexp': bexp,
          'dnufrac': dnufrac, "intpol_ratios": "False"}

n = 1

fitpar0 = ('FeH', 'Teff')
outdir0 = 'output_{:03d}_diffusion_' 

for ratio in ['r01', 'r02', 'r012']:
    outdir1 = outdir0 + '_'.join(fitpar0) + '_'+ ratio + '_'
    fitpar1 = fitpar0 + (ratio,)
    for covariance in [False, True]:
        corr = 'corr_' if covariance else 'uncorr_'
        outdir2 = outdir1 + corr
        for remove in [False, True]:
            trust = 'nottrusted_' if remove else 'trusted_'
            outdir3 = outdir2 + trust
            for threepoint in [True, False]:
                point = '3p' if threepoint else '5p'
                outdir4 = outdir3 + point

                if not threepoint and '1' not in ratio:
                    continue
                if remove and '2' not in ratio:
                    continue
                
                fp = {'correlations': str(covariance),
                      'threepoint': str(threepoint)}
                if remove:
                    fp['nottrustedfile'] = notrust
                
                outputpath = outdir4.format(n)
                xml = generate_xml(grid, asciifile, outputpath, 
                                 inputparams, fitpar1, outparams,
                                 solarmodel=solarmodel,
                                 outputfile=outputfile, priors=priors,
                                 cornerplots=fitpar0+cornerplots, freqplots=freqplots,
                                 freqparams=dict(fp, **fpbase), kielplots=kielplots,
                                 dustframe=dustframe, filters=filters,
                                 plotfmt='pdf')
                with open(xmlfile.format(n), 'w') as inpfile:
                        print(xml, file=inpfile)

                n += 1
                print(outputpath)


freqpars = [dict({'correlations': "False", 'threepoint': "True"}, **fpbase),
            dict({'correlations': "True", 'threepoint': "False", 'nottrustedfile': notrust}, **fpbase)]
freqlabs = ['_uncorr_trusted_3p', '_corr_nottrusted_5p']

fitsets = [('LPhot',), 
           ('logg',), 
           ('rho',),
           ('LPhot', 'logg',), 
           ('LPhot', 'rho',), 
           ('logg', 'rho',), 
           ('LPhot', 'logg', 'rho',),
           ('LPhot', 'logg', 'rho', 'r02'),
           ('parallax',)
          ]

for fitset in fitsets:
    fitparams = fitpar0 + fitset
    if 'r02' not in fitparams:
        fitparams += ('r012',) 
    for fpars, flabel in zip(freqpars, freqlabs):
        outdir1 = outdir0 + '_'.join(fitparams) + flabel
        
        outputpath = outdir1.format(n)
        xml = generate_xml(grid, asciifile, outputpath, 
                         inputparams, fitparams, outparams,
                         solarmodel=solarmodel,
                         outputfile=outputfile, priors=priors,
                         cornerplots=fitparams+cornerplots, freqplots=freqplots,
                         freqparams=fpars, kielplots=kielplots,
                         dustframe=dustframe, filters=filters,
                         plotfmt='pdf')
        with open(xmlfile.format(n), 'w') as inpfile:
                print(xml, file=inpfile)

        n += 1
        print(outputpath)
