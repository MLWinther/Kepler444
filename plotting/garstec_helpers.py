import warnings
import numpy as np

def read_garlog(filename, ver, *arguments):
    """
    Read the .log file produced by GARSTEC

    Parameters
    ----------
    filename : str
        Absolute path and name of the .log file to read
    arguments : str
        Any number of strings to specify which columns to read. If none
        are given the entire logfile will be read

    Returns
    -------
    data : array
        Array containing the colums specified in ``arguments``
    """
    # Definition of column widths
    if ver == "15":
        logformat = [
            6,
            16,
            8,
            2,
            7,
            7,
            9,
            7,
            7,
            7,
            8,
            8,
            6,
            6,
            8,
            8,
            8,
            6,
            5,
            4,
            8,
            8,
            8,
            6,
            2,
        ]
    elif ver == "13":
        logformat = [
            6,
            16,
            8,
            2,
            7,
            7,
            8,
            7,
            7,
            7,
            8,
            6,
            6,
            6,
            8,
            8,
            7,
            6,
            5,
            4,
            9,
            8,
            8,
            6,
            2,
        ]
    else:
        print("Wrong Garstec-logformat chosen!")
        exit(2)

    # Selections of colums
    if arguments:
        columns = []

        # Gartec 13 ?
        if ver == "13":
            for opt in arguments:
                if opt.lower() == "mod":
                    columns.append(0)
                elif (opt.lower() == "age") or (opt.lower() == "age6"):
                    columns.append(1)
                elif (opt.lower() == "dty") or (opt.lower() == "dty3"):
                    columns.append(2)
                elif opt.lower() == "??":
                    columns.append(3)
                elif opt.lower() == "mtot":
                    columns.append(4)
                elif (opt.lower() == "dm") or (opt.lower() == "dm/dt"):
                    columns.append(5)
                elif (opt.lower() == "l") or (opt.lower() == "lgl"):
                    columns.append(6)
                elif (opt.lower() == "lh") or (opt.lower() == "lglh"):
                    columns.append(7)
                elif (opt.lower() == "lhe") or (opt.lower() == "lglhe"):
                    columns.append(8)
                elif (opt.lower() == "lnu") or (opt.lower() == "lglnu"):
                    columns.append(9)
                elif (opt.lower() == "te") or (opt.lower() == "lgte"):
                    columns.append(10)
                elif (opt.lower() == "pc") or (opt.lower() == "lgpc"):
                    columns.append(11)
                elif (opt.lower() == "tc") or (opt.lower() == "lgtc"):
                    columns.append(12)
                elif (opt.lower() == "rhc") or (opt.lower() == "lgrhc"):
                    columns.append(13)
                elif (opt.lower() == "xc") or (opt.lower() == "xc/mh"):
                    columns.append(14)
                elif (opt.lower() == "yc") or (opt.lower() == "yc/mhe"):
                    columns.append(15)
                elif opt.lower() == "mcnvc":
                    columns.append(16)
                elif (opt.lower() == "y") or (opt.lower() == "y(1)"):
                    columns.append(17)
                elif opt.lower() == "m":
                    columns.append(18)
                elif opt.lower() == "it":
                    columns.append(19)
                elif (opt.lower() == "eg") or (opt.lower() == "lgeg+"):
                    columns.append(20)
                elif opt.lower() == "fhydm":
                    columns.append(21)
                elif (opt.lower() == "tau") or (opt.lower() == "tau_f"):
                    columns.append(22)
                elif opt.lower() == "gmax":
                    columns.append(23)
                else:
                    print("Bad argument: %s" % opt)

        # .. or Garstec 15 ?
        else:
            for opt in arguments:
                if opt.lower() == "mod":
                    columns.append(0)
                elif (opt.lower() == "age") or (opt.lower() == "age6"):
                    columns.append(1)
                elif (opt.lower() == "dty") or (opt.lower() == "dty3"):
                    columns.append(2)
                elif opt.lower() == "??":
                    columns.append(3)
                elif opt.lower() == "mtot":
                    columns.append(4)
                elif (opt.lower() == "dm") or (opt.lower() == "dm/dt"):
                    columns.append(5)
                elif (opt.lower() == "l") or (opt.lower() == "lgl"):
                    columns.append(6)
                elif (opt.lower() == "lh") or (opt.lower() == "lglh"):
                    columns.append(7)
                elif (opt.lower() == "lhe") or (opt.lower() == "lglhe"):
                    columns.append(8)
                elif (opt.lower() == "lnu") or (opt.lower() == "lglnu"):
                    columns.append(9)
                elif (opt.lower() == "te") or (opt.lower() == "lgte"):
                    columns.append(10)
                elif (opt.lower() == "r") or (opt.lower() == "r/rs"):
                    columns.append(11)
                elif (opt.lower() == "tc") or (opt.lower() == "lgtc"):
                    columns.append(12)
                elif (opt.lower() == "rhc") or (opt.lower() == "lgrhc"):
                    columns.append(13)
                elif (opt.lower() == "xc") or (opt.lower() == "xc/mh"):
                    columns.append(14)
                elif (opt.lower() == "yc") or (opt.lower() == "yc/mhe"):
                    columns.append(15)
                elif opt.lower() == "mcnvc":
                    columns.append(16)
                elif (opt.lower() == "y") or (opt.lower() == "y(1)"):
                    columns.append(17)
                elif opt.lower() == "m":
                    columns.append(18)
                elif opt.lower() == "it":
                    columns.append(19)
                elif (opt.lower() == "eg") or (opt.lower() == "lgeg+"):
                    columns.append(20)
                elif opt.lower() == "fhydm":
                    columns.append(21)
                elif (opt.lower() == "tau") or (opt.lower() == "tau_f"):
                    columns.append(22)
                elif opt.lower() == "gmax":
                    columns.append(23)
                else:
                    print("Bad argument: %s" % opt)
    else:
        columns = None

    # Read the logfile
    data = np.genfromtxt(
        filename, comments="#", delimiter=logformat, usecols=columns, encoding=None
    )

    # Return
    return np.transpose(data)

def read_fgong(filename):
    """
    Read in an FGONG file.
    This function can read in FONG versions 250 and 300.
    The output gives the center first, surface last.

    Parameters
    ----------
    filename : str
        Absolute path an name of the FGONG file to read

    Returns
    -------
    starg : dict
        Global parameters of the model
    starl : dict
        Local parameters of the model
    """
    # Standard definitions and units for FGONG
    glob_pars = [
        ("mass", "g"),
        ("Rphot", "cm"),
        ("Lphot", "erg/s"),
        ("Zini", None),
        ("Xini", None),
        ("alphaMLT", None),
        ("phi", None),
        ("xi", None),
        ("beta", None),
        ("dp", "s"),
        ("ddP_drr_c", None),
        ("ddrho_drr_c", None),
        ("Age", "yr"),
        ("teff", "K"),
        ("Gconst", "cm3/gs2"),
    ]
    loc_pars = [
        ("radius", "cm"),  # Var 1
        ("ln(m/M)", None),
        ("Temp", "K"),
        ("P", "kg/m/s2"),
        ("Rho", "g/cm3"),
        ("X", None),
        ("Lumi", "erg/s"),
        ("opacity", "cm2/g"),
        ("eps_nuc", None),
        ("gamma1", None),  # Var 10
        ("grada", None),
        ("delta", None),
        ("cp", None),
        ("free_e", None),
        ("brunt_A", None),  # Var 15 (!!)
        ("rx", None),
        ("Z", None),
        ("R-r", "cm"),
        ("eps_logg", None),
        ("Lg", "erg/s"),  # Var 20
        ("xhe3", None),
        ("xc12", None),
        ("xc13", None),
        ("xn14", None),
        ("xo16", None),
        ("dG1_drho", None),
        ("dG1_dp", None),
        ("dG1_dY", None),
        ("xh2", None),
        ("xhe4", None),  # Var 30
        ("xli7", None),
        ("xbe7", None),
        ("xn15", None),
        ("xo17", None),
        ("xo18", None),
        ("xne20", None),  # Var 36 (end of format spec.)
        ("xmg24", None),  # Extension. Added in Garstec
        ("xsi28", None),  # ibid.
        ("NA1", None),  # NOT IN USE
        ("NA2", None),
    ]  # NOT IN USE  -- Var 40

    # Start reading the file
    ff = open(filename, "r")
    lines = ff.readlines()

    # Read file definitions from the fifth line (first four is comments)
    NN, ICONST, IVAR, IVERS = [int(i) for i in lines[4].strip().split()]
    if not ICONST == 15:
        raise ValueError("cannot interpret FGONG file: wrong ICONST")

    # Data storage
    data = []
    starg = {}

    # Read the file from the fifth line onwards
    # Change in the format for storing the numbers (February 2017):
    #  - If IVERS <= 1000, 1p5e16.9
    #  - If IVERS  > 1000, 1p,5(x,e26.18e3)
    if IVERS <= 1000:
        for line in lines[5:]:
            data.append(
                [
                    line[0 * 16 : 1 * 16],
                    line[1 * 16 : 2 * 16],
                    line[2 * 16 : 3 * 16],
                    line[3 * 16 : 4 * 16],
                    line[4 * 16 : 5 * 16],
                ]
            )
    else:
        for line in lines[5:]:
            data.append(
                [
                    line[0 * 27 : 1 * 27],
                    line[1 * 27 : 2 * 27],
                    line[2 * 27 : 3 * 27],
                    line[3 * 27 : 4 * 27],
                    line[4 * 27 : 5 * 27],
                ]
            )

    # Put the data into arrays
    data = np.ravel(np.array(data, float))
    for i in range(ICONST):
        starg[glob_pars[i][0]] = data[i]
    data = data[15:].reshape((NN, IVAR)).T

    # Reverse the profile to get center ---> surface
    data = data[:, ::-1]

    # Make it into a record array and return the data
    starl = np.rec.fromarrays(data, names=[lp[0] for lp in loc_pars])

    # Exclude the center r = 0. mesh (MESA includes it)
    if starl["radius"][0] < 1.0e-14:
        starl = starl[1:]

    return starg, starl
