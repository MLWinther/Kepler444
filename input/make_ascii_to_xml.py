import os
import numpy as np
import basta.fileio as fio

c = 299792.458 # km/s

Vr = -120.78367 # km/s

freqs = np.loadtxt("Kepler444_uncorrected.fre")
freqs[:,2] *= (1 + Vr/c)
np.savetxt("Kepler444.fre", freqs, fmt="%d  %d  %.2f  %.2f  %d")

fio.freqs_ascii_to_xml('.', 'Kepler444',check_radial_orders=False)
