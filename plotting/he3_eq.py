import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Set the style of all plots
plt.style.use(os.path.join(os.environ['BASTADIR'], 'basta/plots.mplstyle'))

def he3_frac_easy(xh, xhe4, temp):

    t = lambda w: 19.721*(w/temp*1e-7)**(1./3)
    
    sv_pp = t(1/2)**2*np.exp(-t(1/2))
    sv_HeHe = t(16*9/6)**2*np.exp(-t(16*9/6))
    sv_aHe = t(16*3*4/7)**2*np.exp(-t(16*3*4/7))

    term1 = -3./4*xhe4*sv_aHe
    term2 = term1**2.+2.*(3*xh)**2*sv_pp*sv_HeHe
    term3 = 2.*sv_HeHe
    
    frac = (term1+np.sqrt(term2))/term3
    
    return frac

def S0(species, temp):
    
    if species == 'pp':
        s0 = 4.01e-25
        s1 = 4.49e-24
        s2 = 0.0
    elif species == 'HeHe':
        s0 = 5.21
        s1 = -4.49
        s2 = 2.2e1
    elif species == 'aHe':
        s0 = 5.6e-4
        s1 = -3.6e-4
        s2 = 1.51e-4
    else:
        print('Incorrect species in S0')
        sys.exit()
    t = tau(species, temp)
    kT = 0.086173*temp*1e-9
    E = t*kT/3

    seff = s0*(1+5/(12*t) + s1/s0*(E+35./36*kT) + 1/2*s2/s0*(E**2.+89./36*E*kT))
    return seff

def tau(species, temp):
    if species == 'pp':
        z1 = 1; z2 = 1
        m1 = 1; m2 = 1
    elif species == 'HeHe':
        z1 = 2; z2 = 2
        m1 = 3; m2 = 3
    elif species == 'aHe':
        z1 = 2; z2 = 2
        m1 = 3; m2 = 4
    
    # Iliadis eq. 3.88
    t = 4.2487*np.cbrt(z1**2.*z2**2.*(m1*m2/(m1+m2))/(temp*1e-9))
    return t

def cross_section(species, temp):
    if species == 'pp':
        z1 = 1; z2 = 1
        m1 = 1; m2 = 1
    elif species == 'HeHe':
        z1 = 2; z2 = 2
        m1 = 3; m2 = 3
    elif species == 'aHe':
        z1 = 2; z2 = 2
        m1 = 3; m2 = 4
    
    t = tau(species, temp)
    s = S0(species, temp)
    
    m = m1*m2/(m1+m2)
    
    # Iliadis eq. 3.81 without constants, dissappears in frac
    sv = s*t**2.*np.exp(-t)/(m*z1*z2)

    return sv

def he3_frac(xh, xhe4, temp):
    ind = np.where(temp*1e-7<1)[0]
    temp[ind] = np.nan
    sv_pp = cross_section('pp', temp)
    sv_HeHe = cross_section('HeHe', temp)
    sv_aHe = cross_section('aHe', temp)

    term1 = -3./4*xhe4*sv_aHe
    term2 = term1**2.+2.*(3*xh)**2.*sv_pp*sv_HeHe
    term3 = 2.*sv_HeHe
    
    frac = (term1+np.sqrt(term2))/term3
    
    return frac
