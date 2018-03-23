# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:06:32 2018

@author: jbr2

variational code for determining the complex index of a suspended film given
R and T measurements

based on method described by Soldera and Monterrat in Polymer 43, 6027 (2002)
"""

"""
initialize environment etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.seterr(divide='ignore', invalid='ignore', over='ignore')


"""
Function definitions

optical function definitions for normal incidence

definition: n_c = n + ik = (n2 + ik2)/(n1 + ik1)

"""

def r_interface(n, k):
    return -(n+k*1j-1)/(n+k*1j+1) 

def t_interface(n, k):
    return 2/(n+k*1j+1) 

#specfically suspended film with medium of n=1 on either side
    
def r_threephase(n, k, d, wavelength):
    up = r_interface(n, k) + r_interface(n/(n**2+k**2), -k/(n**2+k**2))*np.e**(4j*np.pi*d/wavelength*(n + k*1j))
    down = 1 + r_interface(n, k) * r_interface(n/(n**2+k**2), -k/(n**2+k**2))*np.e**(4j*np.pi*d/wavelength*(n + k*1j))
    return up/down 
def t_threephase(n, k, d, wavelength):
    up = t_interface(n, k) * t_interface(n/(n**2+k**2), -k/(n**2+k**2))*np.e**(2j*np.pi*d/wavelength*(n + k*1j))
    down = 1 + r_interface(n, k) * r_interface(n/(n**2+k**2), -k/(n**2+k**2))*np.e**(4j*np.pi*d/wavelength*(n + k*1j))
    return up/down 

"""
functions for computation of n and k variations
"""

def diffRn(n, k, step, thick, L):
    forward = abs(r_threephase(n + step, k, thick, L))**2 
    forward2 = abs(r_threephase(n + 2*step, k, thick, L))**2 
    backward = abs(r_threephase(n - step, k, thick, L))**2 
    backward2 = abs(r_threephase(n - 2*step, k, thick, L))**2 
    return (backward2/12-2/3*backward+2/3*forward-forward2/12)/(step)
def diffRk(n, k, step, thick, L):
    forward = abs(r_threephase(n, k + step, thick, L))**2 
    forward2 = abs(r_threephase(n, k + 2*step, thick, L))**2 
    backward = abs(r_threephase(n, k - step, thick, L))**2 
    backward2 = abs(r_threephase(n, k - 2*step, thick, L))**2 
    return (backward2/12-2/3*backward+2/3*forward-forward2/12)/(step)
def diffTn(n, k, step, thick, L):
    forward = abs(t_threephase(n + step, k, thick, L))**2 
    forward2 = abs(t_threephase(n + 2*step, k, thick, L))**2 
    backward = abs(t_threephase(n - step, k, thick, L))**2 
    backward2 = abs(t_threephase(n - 2*step, k, thick, L))**2 
    return (backward2/12-2/3*backward+2/3*forward-forward2/12)/(step)
def diffTk(n, k, step, thick, L):
    forward = abs(t_threephase(n, k + step, thick, L))**2 
    forward2 = abs(t_threephase(n, k + 2*step, thick, L))**2 
    backward = abs(t_threephase(n, k - step, thick, L))**2 
    backward2 = abs(t_threephase(n, k - 2*step, thick, L))**2  
    return (backward2/12-2/3*backward+2/3*forward-forward2/12)/(step)
#def diffRn(n, k, step, thick, L):
#    forward = abs(r_threephase(n + step, k, thick, L))**2 
#    backward = abs(r_threephase(n, k, thick, L))**2 
#    return (forward - backward)/(step)
#def diffRk(n, k, step, thick, L):
#    forward = abs(r_threephase(n, k + step, thick, L))**2 
#    backward = abs(r_threephase(n, k, thick, L))**2 
#    return (forward - backward)/(step)
#def diffTn(n, k, step, thick, L):
#    forward = abs(t_threephase(n + step, k, thick, L))**2 
#    backward = abs(t_threephase(n, k, thick, L))**2 
#    return (forward - backward)/(step)
#def diffTk(n, k, step, thick, L):
#    forward = abs(t_threephase(n, k + step, thick, L))**2 
#    backward = abs(t_threephase(n, k, thick, L))**2 
#    return (forward - backward)/(step)

"""
initialize measurement 
import data to be examined and setup initial parameter guesses
"""


def variational_nk(data, n_guess, k_guess, d_guess, step_size, N_steps):
    """
    variational_nk has 6 arguments
    (1) data as dataframe with wavenumber R and T columnds, 
    (2 and 3 and 4) initial n and k guesses and d guess,
    (5) stepsize for derivatives, 
    and (6) total number of iterations to undergo
    """
    WL = 10**4/np.array(data['wavenumber'])
    data_R = 0.90*np.array(data['R'])
    data_T = np.array(data['T'])
    """
    initial model guess
    """
    n = n_guess
    k = k_guess
    
    R = np.array([abs(r_threephase(n, k, d_guess, L))**2 for L in WL])
    T = np.array([abs(t_threephase(n_guess, k_guess, d_guess, L))**2 for L in WL])   
        
    """
        loop iterators
    """
    
    for i in range(N_steps):
        dRdn = diffRn(n, k, step_size, d_guess, WL)
        dRdk = diffRk(n, k, step_size, d_guess, WL)
        dTdn = diffTn(n, k, step_size, d_guess, WL)
        dTdk = diffTk(n, k, step_size, d_guess, WL)
        A = dTdk*dRdn-dRdk*dTdn
        delta_R = -(data_R - R)
        delta_T = -(data_T - T)
        delta_n = (delta_R*dTdk - delta_T*dRdk)/A
        delta_k = (delta_T*dRdn - delta_R*dTdn)/A
        n = n - delta_n
        k = k - delta_k
        R = abs(r_threephase(n, k, d_guess, WL))**2
        T = abs(t_threephase(n, k, d_guess, WL))**2
 
    """
    package data
    """
    
    calc_data = {'wavelength' : WL, 'R' : R, 'T' : T, 'n' : n, 'k' : k}
    return pd.DataFrame(calc_data)

data = pd.read_csv("IPDIP3.csv", header = None)
data = data.rename(index = str, columns = {0 : 'wavenumber', 1 : 'T', 2 : 'R'})

WL = 10**4/np.array(data['wavenumber'])

(n0, k0) = (1.6,0)
#d_guess = 2.5
n_step = 0.0005
N_steps = 10
errors = []
ds = []

for d_guess in np.arange(3.0,3.5,0.01):
    calc = variational_nk(data, n0, k0, d_guess, n_step, N_steps)
    print(d_guess)
    plt.figure(d_guess)
    plt.subplot(2,1,1)
    plt.plot(calc['wavelength'],calc['R'],"red")
    plt.ylim(-0.1,1.1)
    plt.subplot(2,1,2)
    plt.plot(calc['wavelength'],calc['n'])
    plt.ylim(1,2)
    plt.show()
    errors.append(np.sqrt(calc['n'].std()**2+calc['k'].std()**2))
    ds.append(d_guess)
    
plt.plot(ds,errors)



#"""
#look at results
#"""
#
#plt.figure(6)
#plt.subplot(211)
#plt.plot(WL,modelR0)
#plt.plot(WL,R,".")
#plt.plot(WL,data['R'])
#plt.ylim((-0.1, 0.75))
#
#plt.subplot(212)
#plt.plot(WL,modelT0)
#plt.plot(WL,T,"o")
#plt.plot(WL,data['T'])
#plt.ylim((-0.1,1.2))
#plt.show()
#
#plt.figure(7)
#plt.subplot(211)
#plt.plot(WL,n)
#plt.ylim((0,3.5))
#
#plt.subplot(212)
#plt.plot(WL,k/WL)
#plt.ylim((-0.1,0.5))
#plt.show()
# print((np.sum(delta_R**2+delta_T**2),np.std(delta_R**2+delta_T**2),n.isnull().sum()+n.isnull().sum(),np.max(np.abs(n)),np.max(np.abs(k))))