# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:06:32 2018

@author: jbr2

variational code for determining the complex index of a suspended film given
R and T measurements

based on method described by Soldera and Monterrat in Polymer 43, 6027 (2002)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
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

data = pd.read_csv("IPDIP.csv", header = None)
data = data.rename(index = str, columns = {0 : 'wavenumber', 1 : 'T', 2 : 'R'})

WL = 10**4/np.array(data['wavenumber'])

print(WL[0])

"""
initialize model and define differential functions
"""

(n0, k0) = (1.7,0.05)
d_guess = 0.5


n_step = 0.0005
k_step = 0.0005

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

def damping(val,damp_par):
    return damp_par*np.arcsinh((val-np.mean(val))/damp_par)+np.mean(val)

modelR0 = np.array([abs(r_threephase(n0, k0, d_guess, L))**2 for L in WL])
dRdn = np.array([diffRn(n0, k0, n_step, d_guess, L) for L in WL])
dRdk = np.array([diffRk(n0, k0, k_step, d_guess,L) for L in WL])


modelT0 = np.array([abs(t_threephase(n0, k0, d_guess, L))**2 for L in WL])
dTdn = np.array([diffTn(n0, k0, n_step, d_guess, L) for L in WL])
dTdk = np.array([diffTk(n0, k0, k_step, d_guess, L) for L in WL])




"""
set up iterators 
"""

delta_R = -(data['R'] - modelR0)
delta_T = -(data['T'] - modelT0)

A = (dRdn)**2+(dRdk)**2

delta_n = (delta_R*dRdn)/A
delta_k = (delta_R*dRdk)/A

new_n = n0 - delta_n
new_k = k0 - delta_k

new_R = abs(r_threephase(new_n, new_k, d_guess, WL))**2
#new_T = abs(t_threephase(new_n, new_k, d_guess, WL))**2

new_dRdn = diffRn(new_n, new_k, n_step, d_guess, WL)
new_dRdk = diffRk(new_n, new_k, k_step, d_guess, WL) 
#new_dTdn = diffTn(new_n, new_k, n_step, d_guess, WL)
#new_dTdk = diffTk(new_n, new_k, k_step, d_guess, WL)

new_A = new_dRdn**2+new_dRdk**2


damp = 1000
"""
loop iterators
"""

for i in range(20):
    delta_R = -(data['R'] - new_R)
    #delta_T = -(data['T'] - new_T)
    delta_n = (delta_R*new_dRdn)/new_A
    delta_k = (delta_R*new_dRdk)/new_A
    new_n = new_n - delta_n
    new_k = new_k - delta_k
    new_R = abs(r_threephase(new_n, new_k, d_guess, WL))**2
    #new_T = abs(t_threephase(new_n, new_k, d_guess, WL))**2
    new_dRdn = diffRn(new_n, new_k, n_step, d_guess, WL)
    new_dRdk = diffRk(new_n, new_k, k_step, d_guess, WL)
    #new_dTdn = diffTn(new_n, new_k, n_step, d_guess, WL)
    #new_dTdk = diffTk(new_n, new_k, k_step, d_guess, WL)
    new_A = new_dRdn**2+new_dRdk**2
 
 


    
    
#plt.figure(1)
#plt.plot(WL, dRdn)
#plt.plot(WL, new_dRdn)
#plt.ylim((-5,5))
#plt.xlim((5.6,5.8))
#plt.show()
#
#plt.figure(2)
#plt.plot(WL, dRdk)
#plt.plot(WL, new_dRdk)
#plt.xlim((5.6,5.8))
#plt.show()
#
#plt.figure(3)
#plt.plot(WL, dTdn)
#plt.plot(WL, new_dTdn)
#plt.xlim((5.6,5.8))
#plt.show()
#
#plt.figure(4)
#plt.plot(WL, dTdk)
#plt.plot(WL, new_dTdk)
#plt.xlim((5.6,5.8))
#plt.show()

#plt.figure(5)
#plt.plot(WL, A)
#plt.plot(WL, new_A)
#plt.show()

plt.figure(6)
plt.subplot(211)
plt.plot(WL,modelR0)
plt.plot(WL,new_R)
plt.plot(WL,data['R'])
plt.ylim((-0.1, 0.75))

#plt.subplot(212)
#plt.plot(WL,modelT0)
#plt.plot(WL,new_T)
#plt.plot(WL,data['T'])
#plt.ylim((-0.1,1.2))
#plt.show()

plt.figure(7)
plt.subplot(211)
plt.plot(WL,new_n)
plt.ylim((0,3.5))

plt.subplot(212)
plt.plot(WL,new_k/WL)
plt.ylim((0,1))
plt.show()