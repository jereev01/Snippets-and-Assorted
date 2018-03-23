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

data = pd.read_csv("AndersonRTdata2um.csv", header = 0)
measured = pd.read_csv("Anderson.csv", header = None)
measured = measured.rename(index = str, columns = {0 : 'wavelength [um]', 1 : 'n', 2 : 'k'})
meas_n = np.array(measured['n'])
meas_k = np.array(measured['k'])

WL = data['wavelength']

"""
initialize model and define differential functions
"""

(n0, k0) = (1.3205, 0.025)
d_guess = 2

n_step = 0.005
k_step = 0.005

def diffRn(n, k, step, thick, L):
    forward = abs(r_threephase(n + step, k, thick, L))**2 
    backward = abs(r_threephase(n - step, k, thick, L))**2 
    return (forward - backward)/(2*step)
def diffRk(n, k, step, thick, L):
    forward = abs(r_threephase(n, k + step, thick, L))**2 
    backward = abs(r_threephase(n, k - step, thick, L))**2 
    return (forward - backward)/(2*step)
def diffTn(n, k, step, thick, L):
    forward = abs(t_threephase(n + step, k, thick, L))**2 
    backward = abs(t_threephase(n - step, k, thick, L))**2 
    return (forward - backward)/(2*step)
def diffTk(n, k, step, thick, L):
    forward = abs(t_threephase(n, k + step, thick, L))**2 
    backward = abs(t_threephase(n, k - step, thick, L))**2 
    return (forward - backward)/(2*step)

modelR0 = np.array([abs(r_threephase(n0, k0, d_guess, L))**2 for L in WL])
dRdn = np.array([diffRn(n0, k0, n_step, d_guess, L) for L in WL])
dRdk = np.array([diffRk(n0, k0, k_step, d_guess, L) for L in WL])


modelT0 = np.array([abs(t_threephase(n0, k0, d_guess, L))**2 for L in WL])
dTdn = np.array([diffTn(n0, k0, n_step, d_guess, L) for L in WL])
dTdk = np.array([diffTk(n0, k0, k_step, d_guess, L) for L in WL])


"""
set up iterators 
"""

delta_R = data['R'] - modelR0
delta_T = data['T'] - modelT0

A = dTdk*dRdn-dRdk*dTdn

delta_n = (delta_R*dTdk - delta_T*dRdk)/A
delta_k = (delta_T*dRdn - delta_R*dTdn)/A

new_n = n0 + delta_n
new_k = k0 + delta_k

new_R = abs(r_threephase(new_n, new_k, d_guess, WL))**2
new_T = abs(t_threephase(new_n, new_k, d_guess, WL))**2

damp = 0.01

new_dRdn = damp*np.arcsinh(diffRn(new_n, new_k, n_step, d_guess, WL)/damp)
new_dRdk = damp*np.arcsinh(diffRk(new_n, new_k, k_step, d_guess, WL)/damp)
new_dTdn = damp*np.arcsinh(diffTn(new_n, new_k, n_step, d_guess, WL)/damp)
new_dTdk = damp*np.arcsinh(diffTk(new_n, new_k, k_step, d_guess, WL)/damp)

new_A = new_dTdk*new_dRdn-new_dRdk*new_dTdn



"""
loop iterators
"""



for i in range(100):
    delta_R = data['R'] - new_R
    delta_T = data['T'] - new_T
    delta_n = (delta_R*new_dTdk - delta_T*new_dRdk)/new_A
    delta_k = (delta_T*new_dRdn - delta_R*new_dTdn)/new_A
    new_n = new_n + delta_n
    new_k = new_k + delta_k
    print((np.sum((meas_n-new_n)**2 + (meas_k-new_k)**2),max(np.abs((meas_n-new_n)**2)),max(np.abs((meas_k-new_k)**2))))
    new_R = abs(r_threephase(new_n, new_k, d_guess, WL))**2
    new_T = abs(t_threephase(new_n, new_k, d_guess, WL))**2
    new_dRdn = diffRn(new_n, new_k, n_step, d_guess, WL)
    new_dRdk = diffRk(new_n, new_k, k_step, d_guess, WL) 
    new_dTdn = diffTn(new_n, new_k, n_step, d_guess, WL)
    new_dTdk = diffTk(new_n, new_k, k_step, d_guess, WL)
    new_A = new_dTdk*new_dRdn-new_dRdk*new_dTdn
    print((np.sum(delta_R**2+delta_T**2),np.std(delta_R**2+delta_T**2),new_k.isnull().sum()+new_n.isnull().sum(),np.max(np.abs(new_n)),np.max(np.abs(new_k))))
    
#plt.figure(1)
#plt.plot(WL, dRdn)
#plt.plot(WL, new_dRdn, "o")
#plt.show()
#
#plt.figure(2)
#plt.plot(WL, dRdk)
#plt.plot(WL, new_dRdk, "o")
#plt.show()
#
#plt.figure(3)
#plt.plot(WL, dTdn)
#plt.plot(WL, new_dTdn, "o")
#plt.show()
#
#plt.figure(4)
#plt.plot(WL, dTdk)
#plt.plot(WL, new_dTdk, "o")
#plt.show()
#
#plt.figure(5)
#plt.plot(WL, A)
#plt.plot(WL, new_A, "o")
#plt.show()

plt.figure(6)
plt.subplot(211)
plt.plot(WL,new_R)
plt.plot(WL,data['R'])

plt.subplot(212)
plt.plot(WL,new_T)
plt.plot(WL,data['T'])
plt.show()

plt.figure(7)
plt.subplot(211)
plt.plot(WL,new_n,"o-")
plt.plot(WL,meas_n)
plt.ylim((1.3,1.4))

plt.subplot(212)
plt.plot(WL,new_k,"o")
plt.plot(WL,meas_k)
plt.ylim((-0.05,0.1))
plt.show()