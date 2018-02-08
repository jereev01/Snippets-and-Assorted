# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:02:50 2017

@author: jbr2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

test_data = pd.read_csv('BG_Au_R_int.0.dpt', header = None)
test_data = test_data.rename(index = str, columns = {0 : 'measurement', 1 : 'amplitude'})
test_data = test_data.drop('measurement', axis = 1)

data = np.array(test_data['amplitude'])

ref = pd.read_csv('BG_Au_R_spect.0.dpt', header = None)
ref = ref.rename(index = str, columns = {0 : 'measurement', 1 : 'amplitude'})
refdata = np.array(ref['amplitude'])
refwave = np.array(ref['measurement'])


laser_wavenumber = 15797.09
data_length = int(len(data)/2)

forward = data[:data_length]
backward = data[data_length:]



"""
Normalize
"""

normal_forward = forward - np.mean(forward)

"""
Zero-Fill
"""

log_size = int(np.log2(data_length))
zerofill_factor = 2
num_zeros = 2**(zerofill_factor + log_size) - data_length
zero_pad = [0] * num_zeros
normal_pad_forward = np.append(normal_forward, zero_pad)

"""
Calculate Mertz Phase Correction Data
"""

phase_data_size = 128

zpd = np.argmax(normal_pad_forward)
zpd_data = normal_pad_forward[zpd-phase_data_size:zpd+phase_data_size]
mertz = [0] * len(normal_pad_forward)

for i, j in zip(range(zpd-phase_data_size,zpd+phase_data_size), range(0,phase_data_size*2)):
    mertz[i] = (1-np.abs((zpd - i)/phase_data_size))*zpd_data[j] 
    
mertz = mertz[zpd:] + mertz[:zpd]

fftmertz = np.fft.fft(mertz)
fftmertzR = np.real(fftmertz)
fftmertzI = np.imag(fftmertz)
mertzpwrspct = np.abs(fftmertz)


phase_dataR = mertzpwrspct * fftmertzR
phase_dataI = mertzpwrspct * fftmertzI


"""
Mertz Correction Apodization
"""

apocorrection = []
for i in range(2*zpd):
    apocorrection.append(zpd/0.5 * i)
for i in range(2*zpd,len(normal_forward)):
    apocorrection.append(1 - len(normal_forward) * (i - 2*zpd))
for i in range(len(normal_forward),len(normal_pad_forward)):
    apocorrection.append(0)


apoforward = list(normal_pad_forward * apocorrection)
apoforward = apoforward[zpd:] + apoforward[:zpd]

"""
fft
"""

spectrum_size = len(apoforward)//2
max_wavenumber = laser_wavenumber
spacing = max_wavenumber/spectrum_size
freq_data = [spacing * i for i in range(spectrum_size)]
                

fftspectrum = 1/(4*spectrum_size*laser_wavenumber)*np.fft.fft(apoforward)[:spectrum_size]
fftspectrumR = np.real(fftspectrum)
fftspectrumI = np.imag(fftspectrum)


plt.plot(refwave,refdata,"o")
plt.plot(freq_data,np.abs(fftspectrum),"o")
plt.xlim(2000,2010)

print(max(refdata))
print(max(np.abs(fftspectrum)))

print(max(refdata)/max(np.abs(fftspectrum)))