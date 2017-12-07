# -*- coding: utf-8 -*-
"""
FFT Basics
"""
import numpy as np
import matplotlib.pyplot as plt

f = 18
num = 100
start = -1
end = 1
step = (end - start)/num

test_x = np.array([(step*x + start) for x in range(num+1)])                 
test_y = np.array([np.e**(-x**2/(2*0.1**2))*(np.cos(2*np.pi*f*x)+0.2*np.cos(2*np.pi*0.7*f*x)) for x in test_x])

#plt.plot(test_x,test_y,"o-")
fourier = np.fft.fft(test_y) #question for self: what are different ffts in py

pwr_spct = list(np.abs(fourier)**2)


"""
assign absolute frequencies to the data. 
presently: must have an even number of data points, but can be generalized
"""

fourier_x = [1/(step*num)*x for x in range(num//2+1)]
fx = fourier_x + [-1*x for x in fourier_x[:0:-1]]

plt.plot(fx,pwr_spct,"o-")

