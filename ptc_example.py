#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:19:17 2026

@author: jeremy.rogers@wisc.edu

example script for processing PTC
"""

# %% Imports
import numpy as np
import photontransfercurve as ptc
import matplotlib.pyplot as plt


# %% simulate data



# %% Load small example data
data = np.load('exampledata.npy')


# %% use PTC functions to process data
signal, noise = ptc.ptc(data,binsize=None) # all values
ptcge,ptcsr = ptc.fitptc(signal, noise, maxfit='Noise2')

binnedsignal, binnednoise = ptc.ptc(data,binsize=3) # binned data


# %% display results
f1,a1=plt.subplots(nrows=2,ncols=2,num=1);f1.clear()
f1,a1=plt.subplots(nrows=2,ncols=2,num=1); # resets plot each time

a1[0,0].imshow(data.mean(axis=0),cmap='gray'); a1[0,0].set_title('Average image')
a1[0,1].imshow(data.std(axis=0,ddof=1)); a1[0,1].set_title('Noise')

a1[1,0].loglog(signal,noise,'.', alpha=0.2,label='all data')
a1[1,0].loglog(binnedsignal,binnednoise,'.-', alpha=0.3,label='binned data')
sigvals = np.arange(signal.min(),signal.max())
a1[1,0].loglog(sigvals,np.sqrt(sigvals*ptcge+ptcsr**2),label='fit PTC')
a1[1,0].set_title(f'PTC log-log space')
a1[1,0].set_xlabel(f'signal')
a1[1,0].set_ylabel(f'noise')
a1[1,0].legend()


a1[1,1].plot(signal,noise**2,'.', alpha=0.1,label='all data')
a1[1,1].plot(binnedsignal,binnednoise**2,'.-', alpha=0.3,label='binned data')
a1[1,1].plot(sigvals,(sigvals*ptcge+ptcsr**2),label='PTCfunc fit')
a1[1,1].set_title(f'Variance vs signal')
a1[1,1].set_xlabel(f'signal')
a1[1,1].set_ylabel(f'variance (noise$^2$)')
a1[1,1].set_ylim((a1[1,1].get_ylim()[0],binnednoise.max()**2))
a1[1,1].set_xlim((a1[1,1].get_xlim()[0],binnedsignal.max()))


