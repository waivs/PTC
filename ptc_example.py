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


# %% Load small example data
data = np.load('exampledata.npy')+0

# %% use PTC functions to process data
signal, noise = ptc.ptc(data,binsize=None) # all values
fitd=0
# fitge,fitsr = ptc.fitptc(signal, noise, maxfit='Noise2')
fitge,fitsr,fitd = ptc.fitptcd(signal, noise, maxfit='Noise2')

binnedsignal, binnednoise = ptc.ptc(data,binsize=1) # binned data


# %% display results
f1,a1=plt.subplots(nrows=2,ncols=2,num=1);f1.clear()
f1,a1=plt.subplots(nrows=2,ncols=2,num=1); # resets plot each time

a1[0,0].imshow(data.mean(axis=0),cmap='gray'); a1[0,0].set_title('Average image')
a1[0,1].imshow(data.std(axis=0,ddof=1)); a1[0,1].set_title('Noise image (std)')

a1[1,0].loglog(signal-fitd,noise,'.', alpha=0.1,label='all data')
a1[1,0].loglog(binnedsignal-fitd,binnednoise,'.-', alpha=0.3,label='binned data')
sigvals = np.arange(signal.min(),signal.max());
a1[1,0].loglog(sigvals,np.sqrt((sigvals)*fitge+fitsr**2),label='fit PTC')
a1[1,0].set_title(f'PTC (log-log space)')
a1[1,0].set_xlabel(f'signal')
a1[1,0].set_ylabel(f'noise')
a1[1,0].legend()


a1[1,1].plot(signal,noise**2,'.', alpha=0.1,label='all data')
a1[1,1].plot(binnedsignal,binnednoise**2,'.-', alpha=0.3,label='binned data')
a1[1,1].plot(sigvals,((sigvals-fitd)*fitge+fitsr**2),label='PTCfunc fit')
a1[1,1].set_title(f'Variance vs signal (linear space)')
a1[1,1].set_xlabel(f'signal')
a1[1,1].set_ylabel(f'variance (noise$^2$)')
a1[1,1].set_ylim((a1[1,1].get_ylim()[0],binnednoise.max()**2));
a1[1,1].set_xlim((a1[1,1].get_xlim()[0],binnedsignal.max()));


# %% simulate data and display PTC
N = 256 # image size
frames = 100 # number of frames
bitdepth = 2**13 # max integer that is digitized
simdark = 0 # dark counts
simgain = 408 # effective gain factor
simsr = 35  # read noise

# max photoelectrons (signal brightness). You can change this to a specific 
# value, but assuming the data should approach saturation, we can divide the 
# bitdepth by the gain to get the max # photoelectrons
maxphotoelect = bitdepth/simgain # max signal in photoelectrons before gain


# # First make set of NxN frames with gradient spanning from 0 to 1
# simd = np.tile(np.arange(N)/(N-1),(N,1)) # NxN array with values from 0 to 1 along x-axis
# simd = np.tile(simd,(frames,1,1))  # repeat in z-axis to make frames

# Instead, make a continuous ramp and reshape which provides more intermediate pixels values along the columns
simd = np.reshape(np.arange(N**2)/(N**2-1),(N,-1)).T # NxN array with values from 0 to 1 along x-axis
simd = np.tile(simd,(frames,1,1))  # repeat in z-axis to make frames

simd *= maxphotoelect

# If you want to see how this works with normally distributed noise:
# simd = (simd + simdark # mean signal + dark counts + noise using normal noise
#         + np.random.randn(frames,N,N)*simgain*simd 
#         + np.random.randn(frames,N,N)*simsr )

# We assume that the values will span the bit depth, but the shot noise is
# due to the detected electrons (photons*QE). For example, the max signal
# may be the bit depth, but this was after gain times the max number of 
# photoelectrons that are amplified by gain.

# model shot noise
if 0:  # apply shot noise to photoelectrons and then add dark and apply gain  
    for row in np.arange(N):
        for col in np.arange(N):
            simd[:,row,col]=np.random.poisson(lam=simd[:,row,col].mean(),
                                        size=(frames))*simgain
    simd += simdark # add dark counts
    simd += np.random.randn(frames,N,N)*simsr # add read noise using normal dist 
            
else: # same, but apply random gain
    for row in np.arange(N):
        for col in np.arange(N):
            simd[:,row,col]=np.random.poisson(lam=simd[:,row,col].mean(),
                                        size=(frames))
    simd *= (simgain+np.random.randn(frames,N,N)*.2) # random gain (this is not rigorous yet)
    simd += simdark # add dark counts
    simd += np.random.randn(frames,N,N)*simsr # add read noise using normal dist 

simd[simd>bitdepth]=bitdepth # saturation
# mindetectorval = 0; simd[simd<mindetectorval]=mindetectorval

simsig, simnoise = ptc.ptc(simd,binsize=None) # all values
simfitge,simfitsr = ptc.fitptc(simsig, simnoise, maxfit="Noise2")
# simfitge,simfitsr = ptc.fitptc(simsig, simnoise, maxfit=4000)

simfitdark = 0 # set to zero in case we don't use the fit below
# simfitge,simfitsr,simfitdark = ptc.fitptcd(simsig, simnoise, maxfit="Noise2")
# simfitge,simfitsr,simfitdark = ptc.fitptcd(simsig, simnoise, maxfit=4000)

binnedsimdsignal, binnedsimdnoise = ptc.ptc(simd,binsize=2) # binned data

f2,a2=plt.subplots(nrows=2,ncols=2,num=2);f2.clear()
f2,a2=plt.subplots(nrows=2,ncols=2,num=2); # resets plot each time

a2[0,0].imshow(simd.mean(axis=0),cmap='gray'); a2[0,0].set_title('Average modeled image')
a2[0,1].imshow(simd.std(axis=0,ddof=1)); a2[0,1].set_title('modeled noise')
# a2[0,1].imshow(shotnoise[0]); a1[0,1].set_title('modeled noise')

a2[1,0].loglog(simsig,simnoise,'.', alpha=0.2,label='all data')
a2[1,0].loglog(binnedsimdsignal,binnedsimdnoise,'.-', alpha=0.3,label='binned data')

simsigvals = np.arange(simsig.min(),simsig.max())
a2[1,0].loglog(simsigvals,np.sqrt((simsigvals+simfitdark)*simfitge+simfitsr**2),label='fit PTC')
a2[1,0].loglog(simsigvals,np.sqrt((simsigvals+simdark)*simgain+simsr**2),label='expected PTC')
a2[1,0].set_title(f'PTC (log-log space)')
a2[1,0].set_xlabel(f'signal')
a2[1,0].set_ylabel(f'noise')
a2[1,0].legend()

a2[1,1].plot(simsig,simnoise**2,'.', alpha=0.1,label='all data')
a2[1,1].plot(binnedsimdsignal,binnedsimdnoise**2,'.-', alpha=0.3,label='binned data')
a2[1,1].plot(simsigvals,((simsigvals+simfitdark)*simfitge+simfitsr**2),label='PTCfunc fit')
a2[1,1].plot(simsigvals,((simsigvals+simdark)*simgain+simsr**2),label='expected PTCfunc')
a2[1,1].set_title(f'Variance vs signal (linear space)')
a2[1,1].set_xlabel(f'signal')
a2[1,1].set_ylabel(f'variance (noise$^2$)');
# a2[1,1].set_ylim((a1[1,1].get_ylim()[0],binnesimdnoise.max()**2))
# a2[1,1].set_xlim((a1[1,1].get_xlim()[0],binnesimdsignal.max()))
