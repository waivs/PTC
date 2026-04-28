#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:44:28 2026

@author: jeremy.rogers@wisc.edu

Quickstart:
    import photontransfercurve as ptc
    import matplotlib.plot as plt
    <load data as 3D video with first dimension as time or frames>
    signal,noise = ptc.ptc(data)
    ge,sr = ptc.fitptc(signal,noise)
    plt.figure()
    plt.loglog(signal,noise, label='data')
    plt.loglog(signal, ge*signal+sr)
    plt.legend()
"""



# %% imports
import time
import numpy as np
import matplotlib.pyplot as plt
import photontransfercurve as ptc
from scipy.optimize import curve_fit


# %% functions for calculating PTC
def ptc(d, binsize=None):
    """
    Calculate the noise and signal from an image stack or video. 
    
    Parameters
    ----------
    d : 3D numpy array 
        first axis it time or frame of a video
    binsize : integer, float, or None (default)
        value of size of the bins, optional. If None, return noise (std)
        and signal (mean) from all pixels over time. If a bin size is used, 
        all pixels within the bin will be pooled to calculate the noise and
        signal values. The signal value will be the center of the bin and any
        pixels with mean value across time within half the bin size of the bin
        center will be included. 

    Returns
    -------
    signal : 1D array 
        signal values 
    noise : 1D array 
        noise values
    """
    d=np.reshape(d,shape=(d.shape[0],-1)) # make 2D with time as 1st dim
    signal = np.mean(d,axis=0) # average value of each pixel flattened to a 1D array
    noise = np.std(d,axis=0,ddof=1) # sample standard deviation of each pixel flattened to a 1D array
    
    # if binsize if passed, bin the data according to the bin size
    if binsize==1:
        roundedsignal = np.round(signal) # signal rounded for indexing
        signalbins = np.arange(roundedsignal.min(),roundedsignal.max()) # values for binning the data
        binnedsignal = 0*signalbins
        binnednoise = 0*signalbins # array to hold binned noise vals
        nptsinbin = 0*signalbins # number of pixels in each bin
        for ii in np.arange(binnedsignal.shape[0]):
            binnedsignal[ii]=np.mean(d[:,roundedsignal==signalbins[ii]])
            binnednoise[ii]=np.std(d[:,roundedsignal==signalbins[ii]],ddof=1)
            nptsinbin[ii] = d[:,roundedsignal==signalbins[ii]].shape[1]
        
        # TODO: work on generalizing for different bins sizes ...
        # binsize = 10
        # for ii in np.arange(sigvals.shape[0]):
        #     pointstouse = dmean==sigvals[ii]
        #     for jj in np.arange(int(np.floor(binsize/2)-1))+1:
        #         if ii-jj>=0 and ii+jj<=sigvals.shape[0]-1:
        #             pointstouse+=(dmean==sigvals[ii-jj])
        #             pointstouse+=(dmean==sigvals[ii+jj])
        #     binnednoise[ii]=np.std(d[:,pointstouse],ddof=1)
        #     nptsinbin[ii] = d[:,pointstouse].shape[1]

        t=1 # threshold: remove elements with too few points (i.e. not enough 
            # to calculate std())
            # t=0 is the bare minimum if std(x,ddof=0) is used, but we use
            # the std(x,ddof=1) to reduce bias from a finite sample size. 
            # This means we need at least 2 points to get a valid noise value. 
            # values of zero when only 
        # strip values with not enough points
        binnedsignal=binnedsignal[nptsinbin>t]    
        binnednoise=binnednoise[nptsinbin>t]  
        nptsinbin = nptsinbin[nptsinbin>t]
        
        signal = binnedsignal;
        noise = binnednoise;
    else: print(f'only binsize=1 is currently supported, returning all values')

    return signal, noise


# %% functions for calculating PTC
def fitptc(signal, noise, saturation=None):
    """
    Fit a model of noise as a function of signal and return fit coefficients.
    The fit is performed on variance as a function of signal which is modeled
    as a linear function: 
        noisevar = ge * signal + sr2 
    where: 
    signal = the signal or mean pixel value
    noisevar = noise variance or the square of the standard deviationas a function of signal 
    ge = the effective gain coefficient or the proportionality constant when noise is dominated by shot noise
    sr2 = read noise squared, or the signal independant variance that exists evn when input signal is zero
    
    Note that the input and output are in units of noise (standard deviation),
    but the fit is performed on the variance where the function is linear.
    
    This function performs 2 iterations. Linear least squares fit normally 
    assumes equal variance of the resduals, however, we expect poisson (shot)
    noise, so the variance /standard deviation depends on the signal. The 
    curve_fit function have be passed an array sigma to weight the fit according
    to the expected standard deviation. Thereore, we first fit the noise to get 
    an estimate of sigma, and the repeat the fit using the first fit parameters
    to calculate an estimated weighting function.
    
    Parameters
    ----------
    signal : 1D array of length N
        array of signal values
    noise : 1D array of length N
        array of corresponding noise values
    saturation : value 
        This is the maximum value that the system can produce and is used to 
        limit the fit range so that values that begin to saturate do not 
        contribute to the fit erroneously:
            False: fit all the values provided
            None (default): stop the fit 2x the max noise below the max signal value
            Value: fit only up to the given value
        DESCRIPTION. The default is None.

    Returns
    -------
    ge : float
        fit of the effective gain coefficient 
    sr : float
        fit value of the constant (signal independent) read noise or ciruit noise

    """
    def noisevar(signal, ge, sr2):  return ge*signal+sr2
    
    if saturation == False: maxfitval = signal.max()
    elif saturation == None: maxfitval = signal.max()-2*noise.max()
    else: maxfitval = saturation

    sigvals2fit = signal[signal<maxfitval]
    noisevals2fit = noise[signal<maxfitval]
    
    # first iteration to estimate sigma
    pout,pcov = curve_fit(noisevar, 
                          sigvals2fit, 
                          noisevals2fit**2, 
                          [1,0], # initial guesses for gain and read noise
                          bounds=([0,-1e4],[1e8,1e4])) # bounds are ([a_min,b_min,c_min,...],[a_max,b_max,cmax,...])
    ge, sr2 = pout
    
    # second iteration passing weighted function
    pout,pcov = curve_fit(noisevar, 
                          sigvals2fit, 
                          noisevals2fit**2, 
                          [ge,sr2], # initial guesses for gain and read noise
                          sigma=((sigvals2fit-sigvals2fit.min()+1)*ge+sr2)**.5, 
                          bounds=([0,-1e4],[1e8,1e4])) # bounds are ([a_min,b_min,c_min,...],[a_max,b_max,cmax,...])
    ge, sr2 = pout

    return ge, sr2**.5

