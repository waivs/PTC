# PTC

This repository contains some functions to help generate and use the photon
transfer curve (PTC) method, including binning and fitting the data. 

A small example dataset is also included. The example script analyzes and plots
this example data and also provides a simulation of noise with parameters that
a user can modify. 

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
