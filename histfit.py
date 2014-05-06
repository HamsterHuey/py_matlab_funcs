# -*- coding: utf-8 -*-
"""
Created on Mon May 05 23:10:12 2014

@author: Sudeep Mandal, sudeepmandal@gmail.com
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def histfit(data, ax=None, **kwargs):
    """
    Emulates MATLAB histfit() function. Plots histogram & Gaussian fit to data
    
    Parameters:
    -----------
        data :  1D array data to be plotted as histogram
        ax :    axes handle to use for plotting (eg: pass handle for using histfit
                to plot in a subplot of an existing figure)
        kwargs: 
            bins - Number of bins, defaults to sqrt(N)
            color - Color of histogram bars. eg: 'r','g','b', etc. Default = 'g'
    Returns:
    --------
        ax : axes handle of histfit plot
        fit_coeffs : Gaussian fit parameters from fitting routine
                        fit_coeffs[0] is the Amplitude, fit_coeffs[1] = mu, 
                        fit_coeffs[2] = sigma
    """

    bins = kwargs['bins'] if 'bins' in kwargs else np.sqrt(len(data))
    color = kwargs['color'] if 'color' in kwargs else 'g'
    
    if ax is None:
        ax = plt.gca()
    
    # Plot histogram
    n, bins, patches = ax.hist(data, bins=bins, alpha=0.6, color=color)
    bin_centers = (bins[:-1] + bins[1:])/2
    
    # Define model function to be used to fit to the data above:
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [1., np.mean(data), np.std(data)]
    fit_coeffs, var_matrix = curve_fit(gauss, bin_centers, n, p0=p0)
    mu, sigma = fit_coeffs[1], fit_coeffs[2]
    xnormfit = np.arange(mu-4*sigma,mu+4*sigma,(8*sigma/100))
    ynormfit = gauss(xnormfit, *fit_coeffs)
    
    # Plot Fit
    ax.plot(xnormfit, ynormfit, 'r-', linewidth=2,)
    ax.axis('tight')
    ax.set_ylim(np.array(ax.get_ylim()) * 1.1)
    
    return ax, fit_coeffs