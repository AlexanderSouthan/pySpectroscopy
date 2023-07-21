# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:23:17 2023

@author: southan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from little_helpers.math_functions import gaussian
from pySpectroscopy import spectroscopy_data


# Calculate NMR spectrum
chem_shift = np.linspace(0, 5, 1000)
intensities = np.array([
    gaussian(chem_shift, [3/4, 3/2, 3/4], [1.1, 1.2, 1.3], [0 ,0, 0], [0.01, 0.01, 0.01]) + 
    gaussian(chem_shift, [2/8, 2*3/8, 2*3/8, 2/8], [2.1, 2.2, 2.3, 2.4], [0 ,0, 0, 0], [0.01, 0.01, 0.01, 0.01]) +
    np.random.normal(scale=0.01, size=chem_shift.size)])

# Put NMR spectrum into spectroscopy data class
nmr = spectroscopy_data(
    data_source='DataFrame',
    spectral_data=pd.DataFrame(intensities, columns=chem_shift))

# Define peak ranges for integration
peak_ranges = [[1, 1.4], [2, 2.5]]

# Normalize the integral for the first signal to 3
nmr.normalize('integral', value=3, limits=[0.9, 1.5], apply=True)

# Calculate the integral curves
integral_curves = [
    nmr.integrate_spectra(x_limits=curr_peak) for curr_peak in peak_ranges]
integral_limits = [[curr_curve.columns[0], curr_curve.columns[-1]] for curr_curve in integral_curves]
integral_values = [curr_curve.iloc[0, -1] for curr_curve in integral_curves]

# plot the spectrum itself
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Chemical shift [ppm]')
ax1.set_ylabel('Intensity (normalized)')
ax1.set_xlim([5, 0])
ax1.plot(nmr.spectral_data.columns, nmr.spectral_data.T)

# plot the integral curves
ax2 = ax1.twinx()
ax2.set_ylabel('Integral (normalized)')
for curr_curve in integral_curves:
    ax2.plot(curr_curve.columns, -(curr_curve.T.values-curr_curve.values.max()), c='r')

# Add Integral values and limits to the plot.
for curr_lims, curr_val in zip(integral_limits, integral_values):
    ax1.text(np.mean(curr_lims), -0.1, '{0:.2f}'.format(curr_val), rotation='vertical',
              horizontalalignment='center', verticalalignment='center')
    ax1.axvline(curr_lims[0], c='g', ls='--')
    ax1.axvline(curr_lims[1], c='g', ls='--')
