# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:36:30 2020

@author: aso
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pyRegression.nonlinear_regression import calc_function

from pyAnalytics.hplc_data import hplc_data


class hplc_calibration():
    def __init__(self, data_mode, **kwargs):
        if data_mode == 'DataFrame':
            calibration_data = kwargs.get('calibration_data')
            self.concentrations = kwargs.get('concentrations')[np.newaxis]
            
            self.calibration_data = []
            for curr_data in calibration_data:
                self.calibration_data.append(
                        hplc_data('DataFrame', data=curr_data))
        else:
            raise ValueError('No valid mode given.')
        
        self.wavelengths = self.calibration_data[0].wavelengths
        self.calibration_integrated = np.empty((len(self.wavelengths),
                                                len(np.squeeze(self.concentrations))))
        for index, curr_data in enumerate(self.calibration_data):
            self.calibration_integrated[:, index] = curr_data.integrate_all_elugrams()
            
        self.K = None
            
    def calibration(self, mode):
        if mode == 'multivariate':
            self.K = np.dot(np.dot(self.calibration_integrated, self.concentrations.T), np.linalg.inv(np.dot(self.concentrations, self.concentrations.T)))
        elif mode == 'univariate':
            pass # not yet integrated
        else:
            raise ValueError('No valid calibration mode entered.')
        return self.K
    
    def prediction(self, mode, sample):
        if self.K is None:
            self.calibration(mode)
            
        if mode == 'multivariate':
            prediction = np.dot(
                    np.linalg.inv(np.dot(self.K.T,self.K)),
                    np.dot(self.K.T,
                           sample.integrate_all_elugrams().values[np.newaxis].T))
        elif mode == 'univariate':
            pass # not yet integrated
        else:
            raise ValueError('No valid prediction mode entered.')
            
        return prediction

# calculate simulated hplc calibration data
wavenumbers = np.linspace(200,400,201)
uv_spectrum = calc_function(wavenumbers, [4,200,0,15], 'Gauss') + calc_function(wavenumbers, [0.8,250,0,15], 'Gauss')

times = np.linspace(0,10,1001)
chromatogram = calc_function(times, [2.2,4.3,0,0.2], 'Gauss')

concentrations = np.array([0.1, 0.2, 0.3, 0.4, 0.5])[np.newaxis]

calibration = np.squeeze(concentrations)[:,np.newaxis,np.newaxis] * uv_spectrum * chromatogram[:,np.newaxis]
noise = np.random.standard_normal(calibration.shape)*0.05
calibration = calibration + noise

calibration_dfs = []
for curr_data in calibration:
    calibration_dfs.append(pd.DataFrame(curr_data, index=times, columns=wavenumbers))
    
# generate hplc_calibration instance with simulated calibration data
calibration = hplc_calibration('DataFrame', calibration_data=calibration_dfs, concentrations=np.squeeze(concentrations))

K = calibration.calibration('multivariate')

# classical_calibration = calibration_integrated[51, :]
# slope_calibration = linregress(np.squeeze(concentrations),classical_calibration)[0]

print('Korrekter Wert: 0.89400097')
# print('Steigung klassische Kalibration:', slope_calibration)
print('Steigung aus K-Matrix:', K[51])


# calculate smaple with unknown concentration
unknown_sample_1 = pd.DataFrame(
        0.35 * uv_spectrum * chromatogram[:,np.newaxis], index=times, columns=wavenumbers)
noise = np.random.standard_normal(unknown_sample_1.shape)*0.1
unknown_sample_1 = unknown_sample_1 + noise
unknown_sample_1 = hplc_data('DataFrame', data=unknown_sample_1)

# predict sample concentration with simulated calibration data
unknown_concentrations = calibration.prediction('multivariate', unknown_sample_1)

# plot some data
plt.figure(0)
plt.plot(wavenumbers, calibration.calibration_data[0].raw_data.iloc[430, :])

plt.figure(1)
plt.plot(times, calibration.calibration_data[0].raw_data.iloc[:,51])

plt.figure()
plt.plot(wavenumbers, calibration.calibration_integrated)

plt.figure()
plt.plot(wavenumbers, K)
