# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:07:42 2020

@author: aso
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyRegression.nonlinear_regression import calc_function

from pyAnalytics.hplc_data import hplc_data
from pyAnalytics.hplc_calibration import hplc_calibration

class hplc_prediction():
    def __init__(self, samples, calibrations):
        self.samples = samples
        self.calibrations = calibrations
    
    def simple_prediction(self):
        predictions = np.zeros((len(self.samples), len(self.calibrations)))
        for ii, sample in enumerate(self.samples):
            for jj, calibration in enumerate(self.calibrations):
                prediction = np.dot(
                        np.linalg.inv(np.dot(calibration.K.T, calibration.K)),
                        np.dot(calibration.K.T,
                               sample.integrate_all_data(time_limits=calibration.time_limits, wavelength_limits=calibration.wavelength_limits).values[np.newaxis].T))
                predictions[ii, jj] = prediction.item()

        return predictions
    
    def advanced_prediction(self):
        # both calibrations must have equal time and wavelength range, not the case yet!
        Ks = []
        for calibration in self.calibrations:
            Ks.append(np.squeeze(calibration.K))
        Ks = np.array(Ks).T

        predictions = np.zeros((len(self.samples), len(self.calibrations)))        
        for ii, sample in enumerate(self.samples):
            for jj, calibration in enumerate(self.calibrations):
                sample_cropped = sample.crop_data(time_limits=calibration.time_limits, wavelength_limits=calibration.wavelength_limits)
                
                concentrations = np.dot(np.linalg.inv(np.dot(Ks.T,Ks)), np.dot(Ks.T, sample_cropped.T))
                concentrations = np.trapz(np.squeeze(concentrations), x=sample_cropped.index)
                print(concentrations.shape)
                #concentrations = pd.Series(np.squeeze(concentrations), index=sample_cropped.index)
                predictions[ii, jj] = concentrations[ii]
        return predictions
    
#    def advanced_prediction(self):
#        predictions = []
#        for sample in self.samples:
#            sample_cropped = sample.crop_data(time_limits=self.calibrations.time_limits, wavelength_limits=self.calibrations.wavelength_limits)
#            
#            concentrations = np.dot(np.linalg.inv(np.dot(self.calibrations.K.T,self.calibrations.K)), np.dot(self.calibrations.K.T, sample_cropped.T))
#            concentrations = np.trapz(np.squeeze(concentrations), x=sample_cropped.index)
#            #concentrations = pd.Series(np.squeeze(concentrations), index=sample_cropped.index)
#            predictions.append(concentrations)
#        return predictions

# unknown_concentrations = np.dot(np.linalg.inv(np.dot(K.T,K)), np.dot(K.T, unknown_samples.T))
if __name__ == "__main__":

    # calculate simulated hplc calibration data
    wavenumbers = np.linspace(200, 400, 201)
    uv_spectrum = calc_function(
            wavenumbers, [4, 200, 0, 15], 'Gauss') + calc_function(
                    wavenumbers, [0.8, 250, 0, 15], 'Gauss')
    uv_spectrum_2 = calc_function(
            wavenumbers, [4, 200, 0, 15], 'Gauss') + calc_function(
                    wavenumbers, [1.3, 275, 0, 12], 'Gauss')

    times = np.linspace(0,10,1001)
    chromatogram = calc_function(times, [2.2,4.3,0,0.2], 'Gauss')
    chromatogram_2 = calc_function(times, [2.2,5,0,0.2], 'Gauss')

    concentrations = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    calibration = concentrations[:,np.newaxis,np.newaxis] * uv_spectrum * chromatogram[:,np.newaxis]
    noise = np.random.standard_normal(calibration.shape)*0.05
    calibration = calibration + noise
    
    calibration_2 = concentrations[:,np.newaxis,np.newaxis] * uv_spectrum_2 * chromatogram_2[:,np.newaxis]
    noise = np.random.standard_normal(calibration_2.shape)*0.05
    calibration_2 = calibration_2 + noise

    calibration_dfs = []
    for curr_data in calibration:
        calibration_dfs.append(pd.DataFrame(curr_data, index=times, columns=wavenumbers))
        
    calibration_2_dfs = []
    for curr_data in calibration_2:
        calibration_2_dfs.append(pd.DataFrame(curr_data, index=times, columns=wavenumbers))
    
    # calculate samples with unknown concentration
    unknown_sample_1 = pd.DataFrame(
            0.35 * uv_spectrum * chromatogram[:,np.newaxis] + 0.2 * uv_spectrum_2 * chromatogram_2[:,np.newaxis], index=times, columns=wavenumbers)
    noise = np.random.standard_normal(unknown_sample_1.shape)*0.05
    unknown_sample_1 = unknown_sample_1 + noise
    unknown_sample_1 = hplc_data('DataFrame', data=unknown_sample_1)

    unknown_sample_2 = pd.DataFrame(
            0.27 * uv_spectrum * chromatogram[:,np.newaxis] + 0.55 * uv_spectrum_2 * chromatogram_2[:,np.newaxis], index=times, columns=wavenumbers)
    noise = np.random.standard_normal(unknown_sample_2.shape)*0.05
    unknown_sample_2 = unknown_sample_2 + noise
    unknown_sample_2 = hplc_data('DataFrame', data=unknown_sample_2)
    
    unknown_samples = [unknown_sample_1, unknown_sample_2]

    # generate hplc_calibration instance with simulated calibration data and
    # predict sample concentration with simulated calibration data
    calibration = hplc_calibration('DataFrame',
                                   calibration_data=calibration_dfs,
                                   concentrations=concentrations,
                                   time_limits=[3, 5.8],
                                   wavelength_limits=[225, 275])
    
    calibration_2 = hplc_calibration('DataFrame',
                                     calibration_data=calibration_2_dfs,
                                     concentrations=concentrations,
                                     time_limits=[3.7, 6.5],
                                     wavelength_limits=[250, 300])
    
    K = calibration.K
    
    predicted_concentrations = hplc_prediction(unknown_samples, [calibration, calibration_2])
    unknown_concentrations_simple = predicted_concentrations.simple_prediction()
    unknown_concentrations_advanced = predicted_concentrations.advanced_prediction()

    #K_uni = calibration.calculate_calibration(time_limits=[2, 8],
    #                                          wavelength_limits=[250, 250])
    #unknown_concentrations_uni = predicted_concentrations.simple_prediction()

    print('Correct_concentrations:\n0.35, 0.2\n0.27, 0.55')
    print('Predicted concentrations (multivariate, simple):\n', unknown_concentrations_simple)
    print('Predicted concentrations (multivariate, advanced):', unknown_concentrations_advanced)
    #print('Predicted concentrations (univariate):', unknown_concentrations_uni)

    # plot some data
    plt.figure(0)
    plt.plot(wavenumbers, calibration.calibration_data[4].raw_data.loc[4.3, :],
             wavenumbers, calibration_2.calibration_data[4].raw_data.loc[5, :])

    plt.figure(1)
    plt.plot(times, unknown_sample_1.raw_data.loc[:,200])

    plt.figure()
    plt.plot(K)