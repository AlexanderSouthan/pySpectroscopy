# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:36:30 2020

@author: aso
"""

import numpy as np
import pandas as pd

from pyAnalytics.hplc_data import hplc_data


class hplc_calibration():
    def __init__(self, data_mode, time_limits=None, wavelength_limits=None,
                 **kwargs):
        if data_mode == 'DataFrame':
            calibration_data = kwargs.get('calibration_data')
            self.concentrations = kwargs.get('concentrations')[np.newaxis]
            
            self.calibration_data = []
            for curr_data in calibration_data:
                self.calibration_data.append(
                        hplc_data('DataFrame', data=curr_data))
        else:
            raise ValueError('No valid data_mode given.')

        self.time_limits = time_limits
        self.wavelength_limits = wavelength_limits
        self.calculate_calibration()
            
    def calculate_calibration(self, time_limits=None, wavelength_limits=None):
        if time_limits is not None:
            self.time_limits = time_limits
        if wavelength_limits is not None:
            self.wavelength_limits = wavelength_limits

        self.calibration_integrated = []
        for index, curr_data in enumerate(self.calibration_data):
            self.calibration_integrated.append(
                    curr_data.integrate_all_data(
                            time_limits=self.time_limits,
                            wavelength_limits=self.wavelength_limits))
        self.calibration_integrated = pd.DataFrame(
                self.calibration_integrated).T
        
        self.K = np.dot(
                np.dot(
                        self.calibration_integrated, self.concentrations.T
                        ),
                        np.linalg.inv(
                                np.dot(
                                        self.concentrations,
                                        self.concentrations.T)
                                ))
        self.K = pd.DataFrame(self.K, index=self.calibration_integrated.index)

        return self.K
