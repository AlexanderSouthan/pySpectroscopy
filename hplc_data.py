# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:26:41 2019

@author: aso
"""

import numpy as np
import pandas as pd

############### import own modules###############
from pyAnalytics.measurement_parameters import measurement_parameters
import pyPreprocessing.baseline_correction
#################################################

class hplc_data:
    def __init__(self, mode, **kwargs):
        if mode == 'import':
            self.file_name = kwargs.get('file_name')
            directory = kwargs.get('directory', None)
            if directory is None: 
                self.directory = ''
            else:
                self.directory = directory + '/'

            self.import_from_file(self.file_name, self.directory)
        elif mode == 'DataFrame':
            self.raw_data = kwargs.get('data')
            self.wavelengths = self.raw_data.columns.to_numpy()
            measurement_params = pd.DataFrame(
                    [self.raw_data.index[1] - self.raw_data.index[0],
                     self.raw_data.index[0], self.raw_data.index[-1],
                     self.raw_data.columns[0], self.raw_data.columns[-1],
                     len(self.raw_data.columns), len(self.raw_data.index)],
                     index = ['interval', 'start_time', 'end_time',
                              'start_wavelength', 'end_wavelength',
                              'number_of_wavelength_points',
                              'number_of_time_points'])
            self.PDA_data = measurement_parameters(measurement_params)
        else:
            raise ValueError('No valid mode entered. Allowed modes are \'import\' and \'DataFrame\'.')

        self.time_data = self.raw_data.index.to_numpy()

    def import_from_file(self, file_name, directory):
        # next two lines in case function is called after mode was not 'import'
        self.file_name = file_name
        self.directory = directory

        PDA_data_raw = pd.read_csv(self.directory + self.file_name, sep='\t', skiprows=13, nrows=7)
        self.PDA_data = measurement_parameters(PDA_data_raw, parameter_names = ['interval','start_time','end_time','start_wavelength','end_wavelength','number_of_wavelength_points','number_of_time_points'])
        self.PDA_data.number_of_wavelength_points = int(self.PDA_data.number_of_wavelength_points)
        self.PDA_data.number_of_time_points = int(self.PDA_data.number_of_time_points)

        self.wavelengths= np.linspace(self.PDA_data.start_wavelength,self.PDA_data.end_wavelength,num = self.PDA_data.number_of_wavelength_points)

        self.raw_data = np.loadtxt(self.directory + self.file_name,skiprows=23,delimiter='\t')
        self.raw_data = pd.DataFrame(self.raw_data)
        self.raw_data.set_index(0,inplace=True)
        self.raw_data.columns = self.wavelengths

    def crop_data(self, time_limits=None, wavelength_limits=None,
                  active_data=None):
        active_data = self.check_active_data(active_data)

        if time_limits is None:
            time_limits = [self.PDA_data.start_time, self.PDA_data.end_time]
        elif None in time_limits:
            if time_limits[0] is None:
                time_limits = [self.PDA_data.start_time, time_limits[1]]
            elif time_limits[1] is None:
                time_limits = [time_limits[0], self.PDA_data.end_time]

        if wavelength_limits is None:
            wavelength_limits = [self.PDA_data.start_wavelength, self.PDA_data.end_wavelength]
        elif None in wavelength_limits:
            if wavelength_limits[0] is None:
                wavelength_limits = [self.PDA_data.start_wavelength, wavelength_limits[1]]
            elif wavelength_limits[1] is None:
                wavelength_limits = [wavelength_limits[0], self.PDA_data.end_wavelength]

        cropped_data = active_data.iloc[
                self.closest_index_to_value(
                        active_data.index, time_limits[0]):
                self.closest_index_to_value(
                        active_data.index,time_limits[1]) + 1,
                self.closest_index_to_value(
                        active_data.columns, wavelength_limits[0]):
                self.closest_index_to_value(
                        active_data.columns, wavelength_limits[1]) + 1
                ]

        return cropped_data

    def extract_elugram(self, wavelength, time_limits = None,
                        active_data=None, baseline_correction = False):
        active_data = self.check_active_data(active_data)

        elugram = self.crop_data(
                time_limits=time_limits,
                wavelength_limits=[wavelength, wavelength],
                active_data=active_data
                )

        #if baseline_correction:
            #baseline is not saved in any form and so far no method for full baseline correction exists
        #    baseline = pd.Series(pyPreprocessing.baseline_correction.drPLS_baseline(elugram.values[np.newaxis,:],100000000,0.8,100)[0,:].T,index = elugram.index)
        #    elugram = elugram - baseline

        return elugram

    def integrate_elugram(self, wavelength, time_limits=None,
                          active_data=None, baseline_correction=False):
        active_data = self.check_active_data(active_data)

        curr_elugram = self.extract_elugram(wavelength,
                                            time_limits=time_limits,
                                            baseline_correction=
                                            baseline_correction,
                                            active_data=active_data)
        elugram_integrated = np.trapz(curr_elugram, x=curr_elugram.index,
                                      axis=0).item()

        return elugram_integrated

    def integrate_all_data(self, mode='elugrams', time_limits=None,
                           wavelength_limits=None, active_data=None): # baseline_correction not yet integrated
        active_data = self.check_active_data(active_data)
        active_data = self.crop_data(time_limits=time_limits,
                                     wavelength_limits=wavelength_limits,
                                     active_data=active_data)
        
        if mode == 'elugrams':
            integration_axis = 0
            x_data = active_data.index
            result_index = active_data.columns
        elif mode == 'spectra':
            integration_axis = 1
            x_data = active_data.columns
            result_index = active_data.index
        
        integrated_data = np.trapz(active_data.values, x=x_data,
                                   axis=integration_axis)
        integrated_data = pd.Series(integrated_data,
                                    index=result_index)
        return integrated_data
        
    # next function should not stay in object
    def closest_index_to_value(self, array, value):
        return np.argmin(np.abs(array - value))
        
    def check_active_data(self, active_data):
        if active_data is None:
            active_data = self.raw_data
        return active_data
            
        