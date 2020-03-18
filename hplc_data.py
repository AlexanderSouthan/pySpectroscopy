# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:26:41 2019

@author: aso
"""

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz

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

    def extract_elugram(self,wavelength,time_limits = None,baseline_correction = False):
        if time_limits is None:
            time_limits = [self.PDA_data.start_time, self.PDA_data.end_time]
        else:
            if time_limits[0] is None:
                time_limits = [self.PDA_data.start_time,time_limits[1]]
            elif time_limits[1] is None:
                time_limits = [time_limits[0],self.PDA_data.end_time]
                
        elugram = pd.Series(self.raw_data.iloc[self.closest_index_to_value(self.raw_data.index,time_limits[0]):
                                               self.closest_index_to_value(self.raw_data.index,time_limits[1]),
                                               self.closest_index_to_value(self.wavelengths,wavelength)],
                            index = self.raw_data.index[self.closest_index_to_value(self.raw_data.index,time_limits[0]):
                                                        self.closest_index_to_value(self.raw_data.index,time_limits[1])])
        
        if baseline_correction:
            #baseline is not saved in any form and so far no method for full baseline correction exists
            baseline = pd.Series(pyPreprocessing.baseline_correction.drPLS_baseline(elugram.values[np.newaxis,:],100000000,0.8,100)[0,:].T,index = elugram.index)
            elugram = elugram - baseline
            
            
        return elugram
    
    def integrate_elugram(self,wavelength,time_limits = None,baseline_correction = False):
        curr_elugram = self.extract_elugram(wavelength,time_limits = time_limits,baseline_correction = baseline_correction)
        elugram_integrated = pd.Series(cumtrapz(curr_elugram,x = curr_elugram.index,initial = 0,axis = 0),index = curr_elugram.index)
        
        return elugram_integrated
    
    def integrate_all_elugrams(self): # baseline_correction and time limits not yet integrated
        integrated_elugrams = np.trapz(self.raw_data.values, x=self.time_data, axis=0)
        return pd.Series(integrated_elugrams, index=self.wavelengths)
        
    #next function should not stay in object
    def closest_index_to_value(self,array,value):
        return np.argmin(np.abs(array - value))
        
        

            
        