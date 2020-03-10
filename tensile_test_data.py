# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:31:29 2020

@author: aso
"""

import numpy as np
import pandas as pd
from pyRegression.linear_regression import lin_reg_all_sections
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import sys
from tqdm import tqdm
from copy import deepcopy

class tensile_test():
    def __init__(self, import_file):
        self.import_file = import_file
        self.import_data()
        
        #################### Define constants #################################
        self.R_SQUARED_LOWER_LIMIT = 0.995
        self.LOWER_STRAIN_LIMIT = 0
        self.UPPER_STRAIN_LIMIT = 50

        
        # Instantiate empty DataFrames for results 
        self.emod_df = pd.DataFrame(None)
        self.strength_df = pd.DataFrame(None)
        self.elongation_at_break_df = pd.DataFrame(None)
        self.toughness_df = pd.DataFrame(None)
        self.linear_limit_df = pd.DataFrame(None)
    
    def import_data(self):
        try:
            # Read excel file
            raw_excel = pd.ExcelFile(self.import_file)

            # Save sheets starting with the third into list
            self.raw_original = []
            for sheet_name in raw_excel.sheet_names[3:]:
                self.raw_original.append(raw_excel.parse(sheet_name,header=2,names=['Dehnung','Standardkraft','Zugspannung']))

            # copy original raw data and use this copy for further processing
            self.raw = deepcopy(self.raw_original)
        except Exception as e:
            print('Error while file import in line ' +
                  str(sys.exc_info()[2].tb_lineno) + ': ', e)
            pass
    
    def e_modulus(self):
        # Data pre-processing for E modulus calculation
        
        emodules = []
        slope_limits =[]
        intercept_limits = []
        linear_limits = []
        
        self.SAV_GOL_WINDOW = 1001
        self.SAV_GOL_POLYORDER = 2
        
        self.raw_processed = []
        global_min_stress = 0
        global_max_stress = 0
        global_min_strain = 0
        global_max_strain = 0
        for sample in self.raw:
            sample = self.streamline_data(sample)
            self.raw_processed.append(self.smooth_data(sample,self.SAV_GOL_WINDOW,self.SAV_GOL_POLYORDER))
            if sample['Zugspannung'].max() > global_max_stress:
                global_max_stress = sample['Zugspannung'].max()
            if sample['Dehnung'].max() > global_max_strain:
                global_max_strain = sample['Dehnung'].max()
            if sample['Zugspannung'].min() < global_min_stress:
                global_min_stress = sample['Zugspannung'].min()
            if sample['Dehnung'].min() < global_min_strain:
                global_min_strain = sample['Dehnung'].min()
        
        self.raw_processed_cropped = []
        max_strain_cropped = 0
        for sample in self.raw_processed:
            sample_cropped = sample.copy()
            # Extract relevant strain range for youngs modulus
            indexNames = sample_cropped[(sample_cropped['Dehnung']<=self.LOWER_STRAIN_LIMIT)].index 
            indexNames_2 = sample_cropped[(sample_cropped['Dehnung']>=self.UPPER_STRAIN_LIMIT)].index 
            sample_cropped.drop(indexNames , inplace=True)
            sample_cropped.drop(indexNames_2 , inplace=True)
            sample_cropped.reset_index(drop=True,inplace=True)
        
            #do regression and append results to sample DataFrame
            regression_results,curr_linear_limit,curr_linear_limit_stress,curr_slope_at_limit,curr_intercept_at_limit = lin_reg_all_sections(sample_cropped.loc[:,'Dehnung'].values,sample_cropped.loc[:,'Zugspannung'].values,r_squared_limit = self.R_SQUARED_LOWER_LIMIT,mode = 'both')
            sample_cropped = pd.concat([sample_cropped,regression_results],axis=1)
                
            slope_limits.append(curr_slope_at_limit)
            intercept_limits.append(curr_intercept_at_limit)
            linear_limits.append(curr_linear_limit)

            # Calculate youngs modulus
            emodule = np.around(curr_slope_at_limit,decimals=3)*100#gradient[0]*100

            # Save result in list 
            emodules.append(emodule.round(3))
            self.raw_processed_cropped.append(sample_cropped)
                
            if sample_cropped['Dehnung'].iloc[-1] > max_strain_cropped:
                max_strain_cropped = sample_cropped['Dehnung'].iloc[-1]
        
        return emodules
    
    def strength(self):
        strengths = []
        for sample in self.raw:
            strengths.append(np.around(sample['Zugspannung'].max(),decimals=1))
        return strengths
    
    def toughness(self):
        toughnesses = []
        for sample in self.raw:
            toughnesses.append(np.around(np.trapz(sample['Zugspannung'],x=sample['Dehnung'])/100,decimals=1))
        return toughnesses
    
    def elongation_at_break(self):
        elongations_at_break = []
        for sample in self.raw:
            elongations_at_break.append(np.around(sample['Dehnung'].at[sample['Zugspannung'].idxmax()],decimals=1))
        return elongations_at_break
    
    def streamline_data(self, sample):
        sample.drop_duplicates('Dehnung', keep='first', inplace=True)
        sample.sort_values(by=['Dehnung'], inplace=True)
        sample.dropna(inplace=True)

        return sample

    def smooth_data(self, sample, window, poly_order):
        smoothed_sample = sample.loc[:, 'Zugspannung'].values
    
        itp = interp1d(sample['Dehnung'], smoothed_sample, kind='linear')
        strain_interpolated = np.linspace(sample['Dehnung'].values[0], sample['Dehnung'].values[-1], 10000)
        stress_interpolated = itp(strain_interpolated)

        smoothed_sample = np.concatenate((-stress_interpolated[::-1]+2*stress_interpolated[0], stress_interpolated, -stress_interpolated[::-1]+2*stress_interpolated[-1]))

        smoothed_sample = savgol_filter(smoothed_sample, window, poly_order, mode='nearest')
        smoothed_sample = np.split(smoothed_sample, 3)[1]
    
        sample = pd.DataFrame(list(zip(strain_interpolated, smoothed_sample)), columns=['Dehnung', 'Zugspannung'])
    
        return sample

file = r'Z:\Charakterisierungen und Messungen\Python\zz_not_yet_in_git\04_Zugversuch-Auswertung\62\62.xls'
#file = r'Z:\Charakterisierungen und Messungen\Python\zz_not_yet_in_git\04_Zugversuch-Auswertung\62\Mappe1.xlsx'
tensile_test = tensile_test(file)
emodules = tensile_test.e_modulus()
strengths = tensile_test.strength()
toughnesses = tensile_test.toughness()
elongations_at_break = tensile_test.elongation_at_break()