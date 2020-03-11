# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:31:29 2020

@author: aso
"""

import numpy as np
import pandas as pd
from pyRegression.linear_regression import lin_reg_all_sections
from pyPreprocessing.smoothing import smoothing
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
# from scipy.interpolate import interp1d
import sys
# from tqdm import tqdm
from copy import deepcopy


class tensile_test():
    def __init__(self, import_file):
        self.import_file = import_file
        self.import_data()

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
                self.raw_original.append(
                    raw_excel.parse(sheet_name, header=2,
                                    names=['Dehnung', 'Standardkraft',
                                           'Zugspannung']))

            # copy original raw data and use this copy for further processing
            self.raw = deepcopy(self.raw_original)
        except Exception as e:
            print('Error while file import in line ' +
                  str(sys.exc_info()[2].tb_lineno) + ': ', e)
            pass

    def e_modulus(self, r_squared_lower_limit=0.995, lower_strain_limit=0,
                  upper_strain_limit=50, smoothing=True, **kwargs):

        emodules = []
        slope_limits = []
        intercept_limits = []
        linear_limits = []

        # Data pre-processing for E modulus calculation
        self.raw_processed = []
        global_min_stress = 0
        global_max_stress = 0
        global_min_strain = 0
        global_max_strain = 0
        for sample in self.raw:
            sample = self.streamline_data(sample)

            if sample['Zugspannung'].max() > global_max_stress:
                global_max_stress = sample['Zugspannung'].max()
            if sample['Dehnung'].max() > global_max_strain:
                global_max_strain = sample['Dehnung'].max()
            if sample['Zugspannung'].min() < global_min_stress:
                global_min_stress = sample['Zugspannung'].min()
            if sample['Dehnung'].min() < global_min_strain:
                global_min_strain = sample['Dehnung'].min()

            if smoothing:
                sav_gol_window = kwargs.get('sav_gol_window', 1001)
                sav_gol_polyorder = kwargs.get('sav_gol_polyorder', 2)
                sample = self.smooth_data(sample, sav_gol_window,
                                          sav_gol_polyorder)
            self.raw_processed.append(sample)

        self.raw_processed_cropped = []
        #max_strain_cropped = 0
        for sample in self.raw_processed:
            sample_cropped = sample.copy()
            # Extract relevant strain range for youngs modulus
            indexNames = sample_cropped[(
                sample_cropped['Dehnung'] <= lower_strain_limit)].index
            indexNames_2 = sample_cropped[(
                sample_cropped['Dehnung'] >= upper_strain_limit)].index
            sample_cropped.drop(indexNames, inplace=True)
            sample_cropped.drop(indexNames_2, inplace=True)
            sample_cropped.reset_index(drop=True, inplace=True)

            plt.plot(
                self.raw[0].loc[:, 'Dehnung'].values,self.raw[0].loc[:, 'Zugspannung'].values,
                sample_cropped.loc[:, 'Dehnung'].values,sample_cropped.loc[:, 'Zugspannung'].values)
            # do regression and append results to sample DataFrame
            regression_results, curr_linear_limit, curr_linear_limit_stress, curr_slope_at_limit, curr_intercept_at_limit = lin_reg_all_sections(sample_cropped.loc[:, 'Dehnung'].values, sample_cropped.loc[:, 'Zugspannung'].values, r_squared_limit = r_squared_lower_limit, mode = 'both')
            sample_cropped = pd.concat([sample_cropped, regression_results],
                                       axis=1)

            slope_limits.append(curr_slope_at_limit)
            intercept_limits.append(curr_intercept_at_limit)
            linear_limits.append(curr_linear_limit)

            # Calculate youngs modulus
            emodule = np.around(curr_slope_at_limit, decimals=3)*100

            # Save result in list
            emodules.append(emodule.round(3))
            self.raw_processed_cropped.append(sample_cropped)

            #if sample_cropped['Dehnung'].iloc[-1] > max_strain_cropped:
            #    max_strain_cropped = sample_cropped['Dehnung'].iloc[-1]

        return emodules

    def strength(self):
        strengths = []
        for sample in self.raw:
            strengths.append(
                np.around(sample['Zugspannung'].max(), decimals=1))
        return strengths

    def toughness(self):
        toughnesses = []
        for sample in self.raw:
            toughnesses.append(
                np.around(np.trapz(sample['Zugspannung'],
                                   x=sample['Dehnung'])/100, decimals=1))
        return toughnesses

    def elongation_at_break(self):
        elongations_at_break = []
        for sample in self.raw:
            elongations_at_break.append(
                np.around(sample['Dehnung'].at[sample['Zugspannung'].idxmax()],
                          decimals=1))
        return elongations_at_break

    def streamline_data(self, sample):
        sample.drop_duplicates('Dehnung', keep='first', inplace=True)
        sample.sort_values(by=['Dehnung'], inplace=True)
        sample.dropna(inplace=True)

        return sample

    def smooth_data(self, sample, window, poly_order):
        smoothed_sample = sample.loc[:, 'Zugspannung'].values
        x_coordinate = sample.loc[:, 'Dehnung'].values

        # print(smoothed_sample[np.newaxis].shape)
        # print(x_coordinate.shape)
        strain_interpolated, smoothed_sample = smoothing(
            smoothed_sample[np.newaxis], 'sav_gol', interpolate=True,
            point_mirror=True, x_coordinate=x_coordinate,
            savgol_points=window, poly_order=poly_order)

        sample = pd.DataFrame(list(zip(strain_interpolated, smoothed_sample.T)),
                              columns=['Dehnung', 'Zugspannung'])

        return sample

# file = r'Z:\Charakterisierungen und Messungen\Python\zz_not_yet_in_git\04_Zugversuch-Auswertung\62\62.xls'
# file = r'Z:\Charakterisierungen und Messungen\Python\zz_not_yet_in_git\04_Zugversuch-Auswertung\62\Mappe1.xlsx'
file = r'/home/almami/Alexander/Python_Skripte/yy_Not_yet_in_git/04_Zugversuch-Auswertung/62/62.xls'
tensile_test = tensile_test(file)
emodules = tensile_test.e_modulus()
strengths = tensile_test.strength()
toughnesses = tensile_test.toughness()
elongations_at_break = tensile_test.elongation_at_break()