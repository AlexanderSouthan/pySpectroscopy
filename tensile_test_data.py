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
import sys
import os
# from tqdm import tqdm
from copy import deepcopy


class tensile_test():
    def __init__(self, import_file, import_mode):
        self.import_file = import_file
        self.import_mode = import_mode
        self.import_data()
        # copy original raw data and use this copy for further processing
        self.raw = deepcopy(self.raw_original)
        
        self.e_modulus = None
        self.linear_limit = None
        self.strength = None
        self.toughness = None
        self.elongation_at_break = None
        

        # Instantiate empty DataFrames for results
        self.emod_df = pd.DataFrame(None)
        self.strength_df = pd.DataFrame(None)
        self.elongation_at_break_df = pd.DataFrame(None)
        self.toughness_df = pd.DataFrame(None)
        self.linear_limit_df = pd.DataFrame(None)

    def import_data(self):
        if self.import_mode == 'Marc_Stuhlmueller':
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
            except Exception as e:
                print('Error while file import in line ' +
                      str(sys.exc_info()[2].tb_lineno) + ': ', e)
            pass

        else:
            raise ValueError('No valid import mode entered. Allowed mode is'
                             ' \'Marc_Stuhlmueller\'.')

    def calc_e_modulus(self, r_squared_lower_limit=0.995, lower_strain_limit=0,
                  upper_strain_limit=50, smoothing=True, **kwargs):

        self.e_modulus = []
        self.slope_limit = []
        self.intercept_limit = []
        self.linear_limit = []
        self.r_squared_lower_limit = r_squared_lower_limit

        # Data pre-processing for E modulus calculation
        self.raw_processed = []
        for sample in self.raw:
            sample = self.streamline_data(sample)

            if smoothing:
                sav_gol_window = kwargs.get('sav_gol_window', 500)
                sav_gol_polyorder = kwargs.get('sav_gol_polyorder', 2)
                sample = self.smooth_data(sample, sav_gol_window,
                                          sav_gol_polyorder,
                                          data_points=kwargs.get(
                                                  'data_points', None))
            self.raw_processed.append(sample)

        self.raw_processed_cropped = []
        # max_strain_cropped = 0
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

            # do regression and append results to sample DataFrame
            regression_results, curr_linear_limit, curr_linear_limit_stress, curr_slope_at_limit, curr_intercept_at_limit = lin_reg_all_sections(sample_cropped.loc[:, 'Dehnung'].values, sample_cropped.loc[:, 'Zugspannung'].values, r_squared_limit = r_squared_lower_limit, mode = 'both')
            sample_cropped = pd.concat([sample_cropped, regression_results],
                                       axis=1)

            self.slope_limit.append(curr_slope_at_limit)
            self.intercept_limit.append(curr_intercept_at_limit)
            self.linear_limit.append(curr_linear_limit)

            # Calculate youngs modulus
            e_modulus = np.around(curr_slope_at_limit, decimals=3)*100

            # Save result in list
            self.e_modulus.append(e_modulus.round(3))
            self.raw_processed_cropped.append(sample_cropped)

        return (self.e_modulus, self.linear_limit)

    def calc_strength(self):
        self.strength = []
        for sample in self.raw:
            self.strength.append(
                    np.around(sample['Zugspannung'].max(), decimals=1))
        return self.strength

    def calc_toughness(self):
        self.toughness = []
        for sample in self.raw:
            self.toughness.append(
                    np.around(np.trapz(sample['Zugspannung'],
                                       x=sample['Dehnung'])/100, decimals=1))
        return self.toughness

    def calc_elongation_at_break(self):
        self.elongation_at_break = []
        for sample in self.raw:
            self.elongation_at_break.append(
                    np.around(sample['Dehnung'].at[sample[
                            'Zugspannung'].idxmax()],
                decimals=1))
        return self.elongation_at_break

    def generate_plots(self, export_path=None, **kwargs):
        if (self.e_modulus is None) or (self.linear_limit is None):
            r_squared_lower_limit = kwargs.get('r_squared_lower_limit', 0.995)
            lower_strain_limit = kwargs.get('lower_strain_limit', 0)
            upper_strain_limit=kwargs.get('upper_strain_limit', 15)
            smoothing = kwargs.get('smoothing', True)
            sav_gol_window = kwargs.get('sav_gol_window', 500)
            sav_gol_polyorder = kwargs.get('sav_gol_polyorder', 2)
            data_points=kwargs.get('data_points', 10000)

            self.calc_e_modulus(r_squared_lower_limit=r_squared_lower_limit,
                                lower_strain_limit=lower_strain_limit,
                                upper_strain_limit=upper_strain_limit,
                                smoothing=smoothing,
                                sav_gol_window=sav_gol_window,
                                sav_gol_polyorder=sav_gol_polyorder,
                                data_points=data_points)
            
        if self.strength is None:
            self.calc_strength()
        if self.toughness is None:
            self.calc_toughness()
        if self.elongation_at_break is None:
            self.calc_elongation_at_break()
        
        global_min_stress = 0
        global_max_stress = 0
        global_min_strain = 0
        global_max_strain = 0
        
        for sample in self.raw:
            if sample['Zugspannung'].max() > global_max_stress:
                global_max_stress = sample['Zugspannung'].max()
            if sample['Dehnung'].max() > global_max_strain:
                global_max_strain = sample['Dehnung'].max()
            if sample['Zugspannung'].min() < global_min_stress:
                global_min_stress = sample['Zugspannung'].min()
            if sample['Dehnung'].min() < global_min_strain:
                global_min_strain = sample['Dehnung'].min()
                
        max_strain_cropped = 0
        for sample in self.raw_processed_cropped:
            if sample['Dehnung'].iloc[-1] > max_strain_cropped:
                max_strain_cropped = sample['Dehnung'].iloc[-1]
                
        # Plot whole graph with regression line
        for ii, (raw_sample, processed_sample, el_at_break, strength,
                curr_linear_limit, curr_slope_limit, curr_intercept_limit
                ) in enumerate(zip(self.raw, self.raw_processed, self.elongation_at_break,
                self.strength, self.linear_limit, self.slope_limit, self.intercept_limit)):
            fig = plt.subplot(len(self.raw), 3, 1+ii*3)
            if ii==0: plt.title('Tensile test')
            if ii==int(len(self.raw)/2): plt.ylabel(r'$\sigma$ [kPa]')
            if ii==len(self.raw)-1: 
                plt.xlabel(r'$\epsilon$ [%]')
            else:
                fig.axes.xaxis.set_ticklabels([])
            plt.axvline(el_at_break,ls='--',c='k',lw=0.5)
            plt.axhline(strength,ls='--',c='k',lw=0.5)
            plt.plot(raw_sample['Dehnung'],raw_sample['Zugspannung'], linestyle='-')
            plt.plot(processed_sample['Dehnung'],processed_sample['Zugspannung'], linestyle='-',color='y')
            plt.plot(np.linspace(0,curr_linear_limit),curr_slope_limit*np.linspace(0,curr_linear_limit)+curr_intercept_limit, color='indianred')
            plt.xlim(global_min_strain,1.05*global_max_strain)
            plt.ylim(global_min_stress,1.05*global_max_stress)

        # Plot r-sqaured 
        plt.subplots_adjust(wspace = 0.5)
        for ii,(emod_sample,curr_linear_limit) in enumerate(zip(self.raw_processed_cropped,self.linear_limit)):
            fig = plt.subplot(len(self.raw), 3, 2+ii*3)
            if ii==0: plt.title('Coefficient of determination')
            if ii==int(len(self.raw)/2): plt.ylabel('$R^2$')
            if ii==len(self.raw)-1: 
                plt.xlabel(r'$\epsilon$ [%]')
            else:
                fig.axes.xaxis.set_ticklabels([])
            plt.yticks([0.975,1.0])
            plt.plot(emod_sample.loc[1:,'Dehnung'],emod_sample.loc[1:,'r_squared'], color='indianred')
            plt.axvline(curr_linear_limit,ls='--',c='k',lw=0.5)
            plt.axhline(self.r_squared_lower_limit,ls='--',c='k',lw=0.5)
            plt.xlim(0,max_strain_cropped)
            plt.ylim(0.95, 1)
            
        # Plot E modulus
        for ii,(emod_sample,curr_linear_limit) in enumerate(zip(self.raw_processed_cropped,self.linear_limit)):
            fig = plt.subplot(len(self.raw), 3, 3+ii*3)
            if ii==0: plt.title('E modulus')
            if ii==int(len(self.raw)/2): plt.ylabel('$E$ [kPa]')
            if ii==len(self.raw)-1: 
                plt.xlabel(r'$\epsilon$ [%]')
            else:
                fig.axes.xaxis.set_ticklabels([])
            #plt.ylim(0.95, 1)
            #plt.yticks([0.975,1.0])
            plt.plot(emod_sample.loc[1:,'Dehnung'],emod_sample.loc[1:,'slopes']*100, color='indianred')
            plt.axvline(curr_linear_limit,ls='--',c='k',lw=0.5)
            plt.xlim(0,max_strain_cropped)

        if export_path is not None:
            export_name = kwargs.get('export_name', 'Tensile test')
            # Save figure as svg
            if not os.path.exists(export_path + 'Plots/'):
                os.makedirs(export_path + 'Plots/')
            plt.savefig(export_path + 'Plots/' + export_name + '.png', dpi=500)
            #plt.clf()
            plt.close()

    def streamline_data(self, sample):
        sample.drop_duplicates('Dehnung', keep='first', inplace=True)
        sample.sort_values(by=['Dehnung'], inplace=True)
        sample.dropna(inplace=True)

        return sample

    def smooth_data(self, sample, window, poly_order, data_points=None):
        smoothed_sample = sample.loc[:, 'Zugspannung'].values
        x_coordinate = sample.loc[:, 'Dehnung'].values
        
        if data_points is None:
            data_points = int(10**np.ceil(np.log10(len(x_coordinate))))

        strain_interpolated, smoothed_sample = smoothing(
            smoothed_sample[np.newaxis], 'sav_gol', interpolate=True,
            point_mirror=True, x_coordinate=x_coordinate,
            savgol_points=window, poly_order=poly_order,
            data_points=data_points)

        sample = pd.DataFrame(
                list(zip(strain_interpolated, np.squeeze(smoothed_sample.T))),
                columns=['Dehnung', 'Zugspannung'])

        return sample

#file = r'Z:\Charakterisierungen und Messungen\Python\zz_not_yet_in_git\04_Zugversuch-Auswertung\62\62.xls'
file = r'Z:/Lehre/Studentische Arbeiten/02 Abgeschlossene Arbeiten/2019_Marc Stuhlmüller/Messdaten Zugversuche/03_Messdaten Zugversuche bis 15/23.xls'
# file = r'Z:\Charakterisierungen und Messungen\Python\zz_not_yet_in_git\04_Zugversuch-Auswertung\62\Mappe1.xlsx'
# file = r'/home/almami/Alexander/Python_Skripte/yy_Not_yet_in_git/04_Zugversuch-Auswertung/62/62.xls'
tensile_test = tensile_test(file, 'Marc_Stuhlmueller')
e_moduli, linear_limits = tensile_test.calc_e_modulus(upper_strain_limit=15, sav_gol_window=500, data_points=10000)
strengths = tensile_test.calc_strength()
toughnesses = tensile_test.calc_toughness()
elongations_at_break = tensile_test.calc_elongation_at_break()