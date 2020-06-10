# -*- coding: utf-8 -*-
"""
Provides a class for spectroscopic data preprocessing and analysis.
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
from scipy.integrate import cumtrapz
from sklearn.decomposition import PCA

# import own packagaes ######################
import pyPreprocessing.baseline_correction as baseline_correction
import pyPreprocessing.smoothing as smooth_data
import pyPreprocessing.transform as transform
import pyRegression.linear_regression as l_reg
#############################################


class spectroscopy_data:
    def __init__(self, spectral_data):
        self.spectral_data = spectral_data

        self.wavenumbers = self.spectral_data.columns.to_numpy()

        self.baseline_data = {
            'SNIP': pd.DataFrame(
                np.zeros_like(self.spectral_data.values),
                index=self.spectral_data.index,
                columns=self.spectral_data.columns),
            'ALSS': pd.DataFrame(
                np.zeros_like(self.spectral_data.values),
                index=self.spectral_data.index,
                columns=self.spectral_data.columns),
            'iALSS': pd.DataFrame(
                np.zeros_like(self.spectral_data.values),
                index=self.spectral_data.index,
                columns=self.spectral_data.columns),
            'drPLS': pd.DataFrame(
                np.zeros_like(self.spectral_data.values),
                index=self.spectral_data.index,
                columns=self.spectral_data.columns),
            'ModPoly': pd.DataFrame(
                np.zeros_like(self.spectral_data.values),
                index=self.spectral_data.index,
                columns=self.spectral_data.columns),
            'IModPoly': pd.DataFrame(
                np.zeros_like(self.spectral_data.values),
                index=self.spectral_data.index,
                columns=self.spectral_data.columns),
            'none': pd.DataFrame(
                np.zeros_like(self.spectral_data.values),
                index=self.spectral_data.index,
                columns=self.spectral_data.columns)}

    def check_active_spectra(self, active_spectra):
        if active_spectra is None:
            active_spectra = self.spectral_data
        return active_spectra

###############################
# preprocessing methods
###############################

    def mean_center(self, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        mean_centered_data = pd.DataFrame(
            preprocessing.scale(active_spectra, axis=0, with_std=False),
            index=active_spectra.index, columns=active_spectra.columns)
        return mean_centered_data.round(decimals=6)

    def standard_normal_variate(self, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        SNV_scaled_data = pd.DataFrame(active_spectra.subtract(
            active_spectra.mean(axis=1), axis=0).divide(
                active_spectra.std(axis=1), axis=0),
                index=active_spectra.index, columns=active_spectra.columns)
        return SNV_scaled_data.round(decimals=6)

    def clip_wavenumbers(self, wn_limits, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        wn_limits = np.array(wn_limits)

        lower_wn = wn_limits[:, 0]
        upper_wn = wn_limits[:, 1]

        lower_wn = lower_wn[np.argsort(-lower_wn)]
        upper_wn = upper_wn[np.argsort(-upper_wn)]

        closest_index_to_lower_wn = np.argmin(
            np.abs(active_spectra.columns.values[:, np.newaxis]-lower_wn),
            axis=0)
        closest_index_to_upper_wn = np.argmin(
            np.abs(active_spectra.columns.values[:, np.newaxis]-upper_wn),
            axis=0)

        clipping_index = np.concatenate(
            np.array(
                [np.r_[closest_index_to_upper_wn[ii]:
                       closest_index_to_lower_wn[ii]+1]
                 for ii in np.arange(len(closest_index_to_lower_wn))]))

        clipped_data = active_spectra.iloc[:, clipping_index]
        return clipped_data

    def clip_samples(self, x_limits=None, y_limits=None, z_limits=None,
                     active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        x_clipping_mask = self.generate_sample_clipping_mask(active_spectra,
                                                             x_limits, 0)
        y_clipping_mask = self.generate_sample_clipping_mask(active_spectra,
                                                             y_limits, 1)
        z_clipping_mask = self.generate_sample_clipping_mask(active_spectra,
                                                             z_limits, 2)

        clipped_data = active_spectra.loc[(x_clipping_mask, y_clipping_mask,
                                           z_clipping_mask)]
        return clipped_data

    def integrate_image(self, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        hyperspectral_image_integrated = pd.DataFrame(
            -cumtrapz(active_spectra, x=active_spectra.columns+0, initial=0),
            index=active_spectra.index, columns=active_spectra.columns
            ).round(decimals=6)
        return hyperspectral_image_integrated

    def smoothing(self, mode, active_spectra=None, **kwargs):
        active_spectra = self.check_active_spectra(active_spectra)

        smoothed_data = pd.DataFrame(smooth_data.smoothing(
            active_spectra.values, mode=mode, **kwargs),
            index=active_spectra.index,
            columns=active_spectra.columns).round(decimals=6)

        return smoothed_data

    def normalize(self, mode, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        normalize_modes = ['total_intensity']
        assert mode in normalize_modes, 'normalize mode unknown'

        if mode == normalize_modes[0]:  # total_intensity
            normalized_data = pd.DataFrame(transform.normalize(
                active_spectra.values, mode,
                x_data=active_spectra.columns.to_numpy()),
                index=active_spectra.index, columns=active_spectra.columns)
            
        return -normalized_data

    # needs to be adapted to fit all baseline correction modes
    def baseline_correction(self, lam=100, lam_1=100, p=0.01, eta=0.5,
                            n_iter=100, conv_crit=0.001, active_spectra=None,
                            alg='SNIP', wavenumbers=[], poly_order=5):
        active_spectra = self.check_active_spectra(active_spectra)

        baseline_types = ['ALSS', 'iALSS', 'drPLS', 'SNIP', 'ModPoly',
                          'IModPoly']
        assert alg in baseline_types, 'baseline type unknown'

        if alg == baseline_types[0]:  # ALSS
            self.baseline_data[alg] = pd.DataFrame(
                baseline_correction.generate_baseline(
                    active_spectra.values, baseline_types[0], lam=lam, p=p,
                    n_iter=n_iter, conv_crit=conv_crit),
                index=active_spectra.index, columns=active_spectra.columns)
        elif alg == baseline_types[1]:  # iALSS
            self.baseline_data[alg] = pd.DataFrame(
                baseline_correction.generate_baseline(
                    active_spectra.values, baseline_types[1],
                    wavenumbers=active_spectra.columns.to_numpy(), lam=lam,
                    lam_1=lam_1, p=p, n_iter=n_iter, conv_crit=conv_crit),
                index=active_spectra.index, columns=active_spectra.columns)
        elif alg == baseline_types[2]:  # drPLS
            self.baseline_data[alg] = pd.DataFrame(
                baseline_correction.generate_baseline(
                    active_spectra.values, baseline_types[2], lam=lam, eta=eta,
                    n_iter=n_iter, conv_crit=conv_crit),
                index=active_spectra.index, columns=active_spectra.columns)
        elif alg == baseline_types[3]:  # SNIP
            self.baseline_data[alg] = pd.DataFrame(
                baseline_correction.generate_baseline(
                    active_spectra.values, baseline_types[3], n_iter=n_iter),
                index=active_spectra.index, columns=active_spectra.columns)
        elif alg in baseline_types[4:6]:  # ModPoly, IModPoly
            self.baseline_data[alg] = pd.DataFrame(
                baseline_correction.generate_baseline(
                    active_spectra.values, baseline_types[4],
                    wavenumbers=wavenumbers, n_iter=n_iter,
                    poly_order=poly_order),
                index=active_spectra.index, columns=active_spectra.columns)

        corrected_data = active_spectra - self.baseline_data[alg]
        return corrected_data.round(decimals=6)

    def generate_sample_clipping_mask(self, active_spectra, limits, dimension):
        if limits is not None:
            limits = np.array(limits)
            lower_index = limits[:, 0]
            lower_index = lower_index[np.argsort(lower_index)]
            upper_index = limits[:, 1]
            upper_index = upper_index[np.argsort(upper_index)]
            lower_clipping = np.empty(
                (len(lower_index), len(active_spectra.index)), dtype=bool)
            upper_clipping = np.empty(
                (len(upper_index), len(active_spectra.index)), dtype=bool)

            for ii, (curr_lower_index, curr_upper_index) in enumerate(
                    zip(lower_index, upper_index)):
                lower_clipping[ii] = active_spectra.index.get_level_values(
                    dimension) >= curr_lower_index
                upper_clipping[ii] = active_spectra.index.get_level_values(
                    dimension) <= curr_upper_index

            clipping_mask = np.sum(lower_clipping*upper_clipping, axis=0,
                                   dtype=bool)
        else:
            clipping_mask = np.full(len(active_spectra.index), True)
        return clipping_mask

####################################
# spectrum characteristics methods
####################################

    def mean_spectrum(self, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        mean_spectrum = active_spectra.mean()
        return mean_spectrum.round(decimals=6)

    def max_spectrum(self, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        max_spectrum = active_spectra.max()
        return max_spectrum

    def min_spectrum(self, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        min_spectrum = active_spectra.min()
        return min_spectrum

    def std(self, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        std = active_spectra.std()
        return std.round(decimals=6)

####################################
# spectrum analysis methods
####################################

    def principal_component_analysis(self, pca_components,
                                     active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        pca = PCA(n_components=pca_components)
        scores = pca.fit_transform(active_spectra)
        loadings = pca.components_.T  # * np.sqrt(pca.explained_variance_)

        list_pc_components = ['pc' + str(ii)
                              for ii in np.arange(pca_components)]

        return {'scores': pd.DataFrame(scores, index=active_spectra.index,
                                       columns=list_pc_components),
                'loadings': pd.DataFrame(loadings.T, index=list_pc_components,
                                         columns=active_spectra.columns),
                'explained_variance': pca.explained_variance_ratio_}

    def reference_spectra_fit(self, reference_data, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        reference_components = reference_data.shape[0]
        list_reference_components = ['comp' + str(ii) for ii in np.arange(
            reference_components)]

        fit_coeffs = pd.DataFrame(
            np.empty((len(active_spectra.index), len(reference_data.index))),
            index=active_spectra.index, columns=list_reference_components)
        # self.fit_results = np.empty(len(active_spectra.index),dtype=object)

        for ii in tqdm(np.arange(len(active_spectra.index))):
            current_fit_result = l_reg.dataset_regression(
                active_spectra.iloc[ii, :], reference_data)
            fit_coeffs.iloc[ii, :] = current_fit_result
            # self.fit_results[ii] = current_fit_result
        return fit_coeffs

#     def find_peaks(self,active_spectra=None):#is still experimental
#         active_spectra = self.check_active_spectra(active_spectra)
#
#         processed_data = [find_peaks(row, height=200, prominence=100)
#                           for row in active_spectra.values]
#
#         return processed_data

####################################
# export methods
####################################
    def export_spectra(self, export_path, export_name, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        active_spectra.to_csv(export_path + export_name + '.txt', sep='\t',
                              header=True)
