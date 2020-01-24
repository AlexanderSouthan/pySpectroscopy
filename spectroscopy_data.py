# -*- coding: utf-8 -*-
"""
Provides a class for spectroscopic data preprocessing and analysis.

@author: Snijderfrey
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter  # ,find_peaks
from sklearn.decomposition import PCA

# import own packagaes ######################
import preprocessing.baseline_correction as baseline_correction
import regression.nonlinear_regression as nl_reg
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

    def savitzky_golay(self, deriv=0, savgol_points=9, poly_order=2,
                       active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        sav_gol_filtered_data = pd.DataFrame(
            savgol_filter(
                active_spectra, 1+2*savgol_points, poly_order, deriv=deriv,
                axis=1), index=active_spectra.index,
            columns=active_spectra.columns).round(decimals=6)
        return sav_gol_filtered_data

    def median_filter(self, window=5, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        edge_value_count = int((window-1)/2)
        median_filtered_data = active_spectra.rolling(
            window, axis=1, center=True).median().iloc[
                :, edge_value_count:-edge_value_count]

        return median_filtered_data.round(decimals=6)

    def pca_smoothing(self, pca_components=3, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        pca_results = self.principal_component_analysis(
            pca_components, active_spectra=active_spectra)
        # pca_result is also calculated in multi2monochrome,
        # possibly it can be bundled in one place

        reconstructed_pca_image = pd.DataFrame(
            np.dot(pca_results['scores'], pca_results['loadings'])
            + self.mean_spectrum(active_spectra=active_spectra).values,
            index=active_spectra.index, columns=active_spectra.columns)

        return reconstructed_pca_image

    def integrate_image(self, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        hyperspectral_image_integrated = pd.DataFrame(
            -cumtrapz(active_spectra, x=active_spectra.columns+0, initial=0),
            index=active_spectra.index, columns=active_spectra.columns
            ).round(decimals=6)
        return hyperspectral_image_integrated

    def baseline_correction(self, lam=100, lam_1=100, p=0.01, eta=0.5,
                            n_iter=100, conv_crit=0.001, active_spectra=None,
                            alg='SNIP'):
        active_spectra = self.check_active_spectra(active_spectra)

        baseline_types = ['ALSS', 'iALSS', 'drPLS', 'SNIP']
        assert alg in baseline_types, 'baseline type unknown'

        if alg == baseline_types[0]:
            self.baseline_data[alg] = pd.DataFrame(
                baseline_correction.generate_baseline(
                    active_spectra.values, baseline_types[0], lam=lam, p=p,
                    n_iter=n_iter, conv_crit=conv_crit),
                index=active_spectra.index, columns=active_spectra.columns)
        elif alg == baseline_types[1]:
            self.baseline_data[alg] = pd.DataFrame(
                baseline_correction.generate_baseline(
                    active_spectra.values, baseline_types[1],
                    wavenumbers=active_spectra.columns.to_numpy(), lam=lam,
                    lam_1=lam_1, p=p, n_iter=n_iter, conv_crit=conv_crit),
                index=active_spectra.index, columns=active_spectra.columns)
        elif alg == baseline_types[2]:
            self.baseline_data[alg] = pd.DataFrame(
                baseline_correction.generate_baseline(
                    active_spectra.values, baseline_types[2], lam=lam, eta=eta,
                    n_iter=n_iter, conv_crit=conv_crit),
                index=active_spectra.index, columns=active_spectra.columns)
        elif alg == baseline_types[3]:
            self.baseline_data[alg] = pd.DataFrame(
                baseline_correction.generate_baseline(
                    active_spectra.values, baseline_types[3], n_iter=n_iter),
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

    # PCA reconstruction is implemented in PCA_viewer and
    # should be transferred to here

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

    def reference_spectra_fit(self, reference_data, mode, maxiter,
                              initial_guess, lower_bounds, upper_bounds,
                              active_spectra=None):  # experimental
        active_spectra = self.check_active_spectra(active_spectra)

        reference_components = len(reference_data.index)
        if lower_bounds is None:
            lower_bounds = np.zeros(reference_components)
        if upper_bounds is None:
            upper_bounds = np.full(reference_components, 1000)
        list_reference_components = ['comp' + str(ii) for ii in np.arange(
            reference_components)]

        fit_coeffs = pd.DataFrame(
            np.empty((len(active_spectra.index), len(reference_data.index))),
            index=active_spectra.index, columns=list_reference_components)
        # self.fit_results = np.empty(len(active_spectra.index),dtype=object)

        for ii in tqdm(np.arange(len(active_spectra.index))):
            if mode == 'Evolutionary':
                boundaries = list(zip(lower_bounds, upper_bounds))
                current_fit_result = nl_reg.dataset_regression(
                    active_spectra.iloc[ii, :], reference_data,
                    boundaries=boundaries, max_iter=maxiter, alg='evo')
            elif mode == 'Levenberg-Marquardt':
                if initial_guess is None:
                    initial_guess = np.ones(reference_components)
                current_fit_result = nl_reg.dataset_regression(
                    active_spectra.iloc[ii, :], reference_data,
                    initial_guess=initial_guess, alg='lm')
            fit_coeffs.iloc[ii, :] = current_fit_result.x
            # self.fit_results[ii] = current_fit_result
        return fit_coeffs

#     def find_peaks(self,active_spectra=None):#is still experimental
#         active_spectra = self.check_active_spectra(active_spectra)
#
#         processed_data = [find_peaks(row, height=200, prominence=100)
#                           for row in active_spectra.values]
#
#         return processed_data
