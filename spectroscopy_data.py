# -*- coding: utf-8 -*-
"""
Provides a class for spectroscopic data preprocessing and analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from tqdm import tqdm
from scipy.integrate import cumtrapz, trapz
# from sklearn.decomposition import PCA

from pyPreprocessing.baseline_correction import generate_baseline
from pyPreprocessing.smoothing import smoothing as smooth_data
import pyPreprocessing.transform as transform
import pyRegression.linear_regression as l_reg
from pyRegression.multivariate_regression import principal_component_regression


class spectroscopy_data:
    def __init__(self, data_source='import', **kwargs):
        """
        Initialize the spectroscopy_data instance.

        Parameters
        ----------
        data_source : str, optional
            The data source of the spectral data. Default is import which
            means that the data is imported from ASCII files. Alternatively,
            the data can be given in a DataFrame with 'DataFrame'.

        **kwargs for the different data_source values
        data_source == 'import'
            file_names : list of str
                A list of strings giving full paths to the files to be
                imported. The files must contain the spectral data in two
                columns. The first column must contain the wavenumber data and
                the second column the intensity data. All spectra must have
                identical wavenumber data. The file names will result as the
                index in the self.spectral_data DataFrame.
        data_source == 'DataFrame'
            spectral_data : pandas DataFrame
                A pandas DataFrame containing the spectral information. Each
                line contains one spectrum. The index may contain sample names,
                numbers, or e.g. coordinates from Raman maps. The column index
                must be the wavenumber/wavelength scale.

        Returns
        -------
        None.

        """
        self.data_source = data_source
        self.kwargs = kwargs

        self.__import_data()

        if self.spectral_data.columns[1] < self.spectral_data.columns[0]:
            self.spectral_data = self.spectral_data.iloc[:, ::-1]

        self.reset_processed_data()
        self.wavenumbers = self.spectral_data.columns.to_numpy()
        self.baseline_data = {}
        self.monochrome_data = {}

    def __import_data(self):
        if self.data_source == 'DataFrame':
            self.spectral_data = self.kwargs.get('spectral_data')

        elif self.data_source == 'import':
            self.file_names = self.kwargs.get('file_names')

            wavenumbers = np.fromfile(self.file_names[0], sep=' ')[::2]
            intensities = np.zeros((len(self.file_names), wavenumbers.size))
            for idx, curr_file in enumerate(tqdm(self.file_names)):
                intensities[idx] = np.fromfile(curr_file, sep=' ')[1::2]

            fname_index = pd.Series(self.file_names).str.rsplit(
                '/', n=1, expand=True)[1]
            self.spectral_data = pd.DataFrame(
                intensities, index=fname_index,
                columns=np.around(wavenumbers, 2))

        else:
            raise ValueError('Value of data_source not understood.')

    def reset_processed_data(self):
        self.spectral_data_processed = self.spectral_data.copy()

    def check_active_data(self, active_data):
        if active_data is None:
            active_data = self.spectral_data_processed
        return active_data

###############################
# preprocessing methods
###############################

    def mean_center(self, active_data=None):
        """
        Subtract the mean spectrum from each spectrum im self.spectral_data.

        Parameters
        ----------
        active_data : pandas DataFrame or None, optional
            None means that self.spectral_data_processed is used for baseline
            correction. Alternatively, a pandas DataFrame in the same format as
            self.spectral_data_processed can be passed that is used for the
            calculations. The default is None.

        Returns
        -------
        pandas DataFrame
            The mean centered spectral data.

        """
        active_data = self.check_active_data(active_data)

        mean_centered_data = pd.DataFrame(
            scale(active_data, axis=0, with_std=False),
            index=active_data.index, columns=active_data.columns).round(
                decimals=6)

        self.spectral_data_processed = mean_centered_data

        return mean_centered_data

    def standard_normal_variate(self, active_data=None):
        """
        Mean center spectra and scale to unit variance (SNV).

        Mean centering is perforemd by subtracting the mean spectrum from each
        spectrum. Scaling to unit variance is done by calculating the standard
        deviation at each wavenumber and subsequent division of each intensity
        by the respective standard deviation at that wavenumber. The resulting
        dataset has a mean of zero and a standrad deviation of one at each
        wavenumber.

        Parameters
        ----------
        active_data : pandas DataFrame or None, optional
            None means that self.spectral_data_processed is used for baseline
            correction. Alternatively, a pandas DataFrame in the same format as
            self.spectral_data_processed can be passed that is used for the
            calculations. The default is None.

        Returns
        -------
        pandas DataFrame
            The mean centered and scaled spectral data.

        """
        active_data = self.check_active_data(active_data)

        SNV_scaled_data = pd.DataFrame(active_data.subtract(
            active_data.mean(axis=1), axis=0).divide(
                active_data.std(axis=1), axis=0),
                index=active_data.index,
                columns=active_data.columns).round(decimals=6)

        self.spectral_data_processed = SNV_scaled_data

        return SNV_scaled_data

    def clip_wavenumbers(self, wn_limits, active_data=None):
        """
        Select certain wavenumber ranges from spectral data.

        Parameters
        ----------
        wn_limits : list of tuples
            A list containing tuples with two numbers. Each tuple gives a lower
            and an upper wavenumber defining a wavenumber range. When more than
            one tuple is given, the remaining spectral regions are not
            necessarily neighboring spectral regions in the original data.
        active_data : pandas DataFrame or None, optional
            None means that self.spectral_data_processed is used for baseline
            correction. Alternatively, a pandas DataFrame in the same format as
            self.spectral_data_processed can be passed that is used for the
            calculations. The default is None.

        Returns
        -------
        clipped_data : pandas DataFrame
            The spectral regions collected in one DataFrame defined by the
            wn_limits.

        """
        active_data = self.check_active_data(active_data)

        wn_limits = np.array(wn_limits)

        lower_wn = wn_limits[:, 0]
        upper_wn = wn_limits[:, 1]

        lower_wn = lower_wn[np.argsort(-lower_wn)]
        upper_wn = upper_wn[np.argsort(-upper_wn)]

        closest_index_to_lower_wn = np.argmin(
            np.abs(active_data.columns.values[:, np.newaxis]-lower_wn),
            axis=0)
        closest_index_to_upper_wn = np.argmin(
            np.abs(active_data.columns.values[:, np.newaxis]-upper_wn),
            axis=0)

        clipping_index = np.concatenate(
            np.array(
                [np.r_[closest_index_to_upper_wn[ii]:
                       closest_index_to_lower_wn[ii]+1]
                 for ii in np.arange(len(closest_index_to_lower_wn))]))

        clipped_data = active_data.iloc[:, clipping_index]

        self.spectral_data_processed = clipped_data

        return clipped_data

    def clip_samples(self, x_limits=None, y_limits=None, z_limits=None,
                     active_data=None):
        # This method is not good for this class. It assumes that the index
        # contains info only present for Raman images, so it must be improved
        # to accept all kinds on index values, such as sample names or simply
        # numbers.
        active_data = self.check_active_data(active_data)

        x_clipping_mask = self.generate_sample_clipping_mask(active_data,
                                                             x_limits, 0)
        y_clipping_mask = self.generate_sample_clipping_mask(active_data,
                                                             y_limits, 1)
        z_clipping_mask = self.generate_sample_clipping_mask(active_data,
                                                             z_limits, 2)

        clipped_data = active_data.loc[(x_clipping_mask, y_clipping_mask,
                                           z_clipping_mask)]

        self.spectral_data_processed = clipped_data

        return clipped_data

    def smoothing(self, mode, active_data=None, **kwargs):
        active_data = self.check_active_data(active_data)

        smoothed_data = pd.DataFrame(
            smooth_data(active_data.values, mode=mode, **kwargs),
            index=active_data.index, columns=active_data.columns).round(
                decimals=6)

        self.spectral_data_processed = smoothed_data

        return smoothed_data

    def normalize(self, mode, active_data=None):
        active_data = self.check_active_data(active_data)

        normalize_modes = ['total_intensity']
        assert mode in normalize_modes, 'normalize mode unknown'

        if mode == normalize_modes[0]:  # total_intensity
            normalized_data = pd.DataFrame(-transform.normalize(
                active_data.values, mode,
                x_data=active_data.columns.to_numpy()),
                index=active_data.index, columns=active_data.columns)

        self.spectral_data_processed = normalized_data

        return normalized_data

    def baseline_correction(self, mode='ModPoly', smoothing=True,
                            transform=False, active_data=None, **kwargs):
        """
        Correct baseline with methods from pyPreprocessing.baseline_correction.

        The details about the function arguments and kwargs can be found in the
        docstrings in pyPreprocessing.baseline_correction.

        Parameters
        ----------
        mode : str
            The baseline correction algorithm.
        smoothing : bool, optional
            True means the spectra are smoothed before baseline correction. The
            default is True.
        transform : bool, optional
            True means the spectra are transformed before baseline correction.
            The default is False.
        active_data : pandas DataFrame or None, optional
            None means that self.spectral_data is used for baseline correction.
            Alternatively, a pandas DataFrame in the same format as
            self.spectral_data can be passed that is used for the calculations.
            The default is None.
        **kwargs
            All kwargs necessary for the respective baseline correction mode.
            If wavenumbers are needed for the calculations, these do not need
            to be given because they are already known. If they are passed
            anyway, the passed values are ignored.

        Returns
        -------
        pd.DataFrame
            The spectral data with the subtracted baseline.

        """
        active_data = self.check_active_data(active_data)

        if mode in ['convex_hull', 'ModPoly', 'IModPoly', 'PPF', 'iALSS']:
            kwargs['wavenumbers'] = active_data.columns.to_numpy()

        self.baseline_data[mode] = pd.DataFrame(
            generate_baseline(active_data.values, mode, smoothing=True,
                              transform=False, **kwargs),
            index=active_data.index, columns=active_data.columns)

        corrected_data = (active_data -
                          self.baseline_data[mode]).round(decimals=6)

        self.spectral_data_processed = corrected_data

        return corrected_data

    def generate_sample_clipping_mask(self, active_data, limits, dimension):
        if limits is not None:
            limits = np.array(limits)
            lower_index = limits[:, 0]
            lower_index = lower_index[np.argsort(lower_index)]
            upper_index = limits[:, 1]
            upper_index = upper_index[np.argsort(upper_index)]
            lower_clipping = np.empty(
                (len(lower_index), len(active_data.index)), dtype=bool)
            upper_clipping = np.empty(
                (len(upper_index), len(active_data.index)), dtype=bool)

            for ii, (curr_lower_index, curr_upper_index) in enumerate(
                    zip(lower_index, upper_index)):
                lower_clipping[ii] = active_data.index.get_level_values(
                    dimension) >= curr_lower_index
                upper_clipping[ii] = active_data.index.get_level_values(
                    dimension) <= curr_upper_index

            clipping_mask = np.sum(lower_clipping*upper_clipping, axis=0,
                                   dtype=bool)
        else:
            clipping_mask = np.full(len(active_data.index), True)
        return clipping_mask

####################################
# spectrum characteristics methods
####################################

    def mean_spectrum(self, active_data=None):
        active_data = self.check_active_data(active_data)

        mean_spectrum = active_data.mean()
        return mean_spectrum.round(decimals=6)

    def max_spectrum(self, active_data=None):
        active_data = self.check_active_data(active_data)

        max_spectrum = active_data.max()
        return max_spectrum

    def min_spectrum(self, active_data=None):
        active_data = self.check_active_data(active_data)

        min_spectrum = active_data.min()
        return min_spectrum

    def std(self, active_data=None):
        active_data = self.check_active_data(active_data)

        std = active_data.std()
        return std.round(decimals=6)

####################################
# spectrum analysis methods
####################################

    def univariate_analysis(self, mode, active_data=None, **kwargs):
        active_data = self.check_active_data(active_data)

        modes = ['int_at_point', 'sig_to_base', 'sig_to_axis']

        if mode == modes[0]:  # 'int_at_point'
            if not modes[0] in self.monochrome_data:
                self.monochrome_data[modes[0]] = pd.DataFrame(
                    [], index=active_data.index)

            wn = kwargs.get('wn', 1000)

            curr_int_at_point = active_data.loc[
                :, active_data.columns[
                    np.argmin(np.abs(active_data.columns - wn))]]

            self.monochrome_data[modes[0]][wn] = curr_int_at_point

            return self.monochrome_data[modes[0]]

        elif mode in modes[1:3]:  # 'sig_to_base', 'sig_to_axis'
            if (not modes[1] in self.monochrome_data) and (
                    not modes[2] in self.monochrome_data):
                self.spectral_data_integrated = self.integrate_spectra()
            if (not modes[1] in self.monochrome_data) and (mode == modes[1]):
                self.monochrome_data[modes[1]]= pd.DataFrame(
                    [], index=active_data.index)
            elif (not modes[2] in self.monochrome_data) and (mode == modes[2]):
                self.monochrome_data[modes[2]] = pd.DataFrame(
                    [], index=active_data.index)

            wn = np.sort(kwargs.get('wn', [1000, 2000]))

            closest_index_lower = np.argmin(np.abs(
                self.spectral_data_integrated.columns-wn[0]))
            closest_index_upper = np.argmin(np.abs(
                self.spectral_data_integrated.columns-wn[1]))

            baseline_x_array = np.array(
                [self.spectral_data_integrated.columns[
                    closest_index_lower],
                    self.spectral_data_integrated.columns[
                        closest_index_upper]])
            baseline_y_array = active_data.loc[:, baseline_x_array]
            self.area_under_baseline = trapz(
                baseline_y_array, x=baseline_x_array) if mode == modes[1] else 0

            curr_sig = (self.spectral_data_integrated.iloc[
                :, closest_index_upper].values -
                self.spectral_data_integrated.iloc[
                    :, closest_index_lower].values - self.area_under_baseline)

            if mode == modes[1]:  # 'sig_to_base'
                self.monochrome_data[modes[1]][str(wn)] = curr_sig
                return self.monochrome_data[modes[1]]
            elif mode == modes[2]:  # 'sig_to_axis'
                self.monochrome_data[modes[2]][str(wn)] = curr_sig
                return self.monochrome_data[modes[2]]

    def integrate_spectra(self, active_data=None):
        active_data = self.check_active_data(active_data)

        spectra_integrated = pd.DataFrame(
            cumtrapz(active_data, x=active_data.columns+0, initial=0),
            index=active_data.index, columns=active_data.columns
            ).round(decimals=6)
        return spectra_integrated

    def principal_component_analysis(self, pca_components,
                                     active_data=None):
        active_data = self.check_active_data(active_data)

        self.pca = principal_component_regression(
            active_data)

        self.pca.perform_pca(pca_components)
        self.monochrome_data['pca'] = self.pca.pca_scores

        return self.pca

    def reference_spectra_fit(self, reference_data, active_data=None):
        active_data = self.check_active_data(active_data)

        reference_components = reference_data.shape[0]
        list_reference_components = ['comp' + str(ii) for ii in np.arange(
            reference_components)]

        self.ref_coefs = pd.DataFrame(l_reg.dataset_regression(
            active_data.values, reference_data), index=active_data.index,
            columns=list_reference_components)

        self.fitted_spectra = pd.DataFrame(
            np.dot(self.ref_coefs.values, reference_data),
            index=active_data.index, columns=active_data.columns)

        self.monochrome_data['ref_spec_fit'] = self.ref_coefs

        return (self.fitted_spectra, self.ref_coefs)

#     def find_peaks(self,active_data=None):#is still experimental
#         active_data = self.check_active_data(active_data)
#
#         processed_data = [find_peaks(row, height=200, prominence=100)
#                           for row in active_data.values]
#
#         return processed_data

####################################
# export methods
####################################

    def export_spectra(self, export_path, export_name, active_data=None):
        active_data = self.check_active_data(active_data)

        active_data.to_csv(export_path + export_name + '.txt', sep='\t',
                              header=True)
