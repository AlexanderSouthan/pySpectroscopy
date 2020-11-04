# -*- coding: utf-8 -*-
"""
Provides a class for spectroscopic data preprocessing and analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from tqdm import tqdm
from scipy.integrate import cumtrapz
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
                must be the wavenumber/wavelength scale sorted in decreasing
                order.

        Returns
        -------
        None.

        """
        self.data_source = data_source
        self.kwargs = kwargs

        self.__import_data()
        self.reset_processed_data()
        self.wavenumbers = self.spectral_data.columns.to_numpy()
        self.baseline_data = {}

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

    def check_active_spectra(self, active_spectra):
        if active_spectra is None:
            active_spectra = self.spectral_data_processed
        return active_spectra

###############################
# preprocessing methods
###############################

    def mean_center(self, active_spectra=None):
        """
        Subtract the mean spectrum from each spectrum im self.spectral_data.

        Parameters
        ----------
        active_spectra : pandas DataFrame or None, optional
            None means that self.spectral_data_processed is used for baseline
            correction. Alternatively, a pandas DataFrame in the same format as
            self.spectral_data_processed can be passed that is used for the
            calculations. The default is None.

        Returns
        -------
        pandas DataFrame
            The mean centered spectral data.

        """
        active_spectra = self.check_active_spectra(active_spectra)

        mean_centered_data = pd.DataFrame(
            scale(active_spectra, axis=0, with_std=False),
            index=active_spectra.index, columns=active_spectra.columns).round(
                decimals=6)

        self.spectral_data_processed = mean_centered_data

        return mean_centered_data

    def standard_normal_variate(self, active_spectra=None):
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
        active_spectra : pandas DataFrame or None, optional
            None means that self.spectral_data_processed is used for baseline
            correction. Alternatively, a pandas DataFrame in the same format as
            self.spectral_data_processed can be passed that is used for the
            calculations. The default is None.

        Returns
        -------
        pandas DataFrame
            The mean centered and scaled spectral data.

        """
        active_spectra = self.check_active_spectra(active_spectra)

        SNV_scaled_data = pd.DataFrame(active_spectra.subtract(
            active_spectra.mean(axis=1), axis=0).divide(
                active_spectra.std(axis=1), axis=0),
                index=active_spectra.index,
                columns=active_spectra.columns).round(decimals=6)

        self.spectral_data_processed = SNV_scaled_data

        return SNV_scaled_data

    def clip_wavenumbers(self, wn_limits, active_spectra=None):
        """
        Select certain wavenumber ranges from spectral data.

        Parameters
        ----------
        wn_limits : list of tuples
            A list containing tuples with two numbers. Each tuple gives a lower
            and an upper wavenumber defining a wavenumber range. When more than
            one tuple is given, the remaining spectral regions are not
            necessarily neighboring spectral regions in the original data.
        active_spectra : pandas DataFrame or None, optional
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

        self.spectral_data_processed = clipped_data

        return clipped_data

    def clip_samples(self, x_limits=None, y_limits=None, z_limits=None,
                     active_spectra=None):
        # This method is not good for this class. It assumes that the index
        # contains info only present for Raman images, so it must be improved
        # to accept all kinds on index values, such as sample names or simply
        # numbers.
        active_spectra = self.check_active_spectra(active_spectra)

        x_clipping_mask = self.generate_sample_clipping_mask(active_spectra,
                                                             x_limits, 0)
        y_clipping_mask = self.generate_sample_clipping_mask(active_spectra,
                                                             y_limits, 1)
        z_clipping_mask = self.generate_sample_clipping_mask(active_spectra,
                                                             z_limits, 2)

        clipped_data = active_spectra.loc[(x_clipping_mask, y_clipping_mask,
                                           z_clipping_mask)]

        self.spectral_data_processed = clipped_data

        return clipped_data

    def smoothing(self, mode, active_spectra=None, **kwargs):
        active_spectra = self.check_active_spectra(active_spectra)

        smoothed_data = pd.DataFrame(
            smooth_data(active_spectra.values, mode=mode, **kwargs),
            index=active_spectra.index, columns=active_spectra.columns).round(
                decimals=6)

        self.spectral_data_processed = smoothed_data

        return smoothed_data

    def normalize(self, mode, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        normalize_modes = ['total_intensity']
        assert mode in normalize_modes, 'normalize mode unknown'

        if mode == normalize_modes[0]:  # total_intensity
            normalized_data = pd.DataFrame(-transform.normalize(
                active_spectra.values, mode,
                x_data=active_spectra.columns.to_numpy()),
                index=active_spectra.index, columns=active_spectra.columns)

        self.spectral_data_processed = normalized_data

        return normalized_data

    def baseline_correction(self, mode='ModPoly', smoothing=True,
                            transform=False, active_spectra=None, **kwargs):
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
        active_spectra : pandas DataFrame or None, optional
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
        active_spectra = self.check_active_spectra(active_spectra)

        if mode in ['convex_hull', 'ModPoly', 'IModPoly', 'PPF', 'iALSS']:
            kwargs['wavenumbers'] = active_spectra.columns.to_numpy()

        self.baseline_data[mode] = pd.DataFrame(
            generate_baseline(active_spectra.values, mode, smoothing=True,
                              tranform=False, **kwargs),
            index=active_spectra.index, columns=active_spectra.columns)

        corrected_data = (active_spectra -
                          self.baseline_data[mode]).round(decimals=6)

        self.spectral_data_processed = corrected_data

        return corrected_data

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

    def integrate_spectra(self, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        spectra_integrated = pd.DataFrame(
            -cumtrapz(active_spectra, x=active_spectra.columns+0, initial=0),
            index=active_spectra.index, columns=active_spectra.columns
            ).round(decimals=6)
        return spectra_integrated

    def principal_component_analysis(self, pca_components,
                                     active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        self.pca = principal_component_regression(
            active_spectra)

        self.pca.perform_pca(pca_components)

        return self.pca

    def reference_spectra_fit(self, reference_data, active_spectra=None):
        active_spectra = self.check_active_spectra(active_spectra)

        reference_components = reference_data.shape[0]
        list_reference_components = ['comp' + str(ii) for ii in np.arange(
            reference_components)]

        self.ref_coefs = pd.DataFrame(l_reg.dataset_regression(
            active_spectra.values, reference_data), index=active_spectra.index,
            columns=list_reference_components)

        self.fitted_spectra = pd.DataFrame(
            np.dot(self.ref_coefs.values, reference_data),
            index=active_spectra.index, columns=active_spectra.columns)

        return (self.fitted_spectra, self.ref_coefs)

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
