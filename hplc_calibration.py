# -*- coding: utf-8 -*-
"""
Contains class hplc_calibration.

Can only be used in combination with hplc_data and makes most sense in
combination with hplc_prediction.
"""

import numpy as np
import pandas as pd

from pyAnalytics.hplc_data import hplc_data


class hplc_calibration():
    """
    Class holding and analyzing HPLC calibration measurements.

    Contains one calibration set of one substance per instance.
    """

    def __init__(self, data_mode, calibration_data, concentrations,
                 time_limits=None, wavelength_limits=None):
        """
        Use input data and store them internally in hplc_data instances.

        Parameters
        ----------
        data_mode : string
            Defines the way the data is imported into the instance. At the
            moment, input as instances of hplc_data or direct input of
            DataFrames with the wavelengths as columns and the elution time as
            index is possible, so the allowed values are 'hplc_data' or
            'DataFrame'. Might be extended in the future for direct data import
            from files.
        calibration_data : list of DataFrames
            Contains the data from the n calibration measurements as
            DataFrames with the wavelengths as columns and the elution time
            as index.
        concentrations : list
            Contains the n concentrations present in the n calibration
            measurements contained in calibration_data in the same order.
        time_limits : list, optional
            A list containing two elements: The start time in the elugrams and
            the end time in the elugrams used for calibration. Each element can
            either be a number or None. In the latter case, no data is removed
            from the respective side. The default is None.
        wavelength_limits : list, optional
            A list containing two elements: The start wavelength in the spectra
            and the end wavelength in the spectra used for calibration. Each
            element can either be a number or None. In the latter case, no data
            is removed from the respective side. The default is None.

        Raises
        ------
        ValueError
            If no valid data_mode is given.

        Returns
        -------
        None.

        """
        self.concentrations = np.array(concentrations)[np.newaxis]

        if data_mode == 'DataFrame':
            self.calibration_data = []
            for curr_data in calibration_data:
                self.calibration_data.append(
                    hplc_data('DataFrame', data=curr_data))
        elif data_mode == 'hplc_data':
            self.calibration_data = calibration_data
        else:
            raise ValueError('No valid data_mode given.')

        self.time_limits = time_limits
        self.wavelength_limits = wavelength_limits
        self.calculate_calibration()

    def calculate_calibration(self, time_limits=None, wavelength_limits=None):
        """
        Calculate the calibration by classical least squares fitting.

        The result is stored in self.K.

        Parameters
        ----------
        time_limits : list, optional
            A list containing two elements: The start time in the elugrams and
            the end time in the elugrams used for calibration. Each element can
            either be a number or None. In the latter case, no data is removed
            from the respective side. The default is None.
        wavelength_limits : list, optional
            A list containing two elements: The start wavelength in the spectra
            and the end wavelength in the spectra used for calibration. Each
            element can either be a number or None. In the latter case, no data
            is removed from the respective side. The default is None.

        Returns
        -------
        DataFrame
            The calibration result containing the slopes at the different
            wavelengths obtained by classical least squares fitting.

        """
        if time_limits is not None:
            self.time_limits = time_limits
        if wavelength_limits is not None:
            self.wavelength_limits = wavelength_limits

        self.calibration_integrated = []
        for curr_data in self.calibration_data:
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
